#!/usr/bin/env python3

"""
Enhanced hyperparameter tuning module for face recognition models.
"""

import os
import json
import torch
import optuna
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from datetime import datetime
import logging

from .base_config import PROC_DATA_DIR, CHECKPOINTS_DIR, logger, OUT_DIR
from .face_models import get_model, get_criterion
from .data_utils import SiameseDataset
from .lr_finder import LearningRateFinder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
HYPEROPT_DIR = OUT_DIR / "hyperopt_runs"
MODEL_TYPES = ["baseline", "cnn", "siamese", "attention", "arcface", "hybrid", "ensemble"]

# Create hyperopt directory if it doesn't exist
HYPEROPT_DIR.mkdir(parents=True, exist_ok=True)

# Define trial-0 baselines (good starting points)
TRIAL0_BASELINES = {
    "hybrid": {
        "epochs": 50, "batch_size": 32, "learning_rate": 3e-4,
        "weight_decay": 1e-4, "dropout": 0.3, "scheduler": "cosine"
    },
    "arcface": {
        "epochs": 60, "batch_size": 128, "learning_rate": 1e-4,
        "weight_decay": 5e-5, "dropout": 0.2, "scheduler": "cosine",
        "arcface_margin": 0.45, "arcface_scale": 48
    },
    "cnn": {
        "epochs": 40, "batch_size": 64, "learning_rate": 1e-3,
        "weight_decay": 1e-5, "dropout": 0.35, "scheduler": "onecycle"
    },
    # Adding baselines for the remaining model types
    "baseline": {
        "epochs": 30, "batch_size": 32, "learning_rate": 5e-3,
        "weight_decay": 1e-4, "dropout": 0.5, "scheduler": "reduce_lr",
        "scheduler_patience": 5, "scheduler_factor": 0.5
    },
    "siamese": {
        "epochs": 45, "batch_size": 32, "learning_rate": 2e-4,
        "weight_decay": 1e-4, "dropout": 0.3, "scheduler": "cosine",
        "margin": 2.0  # Contrastive loss margin
    },
    "attention": {
        "epochs": 40, "batch_size": 48, "learning_rate": 5e-4,
        "weight_decay": 2e-4, "dropout": 0.25, "scheduler": "cosine",
        "num_heads": 2, "reduction_ratio": 8
    },
    "ensemble": {
        "epochs": 30, "batch_size": 32, "learning_rate": 8e-4,
        "weight_decay": 1e-4, "dropout": 0.2, "scheduler": "cosine",
        "ensemble_method": "weighted"  # Use weighted ensemble method
    }
}

def create_optimizer(model: torch.nn.Module, params: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on parameters.
    
    Args:
        model: The model to optimize
        params: Dictionary containing optimizer parameters including:
            - optimizer: Type of optimizer ('AdamW', 'RAdam', 'SGD_momentum')
            - learning_rate: Learning rate
            - weight_decay: Weight decay value
            
    Returns:
        Configured optimizer instance
    """
    optimizer_type = params.get("optimizer", "AdamW")
    lr = params.get("learning_rate", 0.001)
    weight_decay = params.get("weight_decay", 0.0)
    
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == "RAdam":
        # Check if torch has RAdam or use third-party implementation
        if hasattr(torch.optim, "RAdam"):
            return torch.optim.RAdam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            # Fallback to AdamW with a warning
            logger.warning("RAdam not available, falling back to AdamW")
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
    elif optimizer_type == "SGD_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        # Default to Adam
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

def get_scheduler(optimizer: torch.optim.Optimizer, params: Dict[str, Any], epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler based on parameters.
    
    Args:
        optimizer: The optimizer to schedule
        params: Dictionary containing scheduler parameters including:
            - scheduler: Type of scheduler ('cosine', 'onecycle', 'plateau', 'none')
            - learning_rate: Base learning rate
            - scheduler_patience: Patience for plateau scheduler
            - scheduler_factor: Factor for reducing learning rate
        epochs: Total number of epochs
        
    Returns:
        Configured scheduler instance or None
    """
    scheduler_type = params.get("scheduler", "cosine")
    
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=params.get("learning_rate", 0.001) / 100
        )
    elif scheduler_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.get("learning_rate", 0.001),
            epochs=epochs,
            steps_per_epoch=params.get("steps_per_epoch", 100),
            pct_start=0.2
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=params.get("scheduler_factor", 0.5),
            patience=params.get("scheduler_patience", 5),
            verbose=True
        )
    else:
        return None

def find_optimal_lr_for_trial(model_type: str, dataset_path: Path, batch_size: int = 32) -> float:
    """
    Find the optimal learning rate for a specific trial.
    
    Args:
        model_type: Type of model (baseline, cnn, siamese, etc.)
        dataset_path: Path to the processed dataset
        batch_size: Batch size for the dataloader
        
    Returns:
        Suggested learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running LR Finder for trial on device: {device}")
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    if model_type == 'siamese':
        train_dataset = SiameseDataset(str(dataset_path / "train"), transform=transform)
        num_classes = 2
    else:
        train_dataset = datasets.ImageFolder(dataset_path / "train", transform=transform)
        num_classes = len(train_dataset.classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = get_model(model_type, num_classes=num_classes)
    model = model.to(device)
    
    # Setup criterion and optimizer
    criterion = get_criterion(model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    
    # Create output directory
    output_dir = HYPEROPT_DIR / "lr_finder" / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LR Finder with fewer iterations for hyperparameter tuning
    lr_finder = LearningRateFinder(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        start_lr=1e-7,
        end_lr=1.0,
        num_iterations=50,  # Fewer iterations for speed in hyperparameter tuning
        save_dir=output_dir
    )
    
    # Run LR finder
    logger.info(f"Running learning rate finder for trial with {model_type} model...")
    lr_finder.find_lr(train_loader)
    
    # Analyze results
    analysis = lr_finder._analyze_results()
    
    # Get suggested learning rate
    suggested_lr = analysis["overall"]["suggested_learning_rate"]
    logger.info(f"LR Finder suggested learning rate for trial: {suggested_lr:.2e}")
    
    return suggested_lr

def run_hyperparameter_tuning(model_type: Optional[str] = None, 
                             dataset_path: Optional[Path] = None,
                             n_trials: int = 20,
                             timeout: Optional[int] = None,
                             use_trial0_baseline: bool = True,
                             keep_checkpoints: int = 1,
                             use_lr_finder: bool = False) -> Optional[Dict[str, Any]]:
    """Run hyperparameter tuning for a model.
    
    Args:
        model_type: Type of model to tune
        dataset_path: Path to the dataset
        n_trials: Number of trials to run
        timeout: Optional timeout in seconds for the entire optimization
        use_trial0_baseline: Whether to use trial-0 baseline for first trial
        keep_checkpoints: Number of best checkpoints to keep per trial
        use_lr_finder: Whether to use LR Finder to determine optimal learning rates for each trial
        
    Returns:
        Dictionary containing best parameters and results, or None if tuning fails
    """
    # Use interactive selection if model_type or dataset_path not provided
    if model_type is None:
        print("\nAvailable model types:")
        for mt in MODEL_TYPES:
            print(f"- {mt}")
        
        model_type = input("\nEnter model type: ")
        if model_type.lower() not in MODEL_TYPES:
            print("Invalid model type")
            return None
    
    if dataset_path is None:
        # List available processed datasets
        processed_dirs = []
        dataset_names = set()  # For deduplication
        
        # Check for the standard organization: config_name/dataset_name/train
        config_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and not d.name in ["train", "val", "test"]]
        for config_dir in config_dirs:
            if (config_dir / "train").exists():
                # Use this directory with its name
                if config_dir.name not in dataset_names:
                    processed_dirs.append((config_dir, config_dir.name))
                    dataset_names.add(config_dir.name)
            else:
                for dataset_dir in config_dir.iterdir():
                    if dataset_dir.is_dir() and (dataset_dir / "train").exists():
                        # Use config/dataset naming
                        dataset_name = f"{config_dir.name}/{dataset_dir.name}"
                        if dataset_name not in dataset_names:
                            processed_dirs.append((dataset_dir, dataset_name))
                            dataset_names.add(dataset_name)
        
        # Also check for simpler structure where train is directly in PROC_DATA_DIR
        if (PROC_DATA_DIR / "train").exists() and (PROC_DATA_DIR / "val").exists():
            if "root" not in dataset_names:
                processed_dirs.append((PROC_DATA_DIR, "processed (root)"))
                dataset_names.add("root")
        
        if not processed_dirs:
            print("No processed datasets found. Please process raw data first.")
            return None
        
        print("\nAvailable processed datasets:")
        for i, (dir_path, display_name) in enumerate(processed_dirs, 1):
            print(f"{i}. {display_name}")
        
        while True:
            dataset_choice = input("\nEnter dataset number to use for hyperparameter tuning: ")
            try:
                dataset_idx = int(dataset_choice) - 1
                if 0 <= dataset_idx < len(processed_dirs):
                    dataset_path = processed_dirs[dataset_idx][0]  # Get the path part
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    print(f"\nRunning hyperparameter tuning for {model_type} on {dataset_path.name} with {n_trials} trials")
    if use_trial0_baseline:
        print("Using trial-0 baseline for first trial")
    if use_lr_finder:
        print("Using Learning Rate Finder to determine optimal learning rates for each trial")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperopt_output_dir = HYPEROPT_DIR / f"{model_type}_hyperopt_{dataset_path.name}_{timestamp}"
    hyperopt_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging for this run
    log_file = hyperopt_output_dir / "hyperopt.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting hyperparameter tuning for {model_type} on {dataset_path.name}")
    logger.info(f"Keeping only {keep_checkpoints} best checkpoint(s) per trial to save storage")
    
    # Create and run the Optuna study
    study_name = f"{model_type}_{dataset_path.name}"
    storage_name = f"sqlite:///{hyperopt_output_dir}/study.db"
    
    try:
        # Try to continue from existing study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize"
        )
        logger.info(f"Continuing existing study with {len(study.trials)} previous trials")
    except:
        # Create new study if loading fails
        try:
            if Path(storage_name.replace("sqlite:///", "")).exists():
                Path(storage_name.replace("sqlite:///", "")).unlink()
        except:
            pass
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize"
        )
        logger.info("Created new study")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, model_type, dataset_path, use_trial0_baseline, use_lr_finder),
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Get best parameters and results
    best_params = study.best_params
    best_accuracy = study.best_value
    
    # Compile and save final hyperopt results
    hyperopt_final_results = {
        'model_type': model_type,
        'dataset': dataset_path.name,
        'n_trials': n_trials,
        'best_trial': study.best_trial.number,
        'best_accuracy': best_accuracy,
        'best_params': best_params,
        'all_trials': [
            {
                'trial': trial.number,
                'params': trial.params,
                'accuracy': trial.value,
                'state': trial.state.name
            }
            for trial in study.trials
        ]
    }
    
    # Save results
    with open(hyperopt_output_dir / 'hyperopt_results.json', 'w') as f:
        json.dump(hyperopt_final_results, f, indent=2)
    
    # Save study summary
    with open(hyperopt_output_dir / 'study_summary.txt', 'w') as f:
        f.write(f"Study summary for {model_type} on {dataset_path.name}\n")
        f.write(f"Number of completed trials: {len(study.trials)}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best accuracy: {best_accuracy*100:.2f}%\n")
        f.write(f"Best parameters: {json.dumps(best_params, indent=2)}\n")
    
    print(f"\nHyperparameter tuning complete!")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    print(f"Best parameters: {best_params}")
    print(f"Results saved to: {hyperopt_output_dir}")
    
    # Ask if user wants to train a model with these parameters
    if input("\nTrain a model with these parameters? (y/n): ").lower() == 'y':
        from .training import train_model
        
        # Create a good model name
        model_name = f"{model_type}_tuned_{dataset_path.name}"
        
        # Set epochs to a reasonable value for full training
        epochs = int(input(f"Enter number of epochs (default 50): ") or "50")
        
        # Train the model with the best parameters
        trained_model_name = train_model(
            model_type=model_type,
            model_name=model_name,
            batch_size=best_params['batch_size'],
            epochs=epochs,
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay'],
            scheduler_type=best_params.get('scheduler', 'cosine'),
            scheduler_patience=best_params.get('scheduler_patience', 5),
            scheduler_factor=best_params.get('scheduler_factor', 0.5),
            clip_grad_norm=best_params.get('clip_grad_norm', None),
            early_stopping=True,
            early_stopping_patience=best_params.get('early_stopping_patience', 10),
            dataset_path=dataset_path
        )
        
        print(f"\nModel trained and saved as: {trained_model_name}")
    
    return hyperopt_final_results

# Define hyperparameter search space
def create_search_space():
    """Define the hyperparameter search space for Optuna."""
    return {
        # Basic parameters
        'batch_size': [8, 16, 32, 64],
        'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        'weight_decay': [1e-5, 1e-4, 1e-3],
        
        # Scheduler parameters
        'scheduler_type': ['reduce_lr', 'cosine', 'step', 'none'],
        'scheduler_patience': [3, 5, 7, 10],
        'scheduler_factor': [0.1, 0.3, 0.5, 0.7],
        
        # Gradient clipping
        'use_grad_clip': [True, False],
        'clip_grad_norm': [0.5, 1.0, 3.0, 5.0],
        
        # Early stopping
        'early_stopping': [True, False],
        'early_stopping_patience': [5, 10, 15, 20]
    }

def objective(trial: optuna.Trial, model_type: str, dataset_path: Path, 
            use_trial0_baseline: bool, use_lr_finder: bool = False) -> float:
    """Objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        model_type: Type of model to tune
        dataset_path: Path to the dataset
        use_trial0_baseline: Whether to use trial-0 baseline
        use_lr_finder: Whether to use the LR Finder to determine optimal learning rate
        
    Returns:
        Validation accuracy for the trial
    """
    # Use trial-0 baseline if specified and this is trial 0
    if use_trial0_baseline and trial.number == 0:
        if model_type in TRIAL0_BASELINES:
            params = TRIAL0_BASELINES[model_type]
            logger.info(f"Using trial-0 baseline for {model_type}")
        else:
            logger.warning(f"No trial-0 baseline defined for {model_type}, using random sampling")
            params = {}
    else:
        params = {}
    
    # Sample hyperparameters
    params['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    
    # Use LR Finder to determine optimal learning rate if enabled
    if use_lr_finder:
        try:
            optimal_lr = find_optimal_lr_for_trial(model_type, dataset_path, params['batch_size'])
            # Use the suggested learning rate with some randomization for exploration
            min_lr = optimal_lr / 3
            max_lr = optimal_lr * 3
            params['learning_rate'] = trial.suggest_float('learning_rate', min_lr, max_lr, log=True)
            logger.info(f"Using LR Finder suggested range: {min_lr:.2e} to {max_lr:.2e}")
        except Exception as e:
            logger.warning(f"LR Finder failed: {e}. Falling back to default range.")
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    else:
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    params['optimizer'] = trial.suggest_categorical('optimizer', ['AdamW', 'RAdam', 'SGD_momentum'])
    params['scheduler'] = trial.suggest_categorical('scheduler', ['cosine', 'onecycle', 'plateau'])
    params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Scheduler-specific parameters
    if params['scheduler'] == 'plateau':
        params['scheduler_patience'] = trial.suggest_int('scheduler_patience', 3, 10)
        params['scheduler_factor'] = trial.suggest_float('scheduler_factor', 0.1, 0.7)
    
    # Model-specific parameters
    if model_type == 'arcface':
        params['arcface_margin'] = trial.suggest_float('arcface_margin', 0.3, 0.5)
        params['arcface_scale'] = trial.suggest_int('arcface_scale', 32, 64)
    elif model_type == 'hybrid':
        params['num_heads'] = trial.suggest_int('num_heads', 4, 8)
        params['transformer_layers'] = trial.suggest_int('transformer_layers', 2, 4)
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    if model_type == 'siamese':
        train_dataset = SiameseDataset(dataset_path / "train", transform=transform)
        val_dataset = SiameseDataset(dataset_path / "val", transform=transform)
    else:
        train_dataset = datasets.ImageFolder(dataset_path / "train", transform=transform)
        val_dataset = datasets.ImageFolder(dataset_path / "val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
    
    # Initialize model
    model = get_model(model_type, num_classes=len(train_dataset.classes) if hasattr(train_dataset, 'classes') else None)
    if params.get('dropout', 0.0) > 0:
        model.dropout = nn.Dropout(params['dropout'])
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer and criterion
    optimizer = create_optimizer(model, params)
    criterion = get_criterion(model_type)
    
    # Setup scheduler
    scheduler = get_scheduler(optimizer, params, epochs=10)  # Use 10 epochs for quick evaluation
    
    # Training loop
    best_val_acc = 0.0
    early_stopping_counter = 0
    early_stopping_patience = 5
    
    for epoch in range(10):  # Use 10 epochs for quick evaluation
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = val_correct / val_total
        
        # Update scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                break
        
        # Report intermediate value to Optuna
        trial.report(val_acc, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_acc

if __name__ == "__main__":
    run_hyperparameter_tuning() 