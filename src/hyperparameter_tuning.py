#!/usr/bin/env python3

"""
My hyperparameter tuning module for face models.

Written for my senior project in computer vision.
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
import torch.nn.functional as F
from datetime import datetime
import logging
from tqdm.auto import tqdm  # For progress bars

from .base_config import PROC_DATA_DIR, CHECKPOINTS_DIR, logger, OUT_DIR
from .face_models import get_model, get_criterion, ArcMarginProduct
from .data_utils import SiameseDataset
from .lr_finder import LearningRateFinder
import inspect

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
HYPEROPT_DIR = OUT_DIR / "hyperopt_runs"
MODEL_TYPES = ["baseline", "cnn", "siamese", "attention", "arcface", "hybrid", "ensemble"]

# Create hyperopt directory if it doesn't exist
HYPEROPT_DIR.mkdir(parents=True, exist_ok=True)

# My best guesses for good starting parameters
# Found these by running lots of experiments
TRIAL0_BASELINES = {
    "hybrid": {
        "epochs": 50, "batch_size": 32, "learning_rate": 3e-4,
        "weight_decay": 1e-4, "dropout": 0.3, "scheduler": "cosine"
    },
    "arcface": {
        # ArcFace is super picky about these values - these are more conservative now
        "epochs": 100, "batch_size": 32, "learning_rate": 1e-4,  # Much lower LR needed
        "weight_decay": 2e-4, "dropout": 0.3, "scheduler": "cosine",  # Increased regularization
        "arcface_margin": 0.3, "arcface_scale": 20.0,  # Reduced scale to prevent explosion
        "label_smoothing": 0.1,  # Increased label smoothing for stability
        "use_lr_warmup": True,  # Need this or training fails
        "warmup_epochs": 20,  # Extended warmup period
        "use_gradient_clipping": True,  # Always use gradient clipping
        "clip_grad_norm": 0.5,  # More aggressive clipping
        "optimizer": "AdamW",  # Works better than SGD
        "use_amsgrad": True,  # Helps reach convergence
        "use_progressive_margin": True,  # Always use progressive margin
        "initial_margin_factor": 0.0,  # Always start at zero margin
        "easy_margin": True  # Always use easy margin for stability
    },
    "cnn": {
        "epochs": 40, "batch_size": 64, "learning_rate": 1e-3,
        "weight_decay": 1e-5, "dropout": 0.35, "scheduler": "onecycle"
    },
    "baseline": {
        "epochs": 30, "batch_size": 32, "learning_rate": 5e-3,
        "weight_decay": 1e-4, "dropout": 0.5, "scheduler": "reduce_lr",
        "scheduler_patience": 5, "scheduler_factor": 0.5
    },
    "siamese": {
        # Enhanced Siamese network parameters
        "epochs": 45, "batch_size": 32, "learning_rate": 1e-4,  # Slightly lower LR 
        "weight_decay": 2e-4, "dropout": 0.3, "scheduler": "cosine",
        "margin": 2.0,  # Contrastive loss margin
        "pos_weight": 1.2, "neg_weight": 0.8  # Balanced class weighting
    },
    "attention": {
        "epochs": 40, "batch_size": 48, "learning_rate": 5e-4,
        "weight_decay": 2e-4, "dropout": 0.25, "scheduler": "cosine",
        "num_heads": 2, "reduction_ratio": 8
    },
    "ensemble": {
        # Improved ensemble configuration with weighted method
        "epochs": 30, "batch_size": 32, "learning_rate": 5e-4,  # Lower LR for better ensemble training
        "weight_decay": 2e-4, "dropout": 0.2, "scheduler": "cosine",
        "ensemble_method": "weighted",  # Use weighted ensemble method
        "label_smoothing": 0.1  # Label smoothing for better calibration
    }
}

def create_optimizer(model: torch.nn.Module, params: Dict[str, Any]) -> torch.optim.Optimizer:
    """Make an optimizer for the model.
    
    Args:
        model: The PyTorch model to optimize
        params: Dictionary with optimizer settings like:
            - optimizer: Which optimizer to use ('AdamW', 'RAdam', 'SGD_momentum')
            - learning_rate: How fast to learn
            - weight_decay: Regularization strength
            - use_amsgrad: Whether to use AMSGrad (for AdamW)
            
    Returns:
        The optimizer to use for training
    """
    optimizer_type = params.get("optimizer", "AdamW")
    lr = params.get("learning_rate", 0.001)
    weight_decay = params.get("weight_decay", 0.0)
    use_amsgrad = params.get("use_amsgrad", False)
    
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=use_amsgrad  # Use AMSGrad if specified (helps with ArcFace convergence)
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
            patience=params.get("scheduler_patience", 5)
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
    
    # Configure model-specific learning rate ranges
    start_lr = 1e-7  # Standard start LR
    if model_type == 'arcface':
        # ArcFace needs much lower end LR
        end_lr = 0.01
        logger.info(f"Using lower end_lr={end_lr} for ArcFace model")
        num_iterations = 70  # More iterations for better resolution at lower LRs
    elif model_type == 'siamese':
        # Siamese networks also benefit from more controlled LR ranges
        end_lr = 0.1
        logger.info(f"Using restricted end_lr={end_lr} for Siamese model")
        num_iterations = 60
    else:
        # Standard settings for other models
        end_lr = 1.0
        num_iterations = 50
    
    # Initialize LR Finder with model-specific settings
    lr_finder = LearningRateFinder(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iterations=num_iterations,  # Adjusted iterations based on model type
        save_dir=output_dir,
        model_type=model_type  # Pass model type for specialized scaling
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
                             n_trials: int = 20,  # Default 20, ArcFace needs more
                             timeout: Optional[int] = None,
                             use_trial0_baseline: bool = True,
                             keep_checkpoints: int = 1,
                             use_lr_finder: bool = False,
                             optimizer_type: Optional[str] = None,
                             arcface_params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Run the hyperparameter tuning process with user-friendly progress bar.
    
    Args:
        model_type: Which model to tune (baseline, cnn, siamese, etc.)
        dataset_path: Where the training data lives
        n_trials: How many different parameter combos to try
        timeout: Max seconds to run (can be None)
        use_trial0_baseline: Start with my hand-picked values? 
        keep_checkpoints: How many saved models to keep per trial
        use_lr_finder: Use my auto LR finder?
        optimizer_type: Which optimizer ('AdamW', 'RAdam', 'SGD_momentum')
        arcface_params: Extra params for ArcFace:
            - include_progressive_margin: Try different margin growth?
            - include_easy_margin: Test with/without easy margin?
            - wider_margin_scale_range: Try more extreme values?
            - include_amsgrad: Test with/without AMSGrad?
            - include_gradient_clipping: Try different clipping thresholds?
        
    Returns:
        Best parameters and results, or None if it fails
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
    
    # Create progress tracking for trials
    # ArcFace needs more trials but we already fixed this in interactive.py
    original_n_trials = n_trials
    
    print(f"\nStarting hyperparameter tuning for {model_type} with {n_trials} trials")
    print(f"Using dataset: {dataset_path.name}")
    print(f"Trial-0 baseline: {'Enabled' if use_trial0_baseline else 'Disabled'}")
    print(f"LR Finder: {'Enabled' if use_lr_finder else 'Disabled'}")
    
    # Initialize progress bar for trials
    pbar_trials = tqdm(total=n_trials, desc="Hyperparameter Trials")
    
    # Define callback to update progress bar
    def progress_callback(study, trial):
        # Update progress bar with current trial information
        pbar_trials.update(1)
        current_best = study.best_value if study.best_trial else 0
        pbar_trials.set_postfix({
            "best_acc": f"{current_best:.4f}",
            "last_trial": f"{trial.number}",
            "state": trial.state.name
        })
    
    # Run optimization with progress tracking
    study.optimize(
        lambda trial: objective(trial, model_type, dataset_path, use_trial0_baseline, use_lr_finder, optimizer_type, arcface_params),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[progress_callback]
    )
    
    # Close the progress bar
    pbar_trials.close()
    
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

# All possible hyperparameters to try
def create_search_space():
    """Create the search space for all the hyperparameters."""
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
            use_trial0_baseline: bool, use_lr_finder: bool = False, optimizer_type: Optional[str] = None,
            arcface_params: Optional[Dict[str, Any]] = None) -> float:
    """This is the function Optuna calls for each trial.
    
    Args:
        trial: Optuna object that manages the trial
        model_type: Which model we're tuning
        dataset_path: Where to find the data
        use_trial0_baseline: Start with my best guesses?
        use_lr_finder: Try to auto-detect good learning rates?
        optimizer_type: Which optimizer to use
        arcface_params: Special options for ArcFace
        
    Returns:
        How good this trial was (validation accuracy)
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
            
            # Apply model-specific scaling factors
            if model_type == 'arcface':
                # More conservative range for ArcFace
                min_lr = max(1e-5, optimal_lr / 5)
                max_lr = min(1e-3, optimal_lr * 3)
                logger.info(f"Using ArcFace-adjusted LR range: {min_lr:.2e} to {max_lr:.2e}")
            elif model_type == 'siamese':
                # Adjusted range for Siamese
                min_lr = max(1e-5, optimal_lr / 4)
                max_lr = min(5e-4, optimal_lr * 2)
                logger.info(f"Using Siamese-adjusted LR range: {min_lr:.2e} to {max_lr:.2e}")
            else:
                # Standard range for other models
                min_lr = optimal_lr / 3
                max_lr = optimal_lr * 3
                logger.info(f"Using standard LR range: {min_lr:.2e} to {max_lr:.2e}")
                
            params['learning_rate'] = trial.suggest_float('learning_rate', min_lr, max_lr, log=True)
            logger.info(f"Using LR Finder suggested range: {min_lr:.2e} to {max_lr:.2e}")
        except Exception as e:
            logger.warning(f"LR Finder failed: {e}. Falling back to default range.")
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    else:
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # Increased weight decay range for better regularization, especially important for ArcFace
    if model_type == 'arcface':
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 5e-3, log=True)  # Higher weight decay for ArcFace
        logger.info(f"Using increased weight decay range for ArcFace: [1e-4, 5e-3]")
    else:
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # Set optimizer type based on user preference or sample if not specified
    if optimizer_type:
        params['optimizer'] = optimizer_type
        logger.info(f"Using user-specified optimizer: {optimizer_type}")
    else:
        params['optimizer'] = trial.suggest_categorical('optimizer', ['AdamW', 'RAdam', 'SGD_momentum'])
    
    params['scheduler'] = trial.suggest_categorical('scheduler', ['cosine', 'onecycle', 'plateau'])
    params['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Scheduler-specific parameters
    if params['scheduler'] == 'plateau':
        params['scheduler_patience'] = trial.suggest_int('scheduler_patience', 3, 10)
        params['scheduler_factor'] = trial.suggest_float('scheduler_factor', 0.1, 0.7)
    
    # Special stuff for ArcFace
    if model_type == 'arcface':
        # See what options the user selected
        use_wider_range = arcface_params and arcface_params.get('wider_margin_scale_range', False) 
        include_easy_margin = arcface_params and arcface_params.get('include_easy_margin', False)
        include_prog_margin = arcface_params and arcface_params.get('include_progressive_margin', False)
        include_gradient_clip = arcface_params and arcface_params.get('include_gradient_clipping', False)
        include_amsgrad = arcface_params and arcface_params.get('include_amsgrad', False)
        
        # Log what we're trying
        logger.info(f"ArcFace options: wider_range={use_wider_range}, easy_margin={include_easy_margin}, "
                  f"prog_margin={include_prog_margin}, grad_clip={include_gradient_clip}, amsgrad={include_amsgrad}")
        
        # Always use conservative scale values to prevent loss explosion
        # Scale range reduced to 16.0-24.0 max as suggested
        params['arcface_margin'] = trial.suggest_float('arcface_margin', 0.2, 0.5)  # Safer margin range
        params['arcface_scale'] = trial.suggest_float('arcface_scale', 16.0, 24.0)  # Reduced scale range
        logger.info(f"Using conservative ranges: margin=[0.2-0.5], scale=[16.0-24.0]")
        
        # Easy margin makes training more stable
        if include_easy_margin:
            params['easy_margin'] = trial.suggest_categorical('easy_margin', [True, False])
            logger.info(f"Testing both with and without easy margin")
        else:
            # Just use easy margin by default
            params['easy_margin'] = True
        
        # Progressive margin grows the margin during training
        # Always use progressive margin with initial factor of 0.0
        params['use_progressive_margin'] = True
        params['initial_margin_factor'] = 0.0  # Always start at 0.0 to prevent instability
        logger.info(f"Using progressive margin with initial_factor=0.0")
        
        # Gradient clipping prevents exploding gradients
        # Force gradient clipping to always be true
        params['use_gradient_clipping'] = True
        params['clip_grad_norm'] = trial.suggest_float('clip_grad_norm', 0.1, 1.0)
        logger.info(f"Enforcing gradient clipping with clip_norm range=[0.1-1.0]")
        
        # AMSGrad is an Adam optimizer variant
        if include_amsgrad:
            params['use_amsgrad'] = trial.suggest_categorical('use_amsgrad', [True, False])
            logger.info(f"Testing with and without AMSGrad")
        else:
            params['use_amsgrad'] = True  # Just use it by default
        
        # Label smoothing helps prevent overconfidence
        params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.05, 0.15)  # Small range
        
        # Force warm-up for stable training and better convergence
        params['use_lr_warmup'] = True  # Always use warm-up for ArcFace
        logger.info("Enforcing warm-up for ArcFace to improve training stability")
        
        # Add warm-up epochs if using warm-up
        if params['use_lr_warmup']:
            params['warmup_epochs'] = trial.suggest_int('warmup_epochs', 5, 15)
            
        # Add progressive margin options
        params['use_progressive_margin'] = trial.suggest_categorical('use_progressive_margin', [True, False])
        
        # Add gradient clipping for numerical stability
        params['use_gradient_clipping'] = trial.suggest_categorical('use_gradient_clipping', [True, False])
        
        if params['use_gradient_clipping']:
            params['clip_grad_norm'] = trial.suggest_float('clip_grad_norm', 0.1, 1.0)
            
        # Advanced optimizer options
        params['use_amsgrad'] = trial.suggest_categorical('use_amsgrad', [True, False])
        
        # Add initial margin factor for progressive scaling
        if params['use_progressive_margin']:
            params['initial_margin_factor'] = trial.suggest_float('initial_margin_factor', 0.1, 0.5)
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
    num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 2
    
    # For ArcFace, we need to apply custom initialization parameters 
    if model_type == 'arcface':
        # First get the model with default parameters
        model = get_model(model_type, num_classes=num_classes)
        
        # Then customize the model based on hyperparameters
        # Update the ArcFace parameters if provided
        if 'arcface_margin' in params and 'arcface_scale' in params:
            # Create a new ArcMarginProduct layer with the suggested hyperparameters
            # Include warm-up parameters if provided
            use_warm_up = params.get('use_lr_warmup', True)
            new_arcface = ArcMarginProduct(
                in_feats=512,  # Standard embedding dimension for ArcFace
                out_feats=num_classes,
                s=float(params['arcface_scale']),
                m=float(params['arcface_margin']),
                use_warm_up=use_warm_up
            )
            
            # Always use progressive margin for stable training
            # Configure progressive margin parameters for better stability
            new_arcface.use_warm_up = True  # Force warm-up for stability
            new_arcface.warm_up_epochs = params.get('warmup_epochs', 10)
            
            # Set a smaller initial margin factor to reduce initial loss values
            initial_margin_factor = params.get('initial_margin_factor', 0.0) 
            new_arcface.margin_factor = initial_margin_factor
            
            # Set a conservative scale factor to prevent exploding loss
            new_arcface.scale_factor = 0.3  # Start with 30% of final scale
            
            logger.info(f"Configured progressive margin with initial_factor={new_arcface.margin_factor}, scale_factor=0.3")
            logger.info(f"Using forced warm-up for {new_arcface.warm_up_epochs} epochs to improve stability")
                
            # Replace the old layer
            model.arcface = new_arcface
            logger.info(f"Updated ArcFace with margin={params['arcface_margin']}, scale={params['arcface_scale']}, warm_up={use_warm_up}")
    else:
        # For other models, use standard initialization
        model = get_model(model_type, num_classes=num_classes)
        
    # Apply dropout if specified
    if params.get('dropout', 0.0) > 0:
        # Make sure the model has a dropout attribute before trying to modify it
        if hasattr(model, 'dropout'):
            model.dropout = nn.Dropout(params['dropout'])
            logger.info(f"Set dropout rate to {params['dropout']}")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer and criterion
    optimizer = create_optimizer(model, params)
    
    # Set up criterion with label smoothing for ArcFace if specified
    if model_type == 'arcface' and 'label_smoothing' in params:
        criterion = nn.CrossEntropyLoss(label_smoothing=params['label_smoothing'])
        logger.info(f"Using CrossEntropyLoss with label_smoothing={params['label_smoothing']} for ArcFace")
    else:
        criterion = get_criterion(model_type)
    
    # Setup scheduler
    scheduler = get_scheduler(optimizer, params, epochs=10)  # Use 10 epochs for quick evaluation
    
    # Training loop
    best_val_acc = 0.0
    early_stopping_counter = 0
    early_stopping_patience = 5
    
    pbar_epochs = tqdm(range(10), desc=f"Trial {trial.number}", leave=False)
    for epoch in pbar_epochs:  # Use 10 epochs for quick evaluation with progress bar
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Add progress bar for batches
        pbar_train = tqdm(train_loader, desc=f"Train {epoch+1}/10", leave=False)
        for inputs, targets in pbar_train:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Handle ArcFace models which need labels during training
            if model_type == 'arcface' and model.training:
                outputs = model(inputs, labels=targets)  # Pass labels to ArcFace model
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Apply gradient clipping for numerical stability, especially for ArcFace
            if model_type == 'arcface' and params.get('use_gradient_clipping', False):
                clip_value = params.get('clip_grad_norm', 0.5)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = train_correct / train_total
        # Update epoch progress bar with current metrics
        pbar_epochs.set_postfix({"train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.4f}"})
        
        # Update epoch for ArcFace progressive margin
        if model_type == 'arcface' and hasattr(model, 'update_epoch'):
            model.update_epoch(epoch)
            if epoch % 2 == 0 and hasattr(model, 'get_arcface_stats'):
                stats = model.get_arcface_stats()
                logger.info(f"ArcFace epoch {epoch} stats - margin: {stats.get('effective_margin', 0):.3f}, "
                           f"scale: {stats.get('effective_scale', 0):.1f}")
                
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            # Add progress bar for validation batches
            pbar_val = tqdm(val_loader, desc=f"Val {epoch+1}/10", leave=False)
            for inputs, targets in pbar_val:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # For validation, ArcFace needs special handling
                if model_type == 'arcface':
                    # During validation, ArcFace doesn't need labels
                    emb = model(inputs)  # Get embeddings
                    
                    # For ArcFace validation we can use a simple linear layer
                    # to get logits from the embeddings
                    if not hasattr(model, 'val_classifier'):
                        # Create a simple classifier for validation only
                        model.val_classifier = nn.Linear(
                            512,  # ArcFace embedding dimension
                            len(train_dataset.classes)
                        ).to(device)
                        # Initialize with normalized weights for better cosine similarity
                        nn.init.orthogonal_(model.val_classifier.weight)
                    
                    # Normalize embeddings for consistent evaluation
                    emb = F.normalize(emb, p=2, dim=1)
                    # Normalize classifier weights for cosine similarity
                    model.val_classifier.weight.data = F.normalize(
                        model.val_classifier.weight.data, p=2, dim=1)
                    
                    outputs = model.val_classifier(emb)
                else:
                    outputs = model(inputs)
                    
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = val_correct / val_total
        # Update the epoch progress bar with validation metrics
        pbar_epochs.set_postfix({"train_acc": f"{train_acc:.4f}", "val_acc": f"{val_acc:.4f}"})
        
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