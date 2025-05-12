#!/usr/bin/env python3

"""
Simple hyperparameter tuning module for face recognition models.
"""

import os
import json
import torch
import optuna
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn

from .base_config import PROC_DATA_DIR, CHECKPOINTS_DIR, logger, OUT_DIR
from .face_models import get_model, get_criterion
from .data_utils import SiameseDataset

def run_hyperparameter_tuning(model_type: Optional[str] = None, 
                             dataset_path: Optional[Path] = None,
                             n_trials: int = 20,
                             existing_model: Optional[str] = None):
    """Run hyperparameter tuning for a model.
    
    Args:
        model_type: Type of model to tune
        dataset_path: Path to the dataset
        n_trials: Number of trials to run
        existing_model: Optional name of an existing model to use as a starting point
    """
    # Use interactive selection if model_type or dataset_path not provided
    if model_type is None:
        print("\nAvailable model types:")
        print("- baseline: Simple CNN architecture")
        print("- cnn: ResNet18 transfer learning")
        print("- siamese: Siamese network for verification")
        print("- attention: ResNet with attention mechanism")
        print("- arcface: Face recognition with ArcFace loss")
        print("- hybrid: CNN-Transformer hybrid architecture")
        print("- ensemble: Combination of multiple models")
        
        model_type = input("\nEnter model type: ")
        if model_type.lower() not in ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid', 'ensemble']:
            print("Invalid model type")
            return False
    
    # Allow selecting an existing model
    if existing_model is None:
        # List available models of this type
        model_dirs = list(CHECKPOINTS_DIR.glob(f'{model_type}_*'))
        if model_dirs:
            print("\nAvailable trained models for this type:")
            print("0. (None - start from scratch)")
            for i, model_dir in enumerate(model_dirs, 1):
                print(f"{i}. {model_dir.name}")
            
            while True:
                model_choice = input("\nSelect an existing model as starting point (0 for none): ")
                try:
                    model_idx = int(model_choice)
                    if model_idx == 0:
                        existing_model = None
                        break
                    elif 1 <= model_idx <= len(model_dirs):
                        existing_model = model_dirs[model_idx-1].name
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
    
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
                        dataset_path = f"{config_dir.name}/{dataset_dir.name}"
                        if dataset_path not in dataset_names:
                            processed_dirs.append((dataset_dir, dataset_path))
                            dataset_names.add(dataset_path)
        
        # Also check for simpler structure where train is directly in PROC_DATA_DIR
        if (PROC_DATA_DIR / "train").exists() and (PROC_DATA_DIR / "val").exists():
            if "root" not in dataset_names:
                processed_dirs.append((PROC_DATA_DIR, "processed (root)"))
                dataset_names.add("root")
        
        if not processed_dirs:
            print("No processed datasets found. Please process raw data first.")
            return False
        
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
    
    # Number of trials is now passed directly from the interactive menu, no need to ask again
    
    print(f"\nRunning hyperparameter tuning for {model_type} on {dataset_path.name} with {n_trials} trials")
    if existing_model:
        print(f"Using existing model as starting point: {existing_model}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    if model_type == 'siamese':
        train_dataset = SiameseDataset(str(dataset_path / "train"), transform=transform)
        val_dataset = SiameseDataset(str(dataset_path / "val"), transform=transform)
    else:
        train_dataset = datasets.ImageFolder(dataset_path / "train", transform=transform)
        val_dataset = datasets.ImageFolder(dataset_path / "val", transform=transform)
    
    # Create data loaders before defining the objective function
    batch_sizes = [8, 16, 32, 64]
    # Use the smallest batch size for initial loading to ensure it always works
    initial_batch_size = min(batch_sizes)
    
    # Create data loaders with the initial batch size
    if model_type == 'siamese':
        train_loader = DataLoader(train_dataset, batch_size=initial_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=initial_batch_size)
    else:
        train_loader = DataLoader(train_dataset, batch_size=initial_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=initial_batch_size)
    
    # Setup output directory
    hyperopt_output_dir = CHECKPOINTS_DIR / f"{model_type}_hyperopt_{dataset_path.name}"
    hyperopt_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing model to use as a starting point
    initial_model_state = None
    if existing_model:
        model_checkpoint_dir = CHECKPOINTS_DIR / existing_model
        if model_checkpoint_dir.exists() and (model_checkpoint_dir / 'best_model.pth').exists():
            try:
                # Get the class count from the dataset
                num_classes = len(train_dataset.classes) if model_type != 'siamese' else 2
                
                # Create a temporary model to load the state
                temp_model = get_model(model_type, num_classes=num_classes)
                temp_model.load_state_dict(torch.load(model_checkpoint_dir / 'best_model.pth', 
                                                    map_location=device))
                initial_model_state = temp_model.state_dict()
                logger.info(f"Successfully loaded model state from {existing_model}")
            except Exception as e:
                logger.error(f"Failed to load model state: {str(e)}")
                initial_model_state = None
    
    # Objective function for optimization
    def objective(trial):
        """Optuna objective function for hyperparameter optimization."""
        # Sample hyperparameters
        batch_size = trial.suggest_categorical('batch_size', batch_sizes)
        
        # Update data loaders if batch size changed from initial
        nonlocal train_loader, val_loader
        if batch_size != train_loader.batch_size:
            if model_type == 'siamese':
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
            else:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        
        # Scheduler parameters
        scheduler_type = trial.suggest_categorical('scheduler_type', ['reduce_lr', 'cosine', 'step', 'none'])
        if scheduler_type in ['reduce_lr', 'step']:
            scheduler_patience = trial.suggest_categorical('scheduler_patience', [3, 5, 7, 10])
            scheduler_factor = trial.suggest_categorical('scheduler_factor', [0.1, 0.3, 0.5, 0.7])
        else:
            scheduler_patience = 5
            scheduler_factor = 0.5
        
        # Gradient clipping
        use_grad_clip = trial.suggest_categorical('use_grad_clip', [True, False])
        clip_grad_norm = None
        if use_grad_clip:
            clip_grad_norm = trial.suggest_categorical('clip_grad_norm', [0.5, 1.0, 3.0, 5.0])
        
        # Early stopping
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])
        early_stopping_patience = 10
        if early_stopping:
            early_stopping_patience = trial.suggest_categorical('early_stopping_patience', [5, 10, 15, 20])
        
        # Initialize model
        num_classes = len(train_dataset.classes) if model_type != 'siamese' else 2
        model = get_model(model_type, num_classes=num_classes)
        
        # Load weights from existing model if available
        if initial_model_state is not None:
            try:
                model.load_state_dict(initial_model_state)
                logger.info(f"Loaded initial weights for trial {trial.number}")
            except Exception as e:
                logger.error(f"Failed to load initial weights: {str(e)}")
        
        model = model.to(device)
        
        # Initialize optimizer and criterion
        criterion = get_criterion(model_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize scheduler
        if scheduler_type == 'reduce_lr':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=scheduler_patience, 
                factor=scheduler_factor, verbose=False
            )
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=lr/100, verbose=False
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_patience, gamma=scheduler_factor, verbose=False
            )
        else:  # 'none'
            scheduler = None
        
        # Early stopping setup
        early_stopping_counter = 0
        best_val_loss = float('inf')
        
        # Training loop
        epochs = 20
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                # ArcFace needs labels during training
                if model_type == 'arcface':
                    output = model(data, target)
                else:
                    output = model(data)
                    
                loss = criterion(output, target)
                loss.backward()
                
                # Apply gradient clipping if enabled
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    
                optimizer.step()
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    
                    # ArcFace validation
                    if model_type == 'arcface':
                        output = model(data)
                        val_classifier = nn.Linear(512, num_classes).to(device)
                        logits = val_classifier(output)
                    else:
                        output = model(data)
                        logits = output
                    
                    val_loss += criterion(logits, target).item()
                    _, pred = logits.max(1)
                    correct += int(pred.eq(target).sum().item())
                    total += target.size(0)
            
            val_loss /= len(val_loader)
            accuracy = correct / total
            
            # Update learning rate scheduler
            if scheduler_type == 'reduce_lr':
                scheduler.step(val_loss)
            elif scheduler_type in ['cosine', 'step']:
                scheduler.step()
            
            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        break
            
            # Report intermediate result
            trial.report(accuracy, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return accuracy
    
    # Create and run the Optuna study
    study = optuna.create_study(direction="maximize", study_name=f"{model_type}_{dataset_path.name}")
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
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
    
    with open(hyperopt_output_dir / 'hyperopt_results.json', 'w') as f:
        json.dump(hyperopt_final_results, f, indent=2)
    
    print(f"\nHyperparameter tuning complete!")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    print(f"Best parameters: {best_params}")
    print(f"Results saved to: {hyperopt_output_dir / 'hyperopt_results.json'}")
    
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
            scheduler_type=best_params.get('scheduler_type', 'reduce_lr'),
            scheduler_patience=best_params.get('scheduler_patience', 5),
            scheduler_factor=best_params.get('scheduler_factor', 0.5),
            clip_grad_norm=best_params.get('clip_grad_norm', None) if best_params.get('use_grad_clip', False) else None,
            early_stopping=best_params.get('early_stopping', False),
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

if __name__ == "__main__":
    run_hyperparameter_tuning() 