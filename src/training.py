#!/usr/bin/env python3

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time

# Add a type alias for the scheduler types
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
SchedulerType = Union[ReduceLROnPlateau, CosineAnnealingLR, StepLR, None]

from .base_config import PROC_DATA_DIR, CHECKPOINTS_DIR, logger
from .face_models import get_model, get_criterion
from .data_utils import SiameseDataset
from .lr_finder import LearningRateFinder
from .advanced_metrics import plot_confusion_matrix

def plot_learning_curves(train_losses: List[float], val_losses: List[float], 
                       accuracies: List[float], output_dir: str, model_name: str):
    """Plot learning curves during training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    # Create the output directory structure if it doesn't exist
    save_dir = Path(output_dir)
    if model_name:
        save_dir = save_dir / "plots" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the figure to the created directory
    plt.savefig(save_dir / 'learning_curves.png')
    plt.close()

def find_optimal_lr(model_type: str, dataset_path: Path, batch_size: int = 32,
                  start_lr: float = 1e-7, end_lr: float = 1.0, num_iterations: int = 100,
                  model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Find the optimal learning rate for a model using the LR Finder.

    Args:
        model_type: Type of model (baseline, cnn, siamese, etc.)
        dataset_path: Path to the processed dataset
        batch_size: Batch size for the dataloader
        start_lr: Starting learning rate
        end_lr: Maximum learning rate to try
        num_iterations: Number of iterations to run
        model_name: Optional name of the model (to save results in model directory)

    Returns:
        Dict containing the analysis results and suggested learning rates
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running LR Finder on device: {device}")

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
    optimizer = optim.Adam(model.parameters(), lr=start_lr)

    # Determine where to save LR Finder results
    if model_name:
        # If a model name is provided, save only in the model-specific directory
        output_dir = CHECKPOINTS_DIR / model_name / "plots" / "lr_finder"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Fall back to common directory only if no model name is provided
        output_dir = CHECKPOINTS_DIR / "lr_finder" / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LR Finder
    lr_finder = LearningRateFinder(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iterations=num_iterations,
        save_dir=output_dir  # Use the selected output directory
    )

    # Run LR finder
    logger.info(f"Running learning rate finder for {model_type} model...")
    lr_finder.find_lr(train_loader)

    # Save and analyze results to the selected output directory
    analysis = lr_finder.save_results(output_dir)

    # Log the suggested learning rates
    for group_idx, group in enumerate(analysis["groups"]):
        logger.info(f"Group {group_idx} suggested learning rate: {group['suggested_learning_rate']:.2e}")

    # Overall suggestion
    suggested_lr = analysis["overall"]["suggested_learning_rate"]
    logger.info(f"Overall suggested learning rate: {suggested_lr:.2e}")

    return analysis

def train_model(model_type: str, model_name: Optional[str] = None,
                batch_size: int = 32, epochs: int = 50,
                lr: float = 0.001, weight_decay: float = 1e-4,
                scheduler_type: str = 'reduce_lr', scheduler_patience: int = 5,
                scheduler_factor: float = 0.5, clip_grad_norm: Optional[float] = None,
                early_stopping: bool = False, early_stopping_patience: int = 10,
                dataset_path: Optional[Union[Path, List[Path]]] = None,
                use_lr_finder: bool = False):
    """Train a face recognition model with advanced parameters.
    
    Args:
        model_type: Type of model to train (baseline, cnn, siamese, etc.)
        model_name: Optional name for the model
        batch_size: Batch size for training
        epochs: Number of epochs to train
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        scheduler_type: Type of learning rate scheduler ('reduce_lr', 'cosine', 'step', 'none')
        scheduler_patience: Patience for ReduceLROnPlateau scheduler
        scheduler_factor: Factor for ReduceLROnPlateau scheduler
        clip_grad_norm: Max norm for gradient clipping (None to disable)
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        dataset_path: Optional path to the dataset or list of dataset paths
        use_lr_finder: Whether to use the learning rate finder to determine the optimal learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get the dataset - either use provided path, list of paths, or prompt for selection
    if dataset_path is None:
        # List available processed datasets
        processed_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
        if not processed_dirs:
            raise ValueError("No processed datasets found. Please process raw data first.")
        
        print("\nAvailable processed datasets:")
        for i, d in enumerate(processed_dirs, 1):
            print(f"{i}. {d.name}")
        
        # Allow selecting multiple datasets
        selected_data_dirs = []
        while not selected_data_dirs:
            dataset_choice = input("\nEnter dataset number(s) to use for training (comma-separated for multiple): ")
            try:
                # Handle comma-separated choices
                if "," in dataset_choice:
                    indices = [int(idx.strip()) - 1 for idx in dataset_choice.split(",")]
                    for idx in indices:
                        if 0 <= idx < len(processed_dirs):
                            selected_data_dirs.append(processed_dirs[idx])
                        else:
                            print(f"Invalid choice: {idx+1}. Please try again.")
                            selected_data_dirs = []
                            break
                else:
                    # Handle single choice
                    dataset_idx = int(dataset_choice) - 1
                    if 0 <= dataset_idx < len(processed_dirs):
                        selected_data_dirs.append(processed_dirs[dataset_idx])
                    else:
                        print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter valid number(s).")
    else:
        # Convert single path to list
        if isinstance(dataset_path, list):
            selected_data_dirs = dataset_path
        else:
            selected_data_dirs = [dataset_path]
        
        # Validate all paths
        for data_dir in selected_data_dirs:
            if not (data_dir / "train").exists():
                raise ValueError(f"Invalid dataset path: {data_dir}. Missing 'train' directory.")
    
    # Log all selected datasets
    dataset_names = [d.name for d in selected_data_dirs]
    logger.info(f"Using datasets: {', '.join(dataset_names)}")
    
    # Generate model name if not provided
    if model_name is None:
        existing_models = list(CHECKPOINTS_DIR.glob(f'{model_type}_*'))
        version = len(existing_models) + 1
        model_name = f"{model_type}_v{version}"
    else:
        model_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_')).lower()
        model_name = f"{model_type}_{model_name}"
    
    logger.info(f"Training model: {model_name}")
    
    # Create model-specific directories
    model_checkpoint_dir = CHECKPOINTS_DIR / model_name
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots directory
    plots_dir = model_checkpoint_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Use Learning Rate Finder if requested - use first dataset for this
    if use_lr_finder:
        logger.info("Using LR Finder to determine optimal learning rate...")
        # Pass model_name to save results ONLY in the model directory
        lr_analysis = find_optimal_lr(
            model_type=model_type,
            dataset_path=selected_data_dirs[0],  # Use first dataset for LR finder
            batch_size=batch_size,
            model_name=model_name  # Pass model name for model-specific saving only
        )
        suggested_lr = lr_analysis["overall"]["suggested_learning_rate"]
        min_lr = lr_analysis["overall"]["min_learning_rate"]
        max_lr = lr_analysis["overall"]["max_learning_rate"]
        logger.info(f"LR Finder suggested learning rate: {suggested_lr:.2e}")
        logger.info(f"LR Finder min learning rate: {min_lr:.2e}")
        logger.info(f"LR Finder max learning rate: {max_lr:.2e}")

        # Use the suggested learning rate
        lr = suggested_lr

    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize model
    # Use first dataset to determine number of classes
    if model_type == 'siamese':
        first_dataset = SiameseDataset(str(selected_data_dirs[0] / "train"), transform=transform)
        num_classes = 2
    else:
        first_dataset = datasets.ImageFolder(selected_data_dirs[0] / "train", transform=transform)
        num_classes = len(first_dataset.classes)
    
    model = get_model(model_type, num_classes=num_classes)
    model = model.to(device)
    
    # Setup training
    criterion = get_criterion(model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup learning rate scheduler
    scheduler: SchedulerType = None  # Type annotation with default value

    # For LR finder, we use the output values to set appropriate scheduler parameters
    if use_lr_finder and 'min_lr' in locals() and 'max_lr' in locals():
        # We have LR finder results we can use
        if scheduler_type == 'reduce_lr':
            # ReduceLROnPlateau: Use LR finder min_lr as the floor
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=scheduler_patience,
                factor=scheduler_factor, min_lr=min_lr
            )
            logger.info(f"Using ReduceLROnPlateau scheduler with min_lr={min_lr:.2e}")
        elif scheduler_type == 'cosine':
            # CosineAnnealingLR: Use LR finder min_lr as eta_min
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=min_lr
            )
            logger.info(f"Using CosineAnnealingLR scheduler with eta_min={min_lr:.2e}")
        elif scheduler_type == 'step':
            # Keep traditional step scheduler but log the details
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_patience, gamma=scheduler_factor
            )
            logger.info(f"Using StepLR scheduler: LR will decrease from {lr:.2e} by factor {scheduler_factor} every {scheduler_patience} epochs")
            logger.info(f"After {scheduler_patience} epochs, LR will be {lr * scheduler_factor:.2e}")
    else:
        # Standard scheduler setup without LR finder
        if scheduler_type == 'reduce_lr':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=scheduler_patience,
                factor=scheduler_factor
            )
            logger.info(f"Using ReduceLROnPlateau scheduler")
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr/100
            )
            logger.info(f"Using CosineAnnealingLR scheduler with eta_min={lr/100:.2e}")
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_patience, gamma=scheduler_factor
            )
            logger.info(f"Using StepLR scheduler with step={scheduler_patience}, gamma={scheduler_factor}")
    
    # Training variables
    train_losses = []
    val_losses = []
    accuracies = []
    best_val_acc = 0.0
    
    # Early stopping setup
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    # Train on each dataset sequentially
    for dataset_idx, selected_data_dir in enumerate(selected_data_dirs):
        logger.info(f"Training on dataset {dataset_idx+1}/{len(selected_data_dirs)}: {selected_data_dir.name}")
        
        # Load datasets for the current dataset
        if model_type == 'siamese':
            train_dataset = SiameseDataset(str(selected_data_dir / "train"), transform=transform)
            val_dataset = SiameseDataset(str(selected_data_dir / "val"), transform=transform)
            test_dataset = SiameseDataset(str(selected_data_dir / "test"), transform=transform)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        else:
            train_dataset = datasets.ImageFolder(selected_data_dir / "train", transform=transform)
            val_dataset = datasets.ImageFolder(selected_data_dir / "val", transform=transform)
            test_dataset = datasets.ImageFolder(selected_data_dir / "test", transform=transform)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Training loop for the current dataset
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            start_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                if model_type == 'siamese':
                    img1, img2, target = batch
                    img1, img2, target = img1.to(device), img2.to(device), target.to(device)
                    optimizer.zero_grad()
                    out1, out2 = model(img1, img2)
                    loss = criterion(out1, out2, target)
                else:
                    data, target = batch
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    
                    # Handle ArcFace differently
                    if model_type == 'arcface':
                        output = model(data, target)  # ArcFace needs labels during forward pass
                    else:
                        output = model(data)
                        
                    loss = criterion(output, target)
                
                loss.backward()
                
                # Apply gradient clipping if enabled
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if model_type == 'siamese':
                        img1, img2, target = batch
                        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
                        out1, out2 = model(img1, img2)
                        val_loss += criterion(out1, out2, target).item()
                        # Calculate distances and predict
                        dist = F.pairwise_distance(out1, out2)
                        pred = (dist < 0.5).float()
                        correct += int(pred.eq(target.view_as(pred)).sum().item())
                        total += target.size(0)
                        
                        # Collect predictions and targets for metrics
                        y_true.extend(target.cpu().numpy().tolist())
                        y_pred.extend(pred.cpu().numpy().tolist())
                    else:
                        data, target = batch
                        data, target = data.to(device), target.to(device)
                        
                        # Handle ArcFace differently during validation
                        if model_type == 'arcface':
                            # In validation, we just need the embeddings
                            output = model(data)
                            # For validation purposes, use separate classifier layer
                            val_classifier = nn.Linear(512, num_classes).to(device)
                            logits = val_classifier(output)
                        else:
                            output = model(data)
                            logits = output
                        
                        val_loss += criterion(logits, target).item()
                        _, pred = logits.max(1)
                        correct += int(pred.eq(target).sum().item())
                        total += target.size(0)
                        
                        # Collect predictions and targets for metrics
                        y_true.extend(target.cpu().numpy().tolist())
                        y_pred.extend(pred.cpu().numpy().tolist())
            
            # Calculate epoch metrics
            epoch_loss = train_loss / len(train_loader)
            val_epoch_loss = val_loss / len(val_loader)
            accuracy = correct / total
            
            # Store metrics for plotting
            train_losses.append(epoch_loss)
            val_losses.append(val_epoch_loss)
            accuracies.append(accuracy)
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Log metrics to console
            dataset_prefix = f"[Dataset {dataset_idx+1}/{len(selected_data_dirs)}] "
            logger.info(f'{dataset_prefix}Epoch {epoch+1}/{epochs}:')
            logger.info(f'{dataset_prefix}Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
            
            # Save best model
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                torch.save(model.state_dict(), model_checkpoint_dir / 'best_model.pth')
                print(f"{dataset_prefix}Saved best model with accuracy: {accuracy*100:.2f}%")
            
            # Update learning rate scheduler
            if scheduler_type == 'reduce_lr' and scheduler is not None:
                scheduler.step(val_epoch_loss)
            elif scheduler_type in ['cosine', 'step'] and scheduler is not None:
                scheduler.step()
            
            # Early stopping check
            if early_stopping:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"{dataset_prefix}Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Plot learning curves every 5 epochs
            if (epoch + 1) % 5 == 0:
                plot_learning_curves(train_losses, val_losses, accuracies, 
                                   str(model_checkpoint_dir), model_name)
        
        # Save checkpoint after finishing each dataset
        checkpoint_path = model_checkpoint_dir / f'checkpoint_dataset_{dataset_idx+1}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataset_name': selected_data_dir.name,
            'epoch': epochs,
            'accuracy': accuracies[-1]
        }, checkpoint_path)
        logger.info(f"Saved checkpoint after training on {selected_data_dir.name}")
    
    # Plot final learning curves
    plot_learning_curves(train_losses, val_losses, accuracies, 
                       str(model_checkpoint_dir), model_name)
    
    # Save final model
    torch.save(model.state_dict(), model_checkpoint_dir / 'final_model.pth')
    
    # Evaluation on test set of the last dataset
    logger.info("Evaluating on test set of the last dataset...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'siamese':
                img1, img2, target = batch
                img1, img2, target = img1.to(device), img2.to(device), target.to(device)
                out1, out2 = model(img1, img2)
                # Calculate distances and predict
                dist = F.pairwise_distance(out1, out2)
                pred = (dist < 0.5).float()
                test_loss += criterion(out1, out2, target).item()
                correct += int(pred.eq(target.view_as(pred)).sum().item())
                total += target.size(0)
                
                # Collect predictions and targets for metrics
                all_y_true.extend(target.cpu().numpy().tolist())
                all_y_pred.extend(pred.cpu().numpy().tolist())
            else:
                data, target = batch
                data, target = data.to(device), target.to(device)
                
                # Handle ArcFace differently during evaluation
                if model_type == 'arcface':
                    output = model(data)
                    # For evaluation purposes, use separate classifier layer
                    test_classifier = nn.Linear(512, num_classes).to(device)
                    logits = test_classifier(output)
                else:
                    output = model(data)
                    logits = output
                
                test_loss += criterion(logits, target).item()
                _, pred = logits.max(1)
                correct += int(pred.eq(target).sum().item())
                total += target.size(0)
                
                # Collect predictions and targets for metrics
                all_y_true.extend(target.cpu().numpy().tolist())
                all_y_pred.extend(pred.cpu().numpy().tolist())
    
    # Calculate test accuracy
    test_accuracy = correct / total
    logger.info(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Generate confusion matrix for non-siamese models
    if model_type != 'siamese':
        all_y_true_arr = np.array(all_y_true)
        all_y_pred_arr = np.array(all_y_pred)
        
        cm = confusion_matrix(all_y_true_arr, all_y_pred_arr)
        plot_confusion_matrix(
            y_true=all_y_true_arr,
            y_pred=all_y_pred_arr,
            classes=train_dataset.classes, 
            output_dir=str(plots_dir),
            model_name=model_name,
            detailed=True
        )
    
    # Save model info
    model_info = {
        'model_type': model_type,
        'datasets': [d.name for d in selected_data_dirs],
        'num_classes': num_classes,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'scheduler_type': scheduler_type,
        'test_accuracy': test_accuracy,
        'best_validation_accuracy': best_val_acc
    }
    
    with open(model_checkpoint_dir / 'model_info.json', 'w') as f:
        import json
        json.dump(model_info, f, indent=4)
    
    logger.info(f"Model training complete: {model_name}")
    
    return model_name

def tune_hyperparameters(model_type: str, dataset_path: Path, n_trials: int = 10) -> Dict[str, Any]:
    """Run hyperparameter tuning for a model (simplified version)."""
    print("Using hyperparameter_tuning.py for this functionality.")
    print("Please use 'python run.py hyperopt' instead.")
    return None 