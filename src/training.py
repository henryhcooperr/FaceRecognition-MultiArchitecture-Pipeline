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
from sklearn.metrics import confusion_matrix, classification_report
import time
import threading
import math  # For cosine scheduler calculations
import pandas as pd  # Also import pandas since we use it for saving metrics

# Add a type alias for the scheduler types
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
SchedulerType = Union[ReduceLROnPlateau, CosineAnnealingLR, StepLR, None]

from .base_config import PROC_DATA_DIR, CHECKPOINTS_DIR, logger
from .face_models import get_model, get_criterion
from .data_utils import SiameseDataset
from .lr_finder import LearningRateFinder
from .advanced_metrics import plot_confusion_matrix

def plot_learning_curves(train_losses: List[float], val_losses: List[float], 
                       accuracies: List[float], output_dir: str, model_name: str,
                       train_accuracies: Optional[List[float]] = None):
    """
    Record learning curves metrics without plotting. Maintains API compatibility.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        accuracies: List of validation accuracies
        output_dir: Directory to save metrics (not used for visualization)
        model_name: Name of the model
        train_accuracies: Optional list of training accuracies
    """
    # Create the output directory structure for metrics
    save_dir = Path(output_dir)
    if model_name:
        save_dir = save_dir / "metrics" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to a CSV file for future reference
    epochs = list(range(1, len(train_losses) + 1))
    metrics_dict = {
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_accuracy': accuracies
    }
    
    # Add train accuracies if available
    if train_accuracies is not None:
        metrics_dict['train_accuracy'] = train_accuracies
    
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(save_dir / 'learning_curves_data.csv', index=False)
    
    # Log summary statistics
    logger.info(f"Learning curves data saved to {save_dir / 'learning_curves_data.csv'}")
    logger.info(f"Final metrics - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {accuracies[-1]:.4f}")

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
    
    # Initialize LR Finder with model-specific adjustments
    lr_finder = LearningRateFinder(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        start_lr=start_lr,
        # Use model-specific end_lr
        end_lr=0.01 if model_type == 'arcface' else (0.1 if model_type == 'siamese' else end_lr),
        num_iterations=num_iterations,
        save_dir=output_dir,  # Use the selected output directory
        model_type=model_type  # Pass model type for specialized scaling
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

def get_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    """Creates learning rate scheduler with warm-up phase followed by cosine annealing.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_epochs: Number of epochs for the warm-up phase
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        A learning rate scheduler with warm-up
    """
    def lr_lambda(current_step):
        warmup_steps = warmup_epochs * steps_per_epoch
        if current_step < warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing after warm-up
            progress = float(current_step - warmup_steps) / float(max(1, total_epochs * steps_per_epoch - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(model_type: str, model_name: Optional[str] = None,
                batch_size: int = 32, epochs: int = 50,
                lr: float = 0.001, weight_decay: float = 1e-4,
                scheduler_type: str = 'reduce_lr', scheduler_patience: int = 5,
                scheduler_factor: float = 0.5, clip_grad_norm: Optional[float] = None,
                early_stopping: bool = False, early_stopping_patience: int = 10,
                dataset_path: Optional[Union[Path, List[Path]]] = None,
                use_lr_finder: bool = False, use_warmup: bool = False,
                warmup_epochs: int = 5, easy_margin: bool = False,
                two_phase_training: bool = False):
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
        logger.info(f"Using model-specific learning rate ranges for {model_type}")
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
    
    # Create model with special handling for ArcFace
    if model_type == 'arcface':
        # Initialize ArcFace with advanced parameters
        from .face_models import ArcFaceNet  # Import for direct initialization
        model = ArcFaceNet(
            num_classes=num_classes,
            dropout_rate=0.2,  # Lower dropout for better convergence
            s=32.0,  # Increased scale from 16.0 to 32.0
            m=0.5,  # Standard margin
            easy_margin=easy_margin  # Support for easy margin option
        )
        logger.info(f"Created ArcFace model with enhanced parameters: easy_margin={easy_margin}")
    else:
        model = get_model(model_type, num_classes=num_classes)
    
    model = model.to(device)
    
    # Setup training
    # Setup criterion and optimizer with special handling for ArcFace
    if model_type == 'arcface':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Use label smoothing for ArcFace
        # Use AMSGrad variant for better convergence with ArcFace
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            amsgrad=True  # Use AMSGrad for better convergence
        )
        logger.info(f"Using AdamW with AMSGrad for ArcFace training")
    else:
        criterion = get_criterion(model_type)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup learning rate scheduler
    scheduler: SchedulerType = None  # Type annotation with default value
    
    # Apply special warmup scheduler for ArcFace if needed
    if model_type == 'arcface' and use_warmup:
        # Setup learning rate warm-up
        steps_per_epoch = len(train_loader) if 'train_loader' in locals() else 100
        scheduler = get_warmup_scheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            steps_per_epoch=steps_per_epoch
        )
        logger.info(f"Using warm-up scheduler for ArcFace with {warmup_epochs} epochs of warm-up")
        
        # Also configure model-specific settings if available
        if hasattr(model, 'set_max_grad_norm') and clip_grad_norm is not None:
            model.set_max_grad_norm(clip_grad_norm)
            logger.info(f"Set ArcFace model max_grad_norm to {clip_grad_norm}")
            
        # Initialize two-phase training if requested
        if two_phase_training and hasattr(model, 'freeze_backbone'):
            model.freeze_backbone()
            logger.info("Initialized two-phase training: backbone frozen for initial training phase")
            
        # Skip the regular scheduler setup since we're using custom warmup
        if scheduler_type != 'warmup':
            logger.info(f"Note: Ignoring {scheduler_type} scheduler since warm-up scheduler is being used")

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
    train_accuracies = []  # Store training accuracies for metrics
    best_val_acc = 0.0
    
    # Create CSV file for detailed epoch-by-epoch metrics
    metrics_dir = Path(model_checkpoint_dir) / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)
    metrics_csv_path = metrics_dir / f"{model_name}_training_metrics.csv"
    
    # Write header to CSV file
    with open(metrics_csv_path, 'w') as f:
        header = "epoch,dataset,train_loss,train_acc,val_loss,val_acc,best_val_acc,lr,time_elapsed\n"
        f.write(header)
    
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
            
            # Calculate number of batches to process (limit to prevent freezing)
            max_train_batches = min(len(train_loader), 100)  # Limit to 100 batches max
            
            for batch_idx, batch in enumerate(train_loader):
                # Break after maximum number of batches to prevent freezing
                if batch_idx >= max_train_batches:
                    logger.info(f"Reached maximum training batches ({max_train_batches}/{len(train_loader)}). Moving to validation...")
                    break
                
                if model_type == 'siamese':
                    try:
                        img1, img2, target = batch
                        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
                        optimizer.zero_grad()
                        out1, out2 = model(img1, img2)
                        loss = criterion(out1, out2, target)
                        
                        # Add progress logging
                        if batch_idx % 5 == 0:
                            logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{max_train_batches} | Loss: {loss.item():.4f}")
                        
                        # Add timeout check for long batches
                        if time.time() - start_time > 30:  # 30 second timeout per batch
                            logger.warning(f"Batch {batch_idx} taking too long (>30s). Skipping to next batch.")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error in training batch {batch_idx}: {e}")
                        continue
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
                
                # Safely perform backpropagation with timeouts and error handling
                try:
                    # Set a timer for gradient computation
                    start_backward = time.time()
                    loss.backward()
                    
                    # Check if backward pass took too long
                    if time.time() - start_backward > 10:  # 10 second timeout for backward pass
                        logger.warning(f"Backward pass taking too long (>{time.time() - start_backward:.1f}s) for batch {batch_idx}")
                    
                    # Apply gradient clipping if enabled
                    if clip_grad_norm is not None:
                        # Enhanced gradient clipping for ArcFace
                        if model_type == 'arcface':
                            # Use model's max_grad_norm if available, or the provided clip_grad_norm
                            max_norm = model.max_grad_norm if hasattr(model, 'max_grad_norm') else clip_grad_norm
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                            
                            # Log gradient norms occasionally
                            if batch_idx % 50 == 0 and hasattr(model, 'get_arcface_stats'):
                                stats = model.get_arcface_stats()
                                if stats.get('grad_norm', 0) > 0.5 * max_norm:
                                    logger.info(f"High gradient norm: {stats.get('grad_norm', 0):.3f} (threshold: {max_norm})")
                        else:
                            # Standard gradient clipping for other models
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    
                    # Set timer for optimizer step
                    start_optim = time.time()
                    optimizer.step()
                    
                    # Check if optimizer step took too long
                    if time.time() - start_optim > 5:  # 5 second timeout for optimizer step
                        logger.warning(f"Optimizer step taking too long (>{time.time() - start_optim:.1f}s) for batch {batch_idx}")
                    
                    train_loss += loss.item()
                    
                except RuntimeError as e:
                    logger.error(f"Runtime error in batch {batch_idx}: {e}")
                    if "out of memory" in str(e).lower():
                        logger.error("GPU out of memory error. Skipping this batch.")
                    continue
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            y_true = []
            y_pred = []
            
            # Optimize by using smaller validation samples to prevent freezing
            val_sample_size = min(len(val_loader), 20)  # Limit validation to 20 batches max
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Break after processing val_sample_size batches to prevent freezing
                    if batch_idx >= val_sample_size:
                        break
                        
                    if model_type == 'siamese':
                        img1, img2, target = batch
                        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
                        
                        # Forward pass with error handling
                        try:
                            out1, out2 = model(img1, img2)
                            batch_loss = criterion(out1, out2, target).item()
                            val_loss += batch_loss
                            
                            # Calculate distances and predict
                            dist = F.pairwise_distance(out1, out2)
                            pred = (dist < 0.5).float()
                            correct += int(pred.eq(target.view_as(pred)).sum().item())
                            total += target.size(0)
                            
                            # Log validation progress
                            if batch_idx % 2 == 0:  # Increase logging frequency
                                current_acc = correct / max(1, total)
                                logger.info(f"Validation | Batch {batch_idx}/{val_sample_size} | Loss: {batch_loss:.4f} | Current Acc: {current_acc*100:.2f}%")
                            
                            # Collect predictions and targets for metrics
                            y_true.extend(target.cpu().numpy().tolist())
                            y_pred.extend(pred.cpu().numpy().tolist())
                            
                        except Exception as e:
                            logger.error(f"Error during validation: {e}")
                            # Continue with next batch if there's an error
                            continue
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
            train_accuracies.append(train_acc)  # Store the training accuracy
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print validation accuracy to console for every epoch
            print(f"\nEpoch {epoch+1}/{epochs} - Dataset: {selected_data_dir.name}")
            print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {accuracy:.4f}, Best Val Acc: {best_val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Log metrics to CSV file
            with open(metrics_csv_path, 'a') as f:
                f.write(f"{epoch+1},{selected_data_dir.name},{epoch_loss:.6f},{train_acc:.6f},")
                f.write(f"{val_epoch_loss:.6f},{accuracy:.6f},{best_val_acc:.6f},{current_lr:.8f},{epoch_time:.2f}\n")
            
            # Print more detailed accuracy metrics for Siamese networks
            if model_type == 'siamese':
                # Calculate metrics by class
                y_true_np = np.array(y_true)
                y_pred_np = np.array(y_pred)
                
                # Class 0 = same person, Class 1 = different person
                class0_correct = np.sum((y_true_np == 0) & (y_pred_np == 0))
                class0_total = np.sum(y_true_np == 0)
                class1_correct = np.sum((y_true_np == 1) & (y_pred_np == 1))
                class1_total = np.sum(y_true_np == 1)
                
                # Calculate class accuracies
                class0_acc = class0_correct / max(1, class0_total)
                class1_acc = class1_correct / max(1, class1_total)
                
                logger.info(f"Same Person Accuracy: {class0_acc*100:.2f}% ({class0_correct}/{class0_total})")
                logger.info(f"Different Person Accuracy: {class1_acc*100:.2f}% ({class1_correct}/{class1_total})")
            
            # Log metrics to console
            dataset_prefix = f"[Dataset {dataset_idx+1}/{len(selected_data_dirs)}] "
            logger.info(f'{dataset_prefix}Epoch {epoch+1}/{epochs}:')
            logger.info(f'{dataset_prefix}Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
            logger.info(f'{dataset_prefix}Epoch time: {epoch_time:.2f}s')
            
            # Check if we need to transition to phase 2 for ArcFace with two-phase training
            # This transition happens after approximately 1/3 of total epochs
            if model_type == 'arcface' and two_phase_training and hasattr(model, 'phase') and hasattr(model, 'unfreeze_backbone'):
                phase1_epochs = max(10, epochs // 3)
                if epoch == phase1_epochs and model.phase == 1:
                    logger.info(f"Transitioning ArcFace model to phase 2 (full fine-tuning) after {phase1_epochs} epochs")
                    model.unfreeze_backbone()
                    
                    # Optionally reduce learning rate for fine-tuning phase
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
                    logger.info(f"Reduced learning rate to {optimizer.param_groups[0]['lr']:.2e} for phase 2")
                    
                    # Log model stats after transition
                    if hasattr(model, 'get_arcface_stats'):
                        stats = model.get_arcface_stats()
                        logger.info(f"ArcFace phase 2 stats - phase: {stats['phase']}, margin: {stats['effective_margin']:.4f}")
                        logger.info(f"Backbone frozen: {stats['backbone_frozen']}")
            
            # Update ArcFace epoch tracking for progressive margin
            if model_type == 'arcface' and hasattr(model, 'update_epoch'):
                model.update_epoch(epoch)
                
                # Log ArcFace stats every 5 epochs
                if epoch % 5 == 0 and hasattr(model, 'get_arcface_stats'):
                    stats = model.get_arcface_stats()
                    logger.info(f"ArcFace epoch {epoch} stats - margin: {stats.get('effective_margin', 0):.3f}, "
                               f"scale: {stats.get('effective_scale', 0):.1f}, "
                               f"grad_norm: {stats.get('grad_norm', 0):.3f}")
            
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
            
            # Save metrics every 5 epochs, but only if we have enough data
            if (epoch + 1) % 5 == 0 and len(train_losses) >= 3:
                try:
                    # Save metrics data
                    plot_learning_curves(train_losses, val_losses, accuracies, 
                                        str(model_checkpoint_dir), model_name)
                    logger.info(f"Saved learning curve metrics for epoch {epoch+1}")
                except Exception as e:
                    logger.error(f"Error saving metrics: {e}")
        
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
    
    # Save final metrics data
    logger.info("Saving final learning curves metrics...")
    try:
        plot_learning_curves(train_losses, val_losses, accuracies, 
                          str(model_checkpoint_dir), model_name,
                          train_accuracies=train_accuracies)
        
        # Show final summary of training results
        print("\n========== TRAINING COMPLETE ==========")
        print(f"Model: {model_name}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Final validation accuracy: {accuracies[-1]:.4f}")
        print(f"Detailed metrics saved to: {metrics_csv_path}")
        print("======================================")
        
    except Exception as e:
        logger.error(f"Error saving final metrics: {e}")
    
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
    
    # Limit test evaluation sample size to prevent freezing
    test_sample_size = min(len(test_loader), 30)  # Use max 30 batches for testing
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Break after processing test_sample_size batches to prevent freezing
            if batch_idx >= test_sample_size:
                logger.info(f"Using the first {test_sample_size} batches for evaluation to prevent freezing")
                break
                
            # Log progress more frequently
            if batch_idx % 5 == 0:
                logger.info(f"Test evaluation: batch {batch_idx}/{test_sample_size}")
                
            if model_type == 'siamese':
                try:
                    img1, img2, target = batch
                    img1, img2, target = img1.to(device), img2.to(device), target.to(device)
                    out1, out2 = model(img1, img2)
                    # Calculate distances and predict
                    dist = F.pairwise_distance(out1, out2)
                    pred = (dist < 0.5).float()
                    batch_loss = criterion(out1, out2, target).item()
                    test_loss += batch_loss
                    correct += int(pred.eq(target.view_as(pred)).sum().item())
                    total += target.size(0)
                    
                    # Log test progress with accuracy
                    if batch_idx % 5 == 0:
                        current_acc = correct / max(1, total)
                        logger.info(f"Test | Batch {batch_idx}/{test_sample_size} | Loss: {batch_loss:.4f} | Current Acc: {current_acc*100:.2f}%")
                    
                    # Collect predictions and targets for metrics
                    all_y_true.extend(target.cpu().numpy().tolist())
                    all_y_pred.extend(pred.cpu().numpy().tolist())
                    
                except Exception as e:
                    logger.error(f"Error during test evaluation: {e}")
                    continue
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
    
    # Create a more comprehensive metrics file
    metrics_dir = model_checkpoint_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    # Save detailed metrics for analysis
    all_metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'accuracies': accuracies,
        'epochs': list(range(1, len(accuracies) + 1)),
        'final_test_accuracy': test_accuracy,
        'best_validation_accuracy': best_val_acc
    }
    
    # Save metrics as CSV for easy import to Excel/plotting tools
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(accuracies) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'accuracy': accuracies
    })
    metrics_df.to_csv(metrics_dir / 'learning_curves.csv', index=False)
    logger.info(f"Saved metrics CSV to {metrics_dir / 'learning_curves.csv'}")
    
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
        'best_validation_accuracy': best_val_acc,
        'metrics_saved_at': str(metrics_dir),
        'checkpoint_dir': str(model_checkpoint_dir)
    }
    
    # Save to both locations for convenience
    with open(model_checkpoint_dir / 'model_info.json', 'w') as f:
        import json
        json.dump(model_info, f, indent=4)
        
    with open(metrics_dir / 'model_info.json', 'w') as f:
        import json
        json.dump(model_info, f, indent=4)
        
    # Log the locations for user reference
    logger.info(f"Model checkpoints saved to: {model_checkpoint_dir}")
    logger.info(f"Metrics saved to: {metrics_dir}")
    
    logger.info(f"Model training complete: {model_name}")
    
    return model_name

def tune_hyperparameters(model_type: str, dataset_path: Path, n_trials: int = 10) -> Dict[str, Any]:
    """Run hyperparameter tuning for a model (simplified version)."""
    print("Using hyperparameter_tuning.py for this functionality.")
    print("Please use 'python run.py hyperopt' instead.")
    return None 