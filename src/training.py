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

def plot_confusion_matrix(cm, class_names, output_dir, model_name):
    """
    Plot confusion matrix as a heatmap.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Create the output directory structure if it doesn't exist
    save_dir = Path(output_dir)
    if model_name:
        save_dir = save_dir / "plots" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()

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

def train_model(model_type: str, model_name: Optional[str] = None,
                batch_size: int = 32, epochs: int = 50,
                lr: float = 0.001, weight_decay: float = 1e-4,
                scheduler_type: str = 'reduce_lr', scheduler_patience: int = 5,
                scheduler_factor: float = 0.5, clip_grad_norm: Optional[float] = None,
                early_stopping: bool = False, early_stopping_patience: int = 10,
                dataset_path: Optional[Path] = None):
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
        dataset_path: Optional path to the dataset (if None, will prompt for selection)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get the dataset - either use provided path or prompt for selection
    if dataset_path is None:
        # List available processed datasets
        processed_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
        if not processed_dirs:
            raise ValueError("No processed datasets found. Please process raw data first.")
        
        print("\nAvailable processed datasets:")
        for i, d in enumerate(processed_dirs, 1):
            print(f"{i}. {d.name}")
        
        while True:
            dataset_choice = input("\nEnter dataset number to use for training: ")
            try:
                dataset_idx = int(dataset_choice) - 1
                if 0 <= dataset_idx < len(processed_dirs):
                    selected_data_dir = processed_dirs[dataset_idx]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        # Use provided dataset path
        selected_data_dir = dataset_path
        if not (selected_data_dir / "train").exists():
            raise ValueError(f"Invalid dataset path: {selected_data_dir}. Missing 'train' directory.")
    
    logger.info(f"Using dataset: {selected_data_dir.name}")
    
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
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
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
    
    # Initialize model
    num_classes = len(train_dataset.classes) if model_type != 'siamese' else 2
    model = get_model(model_type, num_classes=num_classes)
    model = model.to(device)
    
    # Setup training
    criterion = get_criterion(model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup learning rate scheduler
    scheduler: SchedulerType = None  # Type annotation with default value
    if scheduler_type == 'reduce_lr':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=scheduler_patience, 
            factor=scheduler_factor, verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr/100, verbose=True
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_patience, gamma=scheduler_factor, verbose=True
        )
    
    # Training loop
    train_losses = []
    val_losses = []
    accuracies = []
    best_val_acc = 0.0
    
    # Early stopping setup
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
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
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
        
        # Save best model
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), model_checkpoint_dir / 'best_model.pth')
            print(f"Saved best model with accuracy: {accuracy*100:.2f}%")
        
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
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Plot learning curves every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_learning_curves(train_losses, val_losses, accuracies, 
                               str(model_checkpoint_dir), model_name)
    
    # Plot final learning curves
    plot_learning_curves(train_losses, val_losses, accuracies, 
                       str(model_checkpoint_dir), model_name)
    
    # Evaluation on test set
    logger.info("Evaluating on test set...")
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
                    eval_classifier = nn.Linear(512, num_classes).to(device)
                    logits = eval_classifier(output)
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
    
    # Calculate test metrics
    test_loss /= len(test_loader)
    accuracy = correct / total
    
    # Print test results
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    
    # Save test results
    test_results = {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "model_type": model_type,
        "dataset": selected_data_dir.name,
        "training_config": {
            "batch_size": batch_size,
            "epochs": epoch + 1,  # Actual number of epochs run
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "scheduler": scheduler_type,
            "gradient_clipping": clip_grad_norm,
            "early_stopping": early_stopping,
        }
    }
    
    with open(model_checkpoint_dir / "test_results.json", "w") as f:
        import json
        json.dump(test_results, f, indent=2)
    
    # Generate confusion matrix for non-siamese models
    if model_type != 'siamese':
        all_y_true_arr = np.array(all_y_true)
        all_y_pred_arr = np.array(all_y_pred)
        
        cm = confusion_matrix(all_y_true_arr, all_y_pred_arr)
        plot_confusion_matrix(
            cm, train_dataset.classes, 
            str(model_checkpoint_dir), model_name
        )
        
        # Generate classification report
        report = classification_report(
            all_y_true_arr, all_y_pred_arr, 
            target_names=train_dataset.classes, 
            output_dict=True
        )
        
        # Save classification report
        with open(model_checkpoint_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
    
    print(f"\nTraining complete! Model saved at {model_checkpoint_dir / 'best_model.pth'}")
    print(f"Visualizations saved in {model_checkpoint_dir / 'plots'}")
    
    return model_name

def tune_hyperparameters(model_type: str, dataset_path: Path, n_trials: int = 10) -> Dict[str, Any]:
    """Run hyperparameter tuning for a model (simplified version)."""
    print("Using hyperparameter_tuning.py for this functionality.")
    print("Please use 'python run.py hyperopt' instead.")
    return None 