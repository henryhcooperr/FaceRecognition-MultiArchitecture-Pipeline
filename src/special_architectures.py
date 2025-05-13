#!/usr/bin/env python3

"""
Special handlers for architecture-specific training and evaluation routines.
This file provides custom implementations for architectures that require
special handling like Siamese networks and ArcFace.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .face_models import get_model, get_criterion
from .base_config import logger
from .training_utils import apply_gradient_clipping, save_checkpoint
from .advanced_metrics import create_enhanced_confusion_matrix, calculate_per_class_metrics, expected_calibration_error

class ArcFaceTrainer:
    """Trainer for ArcFace-based models."""
    
    def __init__(self):
        """Initialize the ArcFace trainer."""
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.config = None
        self.results_manager = None
    
    def train(self):
        """Train the ArcFace model."""
        if not all([self.model, self.train_dataset, self.val_dataset, self.config, self.results_manager]):
            raise ValueError("ArcFaceTrainer not fully initialized. Set model, datasets, config, and results_manager attributes.")
        
        # Call the existing training method
        return self.train_arcface_network()
    
    def test(self, test_dataset):
        """Test the ArcFace model on the provided test dataset."""
        from torch.utils.data import DataLoader
        import torch
        
        # Create test dataloader with appropriate transforms
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test the model using the existing test_model function
        from .testing import test_model
        test_result = test_model(self.model, test_dataloader, device)
        
        # Record test metrics
        self.results_manager.record_test_metrics(test_result)
        
        # Generate visualizations
        try:
            # Get predictions and true labels
            y_true = test_result.get('y_true', [])
            y_pred = test_result.get('y_pred', [])
            y_score = test_result.get('y_score', [])
            
            # Convert to numpy arrays
            import numpy as np
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_score = np.array(y_score)
            
            # Make sure we have data to create visualizations
            if len(y_true) > 0 and len(y_pred) > 0 and len(y_score) > 0:
                # Get class names
                classes = test_dataset.classes if hasattr(test_dataset, 'classes') else None
                
                # Generate confusion matrix visualization
                self.results_manager.record_confusion_matrix(y_true, y_pred, classes)
                
                # Generate per-class metrics and visualizations
                if hasattr(self.config, 'per_class_analysis') and self.config.per_class_analysis:
                    self.results_manager.record_per_class_metrics(y_true, y_pred, y_score, classes)
                
                # Generate calibration visualizations
                if hasattr(self.config, 'calibration_analysis') and self.config.calibration_analysis:
                    self.results_manager.record_calibration_metrics(y_true, y_pred, y_score)
                    
                print(f"Generated visualizations for ArcFace model")
        except Exception as viz_error:
            print(f"Error generating visualizations for ArcFace model: {str(viz_error)}")
        
        return test_result

    def train_arcface_network(self):
        """
        Custom training routine for ArcFace networks.
        
        Returns:
            Dict: Training results
        """
        import torch
        import torch.optim as optim
        import torch.nn.functional as F
        from pathlib import Path
        import time
        import numpy as np
        from torch.utils.data import DataLoader
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Get criterion
        criterion = get_criterion('arcface')
        
        # Set up optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Set up learning rate scheduler if requested
        scheduler = None
        if hasattr(self.config, 'lr_scheduler_type') and self.config.lr_scheduler_type != 'none':
            from .training_utils import get_scheduler
            scheduler_type = self.config.lr_scheduler_type
            scheduler_params = self.config.lr_scheduler_params or {}
            
            scheduler = get_scheduler(
                scheduler_type=scheduler_type,
                optimizer=optimizer,
                **scheduler_params
            )
        
        # Set up early stopping if requested
        early_stopping = None
        if hasattr(self.config, 'use_early_stopping') and self.config.use_early_stopping:
            from .training_utils import EarlyStopping
            early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                mode=self.config.early_stopping_mode
            )
        
        # Training loop variables
        best_val_accuracy = 0
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        start_epoch = 1
        
        # Check if training should be resumed from checkpoint
        if hasattr(self.config, 'resumable_training') and self.config.resumable_training:
            checkpoints_dir = Path(self.config.results_dir) / "checkpoints"
            if checkpoints_dir.exists():
                # Fix: correctly use list() on the result of glob()
                checkpoint_files = list((Path(self.config.results_dir) / "checkpoints").glob("checkpoint_epoch_*.pth"))
                if checkpoint_files:
                    # Sort by epoch number
                    checkpoint_files.sort(key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
                    latest_checkpoint = checkpoint_files[-1]
                    
                    print(f"Resuming training from checkpoint: {latest_checkpoint}")
                    checkpoint = torch.load(latest_checkpoint, map_location=device)
                    
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    # Restore scheduler state if available
                    if scheduler and 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                    # Set start epoch
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Starting from epoch {start_epoch}")
        
        # Start training
        training_start_time = time.time()
        
        for epoch in range(start_epoch, self.config.epochs + 1):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            epoch_start_time = time.time()
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - ArcFace requires labels during training
                outputs = self.model(inputs, labels)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping if enabled
                if hasattr(self.config, 'use_gradient_clipping') and self.config.use_gradient_clipping:
                    from .training_utils import apply_gradient_clipping
                    apply_gradient_clipping(
                        model=self.model,
                        max_norm=self.config.gradient_clipping_max_norm,
                        adaptive=self.config.gradient_clipping_adaptive,
                        model_type='arcface'
                    )
                
                # Optimize
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            # Record training metrics
            self.results_manager.record_training_metrics(epoch, {
                "loss": epoch_loss,
                "epoch_time": time.time() - epoch_start_time
            })
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            val_embeddings = []
            val_labels_list = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Get embeddings for evaluation
                    embeddings = self.model(inputs)
                    val_embeddings.append(embeddings)
                    val_labels_list.append(labels)
                    
                    # Calculate similarity for classification - use same approach as main branch
                    # Skip normalization since it's now handled in the ArcFace module
                    logits = F.linear(
                        embeddings, 
                        self.model.arcface.weight
                    )
                    loss = criterion(logits, labels)
                    
                    _, preds = torch.max(logits, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    val_loss += loss.item()
            
            epoch_val_loss = val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            
            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            
            # Record validation metrics
            val_metrics = {
                "loss": epoch_val_loss,
                "accuracy": accuracy
            }
            self.results_manager.record_evaluation_metrics(epoch, val_metrics)
            
            # Step learning rate scheduler if it's a validation-based scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric_value = epoch_val_loss if hasattr(self.config, 'early_stopping_metric') and self.config.early_stopping_metric == "loss" else -accuracy
                    scheduler.step(metric_value)
            
            # Save checkpoint if this is the best model
            is_best = False
            if hasattr(self.config, 'early_stopping_mode'):
                if self.config.early_stopping_mode == 'max':
                    # For metrics like accuracy where higher is better
                    is_best = accuracy > best_val_accuracy
                    if is_best:
                        best_val_accuracy = accuracy
                else:
                    # For metrics like loss where lower is better
                    is_best = epoch_val_loss < best_val_loss
                    if is_best:
                        best_val_loss = epoch_val_loss
            else:
                # Default to accuracy if not specified
                is_best = accuracy > best_val_accuracy
                if is_best:
                    best_val_accuracy = accuracy
            
            # Save checkpoint with appropriate frequency
            if hasattr(self.config, 'checkpoint_frequency') and (epoch % self.config.checkpoint_frequency == 0 or is_best):
                self.results_manager.save_model_checkpoint(
                    model=self.model, 
                    optimizer=optimizer, 
                    epoch=epoch, 
                    is_best=is_best,
                    scheduler=scheduler,
                    metrics=val_metrics
                )
            
            # Log progress
            print(f'Epoch {epoch}/{self.config.epochs}, '
                  f'Train Loss: {epoch_loss:.4f}, '
                  f'Val Loss: {epoch_val_loss:.4f}, '
                  f'Accuracy: {accuracy:.2f}%, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
                  f'Time: {time.time() - epoch_start_time:.2f}s')
            
            # Step learning rate scheduler if it's an epoch-based scheduler
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            
            # Check early stopping if enabled
            if early_stopping:
                # Use appropriate metric for early stopping
                if hasattr(self.config, 'early_stopping_metric') and self.config.early_stopping_metric == "loss":
                    early_stop = early_stopping(epoch_val_loss)
                else:  # Use accuracy
                    # Negate accuracy for 'min' mode
                    es_value = -accuracy if early_stopping.mode == 'min' else accuracy
                    early_stop = early_stopping(es_value)
                
                # Check if training should be stopped
                if early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    # Save early stopping trace to a file
                    with open(Path(self.config.results_dir) / "logs" / "early_stopping_trace.json", 'w') as f:
                        import json
                        json.dump({
                            "trace": early_stopping.trace,
                            "stopped_epoch": epoch,
                            "best_score": early_stopping.best_score,
                            "mode": early_stopping.mode
                        }, f, indent=2)
                    break
        
        total_training_time = time.time() - training_start_time
        print(f"Training completed in {total_training_time:.2f}s")
        
        # Record learning curves
        self.results_manager.record_learning_curves(train_losses, val_losses, val_accuracies)
        
        # Return training summary
        return {
            "training_time": total_training_time,
            "epochs": epoch - start_epoch + 1,
            "best_validation_accuracy": best_val_accuracy,
            "best_validation_loss": best_val_loss
        }

def handle_special_architecture(architecture, model, train_dataset, val_dataset, test_dataset, config, results_manager):
    """
    Special handling for architectures that require custom training procedures.
    
    Returns:
        tuple: (results, was_handled) - results dictionary and whether architecture was handled
    """
    try:
        # Handle ArcFace network specially
        if architecture == "arcface":
            if hasattr(model, 'arcface'):
                print(f"Using special handling for ArcFace architecture")
                
                # Fix for ArcFaceTrainer - instantiate properly based on its constructor
                # If it takes no arguments, create it first then set attributes
                trainer = ArcFaceTrainer()
                trainer.model = model
                trainer.train_dataset = train_dataset
                trainer.val_dataset = val_dataset
                trainer.config = config
                trainer.results_manager = results_manager
                
                # Train the model
                results = trainer.train()
                
                # Test the model
                print(f"Running specialized testing for ArcFace...")
                results = trainer.test(test_dataset)
                
                return results, True
                
        return None, False
    
    except Exception as e:
        print(f"Error handling special architecture {architecture}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return None, False