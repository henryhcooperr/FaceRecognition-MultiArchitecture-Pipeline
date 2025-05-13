#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any, Callable
from enum import Enum
import datetime
import json
import glob

from .base_config import logger


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss doesn't improve after
    a given patience period.
    
    Attributes:
        patience (int): How many epochs to wait after last improvement
        min_delta (float): Minimum change to qualify as an improvement
        counter (int): Current count of epochs with no improvement
        best_score (float): Best score observed so far
        early_stop (bool): Whether to stop training
        trace (list): History of validation scores
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping object.
        
        Args:
            patience: Number of epochs with no improvement after which training will stop
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for metrics where lower is better (e.g., loss), 'max' where higher is better (e.g., accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        self.early_stop = False
        self.trace = []
        
        if mode == 'min':
            self.best_score = float('inf')
            self.val_comp = lambda score: score < (self.best_score - self.min_delta)
        elif mode == 'max':
            self.best_score = float('-inf')
            self.val_comp = lambda score: score > (self.best_score + self.min_delta)
        else:
            raise ValueError(f"Mode {mode} is not supported. Use 'min' or 'max'")
    
    def __call__(self, val_score: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_score: Current validation score
            
        Returns:
            bool: True if improvement detected, False otherwise
        """
        self.trace.append(val_score)
        
        if self.val_comp(val_score):
            # Score improved
            self.best_score = val_score
            self.counter = 0
            return True
        else:
            # Score did not improve
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class SchedulerType(Enum):
    """Types of learning rate schedulers supported."""
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    ONE_CYCLE = "one_cycle"
    

def get_scheduler(scheduler_type: Union[str, SchedulerType], 
                 optimizer: torch.optim.Optimizer, 
                 **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler based on type.
    
    Args:
        scheduler_type: Type of scheduler to create
        optimizer: PyTorch optimizer to schedule
        **kwargs: Additional arguments for specific scheduler
        
    Returns:
        PyTorch learning rate scheduler
    """
    if isinstance(scheduler_type, str):
        try:
            scheduler_type = SchedulerType(scheduler_type)
        except ValueError:
            # Handle string type directly
            if scheduler_type == "step":
                scheduler_type = SchedulerType.STEP
            elif scheduler_type == "exponential":
                scheduler_type = SchedulerType.EXPONENTIAL
            elif scheduler_type == "cosine":
                scheduler_type = SchedulerType.COSINE
            elif scheduler_type in ["reduce_on_plateau", "reduce_lr"]:
                scheduler_type = SchedulerType.REDUCE_ON_PLATEAU
            elif scheduler_type in ["one_cycle", "onecycle"]:
                scheduler_type = SchedulerType.ONE_CYCLE
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    if scheduler_type == SchedulerType.STEP:
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == SchedulerType.EXPONENTIAL:
        gamma = kwargs.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == SchedulerType.COSINE:
        T_max = kwargs.get('T_max', 50)  # Usually num_epochs
        eta_min = kwargs.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        verbose = kwargs.get('verbose', False)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose
        )
    
    elif scheduler_type == SchedulerType.ONE_CYCLE:
        max_lr = kwargs.get('max_lr', 0.01)
        steps_per_epoch = kwargs.get('steps_per_epoch', 100)
        epochs = kwargs.get('epochs', 50)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs
        )
    
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")
        return None


def apply_gradient_clipping(model: nn.Module, max_norm: float = 1.0, adaptive: bool = False, 
                          model_type: Optional[str] = None):
    """
    Apply gradient clipping to model parameters.
    
    Args:
        model: PyTorch model
        max_norm: Maximum norm for gradient clipping
        adaptive: Whether to use adaptive clipping based on model type
        model_type: Type of model for adaptive clipping
    """
    # Apply adaptive clipping if requested
    if adaptive and model_type:
        # Adjust max_norm based on model type
        if model_type == 'siamese':
            # Siamese networks can be sensitive to gradient spikes
            max_norm = 0.5
        elif model_type == 'attention':
            # Attention mechanisms may need more controlled gradients
            max_norm = 0.75
        elif model_type == 'arcface':
            # ArcFace typically needs tighter gradient control
            max_norm = 0.3
    
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def save_checkpoint(model: nn.Module, 
                  optimizer: torch.optim.Optimizer,
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                  epoch: int,
                  validation_metrics: Dict[str, float],
                  checkpoint_dir: Path,
                  filename: str,
                  metadata: Optional[Dict[str, Any]] = None,
                  keep_best_only: bool = False) -> Path:
    """
    Save a comprehensive model checkpoint with all necessary state for resuming.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler (optional)
        epoch: Current epoch number
        validation_metrics: Dict of validation metrics to save
        checkpoint_dir: Directory to save checkpoint
        filename: Name of checkpoint file
        metadata: Additional metadata to save with checkpoint
        keep_best_only: Whether to keep only the best checkpoint
        
    Returns:
        Path to saved checkpoint
    """
    # Ensure directory exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint path
    checkpoint_path = checkpoint_dir / filename
    
    # Create checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": validation_metrics,
        "date_saved": datetime.datetime.now().isoformat(),
    }
    
    # Add scheduler state if available
    if scheduler is not None:
        if hasattr(scheduler, 'state_dict'):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Add metadata if provided
    if metadata:
        checkpoint["metadata"] = metadata
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # If keep_best_only, prune old checkpoints
    if keep_best_only:
        prune_checkpoints(checkpoint_dir, keep=1)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, 
                  model: nn.Module, 
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                  device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model, optimizer, and scheduler state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to restore state into
        optimizer: PyTorch optimizer to restore state into (optional)
        scheduler: PyTorch learning rate scheduler to restore state into (optional)
        device: Device to load model onto
        
    Returns:
        Dict containing checkpoint information
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def prune_checkpoints(checkpoint_dir: Path, keep: int = 5, pattern: str = "checkpoint_*.pth"):
    """
    Prune old checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep: Number of most recent checkpoints to keep
        pattern: Glob pattern to match checkpoint files
    """
    # Get all checkpoint files matching the pattern
    checkpoint_files = glob.glob(str(checkpoint_dir / pattern))
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    # Delete old checkpoints beyond the number to keep
    for filepath in checkpoint_files[keep:]:
        try:
            os.remove(filepath)
            logger.info(f"Pruned old checkpoint: {filepath}")
        except Exception as e:
            logger.error(f"Error removing checkpoint {filepath}: {str(e)}")


class SimpleResultsManager:
    """
    Simple results manager to record training metrics.
    Required by the specialized architecture handlers.
    """
    
    def __init__(self, output_dir: Path):
        """Initialize with output directory."""
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def record_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Record training metrics."""
        file_path = self.metrics_dir / f"training_epoch_{epoch}.json"
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def record_evaluation_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Record validation metrics."""
        file_path = self.metrics_dir / f"validation_epoch_{epoch}.json"
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def record_test_metrics(self, metrics: Dict[str, float]):
        """Record test metrics."""
        file_path = self.metrics_dir / "test_metrics.json"
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def record_learning_curves(self, train_losses: List[float], val_losses: List[float], val_accuracies: List[float]):
        """Record and plot learning curves."""
        # Save the raw data
        curves_data = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
        
        with open(self.metrics_dir / "learning_curves.json", 'w') as f:
            json.dump(curves_data, f, indent=2)
        
        # Plot learning curves using pyplot
        from .training import plot_learning_curves
        plot_learning_curves(
            train_losses, val_losses, val_accuracies,
            str(self.output_dir), "model"
        )
    
    def save_model_checkpoint(self, model, optimizer, epoch, is_best=False, 
                            scheduler=None, metrics=None):
        """Save model checkpoint."""
        # Create checkpoint filename
        accuracy = metrics.get('accuracy', 0) if metrics else 0
        filename = f"checkpoint_epoch_{epoch}_acc{accuracy:.4f}.pth"
        
        # Save using the existing function
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            validation_metrics=metrics or {},
            checkpoint_dir=self.checkpoints_dir,
            filename=filename,
            metadata={"is_best": is_best}
        )
        
        # If this is the best model, also save a best_model.pth
        if is_best:
            best_model_path = self.checkpoints_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
    
    def record_confusion_matrix(self, y_true, y_pred, class_names):
        """Record confusion matrix."""
        from .advanced_metrics import create_enhanced_confusion_matrix
        cm_data = create_enhanced_confusion_matrix(y_true, y_pred, class_names)
        
        with open(self.metrics_dir / "confusion_matrix.json", 'w') as f:
            json.dump(cm_data, f, indent=2)
    
    def record_per_class_metrics(self, y_true, y_pred, y_score, class_names):
        """Record per-class metrics."""
        from .advanced_metrics import calculate_per_class_metrics
        metrics = calculate_per_class_metrics(y_true, y_pred, y_score, class_names)
        
        with open(self.metrics_dir / "per_class_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def record_calibration_metrics(self, y_true, y_pred, y_score):
        """Record calibration metrics."""
        from .advanced_metrics import expected_calibration_error
        metrics = expected_calibration_error(y_true, y_pred, y_score)
        
        with open(self.metrics_dir / "calibration_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)