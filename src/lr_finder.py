#!/usr/bin/env python3

import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import logging

from .base_config import logger

class LearningRateFinder:
    """Implements Leslie Smith's Learning Rate Range Test for finding optimal learning rates."""
    
    def __init__(self, 
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 start_lr: float = 1e-7,
                 end_lr: float = 10.0,
                 num_iterations: int = 100,
                 diverge_threshold: float = 4.0,
                 save_dir: Optional[Path] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iterations = num_iterations
        self.diverge_threshold = diverge_threshold
        self.save_dir = save_dir
        
        # Initialize tracking for multiple parameter groups
        self.num_groups = len(optimizer.param_groups)
        self.group_learning_rates: List[List[float]] = [[] for _ in range(self.num_groups)]
        self.losses: List[float] = []
        self.best_loss = float('inf')
        
        # Save original model state
        self.original_state = copy.deepcopy(model.state_dict())
        
        # Calculate learning rate multiplier
        self.lr_multiplier = (end_lr / start_lr) ** (1 / num_iterations)
    
    def find_lr(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the learning rate finder with support for multiple parameter groups.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing learning rates and losses for each parameter group
        """
        # Reset tracking
        self.group_learning_rates = [[] for _ in range(self.num_groups)]
        self.losses = []
        self.best_loss = float('inf')
        
        # Set initial learning rate for each parameter group
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.start_lr
        
        # Put model in training mode
        self.model.train()
        
        try:
            # Run the learning rate finder
            for iteration, batch in enumerate(itertools.islice(
                    itertools.cycle(train_loader), self.num_iterations)):
                
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        # Standard (inputs, targets) format
                        inputs, targets = batch
                        is_siamese = False
                    elif len(batch) == 3:
                        # Siamese network format (img1, img2, target)
                        img1, img2, targets = batch
                        is_siamese = True
                    else:
                        raise ValueError(f"Unsupported batch format with {len(batch)} items")
                else:
                    raise ValueError(f"Expected batch to be tuple or list, got {type(batch)}")
                
                # Move data to device
                if is_siamese:
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if is_siamese:
                    # Handle Siamese networks
                    outputs = self.model(img1, img2)
                    # Check if model returns a tuple of outputs or a distance
                    if isinstance(outputs, tuple):
                        out1, out2 = outputs
                        loss = self.criterion(out1, out2, targets)
                    else:
                        # For models that might return a distance directly
                        loss = self.criterion(outputs, targets)
                else:
                    # Handle regular networks
                    if hasattr(self.model, 'forward'):
                        # Check if model's forward method requires labels during training
                        import inspect
                        forward_sig = inspect.signature(self.model.forward)
                        if 'labels' in forward_sig.parameters:
                            outputs = self.model(inputs, labels=targets)
                            # If model returns tuple (outputs, loss), use the loss directly
                            if isinstance(outputs, tuple):
                                loss = outputs[1]
                                outputs = outputs[0]
                            else:
                                loss = self.criterion(outputs, targets)
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Record stats for each parameter group
                for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    self.group_learning_rates[group_idx].append(param_group['lr'])
                    # Update learning rate for this group
                    param_group['lr'] *= self.lr_multiplier
                
                # Record overall loss
                self.losses.append(loss.item())
                
                # Check for divergence
                if loss.item() > self.diverge_threshold * self.best_loss or torch.isnan(loss):
                    logger.info(f"Learning rate finder stopped at iteration {iteration} due to divergence")
                    break
                
                if loss.item() < self.best_loss:
                    self.best_loss = loss.item()
                
                # Log progress
                if iteration % 10 == 0:
                    lr_str = ", ".join([f"Group {i}: {lrs[-1]:.2e}" for i, lrs in enumerate(self.group_learning_rates)])
                    logger.info(f"LR Finder: Iteration {iteration}, LRs: [{lr_str}], Loss: {loss.item():.4f}")
        
        finally:
            # Restore original model state
            self.model.load_state_dict(self.original_state)
        
        # Store results with information per group
        results = {
            "losses": self.losses,
            "group_learning_rates": self.group_learning_rates,
            "num_groups": self.num_groups,
            "best_loss": self.best_loss
        }
        
        # For backward compatibility, also set the learning_rates attribute
        self.learning_rates = self.group_learning_rates[0] if self.group_learning_rates else []
        
        return results
    
    def plot_results(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Plot the learning rate finder results for all parameter groups.
        
        Args:
            save_path: Optional path to save the plots
            
        Returns:
            Dictionary containing suggested learning rates and analysis for each group
        """
        if not any(self.group_learning_rates) or not self.losses:
            raise ValueError("No learning rate finder results to plot. Run find_lr first.")
        
        # Create figure with subplots for each parameter group
        n_groups = len(self.group_learning_rates)
        fig, axes = plt.subplots(n_groups + 1, 1, figsize=(12, 6 * (n_groups + 1)))
        
        # Plot loss curve
        axes[0].plot(self.group_learning_rates[0], self.losses, 'b-', label='Loss')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Learning Rate (log scale)')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Overall Loss vs Learning Rate')
        axes[0].grid(True)
        
        # Analyze results for each parameter group
        analysis = self._analyze_results()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        # Mark suggested learning rates and steepest points
        for group_idx, group_analysis in enumerate(analysis["groups"]):
            suggested_lr = group_analysis["suggested_learning_rate"]
            axes[0].axvline(suggested_lr, color=colors[group_idx % len(colors)], linestyle='--',
                            label=f'Group {group_idx} Suggested LR: {suggested_lr:.2e}')
            # Optionally, mark steepest point
            if group_analysis.get("steepest_point_lr"):
                axes[0].axvline(group_analysis["steepest_point_lr"], color=colors[group_idx % len(colors)], linestyle=':',
                                label=f'Group {group_idx} Steepest: {group_analysis["steepest_point_lr"]:.2e}')
        axes[0].legend()
        
        # Plot learning rate curves for each parameter group
        for group_idx, lrs in enumerate(self.group_learning_rates):
            ax = axes[group_idx + 1]
            ax.plot(range(len(lrs)), lrs, color=colors[group_idx % len(colors)], label=f'Group {group_idx} LR')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.set_title(f'Parameter Group {group_idx} Learning Rate Progression')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        
        return analysis
    
    def _analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the learning rate finder results to suggest optimal learning rates for each group.
        
        Returns:
            Dictionary containing suggested learning rates and analysis for each group
        """
        if not any(self.group_learning_rates) or not self.losses:
            raise ValueError("No learning rate finder results to analyze")
        
        analysis = {
            "overall": self._analyze_single_group(self.group_learning_rates[0], self.losses),
            "groups": []  # Initialize as empty list
        }
        
        # Analyze each parameter group
        for group_idx, lrs in enumerate(self.group_learning_rates):
            group_analysis = self._analyze_single_group(lrs, self.losses)
            group_analysis["group_idx"] = group_idx
            analysis["groups"].append(group_analysis)
        
        return analysis
    
    def _analyze_single_group(self, lrs: List[float], losses: List[float]) -> Dict[str, Any]:
        """Helper method to analyze a single parameter group's results."""
        # Convert to numpy arrays for analysis
        lrs_array = np.array(lrs)
        losses_array = np.array(losses)
        
        # Find the point of steepest descent
        window_length = min(21, len(losses_array) - 2)
        if window_length > 2:
            smoothed_losses = savgol_filter(losses_array, window_length=window_length, polyorder=3)
            gradients = np.gradient(smoothed_losses)
            steepest_idx = np.argmin(gradients)
            steepest_lr = lrs_array[steepest_idx]
        else:
            steepest_lr = None
        
        # Find the point where loss starts increasing significantly
        min_loss_idx = np.argmin(losses_array)
        min_loss = losses_array[min_loss_idx]
        increasing_idx = np.where(losses_array > 3 * min_loss)[0]
        if len(increasing_idx) > 0:
            max_lr = lrs_array[increasing_idx[0]]
        else:
            max_lr = lrs_array[-1]
        
        # Suggest learning rates
        suggested_lr = steepest_lr if steepest_lr is not None else max_lr / 10
        min_lr = suggested_lr / 10
        max_lr = suggested_lr * 10
        
        return {
            "suggested_learning_rate": float(suggested_lr),
            "min_learning_rate": float(min_lr),
            "max_learning_rate": float(max_lr),
            "steepest_point_lr": float(steepest_lr) if steepest_lr is not None else None,
            "analysis": {
                "num_iterations": len(lrs),
                "min_loss": float(min_loss),
                "max_loss": float(np.max(losses_array)),
                "final_loss": float(losses_array[-1])
            }
        }
    
    def save_results(self, output_dir: Path) -> Dict[str, Any]:
        """
        Save learning rate finder results and plots.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing the analysis results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the raw data
        data = {
            "group_learning_rates": self.group_learning_rates,
            "losses": self.losses,
            "best_loss": self.best_loss,
            "start_lr": self.start_lr,
            "end_lr": self.end_lr,
            "num_iterations": self.num_iterations,
            "num_groups": self.num_groups
        }
        
        import json
        with open(output_dir / "lr_finder_results.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        # Generate and save plots
        analysis = self.plot_results(output_dir / "lr_finder_plot.png")
        
        # Save analysis
        with open(output_dir / "lr_finder_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis 