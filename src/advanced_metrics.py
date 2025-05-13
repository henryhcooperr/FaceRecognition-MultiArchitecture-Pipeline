#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .base_config import logger

def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               y_score: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Gets detailed metrics per class so we can see which classes are hard to classify
    
    y_true: ground truth labels  
    y_pred: what the model predicted
    y_score: confidence scores
    class_names: list of class names
    
    Returns a dict with metrics for each class
    """
    # Get precision, recall, etc. for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # results dict - will store everything here
    results = {}
    
    for i, cls in enumerate(class_names):
        # Skip non-existent classes
        if i >= len(precision):
            continue
            
        # Get ROC AUC - this took me forever to get right ugh
        # Create binary labels for this class
        true_bin = (np.array(y_true) == i).astype(int)
        
        try:
            # Handle different score formats
            if y_score.ndim > 1 and y_score.shape[1] > 1:
                scores_i = y_score[:, i]
            else:
                # For binary cases
                scores_i = y_score if i == 1 else 1 - y_score
            
            # Could do this in one line but breaking it up for readability
            fpr, tpr, _ = roc_curve(true_bin, scores_i)
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            logger.warning(f"ROC AUC failed for {cls}: {str(e)}")
            roc_auc = float('nan')
        
        # Calculate accuracy for this class
        n_right = sum((y_true == i) & (y_pred == i))
        n_total = sum(y_true == i)
        accuracy = n_right / n_total if n_total > 0 else 0
        
        # Store all metrics for this class
        results[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc)
        }
    
    return results


def create_enhanced_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                                    class_names: List[str]) -> Dict[str, Any]:
    """
    Makes a beefed-up confusion matrix with extra stats
    """
    # Regular confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate the totals
    rows_total = cm.sum(axis=1)
    cols_total = cm.sum(axis=0)
    
    # Per-class stats
    class_info = {}
    
    for i, cls in enumerate(class_names):
        if i < len(rows_total):
            # TP, FP, FN
            true_positives = cm[i, i]
            false_positives = cols_total[i] - true_positives
            false_negatives = rows_total[i] - true_positives
            
            # Precision, recall
            prec = true_positives / cols_total[i] if cols_total[i] > 0 else 0
            rec = true_positives / rows_total[i] if rows_total[i] > 0 else 0
            
            # Find where this class gets misclassified to
            if rows_total[i] > 0:
                # This is a neat trick to find where misclassifications are going
                misclass = []
                for j in range(len(class_names)):
                    if j != i and cm[i, j] > 0:
                        # Calculate % misclassified
                        misclass.append((class_names[j], float(cm[i, j] / rows_total[i])))
                
                # Sort by most common misclassification
                misclass.sort(key=lambda x: x[1], reverse=True)
            else:
                misclass = []
            
            # Store everything
            class_info[cls] = {
                "true_positives": int(true_positives),
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
                "precision": float(prec),
                "recall": float(rec),
                "support": int(rows_total[i]),
                "misclassified_to": misclass[:3]  # Top 3 misclassifications
            }
    
    return {
        "matrix": cm.tolist(),
        "class_names": class_names,
        "class_statistics": class_info
    }


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_score: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Calculates the Expected Calibration Error (ECE)
    
    ECE measures how well model confidence matches actual accuracy
    For face recognition, we need well-calibrated models!
    
    Returns dict with calibration metrics
    """
    # Get confidence scores in right format
    if y_score.ndim > 1:
        # Multi-class - get confidence for predicted class
        confs = np.array([y_score[i, pred] for i, pred in enumerate(y_pred)])
    else:
        # Binary case
        confs = y_score
    
    # Split into bins
    bin_indices = np.digitize(confs, np.linspace(0, 1, n_bins))
    
    # Initialize arrays
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Calculate accuracy and confidence per bin
    for i in range(n_bins):
        bin_idx = i + 1  # np.digitize starts at 1
        mask = bin_indices == bin_idx
        
        if np.sum(mask) > 0:
            bin_accs[i] = np.mean(y_true[mask] == y_pred[mask])
            bin_confs[i] = np.mean(confs[mask])
            bin_counts[i] = np.sum(mask)
    
    # Calculate ECE
    ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_counts / len(y_true)))
    
    # Maximum Calibration Error (MCE) - worst bin
    mce = np.max(np.abs(bin_accs - bin_confs))
    
    # Return everything 
    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "bin_accuracies": bin_accs.tolist(),
        "bin_confidences": bin_confs.tolist(),
        "bin_counts": bin_counts.tolist(),
        "n_bins": n_bins
    }


class TimerContext:
    """
    Simple timer you can use with 'with' statements
    
    Example:
        with TimerContext("Training") as timer:
            # training code here
        print(f"Training took {timer.elapsed_time:.2f}s")
    """
    
    def __init__(self, name: str):
        """Initialize with a name for the timer."""
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {self.elapsed_time:.2f}s")


def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the parameters in a model
    """
    # Total params (trainable + frozen)
    total = sum(p.numel() for p in model.parameters())
    
    # Just trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable
    }