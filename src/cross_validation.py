#!/usr/bin/env python3

"""
Simple cross-validation module for face recognition models.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

from .base_config import PROC_DATA_DIR, CHECKPOINTS_DIR, logger, OUT_DIR
from .face_models import get_model, get_criterion
from .training import train_model
from .data_utils import SiameseDataset

def run_cross_validation(model_type: Optional[str] = None, 
                        dataset_path: Optional[Path] = None,
                        n_folds: int = 5,
                        random_seed: int = 42,
                        existing_model: Optional[str] = None):
    """Run k-fold cross-validation for a model.
    
    Args:
        model_type: Type of model to use for cross-validation
        dataset_path: Path to the dataset
        n_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
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
    
    if dataset_path is None:
        # List available processed datasets
        processed_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
        if not processed_dirs:
            print("No processed datasets found. Please process raw data first.")
            return False
        
        print("\nAvailable processed datasets:")
        for i, d in enumerate(processed_dirs, 1):
            print(f"{i}. {d.name}")
        
        while True:
            dataset_choice = input("\nEnter dataset number to use for cross-validation: ")
            try:
                dataset_idx = int(dataset_choice) - 1
                if 0 <= dataset_idx < len(processed_dirs):
                    dataset_path = processed_dirs[dataset_idx]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    print(f"\nRunning {n_folds}-fold cross-validation for {model_type} on {dataset_path.name}")
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
    
    # Load dataset
    if model_type == 'siamese':
        dataset = SiameseDataset(str(dataset_path / "train"), transform=transform)
    else:
        dataset = datasets.ImageFolder(dataset_path / "train", transform=transform)
    
    # Setup k-fold cross validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Setup results tracking
    fold_results = []
    cv_output_dir = CHECKPOINTS_DIR / f"{model_type}_cv_{dataset_path.name}"
    cv_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing model to use as a starting point
    initial_model_state = None
    if existing_model:
        model_checkpoint_dir = CHECKPOINTS_DIR / existing_model
        if model_checkpoint_dir.exists() and (model_checkpoint_dir / 'best_model.pth').exists():
            try:
                # Get the class count from the dataset
                num_classes = len(dataset.classes) if model_type != 'siamese' else 2
                
                # Create a temporary model to load the state
                temp_model = get_model(model_type, num_classes=num_classes)
                temp_model.load_state_dict(torch.load(model_checkpoint_dir / 'best_model.pth', 
                                                    map_location=device))
                initial_model_state = temp_model.state_dict()
                logger.info(f"Successfully loaded model state from {existing_model}")
            except Exception as e:
                logger.error(f"Failed to load model state: {str(e)}")
                initial_model_state = None
    
    # Cross validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\n{'='*80}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'='*80}")
        
        # Create data samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # Create data loaders
        train_loader = DataLoader(
            dataset, batch_size=32, sampler=train_sampler,
            num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            dataset, batch_size=32, sampler=val_sampler,
            num_workers=2, pin_memory=True
        )
        
        # Initialize model
        num_classes = len(dataset.classes) if model_type != 'siamese' else 2
        model = get_model(model_type, num_classes=num_classes)
        
        # Load weights from existing model if available
        if initial_model_state is not None:
            try:
                model.load_state_dict(initial_model_state)
                logger.info(f"Loaded initial weights for fold {fold+1}")
            except Exception as e:
                logger.error(f"Failed to load initial weights for fold {fold+1}: {str(e)}")
        
        model = model.to(device)
        
        # Setup training
        criterion = get_criterion(model_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        print(f"Training model for fold {fold+1}...")
        fold_dir = cv_output_dir / f"fold_{fold+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop (simplified for CV)
        epochs = 15  # Reduced epochs for CV
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
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
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if model_type == 'siamese':
                        img1, img2, target = batch
                        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
                        out1, out2 = model(img1, img2)
                        val_loss += criterion(out1, out2, target).item()
                        # Calculate distances and predict
                        dist = torch.nn.functional.pairwise_distance(out1, out2)
                        pred = (dist < 0.5).float()
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                    else:
                        data, target = batch
                        data, target = data.to(device), target.to(device)
                        
                        # Handle ArcFace differently during validation
                        if model_type == 'arcface':
                            # In validation, we just need the embeddings
                            output = model(data)
                            # For validation purposes, use separate classifier layer
                            val_classifier = torch.nn.Linear(512, num_classes).to(device)
                            logits = val_classifier(output)
                        else:
                            output = model(data)
                            logits = output
                        
                        val_loss += criterion(logits, target).item()
                        _, pred = logits.max(1)
                        correct += pred.eq(target).sum().item()
                        total += target.size(0)
            
            # Calculate epoch metrics
            epoch_loss = train_loss / len(train_loader)
            val_epoch_loss = val_loss / len(val_loader)
            accuracy = correct / total
            
            # Log metrics to console
            logger.info(f'Fold {fold+1}, Epoch {epoch+1}/{epochs}:')
            logger.info(f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
            
            # Save best model for this fold
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                torch.save(model.state_dict(), fold_dir / 'best_model.pth')
        
        # Record fold results
        fold_results.append({
            'fold': fold + 1,
            'best_validation_accuracy': best_val_acc
        })
        
        # Save fold results
        with open(fold_dir / 'results.json', 'w') as f:
            json.dump(fold_results[-1], f, indent=2)
    
    # Calculate cross-validation statistics
    accuracies = [result['best_validation_accuracy'] for result in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # Compile and save final CV results
    cv_final_results = {
        'model_type': model_type,
        'dataset': dataset_path.name,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'existing_model': existing_model  # Add the existing model info to results
    }
    
    with open(cv_output_dir / 'cv_results.json', 'w') as f:
        json.dump(cv_final_results, f, indent=2)
    
    print(f"\nCross-validation complete!")
    print(f"Mean accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Results saved to: {cv_output_dir / 'cv_results.json'}")
    
    return cv_final_results

if __name__ == "__main__":
    run_cross_validation() 