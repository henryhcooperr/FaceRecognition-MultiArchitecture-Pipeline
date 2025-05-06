#!/usr/bin/env python3

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any
import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .base_config import PROC_DATA_DIR, CHECKPOINTS_DIR, logger
from .face_models import get_model, get_criterion

class SiameseDataset(Dataset):
    """Dataset for Siamese network training."""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and their labels
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*.jpg"):
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img1_path = self.images[idx]
        label1 = self.labels[idx]
        
        # Randomly decide if we want a positive or negative pair
        should_get_same_class = random.random() > 0.5
        
        if should_get_same_class:
            # Get another image from the same class
            while True:
                idx2 = random.randrange(len(self.images))
                if self.labels[idx2] == label1 and idx2 != idx:
                    break
        else:
            # Get an image from a different class
            while True:
                idx2 = random.randrange(len(self.images))
                if self.labels[idx2] != label1:
                    break
        
        img2_path = self.images[idx2]
        label2 = self.labels[idx2]
        
        # Load and transform images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Label is 1 for same class, 0 for different classes
        label = 1 if label1 == label2 else 0
        
        return img1, img2, label

def train_model(model_type: str, model_name: Optional[str] = None,
                batch_size: int = 32, epochs: int = 50,
                lr: float = 0.001, weight_decay: float = 1e-4):
    """Train a face recognition model with simplified parameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
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
    
    logger.info(f"Using dataset: {selected_data_dir.name}")
    
    # Generate model name if not provided
    if model_name is None:
        existing_models = list(CHECKPOINTS_DIR.glob(f'best_model_{model_type}_*.pth'))
        version = len(existing_models) + 1
        model_name = f"{model_type}_v{version}"
    else:
        model_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_')).lower()
        model_name = f"{model_type}_{model_name}"
    
    logger.info(f"Training model: {model_name}")
    
    # Create model-specific directories
    model_checkpoint_dir = CHECKPOINTS_DIR / model_name
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    if model_type == 'siamese':
        train_dataset = SiameseDataset(selected_data_dir / "train", transform=transform)
        val_dataset = SiameseDataset(selected_data_dir / "val", transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        train_dataset = datasets.ImageFolder(selected_data_dir / "train", transform=transform)
        val_dataset = datasets.ImageFolder(selected_data_dir / "val", transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    num_classes = len(train_dataset.classes) if model_type != 'siamese' else 2
    model = get_model(model_type, num_classes=num_classes)
    model = model.to(device)
    
    # Setup training
    criterion = get_criterion(model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
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
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
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
                    correct += pred.eq(target.view_as(pred)).sum().item()
                else:
                    data, target = batch
                    data, target = data.to(device), target.to(device)
                    
                    # Handle ArcFace differently during validation
                    if model_type == 'arcface':
                        # In validation, we just need the embeddings
                        output = model(data)
                        # For validation purposes, use separate classifier layer
                        val_classifier = nn.Linear(512, num_classes).to(device)
                        output = val_classifier(output)
                    else:
                        output = model(data)
                    
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_dataset)
        
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_checkpoint_dir / 'best_model.pth')
        
        scheduler.step(val_loss)
    
    return model_name

def tune_hyperparameters(model_type: str, dataset_path: Path, n_trials: int = 50) -> Dict[str, Any]:
    """Tune model hyperparameters using Optuna."""
    logger.info(f"Tuning hyperparameters for model type: {model_type}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    if model_type == 'siamese':
        train_dataset = SiameseDataset(dataset_path / "train", transform=transform)
        val_dataset = SiameseDataset(dataset_path / "val", transform=transform)
    else:
        train_dataset = datasets.ImageFolder(dataset_path / "train", transform=transform)
        val_dataset = datasets.ImageFolder(dataset_path / "val", transform=transform)
    
    def objective(trial):
        # Define hyperparameter search space
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Additional model-specific hyperparameters
        model_params = {}
        if model_type == 'siamese':
            num_classes = 2  # Binary classification for pairs
        else:
            num_classes = len(train_dataset.classes)
        
        # Initialize model
        model = get_model(model_type, num_classes=num_classes)
        model = model.to(device)
        
        # Setup training
        criterion = get_criterion(model_type)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Quick training for hyperparameter search (small number of epochs)
        model.train()
        for epoch in range(5):  # Small number of epochs for quick evaluation
            # Training phase
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
                optimizer.step()
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
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
                        correct += pred.eq(target.view_as(pred)).sum().item()
                    else:
                        data, target = batch
                        data, target = data.to(device), target.to(device)
                        
                        # Handle ArcFace differently during validation
                        if model_type == 'arcface':
                            # In validation, we just need the embeddings
                            output = model(data)
                            # For validation purposes, use separate classifier layer
                            val_classifier = nn.Linear(512, num_classes).to(device)
                            output = val_classifier(output)
                        else:
                            output = model(data)
                        
                        val_loss += criterion(output, target).item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
            
            val_loss /= len(val_loader)
            accuracy = correct / len(val_dataset)
            
            # Report metrics to Optuna
            trial.report(val_loss, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return val_loss
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Get best hyperparameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Add fixed parameters
    best_params["model_type"] = model_type
    best_params["epochs"] = 50  # Default full training epochs
    
    return best_params 