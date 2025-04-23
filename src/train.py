#!/usr/bin/env python3
"""
Training script for face recognition models.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm

from models import BaselineNet, ResNetTransfer, SiameseNet, ContrastiveLoss
from preprocessing import AugmentationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train face recognition models')
    parser.add_argument('--model', type=str, required=True,
                      choices=['baseline', 'cnn', 'siamese'],
                      help='Model architecture to train')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                      help='Directory containing processed dataset')
    parser.add_argument('--output-dir', type=str, default='models/checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay for optimizer')
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable Weights & Biases logging')
    return parser.parse_args()

def get_model(model_type: str, num_classes: int = 18) -> nn.Module:
    """
    Get model instance based on type.
    
    Args:
        model_type: One of 'baseline', 'cnn', or 'siamese'
        num_classes: Number of celebrity classes
        
    Returns:
        Model instance
    """
    if model_type == 'baseline':
        return BaselineNet(num_classes=num_classes)
    elif model_type == 'cnn':
        return ResNetTransfer(num_classes=num_classes)
    elif model_type == 'siamese':
        return SiameseNet()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def get_criterion(model_type: str) -> nn.Module:
    """
    Get loss function based on model type.
    
    Args:
        model_type: One of 'baseline', 'cnn', or 'siamese'
        
    Returns:
        Loss function
    """
    if model_type in ['baseline', 'cnn']:
        return nn.CrossEntropyLoss()
    elif model_type == 'siamese':
        return ContrastiveLoss()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
               optimizer: optim.Optimizer, device: torch.device, model_type: str) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        model_type: Type of model being trained
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        if model_type == 'siamese':
            # Siamese network expects pairs of images
            img1, img2, label = batch
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Forward pass
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
        else:
            # Standard classification
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
            device: torch.device, model_type: str) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        model_type: Type of model being validated
        
    Returns:
        Tuple of (average validation loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            if model_type == 'siamese':
                # Siamese network validation
                img1, img2, label = batch
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
                
                # Compute accuracy based on distance threshold
                dist = torch.pairwise_distance(out1, out2)
                pred = (dist < 0.5).float()
                correct += (pred == label).sum().item()
                total += label.size(0)
            else:
                # Standard classification validation
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            total_loss += loss.item()
            
    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(project="celebrity-face-recognition",
                  config=vars(args))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model and criterion
    model = get_model(args.model).to(device)
    criterion = get_criterion(args.model)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Setup data loaders
    train_transform = AugmentationPipeline(phase='train')
    val_transform = AugmentationPipeline(phase='val')
    
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.model)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.model)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if not args.no_wandb:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / f"{args.model}_best.pth")
            
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, output_dir / f"{args.model}_latest.pth")
        
    logger.info("Training completed!")
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 