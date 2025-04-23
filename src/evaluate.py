#!/usr/bin/env python3
"""
Evaluation script for face recognition models.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm

from models import BaselineNet, ResNetTransfer, SiameseNet
from preprocessing import AugmentationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate face recognition models')
    parser.add_argument('--model', type=str, required=True,
                      choices=['baseline', 'cnn', 'siamese'],
                      help='Model architecture to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                      help='Directory containing processed dataset')
    parser.add_argument('--output-dir', type=str, default='reports',
                      help='Directory to save evaluation results')
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

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: list, output_path: str) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                  output_path: str) -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        output_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray,
                 output_path: str) -> None:
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        output_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_tsne_embeddings(model: nn.Module, data_loader: DataLoader,
                        device: torch.device, output_path: str) -> None:
    """
    Plot and save t-SNE visualization of embeddings.
    
    Args:
        model: Model to get embeddings from
        data_loader: Data loader for test set
        device: Device to run inference on
        output_path: Path to save the plot
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Computing embeddings'):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                # Siamese network
                img1, _, label = batch
                img1 = img1.to(device)
                emb = model.get_embedding(img1)
            else:
                # Standard classification
                images, label = batch
                images = images.to(device)
                emb = model.get_embedding(images)
                
            embeddings.append(emb.cpu().numpy())
            labels.extend(label.numpy())
            
    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Face Embeddings')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compute_fairness_metrics(model: nn.Module, data_loader: DataLoader,
                           device: torch.device) -> pd.DataFrame:
    """
    Compute fairness metrics across different demographic groups.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for test set
        device: Device to run inference on
        
    Returns:
        DataFrame containing fairness metrics
    """
    model.eval()
    results = []
    
    # Define demographic groups based on celebrity names
    # This is a simplified approach - in a real application, you would have
    # proper demographic annotations
    demographic_groups = {
        'gender': {
            'male': ['tom_cruise', 'leonardo_dicaprio', 'brad_pitt', 'will_smith'],
            'female': ['angelina_jolie', 'jennifer_lawrence', 'emma_watson', 'scarlett_johansson']
        },
        'age_group': {
            'young': ['emma_watson', 'jennifer_lawrence'],
            'middle': ['leonardo_dicaprio', 'brad_pitt', 'will_smith'],
            'senior': ['tom_cruise', 'angelina_jolie', 'scarlett_johansson']
        }
    }
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Computing fairness metrics'):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                # Siamese network
                img1, img2, label = batch
                img1, img2 = img1.to(device), img2.to(device)
                out1, out2 = model(img1, img2)
                dist = torch.pairwise_distance(out1, out2)
                pred = (dist < 0.5).float()
            else:
                # Standard classification
                images, labels = batch
                images = images.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs.data, 1)
                
            # Get celebrity name from dataset
            celeb_name = data_loader.dataset.classes[labels[0].item()]
            
            # Determine demographic groups
            gender = next((g for g, celebs in demographic_groups['gender'].items() 
                          if celeb_name in celebs), 'unknown')
            age_group = next((g for g, celebs in demographic_groups['age_group'].items() 
                            if celeb_name in celebs), 'unknown')
            
            # Compute metrics
            accuracy = (pred == labels).float().mean().item()
            
            results.append({
                'celebrity': celeb_name,
                'gender': gender,
                'age_group': age_group,
                'accuracy': accuracy
            })
            
    # Convert to DataFrame and compute group-wise metrics
    df = pd.DataFrame(results)
    group_metrics = df.groupby(['gender', 'age_group'])['accuracy'].agg(['mean', 'std', 'count'])
    
    # Add overall metrics
    overall_metrics = pd.DataFrame({
        'mean': [df['accuracy'].mean()],
        'std': [df['accuracy'].std()],
        'count': [len(df)]
    }, index=['overall'])
    
    return pd.concat([group_metrics, overall_metrics])

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = get_model(args.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup data loader
    test_transform = AugmentationPipeline(phase='test')
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'test'),
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Get class names
    class_names = test_dataset.classes
    
    # Collect predictions and true labels
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            if args.model == 'siamese':
                img1, img2, label = batch
                img1, img2 = img1.to(device), img2.to(device)
                out1, out2 = model(img1, img2)
                dist = torch.pairwise_distance(out1, out2)
                pred = (dist < 0.5).float()
                score = 1 - dist  # Convert distance to similarity score
            else:
                images, labels = batch
                images = images.to(device)
                outputs = model(images)
                scores = torch.softmax(outputs, dim=1)
                _, pred = torch.max(outputs.data, 1)
                score = scores[:, 1]  # Score for positive class
                
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_scores.extend(score.cpu().numpy())
            
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Generate plots
    plot_confusion_matrix(all_labels, all_preds, class_names,
                         output_dir / 'confusion_matrix.png')
    plot_roc_curve(all_labels, all_scores,
                  output_dir / 'roc_curve.png')
    plot_pr_curve(all_labels, all_scores,
                 output_dir / 'pr_curve.png')
    plot_tsne_embeddings(model, test_loader, device,
                        output_dir / 'tsne_embeddings.png')
    
    # Compute fairness metrics
    fairness_df = compute_fairness_metrics(model, test_loader, device)
    fairness_df.to_csv(output_dir / 'fairness_metrics.csv', index=False)
    
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 