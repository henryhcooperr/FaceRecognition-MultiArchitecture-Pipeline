#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any
from sklearn.manifold import TSNE

from .base_config import OUT_DIR, VIZ_DIR, logger

def plot_tsne_embeddings(model: torch.nn.Module, dataset, 
                        output_dir: str, model_name: str):
    """Generate t-SNE visualization of the embeddings."""
    device = next(model.parameters()).device
    model.eval()
    
    embeddings = []
    labels = []
    classes = []
    
    # Get embeddings for all images
    with torch.no_grad():
        for i in range(len(dataset)):
            if hasattr(dataset, 'imgs'):
                # Standard ImageFolder dataset
                img, label = dataset[i]
                class_name = dataset.classes[label]
            elif hasattr(dataset, 'images'):
                # Siamese dataset
                img1, _, _ = dataset[i]
                img = img1
                label = dataset.labels[i]
                class_name = dataset.classes[label]
            else:
                logger.warning("Unknown dataset format")
                continue
            
            img_tensor = img.unsqueeze(0).to(device)
            embedding = model.get_embedding(img_tensor)
            embeddings.append(embedding.cpu().numpy())
            labels.append(label)
            classes.append(class_name)
    
    # Convert to numpy arrays
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab20')
    plt.colorbar(scatter)
    
    # Add legend
    unique_labels = np.unique(labels)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab20(i/len(unique_labels)), 
                                 label=classes[int(i)], markersize=10)
                      for i in unique_labels]
    plt.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5))
    
    plt.title('t-SNE visualization of face embeddings')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / model_name / 'tsne_embeddings.png', 
                bbox_inches='tight')
    plt.close()

def plot_attention_maps(model: torch.nn.Module, dataset, 
                      num_samples: int, output_dir: str, model_name: str):
    """Visualize attention maps for attention-based models."""
    device = next(model.parameters()).device
    
    # Check if model has attention module
    if not hasattr(model, 'attention'):
        logger.warning("Model does not have attention module, skipping attention visualization")
        return
    
    # Create figure for visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Storage for attention maps
    attention_maps = []
    
    def attention_hook(module, input, output):
        # For AttentionModule, we want to visualize the attention map
        # The map is calculated as batch_matrix_product(proj_query, proj_key)
        batch_size, C, H, W = input[0].size()
        
        # Get query and key projections
        proj_query = module.query(input[0]).view(batch_size, -1, H*W).permute(0, 2, 1)
        proj_key = module.key(input[0]).view(batch_size, -1, H*W)
        
        # Calculate attention map
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        # Reshape attention map to spatial dimensions
        attention_map = attention[0].reshape(H, W, H*W)
        
        # Average attention maps for all queries
        attention_map = attention_map.mean(dim=2).cpu().detach().numpy()
        
        # Save the attention map
        attention_maps.append(attention_map)
    
    # Register hook to attention module
    handle = model.attention.register_forward_hook(attention_hook)
    
    # Process random samples
    model.eval()
    successful_samples = 0
    
    while successful_samples < num_samples and successful_samples < len(dataset):
        try:
            # Get random sample
            idx = np.random.randint(len(dataset))
            if hasattr(dataset, 'imgs'):
                # Standard ImageFolder dataset
                img, label = dataset[idx]
                class_name = dataset.classes[label]
            elif hasattr(dataset, 'images'):
                # Siamese dataset
                img1, _, _ = dataset[idx]
                img = img1
                label = dataset.labels[idx]
                class_name = dataset.classes[label]
            else:
                logger.warning("Unknown dataset format")
                continue
            
            # Process image
            img_tensor = img.unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(img_tensor)
            
            # Convert tensor to numpy for plotting
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            
            # Plot original image
            axes[successful_samples, 0].imshow(img_np)
            axes[successful_samples, 0].set_title(f'Original: {class_name}')
            axes[successful_samples, 0].axis('off')
            
            # Plot attention map
            attention_map = attention_maps[-1]
            
            # Resize attention map to match image size
            attention_resized = cv2.resize(attention_map, (img_np.shape[1], img_np.shape[0]))
            
            axes[successful_samples, 1].imshow(attention_resized, cmap='hot')
            axes[successful_samples, 1].set_title('Attention Map')
            axes[successful_samples, 1].axis('off')
            
            # Plot overlay
            attention_heatmap = cv2.applyColorMap(np.uint8(255 * attention_resized), cv2.COLORMAP_JET)
            attention_heatmap = cv2.cvtColor(attention_heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = (0.7 * img_np + 0.3 * attention_heatmap).clip(0, 1)
            
            axes[successful_samples, 2].imshow(overlay)
            axes[successful_samples, 2].set_title('Overlay')
            axes[successful_samples, 2].axis('off')
            
            successful_samples += 1
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Remove the hook
    handle.remove()
    
    if successful_samples == 0:
        logger.error("Failed to generate any attention visualizations")
        return
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / model_name / 'attention_maps.png')
    plt.close()

def plot_embedding_similarity(model: torch.nn.Module, dataset, output_dir: str, model_name: str):
    """Visualize embedding similarity matrix between classes."""
    device = next(model.parameters()).device
    model.eval()
    
    # Only process for regular datasets with class structure
    if not hasattr(dataset, 'classes'):
        logger.warning("Dataset doesn't have classes attribute")
        return
    
    # Get class names
    class_names = dataset.classes
    n_classes = len(class_names)
    
    # Collect embeddings per class
    class_embeddings = {i: [] for i in range(n_classes)}
    
    with torch.no_grad():
        for i in range(len(dataset)):
            if hasattr(dataset, 'imgs'):
                # Standard ImageFolder dataset
                img, label = dataset[i]
            else:
                continue
            
            img_tensor = img.unsqueeze(0).to(device)
            embedding = model.get_embedding(img_tensor)
            class_embeddings[label].append(embedding.cpu())
    
    # Calculate average embedding per class
    avg_embeddings = []
    for i in range(n_classes):
        if class_embeddings[i]:
            avg_emb = torch.mean(torch.stack(class_embeddings[i]), dim=0)
            avg_embeddings.append(avg_emb)
        else:
            # If no embeddings for a class, add zeros as placeholder
            avg_embeddings.append(torch.zeros(avg_embeddings[0].shape if avg_embeddings else 512))
    
    # Stack into a tensor
    embeddings_tensor = torch.stack(avg_embeddings)
    
    # Normalize embeddings
    embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
    
    # Calculate cosine similarity matrix
    similarity_matrix = torch.mm(embeddings_tensor, embeddings_tensor.t()).numpy()
    
    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Embedding Similarity Between Classes')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / model_name / 'embedding_similarity.png')
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
    plt.savefig(Path(output_dir) / model_name / 'learning_curves.png')
    plt.close()

def visualize_batch_augmentations(dataset, num_samples: int = 5, output_dir: str = None):
    """Visualize augmentations for a batch of samples."""
    if output_dir is None:
        output_dir = VIZ_DIR / 'augmentations'
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure for visualization
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))
    
    # Process random samples
    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(len(dataset))
        orig_img, label = dataset[idx]
        
        # Convert to numpy for visualization
        orig_img_np = orig_img.permute(1, 2, 0).cpu().numpy()
        orig_img_np = (orig_img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        
        # Plot original image
        axes[i, 0].imshow(orig_img_np)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Generate augmented versions
        for j in range(1, 5):
            # Get the same sample again to see different augmentations
            aug_img, _ = dataset[idx]
            
            # Convert to numpy for visualization
            aug_img_np = aug_img.permute(1, 2, 0).cpu().numpy()
            aug_img_np = (aug_img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            
            # Plot augmented image
            axes[i, j].imshow(aug_img_np)
            axes[i, j].set_title(f'Augmentation {j}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'batch_augmentations.png')
    plt.close()
    
    return output_dir / 'batch_augmentations.png' 