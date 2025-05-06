#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import cv2
import time
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from tqdm import tqdm

from .base_config import PROC_DATA_DIR, CHECKPOINTS_DIR, VIZ_DIR, logger
from .face_models import get_model
from .training import SiameseDataset

def evaluate_model(model_type: str, model_name: Optional[str] = None):
    """Evaluate a trained model with comprehensive metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If no specific model name provided, use the latest version
    if model_name is None:
        model_dirs = list(CHECKPOINTS_DIR.glob(f'{model_type}_*'))
        if not model_dirs:
            raise ValueError(f"No trained models found for type: {model_type}")
        model_name = sorted(model_dirs)[-1].name
    
    model_checkpoint_dir = CHECKPOINTS_DIR / model_name
    if not model_checkpoint_dir.exists():
        raise ValueError(f"Model not found: {model_name}")
    
    # List available processed datasets
    processed_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and (d / "test").exists()]
    if not processed_dirs:
        raise ValueError("No processed datasets found with test data.")
    
    print("\nAvailable processed datasets:")
    for i, d in enumerate(processed_dirs, 1):
        print(f"{i}. {d.name}")
    
    while True:
        dataset_choice = input("\nEnter dataset number to use for evaluation: ")
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
    
    # Create model-specific visualization directory
    model_viz_dir = VIZ_DIR / model_name
    model_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    if model_type == 'siamese':
        test_dataset = SiameseDataset(selected_data_dir / "test", transform=transform)
    else:
        test_dataset = datasets.ImageFolder(selected_data_dir / "test", transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, pin_memory=True)
    
    # Load model
    num_classes = len(test_dataset.classes) if model_type != 'siamese' else 2
    model = get_model(model_type, num_classes).to(device)
    model.load_state_dict(torch.load(model_checkpoint_dir / 'best_model.pth', map_location=device))
    model.eval()
    
    # For ArcFace, we need a classifier for evaluation
    arcface_classifier = None
    if model_type == 'arcface':
        arcface_classifier = nn.Linear(512, num_classes).to(device)
    
    # Initialize metrics
    all_predictions = []
    all_targets = []
    all_probs = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss() if model_type != 'siamese' else nn.BCEWithLogitsLoss()
    
    # Measure inference time
    inference_times = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            if model_type == 'siamese':
                img1, img2, labels = batch
                img1, img2 = img1.to(device), img2.to(device)
                
                # Measure inference time
                start_time = time.time()
                out1, out2 = model(img1, img2)
                dist = F.pairwise_distance(out1, out2)
                pred = (dist < 0.5).float()
                inference_times.append(time.time() - start_time)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(labels.numpy())
                all_probs.extend(dist.cpu().numpy()[:, None])
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                
                # Measure inference time
                start_time = time.time()
                
                # Handle different model architectures
                if model_type == 'arcface':
                    # Get embeddings
                    embeddings = model(images)
                    # Use our evaluation classifier or cosine similarity
                    if arcface_classifier is not None:
                        outputs = arcface_classifier(embeddings)
                    else:
                        # Use cosine similarity as a proxy for classification
                        outputs = F.linear(
                            F.normalize(embeddings), 
                            F.normalize(model.arcface.weight)
                        )
                else:
                    outputs = model(images)
                
                inference_times.append(time.time() - start_time)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    # Calculate ROC AUC
    if model_type == 'siamese':
        fpr, tpr, _ = roc_curve(all_targets, -all_probs.ravel())
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    
    # Calculate PR AUC
    if model_type == 'siamese':
        precision_curve, recall_curve, _ = precision_recall_curve(all_targets, -all_probs.ravel())
        pr_auc = auc(recall_curve, precision_curve)
    else:
        pr_auc = average_precision_score(all_targets, all_probs)
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_times)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms")
    if model_type != 'siamese':
        print(f"Test Loss: {total_loss/len(test_loader):.4f}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes if model_type != 'siamese' else ['Same', 'Different'],
                yticklabels=test_dataset.classes if model_type != 'siamese' else ['Same', 'Different'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(model_viz_dir / 'confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    if model_type == 'siamese':
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        for i in range(len(test_dataset.classes)):
            fpr, tpr, _ = roc_curve(all_targets == i, all_probs[:, i])
            plt.plot(fpr, tpr, label=f'{test_dataset.classes[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(model_viz_dir / 'roc_curves.png')
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    if model_type == 'siamese':
        plt.plot(recall_curve, precision_curve, label=f'PR curve (AUC = {pr_auc:.2f})')
    else:
        for i in range(len(test_dataset.classes)):
            precision_i, recall_i, _ = precision_recall_curve(all_targets == i, all_probs[:, i])
            plt.plot(recall_i, precision_i, label=f'{test_dataset.classes[i]} (AUC = {average_precision_score(all_targets == i, all_probs[:, i]):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(model_viz_dir / 'pr_curves.png')
    plt.close()
    
    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'inference_time': avg_inference_time
    }

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         classes: List[str], output_dir: str, model_name: str):
    """Plot detailed confusion matrix with additional metrics."""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute metrics per class
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / model_name / 'confusion_matrix_detailed.png')
    plt.close()
    
    # Plot per-class metrics
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }, index=classes)
    
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar')
    plt.title('Per-Class Performance Metrics')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / model_name / 'per_class_metrics.png')
    plt.close()

def plot_roc_curves(y_true: np.ndarray, y_score: np.ndarray, 
                   classes: List[str], output_dir: str, model_name: str):
    """Plot ROC curves for each class."""
    plt.figure(figsize=(12, 8))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / model_name / 'roc_curves.png')
    plt.close()

def generate_gradcam(model: nn.Module, image_tensor: torch.Tensor, 
                   target_layer: nn.Module, model_type: str = None) -> np.ndarray:
    """Generate Grad-CAM visualization for a given image and model."""
    # Set model to eval mode and register hooks
    model.eval()
    
    # Storage for activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks using the full backward hook
    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        if model_type == 'siamese':
            # For Siamese network, we use the same image as both inputs
            output1, output2 = model(image_tensor, image_tensor)
            output = torch.pairwise_distance(output1, output2)  # Calculate distance
        else:
            output = model(image_tensor)
        
        if isinstance(output, tuple):
            output = output[0]
        
        # Get the score
        score = torch.max(output)
        
        # Backward pass
        model.zero_grad()
        score.backward()
        
        # Get activations and gradients
        activation = activations[0].detach()
        gradient = gradients[0].detach()
        
        # Global average pooling of gradients
        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
        
        # Weight the activations
        cam = torch.sum(weights * activation, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Convert to numpy and return
        return cam.squeeze().cpu().numpy()
    
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        return None
    
    finally:
        # Clean up hooks
        handle1.remove()
        handle2.remove()

def plot_gradcam_visualization(model: nn.Module, dataset, 
                             num_samples: int, output_dir: str, model_name: str):
    """Plot Grad-CAM visualizations for sample images."""
    device = next(model.parameters()).device
    
    # Get target layer based on model type
    if hasattr(model, 'resnet'):
        target_layer = model.resnet.layer4[-1]
        model_type = 'cnn'
    elif hasattr(model, 'conv3'):
        target_layer = model.conv3
        model_type = 'baseline'
    elif hasattr(model, 'conv'):
        target_layer = model.conv[-3]  # Last conv layer
        model_type = 'siamese'
    else:
        logger.warning(f"Could not determine target layer for Grad-CAM")
        return
    
    # Create figure for visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    successful_samples = 0
    max_attempts = num_samples * 3  # Try up to 3 times the number of samples
    attempts = 0
    
    while successful_samples < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            # Get random sample
            idx = random.randint(0, len(dataset)-1)
            if hasattr(dataset, 'imgs'):
                # Standard ImageFolder dataset
                image_path, label = dataset.imgs[idx]
                image = dataset[idx][0]
            elif hasattr(dataset, 'images'):
                # Siamese dataset
                img1, _, _ = dataset[idx]
                image = img1
            else:
                logger.warning("Unknown dataset format")
                continue
            
            # Convert to tensor and add batch dimension
            img_tensor = image.unsqueeze(0).to(device)
            
            # Generate Grad-CAM
            cam = generate_gradcam(model, img_tensor, target_layer, model_type)
            if cam is None:
                continue
            
            # Convert tensor to numpy for plotting
            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            
            # Plot original image
            axes[successful_samples, 0].imshow(img_np)
            axes[successful_samples, 0].set_title('Original Image')
            axes[successful_samples, 0].axis('off')
            
            # Plot heatmap
            axes[successful_samples, 1].imshow(cam, cmap='jet')
            axes[successful_samples, 1].set_title('Grad-CAM Heatmap')
            axes[successful_samples, 1].axis('off')
            
            # Plot overlay
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = (0.7 * img_np + 0.3 * heatmap/255).clip(0, 1)
            
            axes[successful_samples, 2].imshow(overlay)
            axes[successful_samples, 2].set_title('Overlay')
            axes[successful_samples, 2].axis('off')
            
            successful_samples += 1
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            continue
    
    if successful_samples == 0:
        logger.error("Failed to generate any Grad-CAM visualizations")
        return
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / model_name / 'gradcam_visualization.png')
    plt.close()

def predict_image(model_type: str, image_path: str, model_name: Optional[str] = None) -> Tuple[str, float]:
    """Make a prediction for a single image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If no specific model name provided, use the latest version
    if model_name is None:
        model_dirs = list(CHECKPOINTS_DIR.glob(f'{model_type}_*'))
        if not model_dirs:
            raise ValueError(f"No trained models found for type: {model_type}")
        model_name = sorted(model_dirs)[-1].name
    
    model_checkpoint_dir = CHECKPOINTS_DIR / model_name
    if not model_checkpoint_dir.exists():
        raise ValueError(f"Model not found: {model_name}")
    
    # Find a processed dataset to get class names
    processed_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
    if not processed_dirs:
        raise ValueError("No processed datasets found.")
    
    # Load class names from the first dataset
    dataset_path = processed_dirs[0]
    if model_type == 'siamese':
        raise ValueError("Siamese model can't be used for direct prediction. Use it for verification.")
    else:
        dataset = datasets.ImageFolder(dataset_path / "train")
        classes = dataset.classes
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Load model
    model = get_model(model_type, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_checkpoint_dir / 'best_model.pth', map_location=device))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        prob, pred_idx = torch.max(probs, 1)
        
    return classes[pred_idx.item()], prob.item() 