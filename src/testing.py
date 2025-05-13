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
import json
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import random
import os

from .base_config import PROC_DATA_DIR, CHECKPOINTS_DIR, OUT_DIR, logger, check_gpu
from .face_models import get_model, MODEL_TYPES
from .data_utils import SiameseDataset  # Updated import to use data_utils
from .advanced_metrics import plot_confusion_matrix, create_enhanced_confusion_matrix, calculate_per_class_metrics

def evaluate_model(model_type: str, model_name: Optional[str] = None, auto_dataset: bool = False):
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
    
    processed_dirs = []
    dataset_names = set()  # For deduplication
    
    # Check for the standard organization: config_name/dataset_name/test
    config_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and not d.name in ["train", "val", "test"]]
    for config_dir in config_dirs:
        if (config_dir / "test").exists():
            # Use this directory with its name
            if config_dir.name not in dataset_names:
                processed_dirs.append((config_dir, config_dir.name))
                dataset_names.add(config_dir.name)
        else:
            for dataset_dir in config_dir.iterdir():
                if dataset_dir.is_dir() and (dataset_dir / "test").exists():
                    # Use config/dataset naming
                    dataset_path = f"{config_dir.name}/{dataset_dir.name}"
                    if dataset_path not in dataset_names:
                        processed_dirs.append((dataset_dir, dataset_path))
                        dataset_names.add(dataset_path)
    
    if (PROC_DATA_DIR / "test").exists():
        if "root" not in dataset_names:
            processed_dirs.append((PROC_DATA_DIR, "processed (root)"))
            dataset_names.add("root")
    
    if not processed_dirs:
        raise ValueError("No processed datasets found with test data.")
    
    # Get dataset selection
    if auto_dataset:
        # Automatically select the first dataset
        selected_data_dir = processed_dirs[0][0]
        logger.info(f"Auto-selecting dataset: {selected_data_dir.name}")
    else:
        print("\nAvailable processed datasets:")
        for i, (dir_path, display_name) in enumerate(processed_dirs, 1):
            print(f"{i}. {display_name}")
        
        selected_data_dir = None
        while True:
            dataset_choice = input("\nEnter dataset number to use for evaluation: ")
            try:
                dataset_idx = int(dataset_choice) - 1
                if 0 <= dataset_idx < len(processed_dirs):
                    selected_data_dir = processed_dirs[dataset_idx][0]  # Get the path part
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    logger.info(f"Using dataset: {selected_data_dir.name}")
    
    # Create model-specific visualization directory
    model_viz_dir = OUT_DIR / model_name
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
        test_dataset = SiameseDataset(str(selected_data_dir / "test"), transform=transform, test_mode=True)
    else:
        test_dataset = datasets.ImageFolder(selected_data_dir / "test", transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, pin_memory=True)
    
    # Load model
    num_classes = len(test_dataset.classes) if model_type != 'siamese' else 2
    model = get_model(model_type, num_classes).to(device)
    
    # Try to load best_model.pth first, if not found, try best_checkpoint.pth
    best_model_path = model_checkpoint_dir / 'best_model.pth'
    best_checkpoint_path = model_checkpoint_dir / 'best_checkpoint.pth'
    
    if best_model_path.exists():
        logger.info(f"Loading best_model.pth for {model_type} model {model_name}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    elif best_checkpoint_path.exists():
        logger.info(f"Loading best_checkpoint.pth for {model_type} model {model_name}")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError(f"Neither best_model.pth nor best_checkpoint.pth found in {model_checkpoint_dir}")
    
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
    
    # For siamese networks, also track identities for person-by-person analysis
    if model_type == 'siamese':
        all_identities_1 = []  # Identity of first image in pair
        all_identities_2 = []  # Identity of second image in pair
        identity_map = {}      # Map from file path to identity
        
        # Create identity map from test dataset
        for root, dirs, files in os.walk(str(selected_data_dir / "test")):
            for dir_name in dirs:
                identity = dir_name  # Folder name is the identity
                identity_path = os.path.join(root, dir_name)
                for file_name in os.listdir(identity_path):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        file_path = os.path.join(identity_path, file_name)
                        identity_map[file_path] = identity
        
        logger.info(f"Created identity map with {len(identity_map)} images across {len(set(identity_map.values()))} identities")
    
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
                
                # For siamese, also track the image pair identities if we have access to them
                if hasattr(test_dataset, 'image_pairs') and test_dataset.image_pairs:
                    # Get the paths for the current batch
                    batch_indices = list(range(len(all_predictions) - len(pred), len(all_predictions)))
                    for i, idx in enumerate(batch_indices):
                        if idx < len(test_dataset.image_pairs):
                            img1_path, img2_path = test_dataset.image_pairs[idx]
                            img1_identity = identity_map.get(img1_path, 'unknown')
                            img2_identity = identity_map.get(img2_path, 'unknown')
                            
                            # If identity is still unknown, try to get it from the path
                            if img1_identity == 'unknown':
                                img1_identity = Path(img1_path).parent.name
                            if img2_identity == 'unknown':
                                img2_identity = Path(img2_path).parent.name
                                
                            all_identities_1.append(img1_identity)
                            all_identities_2.append(img2_identity)
                else:
                    # Fallback: try to extract identity information directly from dataset
                    logger.info("No image_pairs attribute found in dataset, using fallback method for identity extraction")
                    try:
                        # If the dataset has a get_image_identities method, use it
                        if hasattr(test_dataset, 'get_image_identities'):
                            identities = test_dataset.get_image_identities()
                            logger.info(f"Retrieved {len(identities)} identities using get_image_identities method")
                            
                            # Use these identities to simulate pairs
                            for i in range(len(all_predictions)):
                                if i >= len(all_identities_1):  # Only add new ones
                                    img_idx = i % len(identities)
                                    identity = identities[img_idx]
                                    all_identities_1.append(identity)
                                    
                                    # For the second identity, use either the same or different one
                                    # based on the prediction
                                    if i < len(all_predictions) and all_predictions[i] > 0.5:
                                        all_identities_2.append(identity)  # Same identity
                                    else:
                                        # Get a different identity
                                        other_identities = [id for id in set(identities) if id != identity]
                                        if other_identities:
                                            all_identities_2.append(random.choice(other_identities))
                                        else:
                                            all_identities_2.append("unknown_other")
                        else:
                            # For each prediction, try to extract identities from directory structure
                            # This assumes images are organized in identity-named directories
                            for img_path in test_dataset.images[len(all_identities_1):len(all_predictions)]:
                                # Extract identity from parent directory name
                                identity = Path(img_path).parent.name
                                all_identities_1.append(identity)
                                
                                # For the second identity, we'd need the paired image
                                # Since we don't have it, we'll just use a placeholder based on whether 
                                # the prediction was "same" or "different"
                                idx = len(all_identities_2)
                                if idx < len(all_predictions):
                                    pred_val = all_predictions[idx]
                                    if pred_val > 0.5:  # Predicted as same
                                        all_identities_2.append(identity)  # Same identity
                                    else:
                                        all_identities_2.append("unknown_other")  # Different identity
                    except Exception as e:
                        logger.error(f"Failed to extract identity information using fallback method: {str(e)}")
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
        # Convert to numpy array explicitly to avoid attribute error
        all_probs_np = np.array(all_probs)
        fpr, tpr, _ = roc_curve(all_targets, -all_probs_np.ravel())
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    
    # Calculate PR AUC
    if model_type == 'siamese':
        # Convert to numpy array explicitly to avoid attribute error  
        all_probs_np = np.array(all_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(all_targets, -all_probs_np.ravel())
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
    
    # Determine class names
    if model_type == 'siamese':
        class_names = ['Same', 'Different']
        
        # Get unique identities if we have identity information
        if 'all_identities_1' in locals() and all_identities_1:
            unique_identities = sorted(set(all_identities_1 + all_identities_2))
            logger.info(f"Found {len(unique_identities)} unique identities for person-by-person analysis")
        else:
            unique_identities = None
            logger.warning("No identity information available for siamese network, cannot generate person-by-person analysis")
    else:
        # For non-siamese, use class names from the dataset
        class_names = test_dataset.classes
        logger.info(f"Using {len(class_names)} classes from dataset: {class_names}")
    
    # Save raw predictions for visualization
    model_results = {
        "predictions": all_predictions.tolist() if hasattr(all_predictions, 'tolist') else all_predictions,
        "targets": all_targets.tolist() if hasattr(all_targets, 'tolist') else all_targets,
        "probabilities": all_probs.tolist() if hasattr(all_probs, 'tolist') and (not hasattr(all_probs, 'ndim') or all_probs.ndim <= 2) 
                        else [p.tolist() if hasattr(p, 'tolist') else p for p in all_probs],
        "class_names": class_names,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "inference_time": float(avg_inference_time)
        }
    }
    
    # Save results to model-specific JSON file
    if model_type == 'siamese':
        results_file = model_viz_dir / 'siamese_network_results.json'
    elif model_type == 'arcface':
        results_file = model_viz_dir / 'arcface_model_results.json'
    elif model_type == 'baseline':
        results_file = model_viz_dir / 'baseline_model_results.json'
    elif model_type == 'cnn':
        results_file = model_viz_dir / 'cnn_model_results.json'
    else:
        results_file = model_viz_dir / f'{model_type}_model_results.json'
    
    with open(results_file, 'w') as f:
        json.dump(model_results, f, indent=2)
    
    logger.info(f"Saved model predictions to {results_file}")
    
    # Save to experiment_summary.json as well for compatibility
    summary_file = model_viz_dir / 'experiment_summary.json'
    
    experiment_summary = {
        "model_type": model_type,
        "model_name": model_name,
        "dataset": selected_data_dir.name,
        "metrics": model_results["metrics"],
        "class_names": model_results["class_names"]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    logger.info(f"Saved experiment summary to {summary_file}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=all_targets, 
        y_pred=all_predictions, 
        classes=class_names, 
        output_dir=str(model_viz_dir), 
        model_name=model_name,
        detailed=True  # Use detailed view with per-class metrics
    )
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    if model_type == 'siamese':
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Convert to numpy array explicitly for slicing
        all_probs_np = np.array(all_probs)
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(all_targets == i, all_probs_np[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
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
        # Convert to numpy array explicitly for slicing
        all_probs_np = np.array(all_probs)
        for i in range(len(class_names)):
            precision_i, recall_i, _ = precision_recall_curve(all_targets == i, all_probs_np[:, i])
            plt.plot(recall_i, precision_i, label=f'{class_names[i]} (AUC = {average_precision_score(all_targets == i, all_probs_np[:, i]):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(model_viz_dir / 'pr_curves.png')
    plt.close()
    
    # For siamese networks, also create a person-by-person confusion matrix if we have identity info
    if model_type == 'siamese' and 'all_identities_1' in locals() and all_identities_1 and len(all_identities_1) > 0:
        # Get unique identities as a sorted list
        unique_identities = sorted(set(all_identities_1 + all_identities_2))
        logger.info(f"Found {len(unique_identities)} unique identities for person-by-person analysis")
        
        # Skip if we don't have at least 2 identities
        if len(unique_identities) < 2:
            logger.warning("Not enough unique identities for person-by-person analysis")
        else:
            # Create a matrix of recognition rates between each identity
            identity_cm = np.zeros((len(unique_identities), len(unique_identities)))
            identity_counts = np.zeros((len(unique_identities), len(unique_identities)))
            
            # Fill the matrix
            for i in range(len(all_predictions)):
                if i < len(all_identities_1) and i < len(all_identities_2):
                    id1 = all_identities_1[i]
                    id2 = all_identities_2[i]
                    pred = all_predictions[i]
                    
                    # Only consider pairs where both identities are known
                    if id1 in unique_identities and id2 in unique_identities:
                        idx1 = unique_identities.index(id1)
                        idx2 = unique_identities.index(id2)
                        
                        # Same person (diagonal elements)
                        if id1 == id2:
                            # Increment correct recognition count if predicted as same
                            if pred == 1:  # Correctly predicted as same
                                identity_cm[idx1, idx2] += 1
                            identity_counts[idx1, idx2] += 1
                        # Different people (off-diagonal elements)
                        else:
                            # Increment correct recognition count if predicted as different
                            if pred == 0:  # Correctly predicted as different
                                identity_cm[idx1, idx2] += 1
                                identity_cm[idx2, idx1] += 1  # Mirror the matrix
                            identity_counts[idx1, idx2] += 1
                            identity_counts[idx2, idx1] += 1  # Mirror the matrix
            
            # Calculate recognition rates (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                identity_rates = np.divide(identity_cm, identity_counts)
                identity_rates = np.nan_to_num(identity_rates)  # Replace NaN with 0
            
            # Plot person-by-person recognition rate matrix
            plt.figure(figsize=(15, 12))
            sns.heatmap(identity_rates, annot=True, fmt='.2f', cmap='viridis',
                        xticklabels=unique_identities,
                        yticklabels=unique_identities)
            plt.title('Person-by-Person Recognition Rate')
            plt.xlabel('Person')
            plt.ylabel('Person')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(model_viz_dir / 'person_recognition_matrix.png')
            plt.close()
            
            # Also create a more compact visualization showing per-person performance
            per_person_accuracy = np.diag(identity_rates)
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(unique_identities, per_person_accuracy)
            plt.title('Recognition Accuracy per Person')
            plt.xlabel('Person')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1.0)
            plt.axhline(y=np.mean(per_person_accuracy), color='r', linestyle='--', label=f'Average: {np.mean(per_person_accuracy):.2f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(model_viz_dir / 'per_person_accuracy.png')
            plt.close()
    
    # Generate Grad-CAM visualizations
    logger.info("Generating Grad-CAM visualizations...")
    try:
        plot_gradcam_visualization(model, test_dataset, 5, str(model_viz_dir), model_name)
    except Exception as e:
        logger.error(f"Failed to generate Grad-CAM visualizations: {str(e)}")
    
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

def calculate_detailed_metrics(all_targets, all_predictions, all_probs, class_names):
    """Calculate detailed performance metrics for model evaluation."""
    # ...
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'inference_time': avg_inference_time
    }

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
                   target_layer: nn.Module, model_type: Optional[str] = None) -> np.ndarray:
    """Generate Grad-CAM visualization for a given image and model."""
    # Ensure image_tensor is a single image (add batch dimension if needed)
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
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
        logger.info(f"Generating Grad-CAM with model type: {model_type}")
        
        # Forward pass
        if model_type == 'siamese':
            # For Siamese network, we use the same image as both inputs
            output1, output2 = model(image_tensor, image_tensor)
            output = torch.pairwise_distance(output1, output2)  # Calculate distance
        else:
            try:
                output = model(image_tensor)
                logger.info(f"Model output shape: {output.shape if isinstance(output, torch.Tensor) else 'not a tensor'}")
            except Exception as e:
                logger.error(f"Error in model forward pass: {str(e)}")
                return np.zeros((224, 224), dtype=np.float32)
        
        if isinstance(output, tuple):
            output = output[0]
            logger.info("Output was a tuple, using first element")
        
        # If output doesn't have a batch dimension or is not a tensor with values,
        # we can't compute gradients - return empty heatmap
        if not isinstance(output, torch.Tensor) or output.numel() == 0:
            logger.error("Output is not a valid tensor or has no elements")
            return np.zeros((224, 224), dtype=np.float32)
            

        if model_type == 'siamese':
            score = output  # Use distance as score for siamese
            logger.info(f"Using distance as score for siamese model")
        else:
            # For classification models, use the predicted class's score
            score = torch.max(output) if output.numel() > 0 else None
            logger.info(f"Using max score for classification: {score.item() if score is not None else 'None'}")
        
        if score is None:
            logger.error("Could not compute score from output")
            return np.zeros((224, 224), dtype=np.float32)
        
        # Backward pass
        model.zero_grad()
        try:
            score.backward()
        except Exception as e:
            logger.error(f"Error in backward pass: {str(e)}")
            return np.zeros((224, 224), dtype=np.float32)
        
        # Check if we got any gradients
        if not gradients or len(gradients) == 0:
            logger.error("No gradients captured by hook")
            return np.zeros((224, 224), dtype=np.float32)
            
        # Get activations and gradients
        activation = activations[0].detach()
        gradient = gradients[0].detach()
        
        logger.info(f"Activation shape: {activation.shape}, Gradient shape: {gradient.shape}")
        
        # Global average pooling of gradients
        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
        
        # Weight the activations
        cam = torch.sum(weights * activation, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        if cam.max() > 1e-8: 
            cam = cam / (cam.max() + 1e-8)
        else:
            logger.warning("CAM has very small values, may result in poor visualization")
        

        cam_np = cam.squeeze().cpu().numpy()
        logger.info(f"Generated CAM with shape: {cam_np.shape}")
        

        if len(cam_np.shape) > 2:
            # If it has more than 2 dimensions, take the first channel
            cam_np = cam_np[0]
            logger.info(f"Adjusted CAM shape to 2D: {cam_np.shape}")
        
        return cam_np
    
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        # Return an empty heatmap instead of None
        return np.zeros((224, 224), dtype=np.float32)
    
    finally:
        # Clean up hooks
        handle1.remove()
        handle2.remove()

def plot_gradcam_visualization(model: nn.Module, images, 
                             num_samples: int, output_dir: str, model_name: str):
    """
    Plot Grad-CAM visualizations for sample images.
    
    Args:
        model: The model to visualize
        images: Either a dataset or a tensor of images
        num_samples: Maximum number of samples to visualize (will be capped at 5)
        output_dir: Directory to save visualizations
        model_name: Name of the model for file naming
    """
    device = next(model.parameters()).device
    
    # Limit to 5 samples maximum
    num_samples = min(5, num_samples)
    
    # Get target layer based on model type
    target_layer = None
    model_type = None
    
    logger.info(f"Finding target layer for model: {model_name}")
    
    # Log model architecture to help with debugging
    logger.info(f"Model architecture: {model.__class__.__name__}")
    top_level_attributes = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr))]
    logger.info(f"Top-level model attributes: {top_level_attributes}")
    
    if hasattr(model, 'models') and isinstance(model.models, nn.ModuleList):
        # This is an ensemble model, use the first model's layer
        if len(model.models) > 0 and hasattr(model.models[0], 'resnet'):
            target_layer = model.models[0].resnet.layer4[-1]
            model_type = 'ensemble'
            logger.info("Using ensemble model's first model's resnet.layer4[-1]")
        elif len(model.models) > 0 and hasattr(model.models[0], 'features'):
            target_layer = model.models[0].features[-1]
            model_type = 'ensemble'
            logger.info("Using ensemble model's first model's features[-1]")
    elif hasattr(model, 'resnet'):
        target_layer = model.resnet.layer4[-1]
        model_type = 'cnn'
        logger.info("Using resnet.layer4[-1] as target layer")
    elif 'cnn' in model_name.lower() or ('baseline' not in model_name.lower() and hasattr(model, 'layer4')):
        # Direct ResNet structure or CNN model
        target_layer = model.layer4[-1] if hasattr(model, 'layer4') else None
        model_type = 'cnn'
        logger.info("Using layer4[-1] as target layer (direct ResNet structure)")
    elif hasattr(model, 'conv3'):
        target_layer = model.conv3
        model_type = 'baseline'
        logger.info("Using conv3 as target layer (baseline model)")
    elif hasattr(model, 'conv'):
        if isinstance(model.conv, nn.ModuleList) or isinstance(model.conv, list):
            target_layer = model.conv[-3]  # Last conv layer
            logger.info("Using conv[-3] as target layer (siamese model)")
        else:
            target_layer = model.conv
            logger.info("Using conv as target layer (siamese model)")
        model_type = 'siamese'
    elif hasattr(model, 'attention_layers') and len(model.attention_layers) > 0:
        target_layer = model.attention_layers[-1]
        model_type = 'attention'
        logger.info("Using attention_layers[-1] as target layer")
    else:
        logger.warning(f"Could not determine target layer for Grad-CAM for model type: {model_name}")
        # Use a fallback approach - look for any convolutional layer
        found_layer = False
        # First try to find the last convolutional layer
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                logger.info(f"Found conv layer: {name}")
        
        if last_conv is not None:
            target_layer = last_conv
            model_type = 'unknown'
            found_layer = True
            logger.info(f"Using last found Conv2d layer as target layer for Grad-CAM")
        else:
            # If still no convolutional layer found
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    model_type = 'unknown'
                    found_layer = True
                    logger.info(f"Using {name} as target layer for Grad-CAM")
                    break
        
        if not found_layer:
            logger.error(f"No suitable layer found for Grad-CAM visualization")
            return
    
    # Verify target layer was found
    if target_layer is None:
        logger.error("Target layer is None, cannot generate Grad-CAM")
        return
    
    logger.info(f"Using target layer: {target_layer.__class__.__name__} for model type: {model_type}")
    
    # Create standard output directories matching experiment_manager's structure
    output_path = Path(output_dir)
    
    # Ensure standard plots directory exists (important for consistent structure)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create resources directory (where your screenshot shows visualizations)
    resources_dir = plots_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create model_name directory for backward compatibility
    if model_name and model_name.strip():
        model_dir = output_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving visualizations to multiple locations for compatibility:")
    logger.info(f"1. {resources_dir}")
    logger.info(f"2. {plots_dir}")
    if model_name and model_name.strip():
        logger.info(f"3. {model_dir}")
    
    # Create figure for visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    successful_samples = 0
    max_attempts = num_samples * 3  # Try up to 3 times the number of samples
    attempts = 0
    
    # Create a dataset iterator if passed a dataloader or dataset
    if hasattr(images, 'dataset'):  # DataLoader
        dataset = images.dataset
    elif torch.is_tensor(images):  # Tensor of images
        # Process each image in the batch individually
        for i in range(min(num_samples, images.size(0))):
            try:
                image_tensor = images[i:i+1].to(device)  # Keep batch dimension but just one image
                
                # Generate Grad-CAM
                if target_layer is not None:
                    cam = generate_gradcam(model, image_tensor, target_layer, model_type)
                else:
                    # If target layer is None, return empty heatmap
                    cam = np.zeros((224, 224), dtype=np.float32)
                
                # Convert tensor to numpy for plotting (remove batch dimension)
                img_np = images[i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
                
                # Plot original image
                axes[successful_samples, 0].imshow(img_np)
                axes[successful_samples, 0].set_title('Original Image')
                axes[successful_samples, 0].axis('off')
                
                # Plot heatmap
                axes[successful_samples, 1].imshow(cam, cmap='jet')
                axes[successful_samples, 1].set_title('Grad-CAM Heatmap')
                axes[successful_samples, 1].axis('off')
                
                # Plot overlay - ensure cam and img_np have compatible shapes
                if cam.shape[:2] != img_np.shape[:2]:
                    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                else:
                    cam_resized = cam
                
                # Create heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), 2)  # 2 is COLORMAP_JET
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Ensure shapes match for overlay
                if heatmap.shape[:2] != img_np.shape[:2]:
                    logger.warning(f"Shape mismatch: heatmap {heatmap.shape}, image {img_np.shape}")
                    # Resize heatmap to match image
                    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                
                # Create overlay by properly broadcasting the shapes
                if len(img_np.shape) == 3 and len(heatmap.shape) == 3:
                    overlay = (0.7 * img_np + 0.3 * heatmap/255).clip(0, 1)
                    axes[successful_samples, 2].imshow(overlay)
                else:
                    # If shapes are incompatible, show just the heatmap
                    axes[successful_samples, 2].imshow(heatmap/255)
                
                axes[successful_samples, 2].set_title('Overlay')
                axes[successful_samples, 2].axis('off')
                
                successful_samples += 1
                if successful_samples >= num_samples:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing image tensor {i}: {str(e)}")
                continue
                
        # Save the figure with the samples we've processed
        if successful_samples > 0:
            # If we didn't fill all rows, remove empty subplots
            if successful_samples < num_samples:
                for i in range(successful_samples, num_samples):
                    for j in range(3):
                        fig.delaxes(axes[i, j])
            
            plt.tight_layout()
            viz_file1 = resources_dir / 'gradcam_visualization.png'
            plt.savefig(viz_file1)
            logger.info(f"Saved visualizations to {viz_file1}")
            
            # Save to model name directory for backward compatibility
            if model_name and model_name.strip():
                viz_file2 = model_dir / 'gradcam_visualization.png'
                plt.savefig(viz_file2)
                logger.info(f"Saved visualizations to {viz_file2}")
            
            plt.close()
            
            # Return success with clear path info
            logger.info(f"Successfully saved Grad-CAM visualizations to multiple locations for compatibility")
            return
            
        # If we couldn't process any images in the tensor, show error
        logger.error("Failed to generate any Grad-CAM visualizations from tensor")
        return
    
    # If we have a dataset, sample from it
    else:
        dataset = images
        while successful_samples < num_samples and attempts < max_attempts:
            attempts += 1
            try:
                # Get random sample
                idx = random.randint(0, len(dataset)-1)
                
                # Handle different dataset types
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
                if target_layer is not None:
                    cam = generate_gradcam(model, img_tensor, target_layer, model_type)
                else:
                    # If target layer is None, return empty heatmap
                    cam = np.zeros((224, 224), dtype=np.float32)
                
                # Convert tensor to numpy for plotting
                img_np = image.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
                
                # Plot original image
                axes[successful_samples, 0].imshow(img_np)
                axes[successful_samples, 0].set_title('Original Image')
                axes[successful_samples, 0].axis('off')
                
                # Plot heatmap
                axes[successful_samples, 1].imshow(cam, cmap='jet')
                axes[successful_samples, 1].set_title('Grad-CAM Heatmap')
                axes[successful_samples, 1].axis('off')
                
                # Plot overlay - ensure cam and img_np have compatible shapes
                if cam.shape[:2] != img_np.shape[:2]:
                    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                else:
                    cam_resized = cam
                
                # Create heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), 2)  # 2 is COLORMAP_JET
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Create overlay
                overlay = (0.7 * img_np + 0.3 * heatmap/255).clip(0, 1)
                
                axes[successful_samples, 2].imshow(overlay)
                axes[successful_samples, 2].set_title('Overlay')
                axes[successful_samples, 2].axis('off')
                
                successful_samples += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
    
    # Save or display error message
    if successful_samples == 0:
        logger.error("Failed to generate any Grad-CAM visualizations")
        return
    
    # If we didn't fill all rows, remove empty subplots
    if successful_samples < num_samples:
        for i in range(successful_samples, num_samples):
            for j in range(3):
                fig.delaxes(axes[i, j])
    
    plt.tight_layout()
    viz_file1 = resources_dir / 'gradcam_visualization.png'
    plt.savefig(viz_file1)
    logger.info(f"Saved visualizations to {viz_file1}")
    
    # Save to model name directory for backward compatibility
    if model_name and model_name.strip():
        viz_file2 = model_dir / 'gradcam_visualization.png'
        plt.savefig(viz_file2)
        logger.info(f"Saved visualizations to {viz_file2}")
    
    plt.close()
    
    # Return success with clear path info
    logger.info(f"Successfully saved Grad-CAM visualizations to multiple locations for compatibility")

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
    
    # Try to load best_model.pth first, if not found, try best_checkpoint.pth
    best_model_path = model_checkpoint_dir / 'best_model.pth'
    best_checkpoint_path = model_checkpoint_dir / 'best_checkpoint.pth'
    
    if best_model_path.exists():
        logger.info(f"Loading best_model.pth for {model_type} model {model_name}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    elif best_checkpoint_path.exists():
        logger.info(f"Loading best_checkpoint.pth for {model_type} model {model_name}")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError(f"Neither best_model.pth nor best_checkpoint.pth found in {model_checkpoint_dir}")
    
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        prob, pred_idx = torch.max(probs, 1)
        
    return classes[pred_idx.item()], prob.item() 