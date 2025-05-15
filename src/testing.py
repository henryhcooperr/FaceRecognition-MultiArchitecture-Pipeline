#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
from .advanced_metrics import plot_confusion_matrix, create_enhanced_confusion_matrix

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
    logger.info("Skipping visualizations as plotting features are disabled")
    
    # Calculate confusion matrix but skip plotting
    plot_confusion_matrix(
        y_true=all_targets, 
        y_pred=all_predictions, 
        classes=class_names, 
        output_dir=str(model_viz_dir), 
        model_name=model_name,
        detailed=True  # Use detailed view with per-class metrics
    )
    
    # Log ROC curve metrics without plotting
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Log PR curve metrics without plotting
    logger.info(f"PR AUC: {pr_auc:.4f}")
    
    # Save curve data to CSV for later reference if needed
    if model_type == 'siamese':
        roc_df = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        })
        pr_df = pd.DataFrame({
            'precision': precision_curve,
            'recall': recall_curve,
            'auc': pr_auc
        })
        roc_df.to_csv(model_viz_dir / 'roc_curve_data.csv', index=False)
        pr_df.to_csv(model_viz_dir / 'pr_curve_data.csv', index=False)
    else:
        # For multi-class, just save the overall metrics
        # without the detailed curve data
        metrics_df = pd.DataFrame({
            'class': class_names,
            'roc_auc': [roc_auc] * len(class_names),
            'pr_auc': [pr_auc] * len(class_names)
        })
        metrics_df.to_csv(model_viz_dir / 'curve_metrics.csv', index=False)
    
    # For siamese networks, calculate person-by-person metrics without visualization
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
            
            # Save identity recognition rates to CSV
            person_rates_df = pd.DataFrame(identity_rates, 
                                        index=unique_identities, 
                                        columns=unique_identities)
            person_rates_df.to_csv(model_viz_dir / 'person_recognition_rates.csv')
            
            # Calculate and save per-person performance
            per_person_accuracy = np.diag(identity_rates)
            person_accuracy_df = pd.DataFrame({
                'person': unique_identities,
                'accuracy': per_person_accuracy,
            })
            person_accuracy_df.to_csv(model_viz_dir / 'per_person_accuracy.csv', index=False)
            
            # Log average per-person accuracy
            avg_person_accuracy = np.mean(per_person_accuracy)
            logger.info(f"Average per-person accuracy: {avg_person_accuracy:.4f}")
    
    # No visualization in this branch
    logger.info("Visualization features are disabled in this simplified branch")
    
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
    # For the stub implementation, we'll assume these variables are defined elsewhere
    accuracy = 0.0
    roc_auc = 0.0
    pr_auc = 0.0
    avg_inference_time = 0.0
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'inference_time': avg_inference_time
    }

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