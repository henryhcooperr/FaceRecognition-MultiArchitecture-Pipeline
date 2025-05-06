#!/usr/bin/env python3

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
from PIL import Image
from random import shuffle
import random
from facenet_pytorch import MTCNN
import cv2
import json
import albumentations as A
import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                CHECKPOINTS_DIR, VISUALIZATIONS_DIR,
                PROCESSED_DATA_DIR / "train",
                PROCESSED_DATA_DIR / "val",
                PROCESSED_DATA_DIR / "test"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    def __init__(self,
                 name: str,
                 use_mtcnn: bool = True,
                 face_margin: float = 0.4,
                 final_size: Tuple[int, int] = (224, 224),
                 augmentation: bool = True):
        """Initialize preprocessing configuration with simplified parameters."""
        self.name = name
        self.use_mtcnn = use_mtcnn
        self.face_margin = face_margin
        self.final_size = final_size
        self.min_face_size = 20  # Fixed reasonable default
        self.thresholds = [0.6, 0.7, 0.7]  # Fixed MTCNN defaults
        self.augmentation = augmentation
        # Fixed augmentation parameters with reasonable defaults
        self.aug_rotation_range = 20
        self.aug_brightness_range = 0.2
        self.aug_contrast_range = 0.2
        self.aug_scale_range = 0.1
        self.horizontal_flip = True

    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PreprocessingConfig':
        """Create config from dictionary."""
        # Extract only the parameters needed for the constructor
        constructor_params = {
            'name': config_dict['name'],
            'use_mtcnn': config_dict['use_mtcnn'],
            'face_margin': config_dict['face_margin'],
            'final_size': config_dict['final_size'],
            'augmentation': config_dict['augmentation']
        }
        
        # Create the config with constructor params
        config = cls(**constructor_params)
        
        # Set any additional attributes that may have been added
        for key, value in config_dict.items():
            if key not in constructor_params:
                setattr(config, key, value)
                
        return config

def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Align face based on eye landmarks."""
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    
    # Calculate angle to rotate image
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Get the center point between the eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
    
    # Rotate the image
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return aligned

def get_face_bbox_with_margin(bbox: np.ndarray, margin: float, 
                            img_shape: Tuple[int, int]) -> np.ndarray:
    """Get face bounding box with margin."""
    height, width = img_shape[:2]
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    margin_x = int(width * margin)
    margin_y = int(height * margin)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img_shape[1], x2 + margin_x)
    y2 = min(img_shape[0], y2 + margin_y)
    
    return np.array([x1, y1, x2, y2])

def preprocess_image(image_path: str, config: PreprocessingConfig) -> Optional[Image.Image]:
    """Preprocess a single image according to configuration."""
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if config.use_mtcnn:
            mtcnn = MTCNN(
                image_size=config.final_size[0],
                margin=config.face_margin,
                min_face_size=config.min_face_size,
                thresholds=config.thresholds,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            
            boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                logger.warning(f"No face detected in {image_path}")
                return None
            
            # Use the face with highest probability
            box = boxes[0]
            landmark = landmarks[0]
            
            # Get face bbox with margin
            bbox = get_face_bbox_with_margin(box, config.face_margin, image.shape)
            
            # Align face using landmarks
            aligned_face = align_face(image, landmark)
            
            # Crop to face region
            face = aligned_face[int(bbox[1]):int(bbox[3]), 
                              int(bbox[0]):int(bbox[2])]
        else:
            face = image
        
        # Resize
        face = cv2.resize(face, config.final_size)
        
        # Convert to PIL Image
        face_pil = Image.fromarray(face)
        
        if config.augmentation:
            # Define augmentation pipeline
            transform = A.Compose([
                A.Rotate(limit=config.aug_rotation_range, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=config.aug_brightness_range,
                    contrast_limit=config.aug_contrast_range,
                    p=0.5
                ),
                A.RandomScale(scale_limit=config.aug_scale_range, p=0.5),
                A.HorizontalFlip(p=0.5 if config.horizontal_flip else 0),
            ])
            
            # Apply augmentations
            augmented = transform(image=np.array(face_pil))
            face_pil = Image.fromarray(augmented['image'])
        
        return face_pil
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None

def visualize_preprocessing_steps(image_path: str, config: PreprocessingConfig, output_dir: Path):
    """Visualize preprocessing steps for a single image."""
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure for visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Preprocessing Steps', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        if config.use_mtcnn:
            # Initialize MTCNN
            mtcnn = MTCNN(
                image_size=config.final_size[0],
                margin=config.face_margin,
                min_face_size=config.min_face_size,
                thresholds=config.thresholds,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            
            # Detect face and get landmarks
            boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                logger.warning(f"No face detected in {image_path}")
                return None
            
            # Use the face with highest probability
            box = boxes[0]
            landmark = landmarks[0]
            
            # Draw bounding box and landmarks on original image
            img_with_boxes = image.copy()
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for point in landmark:
                cv2.circle(img_with_boxes, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
            
            axes[0, 1].imshow(img_with_boxes)
            axes[0, 1].set_title('Face Detection with Landmarks')
            axes[0, 1].axis('off')
            
            # Get face bbox with margin
            bbox = get_face_bbox_with_margin(box, config.face_margin, image.shape)
            
            # Align face using landmarks
            aligned_face = align_face(image, landmark)
            
            # Draw aligned face with bounding box
            aligned_with_box = aligned_face.copy()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(aligned_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            axes[0, 2].imshow(aligned_with_box)
            axes[0, 2].set_title('Aligned Face with Margin')
            axes[0, 2].axis('off')
            
            # Final cropped and resized face
            face = aligned_face[int(bbox[1]):int(bbox[3]), 
                              int(bbox[0]):int(bbox[2])]
            face = cv2.resize(face, config.final_size)
            
            axes[1, 0].imshow(face)
            axes[1, 0].set_title('Final Processed Face')
            axes[1, 0].axis('off')
            
            if config.augmentation:
                # Define augmentation pipeline
                transform = A.Compose([
                    A.Rotate(limit=config.aug_rotation_range, p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=config.aug_brightness_range,
                        contrast_limit=config.aug_contrast_range,
                        p=1.0
                    ),
                    A.RandomScale(scale_limit=config.aug_scale_range, p=1.0),
                    A.HorizontalFlip(p=1.0 if config.horizontal_flip else 0),
                ])
                
                # Apply augmentations
                augmented = transform(image=face)
                augmented_face = augmented['image']
                
                # Show different augmentations
                axes[1, 1].imshow(augmented_face)
                axes[1, 1].set_title('Augmented Face')
                axes[1, 1].axis('off')
                
                # Show another augmentation with different parameters
                transform2 = A.Compose([
                    A.Rotate(limit=config.aug_rotation_range, p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=config.aug_brightness_range,
                        contrast_limit=config.aug_contrast_range,
                        p=1.0
                    ),
                    A.RandomScale(scale_limit=config.aug_scale_range, p=1.0),
                    A.HorizontalFlip(p=1.0 if config.horizontal_flip else 0),
                ])
                
                augmented2 = transform2(image=face)
                augmented_face2 = augmented2['image']
                
                axes[1, 2].imshow(augmented_face2)
                axes[1, 2].set_title('Another Augmentation')
                axes[1, 2].axis('off')
                
                # Show augmentation parameters
                params_text = f"Augmentation Parameters:\n"
                params_text += f"Rotation Range: ±{config.aug_rotation_range}°\n"
                params_text += f"Brightness Range: ±{config.aug_brightness_range}\n"
                params_text += f"Contrast Range: ±{config.aug_contrast_range}\n"
                params_text += f"Scale Range: ±{config.aug_scale_range}\n"
                params_text += f"Horizontal Flip: {config.horizontal_flip}"
                
                # Add text box with parameters
                plt.figtext(0.02, 0.02, params_text, fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.8))
            
            # Save visualization
            plt.tight_layout()
            plt.savefig(output_dir / f'preprocessing_{Path(image_path).stem}.png')
            plt.close()
            
            return face
        
        return image
    
    except Exception as e:
        logger.error(f"Error visualizing preprocessing for {image_path}: {str(e)}")
        return None

def process_raw_data(config: PreprocessingConfig,
                    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                    test_mode: bool = False):
    """Process raw data with given preprocessing configuration."""
    logger.info(f"Processing raw data with config: {config.name}")
    
    # Create preprocessing-specific directories
    processed_base = PROCESSED_DATA_DIR / config.name
    for split in ["train", "val", "test"]:
        split_dir = processed_base / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
    
    # Create visualization directory
    viz_dir = processed_base / "preprocessing_visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Save preprocessing configuration
    config_path = processed_base / "preprocessing_config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    
    # Process each class in raw data
    for class_dir in RAW_DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not image_files:
            logger.warning(f"No images found in {class_dir}")
            continue
            
        shuffle(image_files)
        
        # In test mode, limit to 2 images per class
        if test_mode:
            image_files = image_files[:2]
        
        # Calculate split sizes
        n_images = len(image_files)
        n_train = int(n_images * split_ratio[0])
        n_val = int(n_images * split_ratio[1])
        
        # Split indices
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Create class-specific visualization directory
        class_viz_dir = viz_dir / class_name
        class_viz_dir.mkdir(exist_ok=True)
        
        # Visualize preprocessing for first 3 images
        for i, file in enumerate(image_files[:3]):
            visualize_preprocessing_steps(str(file), config, class_viz_dir)
        
        # Process and save files
        for split, files in [("train", train_files), 
                           ("val", val_files), 
                           ("test", test_files)]:
            split_class_dir = processed_base / split / class_name
            split_class_dir.mkdir(exist_ok=True)
            
            # Use tqdm safely, handle cases where it might be mocked
            try:
                # Try using tqdm normally
                files_iter = tqdm(files, desc=f"Processing {split}/{class_name}")
            except Exception:
                # If tqdm fails (e.g., during testing), fall back to normal iteration
                logger.warning("Could not use tqdm, falling back to normal iteration")
                files_iter = files
            
            for file in files_iter:
                try:
                    processed_face = preprocess_image(str(file), config)
                    if processed_face is not None:
                        save_path = split_class_dir / file.name
                        processed_face.save(save_path)
                except Exception as e:
                    logger.error(f"Error processing {file}: {str(e)}")
                    continue
    
    logger.info("Data processing complete!")
    logger.info(f"Preprocessing visualizations saved in: {viz_dir}")
    return processed_base

def get_preprocessing_config() -> PreprocessingConfig:
    """Interactive function to get preprocessing configuration from user."""
    print("\nPreprocessing Configuration")
    
    name = input("Enter a name for this preprocessing configuration: ")
    
    use_mtcnn = get_user_confirmation("Use MTCNN for face detection? (y/n): ")
    
    if use_mtcnn:
        face_margin = float(input("Enter face margin (default 0.4): ") or "0.4")
    else:
        face_margin = 0.4
    
    size_input = input("Enter final image size as width,height (default 224,224): ")
    if size_input:
        final_size = tuple(map(int, size_input.split(",")))
    else:
        final_size = (224, 224)
    
    use_augmentation = get_user_confirmation("Use data augmentation? (y/n): ")
    
    return PreprocessingConfig(
        name=name,
        use_mtcnn=use_mtcnn,
        face_margin=face_margin,
        final_size=final_size,
        augmentation=use_augmentation
    )

class BaselineNet(nn.Module):
    """Baseline CNN model for face recognition."""
    def __init__(self, num_classes: int = 18, input_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate feature map size after convolutions and pooling
        # Input -> Conv1 -> Pool -> Conv2 -> Pool -> Conv3 -> Pool
        h, w = input_size
        # Conv1: no padding, kernel size 3
        h, w = h - 2, w - 2
        # Pool: kernel size 2, stride 2
        h, w = h // 2, w // 2
        # Conv2: no padding, kernel size 3
        h, w = h - 2, w - 2
        # Pool: kernel size 2, stride 2
        h, w = h // 2, w // 2
        # Conv3: no padding, kernel size 3
        h, w = h - 2, w - 2
        # Pool: kernel size 2, stride 2
        h, w = h // 2, w // 2
        
        # Calculate final feature size
        self.features_size = 128 * h * w
        
        self.fc1 = nn.Linear(self.features_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        return x

class ResNetTransfer(nn.Module):
    """Transfer learning model based on ResNet-18."""
    def __init__(self, num_classes: int = 18):
        super().__init__()
        # Use weights parameter instead of pretrained to avoid deprecation warning
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def get_embedding(self, x):
        modules = list(self.resnet.children())[:-1]
        resnet_wo_fc = nn.Sequential(*modules)
        return resnet_wo_fc(x).squeeze()

class SiameseNet(nn.Module):
    """Siamese network for face verification."""
    def __init__(self):
        super().__init__()
        # Modified architecture for 224x224 input images
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4),  # Output: 54x54
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # Output: 26x26
            nn.Conv2d(64, 128, kernel_size=5, padding=2),  # Output: 26x26
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # Output: 12x12
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output: 12x12
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Output: 6x6
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Output: 6x6
            nn.ReLU(),
        )
        
        # Calculate the size of flattened features
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

    def get_embedding(self, x):
        return self.forward_one(x)

class ContrastiveLoss(nn.Module):
    """Contrastive loss function for Siamese network."""
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

def get_model(model_type: str, num_classes: int = 18, input_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    if model_type == 'baseline':
        return BaselineNet(num_classes=num_classes, input_size=input_size)
    elif model_type == 'cnn':
        return ResNetTransfer(num_classes=num_classes)
    elif model_type == 'siamese':
        return SiameseNet()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def get_criterion(model_type: str) -> nn.Module:
    if model_type in ['baseline', 'cnn']:
        return nn.CrossEntropyLoss()
    elif model_type == 'siamese':
        return ContrastiveLoss()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

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
    processed_dirs = [d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
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
    train_dataset = datasets.ImageFolder(selected_data_dir / "train", transform=transform)
    val_dataset = datasets.ImageFolder(selected_data_dir / "val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = get_model(model_type, num_classes=len(train_dataset.classes))
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
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
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
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
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

def plot_gradcam_visualization(model: nn.Module, dataset: Dataset, 
                             num_samples: int, output_dir: str, model_name: str):
    """Plot Grad-CAM visualizations for sample images."""
    device = next(model.parameters()).device
    
    # Get target layer based on model type
    if isinstance(model, ResNetTransfer):
        target_layer = model.resnet.layer4[-1]
        model_type = 'cnn'
    elif isinstance(model, BaselineNet):
        target_layer = model.conv3
        model_type = 'baseline'
    else:  # SiameseNet
        target_layer = model.conv[-3]  # Last conv layer
        model_type = 'siamese'
    
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
            if isinstance(dataset, SiameseDataset):
                img1, _, _ = dataset[idx]
                image = img1
            else:
                image, label = dataset[idx]
            
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

def plot_tsne_embeddings(model: nn.Module, dataset: Dataset, 
                        output_dir: str, model_name: str):
    """Generate t-SNE visualization of the embeddings."""
    device = next(model.parameters()).device
    model.eval()
    
    embeddings = []
    labels = []
    classes = []
    
    # Get embeddings for all images
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Generating embeddings"):
            if isinstance(dataset, SiameseDataset):
                img1, _, _ = dataset[i]
                label = dataset.labels[i]
                class_name = dataset.classes[label]
            else:
                img1, label = dataset[i]
                class_name = dataset.classes[label]
            
            img_tensor = img1.unsqueeze(0).to(device)
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
                                 label=classes[i], markersize=10)
                      for i in unique_labels]
    plt.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5))
    
    plt.title('t-SNE visualization of face embeddings')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / model_name / 'tsne_embeddings.png', 
                bbox_inches='tight')
    plt.close()

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
    processed_dirs = [d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir() and (d / "test").exists()]
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
    model_viz_dir = VISUALIZATIONS_DIR / model_name
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
    model.load_state_dict(torch.load(model_checkpoint_dir / 'best_model.pth'))
    model.eval()
    
    # Initialize metrics
    all_predictions = []
    all_targets = []
    all_probs = []
    total_loss = 0
    criterion = get_criterion(model_type)
    
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
                
                # Measure inference time
                start_time = time.time()
                outputs = model(images)
                inference_times.append(time.time() - start_time)
                
                loss = criterion(outputs, labels.to(device))
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.numpy())
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
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
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
            precision_curve, recall_curve, _ = precision_recall_curve(all_targets == i, all_probs[:, i])
            plt.plot(recall_curve, precision_curve, label=f'{test_dataset.classes[i]} (AUC = {auc(recall_curve, precision_curve):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(model_viz_dir / 'pr_curves.png')
    plt.close()
    
    # Generate Grad-CAM visualizations
    logger.info("Generating Grad-CAM visualizations...")
    plot_gradcam_visualization(model, test_dataset, num_samples=5, 
                             output_dir=str(VISUALIZATIONS_DIR), model_name=model_name)
    
    # Generate t-SNE embeddings
    logger.info("Generating t-SNE embeddings...")
    plot_tsne_embeddings(model, test_dataset, 
                        str(VISUALIZATIONS_DIR), model_name)
    
    logger.info("Evaluation complete! Check the visualizations directory for results.")

def predict_image(model_type: str, image_path: str, model_name: Optional[str] = None) -> Tuple[str, float]:
    """Predict the class of a single image."""
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
    
    # Load class names from training data
    class_names = sorted(os.listdir(PROCESSED_DATA_DIR / "train"))
    
    # Load model
    model = get_model(model_type, len(class_names)).to(device)
    model.load_state_dict(torch.load(model_checkpoint_dir / 'best_model.pth'))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        
    return class_names[predicted.item()], max_prob.item()

def check_gpu():
    """Check if GPU is available and print device information."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU is available! Found {gpu_count} GPU(s)")
        logger.info(f"Using: {gpu_name}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        # Print memory information
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_cached = torch.cuda.memory_reserved(0) / 1024**2
        logger.info(f"Memory Allocated: {memory_allocated:.2f} MB")
        logger.info(f"Memory Cached: {memory_cached:.2f} MB")
    else:
        device = torch.device('cpu')
        logger.warning("No GPU available, using CPU instead")
        logger.info(f"PyTorch Version: {torch.__version__}")
    
    return device

def cleanup_old_model(model_type: str):
    """Remove old model files for the specified model type."""
    old_model = CHECKPOINTS_DIR / f'best_model_{model_type}.pth'
    old_checkpoint = CHECKPOINTS_DIR / f'checkpoint_{model_type}.pth'
    
    if old_model.exists():
        old_model.unlink()
        logger.info(f"Removed old best model: {old_model}")
    if old_checkpoint.exists():
        old_checkpoint.unlink()
        logger.info(f"Removed old checkpoint: {old_checkpoint}")

def get_user_confirmation(prompt: str = "Continue? (y/n): ") -> bool:
    """Get user confirmation for an action."""
    while True:
        choice = input(prompt).lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no', 'b', 'back']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no/back")

def tune_hyperparameters(model_type: str, dataset_path: Path, n_trials: int = 50) -> Dict[str, Any]:
    """Tune hyperparameters for a given model type using Optuna."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting hyperparameter tuning for {model_type} model")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    def objective(trial):
        # Define hyperparameter search space
        if model_type == 'siamese':
            params = {
                'batch_size': trial.suggest_int('batch_size', 16, 128),
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                'margin': trial.suggest_float('margin', 1.0, 3.0)
            }
        else:
            params = {
                'batch_size': trial.suggest_int('batch_size', 16, 128),
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7)
            }
        
        # Initialize model
        num_classes = len(train_dataset.classes) if model_type != 'siamese' else 2
        model = get_model(model_type, num_classes).to(device)
        
        # Set dropout rate for baseline and CNN models
        if model_type in ['baseline', 'cnn']:
            if model_type == 'baseline':
                model.dropout.p = params['dropout_rate']
            else:  # CNN
                model.resnet.fc = nn.Sequential(
                    nn.Dropout(p=params['dropout_rate']),
                    nn.Linear(model.resnet.fc.in_features, num_classes)
                )
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = get_criterion(model_type)
        if model_type == 'siamese':
            criterion = ContrastiveLoss(margin=params['margin'])
        
        # Train for a few epochs to evaluate
        best_val_acc = 0
        for epoch in range(5):  # Use fewer epochs for tuning
            model.train()
            train_loss = 0
            for batch in train_loader:
                if model_type == 'siamese':
                    img1, img2, label = batch
                    img1, img2 = img1.to(device), img2.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    out1, out2 = model(img1, img2)
                    loss = criterion(out1, out2, label)
                else:
                    data, target = batch
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    if model_type == 'siamese':
                        img1, img2, label = batch
                        img1, img2 = img1.to(device), img2.to(device)
                        label = label.to(device)
                        out1, out2 = model(img1, img2)
                        dist = F.pairwise_distance(out1, out2)
                        pred = (dist < 0.5).float()
                        correct += pred.eq(label).sum().item()
                    else:
                        data, target = batch
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += criterion(output, target).item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
            
            val_acc = 100. * correct / len(val_dataset)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        return best_val_acc
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print("\nHyperparameter Tuning Results:")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_trial.value:.2f}%")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / f'{model_type}_tuning_history.png')
    plt.close()
    
    # Plot parameter importance
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Parameter Importance')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / f'{model_type}_param_importance.png')
    plt.close()
    
    return study.best_trial.params

def main():
    """Interactive interface for the face recognition system."""
    while True:
        print("\nFace Recognition System")
        print("1. Process Raw Data")
        print("2. Train Model")
        print("3. Evaluate Model")
        print("4. Tune Hyperparameters")
        print("5. List Processed Datasets")
        print("6. List Trained Models")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            print("\nData Processing")
            if not get_user_confirmation("This will create a new preprocessed dataset. Continue? (y/n): "):
                continue
            
            config = get_preprocessing_config()
            if get_user_confirmation("Start processing? (y/n): "):
                processed_dir = process_raw_data(config)
                print(f"\nProcessed data saved in: {processed_dir}")
        
        elif choice == '2':
            print("\nModel Training")
            model_type = input("Enter model type (baseline/cnn/siamese): ")
            if model_type.lower() not in ['baseline', 'cnn', 'siamese']:
                print("Invalid model type")
                continue
            
            # List available processed datasets
            processed_dirs = [d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
            if not processed_dirs:
                print("No processed datasets found. Please process raw data first.")
                continue
            
            print("\nAvailable processed datasets:")
            for i, d in enumerate(processed_dirs, 1):
                print(f"{i}. {d.name}")
                # Try to load and display config info
                config_file = d / "preprocessing_config.json"
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                        print(f"   - MTCNN: {config.get('use_mtcnn', 'N/A')}")
                        print(f"   - Face Margin: {config.get('face_margin', 'N/A')}")
                        print(f"   - Image Size: {config.get('final_size', 'N/A')}")
                    except:
                        pass
            
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
            
            model_name = input("Enter model name (optional, press Enter for automatic versioning): ")
            if not model_name:
                model_name = None
            
            epochs = int(input("Enter number of epochs (default 50): ") or "50")
            batch_size = int(input("Enter batch size (default 32): ") or "32")
            lr = float(input("Enter learning rate (default 0.001): ") or "0.001")
            
            if get_user_confirmation("Start training? (y/n): "):
                trained_model_name = train_model(
                    model_type, model_name, batch_size, epochs, lr=lr
                )
                print(f"\nModel trained and saved as: {trained_model_name}")
        
        elif choice == '3':
            print("\nModel Evaluation")
            model_type = input("Enter model type (baseline/cnn/siamese): ")
            if model_type.lower() not in ['baseline', 'cnn', 'siamese']:
                print("Invalid model type")
                continue
            
            # List available models of this type
            model_dirs = list(CHECKPOINTS_DIR.glob(f'{model_type}_*'))
            if not model_dirs:
                print(f"No trained models found for type: {model_type}")
                continue
            
            print("\nAvailable models:")
            for i, model_dir in enumerate(model_dirs, 1):
                print(f"{i}. {model_dir.name}")
            
            while True:
                model_choice = input("\nEnter model number (or press Enter for latest): ")
                if not model_choice:
                    model_name = None
                    break
                try:
                    model_idx = int(model_choice) - 1
                    if 0 <= model_idx < len(model_dirs):
                        model_name = model_dirs[model_idx].name
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            try:
                evaluate_model(model_type, model_name)
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
        
        elif choice == '4':
            print("\nHyperparameter Tuning")
            model_type = input("Enter model type (baseline/cnn/siamese): ")
            if model_type.lower() not in ['baseline', 'cnn', 'siamese']:
                print("Invalid model type")
                continue
            
            # List available processed datasets
            processed_dirs = [d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
            if not processed_dirs:
                print("No processed datasets found. Please process raw data first.")
                continue
            
            print("\nAvailable processed datasets:")
            for i, d in enumerate(processed_dirs, 1):
                print(f"{i}. {d.name}")
            
            while True:
                dataset_choice = input("\nEnter dataset number to use for tuning: ")
                try:
                    dataset_idx = int(dataset_choice) - 1
                    if 0 <= dataset_idx < len(processed_dirs):
                        selected_data_dir = processed_dirs[dataset_idx]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            n_trials = int(input("Enter number of trials (default 50): ") or "50")
            
            if get_user_confirmation("Start hyperparameter tuning? (y/n): "):
                best_params = tune_hyperparameters(model_type, selected_data_dir, n_trials)
                print("\nWould you like to train a model with these parameters?")
                if get_user_confirmation("Train model with best parameters? (y/n): "):
                    model_name = f"{model_type}_tuned"
                    epochs = int(input("Enter number of epochs (default 50): ") or "50")
                    trained_model_name = train_model(
                        model_type, model_name, 
                        batch_size=best_params['batch_size'],
                        epochs=epochs,
                        lr=best_params['lr']
                    )
                    print(f"\nModel trained and saved as: {trained_model_name}")
        
        elif choice == '5':
            print("\nProcessed Datasets:")
            processed_dirs = [d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()]
            if not processed_dirs:
                print("No processed datasets found")
            else:
                for d in processed_dirs:
                    print(f"- {d.name}")
                    # Try to load and display config info
                    config_file = d / "preprocessing_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file) as f:
                                config = json.load(f)
                            print(f"   - MTCNN: {config.get('use_mtcnn', 'N/A')}")
                            print(f"   - Face Margin: {config.get('face_margin', 'N/A')}")
                            print(f"   - Image Size: {config.get('final_size', 'N/A')}")
                        except:
                            pass
        
        elif choice == '6':
            print("\nTrained Models:")
            model_dirs = list(CHECKPOINTS_DIR.glob('*'))
            if not model_dirs:
                print("No trained models found")
            else:
                for model_dir in sorted(model_dirs):
                    if model_dir.is_dir():
                        print(f"- {model_dir.name}")
        
        elif choice == '7':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == '__main__':
    main() 