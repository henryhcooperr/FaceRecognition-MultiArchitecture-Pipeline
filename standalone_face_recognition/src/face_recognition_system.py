#!/usr/bin/env python3
"""
Standalone Face Recognition System with Automatic Data Management
This file contains all the functionality of the face recognition system in one place,
with automatic data organization and processing.
"""

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
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
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
                 min_face_size: int = 20,
                 thresholds: List[float] = [0.6, 0.7, 0.7],
                 augmentation: bool = True,
                 aug_rotation_range: int = 20,
                 aug_brightness_range: float = 0.2,
                 aug_contrast_range: float = 0.2,
                 aug_scale_range: float = 0.1,
                 horizontal_flip: bool = True):
        """Initialize preprocessing configuration."""
        self.name = name
        self.use_mtcnn = use_mtcnn
        self.face_margin = face_margin
        self.final_size = final_size
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.augmentation = augmentation
        self.aug_rotation_range = aug_rotation_range
        self.aug_brightness_range = aug_brightness_range
        self.aug_contrast_range = aug_contrast_range
        self.aug_scale_range = aug_scale_range
        self.horizontal_flip = horizontal_flip

    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PreprocessingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

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
            # Initialize MTCNN if not already initialized
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

def process_raw_data(config: PreprocessingConfig,
                    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
    """Process raw data with given preprocessing configuration."""
    logger.info(f"Processing raw data with config: {config.name}")
    
    # Create preprocessing-specific directories
    processed_base = PROCESSED_DATA_DIR / config.name
    for split in ["train", "val", "test"]:
        split_dir = processed_base / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
    
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
        shuffle(image_files)
        
        # Calculate split sizes
        n_images = len(image_files)
        n_train = int(n_images * split_ratio[0])
        n_val = int(n_images * split_ratio[1])
        
        # Split indices
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Process and save files
        for split, files in [("train", train_files), 
                           ("val", val_files), 
                           ("test", test_files)]:
            split_class_dir = processed_base / split / class_name
            split_class_dir.mkdir(exist_ok=True)
            
            for file in tqdm(files, desc=f"Processing {split}/{class_name}"):
                processed_face = preprocess_image(str(file), config)
                if processed_face is not None:
                    save_path = split_class_dir / file.name
                    processed_face.save(save_path)
    
    logger.info("Data processing complete!")
    return processed_base

def get_preprocessing_config() -> PreprocessingConfig:
    """Interactive function to get preprocessing configuration from user."""
    print("\nPreprocessing Configuration")
    
    name = input("Enter a name for this preprocessing configuration: ")
    
    use_mtcnn = get_user_confirmation("Use MTCNN for face detection? (y/n): ")
    
    if use_mtcnn:
        face_margin = float(input("Enter face margin (default 0.4): ") or "0.4")
        min_face_size = int(input("Enter minimum face size (default 20): ") or "20")
        thresholds = [0.6, 0.7, 0.7]  # Default MTCNN thresholds
        thresh_input = input("Enter MTCNN thresholds (default 0.6,0.7,0.7): ")
        if thresh_input:
            thresholds = [float(x) for x in thresh_input.split(",")]
    else:
        face_margin = 0.4
        min_face_size = 20
        thresholds = [0.6, 0.7, 0.7]
    
    size_input = input("Enter final image size as width,height (default 224,224): ")
    if size_input:
        final_size = tuple(map(int, size_input.split(",")))
    else:
        final_size = (224, 224)
    
    use_augmentation = get_user_confirmation("Use data augmentation? (y/n): ")
    
    if use_augmentation:
        rotation_range = int(input("Enter rotation range in degrees (default 20): ") or "20")
        brightness_range = float(input("Enter brightness adjustment range (default 0.2): ") or "0.2")
        contrast_range = float(input("Enter contrast adjustment range (default 0.2): ") or "0.2")
        scale_range = float(input("Enter scale adjustment range (default 0.1): ") or "0.1")
        horizontal_flip = get_user_confirmation("Enable horizontal flip? (y/n): ")
    else:
        rotation_range = 20
        brightness_range = 0.2
        contrast_range = 0.2
        scale_range = 0.1
        horizontal_flip = True
    
    return PreprocessingConfig(
        name=name,
        use_mtcnn=use_mtcnn,
        face_margin=face_margin,
        final_size=final_size,
        min_face_size=min_face_size,
        thresholds=thresholds,
        augmentation=use_augmentation,
        aug_rotation_range=rotation_range,
        aug_brightness_range=brightness_range,
        aug_contrast_range=contrast_range,
        aug_scale_range=scale_range,
        horizontal_flip=horizontal_flip
    )

class BaselineNet(nn.Module):
    """Baseline CNN model for face recognition."""
    def __init__(self, num_classes: int = 18):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 26 * 26, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 26 * 26)
        x = F.relu(self.fc1(x))
        return x

class ResNetTransfer(nn.Module):
    """Transfer learning model based on ResNet-18."""
    def __init__(self, num_classes: int = 18):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
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
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        if model_type == 'siamese':
            img1, img2, label = batch
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
            device: torch.device, model_type: str) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            if model_type == 'siamese':
                img1, img2, label = batch
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
                dist = torch.pairwise_distance(out1, out2)
                pred = (dist < 0.5).float()
                correct += (pred == label).sum().item()
                total += label.size(0)
            else:
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

def plot_training_progress(train_losses: List[float], val_losses: List[float],
                         val_accuracies: List[float], output_dir: str, model_name: str):
    """Plot and save training progress."""
    epochs = range(1, len(train_losses) + 1)
    
    # Create model-specific directory
    model_viz_dir = Path(output_dir) / model_name
    model_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(model_viz_dir / 'training_progress.png')
    plt.close()

class SiameseDataset(Dataset):
    """Dataset for Siamese network training."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all valid class directories (only directories, not files)
        self.classes = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        if not self.classes:
            raise ValueError(f"No class directories found in {root_dir}")
        
        self.classes = sorted(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and their labels
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue
                
            for img_path in class_dir.glob("*.[jp][pn][g]"):  # matches .jpg, .png, .jpeg
                if img_path.is_file():  # Make sure it's a file
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        
        if not self.images:
            raise ValueError(f"No valid images found in {root_dir}")
            
        self.labels = np.array(self.labels)
        logger.info(f"Loaded {len(self.images)} images from {len(self.classes)} classes in {root_dir}")
        
    def __getitem__(self, index):
        # Get the first image
        img1_path = self.images[index]
        img1 = Image.open(img1_path).convert('RGB')
        label1 = self.labels[index]
        
        # Randomly decide if we want a same-class pair (1) or different-class pair (0)
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # Get another image from the same class
            same_class_indices = np.where(self.labels == label1)[0]
            same_class_indices = same_class_indices[same_class_indices != index]  # Remove current index
            if len(same_class_indices) == 0:  # If no other images in class, use same image
                img2_path = img1_path
            else:
                img2_path = self.images[np.random.choice(same_class_indices)]
        else:
            # Get an image from a different class
            other_class_indices = np.where(self.labels != label1)[0]
            img2_path = self.images[np.random.choice(other_class_indices)]
        
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.FloatTensor([float(not should_get_same_class)])  # Convert to float for BCE loss
    
    def __len__(self):
        return len(self.images)

def train_model(model_type: str, model_name: Optional[str] = None, batch_size: int = 32, epochs: int = 50,
                lr: float = 0.001, weight_decay: float = 1e-4):
    """Train a face recognition model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # List available processed datasets
    processed_dirs = [d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
    if not processed_dirs:
        raise ValueError("No processed datasets found. Please process raw data first.")
    
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
    
    logger.info(f"Using dataset: {selected_data_dir.name}")
    
    # Generate model name if not provided
    if model_name is None:
        # Find existing models of this type
        existing_models = list(CHECKPOINTS_DIR.glob(f'best_model_{model_type}_*.pth'))
        version = len(existing_models) + 1
        model_name = f"{model_type}_v{version}"
    else:
        # Clean the model name to be filesystem friendly
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
    
    # Load datasets based on model type
    if model_type == 'siamese':
        train_dataset = SiameseDataset(selected_data_dir / "train", transform=transform)
        val_dataset = SiameseDataset(selected_data_dir / "val", transform=transform)
    else:
        train_dataset = datasets.ImageFolder(selected_data_dir / "train", transform=transform)
        val_dataset = datasets.ImageFolder(selected_data_dir / "val", transform=transform)
    
    # For Siamese network, use simpler DataLoader config to avoid multiprocessing issues
    if model_type == 'siamese':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=2, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              num_workers=2, persistent_workers=True)
    
    # Initialize model
    num_classes = len(train_dataset.classes) if model_type != 'siamese' else 2
    model = get_model(model_type, num_classes).to(device)
    
    # Setup training
    criterion = get_criterion(model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    best_val_acc: float = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    try:
        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, model_type)
            val_loss, val_acc = validate(model, val_loader, criterion, device, model_type)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 
                          model_checkpoint_dir / 'best_model.pth')
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, model_checkpoint_dir / 'checkpoint.pth')
            
            # Plot progress
            plot_training_progress(train_losses, val_losses, val_accuracies,
                                 str(VISUALIZATIONS_DIR), model_name)
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
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
    
    # Register hooks
    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_backward_hook(backward_hook)
    
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
    
    # Clean up
    handle1.remove()
    handle2.remove()
    
    # Convert to numpy and return
    return cam.squeeze().cpu().numpy()

def plot_gradcam_visualization(model: nn.Module, dataset: Dataset, 
                             num_samples: int, output_dir: str, model_name: str):
    """Plot Grad-CAM visualizations for sample images."""
    device = next(model.parameters()).device
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
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
    
    for i in range(num_samples):
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
        
        # Convert tensor to numpy for plotting
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        
        # Plot original image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Plot heatmap
        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].set_title('Grad-CAM Heatmap')
        axes[i, 1].axis('off')
        
        # Plot overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.7 * img_np + 0.3 * heatmap/255).clip(0, 1)
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
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
    """Evaluate a trained model with comprehensive visualizations."""
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
    
    # Load test dataset based on model type
    if model_type == 'siamese':
        test_dataset = SiameseDataset(selected_data_dir / "test", transform=transform)
    else:
        test_dataset = datasets.ImageFolder(selected_data_dir / "test", transform=transform)
    
    # Use simple DataLoader configuration to avoid multiprocessing issues
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, pin_memory=True)
    
    # Load model
    num_classes = len(test_dataset.classes) if model_type != 'siamese' else 2
    model = get_model(model_type, num_classes).to(device)
    model.load_state_dict(torch.load(model_checkpoint_dir / 'best_model.pth'))
    model.eval()
    
    # Generate Grad-CAM visualizations
    logger.info("Generating Grad-CAM visualizations...")
    plot_gradcam_visualization(model, test_dataset, num_samples=5, 
                             output_dir=str(VISUALIZATIONS_DIR), model_name=model_name)
    
    # Generate t-SNE embeddings
    logger.info("Generating t-SNE embeddings...")
    plot_tsne_embeddings(model, test_dataset, 
                        str(VISUALIZATIONS_DIR), model_name)
    
    # Evaluate model performance
    logger.info("Evaluating model performance...")
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            if model_type == 'siamese':
                img1, img2, labels = batch
                img1, img2 = img1.to(device), img2.to(device)
                out1, out2 = model(img1, img2)
                dist = F.pairwise_distance(out1, out2)
                pred = (dist < 0.5).float()
                y_true.extend(labels.numpy())
                y_pred.extend(pred.cpu().numpy())
                y_score.extend(dist.cpu().numpy()[:, None])  # Add dimension for consistency
            else:
                images, labels = batch
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                probs = F.softmax(outputs, dim=1)
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_score.extend(probs.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    
    # Plot detailed confusion matrix and metrics
    logger.info("Generating performance visualizations...")
    if model_type != 'siamese':
        plot_confusion_matrix(y_true, y_pred, test_dataset.classes,
                            str(VISUALIZATIONS_DIR), model_name)
        plot_roc_curves(y_true, y_score, test_dataset.classes,
                        str(VISUALIZATIONS_DIR), model_name)
    else:
        # For Siamese network, plot simplified metrics
        accuracy = (y_true == y_pred).mean()
        logger.info(f"Siamese Network Accuracy: {accuracy:.2%}")
        
        # Plot ROC curve for verification
        fpr, tpr, _ = roc_curve(y_true, -y_score.ravel())  # Negative because smaller distance = more similar
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Face Verification')
        plt.legend(loc="lower right")
        plt.savefig(model_viz_dir / 'verification_roc.png')
        plt.close()
    
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

def main():
    """Interactive interface for the face recognition system."""
    # Check GPU availability at startup
    device = check_gpu()
    
    while True:
        print("\nFace Recognition System")
        print("1. Process raw data")
        print("2. Train new model")
        print("3. Evaluate model")
        print("4. Predict single image")
        print("5. Check GPU Status")
        print("6. List trained models")
        print("7. List preprocessing configs")
        print("8. Exit")
        print("\nType 'b' or 'back' at any prompt to return to main menu")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice.lower() in ['b', 'back']:
            continue
        
        try:
            if choice == '1':
                print("\nData Processing")
                if not get_user_confirmation("This will create a new preprocessed dataset. Continue? (y/n): "):
                    continue
                
                # Get preprocessing configuration
                config = get_preprocessing_config()
                
                # Get split ratios
                train_ratio = input("Enter training data ratio (default 0.7): ") or "0.7"
                if train_ratio.lower() in ['b', 'back']:
                    continue
                train_ratio = float(train_ratio)
                
                val_ratio = input("Enter validation data ratio (default 0.15): ") or "0.15"
                if val_ratio.lower() in ['b', 'back']:
                    continue
                val_ratio = float(val_ratio)
                
                test_ratio = input("Enter test data ratio (default 0.15): ") or "0.15"
                if test_ratio.lower() in ['b', 'back']:
                    continue
                test_ratio = float(test_ratio)
                
                if get_user_confirmation("Start processing? (y/n): "):
                    processed_dir = process_raw_data(config, (train_ratio, val_ratio, test_ratio))
                    print(f"\nProcessed data saved in: {processed_dir}")
                
            elif choice == '2':
                print("\nModel Training")
                model_type = input("Enter model type (baseline/cnn/siamese): ")
                if model_type.lower() in ['b', 'back']:
                    continue
                
                model_name = input("Enter model name (optional, press Enter for automatic versioning): ")
                if model_name.lower() in ['b', 'back']:
                    continue
                if not model_name:
                    model_name = None
                
                batch_size = input("Enter batch size (default 32): ") or "32"
                if batch_size.lower() in ['b', 'back']:
                    continue
                batch_size = int(batch_size)
                
                epochs = input("Enter number of epochs (default 50): ") or "50"
                if epochs.lower() in ['b', 'back']:
                    continue
                epochs = int(epochs)
                
                if get_user_confirmation("Start training? (y/n): "):
                    trained_model_name = train_model(model_type, model_name, batch_size, epochs)
                    print(f"\nModel trained and saved as: {trained_model_name}")
                
            elif choice == '3':
                print("\nModel Evaluation")
                model_type = input("Enter model type (baseline/cnn/siamese): ")
                if model_type.lower() in ['b', 'back']:
                    continue
                
                # List available models of this type
                model_dirs = list(CHECKPOINTS_DIR.glob(f'{model_type}_*'))
                if not model_dirs:
                    print(f"No trained models found for type: {model_type}")
                    continue
                
                print("\nAvailable models:")
                for i, model_dir in enumerate(model_dirs, 1):
                    print(f"{i}. {model_dir.name}")
                
                model_choice = input("\nEnter model number (or press Enter for latest): ")
                if model_choice.lower() in ['b', 'back']:
                    continue
                
                model_name = None if not model_choice else model_dirs[int(model_choice)-1].name
                
                if get_user_confirmation("Start evaluation? (y/n): "):
                    evaluate_model(model_type, model_name)
                
            elif choice == '4':
                print("\nSingle Image Prediction")
                model_type = input("Enter model type (baseline/cnn/siamese): ")
                if model_type.lower() in ['b', 'back']:
                    continue
                
                # List available models of this type
                model_dirs = list(CHECKPOINTS_DIR.glob(f'{model_type}_*'))
                if not model_dirs:
                    print(f"No trained models found for type: {model_type}")
                    continue
                
                print("\nAvailable models:")
                for i, model_dir in enumerate(model_dirs, 1):
                    print(f"{i}. {model_dir.name}")
                
                model_choice = input("\nEnter model number (or press Enter for latest): ")
                if model_choice.lower() in ['b', 'back']:
                    continue
                
                model_name = None if not model_choice else model_dirs[int(model_choice)-1].name
                
                image_path = input("Enter image path: ")
                if image_path.lower() in ['b', 'back']:
                    continue
                
                if get_user_confirmation("Make prediction? (y/n): "):
                    predicted_class, confidence = predict_image(model_type, image_path, model_name)
                    print(f"\nPredicted class: {predicted_class}")
                    print(f"Confidence: {confidence:.2%}")
                
            elif choice == '5':
                print("\nChecking GPU Status...")
                device = check_gpu()
                input("\nPress Enter to continue...")
                
            elif choice == '6':
                print("\nTrained Models:")
                model_dirs = list(CHECKPOINTS_DIR.glob('*'))
                if not model_dirs:
                    print("No trained models found")
                else:
                    for model_dir in sorted(model_dirs):
                        if model_dir.is_dir():
                            print(f"- {model_dir.name}")
                input("\nPress Enter to continue...")
                
            elif choice == '7':
                print("\nPreprocessing Configurations:")
                configs = list(PROCESSED_DATA_DIR.glob("*/preprocessing_config.json"))
                if not configs:
                    print("No preprocessing configurations found")
                else:
                    for config_path in configs:
                        with open(config_path) as f:
                            config = json.load(f)
                        print(f"\n- {config['name']}:")
                        print(f"  MTCNN: {config['use_mtcnn']}")
                        print(f"  Face Margin: {config['face_margin']}")
                        print(f"  Image Size: {config['final_size']}")
                        print(f"  Augmentation: {config['augmentation']}")
                input("\nPress Enter to continue...")
                
            elif choice == '8':
                if get_user_confirmation("Are you sure you want to exit? (y/n): "):
                    print("\nExiting...")
                    break
                
            else:
                print("\nInvalid choice. Please try again.")
                
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            print("Returning to main menu...")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print("Returning to main menu...")

if __name__ == '__main__':
    main() 