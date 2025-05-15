#!/usr/bin/env python3

import os
import shutil
import json
import logging
import random
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import albumentations as A
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any
from tqdm import tqdm

from .base_config import PROJECT_ROOT, RAW_DATA_DIR, PROC_DATA_DIR, VIZ_DIR, logger, get_user_confirmation

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
    """
    Process an image without visualizing steps, maintains API compatibility.
    
    Args:
        image_path: Path to the image
        config: Preprocessing configuration
        output_dir: Output directory (not used for visualization in this simplified version)
        
    Returns:
        Processed face image or None
    """
    try:
        # Simply process the image using the regular preprocess_image function
        processed_img = preprocess_image(image_path, config)
        
        # Log that visualization is disabled
        logger.info(f"Preprocessing visualization is disabled. Image processed: {image_path}")
        
        return processed_img
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def process_raw_data(raw_data_dir, output_dir, config=None, test_mode=False, max_samples_per_class=None):
    """Process raw image data for face recognition.
    
    Args:
        raw_data_dir: Path to raw data directory
        output_dir: Path to output directory for processed data
        config: PreprocessingConfig object
        test_mode: If True, only process a small subset of data for testing
        max_samples_per_class: Maximum number of samples to use per class
        
    Returns:
        Path: Base output directory containing processed data
    """
    import random
    from tqdm import tqdm
    
    # Look for the raw folders with your specific names
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    
    # Map your folder names to the expected dataset names
    dataset_mapping = {
        "dataset1": "dataset1",  # 36 celebrities, 49 images each
        "dataset2": "dataset2"    # 18 celebrities, 100 images each
    }
    
    # Set up preprocessing config if not provided
    if config is None:
        config = PreprocessingConfig(
            name="default",
            use_mtcnn=True,
            face_margin=0.4,
            final_size=(224, 224),
            augmentation=True
        )
    
    # Set the output subdirectory based on the config name
    base_output_dir = output_dir / config.name
    
    # If max_samples_per_class is set, update the output dir name to reflect it
    if max_samples_per_class is not None:
        base_output_dir = output_dir / f"{config.name}_max{max_samples_per_class}"
        print(f"Limiting to {max_samples_per_class} samples per class")
    
    # Create MTCNN detector if needed
    mtcnn = None
    if config.use_mtcnn:
        from facenet_pytorch import MTCNN
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(
            image_size=config.final_size[0],
            margin=config.face_margin,
            min_face_size=config.min_face_size,
            thresholds=config.thresholds,
            device=device
        )
    
    # Automatically detect and process each dataset
    for source_name, target_name in dataset_mapping.items():
        source_path = raw_data_dir / source_name
        if source_path.exists():
            print(f"Processing {source_name} as {target_name}...")
            
            # Set up output directory for this dataset
            dataset_output_dir = base_output_dir / target_name
            
            # Create train/val/test subdirectories
            train_dir = dataset_output_dir / "train"
            val_dir = dataset_output_dir / "val"
            test_dir = dataset_output_dir / "test"
            
            # Create directories
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Get list of person directories
            person_dirs = [d for d in source_path.iterdir() if d.is_dir()]
            
            # Limit the number of persons in test mode
            if test_mode:
                person_dirs = person_dirs[:3]  # Only process 3 persons for testing
                print(f"Test mode: only processing {len(person_dirs)} persons")
            
            # Process each person's directory
            for person_dir in tqdm(person_dirs, desc=f"Processing {source_name}"):
                person_name = person_dir.name
                
                # Create person directories in train/val/test
                train_person_dir = train_dir / person_name
                val_person_dir = val_dir / person_name
                test_person_dir = test_dir / person_name
                
                train_person_dir.mkdir(exist_ok=True)
                val_person_dir.mkdir(exist_ok=True)
                test_person_dir.mkdir(exist_ok=True)
                
                # Get all image files for this person
                image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
                
                # Skip if no images found
                if not image_files:
                    print(f"No images found for {person_name}, skipping")
                    continue
                
                # Shuffle images for random selection and split
                random.shuffle(image_files)
                
                # Limit the number of images if max_samples_per_class is set
                if max_samples_per_class is not None:
                    image_files = image_files[:max_samples_per_class]
                    print(f"Using {len(image_files)} images for {person_name}")
                
                # Limit the number of images in test mode
                if test_mode:
                    image_files = image_files[:10]  # Only process 10 images per person for testing
                
                # Split into train/val/test
                train_ratio, val_ratio = 0.7, 0.15  # 70% train, 15% val, 15% test
                
                train_size = int(len(image_files) * train_ratio)
                val_size = int(len(image_files) * val_ratio)
                
                train_files = image_files[:train_size]
                val_files = image_files[train_size:train_size + val_size]
                test_files = image_files[train_size + val_size:]
                
                # Process training images
                for img_path in train_files:
                    # Process and save image
                    processed_img = preprocess_image(str(img_path), config)
                    if processed_img:
                        output_path = train_person_dir / img_path.name
                        processed_img.save(str(output_path))
                
                # Process validation images
                for img_path in val_files:
                    processed_img = preprocess_image(str(img_path), config)
                    if processed_img:
                        output_path = val_person_dir / img_path.name
                        processed_img.save(str(output_path))
                
                # Process test images
                for img_path in test_files:
                    processed_img = preprocess_image(str(img_path), config)
                    if processed_img:
                        output_path = test_person_dir / img_path.name
                        processed_img.save(str(output_path))
                
                # Apply augmentation to training set if enabled and there are few images
                if config.augmentation and len(train_files) < 20:
                    # Add 5 augmented images for each original image
                    print(f"Augmenting training data for {person_name}")
                    
                    # Get existing processed images
                    processed_train_files = list(train_person_dir.glob("*.jpg"))
                    
                    # Skip if no processed images
                    if not processed_train_files:
                        continue
                    
                    # Apply augmentation
                    from PIL import Image
                    import albumentations as A
                    
                    # Define augmentation pipeline
                    transform = A.Compose([
                        A.Rotate(limit=config.aug_rotation_range, p=0.7),
                        A.RandomBrightnessContrast(
                            brightness_limit=config.aug_brightness_range,
                            contrast_limit=config.aug_contrast_range,
                            p=0.7
                        ),
                        A.RandomScale(scale_limit=config.aug_scale_range, p=0.5),
                        A.HorizontalFlip(p=0.5 if config.horizontal_flip else 0),
                    ])
                    
                    for idx, img_path in enumerate(processed_train_files):
                        # Only augment a subset of images to avoid too many images
                        if idx >= min(10, len(processed_train_files)):
                            break
                            
                        # Load image
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        
                        # Create 5 augmented versions
                        for aug_idx in range(5):
                            augmented = transform(image=img_array)
                            aug_img = Image.fromarray(augmented['image'])
                            
                            # Save augmented image
                            aug_path = train_person_dir / f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
                            aug_img.save(str(aug_path))
            
            print(f"Finished processing {source_name} as {target_name}")
            
    print("Data preprocessing complete!")
    
    # Return the base output directory so tests can check if it exists
    return base_output_dir

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