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
    """Visualize preprocessing steps for a single image."""
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure for visualization
        import matplotlib.pyplot as plt
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
    processed_base = PROC_DATA_DIR / config.name
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
            
        random.shuffle(image_files)
        
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