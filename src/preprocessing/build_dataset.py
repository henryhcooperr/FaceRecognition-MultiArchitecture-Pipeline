#!/usr/bin/env python3
"""
Script to build the processed dataset with face detection and augmentation.
"""

import os
import sys
import logging
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import cv2
import numpy as np

from face_detector import FaceDetector
from augmentations import AugmentationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(base_dir: str) -> None:
    """
    Create necessary directories for processed dataset.
    
    Args:
        base_dir: Base directory for processed data
    """
    dirs = [
        os.path.join(base_dir, 'train'),
        os.path.join(base_dir, 'val'),
        os.path.join(base_dir, 'test')
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def split_dataset(input_dir: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15) -> None:
    """
    Split dataset into train/val/test sets and process images.
    
    Args:
        input_dir: Input directory containing raw images
        output_dir: Output directory for processed dataset
        train_ratio: Ratio of images to use for training
        val_ratio: Ratio of images to use for validation
    """
    # Initialize face detector and augmentation pipelines
    detector = FaceDetector()
    train_aug = AugmentationPipeline(phase='train')
    val_aug = AugmentationPipeline(phase='val')
    test_aug = AugmentationPipeline(phase='test')
    
    # Process each celebrity directory
    for celeb_dir in tqdm(list(Path(input_dir).iterdir()), desc="Processing celebrities"):
        if not celeb_dir.is_dir():
            continue
            
        # Get all images for this celebrity
        images = list(celeb_dir.glob('*.jpg'))
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Process each split
        for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            split_dir = os.path.join(output_dir, split, celeb_dir.name)
            Path(split_dir).mkdir(parents=True, exist_ok=True)
            
            for img_path in tqdm(images, desc=f"Processing {split} images for {celeb_dir.name}"):
                # Detect and align face
                face_img = detector.process_image(str(img_path))
                if face_img is None:
                    continue
                    
                # Convert BGR to RGB for augmentation
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Apply appropriate augmentation
                if split == 'train':
                    augmented = train_aug(face_img_rgb)
                elif split == 'val':
                    augmented = val_aug(face_img_rgb)
                else:
                    augmented = test_aug(face_img_rgb)
                
                # Save processed image
                output_path = os.path.join(split_dir, img_path.name)
                cv2.imwrite(output_path, face_img)

def main():
    """Main function to orchestrate dataset building."""
    # Setup paths
    raw_dir = 'data/raw/celebrity_faces'
    processed_dir = 'data/processed'
    
    # Verify input directory exists
    if not os.path.exists(raw_dir):
        logger.error(f"Raw data directory not found: {raw_dir}")
        sys.exit(1)
        
    # Create output directories
    setup_directories(processed_dir)
    
    # Process and split dataset
    logger.info("Starting dataset processing...")
    split_dataset(raw_dir, processed_dir)
    logger.info("Dataset processing completed!")

if __name__ == "__main__":
    main() 