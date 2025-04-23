#!/usr/bin/env python3
"""
Data augmentation pipeline using Albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any

def get_training_augmentation() -> A.Compose:
    """
    Get augmentation pipeline for training.
    
    Returns:
        Albumentations Compose object with training augmentations
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=5, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_augmentation() -> A.Compose:
    """
    Get augmentation pipeline for validation.
    
    Returns:
        Albumentations Compose object with validation augmentations
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_test_augmentation() -> A.Compose:
    """
    Get augmentation pipeline for testing.
    
    Returns:
        Albumentations Compose object with test augmentations
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class AugmentationPipeline:
    def __init__(self, phase: str = 'train'):
        """
        Initialize augmentation pipeline.
        
        Args:
            phase: One of 'train', 'val', or 'test'
        """
        if phase == 'train':
            self.transform = get_training_augmentation()
        elif phase == 'val':
            self.transform = get_validation_augmentation()
        elif phase == 'test':
            self.transform = get_test_augmentation()
        else:
            raise ValueError(f"Invalid phase: {phase}")
            
    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Apply augmentations to image.
        
        Args:
            image: Input image as numpy array (RGB format)
            
        Returns:
            Dictionary containing augmented image and metadata
        """
        augmented = self.transform(image=image)
        return augmented 