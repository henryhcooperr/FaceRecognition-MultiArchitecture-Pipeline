#!/usr/bin/env python3

import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional

class SiameseDataset(Dataset):
    """Dataset for Siamese network training."""
    def __init__(self, root_dir: str, transform=None, test_mode: bool = False, fixed_pairs: bool = False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.test_mode = test_mode  # If in test mode, use fixed pairs
        self.fixed_pairs = fixed_pairs  # If true, generate fixed pairs for testing
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
        
        # Store image pairs for tracking identities
        self.image_pairs = []
        
        # If in test mode, generate fixed pairs instead of random ones
        if test_mode or fixed_pairs:
            self._generate_fixed_test_pairs()
    
    def _generate_fixed_test_pairs(self):
        """Generate fixed pairs for testing to ensure consistent evaluation."""
        # First clear existing pairs
        self.fixed_pairs = []
        
        # Create pairs: both positive (same identity) and negative (different identity)
        for i, (img1_path, label1) in enumerate(zip(self.images, self.labels)):
            # For every image, create one positive and one negative pair
            
            # Find a positive pair (same identity)
            positive_indices = [j for j, lbl in enumerate(self.labels) if lbl == label1 and j != i]
            if positive_indices:
                pos_idx = random.choice(positive_indices)
                self.fixed_pairs.append((i, pos_idx, 1))  # (idx1, idx2, same=1)
            
            # Find a negative pair (different identity)
            negative_indices = [j for j, lbl in enumerate(self.labels) if lbl != label1]
            if negative_indices:
                neg_idx = random.choice(negative_indices)
                self.fixed_pairs.append((i, neg_idx, 0))  # (idx1, idx2, same=0)
        
        # Shuffle the pairs
        random.shuffle(self.fixed_pairs)
    
    def __len__(self):
        if self.test_mode and self.fixed_pairs:
            return len(self.fixed_pairs)
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.test_mode and self.fixed_pairs:
            # Use pre-generated fixed pairs
            idx1, idx2, label = self.fixed_pairs[idx]
            img1_path = self.images[idx1]
            img2_path = self.images[idx2]
            
            # Load and transform images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            # Store image pair paths
            if len(self.image_pairs) <= idx:
                self.image_pairs.append((str(img1_path), str(img2_path)))
            else:
                self.image_pairs[idx] = (str(img1_path), str(img2_path))
            
            return img1, img2, label
        
        # Regular random pairing for training
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
        
        # Store the image pair paths as strings for identity tracking
        if len(self.image_pairs) <= idx:
            self.image_pairs.append((str(img1_path), str(img2_path)))
        else:
            self.image_pairs[idx] = (str(img1_path), str(img2_path))
        
        return img1, img2, label
    
    def get_image_identities(self):
        """Get image identities (person names) for all images in the dataset."""
        identities = []
        for img_path in self.images:
            # Extract identity from parent directory name
            identity = img_path.parent.name
            identities.append(identity)
        return identities 