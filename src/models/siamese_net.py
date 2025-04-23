#!/usr/bin/env python3
"""
Siamese network for metric learning in face recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize the Siamese network.
        
        Args:
            embedding_dim: Dimension of the embedding space
        """
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, embedding_dim)
        
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single input.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 512 * 14 * 14)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair of inputs.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Tuple of (embedding1, embedding2)
        """
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2
        
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding vector for an input image.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.forward_one(x)
        
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 2.0):
        """
        Initialize the contrastive loss.
        
        Args:
            margin: Margin for the contrastive loss
        """
        super().__init__()
        self.margin = margin
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss.
        
        Args:
            x1: First embedding tensor
            x2: Second embedding tensor
            y: Labels (1 for same person, 0 for different)
            
        Returns:
            Loss tensor
        """
        # Compute Euclidean distance
        dist = F.pairwise_distance(x1, x2)
        
        # Compute loss
        loss = y * torch.pow(dist, 2) + (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        
        return loss.mean() 