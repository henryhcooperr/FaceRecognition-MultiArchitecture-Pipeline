#!/usr/bin/env python3
"""
Baseline fully-connected neural network for face recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class BaselineNet(nn.Module):
    def __init__(self, num_classes: int = 18, input_size: int = 224 * 224 * 3):
        """
        Initialize the baseline network.
        
        Args:
            num_classes: Number of celebrity classes
            input_size: Size of input flattened image
        """
        super().__init__()
        
        # Flatten input
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten input
        x = self.flatten(x)
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        
        return x
        
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding vector for an input image.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Embedding tensor of shape (batch_size, 256)
        """
        # Flatten input
        x = self.flatten(x)
        
        # Get embedding from last hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        
        return x 