#!/usr/bin/env python3
"""
Transfer learning model using ResNet-18 for face recognition.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional

class ResNetTransfer(nn.Module):
    def __init__(self, num_classes: int = 18, pretrained: bool = True):
        """
        Initialize the ResNet transfer learning model.
        
        Args:
            num_classes: Number of celebrity classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-4]:
            param.requires_grad = False
            
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.resnet(x)
        
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding vector for an input image.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Embedding tensor of shape (batch_size, 512)
        """
        # Get features from ResNet
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get embedding from first layer of new classifier
        x = self.resnet.fc[0](x)
        x = self.resnet.fc[1](x)  # ReLU
        
        return x 