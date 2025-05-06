#!/usr/bin/env python3

from typing import Dict, Optional, List, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

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

class AttentionModule(nn.Module):
    """Self-attention module for face recognition."""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Project to query, key, value
        proj_query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H*W)
        proj_value = self.value(x).view(batch_size, -1, H*W)
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection
        return self.gamma * out + x

class AttentionNet(nn.Module):
    """ResNet with self-attention mechanism for face recognition."""
    def __init__(self, num_classes=18):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.attention = AttentionModule(512)  # ResNet18 has 512 channels in last layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def get_embedding(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)

class ArcMarginProduct(nn.Module):
    """ArcFace loss implementation."""
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # Scale factor
        self.m = m  # Margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input, label):
        # Normalize features and weights
        x = F.normalize(input)
        w = F.normalize(self.weight)
        
        # Compute cosine similarity
        cos_theta = F.linear(x, w)
        cos_theta = cos_theta.clamp(-1, 1)
        
        # Add margin to target class
        phi = cos_theta.clone()
        target_mask = torch.zeros_like(cos_theta)
        target_mask.scatter_(1, label.view(-1, 1), 1)
        
        # Apply margin to target class
        phi = torch.where(target_mask.bool(), 
                          torch.cos(torch.acos(cos_theta) + self.m),
                          cos_theta)
        
        # Scale output
        output = phi * self.s
        return output

class ArcFaceNet(nn.Module):
    """Face recognition model using ArcFace loss."""
    def __init__(self, num_classes=18):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.embedding = nn.Linear(512, 512)  # Embedding dimension
        self.arcface = ArcMarginProduct(512, num_classes)
        
    def forward(self, x, labels=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)
        
        if self.training:
            if labels is None:
                raise ValueError("Labels must be provided during training")
            output = self.arcface(embedding, labels)
            return output
        else:
            return embedding
            
    def get_embedding(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.embedding(x)

class TransformerBlock(nn.Module):
    """Simple transformer block for feature refinement."""
    def __init__(self, embed_dim, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (seq_len, batch, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm2(x)
        return x

class HybridNet(nn.Module):
    """Hybrid CNN-Transformer architecture for face recognition."""
    def __init__(self, num_classes=18):
        super().__init__()
        # CNN Feature Extractor
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove classification head
        self.features = nn.Sequential(*list(self.cnn.children())[:-2])
        
        # Feature dimensions
        self.feature_dim = 512
        self.sequence_length = 49  # 7x7 feature map flattened
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.zeros(self.sequence_length, 1, self.feature_dim))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
        # Transformer blocks
        self.transformer = TransformerBlock(self.feature_dim)
        
        # Output layers
        self.norm = nn.LayerNorm(self.feature_dim)
        self.fc = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.features(x)  # [batch, 512, 7, 7]
        batch_size = x.shape[0]
        
        # Reshape for transformer
        x = x.view(batch_size, self.feature_dim, -1)  # [batch, 512, 49]
        x = x.permute(2, 0, 1)  # [49, batch, 512]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global pooling (use first token)
        x = x.mean(dim=0)  # [batch, 512]
        
        # Normalization and classification
        x = self.norm(x)
        x = self.fc(x)
        
        return x
        
    def get_embedding(self, x):
        # CNN feature extraction
        x = self.features(x)  # [batch, 512, 7, 7]
        batch_size = x.shape[0]
        
        # Reshape for transformer
        x = x.view(batch_size, self.feature_dim, -1)  # [batch, 512, 49]
        x = x.permute(2, 0, 1)  # [49, batch, 512]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global pooling (use first token)
        x = x.mean(dim=0)  # [batch, 512]
        
        # Normalization
        x = self.norm(x)
        
        return x

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
    elif model_type == 'attention':
        return AttentionNet(num_classes=num_classes)
    elif model_type == 'arcface':
        return ArcFaceNet(num_classes=num_classes)
    elif model_type == 'hybrid':
        return HybridNet(num_classes=num_classes)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def get_criterion(model_type: str) -> nn.Module:
    if model_type in ['baseline', 'cnn', 'attention', 'hybrid']:
        return nn.CrossEntropyLoss()
    elif model_type == 'siamese':
        return ContrastiveLoss()
    elif model_type == 'arcface':
        # ArcFace models handle loss internally
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid model type: {model_type}") 