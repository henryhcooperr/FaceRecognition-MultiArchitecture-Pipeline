#!/usr/bin/env python3

from typing import Dict, Optional, List, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class BaselineNet(nn.Module):
    """Basic CNN model I built for initial testing."""
    def __init__(self, num_classes: int = 18, input_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # got tired of calculating this manually every time so I wrote it out
        # TODO: make this more elegant later when I have time
        h, w = input_size
        h, w = h - 2, w - 2  # conv1
        h, w = h // 2, w // 2  # pool
        h, w = h - 2, w - 2  # conv2
        h, w = h // 2, w // 2  # pool
        h, w = h - 2, w - 2  # conv3
        h, w = h // 2, w // 2  # pool
        
        # final feature size
        self.feat_size = 128 * h * w
        
        self.fc1 = nn.Linear(self.feat_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  # raised from 0.3 after seeing overfitting

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.feat_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.feat_size)
        x = F.relu(self.fc1(x))
        return x

class ResNetTransfer(nn.Module):
    """ResNet transfer learning - got much better results with this!"""
    def __init__(self, num_classes: int = 18):
        super().__init__()
        # Fixed deprecation warning after spending 2 hours debugging
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def get_embedding(self, x):
        # this extracts features before the final layer
        modules = list(self.resnet.children())[:-1]
        resnet_feats = nn.Sequential(*modules)
        return resnet_feats(x).squeeze()

class SiameseNet(nn.Module):
    # Siamese network implementation based on that CVPR paper
    # Works well when we don't have much data per person
    def __init__(self):
        super().__init__()
        # Modified for 224x224 input images after testing
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
        
        # had to calculate this manually - don't mess with these values!
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )

    def forward_one(self, x):
        feats = self.conv(x)
        feats = feats.view(feats.size(0), -1)  # Flatten
        feats = self.fc(feats)
        return feats

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

    def get_embedding(self, x):
        return self.forward_one(x)

class AttentionModule(nn.Module):
    """Self-attention for CNN features - added this after reading that ICCV paper"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # learned weight
        
    def forward(self, x):
        batch, C, H, W = x.size()
        
        # Project to q, k, v (terminology from the paper)
        q = self.query(x).view(batch, -1, H*W).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, H*W)
        v = self.value(x).view(batch, -1, H*W)
        
        # Calculate attention map - this is the key insight from the paper
        energy = torch.bmm(q, k)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        
        # Tried without residual first, but works much better with it
        return self.gamma * out + x

class AttentionNet(nn.Module):
    """ResNet with self-attention - my attempt at improving the model."""
    def __init__(self, num_classes=18):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        # This attention module was a pain to debug but works great now
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
    """ArcFace loss implementation.
    
    """
    def __init__(self, in_feats, out_feats, s=30.0, m=0.5):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.s = s  # scale factor
        self.m = m  # margin - played with this value a lot, 0.5 seems best
        self.weight = nn.Parameter(torch.FloatTensor(out_feats, in_feats))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input, label):
        # Normalize features and weights
        x = F.normalize(input)
        w = F.normalize(self.weight)
        
        # Compute cosine similarity
        cos_theta = F.linear(x, w)
        cos_theta = cos_theta.clamp(-1, 1)  # numerical stability
        
        # Add margin to target class
        phi = cos_theta.clone()
        target_mask = torch.zeros_like(cos_theta)
        target_mask.scatter_(1, label.view(-1, 1), 1)
        
        # Apply margin to target class - this is where the magic happens
        phi = torch.where(target_mask.bool(), 
                          torch.cos(torch.acos(cos_theta) + self.m),
                          cos_theta)
        
        # Scale output
        output = phi * self.s
        return output

class ArcFaceNet(nn.Module):
    """Face recognition using ArcFace loss.
    
    """
    def __init__(self, num_classes=18):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.embedding = nn.Linear(512, 512)  # embedding dimension
        self.arcface = ArcMarginProduct(512, num_classes)
        
    def forward(self, x, labels=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        emb = self.embedding(x)
        
        if self.training:
            if labels is None:
                # This kept causing issues during training
                raise ValueError("Labels must be provided during training")
            output = self.arcface(emb, labels)
            return output
        else:
            return emb
            
    def get_embedding(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.embedding(x)

# I tried implementing a transformer block for my presentation
# It's based on "Attention Is All You Need" but modified for vision
# Not sure if it's worth keeping but I'll leave it for now
class TransformerBlock(nn.Module):
    """Simple transformer block for feature refinement."""
    def __init__(self, embed_dim, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # tried ReLU first but GELU works better
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (seq_len, batch, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # residual connection
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + ff_out  # another residual
        x = self.norm2(x)
        return x

class HybridNet(nn.Module):
    """My experimental hybrid CNN-Transformer architecture.
    
    This is my attempt at combining traditional CNNs with transformer attention.
    """
    def __init__(self, num_classes=18):
        super().__init__()
        # CNN Feature Extractor - keeping it simple with ResNet18
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove classification head
        self.features = nn.Sequential(*list(self.cnn.children())[:-2])
        
        # Feature dimensions
        self.fdim = 512
        self.seq_len = 49  # 7x7 feature map flattened
        
        # Position encoding - crucial for transformers to understand spatial relationships
        self.pos_encoding = nn.Parameter(torch.zeros(self.seq_len, 1, self.fdim))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
        # Transformer blocks - tried with more blocks but 1 seems sufficient
        self.transformer = TransformerBlock(self.fdim)
        
        # Output layers
        self.norm = nn.LayerNorm(self.fdim)
        self.fc = nn.Linear(self.fdim, num_classes)
    
    def forward(self, x):
        # Extract CNN features
        feats = self.features(x)  # [batch, 512, 7, 7]
        batch_sz = feats.shape[0]
        
        # Reshape for transformer 
        feats = feats.view(batch_sz, self.fdim, -1)  # [batch, 512, 49]
        feats = feats.permute(2, 0, 1)  # [49, batch, 512]
        
        # Add positional encoding
        feats = feats + self.pos_encoding
        
        # Apply transformer
        feats = self.transformer(feats)
        
        # Global pooling (mean) - tried different pooling methods
        feats = feats.mean(dim=0)  # [batch, 512]
        
        # Normalization and classification
        feats = self.norm(feats)
        feats = self.fc(feats)
        
        return feats
        
    def get_embedding(self, x):
        # Pretty much the same as forward but without final classification
        feats = self.features(x)
        batch_sz = feats.shape[0]
        
        feats = feats.view(batch_sz, self.fdim, -1)
        feats = feats.permute(2, 0, 1)
        
        feats = feats + self.pos_encoding
        feats = self.transformer(feats)
        feats = feats.mean(dim=0)
        feats = self.norm(feats)
        
        return feats

# Tried both contrastive and triplet loss
# Contrastive worked better in my experiments
class ContrastiveLoss(nn.Module):
    """Loss function for Siamese networks"""
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin  # increased from 1.0 after seeing poor separation

    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                         label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss

# Function to get the requested model type
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
        # ArcFace handles loss internally - this confused me at first
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

# Old implementation I tried first - keeping for reference
# def get_model_v1(model_type, num_classes):
#     if model_type == 'baseline':
#         return BaselineNet(num_classes)
#     elif model_type == 'cnn':
#         model = models.resnet18(pretrained=True)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model
#     else:
#         raise ValueError(f"Unknown model: {model_type}") 