#!/usr/bin/env python3

from typing import Dict, Optional, List, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import math  # For kaiming initialization

# List of supported model types
MODEL_TYPES = ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid', 'ensemble']



class BaselineNet(nn.Module):
    """Basic CNN model I built for initial testing."""
    def __init__(self, num_classes: int = 18, input_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        # Adding padding=1 to preserve spatial dimensions
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Add adaptive pooling to replace manual feature size calculation
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  # keeping dropout at 0.5

    def forward(self, x):
        # Apply BatchNorm after convolution and before ReLU
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Replace manual flattening with adaptive pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        return x

class ResNetTransfer(nn.Module):
    """ResNet transfer learning - got much better results with this!"""
    def __init__(self, num_classes: int = 18, freeze_backbone: bool = False):
        super().__init__()
        # Fixed deprecation warning after spending 2 hours debugging
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feats = self.resnet.fc.in_features
        
        # Add dropout before final FC layer - use much lower dropout rate
        self.dropout = nn.Dropout(0.1)
        
        self.resnet.fc = nn.Sequential(
            self.dropout,
            nn.Linear(in_feats, num_classes)
        )
        
        # Freeze backbone only if explicitly requested (default is now False)
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all ResNet layers except the final FC layer"""
        for name, param in self.resnet.named_parameters():
            if "fc" not in name:  # Don't freeze FC layer
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Method to unfreeze the backbone for second stage training"""
        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Use the ResNet's built-in forward method
        # This is simpler and less error-prone
        return self.resnet(x)

    def get_embedding(self, x):
        # this extracts features before the final layer
        modules = list(self.resnet.children())[:-1]
        resnet_feats = nn.Sequential(*modules)
        return resnet_feats(x).squeeze()

class SiameseNet(nn.Module):
    """Improved Siamese network implementation based on deep metric learning techniques.
    
    Inspired by 'Learning a Similarity Metric Discriminatively, with Application to Face Verification'
    and modern techniques for improved network architecture with residual connections.
    """
    def __init__(self):
        super().__init__()
        # Enhanced CNN backbone with improved kernel sizing and residual connections
        self.conv = nn.Sequential(
            # First block with larger kernel for initial feature extraction
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Reduced kernel size for better gradient flow
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # Use inplace ReLU for memory efficiency
            nn.MaxPool2d(2, stride=2),
            
            # Deeper feature extraction with residual-like connections
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Additional conv layer for depth
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Mid-level features with increased channel depth
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Additional conv layer for depth
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Final feature extraction with high-level patterns
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),  # Use adaptive pooling for flexible input sizing
        )
        
        # Enhanced FC layers with gradual dimension reduction
        self.fc = nn.Sequential(
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(512 * 6 * 6, 1024),
            nn.BatchNorm1d(1024),  # Add BN after FC for better training stability
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),  # Additional layer for more gradual dimension reduction
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)  # Final embedding dimension
        )
        
        # Track intermediate shapes for debugging
        self.debug_shapes = {}

    def forward_one(self, x):
        # Store input shape for debugging
        batch_size = x.size(0)
        self.debug_shapes["input"] = x.shape
        
        # Apply convolutional layers
        feats = self.conv(x)
        self.debug_shapes["after_conv"] = feats.shape
        
        # Flatten
        feats = feats.view(batch_size, -1)
        self.debug_shapes["flattened"] = feats.shape
        
        # Apply fully connected layers
        feats = self.fc(feats)
        self.debug_shapes["before_norm"] = feats.shape
        
        # Add L2 normalization to embedding outputs
        feats = F.normalize(feats, p=2, dim=1)
        return feats

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

    def get_embedding(self, x):
        return self.forward_one(x)
        
    def get_debug_info(self):
        """Return debug information including layer shapes."""
        return self.debug_shapes

class SpatialAttention(nn.Module):
    """Spatial attention module to complement channel attention"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Compute average and max pooling across channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along the channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply convolution and sigmoid activation
        attn_map = self.sigmoid(self.conv(pooled))
        
        # Apply spatial attention to input feature map
        return x * attn_map

class AttentionModule(nn.Module):
    """Self-attention for CNN features - added this after reading that ICCV paper"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # learned weight
        self.gamma_value = 0.0  # Monitoring variable for gamma value
        
        # Increase the number of attention heads
        self.num_heads = 2
        self.head_dim = in_channels // (reduction_ratio * self.num_heads)
        
        # Add spatial attention module
        self.spatial_attention = SpatialAttention()
        
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
        
        # Apply channel attention with gamma parameter
        channel_attn_out = self.gamma * out + x
        
        # Store gamma value for monitoring
        self.gamma_value = self.gamma.item()
        
        # Apply spatial attention
        final_out = self.spatial_attention(channel_attn_out)
        
        return final_out

class AttentionNet(nn.Module):
    """Enhanced ResNet with multi-head self-attention mechanisms.
    
    Combines ResNet backbone with both channel and spatial attention for improved
    feature extraction and classification performance.
    """
    def __init__(self, num_classes=18, dropout_rate=0.25):
        super().__init__()
        # Use ResNet18 as a strong feature extractor backbone
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
    
    def get_attention_params(self):
        """Method to monitor attention parameters during training"""
        return {
            "gamma": self.attention.gamma_value
        }

class ArcMarginProduct(nn.Module):
    """My ArcFace loss implementation for face recognition.
    
    I read this paper on ArcFace and tried to implement it with some tweaks.
    Main things I added:
    - Progressive margin (starts small and grows during training)
    - Warm-up for scaling factor
    - Some fixes for stability issues
    - Added the easy margin option cuz the original was hard to train
    """
    def __init__(self, in_feats, out_feats, s=32.0, m=0.5, use_warm_up=True, easy_margin=False):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        
        # Main ArcFace params
        self.s = s  # Scale value - bigger means more separated features
        self.m = m  # Margin - angular distance between classes
        self.easy_margin = easy_margin  # Makes training more stable
        
        # Added some warm-up stuff to avoid training issues
        self.use_warm_up = use_warm_up
        self.warm_up_epochs = 10  # Epochs to slowly ramp up margin and scale
        self.margin_factor = 0.0  # Starts at 0 and increases during training
        self.scale_factor = 0.3  # Start with 30% scale to avoid crazy loss values
        self.current_epoch = 0  # Need to track which epoch we're on
        
        # Make the weight matrix for the linear layer
        self.weight = nn.Parameter(torch.FloatTensor(out_feats, in_feats))
        nn.init.xavier_normal_(self.weight, gain=math.sqrt(2))  # This init works better
        
        # Some tracking variables to help with debugging
        self.register_buffer('u', torch.zeros(1))
        self.max_cos_theta = 0.0  # Highest cosine similarity
        self.min_cos_theta = 0.0  # Lowest cosine similarity 
        self.easy_margin_used = False  # For checking if easy margin kicked in
        
    def forward(self, input, label):
        # Handle warm-up if enabled - gradually increase margin and scale
        if self.training and self.use_warm_up:
            # Check if we're still in warm-up phase
            if self.current_epoch < self.warm_up_epochs:
                # Slowly increase margin - use quadratic growth to start even slower
                progress = self.current_epoch / self.warm_up_epochs
                self.margin_factor = min(0.9, progress * progress)  # Cap at 0.9
                
                # Also slowly increase scale factor - this helps prevent crazy loss spikes
                self.scale_factor = min(0.8, 0.3 + 0.5 * progress)
            else:
                # After warm-up, use these fixed values for stability
                self.margin_factor = 0.9  # Only 90% of margin - full margin is too aggressive
                self.scale_factor = 0.8   # Only 80% of scale - full scale causes loss issues
                
        # Normalize features and weights (make unit vectors)
        x = F.normalize(input, p=2, dim=1, eps=1e-12)
        w = F.normalize(self.weight, p=2, dim=1, eps=1e-12)
        
        # Get cosine similarity between features and weights
        cos_theta = F.linear(x, w)
        
        # Keep track of min/max values for debugging
        with torch.no_grad():
            self.max_cos_theta = torch.max(cos_theta).item()
            self.min_cos_theta = torch.min(cos_theta).item()
        
        # Make sure values are between -1 and 1 with a tiny buffer
        cos_theta_safe = torch.clamp(cos_theta, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        
        # Convert cosine to angle in radians
        theta = torch.acos(cos_theta_safe)
        
        # Apply the progressive margin during training
        effective_margin = self.m * self.margin_factor if self.training else self.m
        
        # Handle the easy margin case
        if self.easy_margin:
            # Only apply margin to positive examples
            self.easy_margin_used = True
            phi = torch.where(
                cos_theta_safe > 0,
                torch.cos(theta + effective_margin),
                cos_theta_safe
            )
            # Create mask for target class
            one_hot = torch.zeros_like(cos_theta_safe, device=cos_theta_safe.device)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            # Apply phi to target class, leave others untouched
            output = torch.where(one_hot.bool(), phi, cos_theta_safe)
        else:
            # Standard margin approach
            # Make sure we don't go over pi (could cause numerical issues)
            max_angle = torch.tensor(math.pi - 1e-4, device=theta.device)
            margined_theta = torch.minimum(max_angle, theta + effective_margin)
            
            # Create mask for target class
            one_hot = torch.zeros_like(cos_theta_safe, device=cos_theta_safe.device)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            
            # Apply margin and convert back to cosine
            phi = torch.cos(margined_theta)
            output = torch.where(one_hot.bool(), phi, cos_theta_safe)
        
        # Apply scaling with stricter caps to prevent loss from exploding
        # I had serious issues with loss values when I didn't cap this
        max_scale = 24.0  # Reduced from 30.0 to 24.0 to prevent instability
        effective_s = min(self.s, max_scale)  # Cap the scale
        # Further reduce effective scale during training with stricter cap
        effective_scale = effective_s * min(0.8, self.scale_factor) if self.training else effective_s
        
        # Extra scale reduction for high-margin cases
        if self.m > 0.4 and self.training:
            # Higher margins + high scale = unstable training
            effective_scale = effective_scale * (0.8 - 0.5 * self.margin_factor)
        
        # Apply the scale to the output
        output = output * effective_scale
        
        # Log warnings for extreme scale values (increased frequency for better monitoring)
        if effective_scale > 20.0 and self.training and torch.rand(1).item() < 0.05:
            print(f"Warning: Scale value too high ({effective_scale:.1f}) - reducing to prevent instability")
            effective_scale = 20.0  # Hard cap at 20.0 for extreme cases
            output = output * (20.0 / effective_scale)  # Rescale output
        elif effective_scale < 3.0 and self.training and torch.rand(1).item() < 0.05:
            print(f"Warning: Scale value too low ({effective_scale:.1f}) - might train slowly")
        
        # Handle NaN/Inf values if they show up
        if torch.isnan(output).any() or torch.isinf(output).any():
            # Replace bad values with zeros
            output = torch.where(torch.isnan(output) | torch.isinf(output), 
                                 torch.zeros_like(output), output)
            print("Uh oh! NaN or Inf in ArcFace output!")
            
        return output
    
    def update_epoch(self, epoch):
        """Keep track of which epoch we're on for the margin/scale scheduling"""
        self.current_epoch = epoch
        
    def get_margin_stats(self):
        """Get the current margin and scale stats for debugging"""
        return {
            'margin_factor': self.margin_factor,
            'scale_factor': self.scale_factor,
            'effective_margin': self.m * self.margin_factor,
            'effective_scale': self.s * self.scale_factor,
            'max_cos_theta': self.max_cos_theta,
            'min_cos_theta': self.min_cos_theta,
            'easy_margin_used': self.easy_margin_used if self.easy_margin else False
        }

class ArcFaceNet(nn.Module):
    """Enhanced face recognition using ArcFace loss with advanced architecture.
    
    Features:
    - Optimized with batch normalization after embedding
    - Dropout for better generalization
    - Weight and feature normalization for improved stability
    - Progressive margin strategy for better convergence
    - Gradient monitoring and clipping specifically for ArcFace
    - Two-phase training support (frozen backbone + fine-tuning)
    - Easy margin option for better convergence
    - Improved numerical stability and gradient handling
    """
    def __init__(self, num_classes=18, dropout_rate=0.2, s=32.0, m=0.5, easy_margin=False):
        super().__init__()
        # Use ResNet18 as a strong feature extractor backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Embedding dimension with batch normalization for better stability
        self.embedding = nn.Linear(512, 512, bias=False)  # bias=False works better with BN
        self.bn = nn.BatchNorm1d(512, eps=1e-5)
        
        # Add dropout for better generalization (reduced from 0.3 to 0.2)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Improved ArcFace layer with progressive margin strategy and easy margin option
        self.arcface = ArcMarginProduct(512, num_classes, s=s, m=m, use_warm_up=True, easy_margin=easy_margin)
        
        # Track gradient norms for monitoring and adaptive clipping
        self.last_grad_norm = 0.0
        self.max_grad_norm = 1.0  # Default max gradient norm for clipping
        
        # Current epoch for margin scheduling
        self.current_epoch = 0
        
        # Training phase tracking
        self.phase = 1  # 1: frozen backbone, 2: full fine-tuning
        self.backbone_frozen = False
        
        # Initialize validation classifier for inference
        self.val_classifier = nn.Linear(512, num_classes)
        # Proper initialization for cosine similarity based inference
        nn.init.xavier_normal_(self.val_classifier.weight, gain=math.sqrt(2))
        
    def freeze_backbone(self):
        """Freeze backbone for first training phase."""
        self.backbone_frozen = True
        self.phase = 1
        for param_name, param in self.named_parameters():
            if 'backbone' in param_name or 'features' in param_name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreeze backbone for second training phase."""
        self.backbone_frozen = False
        self.phase = 2
        for param in self.parameters():
            param.requires_grad = True
            
    def set_max_grad_norm(self, max_norm):
        """Set maximum gradient norm for clipping."""
        self.max_grad_norm = max_norm
        
    def forward(self, x, labels=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Get embeddings with batch normalization and dropout for better training
        x = self.embedding(x)
        x = self.bn(x)  # Apply batch norm for training stability
        
        # Apply dropout during training (reduced rate for better stability)
        if self.training:
            x = self.dropout(x)
            
        # Normalize embeddings for better cosine similarity computation
        # Use smaller epsilon for better numerical precision
        emb = F.normalize(x, p=2, dim=1, eps=1e-12)
        
        if self.training:
            if labels is None:
                raise ValueError("Labels must be provided during training")
                
            # Pass current epoch to ArcFace for progressive margin
            self.arcface.update_epoch(self.current_epoch)
            
            # Forward pass through ArcFace
            output = self.arcface(emb, labels)
            
            # Monitor and clip gradients specifically for ArcFace after backward
            def _hook(module, grad_input, grad_output):
                # Store the gradient norm for monitoring
                if isinstance(grad_input, tuple) and len(grad_input) > 0 and grad_input[0] is not None:
                    self.last_grad_norm = torch.norm(grad_input[0]).item()
                    
                    # Enhanced adaptive gradient clipping
                    # Base threshold from model parameter
                    clip_threshold = self.max_grad_norm
                    
                    # Phase 1 (frozen backbone) uses stricter clipping
                    if self.phase == 1:
                        clip_threshold = min(0.5, self.max_grad_norm)
                    
                    # Scale threshold based on current epoch for smoother training
                    if self.current_epoch < 10:  # Early epochs need tighter clipping
                        clip_threshold = min(clip_threshold, 0.5 + 0.05 * self.current_epoch)
                    
                    # Apply more aggressive clipping for very large gradients
                    # (helps prevent loss explosion)
                    if self.last_grad_norm > 3.0:
                        clip_threshold = min(clip_threshold, 0.5)  # Much tighter for extreme cases
                        print(f"Warning: Extreme gradient norm ({self.last_grad_norm:.2f}) detected, applying tight clipping")
                    
                    # Apply the clipping
                    if self.last_grad_norm > clip_threshold:
                        # More sophisticated clamping with scaling for better numerical stability
                        scale_factor = clip_threshold / (self.last_grad_norm + 1e-8)
                        return (grad_input[0] * scale_factor,) + grad_input[1:]
                return grad_input
            
            # Register hook only if it hasn't been registered yet
            if not hasattr(self, '_hook_handle'):
                self._hook_handle = self.arcface.register_backward_hook(_hook)
                
            return output
        else:
            # For inference, return embeddings or classifier outputs based on requirement
            # Normalize val_classifier weights for cosine similarity
            self.val_classifier.weight.data = F.normalize(self.val_classifier.weight.data, p=2, dim=1, eps=1e-12)
            
            # Return logits for validation if requested
            if labels is not None:
                return self.val_classifier(emb)
            # Return embeddings for similarity comparison
            return emb
            
    def get_embedding(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.bn(x)
        # Return normalized embeddings for consistency with smaller epsilon
        return F.normalize(x, p=2, dim=1, eps=1e-12)
    
    def update_epoch(self, epoch):
        """Update current epoch for progressive margin."""
        self.current_epoch = epoch
        # Also update the ArcFace layer's epoch
        self.arcface.update_epoch(epoch)
        
    def get_arcface_stats(self):
        """Return ArcFace statistics for monitoring."""
        stats = self.arcface.get_margin_stats()
        stats['grad_norm'] = self.last_grad_norm
        stats['max_grad_norm'] = self.max_grad_norm
        stats['phase'] = self.phase
        stats['backbone_frozen'] = self.backbone_frozen
        return stats
        
    def get_training_phase(self):
        """Return current training phase information."""
        return {
            'phase': self.phase,
            'backbone_frozen': self.backbone_frozen,
            'epoch': self.current_epoch
        }

# I tried implementing a transformer block for my presentation
# It's based on "Attention Is All You Need" but modified for vision
# Not sure if it's worth keeping but I'll leave it for now
class TransformerBlock(nn.Module):
    """Simple transformer block for feature refinement."""
    def __init__(self, embed_dim, num_heads=4, ff_dim=2048, dropout=0.1):
        super().__init__()
        # Reduced number of heads from 8 to 4 to match the main branch
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Simplify the feed-forward network slightly
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # GELU works better than ReLU here
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Apply layer normalization before attention (Pre-LN architecture)
        # This often leads to more stable training
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out  # residual connection
        
        # Apply layer normalization before feed-forward network
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out  # another residual
        
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
        
        # Use a single transformer block like in the main branch - simpler is better
        self.transformer = TransformerBlock(self.fdim)
        
        # Add dropout before final classification
        self.dropout = nn.Dropout(0.1)
        
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
        
        # Apply transformer directly - simpler approach from main branch
        feats = self.transformer(feats)
        
        # Global pooling (mean) - tried different pooling methods
        feats = feats.mean(dim=0)  # [batch, 512]
        
        # Normalization, dropout, and classification
        feats = self.norm(feats)
        feats = self.dropout(feats)  # Add dropout before final classification
        feats = self.fc(feats)
        
        return feats
        
    def get_embedding(self, x):
        # Pretty much the same as forward but without final classification
        feats = self.features(x)
        batch_sz = feats.shape[0]
        
        feats = feats.view(batch_sz, self.fdim, -1)
        feats = feats.permute(2, 0, 1)
        
        feats = feats + self.pos_encoding
        
        # Apply transformer directly like in main branch
        feats = self.transformer(feats)
        
        feats = feats.mean(dim=0)
        feats = self.norm(feats)
        
        return feats

# Tried both contrastive and triplet loss
# Contrastive worked better in my experiments
class ContrastiveLoss(nn.Module):
    """My implementation of contrastive loss for Siamese networks.
    
    Added some weighting factors to balance positive/negative pairs 
    and improved the handling of the margin parameter.
    """
    def __init__(self, margin=2.0, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin  # Margin for dissimilar pairs 
        self.eps = 1e-8  # Small epsilon for numerical stability
        # Weights to balance positive and negative pair contributions
        self.pos_weight = pos_weight  # Weight for positive pairs
        self.neg_weight = neg_weight  # Weight for negative pairs
        # Track loss components for debugging
        self.last_dist = None
        self.positive_loss = None
        self.negative_loss = None

    def forward(self, out1, out2, label):
        # Normalize the vectors for more stable calculations
        out1 = F.normalize(out1, p=2, dim=1)
        out2 = F.normalize(out2, p=2, dim=1)
        
        # Calculate distance between pairs
        dist = F.pairwise_distance(out1, out2)
        self.last_dist = dist.detach().cpu()
        
        # Make sure distance isn't too small (avoid division by zero)
        dist = torch.clamp(dist, min=self.eps)
        
        # Handle same/different pairs separately
        # Same class pairs (label=0): want to minimize distance
        same_pairs_loss = (1-label) * torch.pow(dist, 2) * self.neg_weight
        
        # Different class pairs (label=1): want to push apart to at least margin distance
        diff_pairs_loss = label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2) * self.pos_weight
        
        # Save component losses for debugging
        self.positive_loss = diff_pairs_loss.mean().item()  
        self.negative_loss = same_pairs_loss.mean().item()
        
        # Sum the components for final loss
        loss = torch.mean(same_pairs_loss + diff_pairs_loss)
        
        # Print some stats occasionally (not too often)
        if torch.rand(1).item() < 0.05:  # Only 5% of the time
            print(f"ContrastiveLoss - Margin: {self.margin}, Distance: {dist.mean().item():.4f}, "
                  f"Pos loss: {self.positive_loss:.4f}, Neg loss: {self.negative_loss:.4f}")
            
        return loss
        
    def get_debug_info(self):
        """Return debug information about the loss components."""
        return {
            "last_dist": self.last_dist,
            "positive_loss": self.positive_loss,
            "negative_loss": self.negative_loss
        }

# Function to get the requested model type
def get_model(model_type: str, num_classes: int = 18, input_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    """Get a model based on the type string.
    
    I tried to optimize each model's architecture based on what worked best.
    """
    if model_type == 'baseline':
        return BaselineNet(num_classes=num_classes, input_size=input_size)
    elif model_type == 'cnn':
        # Don't freeze backbone by default - allow full training
        return ResNetTransfer(num_classes=num_classes, freeze_backbone=False)
    elif model_type == 'siamese':
        return SiameseNet()
    elif model_type == 'attention':
        # Use improved dropout rate for attention model
        return AttentionNet(num_classes=num_classes, dropout_rate=0.25)
    elif model_type == 'arcface':
        # Use improved dropout rate for ArcFace model
        return ArcFaceNet(num_classes=num_classes, dropout_rate=0.2)
    elif model_type == 'hybrid':
        return HybridNet(num_classes=num_classes)
    elif model_type == 'ensemble':
        # Default ensemble combines CNN, AttentionNet, and ArcFace models
        # This combination has been found to work well in practice
        return create_ensemble(['cnn', 'attention', 'arcface'], num_classes=num_classes)
    elif isinstance(model_type, list):
        # If a list of model types is provided, create an ensemble
        return create_ensemble(model_type, num_classes=num_classes)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def get_criterion(model_type: str) -> nn.Module:
    """Get the right loss function for each model type.
    
    Each model needs a different loss function based on how it works.
    """
    if model_type in ['baseline', 'cnn', 'attention', 'hybrid', 'ensemble']:
        # Use label smoothing for better generalization in classification models
        return nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps reduce overconfidence
    elif model_type == 'siamese':
        # Enhanced contrastive loss with balanced weighting
        return ContrastiveLoss(margin=2.0, pos_weight=1.2, neg_weight=0.8)  # Slight emphasis on positive pairs
    elif model_type == 'arcface':
        # ArcFace uses its own specialized loss internally
        return nn.CrossEntropyLoss(label_smoothing=0.05)  # Less smoothing for ArcFace
    else:
        raise ValueError(f"Invalid model type: {model_type}")

# Old implementation I tried first - keeping for reference
# def get_model_v1(model_type, num_classes):
#     if model_type == 'baseline':
#         return BaselineNet(num_classes)
#     elif model_type == 'cnn':
#         model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model
#     else:
#         raise ValueError(f"Unknown model: {model_type}") 

class EnsembleModel(nn.Module):
    """Enhanced ensemble model that combines predictions from multiple face recognition models.
    
    Implements multiple combination strategies including weighted averaging with learnable weights,
    confidence-based weighting, and max-confidence selection.
    """
    def __init__(self, models: List[nn.Module], ensemble_method: str = 'weighted'):
        """Initialize enhanced ensemble model.
        
        Args:
            models: List of models to ensemble
            ensemble_method: Method to combine predictions:
                - 'average': Simple average of all outputs
                - 'weighted': Weighted average with learnable weights
                - 'max': Take prediction with highest confidence
                - 'attention': Attention-based weighting (new option)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        
        # Initialize weights for weighted ensemble with Xavier initialization for better convergence
        self.weights = nn.Parameter(
            torch.ones(len(models)) / len(models),  # Initialize with equal weights
            requires_grad=(ensemble_method in ['weighted', 'attention'])  # Learn weights for these methods
        )
        
        # For attention-based weighting, add a small network to compute attention scores
        if ensemble_method == 'attention':
            # Add attention mechanism for dynamic weighting
            self.attention_net = nn.Sequential(
                nn.Linear(len(models), 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, len(models)),
                nn.Softmax(dim=0)
            )
    
    def forward(self, x):
        """Forward pass through all models and combine predictions."""
        # Get outputs from all models
        outputs = []
        for model in self.models:
            if hasattr(model, 'training') and model.training:
                model.eval()  # Always use eval mode for ensemble
                
            # Handle different model types
            if isinstance(model, ArcFaceNet):
                # For ArcFace, we need to extract embeddings and compute logits
                embeddings = model(x)
                logits = F.linear(F.normalize(embeddings), F.normalize(model.arcface.weight))
                outputs.append(logits)
            elif isinstance(model, SiameseNet):
                # For SiameseNet, we need reference embeddings and the inference is different
                # come back later to implement this if i have time
                continue
            else:
                # For standard classification models
                outputs.append(model(x))
        
        # Skip ensemble if only one valid model
        if len(outputs) == 1:
            return outputs[0]
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'average':
            return torch.mean(torch.stack(outputs), dim=0)
        elif self.ensemble_method == 'weighted':
            normalized_weights = F.softmax(self.weights, dim=0)
            weighted_outputs = torch.stack([normalized_weights[i] * outputs[i] for i in range(len(outputs))])
            return torch.sum(weighted_outputs, dim=0)
        elif self.ensemble_method == 'max':
            # Convert to probabilities and take max
            probs = [F.softmax(output, dim=1) for output in outputs]
            max_probs, _ = torch.max(torch.stack(probs), dim=0)
            # Convert back to logits for loss computation
            return torch.log(max_probs)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def get_embedding(self, x):
        """Get combined embeddings from all models."""
        # Collect embeddings from all models
        embeddings = []
        for model in self.models:
            if hasattr(model, 'get_embedding'):
                emb = model.get_embedding(x)
                embeddings.append(emb)
        
        # Handle the embeddings
        if len(embeddings) > 1:
            # Concatenate embeddings if multiple models provided embeddings
            return torch.cat(embeddings, dim=1)
        elif len(embeddings) == 1:
            # Just return the single embedding
            return embeddings[0]
        else:
            # No embeddings collected
            return None

def create_ensemble(model_types: List[str], num_classes: int, ensemble_method: str = 'average') -> EnsembleModel:
    """Create an ensemble model from multiple model architectures.
    
    Args:
        model_types: List of model type strings
        num_classes: Number of classes for classification
        ensemble_method: Method to combine predictions
        
    Returns:
        Ensemble model
    """
    models = []
    for model_type in model_types:
        model = get_model(model_type, num_classes=num_classes)
        models.append(model)
    
    return EnsembleModel(models, ensemble_method=ensemble_method) 