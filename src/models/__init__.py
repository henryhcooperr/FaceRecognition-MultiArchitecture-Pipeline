"""
Model architectures for face recognition.
"""

from .baseline_backprop import BaselineNet
from .cnn_transfer import ResNetTransfer
from .siamese_net import SiameseNet, ContrastiveLoss

__all__ = [
    'BaselineNet',
    'ResNetTransfer',
    'SiameseNet',
    'ContrastiveLoss'
] 