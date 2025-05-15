"""Face Recognition System.

This started as a class project but evolved into something more personal.
A system to help those with Alzheimer's recognize their loved ones through
computer vision and machine learning.
"""

# Updated imports to match my renamed variables from base_config
from .base_config import PROJECT_ROOT, DATA_DIR, MODELS_DIR, OUT_DIR, PROC_DATA_DIR
from .face_models import (
    BaselineNet, ResNetTransfer, SiameseNet, AttentionNet, ArcFaceNet, HybridNet,
    get_model, get_criterion
)
from .data_prep import (
    PreprocessingConfig, process_raw_data, get_preprocessing_config,
    preprocess_image, align_face
)
from .training import (
    train_model, tune_hyperparameters, SiameseDataset
)
from .testing import (
    evaluate_model, predict_image
)

# This helps with proper importing, easier to add modules this way
__all__ = [
    # Base config
    'PROJECT_ROOT', 'DATA_DIR', 'MODELS_DIR', 'OUT_DIR', 'PROC_DATA_DIR',
    
    # Models - implemented in order of complexity
    'BaselineNet', 'ResNetTransfer', 'SiameseNet', 'AttentionNet', 
    'ArcFaceNet', 'HybridNet', 'get_model', 'get_criterion',
    
    # Data processing
    'PreprocessingConfig', 'process_raw_data', 'get_preprocessing_config',
    'preprocess_image', 'align_face',
    
    # Training tools
    'train_model', 'tune_hyperparameters', 'SiameseDataset',
    
    # Evaluation
    'evaluate_model', 'predict_image'
] 