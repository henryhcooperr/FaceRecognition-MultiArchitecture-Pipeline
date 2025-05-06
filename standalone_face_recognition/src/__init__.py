"""Face Recognition System.

A modular face recognition system with multiple model architectures,
preprocessing options, and visualization tools.
"""

from .base_config import PROJECT_ROOT, DATA_DIR, MODELS_DIR, OUTPUTS_DIR
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
    evaluate_model, predict_image, plot_gradcam_visualization, 
    generate_gradcam
)
from .visualize import (
    plot_tsne_embeddings, plot_attention_maps, plot_embedding_similarity,
    plot_learning_curves, visualize_batch_augmentations
)

__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'MODELS_DIR', 'OUTPUTS_DIR',
    'BaselineNet', 'ResNetTransfer', 'SiameseNet', 'AttentionNet', 
    'ArcFaceNet', 'HybridNet', 'get_model', 'get_criterion',
    'PreprocessingConfig', 'process_raw_data', 'get_preprocessing_config',
    'preprocess_image', 'align_face',
    'train_model', 'tune_hyperparameters', 'SiameseDataset',
    'evaluate_model', 'predict_image', 'plot_gradcam_visualization', 
    'generate_gradcam',
    'plot_tsne_embeddings', 'plot_attention_maps', 'plot_embedding_similarity',
    'plot_learning_curves', 'visualize_batch_augmentations',
] 