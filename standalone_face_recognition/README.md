# Standalone Face Recognition System

This is a standalone version of the face recognition system that automatically handles data organization and processing.

## Project Structure

```
standalone_face_recognition/
├── data/
│   ├── raw/              # Place your raw image data here
│   └── processed/        # Automatically organized train/val/test splits
│       ├── train/
│       ├── val/
│       └── test/
├── models/              # Model architecture definitions
├── outputs/
│   ├── checkpoints/    # Saved model checkpoints
│   └── visualizations/ # Training plots and evaluation visualizations
└── src/
    └── face_recognition_system.py  # Main script
```

## Setup

1. Install the required dependencies:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas tqdm pillow facenet-pytorch albumentations optuna ray
```

2. Organize your raw data:
   - Create a folder for each class in `data/raw/`
   - Place all images for each class in their respective folders
   - Supported image formats: JPG, PNG

Example:
```
data/raw/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   └── ...
└── ...
```

## Usage

Run the main script:
```bash
python src/face_recognition_system.py
```

The system provides an interactive menu with the following options:

1. **Process Raw Data**
   - Configure preprocessing settings
   - Apply face detection and alignment
   - Split data into train/val/test sets

2. **Train Model**
   - Choose model type (baseline/cnn/siamese)
   - Select processed dataset
   - Configure training parameters

3. **Evaluate Model**
   - Comprehensive evaluation metrics
   - Visualizations and analysis
   - Performance benchmarking

4. **List Processed Datasets**
   - View available processed datasets
   - Check preprocessing configurations

5. **List Trained Models**
   - View available trained models
   - Check model versions and types

## Model Types

1. **Baseline (baseline)**
   - Simple CNN architecture
   - Good for initial experiments

2. **CNN Transfer Learning (cnn)**
   - Based on ResNet-18
   - Pre-trained on ImageNet
   - Good for general face recognition

3. **Siamese Network (siamese)**
   - For face verification/comparison
   - Good for few-shot learning

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

1. **Classification Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC AUC
   - PR AUC

2. **Visualizations**
   - Confusion Matrix
   - ROC Curves
   - Precision-Recall Curves
   - Grad-CAM Visualizations
   - t-SNE Embeddings

3. **Performance Metrics**
   - Training/Validation/Test Loss
   - Inference Time
   - Cross-entropy Loss

## Outputs

- **Checkpoints**: Saved in `outputs/checkpoints/`
  - `best_model_{model_type}.pth`: Best performing model
  - `checkpoint_{model_type}.pth`: Latest training checkpoint

- **Visualizations**: Saved in `outputs/visualizations/`
  - Training progress plots
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
  - t-SNE visualization of embeddings
  - Grad-CAM visualizations

## Notes

- The system automatically handles data organization and augmentation
- All paths are relative to the project root
- GPU support is automatic if available
- Models can be resumed from checkpoints
- Visualizations are automatically generated and saved
- Comprehensive error handling and logging 