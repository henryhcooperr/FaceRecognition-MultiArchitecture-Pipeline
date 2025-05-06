# Standalone Face Recognition System

This is a standalone face recognition system that provides both a training pipeline and a live webcam demo for real-time face detection and recognition.

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
├── src/
│   ├── face_recognition_system.py  # Training pipeline
│   └── app.py          # Live webcam demo
├── tests/              # Test suite for the system
│   └── test_face_recognition_system.py
├── run_tests.py        # Script to run all tests
└── face_references/    # Storage for recognized faces in the live demo
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For training:
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

### Training Pipeline
Run the training script:
```bash
python src/face_recognition_system.py
```

### Live Webcam Demo
Run the Streamlit app:
```bash
streamlit run src/app.py
```

### Running Tests
To run the automated test suite:
```bash
python run_tests.py
```

The system includes comprehensive tests for:
- Model creation and forward passes with different input sizes
- Preprocessing with various configurations
- Data processing pipeline
- Training and evaluation (requires GPU)

The test suite uses a small synthetic dataset to ensure fast execution.

## Features

### Training Pipeline
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

### Live Demo
1. **Real-time Face Detection**
   - Multiple face detection
   - Confidence scores
   - Face tracking

2. **Face Recognition**
   - Real-time recognition of known faces
   - Adjustable recognition threshold
   - Distance-based matching

3. **Face Management**
   - Add new faces through webcam
   - View stored reference faces
   - Clear reference database

## Model Types

1. **Baseline (baseline)**
   - Simple CNN architecture
   - Dynamically adjusts to different input image sizes
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

- **Face References**: Saved in `face_references/`
  - Stored face images and embeddings
  - Persistent across sessions

## Notes

- The system automatically handles data organization and augmentation
- All paths are relative to the project root
- GPU support is automatic if available
- Models can be resumed from checkpoints
- Visualizations are automatically generated and saved
- Comprehensive error handling and logging
- Live demo supports multiple face tracking and recognition
- The baseline model now supports variable input sizes for increased flexibility 