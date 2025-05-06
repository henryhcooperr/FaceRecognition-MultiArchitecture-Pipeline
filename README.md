# Standalone Face Recognition System

This is a standalone face recognition system that provides both a training pipeline and a live webcam demo for real-time face detection and recognition.

## Project Structure

```
standalone_face_recognition/
├── data/
│   ├── raw/              # Place your raw image data here
│   └── processed/        # Automatically organized train/val/test splits
├── models/               # Saved model definitions
├── outputs/
│   ├── checkpoints/      # Saved model checkpoints
│   └── visualizations/   # Training plots and evaluation visualizations
├── src/
│   ├── __init__.py       # Package initialization
│   ├── base_config.py    # Paths and configuration settings
│   ├── face_models.py    # All model definitions
│   ├── data_prep.py      # Face detection and preprocessing
│   ├── training.py       # Training functions
│   ├── testing.py        # Evaluation and metrics
│   ├── visualize.py      # Visualization functions
│   ├── app.py            # Live webcam demo (Streamlit)
│   └── main.py           # Command-line interface
├── tests/                # Test suite for the system
├── run_tests.py          # Script to run all tests
└── run.py                # Main entry point script
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

### Command-Line Interface

The system now uses a unified command-line interface:

```bash
python run.py <command> [options]
```

Available commands:

1. **Preprocess** data:
```bash
python run.py preprocess
```

2. **Train** a model:
```bash
python run.py train --model-type cnn --epochs 50
```

3. **Evaluate** a model:
```bash
python run.py evaluate --model-type cnn
```

4. **Predict** on a single image:
```bash
python run.py predict --model-type cnn --image-path path/to/image.jpg
```

5. **Tune** hyperparameters:
```bash
python run.py tune --model-type cnn --n-trials 50
```

6. Check GPU availability:
```bash
python run.py check-gpu
```

7. List available trained models:
```bash
python run.py list-models
```

For help on any command:
```bash
python run.py <command> --help
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

## System Architecture

The system is organized in a modular structure:

- **base_config.py**: Contains paths, constants, and basic utility functions
- **face_models.py**: All model definitions (BaselineNet, ResNetTransfer, SiameseNet, etc.)
- **data_prep.py**: Face detection, alignment, preprocessing, and augmentation
- **training.py**: Dataset classes and training functions
- **testing.py**: Evaluation and metrics calculations
- **visualize.py**: Visualization functions for model outputs and embeddings
- **main.py**: Command-line interface with all available commands
- **app.py**: Streamlit-based live webcam interface

This modular design makes the code easier to understand, maintain, and extend.

## Features

### Training Pipeline
1. **Process Raw Data**
   - Configure preprocessing settings
   - Apply face detection and alignment
   - Split data into train/val/test sets

2. **Train Model**
   - Choose model type (baseline/cnn/siamese/attention/arcface/hybrid)
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

4. **Attention Network (attention)**
   - ResNet with self-attention mechanism
   - Better focus on discriminative facial regions

5. **ArcFace Network (arcface)**
   - Uses Angular margin loss for better discrimination
   - State-of-the-art approach for face recognition

6. **Hybrid Network (hybrid)**
   - CNN-Transformer hybrid architecture
   - Combines local and global feature representation

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
   - Attention Maps (for attention models)

3. **Performance Metrics**
   - Training/Validation/Test Loss
   - Inference Time
   - Cross-entropy Loss

## Outputs

- **Checkpoints**: Saved in `outputs/checkpoints/`
  - `best_model.pth`: Best performing model

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
- Live demo supports multiple face tracking and recognition
- Models support different input sizes for increased flexibility 