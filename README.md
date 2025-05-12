# Face Recognition Project - Simplified Version

This is a simplified version of the face recognition project, designed to be more focused on core functionality and easier to understand for presentation purposes.

## Features

- **Interactive Interface**: User-friendly command-line interface for all operations
- **Multiple Model Architectures**: Support for various face recognition architectures
  - Baseline CNN: Simple CNN architecture
  - CNN (ResNet18): Transfer learning with ResNet18
  - Siamese Networks: Facial verification with similarity learning
  - Attention-based models: Self-attention for better feature extraction
  - ArcFace: Advanced angular margin loss for improved performance
  - Hybrid CNN-Transformer: Combined CNN and transformer architecture
  - Ensemble models: Combine multiple models for better performance
- **Preprocessing Pipeline**:
  - Face detection using MTCNN
  - Face alignment and cropping
  - Data augmentation
  - Visualization of preprocessing steps
- **Training Enhancements**:
  - Multiple learning rate schedulers
  - Gradient clipping
  - Early stopping
  - Cross-validation
- **Visualization Tools**:
  - Preprocessing visualization
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
  - Grad-CAM visualizations
  - Person-by-person performance analysis for Siamese networks
- **Hyperparameter Tuning**: Optimize model parameters with Optuna
- **Live Demo**: Real-time face recognition using a webcam

## Datasets

The project uses two datasets:
- **Dataset1**: 36 celebrities, 49 images each
- **Dataset2**: 18 celebrities, 100 images each

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.13+
- CUDA-capable GPU (recommended but not required)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd face-recognition-project
   git checkout simplified
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Download the datasets:
   ```
   python run.py
   ```
   Then select option 1 to process raw data. The system will automatically download the datasets if they don't exist.

## Usage

The project can be run in several modes:

### Interactive Mode

```
python run.py
```

This launches the interactive menu where you can:
- Process raw data
- Visualize preprocessing steps
- Train models
- Evaluate models
- Tune hyperparameters
- Run cross-validation
- List processed datasets and trained models

### Live Demo

```
python run.py demo
```

Launches the live face recognition demo using your webcam.

### Cross-Validation

```
python run.py cv
```

Run k-fold cross-validation on a selected model and dataset.

### Hyperparameter Tuning

```
python run.py hyperopt
```

Optimize hyperparameters for a selected model and dataset using Optuna.

## Interactive Menu Options

1. **Process Raw Data**: Preprocess face images for training
2. **Visualize Preprocessing**: See detailed visualizations of preprocessing steps
3. **Train Model**: Train a face recognition model
4. **Evaluate Model**: Test model performance with comprehensive metrics
5. **Tune Hyperparameters**: Find optimal model parameters
6. **Cross-Validation**: Evaluate model robustness with k-fold validation
7. **List Processed Datasets**: View available preprocessed datasets
8. **List Trained Models**: View trained models
9. **Exit**: Exit the program

## Project Structure

- `src/` - Source code
  - `interactive.py` - Interactive command-line interface
  - `app.py` - Live demo application
  - `base_config.py` - Configuration settings and paths
  - `data_prep.py` - Data preprocessing utilities
  - `data_utils.py` - Dataset classes and utilities
  - `download_dataset.py` - Dataset download functionality
  - `face_models.py` - Model architecture implementations
  - `training.py` - Model training functionality
  - `testing.py` - Model evaluation and visualization
  - `cross_validation.py` - Cross-validation implementation
  - `hyperparameter_tuning.py` - Hyperparameter optimization

- `data/` - Data directory
  - `raw/` - Raw unprocessed datasets
  - `processed/` - Preprocessed datasets

- `outputs/` - Output directory
  - `checkpoints/` - Trained model weights and results
  - `preprocessing_visualization/` - Preprocessing visualizations
  - `plots/` - Performance visualizations

- `run.py` - Main entry point

## Model Details

### Baseline CNN
A simple convolutional neural network with three convolutional layers followed by fully connected layers.

### CNN (ResNet18)
Uses a pre-trained ResNet18 model, fine-tuned for face recognition.

### Siamese Network
Two identical networks that learn similarity between pairs of images. Good for verification tasks.

### Attention Model
Incorporates self-attention mechanisms to focus on important facial features.

### ArcFace
Advanced face recognition model using angular margin loss for improved discrimination.

### Hybrid CNN-Transformer
Combines convolutional layers with transformer blocks for improved performance.

### Ensemble Model
Combines multiple models for improved accuracy through model averaging.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- MTCNN developers for the face detection implementation
- FaceNet-PyTorch for inspiration and utility functions
