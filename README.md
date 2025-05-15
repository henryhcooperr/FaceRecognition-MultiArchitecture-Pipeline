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
python run_app.py --browser
```

Launches the interactive Streamlit face recognition demo using your webcam. The enhanced demo includes:

- 🎥 Real-time face detection using MTCNN
- 👤 Face recognition with FaceNet embeddings
- ➕ Easy interface to add new faces to the recognition system
- 🔄 Face tracking between frames for stable recognition
- 📝 Edit and manage known faces
- 📊 Recognition history tracking
- 🔧 Automatic prompt for unrecognized faces

#### Command Line Options

- `--browser`: Automatically open the app in a browser
- `--port`: Specify a port number (default: 8501)

Example:
```bash
python run_app.py --browser --port=8502
```

#### Interface

The application has three main tabs in the sidebar:

1. **Controls**: Start/stop webcam and adjust recognition settings
2. **Faces**: Add, edit and view known faces
3. **History**: View recognition history

#### How to Use

1. Start the webcam using the "Start Webcam" button in the Controls tab
2. When a face is detected but not recognized, you'll be prompted to add it
3. Enter a name for the detected face and click "Save"
4. The face will now be recognized in future frames
5. Use the edit buttons (✏️) next to known faces to update or delete them

### Cross-Validation

```
python run.py cv
```

Run k-fold cross-validation on a selected model and dataset.

### Hyperparameter Tuning

The system includes an advanced hyperparameter tuning module that uses Optuna to optimize model performance. You can run hyperparameter tuning either through the interactive menu or via command line.

### Features

- Support for multiple model types (baseline, CNN, Siamese, Attention, ArcFace, Hybrid, Ensemble)
- Multiple optimizer options (AdamW, RAdam, SGD with momentum)
- Advanced learning rate schedulers (cosine, one-cycle, plateau)
- Trial-0 baseline support for better initial exploration
- Automatic checkpoint management
- Early stopping and pruning of unpromising trials
- Comprehensive hyperparameter search space including:
  - Learning rate
  - Batch size
  - Weight decay
  - Dropout rate
  - Optimizer selection
  - Scheduler selection and parameters
  - Model-specific parameters (e.g., ArcFace margin, attention heads)

### Interactive Menu

To run hyperparameter tuning through the interactive menu:

1. Select option 5 (Hyperparameter Tuning)
2. Choose a model type from the available options
3. Select a processed dataset
4. Configure tuning parameters:
   - Number of trials (default: 20)
   - Timeout in seconds (optional)
   - Whether to use trial-0 baseline
   - Number of checkpoints to keep per trial
5. Start the tuning process
6. Optionally train a model with the best parameters after tuning

### Command Line

You can also run hyperparameter tuning via command line:

```bash
python run.py hyperopt --model-type <model_type> --dataset <dataset_path> [options]
```

Required arguments:
- `--model-type`: Type of model to tune (one of: baseline, cnn, siamese, attention, arcface, hybrid, ensemble)
- `--dataset`: Path to the processed dataset directory

Optional arguments:
- `--n-trials`: Number of trials to run (default: 20)
- `--timeout`: Timeout in seconds (optional)
- `--use-trial0-baseline`: Use trial-0 baseline for first trial
- `--keep-checkpoints`: Number of best checkpoints to keep per trial (default: 1)
- `--train-best`: Train a model with the best parameters after tuning
- `--epochs`: Number of epochs for training with best parameters (default: 50)

Example:
```bash
# Run hyperparameter tuning with trial-0 baseline and train best model
python run.py hyperopt --model-type arcface --dataset data/processed/lfw --use-trial0-baseline --train-best --epochs 100

# Run hyperparameter tuning with timeout
python run.py hyperopt --model-type hybrid --dataset data/processed/celeba --n-trials 50 --timeout 3600
```

### Trial-0 Baseline

The system includes predefined baseline configurations for each model type, which can be used as a starting point for hyperparameter tuning. These baselines are based on common best practices and research findings. To use the trial-0 baseline:

- In interactive menu: Answer 'y' to "Use trial-0 baseline for first trial?"
- In command line: Add the `--use-trial0-baseline` flag

### Results

After hyperparameter tuning completes, the system will:
1. Display the best accuracy achieved
2. Show the best hyperparameters found
3. Optionally train a model with these parameters
4. Save all trial results and checkpoints in the `outputs/hyperopt_runs` directory

The best model checkpoints are saved with their trial number and validation accuracy in the filename.

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


## Acknowledgments

- PyTorch team for the deep learning framework
- MTCNN developers for the face detection implementation
- FaceNet-PyTorch for inspiration and utility functions
