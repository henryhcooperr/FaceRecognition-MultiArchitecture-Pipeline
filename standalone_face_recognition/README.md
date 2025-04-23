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
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas tqdm pillow
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

Run the script:
```bash
python src/face_recognition_system.py
```

The interactive menu provides the following options:

1. **Process Raw Data**
   - Automatically splits your raw data into train/val/test sets
   - Customizable split ratios (default: 70% train, 15% val, 15% test)

2. **Train New Model**
   - Choose model type (baseline/cnn/siamese)
   - Set training parameters (batch size, epochs)
   - Automatically saves checkpoints and visualizations

3. **Evaluate Model**
   - Evaluates model on test set
   - Generates confusion matrix, ROC curve, and PR curve
   - Saves visualizations to outputs/visualizations/

4. **Predict Single Image**
   - Load a trained model
   - Make prediction on a single image
   - Shows predicted class and confidence score

5. **Exit**

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

## Notes

- The system automatically handles data organization and augmentation
- All paths are relative to the project root
- GPU support is automatic if available
- Models can be resumed from checkpoints
- Visualizations are automatically generated and saved 