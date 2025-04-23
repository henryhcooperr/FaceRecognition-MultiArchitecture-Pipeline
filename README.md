# Face Recognition System

A standalone face recognition system with automatic data management, model training, and evaluation capabilities.

## Features

- Automatic data organization and preprocessing
- Multiple model architectures:
  - Baseline CNN
  - Transfer Learning (ResNet)
  - Siamese Network
- Face detection and alignment using MTCNN
- Comprehensive visualizations and evaluation metrics
- Interactive command-line interface

## Project Structure

```
standalone_face_recognition/
├── data/
│   ├── raw/           # Place raw images here
│   └── processed/     # Automatically processed datasets
├── outputs/
│   ├── checkpoints/   # Saved models
│   └── visualizations/# Training and evaluation plots
└── src/
    └── face_recognition_system.py  # Main system file
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place raw face images in `data/raw/[person_name]/`

## Usage

Run the main script:
```bash
python src/face_recognition_system.py
```

The interactive menu provides options for:
1. Processing raw data
2. Training new models
3. Evaluating models
4. Making predictions
5. Checking GPU status

## Model Types

- **Baseline**: Simple CNN architecture
- **CNN**: Transfer learning with ResNet-18
- **Siamese**: Face verification network

## Outputs

- Training progress plots
- Confusion matrices
- ROC curves
- t-SNE visualizations
- Grad-CAM heatmaps

## Note

Model weights, processed data, and visualizations are not included in version control. 