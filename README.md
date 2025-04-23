# Celebrity Face Recognition

## Project Overview
This project implements a face recognition system for celebrity identification using deep learning. The system is trained on the [Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset) from Kaggle, which contains 1000 images per celebrity for 18 different celebrities.

### Goals
- Implement robust face detection and alignment
- Train multiple model architectures (baseline, transfer learning, and Siamese networks)
- Evaluate model performance with comprehensive metrics
- Create a real-time webcam demo
- Ensure fair and unbiased model performance

### High-Level Design
The project follows a modular architecture with separate components for:
- Data preprocessing and augmentation
- Model training and evaluation
- Real-time inference
- Performance analysis and visualization

## Installation

### Prerequisites
- Python ≥3.8
- Git
- Kaggle API credentials

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/yourusername/celebrity-face-recognition.git
cd celebrity-face-recognition

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API setup)
kaggle datasets download -d vishesh1412/celebrity-face-image-dataset
unzip celebrity-face-image-dataset.zip -d data/raw/
```

## Directory Structure
```
celebrity-face-recognition/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Preprocessed images
├── src/
│   ├── preprocessing/          # Face detection and augmentation
│   ├── models/                 # Model architectures
│   └── demo/                   # Webcam demo
├── scripts/                    # Utility scripts
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── slides/                     # Presentation slides
├── report/                     # Final report
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Usage

### Data Preprocessing
```bash
# Download and verify dataset
python scripts/download_data.py

# Process and split dataset
python src/preprocessing/build_dataset.py
```

### Training
```bash
# Train baseline model
python src/train.py --model baseline --data-dir data/processed --output-dir models/checkpoints

# Train CNN model
python src/train.py --model cnn --data-dir data/processed --output-dir models/checkpoints

# Train Siamese model
python src/train.py --model siamese --data-dir data/processed --output-dir models/checkpoints
```

### Evaluation
```bash
# Evaluate model
python src/evaluate.py --model <model_type> --checkpoint models/checkpoints/<model_type>_best.pth --data-dir data/processed --output-dir reports
```

### Webcam Demo
```bash
# Run webcam demo
python src/demo/webcam_demo.py --model <model_type> --checkpoint models/checkpoints/<model_type>_best.pth --class-names data/processed/class_names.txt
```

## Roadmap / TODO
- [x] Environment Setup
  - [x] Create requirements.txt
  - [x] Set up virtual environment
  - [x] Install dependencies
- [x] Data Management
  - [x] Download and verify dataset
  - [x] Implement face detection & alignment
  - [x] Build augmentation pipeline
  - [x] Generate processed dataset splits
- [x] Model Development
  - [x] Baseline FC network
  - [x] Transfer-learning CNN
  - [x] Siamese / metric-learning model
- [x] Training & Evaluation
  - [x] Implement train.py
  - [x] Integrate logging & metrics
  - [x] Build evaluate.py with visualizations
- [x] Hyperparameter Tuning
  - [x] Setup and run hyperparameter tuning
- [x] Demo
  - [x] Build real-time webcam demo
- [x] Testing & CI
  - [x] Add unit tests
  - [x] Set up CI pipeline
- [x] Documentation
  - [x] Populate docs/ and slides/
  - [x] Write final report

## Contributors & Roles
- Project Manager & Lead Developer: [Your Name]
  - Overall project coordination
  - Architecture design
  - Code review and quality assurance

## Report
The final report is available in the `report/` directory, containing:
- Introduction and problem statement
- Related work and literature review
- Methodology and implementation details
- Results and analysis
- Discussion and future work
- Conclusion 