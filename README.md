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
*Coming soon as features are implemented*

## Roadmap / TODO
- [ ] Environment Setup
  - [ ] Create requirements.txt
  - [ ] Set up virtual environment
  - [ ] Install dependencies
- [ ] Data Management
  - [ ] Download and verify dataset
  - [ ] Implement face detection & alignment
  - [ ] Build augmentation pipeline
  - [ ] Generate processed dataset splits
- [ ] Model Development
  - [ ] Baseline FC network
  - [ ] Transfer-learning CNN
  - [ ] Siamese / metric-learning model
- [ ] Training & Evaluation
  - [ ] Implement train.py
  - [ ] Integrate logging & metrics
  - [ ] Build evaluate.py with visualizations
- [ ] Hyperparameter Tuning
  - [ ] Setup and run hyperparameter tuning
- [ ] Demo
  - [ ] Build real-time webcam demo
- [ ] Testing & CI
  - [ ] Add unit tests
  - [ ] Set up CI pipeline
- [ ] Documentation
  - [ ] Populate docs/ and slides/
  - [ ] Write final report

## Contributors & Roles
- Project Manager & Lead Developer: [Your Name]
  - Overall project coordination
  - Architecture design
  - Code review and quality assurance

*More roles to be added as the project progresses* 