# Alzheimer's Assistant: Face Recognition System

## Project Overview
This face recognition system was developed as my final project for COMP 380 (Neural Networks). The project aims to help Alzheimer's patients recognize familiar faces, addressing one of the most challenging aspects of the disease - the inability to recognize loved ones. By providing real-time face recognition, the system can serve as a personalized assistant that helps patients maintain connections with family and friends.

## Motivation
As someone whose grandmother was affected by Alzheimer's disease, I've witnessed firsthand how facial recognition deterioration can create emotional distance between patients and their loved ones. This system was designed to bridge that gap by:

- Helping identify family members and caregivers in real-time
- Creating a sense of security through familiar face recognition
- Reducing anxiety associated with meeting "strangers" who are actually loved ones
- Providing a non-intrusive technological solution that preserves dignity

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
   - Create a folder for each person in `data/raw/`
   - Place all images for each person in their respective folders
   - Images should be clear face shots with decent lighting
   - The more varied the poses and lighting, the better the recognition

## Usage

### Interactive Menu
The easiest way to use the system:
```bash
python run.py interactive
```

### Command-Line Interface
For more advanced users:

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

### Live Webcam Demo
Run the Streamlit app for a user-friendly interface:
```bash
streamlit run standalone_face_recognition/src/app.py
```

## Development Journey & Challenges

Throughout developing this project, I faced several key challenges:

### Data Collection and Privacy Concerns
Initially, I struggled with collecting enough face data while respecting privacy. I ended up using a combination of:
- Photos of willing family members (with consent)
- Augmented data to increase the dataset size
- Careful preprocessing to maintain quality despite limited data

### Model Selection Experiments
I implemented multiple model architectures to find the best balance of accuracy and speed:
- Started with a basic CNN, but accuracy wasn't sufficient
- Tried transfer learning with ResNet18, which dramatically improved results
- Implemented a siamese network for better few-shot learning
- Added attention mechanisms which helped with recognition under different lighting conditions
- Experimented with ArcFace and a hybrid CNN-Transformer approach

The CNN transfer learning model offered the best balance of accuracy and speed for the Alzheimer's assistant use case.

### Real-time Performance
Getting the system to work smoothly in real-time was challenging:
- Had to optimize the frame processing pipeline
- Implemented frame skipping to maintain responsiveness
- Struggled with webcam thread management in the Streamlit app
- Discovered that smaller batch sizes work better for training but make the process slower

## Personal Development Changelog

### v0.1 (Week 1-2)
- Initial research on facial recognition techniques
- Set up basic project structure
- Implemented baseline CNN model
- Created data preprocessing pipeline

### v0.2 (Week 3)
- Added ResNet transfer learning approach
- Improved data augmentation techniques
- Fixed major issues with model training
- First working prototype with ~70% accuracy

### v0.3 (Week 4-5)
- Implemented Siamese network for better few-shot learning
- Added attention mechanism after reading about it in a paper
- Created visualization tools for model analysis
- Improved accuracy to ~85%

### v0.4 (Week 6)
- Added interactive menu interface
- Implemented ArcFace for better discrimination between similar faces
- Created comprehensive evaluation metrics
- Bug fixes for face detection edge cases

### v0.5 (Week 7)
- Built Streamlit web interface
- Added real-time face tracking
- Implemented face reference database management
- Memory optimizations for resource-constrained environments 

### v1.0 (Week 8)
- Final performance optimizations
- Comprehensive testing with actual users
- Documentation and code cleanup
- Presentation preparation

## Future Improvements

Areas I'd like to continue working on:
- Add voice output for auditory reinforcement of face recognition
- Implement a mobile app version for wider accessibility
- Create a simplified interface specifically designed for elderly users
- Add a "family mode" that focuses on recognizing specific people
- Improve low-light performance

## Notes

If you encounter any issues with the webcam feed freezing, try:
1. Restarting the application
2. Using a different webcam
3. Checking lighting conditions - the system works best in well-lit environments

Best recognition results are achieved when:
- The face is clearly visible
- The person looks directly at the camera
- Multiple reference images are provided for the same person
- Lighting is consistent between reference and recognition phases

## Acknowledgments

- My grandmother, whose struggle with Alzheimer's inspired this project
- Prof. Thompson for guidance on neural network architecture selection
- My classmates who contributed test data and provided valuable feedback
- Open source face detection libraries that made this project possible 