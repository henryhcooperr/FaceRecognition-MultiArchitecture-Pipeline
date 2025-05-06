# Face Recognition System Tests

This directory contains tests for the face recognition system, including comprehensive tests for the model architectures.

## Running Tests

There are several ways to run the tests for the face recognition system:

### Method 1: Using the run_tests.py script

The simplest way to run all tests:

```bash
cd standalone_face_recognition
python tests/run_tests.py
```

To run specific test classes:

```bash
# Run only tests for the attention model
python tests/run_tests.py tests.test_face_recognition_system.AttentionSpecificTests

# Run only tests for the ArcFace model
python tests/run_tests.py tests.test_face_recognition_system.ArcFaceSpecificTests

# Run only tests for the hybrid model
python tests/run_tests.py tests.test_face_recognition_system.HybridModelSpecificTests
```

### Method 2: Using unittest directly

You can also use Python's unittest module:

```bash
cd standalone_face_recognition
python -m unittest discover -s tests
```

To run specific test classes:

```bash
python -m unittest tests.test_face_recognition_system.AttentionSpecificTests
python -m unittest tests.test_face_recognition_system.ArcFaceSpecificTests
python -m unittest tests.test_face_recognition_system.HybridModelSpecificTests
```

### Method 3: From the main application

You can see information about running tests from the main menu:

```bash
cd standalone_face_recognition
python src/face_recognition_system.py
```

Then select option 7 "Run Tests" from the menu.

## Test Coverage

The tests cover the following model architectures:

1. **BaselineNet**: Simple CNN architecture
2. **ResNetTransfer**: ResNet18 transfer learning implementation
3. **SiameseNet**: Siamese network for face verification
4. **AttentionNet**: ResNet with self-attention mechanism
5. **ArcFaceNet**: Face recognition with ArcFace loss
6. **HybridNet**: CNN-Transformer hybrid architecture

For each model, the tests verify:

- Correct model initialization
- Forward pass with proper output shapes
- Embedding extraction functionality
- Compatibility with visualization tools
- Model-specific parameters and behaviors

## Advanced Model-Specific Tests

### AttentionNet Tests

- Tests different reduction ratios for the attention module
- Verifies attention map generation
- Tests attention visualization hooks

### ArcFaceNet Tests

- Tests different margin values for ArcFace loss
- Tests different scale factors
- Verifies embedding normalization

### HybridNet Tests

- Tests transformer block with different head counts
- Tests transformer block with different feedforward dimensions
- Verifies positional encoding initialization and application

## Creating New Tests

When adding new model architectures, please add corresponding tests in `test_face_recognition_system.py`. Follow the existing test pattern:

1. Basic tests in `FaceRecognitionModelTests` class
2. Model-specific tests in a dedicated test class 
3. Update `test_get_model_function` to include your new model type 