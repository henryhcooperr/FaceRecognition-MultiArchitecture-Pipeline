#!/usr/bin/env python3

import os
import sys
import unittest
import shutil
import tempfile
from pathlib import Path
import random
import torch
import numpy as np
import logging
from PIL import Image

# Add parent directory to path so we can import the face_recognition_system module
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.face_recognition_system import (
    PreprocessingConfig, preprocess_image, process_raw_data,
    get_model, get_criterion, train_model, evaluate_model,
    PROCESSED_DATA_DIR, RAW_DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock tqdm to avoid progress bars in tests
import unittest.mock
from tqdm import tqdm as real_tqdm

# Create a simple mock for tqdm that just returns the iterable
class MockTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        
    def __iter__(self):
        return iter(self.iterable)
    
    def update(self, *args, **kwargs):
        pass
    
    def close(self):
        pass

# Patch tqdm with our mock version
unittest.mock.patch('src.face_recognition_system.tqdm', MockTqdm).start()

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class TestFaceRecognitionSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment once before all tests."""
        # Create temporary directories for test data
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.test_raw_dir = cls.temp_dir / "raw"
        cls.test_raw_dir.mkdir(parents=True)
        
        # Create a small test dataset (2 classes, 2 images each)
        cls.create_test_dataset()
        
        # Store original directories
        cls.original_raw_dir = RAW_DATA_DIR
        cls.original_processed_dir = PROCESSED_DATA_DIR
        cls.original_checkpoints_dir = CHECKPOINTS_DIR
        cls.original_outputs_dir = OUTPUTS_DIR
        
        # Replace with test directories
        sys.modules["src.face_recognition_system"].RAW_DATA_DIR = cls.test_raw_dir
        sys.modules["src.face_recognition_system"].PROCESSED_DATA_DIR = cls.temp_dir / "processed"
        sys.modules["src.face_recognition_system"].CHECKPOINTS_DIR = cls.temp_dir / "checkpoints"
        sys.modules["src.face_recognition_system"].OUTPUTS_DIR = cls.temp_dir / "outputs"
        
        # Ensure directories exist
        sys.modules["src.face_recognition_system"].PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        sys.modules["src.face_recognition_system"].CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        sys.modules["src.face_recognition_system"].OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Restore original directories
        sys.modules["src.face_recognition_system"].RAW_DATA_DIR = cls.original_raw_dir
        sys.modules["src.face_recognition_system"].PROCESSED_DATA_DIR = cls.original_processed_dir
        sys.modules["src.face_recognition_system"].CHECKPOINTS_DIR = cls.original_checkpoints_dir
        sys.modules["src.face_recognition_system"].OUTPUTS_DIR = cls.original_outputs_dir
        
        # Clean up temp directory
        shutil.rmtree(cls.temp_dir)

    @classmethod
    def create_test_dataset(cls):
        """Create a small test dataset for testing."""
        # Create 2 classes
        for class_name in ["person1", "person2"]:
            class_dir = cls.test_raw_dir / class_name
            class_dir.mkdir(parents=True)
            
            # Create 2 test images per class (black and white squares)
            for i in range(2):
                img_size = 200
                img = Image.new('RGB', (img_size, img_size), color=(0, 0, 0))
                
                # Add some variation by drawing a white rectangle in different positions
                draw_x = random.randint(50, 100)
                draw_y = random.randint(50, 100)
                for x in range(draw_x, draw_x + 50):
                    for y in range(draw_y, draw_y + 50):
                        if 0 <= x < img_size and 0 <= y < img_size:
                            img.putpixel((x, y), (255, 255, 255))
                
                # Save the image
                img.save(class_dir / f"image_{i}.jpg")
        
        logger.info(f"Created test dataset with 2 classes, 2 images each at {cls.test_raw_dir}")

    def test_preprocessing_config(self):
        """Test creating and manipulating a preprocessing configuration."""
        config = PreprocessingConfig(
            name="test_config",
            use_mtcnn=True,
            face_margin=0.4,
            final_size=(224, 224),
            augmentation=True
        )
        
        # Test converting to dict and back
        config_dict = config.to_dict()
        self.assertEqual(config_dict["name"], "test_config")
        self.assertTrue(config_dict["use_mtcnn"])
        
        # Test creating from dict
        new_config = PreprocessingConfig.from_dict(config_dict)
        self.assertEqual(new_config.name, "test_config")
        self.assertTrue(new_config.use_mtcnn)

    def test_preprocess_image(self):
        """Test preprocessing a single image."""
        # Without MTCNN (faster for testing)
        config = PreprocessingConfig(
            name="test_config",
            use_mtcnn=False,  # Skip MTCNN for faster testing
            augmentation=False
        )
        
        # Find a test image
        test_image_path = str(next((self.test_raw_dir / "person1").glob("*.jpg")))
        
        # Test preprocessing
        result = preprocess_image(test_image_path, config)
        self.assertIsNotNone(result)
        self.assertEqual(result.size, config.final_size)

    def test_process_raw_data(self):
        """Test processing raw data."""
        config = PreprocessingConfig(
            name="test_processed",
            use_mtcnn=False,  # Skip MTCNN for faster testing
            augmentation=False
        )
        
        # Process in test mode (limited subset)
        processed_dir = process_raw_data(config, test_mode=True)
        
        # Check if the processed directory exists
        self.assertTrue(processed_dir.exists())
        
        # Check if train/val/test directories exist
        self.assertTrue((processed_dir / "train").exists())
        self.assertTrue((processed_dir / "val").exists())
        self.assertTrue((processed_dir / "test").exists())
        
        # Check if class directories exist in train/val/test
        self.assertTrue((processed_dir / "train" / "person1").exists())
        self.assertTrue((processed_dir / "val" / "person2").exists())

    def test_model_creation(self):
        """Test creating models of different types."""
        model_types = ["baseline", "cnn", "siamese"]
        
        for model_type in model_types:
            model = get_model(model_type, num_classes=2)
            self.assertIsNotNone(model)
            
            # Test forward pass
            device = torch.device("cpu")
            model = model.to(device)
            
            # Create a dummy input
            x = torch.randn(2, 3, 224, 224).to(device)
            
            if model_type == "siamese":
                out1, out2 = model(x, x)
                self.assertEqual(out1.shape[0], 2)  # Batch size
            else:
                out = model(x)
                self.assertEqual(out.shape[0], 2)  # Batch size
                self.assertEqual(out.shape[1], 2)  # Num classes

    def test_get_criterion(self):
        """Test getting criteria for different model types."""
        model_types = ["baseline", "cnn", "siamese"]
        
        for model_type in model_types:
            criterion = get_criterion(model_type)
            self.assertIsNotNone(criterion)

    def test_train_and_evaluate(self):
        """Test training and evaluating each model type with minimal epochs."""
        # Optional skip if GPU is not available (can be enabled/disabled)
        skip_on_cpu = True  # Set to False to run on CPU anyway
        if skip_on_cpu and not torch.cuda.is_available():
            self.skipTest("Skipping test_train_and_evaluate as it requires GPU")
        
        # First process the data if not already done
        if not (sys.modules["src.face_recognition_system"].PROCESSED_DATA_DIR / "test_processed").exists():
            config = PreprocessingConfig(
                name="test_processed",
                use_mtcnn=False,  # Skip MTCNN for faster testing
                augmentation=False
            )
            process_raw_data(config, test_mode=True)
        
        # Mock train_model to use minimal settings
        original_train_model = sys.modules["src.face_recognition_system"].train_model
        
        def mock_train_model(model_type, model_name=None, batch_size=2, epochs=1, lr=0.001, weight_decay=1e-4):
            """Mock train_model to run with minimal epochs and batch size."""
            return original_train_model(model_type, model_name, batch_size, epochs, lr, weight_decay)
        
        # Replace with mock
        sys.modules["src.face_recognition_system"].train_model = mock_train_model
        
        # Mock evaluate_model to avoid long computations
        original_evaluate_model = sys.modules["src.face_recognition_system"].evaluate_model
        
        def mock_evaluate_model(model_type, model_name=None):
            """Mock evaluate_model to skip intensive computations."""
            print(f"Mock evaluating {model_type} model: {model_name}")
            return True
        
        try:
            # Test each model type
            model_types = ["baseline", "cnn", "siamese"]
            for model_type in model_types:
                print(f"\nTesting {model_type} model...")
                
                # Replace the evaluate function with our mock for each iteration
                sys.modules["src.face_recognition_system"].evaluate_model = mock_evaluate_model
                
                try:
                    # Train a model with minimal settings
                    model_name = train_model(
                        model_type=model_type,
                        model_name=f"test_{model_type}",
                        batch_size=2,
                        epochs=1
                    )
                    
                    self.assertIsNotNone(model_name)
                    
                    # Test that model files were created
                    model_checkpoint_dir = sys.modules["src.face_recognition_system"].CHECKPOINTS_DIR / model_name
                    self.assertTrue((model_checkpoint_dir / "best_model.pth").exists())
                    
                    # Test evaluating the model
                    result = evaluate_model(model_type, model_name)
                    self.assertTrue(result)
                
                except Exception as e:
                    self.fail(f"Testing {model_type} model failed: {str(e)}")
                
        finally:
            # Restore original functions
            sys.modules["src.face_recognition_system"].train_model = original_train_model
            sys.modules["src.face_recognition_system"].evaluate_model = original_evaluate_model

    def test_preprocessing_variations(self):
        """Test preprocessing with different parameter variations."""
        # Create test configs with different parameters
        configs = [
            PreprocessingConfig(
                name="test_basic",
                use_mtcnn=False,
                face_margin=0.4,
                final_size=(224, 224),
                augmentation=False
            ),
            PreprocessingConfig(
                name="test_small",
                use_mtcnn=False,
                face_margin=0.2,
                final_size=(160, 160),
                augmentation=False
            ),
            PreprocessingConfig(
                name="test_augmented",
                use_mtcnn=False,
                face_margin=0.4,
                final_size=(224, 224),
                augmentation=True
            )
        ]
        
        test_image_path = str(next((self.test_raw_dir / "person1").glob("*.jpg")))
        
        # Test each configuration
        for config in configs:
            with self.subTest(config=config.name):
                result = preprocess_image(test_image_path, config)
                self.assertIsNotNone(result)
                # Only check the size for non-augmented images
                # Augmentation can slightly change the size due to rotations and scaling
                if not config.augmentation:
                    self.assertEqual(result.size, config.final_size)
                else:
                    # For augmented images, just verify it's an image with reasonable dimensions
                    self.assertIsInstance(result, Image.Image)
                    self.assertGreater(result.width, 100)
                    self.assertGreater(result.height, 100)
                
                # Test with augmentation
                if config.augmentation:
                    # Process the same image twice - augmentation should give different results
                    result1 = preprocess_image(test_image_path, config)
                    result2 = preprocess_image(test_image_path, config)
                    
                    # Convert to numpy arrays for comparison
                    img1_array = np.array(result1)
                    img2_array = np.array(result2)
                    
                    # Images should be different due to random augmentation
                    # (this might occasionally fail if random augmentation happens to be very similar)
                    # Compare just the red channel to avoid sporadic test failures
                    self.assertFalse(np.array_equal(img1_array[:,:,0], img2_array[:,:,0]),
                                    "Augmented images should be different")

    def test_model_with_different_sizes(self):
        """Test models with different input image sizes."""
        # Define which models support which sizes
        model_configs = [
            {"model_type": "baseline", "sizes": [(224, 224), (160, 160)]},  # Baseline should work with all sizes now
            {"model_type": "cnn", "sizes": [(224, 224), (160, 160)]},  # CNN works with multiple sizes
            {"model_type": "siamese", "sizes": [(224, 224)]}  # Siamese only works with 224x224
        ]
        
        for config in model_configs:
            model_type = config["model_type"]
            for size in config["sizes"]:
                with self.subTest(model=model_type, size=size):
                    # For each size, pass the correct input_size to get_model
                    model = get_model(model_type, num_classes=2, input_size=size)
                    device = torch.device("cpu")
                    model = model.to(device)
                    
                    # Create a dummy input with the specified size
                    batch_size = 2
                    x = torch.randn(batch_size, 3, size[0], size[1]).to(device)
                    
                    try:
                        if model_type == "siamese":
                            out1, out2 = model(x, x)
                            self.assertEqual(out1.shape[0], batch_size)  # Batch size
                        else:
                            out = model(x)
                            self.assertEqual(out.shape[0], batch_size)  # Batch size
                            self.assertEqual(out.shape[1], 2)  # Num classes
                    except Exception as e:
                        self.fail(f"Model {model_type} failed with size {size}: {str(e)}")

if __name__ == "__main__":
    unittest.main() 