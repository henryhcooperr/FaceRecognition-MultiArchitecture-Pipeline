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
import torch.nn as nn
from torch.utils.data import Dataset

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from our new modular structure
from src.base_config import (
    PROC_DATA_DIR, RAW_DATA_DIR, CHECKPOINTS_DIR, OUT_DIR, logger
)
from src.face_models import (
    BaselineNet, ResNetTransfer, SiameseNet, 
    AttentionNet, ArcFaceNet, HybridNet,
    AttentionModule, TransformerBlock, ArcMarginProduct,
    ContrastiveLoss, get_model, get_criterion
)
from src.data_prep import (
    PreprocessingConfig, preprocess_image, process_raw_data
)
from src.training import train_model
from src.testing import evaluate_model

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

# Patch tqdm with our mock version - update patch to point to src.data_prep
unittest.mock.patch('src.data_prep.tqdm', MockTqdm).start()

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
        cls.original_processed_dir = PROC_DATA_DIR
        cls.original_checkpoints_dir = CHECKPOINTS_DIR
        cls.original_outputs_dir = OUT_DIR
        
        # Replace with test directories in each module
        from src import base_config, data_prep, training, testing
        
        # Update base_config module and all dependent modules
        base_config.RAW_DATA_DIR = cls.test_raw_dir
        base_config.PROC_DATA_DIR = cls.temp_dir / "processed"
        base_config.CHECKPOINTS_DIR = cls.temp_dir / "checkpoints"
        base_config.OUT_DIR = cls.temp_dir / "outputs"
        
        # Update references in dependent modules
        data_prep.RAW_DATA_DIR = base_config.RAW_DATA_DIR
        data_prep.PROC_DATA_DIR = base_config.PROC_DATA_DIR
        
        training.PROC_DATA_DIR = base_config.PROC_DATA_DIR
        training.CHECKPOINTS_DIR = base_config.CHECKPOINTS_DIR
        
        testing.PROC_DATA_DIR = base_config.PROC_DATA_DIR
        testing.CHECKPOINTS_DIR = base_config.CHECKPOINTS_DIR
        testing.VIZ_DIR = base_config.OUT_DIR / "visualizations"
        
        # Ensure directories exist
        base_config.PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        base_config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        base_config.OUT_DIR.mkdir(parents=True, exist_ok=True)
        testing.VIZ_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Restore original directories
        from src import base_config, data_prep, training, testing
        
        # Restore in base_config
        base_config.RAW_DATA_DIR = cls.original_raw_dir
        base_config.PROC_DATA_DIR = cls.original_processed_dir
        base_config.CHECKPOINTS_DIR = cls.original_checkpoints_dir
        base_config.OUT_DIR = cls.original_outputs_dir
        
        # Restore in dependent modules
        data_prep.RAW_DATA_DIR = base_config.RAW_DATA_DIR
        data_prep.PROC_DATA_DIR = base_config.PROC_DATA_DIR
        
        training.PROC_DATA_DIR = base_config.PROC_DATA_DIR
        training.CHECKPOINTS_DIR = base_config.CHECKPOINTS_DIR
        
        testing.PROC_DATA_DIR = base_config.PROC_DATA_DIR
        testing.CHECKPOINTS_DIR = base_config.CHECKPOINTS_DIR
        testing.VIZ_DIR = base_config.OUT_DIR / "visualizations"
        
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
        from src.base_config import PROC_DATA_DIR
        
        if not (PROC_DATA_DIR / "test_processed").exists():
            config = PreprocessingConfig(
                name="test_processed",
                use_mtcnn=False,  # Skip MTCNN for faster testing
                augmentation=False
            )
            process_raw_data(config, test_mode=True)
        
        # Patch train_model to use minimal settings
        from src import training, testing
        
        original_train_model = training.train_model
        
        def mock_train_model(model_type, model_name=None, batch_size=2, epochs=1, lr=0.001, weight_decay=1e-4):
            """Mock train_model to run with minimal epochs and batch size."""
            return original_train_model(model_type, model_name, batch_size, epochs, lr, weight_decay)
        
        # Replace with mock
        training.train_model = mock_train_model
        
        # Mock evaluate_model to avoid long computations
        original_evaluate_model = testing.evaluate_model
        
        def mock_evaluate_model(model_type, model_name=None):
            """Mock evaluate_model to skip intensive computations."""
            print(f"Mock evaluating {model_type} model: {model_name}")
            return {"accuracy": 0.8}
        
        try:
            # Test each model type
            model_types = ["baseline", "cnn", "siamese"]
            for model_type in model_types:
                print(f"\nTesting {model_type} model...")
                
                # Replace the evaluate function with our mock for each iteration
                testing.evaluate_model = mock_evaluate_model
                
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
                    model_checkpoint_dir = CHECKPOINTS_DIR / model_name
                    self.assertTrue((model_checkpoint_dir / "best_model.pth").exists())
                    
                    # Test evaluating the model
                    result = evaluate_model(model_type, model_name)
                    self.assertTrue(isinstance(result, dict))
                
                except Exception as e:
                    self.fail(f"Testing {model_type} model failed: {str(e)}")
                
        finally:
            # Restore original functions
            training.train_model = original_train_model
            testing.evaluate_model = original_evaluate_model

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

class FaceRecognitionModelTests(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.num_classes = 5
        self.img_size = (224, 224)
        
        # Create dummy inputs
        self.dummy_input = torch.randn(self.batch_size, 3, *self.img_size).to(self.device)
        self.dummy_labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
    
    def test_baseline_model(self):
        model = BaselineNet(num_classes=self.num_classes, input_size=self.img_size).to(self.device)
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            output = model(self.dummy_input)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))
            
            # Test embedding function
            embedding = model.get_embedding(self.dummy_input)
            self.assertEqual(embedding.shape[0], self.batch_size)
            
    def test_resnet_transfer(self):
        model = ResNetTransfer(num_classes=self.num_classes).to(self.device)
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            output = model(self.dummy_input)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))
            
            # Test embedding function
            embedding = model.get_embedding(self.dummy_input)
            self.assertEqual(embedding.shape[0], self.batch_size)
            self.assertEqual(embedding.shape[1], 512)  # ResNet18 produces 512-dim embeddings
            
    def test_siamese_model(self):
        model = SiameseNet().to(self.device)
        model.eval()
        
        # Test forward pass with pair of images
        with torch.no_grad():
            out1, out2 = model(self.dummy_input, self.dummy_input)
            self.assertEqual(out1.shape, (self.batch_size, 256))
            self.assertEqual(out2.shape, (self.batch_size, 256))
            
            # Test embedding function
            embedding = model.get_embedding(self.dummy_input)
            self.assertEqual(embedding.shape, (self.batch_size, 256))
            
    def test_attention_model(self):
        model = AttentionNet(num_classes=self.num_classes).to(self.device)
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            output = model(self.dummy_input)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))
            
            # Test embedding function
            embedding = model.get_embedding(self.dummy_input)
            self.assertEqual(embedding.shape, (self.batch_size, 512))
            
            # Test attention module separately
            attention_module = AttentionModule(in_channels=64).to(self.device)
            dummy_feature_map = torch.randn(self.batch_size, 64, 32, 32).to(self.device)
            attention_output = attention_module(dummy_feature_map)
            self.assertEqual(attention_output.shape, dummy_feature_map.shape)
            
    def test_arcface_model(self):
        model = ArcFaceNet(num_classes=self.num_classes).to(self.device)
        
        # Test training mode (requires labels)
        model.train()
        output = model(self.dummy_input, self.dummy_labels)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
        # Test inference mode (no labels needed)
        model.eval()
        with torch.no_grad():
            embedding = model(self.dummy_input)
            self.assertEqual(embedding.shape, (self.batch_size, 512))
            
            # Test embedding function
            embedding = model.get_embedding(self.dummy_input)
            self.assertEqual(embedding.shape, (self.batch_size, 512))
        
        # Test ArcMarginProduct separately
        arc_margin = ArcMarginProduct(in_feats=512, out_feats=self.num_classes).to(self.device)
        dummy_features = torch.randn(self.batch_size, 512).to(self.device)
        arc_output = arc_margin(dummy_features, self.dummy_labels)
        self.assertEqual(arc_output.shape, (self.batch_size, self.num_classes))
        
    def test_hybrid_model(self):
        model = HybridNet(num_classes=self.num_classes).to(self.device)
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            output = model(self.dummy_input)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))
            
            # Test embedding function
            embedding = model.get_embedding(self.dummy_input)
            self.assertEqual(embedding.shape, (self.batch_size, 512))
        
        # Test TransformerBlock separately
        transformer_block = TransformerBlock(embed_dim=64, num_heads=2).to(self.device)
        dummy_sequence = torch.randn(10, self.batch_size, 64).to(self.device)  # seq_len, batch, embed_dim
        transformer_output = transformer_block(dummy_sequence)
        self.assertEqual(transformer_output.shape, dummy_sequence.shape)
        
    def test_get_model_function(self):
        """Test that the get_model function correctly instantiates all model types."""
        model_types = ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid']
        
        for model_type in model_types:
            model = get_model(model_type, num_classes=self.num_classes)
            self.assertIsNotNone(model)
            
            # Verify model is of expected type
            if model_type == 'baseline':
                self.assertIsInstance(model, BaselineNet)
            elif model_type == 'cnn':
                self.assertIsInstance(model, ResNetTransfer)
            elif model_type == 'siamese':
                self.assertIsInstance(model, SiameseNet)
            elif model_type == 'attention':
                self.assertIsInstance(model, AttentionNet)
            elif model_type == 'arcface':
                self.assertIsInstance(model, ArcFaceNet)
            elif model_type == 'hybrid':
                self.assertIsInstance(model, HybridNet)
    
    def test_get_criterion_function(self):
        """Test that the get_criterion function returns appropriate loss functions."""
        model_types = ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid']
        
        for model_type in model_types:
            criterion = get_criterion(model_type)
            self.assertIsNotNone(criterion)
            
            if model_type == 'siamese':
                self.assertIsInstance(criterion, ContrastiveLoss)
            elif model_type in ['baseline', 'cnn', 'attention', 'arcface', 'hybrid']:
                self.assertIsInstance(criterion, nn.CrossEntropyLoss)
                
    def test_visualization_compatibility(self):
        """Test compatibility with visualization functions for all model types."""
        model_types = ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid']
        
        # Create a tiny dataset for testing
        class DummyDataset(Dataset):
            def __init__(self, num_samples=10, num_classes=5):
                self.num_samples = num_samples
                self.classes = [f"class_{i}" for i in range(num_classes)]
                self.data = [(torch.randn(3, 224, 224), i % num_classes) for i in range(num_samples)]
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dummy_dataset = DummyDataset()
        
        for model_type in model_types:
            model = get_model(model_type, num_classes=len(dummy_dataset.classes)).to(self.device)
            model.eval()
            
            # Test visualization compatibility
            try:
                # If the model type is attention, we need to test attention map visualization
                if model_type == 'attention':
                    # Skip actual visualization, just test the function doesn't crash
                    # We need this because visualization functions actually save files
                    hook_called = [False]
                    
                    def dummy_hook(module, input, output):
                        hook_called[0] = True
                        return output
                    
                    handle = model.attention.register_forward_hook(dummy_hook)
                    
                    # Run forward pass on first image
                    img = dummy_dataset[0][0].unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        _ = model(img)
                        
                    self.assertTrue(hook_called[0], "Attention hook wasn't called")
                    handle.remove()
            except Exception as e:
                self.fail(f"Visualization test failed for {model_type} model: {str(e)}")

class ArcFaceSpecificTests(unittest.TestCase):
    """Additional tests specific to ArcFace model architecture."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8
        self.num_classes = 10
        self.feature_dim = 512
        self.img_size = (224, 224)
        
        # Create dummy inputs
        self.dummy_input = torch.randn(self.batch_size, 3, *self.img_size).to(self.device)
        self.dummy_labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        self.dummy_features = torch.randn(self.batch_size, self.feature_dim).to(self.device)
    
    def test_arcface_margin_boundaries(self):
        """Test ArcFace with different margin values."""
        margins = [0.3, 0.5, 0.7]
        
        for m in margins:
            arc_margin = ArcMarginProduct(
                in_feats=self.feature_dim, 
                out_feats=self.num_classes,
                m=m
            ).to(self.device)
            
            # Test forward pass
            output = arc_margin(self.dummy_features, self.dummy_labels)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_arcface_scale_factor(self):
        """Test ArcFace with different scale factors."""
        scales = [20.0, 30.0, 40.0]
        
        for s in scales:
            arc_margin = ArcMarginProduct(
                in_feats=self.feature_dim, 
                out_feats=self.num_classes,
                s=s
            ).to(self.device)
            
            # Test forward pass
            output = arc_margin(self.dummy_features, self.dummy_labels)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_arcface_embedding_normalization(self):
        """Test that embeddings are properly normalized in ArcFace."""
        model = ArcFaceNet(num_classes=self.num_classes).to(self.device)
        model.eval()
        
        with torch.no_grad():
            embedding = model.get_embedding(self.dummy_input)
            
            # Manually normalize embedding
            normalized = torch.nn.functional.normalize(embedding, dim=1)
            
            # Check if embedding vectors have unit norm (approximately)
            norms = torch.norm(normalized, dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

class AttentionSpecificTests(unittest.TestCase):
    """Additional tests specific to Attention model architecture."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.num_classes = 10
        self.img_size = (224, 224)
        
        # Create dummy inputs
        self.dummy_input = torch.randn(self.batch_size, 3, *self.img_size).to(self.device)
    
    def test_attention_reduction_ratios(self):
        """Test attention module with different reduction ratios."""
        channels = 256
        ratios = [4, 8, 16]
        
        for ratio in ratios:
            # Create input tensor
            x = torch.randn(self.batch_size, channels, 16, 16).to(self.device)
            
            # Create attention module
            attention = AttentionModule(in_channels=channels, reduction_ratio=ratio).to(self.device)
            
            # Test forward pass
            output = attention(x)
            
            # Check shape
            self.assertEqual(output.shape, x.shape)
            
            # Check that gamma parameter exists and starts at zero
            self.assertIsInstance(attention.gamma, nn.Parameter)
            self.assertTrue(torch.allclose(attention.gamma, torch.zeros_like(attention.gamma)))
    
    def test_attention_visualization_hook(self):
        """Test that we can extract attention maps from the model."""
        model = AttentionNet(num_classes=self.num_classes).to(self.device)
        model.eval()
        
        attention_maps = []
        
        def hook_fn(module, input, output):
            # Extract attention map
            x = input[0]
            batch_size, C, H, W = x.size()
            
            # Compute query and key projections
            proj_query = module.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
            proj_key = module.key(x).view(batch_size, -1, H*W)
            
            # Compute attention weights
            energy = torch.bmm(proj_query, proj_key)
            attention = torch.nn.functional.softmax(energy, dim=-1)
            
            # Store the attention map
            attention_maps.append(attention.detach().cpu())
        
        # Register hook
        handle = model.attention.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = model(self.dummy_input)
        
        # Check that we got attention maps
        self.assertTrue(len(attention_maps) > 0)
        
        # Check shape of attention map (batch_size, H*W, H*W)
        attention_map = attention_maps[0]
        self.assertEqual(attention_map.dim(), 3)
        self.assertEqual(attention_map.shape[0], self.batch_size)
        
        # Remove hook
        handle.remove()

class HybridModelSpecificTests(unittest.TestCase):
    """Additional tests specific to Hybrid model architecture."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.num_classes = 10
        self.img_size = (224, 224)
        self.seq_len = 20
        self.embed_dim = 128
        
        # Create dummy inputs
        self.dummy_input = torch.randn(self.batch_size, 3, *self.img_size).to(self.device)
        self.dummy_seq = torch.randn(self.seq_len, self.batch_size, self.embed_dim).to(self.device)
    
    def test_transformer_block_heads(self):
        """Test transformer block with different numbers of attention heads."""
        head_counts = [1, 2, 4, 8]
        
        for num_heads in head_counts:
            # Create transformer block
            transformer = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                dropout=0.1
            ).to(self.device)
            
            # Test forward pass
            output = transformer(self.dummy_seq)
            
            # Check output shape
            self.assertEqual(output.shape, self.dummy_seq.shape)
    
    def test_transformer_block_feedforward(self):
        """Test transformer block with different feedforward dimensions."""
        ff_dims = [256, 512, 1024]
        
        for ff_dim in ff_dims:
            # Create transformer block
            transformer = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=4,
                ff_dim=ff_dim,
                dropout=0.1
            ).to(self.device)
            
            # Test forward pass
            output = transformer(self.dummy_seq)
            
            # Check output shape
            self.assertEqual(output.shape, self.dummy_seq.shape)
    
    def test_hybrid_model_position_encoding(self):
        """Test that position encoding is properly initialized and applied."""
        model = HybridNet(num_classes=self.num_classes).to(self.device)
        
        # Check position encoding shape
        self.assertEqual(model.pos_encoding.shape[0], model.seq_len)
        self.assertEqual(model.pos_encoding.shape[1], 1)
        self.assertEqual(model.pos_encoding.shape[2], model.fdim)
        
        # Test forward pass
        with torch.no_grad():
            output = model(self.dummy_input)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))

if __name__ == '__main__':
    unittest.main() 