#!/usr/bin/env python3

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import pandas as pd
from PIL import Image
import time
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .base_config import logger, CHECKPOINTS_DIR, VIZ_DIR
from .face_models import get_model, SiameseNet
from .training_utils import load_checkpoint
from .data_utils import SiameseDataset


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class EmbeddingVisualizer:
    """Visualize face embeddings from a Siamese network."""
    
    def __init__(self, model_dir: Union[str, Path], device=None):
        """Initialize visualizer with model directory."""
        self.model_dir = Path(model_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.metrics = None
        self.embeddings_2d = None
        self.embeddings_3d = None
        self.image_paths = []
        self.identities = []
        
        # Cool color palette 
        self.colors = [
            '#FF595E', '#FFCA3A', '#8AC926', '#1982C4', '#6A4C93',
            '#E63946', '#F1FAEE', '#A8DADC', '#457B9D', '#1D3557',
            '#FB8500', '#219EBC', '#023047', '#FFB703', '#8ECAE6',
            '#F72585', '#7209B7', '#3A0CA3', '#4361EE', '#4CC9F0'
        ]
    
    def load_model(self, model_name: Optional[str] = None):
        """Load the trained model."""
        # Find model checkpoint
        if model_name:
            model_path = CHECKPOINTS_DIR / model_name / 'best_model.pth'
        else:
            # Find most recent siamese model
            model_dirs = list(CHECKPOINTS_DIR.glob('siamese_*'))
            if not model_dirs:
                raise ValueError("No siamese models found")
            model_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = model_dirs[0] / 'best_model.pth'
        
        # Initialize model
        self.model = SiameseNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        
        # Try to load model metrics
        metrics_path = model_path.parent / 'metrics' / 'learning_curves.csv'
        if metrics_path.exists():
            self.metrics = pd.read_csv(metrics_path)
        
        # Store model info for display
        self.model_name = model_path.parent.name
        self.model_path = model_path
        return self.model
    
    def process_dataset(self, dataset_path: Union[str, Path], max_samples: int = 300):
        """Process dataset and extract embeddings."""
        dataset_path = Path(dataset_path)
        
        # If no model loaded, try to load first
        if self.model is None:
            self.load_model()
        
        # Load dataset
        transform = None  # Use dataset's default transform
        dataset = SiameseDataset(str(dataset_path), transform=transform)
        
        # Extract embeddings for a sample of images
        embeddings = []
        self.image_paths = []
        self.identities = []
        
        with torch.no_grad():
            # Get all unique image paths
            all_paths = set()
            all_identities = []
            
            # Collect all unique images from the dataset
            for i in range(min(len(dataset), 1000)):  # Limit initial scan to 1000 pairs
                img1_path, img2_path = dataset.image_pairs[i]
                all_paths.add(img1_path)
                all_paths.add(img2_path)
                
                # Extract identity from parent directory name
                identity1 = Path(img1_path).parent.name
                identity2 = Path(img2_path).parent.name
                
                if identity1 not in all_identities:
                    all_identities.append(identity1)
                if identity2 not in all_identities:
                    all_identities.append(identity2)
            
            # Limit to max_samples
            all_paths = list(all_paths)[:max_samples]
            
            # Map identities to colors
            identity_to_color = {}
            for i, identity in enumerate(all_identities):
                identity_to_color[identity] = self.colors[i % len(self.colors)]
            
            # Process each image
            for img_path in all_paths:
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                img_tensor = dataset.transform(img).unsqueeze(0).to(self.device)
                
                # Get embedding
                embedding = self.model.get_embedding(img_tensor)
                
                # Save data
                embeddings.append(embedding.cpu().numpy().flatten())
                self.image_paths.append(img_path)
                identity = Path(img_path).parent.name
                self.identities.append(identity)
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Reduce dimensionality for visualization
        if len(embeddings) > 5:  # Need at least a few samples
            # PCA first for efficiency if we have many points
            if len(embeddings) > 50:
                pca = PCA(n_components=min(50, len(embeddings)))
                embeddings_50d = pca.fit_transform(embeddings)
            else:
                embeddings_50d = embeddings
            
            # T-SNE for final visualization dimensions
            tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1))
            self.embeddings_2d = tsne.fit_transform(embeddings_50d)
            
            tsne3d = TSNE(n_components=3, perplexity=min(30, len(embeddings) - 1))
            self.embeddings_3d = tsne3d.fit_transform(embeddings_50d)
        
        # Create color map for identities
        self.identity_colors = [identity_to_color.get(identity, '#CCCCCC') for identity in self.identities]
        
        logger.info(f"Processed {len(embeddings)} images from {len(set(self.identities))} identities")
        return self.embeddings_2d
    
    def generate_2d_plot(self, output_path: Optional[Union[str, Path]] = None):
        """
        Save 2D embedding data to CSV instead of plotting.
        Maintains API compatibility with original visualization.
        """
        if self.embeddings_2d is None:
            raise ValueError("No embeddings to visualize. Call process_dataset first.")
        
        # Save embeddings to CSV instead of plotting
        if output_path:
            output_path = Path(output_path)
            csv_output_path = output_path.parent / f"{output_path.stem}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a DataFrame with the embedding data
            df = pd.DataFrame({
                'x': self.embeddings_2d[:, 0],
                'y': self.embeddings_2d[:, 1],
                'identity': self.identities,
                'color': self.identity_colors
            })
            
            # Save to CSV
            df.to_csv(csv_output_path, index=False)
            logger.info(f"Saved 2D embedding data to {csv_output_path}")
        
        # Also log a message about plotting being disabled
        logger.info("2D plot visualization is disabled in this simplified branch")
        
        # Return None to maintain API compatibility
        return None
    
    def generate_3d_plot(self, output_path: Optional[Union[str, Path]] = None):
        """
        Save 3D embedding data to CSV instead of plotting.
        Maintains API compatibility with original visualization.
        """
        if self.embeddings_3d is None:
            raise ValueError("No 3D embeddings to visualize. Call process_dataset first.")
        
        # Save embeddings to CSV instead of plotting
        if output_path:
            output_path = Path(output_path)
            csv_output_path = output_path.parent / f"{output_path.stem}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a DataFrame with the embedding data
            df = pd.DataFrame({
                'x': self.embeddings_3d[:, 0],
                'y': self.embeddings_3d[:, 1],
                'z': self.embeddings_3d[:, 2],
                'identity': self.identities,
                'color': self.identity_colors
            })
            
            # Save to CSV
            df.to_csv(csv_output_path, index=False)
            logger.info(f"Saved 3D embedding data to {csv_output_path}")
        
        # Also log a message about plotting being disabled
        logger.info("3D plot visualization is disabled in this simplified branch")
        
        # Return None to maintain API compatibility
        return None
    
    def generate_animated_plot(self, output_path: Optional[Union[str, Path]] = None, frames: int = 120):
        """
        Skip animated plot generation as it requires matplotlib animation.
        Maintains API compatibility with original visualization.
        """
        if self.embeddings_3d is None:
            raise ValueError("No 3D embeddings to visualize. Call process_dataset first.")
        
        # Log that animation is skipped
        logger.info("Animated plot visualization is disabled in this simplified branch")
        
        # If path provided, save a note file instead
        if output_path:
            output_path = Path(output_path)
            note_path = output_path.parent / f"{output_path.stem}_note.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(note_path, 'w') as f:
                f.write("Animation generation is disabled in this simplified branch.\n")
                f.write(f"This file would have contained an animated visualization of {len(self.identities)} embeddings.")
            
            logger.info(f"Saved note about disabled animation to {note_path}")
        
        # Return None to maintain API compatibility
        return None
    
    def generate_similarity_matrix(self, output_path: Optional[Union[str, Path]] = None):
        """
        Calculate and save similarity matrix data to CSV instead of plotting.
        Maintains API compatibility with original visualization.
        """
        if self.embeddings_2d is None:
            raise ValueError("No embeddings to visualize. Call process_dataset first.")
        
        # Limit to 30 samples for readability
        max_samples = min(30, len(self.image_paths))
        embeddings = self.embeddings_2d[:max_samples]
        identities = self.identities[:max_samples]
        
        # Calculate similarity matrix (using Euclidean distance)
        similarity = np.zeros((max_samples, max_samples))
        for i in range(max_samples):
            for j in range(max_samples):
                # Euclidean distance
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                # Convert to similarity (1 for identical, 0 for very different)
                similarity[i, j] = np.exp(-dist)
        
        # Save similarity matrix to CSV
        if output_path:
            output_path = Path(output_path)
            csv_output_path = output_path.parent / f"{output_path.stem}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a DataFrame with row and column headers
            df = pd.DataFrame(similarity, index=identities, columns=identities)
            
            # Save to CSV
            df.to_csv(csv_output_path)
            logger.info(f"Saved similarity matrix to {csv_output_path}")
        
        # Also log a message about plotting being disabled
        logger.info("Similarity matrix visualization is disabled in this simplified branch")
        
        # Return None to maintain API compatibility
        return None
    
    def create_web_interface(self, dataset_path: Optional[Union[str, Path]] = None):
        """
        The web interface functionality is disabled in this simplified branch.
        Function maintained for API compatibility.
        """
        logger.info("Web interface visualization is disabled in this simplified branch")
        
        # Just print that the web interface is disabled
        print("Web interface visualization is disabled in this simplified branch")
        print("In the original version, this would launch a Gradio web interface")
        
        # Return None to maintain API compatibility
        return None


def generate_visualization_report(model_dir: Union[str, Path], dataset_path: Union[str, Path], 
                                output_dir: Optional[Union[str, Path]] = None):
    """
    Save embedding data to CSV files instead of generating visualizations.
    Maintains API compatibility with original visualization report function.
    """
    model_dir = Path(model_dir)
    dataset_path = Path(dataset_path)
    
    # Create output directory
    if output_dir is None:
        output_dir = VIZ_DIR / model_dir.name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer(model_dir)
    visualizer.load_model()
    
    # Process dataset
    visualizer.process_dataset(dataset_path)
    
    # Generate CSV data files instead of visualizations
    visualizer.generate_2d_plot(output_dir / "embeddings_2d.png")  # This now saves CSV
    visualizer.generate_3d_plot(output_dir / "embeddings_3d.png")  # This now saves CSV
    visualizer.generate_animated_plot(output_dir / "embeddings_animated.gif")  # This now saves a note
    visualizer.generate_similarity_matrix(output_dir / "similarity_matrix.png")  # This now saves CSV
    
    # Generate a simplified text report instead of HTML
    report_text = f"""
    # Siamese Network Visualization Report - {model_dir.name}
    
    ## Overview
    - Model: {model_dir.name}
    - Dataset: {dataset_path}
    - Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
    
    ## Data Files
    - 2D Embeddings: embeddings_2d.csv
    - 3D Embeddings: embeddings_3d.csv
    - Similarity Matrix: similarity_matrix.csv
    
    ## Note
    Visualization features are disabled in this simplified branch.
    The CSV files contain the raw data that would have been visualized.
    
    Generated using Siamese Network Data Export Tool
    """
    
    # Save text report
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)
    
    logger.info(f"Generated data export at {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Network Visualization Tool")
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    
    args = parser.parse_args()
    
    if args.web:
        # Launch web interface
        visualizer = EmbeddingVisualizer(args.model if args.model else None)
        visualizer.create_web_interface(args.dataset if args.dataset else None)
    elif args.model and args.dataset:
        # Generate report
        generate_visualization_report(args.model, args.dataset, args.output)
    else:
        parser.print_help()