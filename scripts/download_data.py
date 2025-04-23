#!/usr/bin/env python3
"""
Script to download and verify the Celebrity Face Image Dataset from Kaggle.
"""

import os
import sys
import logging
from pathlib import Path
import kaggle
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/raw', 'data/processed']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def download_dataset():
    """Download the dataset from Kaggle."""
    try:
        logger.info("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            'vishesh1412/celebrity-face-image-dataset',
            path='data/raw',
            unzip=True
        )
        logger.info("Dataset downloaded successfully!")
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        sys.exit(1)

def verify_dataset():
    """Verify the downloaded dataset structure and image counts."""
    dataset_path = Path('data/raw/celebrity_faces')
    if not dataset_path.exists():
        logger.error("Dataset directory not found!")
        sys.exit(1)

    # Count images per celebrity
    celebrity_counts = {}
    for celeb_dir in dataset_path.iterdir():
        if celeb_dir.is_dir():
            image_count = len(list(celeb_dir.glob('*.jpg')))
            celebrity_counts[celeb_dir.name] = image_count

    # Verify counts
    expected_count = 1000
    total_celebrities = len(celebrity_counts)
    total_images = sum(celebrity_counts.values())

    logger.info(f"\nDataset Verification Results:")
    logger.info(f"Total celebrities: {total_celebrities}")
    logger.info(f"Total images: {total_images}")
    logger.info("\nImages per celebrity:")
    for celeb, count in celebrity_counts.items():
        status = "✅" if count == expected_count else "❌"
        logger.info(f"{status} {celeb}: {count} images")

    # Check for any discrepancies
    if any(count != expected_count for count in celebrity_counts.values()):
        logger.warning("Some celebrities have incorrect image counts!")
    else:
        logger.info("All celebrity directories have the correct number of images!")

def main():
    """Main function to orchestrate the download and verification process."""
    setup_directories()
    download_dataset()
    verify_dataset()

if __name__ == "__main__":
    main() 