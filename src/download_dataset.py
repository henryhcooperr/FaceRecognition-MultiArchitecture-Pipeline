#!/usr/bin/env python3

import os
import shutil
import kagglehub
from pathlib import Path
import zipfile
import glob
import random
import sys
import re

# Get the proper paths from the main config
# Fix the import to work when executed directly as a script
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

# Now we can import from src
from src.base_config import RAW_DATA_DIR, logger

# Get project root directory
project_dir = Path(__file__).resolve().parent.parent

# Datasets available to download
DATASETS = {
    "dataset2": {
        "url": "vishesh1412/celebrity-face-image-dataset",
        "description": "Celebrity Faces Dataset (18 celebrities, 100 images each)",
        "needs_unzip": True
    },
    "dataset1": {
        "url": "cybersimar08/face-recognition-dataset",
        "description": "Celebrity Faces Dataset (36 celebrities, 49 images each)",
        "needs_unzip": True
    }
}

def clean_person_name(name):
    """Clean up person name from dataset-specific formats."""
    # Handle "Celebrity Faces Dataset_Angelina Jolie" format
    if "Celeberity Faces Dataset_" in name or "Celebrity Faces Dataset_" in name:
        return name.split("_", 1)[1].strip()
    
    # Handle other common prefixes
    prefixes = ["lfw_", "face_", "person_", "subj_"]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    
    return name

def scan_for_person_directories(root_dir, dataset_id=None):
    """
    Scan for person directories containing images in dataset.
    
    Some datasets have multiple levels of directories that need to be searched.
    """
    person_dirs = []
    
    # Look for top-level person directories first
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains images directly
            images = glob.glob(os.path.join(item_path, "*.jpg")) + glob.glob(os.path.join(item_path, "*.png"))
            if images:
                # For Celebrity Faces Dataset, clean the name
                if dataset_id == "dataset2" and ("Celeberity Faces Dataset_" in item or "Celebrity Faces Dataset_" in item):
                    person_name = clean_person_name(item)
                    person_dirs.append((person_name, item_path))
                else:
                    person_dirs.append((item, item_path))
            else:
                # No images directly in this directory, check its subdirectories
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        # Check if this subdirectory contains images
                        subimages = glob.glob(os.path.join(subitem_path, "*.jpg")) + glob.glob(os.path.join(subitem_path, "*.png"))
                        if subimages:
                            # For dataset-specific structures
                            if item in ["Detected Faces", "Faces", "faces", "detected_faces"]:
                                person_dirs.append((subitem, subitem_path))
                            else:
                                person_dirs.append((f"{item}_{subitem}", subitem_path))
    
    return person_dirs

def extract_images(dataset_path, target_dir, dataset_id=None):
    """Extract images from the dataset to the target directory with no limits."""
    # Special handling for dataset2 dataset
    if dataset_id == "dataset2":
        # Look for "Celebrity Faces Dataset" directory
        celeb_dir = os.path.join(dataset_path, "Celebrity Faces Dataset")
        if os.path.exists(celeb_dir) and os.path.isdir(celeb_dir):
            dataset_path = celeb_dir
    
    # First, try to find person directories in the dataset
    person_dirs = scan_for_person_directories(dataset_path, dataset_id)
    
    # Process directories if found
    if person_dirs:
        total_images = 0
        # Process each person directory
        for person_name, person_dir in person_dirs:
            # Clean up person name
            person_name = clean_person_name(person_name)
            
            # Create directory for this person
            person_target_dir = target_dir / person_name
            person_target_dir.mkdir(exist_ok=True)
            
            # Get all images for this person - no limit
            images = glob.glob(os.path.join(person_dir, "*.jpg")) + glob.glob(os.path.join(person_dir, "*.png"))
            
            # Copy all images
            for i, img_path in enumerate(images):
                target_path = person_target_dir / f"{person_name}_{i:03d}{os.path.splitext(img_path)[1]}"
                shutil.copy2(img_path, target_path)
                total_images += 1
        
        return len(person_dirs), total_images
    
    # If no directories with images found, look for images directly in the dataset
    images = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))
    
    if not images:
        return 0, 0
    
    # Try to extract person names from filenames
    person_images = {}
    for img_path in images:
        filename = os.path.basename(img_path)
        
        # For dataset2 dataset, handle special format
        if dataset_id == "dataset2" and ("Celeberity Faces Dataset_" in filename or "Celebrity Faces Dataset_" in filename):
            # Format: "Celeberity Faces Dataset_Angelina Jolie_1.jpg"
            parts = os.path.splitext(filename)[0].split("_")
            if len(parts) >= 3:  # Need at least 3 parts for this format
                person_name = "_".join(parts[1:-1])  # Join the middle parts
                person_name = clean_person_name(person_name)
            else:
                person_name = f"person_{len(person_images) + 1}"
        else:
            # Standard processing
            name_parts = os.path.splitext(filename)[0].split('_')
            if len(name_parts) > 1:
                person_name = name_parts[0]
            else:
                name_parts = os.path.splitext(filename)[0].split('-')
                if len(name_parts) > 1:
                    person_name = name_parts[0]
                else:
                    # Use parent directory name as a fallback
                    parent_dir = os.path.basename(os.path.dirname(img_path))
                    if parent_dir and parent_dir not in ["dataset", "images", "imgs", "data"]:
                        person_name = parent_dir
                    else:
                        person_name = f"person_{len(person_images) + 1}"
        
        # Clean up the person name
        person_name = clean_person_name(person_name)
        
        if person_name not in person_images:
            person_images[person_name] = []
        
        person_images[person_name].append(img_path)
    
    # Process all people - no limit
    person_names = list(person_images.keys())
    
    # Process each person's images
    total_images = 0
    for person_name in person_names:
        person_target_dir = target_dir / person_name
        person_target_dir.mkdir(exist_ok=True)
        
        # Process all images - no limit
        person_imgs = person_images[person_name]
        
        # Copy images
        for i, img_path in enumerate(person_imgs):
            target_path = person_target_dir / f"{person_name}_{i:03d}{os.path.splitext(img_path)[1]}"
            shutil.copy2(img_path, target_path)
            total_images += 1
    
    return len(person_names), total_images

def download_dataset(dataset_id):
    """Download a specific dataset and organize it with no limits."""
    if dataset_id not in DATASETS:
        print(f"Unknown dataset: {dataset_id}")
        return False
    
    # Create target directory
    target_dir = RAW_DATA_DIR / dataset_id
    
    # Skip if already downloaded
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Dataset {dataset_id} already exists at {target_dir}")
        return True
    
    # Make sure directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(exist_ok=True)
    
    # Download dataset
    url = DATASETS[dataset_id]["url"]
    logger.info(f"Downloading {dataset_id} ({DATASETS[dataset_id]['description']})...")
    print(f"Downloading {dataset_id} ({DATASETS[dataset_id]['description']})...")
    print(f"URL: {url}")
    
    try:
        # Download from Kaggle using kagglehub
        print(f"Downloading dataset using kagglehub: {url}")
        logger.info(f"Downloading dataset using kagglehub: {url}")
        dataset_path = kagglehub.dataset_download(url)
        logger.info(f"Downloaded to: {dataset_path}")
        print(f"Downloaded to: {dataset_path}")

        # Extract dataset to temp directory
        temp_dir = project_dir / f"temp_{dataset_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        # Look for zip files to extract
        zip_files = glob.glob(os.path.join(dataset_path, "*.zip"))
        if zip_files:
            for zip_file in zip_files:
                logger.info(f"Extracting {zip_file}...")
                print(f"Extracting {zip_file}...")
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                except zipfile.BadZipFile:
                    logger.warning(f"Invalid zip file {zip_file}, skipping...")
                    print(f"Warning: Invalid zip file {zip_file}, skipping...")
                    continue

        # Copy all files from the dataset directory
        for item in os.listdir(dataset_path):
            src = Path(dataset_path) / item
            dst = temp_dir / item
            if src.is_dir():
                if not (dst.exists()):  # Avoid copying if already extracted from zip
                    shutil.copytree(src, dst)
            elif not dst.exists():  # Avoid copying if already extracted
                shutil.copy2(src, dst)
        
        # Extract and organize images
        logger.info(f"Organizing images for {dataset_id}...")
        print(f"Organizing images for {dataset_id}...")
        num_persons, num_images = extract_images(
            temp_dir, target_dir, 
            dataset_id=dataset_id
        )
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        if num_images > 0:
            logger.info(f"Successfully organized {num_images} images for {num_persons} persons")
            print(f"Successfully organized {num_images} images for {num_persons} persons in {target_dir}")
            
            # Create info.txt file with dataset information
            with open(target_dir / "info.txt", "w") as f:
                f.write(f"Dataset: {dataset_id}\n")
                f.write(f"Description: {DATASETS[dataset_id]['description']}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Number of persons: {num_persons}\n")
                f.write(f"Number of images: {num_images}\n")
            
            return True
        else:
            logger.warning(f"No images found in the dataset. Removing {target_dir}")
            print(f"No images found in the dataset. Removing {target_dir}")
            shutil.rmtree(target_dir)
            return False
            
    except Exception as e:
        logger.error(f"Error downloading/extracting dataset {dataset_id}: {e}")
        print(f"Error downloading/extracting dataset {dataset_id}: {e}")
        # Clean up if there was an error
        if target_dir.exists():
            shutil.rmtree(target_dir)
        return False

def download_all_datasets():
    """Download all defined datasets with no limits."""
    success = True
    for dataset_id in DATASETS:
        print(f"\n{'='*60}")
        result = download_dataset(dataset_id)
        print(f"{'='*60}")
        success = success and result
    
    return success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download face recognition datasets')
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()), 
                        help='Specific dataset to download')
    
    args = parser.parse_args()
    
    # Create raw data directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.dataset:
        download_dataset(args.dataset)
    else:
        print("Downloading all datasets...")
        download_all_datasets()
        print("\nDatasets downloaded to subdirectories in:", RAW_DATA_DIR)
        
    print("\nYou can now use this dataset with the face recognition system.")
    print("Example commands:")
    print("  python run.py preprocess")
    print("  python run.py train --model-type cnn")
    print("  python run.py interactive") 