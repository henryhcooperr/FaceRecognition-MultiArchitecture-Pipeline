#!/usr/bin/env python3

import os
import shutil
import kagglehub  # had issues with this lib version 0.2.5
from pathlib import Path
import zipfile
import glob
import random
import sys

# Get the proper paths from the face recognition system
project_dir = Path(__file__).resolve().parent
# Add the project directory to the path so we can import modules
sys.path.insert(0, str(project_dir))

try:
    # Try to import from the face recognition system
    from src.base_config import RAW_DATA_DIR
    print(f"Using RAW_DATA_DIR: {RAW_DATA_DIR}")
except ImportError:
    # Fallback if we can't import
    RAW_DATA_DIR = project_dir / "data" / "raw"
    print(f"Using fallback RAW_DATA_DIR: {RAW_DATA_DIR}")

# Datasets I found that work well for this project
# TODO: add more datasets if I have time before the final demo
DATASETS = {
    "celebrity_faces": {
        "kaggle_id": "vishesh1412/celebrity-face-image-dataset", 
        "description": "18 celebrities with ~100 images each",
    },
    "lfw": {
        "kaggle_id": "atulanandjha/lfwpeople",
        "description": "5749 people with varying numbers of images from LFW dataset",
    },
    "face_recognition": {
        "kaggle_id": "cybersimar08/face-recognition-dataset",
        "description": "Face recognition dataset with multiple subjects",
    }
}

def clean_person_name(name):
    """Cleans up names to remove weird prefixes"""
    # Handle "Celebrity Faces Dataset_Angelina Jolie" format
    if "Celeberity Faces Dataset_" in name or "Celebrity Faces Dataset_" in name:
        return name.split("_", 1)[1].strip()
    
    # Handle other common prefixes
    prefixes = ["lfw_", "face_", "person_", "subj_"]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    
    return name

# FIXME: refactor this func if time permits, got too complex
def scan_for_person_directories(root_dir, dataset_id=None):
    """
    Scan for person directories containing images in dataset.
    
    Some datasets (like LFW) have multiple levels of directories or 
    special folders like "Detected Faces"/"Faces" that need to be searched.
    
    Returns: list of tuples (person_name, dir_path)
    """
    dirs = []
    
    # This approach is more reliable than my previous attempt that just looked at top level 
    # - found this after hours of debugging the LFW dataset structure :/
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains images directly
            images = glob.glob(os.path.join(item_path, "*.jpg")) + glob.glob(os.path.join(item_path, "*.png"))
            if images:
                # For Celebrity Faces Dataset, clean the name
                if dataset_id == "celebrity_faces" and ("Celeberity Faces Dataset_" in item or "Celebrity Faces Dataset_" in item):
                    person_name = clean_person_name(item)
                    dirs.append((person_name, item_path))
                else:
                    dirs.append((item, item_path))
            else:
                # No images directly in this directory, check its subdirectories
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        # Check if this subdirectory contains images
                        subimgs = glob.glob(os.path.join(subitem_path, "*.jpg")) + glob.glob(os.path.join(subitem_path, "*.png"))
                        if subimgs:
                            # For LFW-type datasets, use both the parent and child directory names
                            # (e.g., "Detected Faces/George_W_Bush" -> "George_W_Bush")
                            if item in ["Detected Faces", "Faces", "faces", "detected_faces", "lfw"]:
                                dirs.append((subitem, subitem_path))
                            else:
                                dirs.append((f"{item}_{subitem}", subitem_path))
    
    return dirs

def extract_images(dataset_path, target_dir, max_celebrities=20, images_per_celebrity=20, dataset_id=None):
    """Extract images from the dataset to the target directory."""
    # Special handling for celebrity_faces dataset
    if dataset_id == "celebrity_faces":
        # Look for "Celebrity Faces Dataset" directory
        celeb_dir = os.path.join(dataset_path, "Celebrity Faces Dataset")
        if os.path.exists(celeb_dir) and os.path.isdir(celeb_dir):
            dataset_path = celeb_dir
    
    # First, try to find person directories in the dataset
    person_dirs = scan_for_person_directories(dataset_path, dataset_id)
    
    # Process directories if found
    if person_dirs:
        # Limit to max_celebrities
        if max_celebrities and len(person_dirs) > max_celebrities:
            random.shuffle(person_dirs)
            person_dirs = person_dirs[:max_celebrities]
        
        total_imgs = 0
        # Process each person directory
        for name, dir in person_dirs:
            # Clean up person name
            name = clean_person_name(name)
            
            # Create directory for this person
            person_dir = target_dir / name
            person_dir.mkdir(exist_ok=True)
            
            # Get and limit images
            imgs = glob.glob(os.path.join(dir, "*.jpg")) + glob.glob(os.path.join(dir, "*.png"))
            if images_per_celebrity and len(imgs) > images_per_celebrity:
                random.shuffle(imgs)
                imgs = imgs[:images_per_celebrity]
            
            # Copy images
            for i, img in enumerate(imgs):
                target = person_dir / f"{name}_{i:03d}{os.path.splitext(img)[1]}"
                shutil.copy2(img, target)
                total_imgs += 1
        
        return len(person_dirs), total_imgs
    
    # If no directories with images found, look for images directly in the dataset
    # This was added after finding some datasets don't use the standard directory structure
    imgs = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                imgs.append(os.path.join(root, file))
    
    if not imgs:
        return 0, 0
    
    # Try to extract person names from filenames
    person_imgs = {}
    for img in imgs:
        filename = os.path.basename(img)
        
        # For celebrity faces dataset, handle special format
        if dataset_id == "celebrity_faces" and ("Celeberity Faces Dataset_" in filename or "Celebrity Faces Dataset_" in filename):
            # Format: "Celeberity Faces Dataset_Angelina Jolie_1.jpg"
            parts = os.path.splitext(filename)[0].split("_")
            if len(parts) >= 3:  # Need at least 3 parts for this format
                person_name = "_".join(parts[1:-1])  # Join the middle parts
                person_name = clean_person_name(person_name)
            else:
                person_name = f"person_{len(person_imgs) + 1}"
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
                    parent_dir = os.path.basename(os.path.dirname(img))
                    if parent_dir and parent_dir not in ["dataset", "images", "imgs", "data"]:
                        person_name = parent_dir
                    else:
                        person_name = f"person_{len(person_imgs) + 1}"
        
        # Clean up the person name
        person_name = clean_person_name(person_name)
        
        if person_name not in person_imgs:
            person_imgs[person_name] = []
        
        person_imgs[person_name].append(img)
    
    # Limit to max_celebrities
    person_names = list(person_imgs.keys())
    if max_celebrities and len(person_names) > max_celebrities:
        person_names = person_names[:max_celebrities]
    
    # Process each person's images
    total_imgs = 0
    for name in person_names:
        person_dir = target_dir / name
        person_dir.mkdir(exist_ok=True)
        
        # Limit images per person
        imgs = person_imgs[name]
        if images_per_celebrity and len(imgs) > images_per_celebrity:
            random.shuffle(imgs)
            imgs = imgs[:images_per_celebrity]
        
        # Copy images
        for i, img in enumerate(imgs):
            target = person_dir / f"{name}_{i:03d}{os.path.splitext(img)[1]}"
            shutil.copy2(img, target)
            total_imgs += 1
    
    return len(person_names), total_imgs

# Spent way too long figuring out the LFW dataset structure - it's completely different
# from the other datasets. This handles that special case.
# TODO: merge this with extract_images when I have time to refactor
def handle_lfw_dataset(dataset_path, target_dir, max_celebrities=20, images_per_celebrity=20):
    """Special handling for LFW dataset (pain in the ass)"""
    # First, look for the lfw directory structure
    people_dirs = []
    
    # Check all possible locations for LFW directories
    for root, dirs, _ in os.walk(dataset_path):
        for d in dirs:
            # Look for directories that might be person folders (typical LFW structure)
            person_dir = os.path.join(root, d)
            # Only consider if it has enough images
            imgs = glob.glob(os.path.join(person_dir, "*.jpg"))
            if len(imgs) >= 4:  # Often celebrities have at least 4 images
                people_dirs.append((d, person_dir))
    
    if not people_dirs:
        print("Could not find LFW-formatted directories with person images")
        return 0, 0
    
    # Limit to max_celebrities with most images
    people_dirs.sort(key=lambda x: len(glob.glob(os.path.join(x[1], "*.jpg"))), reverse=True)
    if max_celebrities and len(people_dirs) > max_celebrities:
        people_dirs = people_dirs[:max_celebrities]
    
    total_imgs = 0
    # Process each person directory
    for name, dir in people_dirs:
        name = clean_person_name(name)
        
        # Create directory for this person
        person_dir = target_dir / name
        person_dir.mkdir(exist_ok=True)
        
        # Get all images for this person
        imgs = glob.glob(os.path.join(dir, "*.jpg"))
        if images_per_celebrity and len(imgs) > images_per_celebrity:
            random.shuffle(imgs)
            imgs = imgs[:images_per_celebrity]
        
        # Copy images
        for i, img in enumerate(imgs):
            target = person_dir / f"{name}_{i:03d}{os.path.splitext(img)[1]}"
            shutil.copy2(img, target)
            total_imgs += 1
    
    return len(people_dirs), total_imgs

def download_dataset(dataset_id, max_celebrities=20, images_per_celebrity=20):
    """Download dataset from Kaggle and organize it"""
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
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    kaggle_id = DATASETS[dataset_id]["kaggle_id"]
    print(f"Downloading {dataset_id} ({DATASETS[dataset_id]['description']})...")
    print(f"Kaggle ID: {kaggle_id}")
    
    try:
        # Download from Kaggle
        # NOTE: Initially tried kaggle CLI but switched to kagglehub because it's easier
        dataset_path = kagglehub.dataset_download(kaggle_id)
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
                print(f"Extracting {zip_file}...")
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                except zipfile.BadZipFile:
                    print(f"Warning: Invalid zip file {zip_file}, skipping...")
                    continue
        
        # Copy all files - needed for LFW which may not have zip files
        # This was added later after having issues with LFW dataset
        for item in os.listdir(dataset_path):
            src = Path(dataset_path) / item
            dst = temp_dir / item
            if src.is_dir():
                if not (dst.exists()):  # Avoid copying if already extracted from zip
                    shutil.copytree(src, dst)
            elif not dst.exists():  # Avoid copying if already extracted
                shutil.copy2(src, dst)
        
        # Process based on dataset type
        if dataset_id == "lfw":
            print(f"Processing LFW dataset with special handling...")
            num_persons, num_images = handle_lfw_dataset(
                temp_dir, target_dir, 
                max_celebrities=max_celebrities,
                images_per_celebrity=images_per_celebrity
            )
        else:
            # Extract and organize images
            print(f"Organizing images for {dataset_id}...")
            num_persons, num_images = extract_images(
                temp_dir, target_dir, 
                max_celebrities=max_celebrities,
                images_per_celebrity=images_per_celebrity,
                dataset_id=dataset_id
            )
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        if num_images > 0:
            print(f"Successfully organized {num_images} images for {num_persons} persons in {target_dir}")
            
            # Create info.txt file with dataset information
            with open(target_dir / "info.txt", "w") as f:
                f.write(f"Dataset: {dataset_id}\n")
                f.write(f"Description: {DATASETS[dataset_id]['description']}\n")
                f.write(f"Kaggle ID: {kaggle_id}\n")
                f.write(f"Number of persons: {num_persons}\n")
                f.write(f"Number of images: {num_images}\n")
            
            return True
        else:
            print(f"No images found in the dataset. Removing {target_dir}")
            shutil.rmtree(target_dir)
            return False
            
    except Exception as e:
        print(f"Error downloading/extracting dataset {dataset_id}: {e}")
        # Clean up if there was an error
        if target_dir.exists():
            shutil.rmtree(target_dir)
        return False

# Old version that didn't work well with some datasets - keeping for reference
# def download_all_datasets_v1(max_celebrities=20, images_per_celebrity=20):
#     for dataset_id in DATASETS:
#         download_dataset(dataset_id, max_celebrities, images_per_celebrity)

def download_all_datasets(max_celebrities=20, images_per_celebrity=20):
    """Download all defined datasets."""
    success = True
    for dataset_id in DATASETS:
        print(f"\n{'='*60}")
        result = download_dataset(dataset_id, max_celebrities, images_per_celebrity)
        print(f"{'='*60}")
        success = success and result
    
    return success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download face recognition datasets')
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()), 
                        help='Specific dataset to download')
    parser.add_argument('--max-celebrities', type=int, default=20, 
                        help='Maximum number of celebrities per dataset')
    parser.add_argument('--images-per-celebrity', type=int, default=20,
                        help='Maximum images per celebrity')
    
    args = parser.parse_args()
    
    # Create raw data directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.dataset:
        download_dataset(args.dataset, args.max_celebrities, args.images_per_celebrity)
    else:
        print("Downloading all datasets...")
        download_all_datasets(args.max_celebrities, args.images_per_celebrity)
        print("\nDatasets downloaded to subdirectories in:", RAW_DATA_DIR)
        
    print("\nUse 'python run.py interactive' to process and train models with these datasets.") 