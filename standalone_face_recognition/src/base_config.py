#!/usr/bin/env python3

import os
import logging
from pathlib import Path

# Project paths - changed to use pathlib after fighting with os.path for hours
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROC_DATA_DIR = DATA_DIR / "processed"  # processed data goes here
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"  # shortened this
CHECKPOINTS_DIR = OUT_DIR / "checkpoints"
VIZ_DIR = OUT_DIR / "visualizations"  

# Model parameters I've tuned through trial and error
# Smaller batch size helped with my limited dataset
DEFAULT_BATCH_SIZE = 16  # was 32 but got OOM errors on my laptop
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3  # 0.001 seems to work well for most models
IMG_SIZE = 224  # This is what ResNet expects

# Make sure all our directories exist
# This annoyed me to no end when things failed silently
for dir_path in [RAW_DATA_DIR, PROC_DATA_DIR, MODELS_DIR, 
                CHECKPOINTS_DIR, VIZ_DIR,
                PROC_DATA_DIR / "train",
                PROC_DATA_DIR / "val",
                PROC_DATA_DIR / "test"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging - super helpful for debugging
# Added timestamp after getting confused about when errors happened
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FIXME: This is a bit clunky but works for now
# Should eventually replace with a proper CLI confirmation
def get_user_confirmation(prompt: str = "Continue? (y/n): ") -> bool:
    """Simple yes/no prompt to get user confirmation.
    
    I use this to avoid accidentally overwriting data or models.
    """
    while True:
        response = input(prompt).lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'")

# Old validation split function I replaced - keeping for reference
# def split_dataset(data_dir, train=0.7, val=0.2, test=0.1):
#     """Split dataset into train/val/test."""
#     assert abs(train + val + test - 1.0) < 1e-8, "Ratios must sum to 1"
#     # rest of function... 