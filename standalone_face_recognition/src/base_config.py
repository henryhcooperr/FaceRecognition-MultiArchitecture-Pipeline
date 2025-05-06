#!/usr/bin/env python3

import os
import logging
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                CHECKPOINTS_DIR, VISUALIZATIONS_DIR,
                PROCESSED_DATA_DIR / "train",
                PROCESSED_DATA_DIR / "val",
                PROCESSED_DATA_DIR / "test"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_user_confirmation(prompt: str = "Continue? (y/n): ") -> bool:
    """Get confirmation from user."""
    while True:
        response = input(prompt).lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'") 