#!/usr/bin/env python3

"""
Face Recognition System
-----------------------

This is a command-line interface to a modular face recognition system.
It supports different model architectures, preprocessing steps, and evaluation tools.

Usage:
    python run.py <command> [options]

Commands:
    interactive  - Run the interactive menu interface
    preprocess   - Preprocess raw data for training
    train        - Train a face recognition model
    evaluate     - Evaluate a trained model
    predict      - Make a prediction on a single image
    tune         - Tune hyperparameters for a model
    check-gpu    - Check GPU availability
    list-models  - List all trained models

Run `python run.py <command> --help` for more information on a specific command.
"""

import sys
from src.main import main

if __name__ == "__main__":
    sys.exit(main()) 