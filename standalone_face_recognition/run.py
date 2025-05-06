#!/usr/bin/env python3

"""
Alzheimer's Assistant: Face Recognition System
---------------------------------------------


Usage:
    python run.py <command> [options]

Commands:
    interactive  - Run the menu interface (easiest option for most users)
    preprocess   - Prepare images for training 
    train        - Train a face recognition model
    evaluate     - Test how well a model performs
    predict      - Identify a person in a single photo
    tune         - Find the best model settings automatically
    check-gpu    - See if you can use GPU acceleration
    list-models  - Show all your trained models

For help with any command: python run.py <command> --help

Development note: This started as a simple CNN classifier but evolved into
a comprehensive system with multiple model architectures as I learned more
about face recognition techniques.
"""

import sys
from src.main import main

# Main entry point - kept this simple to maintain compatibility with
# the module structure I set up. The real work happens in src/main.py
if __name__ == "__main__":
    # My professor suggested using sys.exit() for proper error codes
    sys.exit(main()) 