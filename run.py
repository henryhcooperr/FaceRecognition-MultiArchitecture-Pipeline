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
from src.interactive import interactive_menu
from src.app import main as run_app

def show_usage():
    print("Face Recognition Project - Simplified Version")
    print("Usage:")
    print("  python run.py             - Launch interactive menu")
    print("  python run.py interactive - Launch interactive menu")
    print("  python run.py demo        - Launch live demo app")
    print("  python run.py cv          - Run cross-validation")
    print("  python run.py hyperopt    - Run hyperparameter tuning")
    print("  python run.py help        - Show this help message")

if __name__ == "__main__":
    # Process command line arguments
    if len(sys.argv) == 1 or sys.argv[1] == "interactive":
        # Run interactive mode
        interactive_menu()
    elif sys.argv[1] == "demo":
        # Run demo app
        run_app()
    elif sys.argv[1] == "cv":
        # Run cross-validation
        from src.cross_validation import run_cross_validation
        run_cross_validation()
    elif sys.argv[1] == "hyperopt":
        # Run hyperparameter tuning
        from src.hyperparameter_tuning import run_hyperparameter_tuning
        run_hyperparameter_tuning()
    elif sys.argv[1] == "help" or sys.argv[1] == "--help" or sys.argv[1] == "-h":
        show_usage()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        show_usage()
        sys.exit(1) 