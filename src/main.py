#!/usr/bin/env python3

import argparse
import sys
import torch
from pathlib import Path

from .base_config import logger, CHECKPOINTS_DIR
from .data_prep import get_preprocessing_config, process_raw_data
from .training import train_model
from .testing import evaluate_model, predict_image
from .face_models import get_model
from .interactive import interactive_menu

def main():
    """Main entry point for face recognition system.
    Simplified version with only essential components.
    """
    parser = argparse.ArgumentParser(description='Face Recognition System - Simplified')
    subparsers = parser.add_subparsers(dest='cmd', help='Command to run')
    
    # Interactive command 
    subparsers.add_parser('interactive', help='Run the interactive menu interface')
    
    # Demo command
    subparsers.add_parser('demo', help='Run live demo app')
    
    # CV command
    subparsers.add_parser('cv', help='Run cross-validation')
    
    # Hyperopt command
    subparsers.add_parser('hyperopt', help='Run hyperparameter tuning')
    
    # Preprocess command
    preproc = subparsers.add_parser('preprocess', help='Preprocess raw data')
    preproc.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    
    # Train command
    train_p = subparsers.add_parser('train', help='Train a model')
    train_p.add_argument('--model-type', type=str, required=True, 
                            choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid', 'ensemble'],
                            help='Type of model to train')
    train_p.add_argument('--model-name', type=str, help='Name for the trained model')
    train_p.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    train_p.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_p.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_p.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    # Evaluate command
    eval_p = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_p.add_argument('--model-type', type=str, required=True, 
                           choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid', 'ensemble'],
                           help='Type of model to evaluate')
    eval_p.add_argument('--model-name', type=str, help='Name of the model to evaluate')
    
    # Predict command
    pred_p = subparsers.add_parser('predict', help='Predict on a single image')
    pred_p.add_argument('--model-type', type=str, required=True, 
                              choices=['baseline', 'cnn', 'attention', 'arcface', 'hybrid', 'ensemble'],
                              help='Type of model to use (not siamese)')
    pred_p.add_argument('--model-name', type=str, help='Name of the model to use')
    pred_p.add_argument('--image-path', type=str, required=True, help='Path to the image to predict')
    
    # Check GPU command
    subparsers.add_parser('check-gpu', help='Check GPU availability')
    
    # List models command
    list_models_p = subparsers.add_parser('list-models', help='List available trained models')
    
    args = parser.parse_args()
    
    # If no command is provided, show help
    if args.cmd is None:
        parser.print_help()
        return 1
    
    # Execute the appropriate command
    if args.cmd == 'interactive':
        return interactive_menu()
    
    elif args.cmd == 'demo':
        # Run the live demo
        from .app import main as run_app
        return run_app()
    
    elif args.cmd == 'cv':
        # Run cross-validation
        from .cross_validation import run_cross_validation
        run_cross_validation()
    
    elif args.cmd == 'hyperopt':
        # Run hyperparameter tuning
        from .hyperparameter_tuning import run_hyperparameter_tuning
        run_hyperparameter_tuning()
    
    elif args.cmd == 'preprocess':
        config = get_preprocessing_config()
        from .base_config import RAW_DATA_DIR, PROC_DATA_DIR
        process_raw_data(RAW_DATA_DIR, PROC_DATA_DIR, config=config, test_mode=args.test)
    
    elif args.cmd == 'train':
        train_model(
            model_type=args.model_type,
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    elif args.cmd == 'evaluate':
        metrics = evaluate_model(
            model_type=args.model_type,
            model_name=args.model_name
        )
    
    elif args.cmd == 'predict':
        name, conf = predict_image(
            model_type=args.model_type,
            image_path=args.image_path,
            model_name=args.model_name
        )
        print(f"Prediction: {name} (confidence: {conf:.2f})")
    
    elif args.cmd == 'check-gpu':
        print("GPU availability:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
    
    elif args.cmd == 'list-models':
        # List trained models in checkpoints directory
        model_dirs = list(CHECKPOINTS_DIR.glob('*'))
        if not model_dirs:
            print("No trained models found.")
            return 0
        
        print("\nAvailable trained models:")
        for model_dir in sorted(model_dirs):
            model_path = model_dir / 'best_model.pth'
            if model_path.exists():
                print(f"  {model_dir.name}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 