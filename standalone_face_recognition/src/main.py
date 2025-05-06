#!/usr/bin/env python3

import argparse
import sys
import torch
from pathlib import Path

from .base_config import logger, CHECKPOINTS_DIR
from .data_prep import get_preprocessing_config, process_raw_data
from .training import train_model, tune_hyperparameters
from .testing import evaluate_model, predict_image
from .face_models import get_model
from .visualize import plot_tsne_embeddings, plot_attention_maps, plot_embedding_similarity
from .interactive import interactive_menu

def main():
    """Main entry point for the face recognition system."""
    parser = argparse.ArgumentParser(description='Face Recognition System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Run the interactive menu interface')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw data')
    preprocess_parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model-type', type=str, required=True, 
                            choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid'],
                            help='Type of model to train')
    train_parser.add_argument('--model-name', type=str, help='Name for the trained model')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Tune hyperparameters')
    tune_parser.add_argument('--model-type', type=str, required=True, 
                           choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid'],
                           help='Type of model to tune')
    tune_parser.add_argument('--n-trials', type=int, default=50, help='Number of hyperparameter trials')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model-type', type=str, required=True, 
                           choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid'],
                           help='Type of model to evaluate')
    eval_parser.add_argument('--model-name', type=str, help='Name of the model to evaluate')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on a single image')
    predict_parser.add_argument('--model-type', type=str, required=True, 
                              choices=['baseline', 'cnn', 'attention', 'arcface', 'hybrid'],
                              help='Type of model to use (not siamese)')
    predict_parser.add_argument('--model-name', type=str, help='Name of the model to use')
    predict_parser.add_argument('--image-path', type=str, required=True, help='Path to the image to predict')
    
    # Check GPU command
    subparsers.add_parser('check-gpu', help='Check GPU availability')
    
    # List models command
    subparsers.add_parser('list-models', help='List available trained models')
    
    args = parser.parse_args()
    
    # If no command is provided, show help
    if args.command is None:
        parser.print_help()
        return 1
    
    # Execute the appropriate command
    if args.command == 'interactive':
        return interactive_menu()
    
    elif args.command == 'preprocess':
        config = get_preprocessing_config()
        process_raw_data(config, test_mode=args.test)
    
    elif args.command == 'train':
        train_model(
            model_type=args.model_type,
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    elif args.command == 'tune':
        # Interactive selection of dataset path
        from .training import PROCESSED_DATA_DIR
        
        # List available processed datasets
        processed_dirs = [d for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
        if not processed_dirs:
            raise ValueError("No processed datasets found. Please process raw data first.")
        
        print("\nAvailable processed datasets:")
        for i, d in enumerate(processed_dirs, 1):
            print(f"{i}. {d.name}")
        
        while True:
            dataset_choice = input("\nEnter dataset number to use for tuning: ")
            try:
                dataset_idx = int(dataset_choice) - 1
                if 0 <= dataset_idx < len(processed_dirs):
                    selected_data_dir = processed_dirs[dataset_idx]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        best_params = tune_hyperparameters(
            model_type=args.model_type,
            dataset_path=selected_data_dir,
            n_trials=args.n_trials
        )
        print(f"Best hyperparameters: {best_params}")
    
    elif args.command == 'evaluate':
        metrics = evaluate_model(
            model_type=args.model_type,
            model_name=args.model_name
        )
    
    elif args.command == 'predict':
        class_name, confidence = predict_image(
            model_type=args.model_type,
            image_path=args.image_path,
            model_name=args.model_name
        )
        print(f"Prediction: {class_name} (confidence: {confidence:.2f})")
    
    elif args.command == 'check-gpu':
        print("GPU availability:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
    
    elif args.command == 'list-models':
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