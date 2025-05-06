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
    """Main entry point for face recognition system.
    Built this to handle different commands through a unified interface.
    """
    parser = argparse.ArgumentParser(description='Face Recognition System')
    subparsers = parser.add_subparsers(dest='cmd', help='Command to run')
    
    # Interactive command 
    subparsers.add_parser('interactive', help='Run the interactive menu interface')
    
    # Preprocess command
    preproc = subparsers.add_parser('preprocess', help='Preprocess raw data')
    preproc.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    
    # Train command
    train_p = subparsers.add_parser('train', help='Train a model')
    train_p.add_argument('--model-type', type=str, required=True, 
                            choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid'],
                            help='Type of model to train')
    train_p.add_argument('--model-name', type=str, help='Name for the trained model')
    train_p.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    train_p.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_p.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_p.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    # Tune command - added this later when I realized manual hyperparameter tuning was taking too long
    tune_p = subparsers.add_parser('tune', help='Tune hyperparameters')
    tune_p.add_argument('--model-type', type=str, required=True, 
                           choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid'],
                           help='Type of model to tune')
    tune_p.add_argument('--n-trials', type=int, default=50, help='Number of hyperparameter trials')
    
    # Evaluate command
    eval_p = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_p.add_argument('--model-type', type=str, required=True, 
                           choices=['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid'],
                           help='Type of model to evaluate')
    eval_p.add_argument('--model-name', type=str, help='Name of the model to evaluate')
    
    # Predict command
    pred_p = subparsers.add_parser('predict', help='Predict on a single image')
    pred_p.add_argument('--model-type', type=str, required=True, 
                              choices=['baseline', 'cnn', 'attention', 'arcface', 'hybrid'],
                              help='Type of model to use (not siamese)')
    pred_p.add_argument('--model-name', type=str, help='Name of the model to use')
    pred_p.add_argument('--image-path', type=str, required=True, help='Path to the image to predict')
    
    # Check GPU command
    subparsers.add_parser('check-gpu', help='Check GPU availability')
    
    # List models command
    subparsers.add_parser('list-models', help='List available trained models')
    
    args = parser.parse_args()
    
    # If no command is provided, show help
    if args.cmd is None:
        parser.print_help()
        return 1
    
    # Execute the appropriate command
    if args.cmd == 'interactive':
        return interactive_menu()
    
    elif args.cmd == 'preprocess':
        # FIXME: sometimes face detection fails on certain images - need to add better error handling
        config = get_preprocessing_config()
        process_raw_data(config, test_mode=args.test)
    
    elif args.cmd == 'train':
        # training params I found work well after lots of experiments
        # smaller batch size tends to work better but is slower
        train_model(
            model_type=args.model_type,
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    elif args.cmd == 'tune':
        # Interactive selection of dataset path
        from .training import PROC_DATA_DIR
        
        # List available processed datasets
        proc_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
        if not proc_dirs:
            raise ValueError("No processed datasets found. Process raw data first.")
        
        print("\nAvailable processed datasets:")
        for i, d in enumerate(proc_dirs, 1):
            print(f"{i}. {d.name}")
        
        while True:
            dataset_choice = input("\nEnter dataset number to use for tuning: ")
            try:
                idx = int(dataset_choice) - 1
                if 0 <= idx < len(proc_dirs):
                    data_dir = proc_dirs[idx]
                    break
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # My old approach was manual grid search but Optuna is way better
        best_params = tune_hyperparameters(
            model_type=args.model_type,
            dataset_path=data_dir,
            n_trials=args.n_trials
        )
        print(f"Best hyperparameters: {best_params}")
    
    elif args.cmd == 'evaluate':
        # WARNING: evaluation on validation set can be optimistic
        # Always check test set metrics before drawing conclusions
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

# This was the original approach before I reorganized the code
# def list_datasets():
#     from .training import PROC_DATA_DIR
#     ds_dirs = list(PROC_DATA_DIR.glob('*'))
#     if not ds_dirs:
#         print("No processed datasets found.")
#         return
#     
#     print("\nAvailable datasets:")
#     for i, ds in enumerate(ds_dirs, 1):
#         if (ds / 'train').exists():
#             print(f"  {i}. {ds.name}")

if __name__ == '__main__':
    sys.exit(main()) 