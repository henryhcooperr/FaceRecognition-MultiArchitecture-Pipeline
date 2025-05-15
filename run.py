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
from pathlib import Path
import argparse
from src.interactive import interactive_menu
from src.app import main as run_app
from src.hyperparameter_tuning import run_hyperparameter_tuning, MODEL_TYPES

def show_usage():
    print("Face Recognition Project - Simplified Version")
    print("Usage:")
    print("  python run.py             - Launch interactive menu")
    print("  python run.py interactive - Launch interactive menu")
    print("  python run.py demo        - Launch live demo app")
    print("  python run.py cv          - Run cross-validation")
    print("  python run.py hyperopt    - Run hyperparameter tuning")
    print("  python run.py help        - Show this help message")

def parse_args():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Hyperparameter tuning command
    hyperopt_parser = subparsers.add_parser('hyperopt', help='Run hyperparameter tuning')
    hyperopt_parser.add_argument('--model-type', type=str, required=True,
                               help=f'Type of model to tune (one of: {", ".join(MODEL_TYPES)})')
    hyperopt_parser.add_argument('--dataset', type=str, required=True,
                               help='Path to the processed dataset directory')
    hyperopt_parser.add_argument('--n-trials', type=int,
                               help='Number of trials to run (default: 20)')
    hyperopt_parser.add_argument('--timeout', type=int,
                               help='Timeout in seconds (optional)')
    hyperopt_parser.add_argument('--use-trial0-baseline', action='store_true',
                               help='Use trial-0 baseline for first trial')
    hyperopt_parser.add_argument('--keep-checkpoints', type=int, default=1,
                               help='Number of best checkpoints to keep per trial (default: 1)')
    hyperopt_parser.add_argument('--train-best', action='store_true',
                               help='Train a model with the best parameters after tuning')
    hyperopt_parser.add_argument('--epochs', type=int,
                               help='Number of epochs for training with best parameters (default: 50)')
    hyperopt_parser.add_argument('--use-lr-finder', action='store_true',
                               help='Use learning rate finder to determine optimal learning rates')
    
    # ArcFace-specific parameters
    hyperopt_parser.add_argument('--arcface-margin', type=float, default=0.5,
                               help='Margin parameter for ArcFace loss (default: 0.5)')
    hyperopt_parser.add_argument('--arcface-scale', type=float, default=32.0,
                               help='Scale parameter for ArcFace loss (default: 32.0)')
    hyperopt_parser.add_argument('--use-warmup', action='store_true',
                               help='Use learning rate warm-up')
    hyperopt_parser.add_argument('--warmup-epochs', type=int, default=5,
                               help='Number of warm-up epochs (default: 5)')
    hyperopt_parser.add_argument('--use-progressive-margin', action='store_true',
                               help='Use progressive margin strategy for ArcFace')
    hyperopt_parser.add_argument('--easy-margin', action='store_true',
                               help='Use easy margin for better initial convergence in ArcFace')
    hyperopt_parser.add_argument('--two-phase-training', action='store_true',
                               help='Use two-phase training for ArcFace (freeze backbone first, then fine-tune)')
    hyperopt_parser.add_argument('--clip-grad-norm', type=float,
                               help='Max norm for gradient clipping')

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
        # Parse arguments for hyperparameter tuning
        args = parse_args()
        if args.command == 'hyperopt':
            # Convert dataset path to Path object
            dataset_path = Path(args.dataset)
            if not dataset_path.exists():
                print(f"Error: Dataset path {dataset_path} does not exist")
                sys.exit(1)
            
            # Run hyperparameter tuning with parsed arguments
            results = run_hyperparameter_tuning(
                model_type=args.model_type,
                dataset_path=dataset_path,
                n_trials=args.n_trials,
                timeout=args.timeout,
                use_trial0_baseline=args.use_trial0_baseline,
                keep_checkpoints=args.keep_checkpoints
            )
            
            if results and args.train_best:
                from src.training import train_model
                
                # Create a good model name
                model_name = f"{args.model_type}_tuned_{dataset_path.name}"
                
                # Create a parameters dictionary for training with best parameters
                best_params = results['best_params']
                training_params = {
                    'model_type': args.model_type,
                    'model_name': model_name,
                    'batch_size': best_params.get('batch_size', 32),
                    'epochs': args.epochs or 50,
                    'lr': best_params.get('learning_rate', 0.001),
                    'weight_decay': best_params.get('weight_decay', 0.0001),
                    'scheduler_type': best_params.get('scheduler', 'cosine'),
                    'scheduler_patience': best_params.get('scheduler_patience', 5),
                    'scheduler_factor': best_params.get('scheduler_factor', 0.5),
                    'clip_grad_norm': best_params.get('clip_grad_norm', None) or args.clip_grad_norm,
                    'early_stopping': True,
                    'early_stopping_patience': best_params.get('early_stopping_patience', 10),
                    'dataset_path': dataset_path,
                    'use_lr_finder': args.use_lr_finder
                }
                
                # Add ArcFace-specific parameters
                if args.model_type == 'arcface':
                    # Add parameters from best_params
                    if 'arcface_margin' in best_params:
                        training_params['arcface_margin'] = best_params.get('arcface_margin', 0.5)
                    elif args.arcface_margin:
                        training_params['arcface_margin'] = args.arcface_margin
                        
                    if 'arcface_scale' in best_params:
                        training_params['arcface_scale'] = best_params.get('arcface_scale', 32.0)
                    elif args.arcface_scale:
                        training_params['arcface_scale'] = args.arcface_scale
                    
                    # Add warm-up parameters
                    training_params['use_warmup'] = best_params.get('use_lr_warmup', False) or args.use_warmup
                    training_params['warmup_epochs'] = best_params.get('warmup_epochs', 5) or args.warmup_epochs
                    
                    # Add progressive margin and easy margin
                    training_params['use_progressive_margin'] = best_params.get('use_progressive_margin', False) or args.use_progressive_margin
                    training_params['easy_margin'] = best_params.get('easy_margin', False) or args.easy_margin
                    
                    # Add two-phase training option
                    training_params['two_phase_training'] = best_params.get('two_phase_training', False) or args.two_phase_training
                
                # Print the training parameters
                print("\nTraining with parameters:")
                for param, value in training_params.items():
                    if param != 'dataset_path':  # Skip lengthy dataset path
                        print(f"- {param}: {value}")
                
                # Train the model with the best parameters
                trained_model_name = train_model(**training_params)
                print(f"\nModel trained and saved as: {trained_model_name}")
    elif sys.argv[1] == "help" or sys.argv[1] == "--help" or sys.argv[1] == "-h":
        show_usage()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        show_usage()
        sys.exit(1) 