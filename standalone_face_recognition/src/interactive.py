#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import torch

from .base_config import (
    logger, CHECKPOINTS_DIR, PROC_DATA_DIR, 
    get_user_confirmation
)
from .data_prep import get_preprocessing_config, process_raw_data
from .training import train_model, tune_hyperparameters
from .testing import evaluate_model, predict_image
from .face_models import get_model

def interactive_menu():
    """Interactive interface for the face recognition system."""
    # Add a command line parser for test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running model tests...")
        print("Tests are now located in the tests directory.")
        print("Please run: python -m unittest discover -s tests")
        sys.exit(0)
    
    while True:
        print("\nFace Recognition System")
        print("1. Process Raw Data")
        print("2. Train Model")
        print("3. Evaluate Model")
        print("4. Tune Hyperparameters")
        print("5. List Processed Datasets")
        print("6. List Trained Models")
        print("7. Run Tests")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == '1':
            print("\nData Processing")
            if not get_user_confirmation("This will create a new preprocessed dataset. Continue? (y/n): "):
                continue
            
            config = get_preprocessing_config()
            if get_user_confirmation("Start processing? (y/n): "):
                processed_dir = process_raw_data(config)
                print(f"\nProcessed data saved in: {processed_dir}")
        
        elif choice == '2':
            print("\nModel Training")
            print("Available model types:")
            print("- baseline: Simple CNN architecture")
            print("- cnn: ResNet18 transfer learning")
            print("- siamese: Siamese network for verification")
            print("- attention: ResNet with attention mechanism")
            print("- arcface: Face recognition with ArcFace loss")
            print("- hybrid: CNN-Transformer hybrid architecture")
            
            model_type = input("Enter model type: ")
            if model_type.lower() not in ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid']:
                print("Invalid model type")
                continue
            
            # List available processed datasets
            processed_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
            if not processed_dirs:
                print("No processed datasets found. Please process raw data first.")
                continue
            
            print("\nAvailable processed datasets:")
            for i, d in enumerate(processed_dirs, 1):
                print(f"{i}. {d.name}")
                # Try to load and display config info
                config_file = d / "preprocessing_config.json"
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                        print(f"   - MTCNN: {config.get('use_mtcnn', 'N/A')}")
                        print(f"   - Face Margin: {config.get('face_margin', 'N/A')}")
                        print(f"   - Image Size: {config.get('final_size', 'N/A')}")
                    except:
                        pass
            
            while True:
                dataset_choice = input("\nEnter dataset number to use for training: ")
                try:
                    dataset_idx = int(dataset_choice) - 1
                    if 0 <= dataset_idx < len(processed_dirs):
                        selected_data_dir = processed_dirs[dataset_idx]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            model_name = input("Enter model name (optional, press Enter for automatic versioning): ")
            if not model_name:
                model_name = None
            
            epochs = int(input("Enter number of epochs (default 50): ") or "50")
            batch_size = int(input("Enter batch size (default 32): ") or "32")
            lr = float(input("Enter learning rate (default 0.001): ") or "0.001")
            
            if get_user_confirmation("Start training? (y/n): "):
                trained_model_name = train_model(
                    model_type, model_name, batch_size, epochs, lr=lr
                )
                print(f"\nModel trained and saved as: {trained_model_name}")
        
        elif choice == '3':
            print("\nModel Evaluation")
            model_type = input("Enter model type (baseline/cnn/siamese/attention/arcface/hybrid): ")
            if model_type.lower() not in ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid']:
                print("Invalid model type")
                continue
            
            # List available models of this type
            model_dirs = list(CHECKPOINTS_DIR.glob(f'{model_type}_*'))
            if not model_dirs:
                print(f"No trained models found for type: {model_type}")
                continue
            
            print("\nAvailable models:")
            for i, model_dir in enumerate(model_dirs, 1):
                print(f"{i}. {model_dir.name}")
            
            while True:
                model_choice = input("\nEnter model number (or press Enter for latest): ")
                if not model_choice:
                    model_name = None
                    break
                try:
                    model_idx = int(model_choice) - 1
                    if 0 <= model_idx < len(model_dirs):
                        model_name = model_dirs[model_idx].name
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            try:
                evaluate_model(model_type, model_name)
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
        
        elif choice == '4':
            print("\nHyperparameter Tuning")
            print("Available model types:")
            print("- baseline: Simple CNN architecture")
            print("- cnn: ResNet18 transfer learning")
            print("- siamese: Siamese network for verification")
            print("- attention: ResNet with attention mechanism")
            print("- arcface: Face recognition with ArcFace loss")
            print("- hybrid: CNN-Transformer hybrid architecture")
            
            model_type = input("Enter model type: ")
            if model_type.lower() not in ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid']:
                print("Invalid model type")
                continue
            
            # List available processed datasets
            processed_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and (d / "train").exists()]
            if not processed_dirs:
                print("No processed datasets found. Please process raw data first.")
                continue
            
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
            
            n_trials = int(input("Enter number of trials (default 50): ") or "50")
            
            if get_user_confirmation("Start hyperparameter tuning? (y/n): "):
                best_params = tune_hyperparameters(model_type, selected_data_dir, n_trials)
                print("\nWould you like to train a model with these parameters?")
                if get_user_confirmation("Train model with best parameters? (y/n): "):
                    model_name = f"{model_type}_tuned"
                    epochs = int(input("Enter number of epochs (default 50): ") or "50")
                    trained_model_name = train_model(
                        model_type, model_name, 
                        batch_size=best_params['batch_size'],
                        epochs=epochs,
                        lr=best_params['lr']
                    )
                    print(f"\nModel trained and saved as: {trained_model_name}")
                    
        elif choice == '5':
            print("\nProcessed Datasets:")
            processed_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir()]
            if not processed_dirs:
                print("No processed datasets found")
            else:
                for d in processed_dirs:
                    print(f"- {d.name}")
                    # Try to load and display config info
                    config_file = d / "preprocessing_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file) as f:
                                config = json.load(f)
                            print(f"   - MTCNN: {config.get('use_mtcnn', 'N/A')}")
                            print(f"   - Face Margin: {config.get('face_margin', 'N/A')}")
                            print(f"   - Image Size: {config.get('final_size', 'N/A')}")
                        except:
                            pass
        
        elif choice == '6':
            print("\nTrained Models:")
            model_dirs = list(CHECKPOINTS_DIR.glob('*'))
            if not model_dirs:
                print("No trained models found")
            else:
                for model_dir in sorted(model_dirs):
                    if model_dir.is_dir():
                        print(f"- {model_dir.name}")
        
        elif choice == '7':
            print("\nRunning Tests...")
            print("Please run the tests using unittest from the command line:")
            print("python -m unittest discover -s tests")
            
        elif choice == '8':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...") 