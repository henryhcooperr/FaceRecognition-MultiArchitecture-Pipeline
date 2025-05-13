#!/usr/bin/env python3

import sys
import json
import subprocess
from pathlib import Path

import torch

from .base_config import (
    logger, CHECKPOINTS_DIR, PROC_DATA_DIR, RAW_DATA_DIR,
    get_user_confirmation
)
from .data_prep import get_preprocessing_config, process_raw_data, PreprocessingConfig, visualize_preprocessing_steps
from .training import train_model
from .testing import evaluate_model, predict_image
from .face_models import get_model
from .hyperparameter_tuning import run_hyperparameter_tuning, MODEL_TYPES
from .cross_validation import run_cross_validation
from . import download_dataset

# Get path to the downloader script
downloader_script = Path(__file__).parent / "download_dataset.py"

def check_and_download_datasets():
    """Check if datasets exist and download them if not."""
    # Check if any valid dataset directories exist
    dataset_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir() and d.name in ["dataset2", "dataset1"]]
    
    if dataset_dirs:
        print("\nExisting datasets found:")
        for d in dataset_dirs:
            info_file = d / "info.txt"
            if info_file.exists():
                print(f"- {d.name}")
                with open(info_file) as f:
                    for line in f:
                        if line.startswith("Description:") or line.startswith("Number of"):
                            print(f"  {line.strip()}")
        return True
    
    # No datasets found, automatically download all
    print("\nNo face recognition datasets found.")
    print("Automatically downloading all datasets...")
    try:
        # Call the download_all_datasets function directly
        success = download_dataset.download_all_datasets()

        if success:
            # Check if download was successful
            dataset_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
            if dataset_dirs:
                print("\nDatasets downloaded successfully:")
                for d in dataset_dirs:
                    info_file = d / "info.txt"
                    if info_file.exists():
                        print(f"- {d.name}")
                        with open(info_file) as f:
                            for line in f:
                                if line.startswith("Description:") or line.startswith("Number of"):
                                    print(f"  {line.strip()}")
                return True
            else:
                print("No datasets found after download.")
                return False
        else:
            print("Failed to download datasets.")
            return False
    except Exception as e:
        print(f"Error downloading datasets: {e}")
        return False

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
        print("2. Visualize Preprocessing")
        print("3. Train Model")
        print("4. Evaluate Model")
        print("5. Tune Hyperparameters")
        print("6. Cross-Validation")
        print("7. List Processed Datasets")
        print("8. List Trained Models")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ")
        
        if choice == '1':
            print("\nData Processing")
            
            # Check if raw data exists and download if needed
            raw_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
            if not raw_dirs:
                print("No raw data found. Downloading datasets automatically...")
                if not check_and_download_datasets():
                    print("Failed to download datasets. Please check your internet connection.")
                    continue
            
            # List available raw datasets for processing
            raw_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
            if raw_dirs:
                print("\nAvailable raw datasets:")
                for i, d in enumerate(raw_dirs, 1):
                    info_file = d / "info.txt"
                    if info_file.exists():
                        print(f"{i}. {d.name}")
                        with open(info_file) as f:
                            for line in f:
                                if line.startswith("Description:"):
                                    print(f"   {line.strip()}")
                    else:
                        print(f"{i}. {d.name}")
            
            if not get_user_confirmation("This will create a new preprocessed dataset. Continue? (y/n): "):
                continue
            
            config = get_preprocessing_config()
            if get_user_confirmation("Start processing? (y/n): "):
                processed_dir = process_raw_data(RAW_DATA_DIR, PROC_DATA_DIR, config)
                print(f"\nProcessed data saved in: {processed_dir}")
        
        elif choice == '2':
            print("\nVisualize Preprocessing")
            
            # Check if raw data exists and download if needed
            raw_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
            if not raw_dirs:
                print("No raw data found. Downloading datasets automatically...")
                if not check_and_download_datasets():
                    print("Failed to download datasets. Please check your internet connection.")
                    continue
            
            # List available raw datasets for processing
            raw_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
            if not raw_dirs:
                print("No raw datasets found.")
                continue
            
            print("\nAvailable raw datasets:")
            for i, d in enumerate(raw_dirs, 1):
                info_file = d / "info.txt"
                if info_file.exists():
                    print(f"{i}. {d.name}")
                    with open(info_file) as f:
                        for line in f:
                            if line.startswith("Description:"):
                                print(f"   {line.strip()}")
                else:
                    print(f"{i}. {d.name}")
            
            # Select dataset
            selected_dataset = None
            while True:
                dataset_choice = input("\nEnter dataset number to visualize: ")
                try:
                    dataset_idx = int(dataset_choice) - 1
                    if 0 <= dataset_idx < len(raw_dirs):
                        selected_dataset = raw_dirs[dataset_idx]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Get person directories
            person_dirs = [d for d in selected_dataset.iterdir() if d.is_dir()]
            if not person_dirs:
                print(f"No person directories found in {selected_dataset.name}")
                continue
            
            # Select a random person or let user choose
            use_random = get_user_confirmation("Select a random person? (y/n): ")
            
            selected_person = None
            if use_random:
                import random
                selected_person = random.choice(person_dirs)
                print(f"Randomly selected: {selected_person.name}")
            else:
                print("\nAvailable people:")
                for i, d in enumerate(person_dirs, 1):
                    print(f"{i}. {d.name}")
                
                while True:
                    person_choice = input("\nEnter person number: ")
                    try:
                        person_idx = int(person_choice) - 1
                        if 0 <= person_idx < len(person_dirs):
                            selected_person = person_dirs[person_idx]
                            break
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a valid number.")
            
            # Get images for the selected person
            image_files = list(selected_person.glob("*.jpg")) + list(selected_person.glob("*.png")) + list(selected_person.glob("*.jpeg"))
            if not image_files:
                print(f"No images found for {selected_person.name}")
                continue
            
            # Select random images or let user choose
            num_samples = int(input("Enter number of sample images to visualize (default 3): ") or "3")
            num_samples = min(num_samples, len(image_files))  # Don't try to visualize more than available
            
            import random
            sample_images = random.sample(image_files, num_samples)
            
            # Get preprocessing config
            print("\nConfiguring preprocessing parameters...")
            config = get_preprocessing_config()
            
            # Create visualization output directory
            OUTPUT_DIR = Path(CHECKPOINTS_DIR).parent  # Get the parent of CHECKPOINTS_DIR which should be the outputs folder
            viz_dir = OUTPUT_DIR / "preprocessing_visualization" / config.name
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Visualize preprocessing steps for each sample image
            print(f"\nVisualizing preprocessing steps for {num_samples} images...")
            for img_path in sample_images:
                print(f"Processing: {img_path.name}")
                visualize_preprocessing_steps(str(img_path), config, viz_dir)
            
            print(f"\nPreprocessing visualizations saved to: {viz_dir}")
            print(f"Generated {num_samples} visualization images.")
            print("You can find diagrams showing face detection, alignment, and augmentation.")
        
        elif choice == '3':
            print("\nModel Training")
            print("Available model types:")
            print("- baseline: Simple CNN architecture")
            print("- cnn: ResNet18 transfer learning")
            print("- siamese: Siamese network for verification")
            print("- attention: ResNet with attention mechanism")
            print("- arcface: Face recognition with ArcFace loss")
            print("- hybrid: CNN-Transformer hybrid architecture")
            print("- ensemble: Combination of multiple models")
            
            model_type = input("Enter model type: ")
            if model_type.lower() not in ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid', 'ensemble']:
                print("Invalid model type")
                continue
            
            # List available processed datasets
            processed_dirs = []
            dataset_names = set()  # For deduplication
            
            # Check for the standard organization: config_name/dataset_name/train
            config_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and not d.name in ["train", "val", "test"]]
            for config_dir in config_dirs:
                if (config_dir / "train").exists():
                    # Use this directory with its name
                    if config_dir.name not in dataset_names:
                        processed_dirs.append((config_dir, config_dir.name))
                        dataset_names.add(config_dir.name)
                else:
                    for dataset_dir in config_dir.iterdir():
                        if dataset_dir.is_dir() and (dataset_dir / "train").exists():
                            # Use config/dataset naming
                            dataset_path = f"{config_dir.name}/{dataset_dir.name}"
                            if dataset_path not in dataset_names:
                                processed_dirs.append((dataset_dir, dataset_path))
                                dataset_names.add(dataset_path)
            
            # Also check for simpler structure where train is directly in PROC_DATA_DIR
            if (PROC_DATA_DIR / "train").exists() and (PROC_DATA_DIR / "val").exists():
                if "root" not in dataset_names:
                    processed_dirs.append((PROC_DATA_DIR, "processed (root)"))
                    dataset_names.add("root")
            
            if not processed_dirs:
                print("No processed datasets found. Please process raw data first.")
                continue
            
            print("\nAvailable processed datasets:")
            for i, (dir_path, display_name) in enumerate(processed_dirs, 1):
                print(f"{i}. {display_name}")
            
            selected_data_dir = None
            while True:
                dataset_choice = input("\nEnter dataset number to use for training: ")
                try:
                    dataset_idx = int(dataset_choice) - 1
                    if 0 <= dataset_idx < len(processed_dirs):
                        selected_data_dir = processed_dirs[dataset_idx][0]  # Get the path part
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            model_name = input("Enter model name (optional, press Enter for automatic versioning): ")
            if not model_name:
                model_name = None
            
            # Basic parameters
            epochs = int(input("Enter number of epochs (default 50): ") or "50")
            batch_size = int(input("Enter batch size (default 32): ") or "32")
            
            # Ask user if they want to use the Learning Rate Finder
            use_lr_finder = get_user_confirmation("Use Learning Rate Finder to determine optimal learning rate? (y/n): ")
            
            # Only ask for learning rate if not using the LR finder
            lr = 0.001  # Default value
            if not use_lr_finder:
                lr = float(input("Enter learning rate (default 0.001): ") or "0.001")
            else:
                print("Learning rate will be determined automatically by the Learning Rate Finder.")
            
            weight_decay = float(input("Enter weight decay (default 0.0001): ") or "0.0001")
            
            # Learning rate scheduler options
            print("\nLearning Rate Scheduler:")
            print("1. ReduceLROnPlateau (reduces LR when metrics plateau)")
            print("2. CosineAnnealing (gradually reduces LR following a cosine curve)")
            print("3. StepLR (reduces LR by a factor every n epochs)")
            print("4. None (constant learning rate)")
            
            scheduler_choice = input("Select scheduler (1-4, default 1): ") or "1"
            scheduler_type = ['reduce_lr', 'cosine', 'step', 'none'][int(scheduler_choice) - 1]
            
            scheduler_patience = 5
            scheduler_factor = 0.5
            if scheduler_type in ['reduce_lr', 'step']:
                scheduler_patience = int(input(f"Enter patience for scheduler (default 5): ") or "5")
                scheduler_factor = float(input(f"Enter factor for scheduler (default 0.5): ") or "0.5")
            
            # Gradient clipping
            use_grad_clip = get_user_confirmation("Use gradient clipping? (y/n): ")
            clip_grad_norm = None
            if use_grad_clip:
                clip_grad_norm = float(input("Enter max gradient norm (default 1.0): ") or "1.0")
            
            # Early stopping
            early_stopping = get_user_confirmation("Use early stopping? (y/n): ")
            early_stopping_patience = 10
            if early_stopping:
                early_stopping_patience = int(input("Enter patience for early stopping (default 10): ") or "10")
            
            if get_user_confirmation("Start training? (y/n): "):
                # Import the function directly to avoid any shadowing issues
                from src.training import train_model as training_function

                trained_model_name = training_function(
                    model_type=model_type,
                    model_name=model_name,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    scheduler_type=scheduler_type,
                    scheduler_patience=scheduler_patience,
                    scheduler_factor=scheduler_factor,
                    clip_grad_norm=clip_grad_norm,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience,
                    dataset_path=selected_data_dir,  # Pass the selected dataset path
                    use_lr_finder=use_lr_finder  # Add the LR Finder option
                )
                print(f"\nModel trained and saved as: {trained_model_name}")
        
        elif choice == '4':
            print("\nModel Evaluation")
            model_type = input("Enter model type (baseline/cnn/siamese/attention/arcface/hybrid/ensemble): ")
            if model_type.lower() not in ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid', 'ensemble']:
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
        
        elif choice == '5':
            print("\nHyperparameter Tuning")
            print("Available model types:")
            for mt in MODEL_TYPES:
                print(f"- {mt}")
            
            model_type = input("Enter model type: ")
            if model_type.lower() not in MODEL_TYPES:
                print("Invalid model type")
                continue
            
            # List available processed datasets
            processed_dirs = []
            dataset_names = set()  # For deduplication
            
            # Check for the standard organization: config_name/dataset_name/train
            config_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and not d.name in ["train", "val", "test"]]
            for config_dir in config_dirs:
                if (config_dir / "train").exists():
                    # Use this directory with its name
                    if config_dir.name not in dataset_names:
                        processed_dirs.append((config_dir, config_dir.name))
                        dataset_names.add(config_dir.name)
                else:
                    for dataset_dir in config_dir.iterdir():
                        if dataset_dir.is_dir() and (dataset_dir / "train").exists():
                            # Use config/dataset naming
                            dataset_name = f"{config_dir.name}/{dataset_dir.name}"
                            if dataset_name not in dataset_names:
                                processed_dirs.append((dataset_dir, dataset_name))
                                dataset_names.add(dataset_name)
            
            # Also check for simpler structure where train is directly in PROC_DATA_DIR
            if (PROC_DATA_DIR / "train").exists() and (PROC_DATA_DIR / "val").exists():
                if "root" not in dataset_names:
                    processed_dirs.append((PROC_DATA_DIR, "processed (root)"))
                    dataset_names.add("root")
            
            if not processed_dirs:
                print("No processed datasets found. Please process raw data first.")
                continue
            
            print("\nAvailable processed datasets:")
            for i, (dir_path, display_name) in enumerate(processed_dirs, 1):
                print(f"{i}. {display_name}")
            
            while True:
                dataset_choice = input("\nEnter dataset number to use for tuning: ")
                try:
                    dataset_idx = int(dataset_choice) - 1
                    if 0 <= dataset_idx < len(processed_dirs):
                        selected_data_dir = processed_dirs[dataset_idx][0]  # Get the path part
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Get hyperparameter tuning options
            n_trials = int(input("Enter number of trials (default 20): ") or "20")
            timeout = input("Enter timeout in seconds (optional, press Enter for no timeout): ")
            timeout = int(timeout) if timeout else None
            
            use_trial0_baseline = get_user_confirmation("Use trial-0 baseline for first trial? (y/n): ")
            keep_checkpoints = int(input("Number of best checkpoints to keep per trial (default 1): ") or "1")
            
            # Add option for Learning Rate Finder
            use_lr_finder = get_user_confirmation("Use Learning Rate Finder to determine optimal learning rates? (y/n): ")
            
            if get_user_confirmation("Start hyperparameter tuning? (y/n): "):
                try:
                    # Call the run_hyperparameter_tuning function with all options
                    results = run_hyperparameter_tuning(
                        model_type=model_type,
                        dataset_path=selected_data_dir,
                        n_trials=n_trials,
                        timeout=timeout,
                        use_trial0_baseline=use_trial0_baseline,
                        keep_checkpoints=keep_checkpoints,
                        use_lr_finder=use_lr_finder
                    )
                    
                    if results:
                        print("\nHyperparameter tuning complete!")
                        print(f"Best accuracy: {results['best_accuracy']*100:.2f}%")
                        print(f"Best parameters: {results['best_params']}")
                        
                        # Ask if user wants to train a model with these parameters
                        if get_user_confirmation("\nTrain a model with these parameters? (y/n): "):
                            # Import the function directly to avoid any shadowing issues
                            from src.training import train_model as training_function

                            # Create a good model name
                            model_name = f"{model_type}_tuned_{selected_data_dir.name}"

                            # Set epochs to a reasonable value for full training
                            epochs = int(input(f"Enter number of epochs (default 50): ") or "50")

                            # Train the model with the best parameters
                            trained_model_name = training_function(
                                model_type=model_type,
                                model_name=model_name,
                                batch_size=results['best_params']['batch_size'],
                                epochs=epochs,
                                lr=results['best_params']['learning_rate'],
                                weight_decay=results['best_params']['weight_decay'],
                                scheduler_type=results['best_params'].get('scheduler', 'cosine'),
                                scheduler_patience=results['best_params'].get('scheduler_patience', 5),
                                scheduler_factor=results['best_params'].get('scheduler_factor', 0.5),
                                clip_grad_norm=results['best_params'].get('gradient_clip_val', None),  # Map gradient_clip_val to clip_grad_norm
                                early_stopping=True,
                                early_stopping_patience=results['best_params'].get('early_stopping_patience', 10),
                                dataset_path=selected_data_dir,
                                use_lr_finder=False  # Don't use LR finder since we already have optimal parameters
                            )
                            print(f"\nModel trained and saved as: {trained_model_name}")
                except Exception as e:
                    print(f"Error during hyperparameter tuning: {str(e)}")
        
        elif choice == '6':
            print("\nCross-Validation")
            print("Available model types:")
            print("- baseline: Simple CNN architecture")
            print("- cnn: ResNet18 transfer learning")
            print("- siamese: Siamese network for verification")
            print("- attention: ResNet with attention mechanism")
            print("- arcface: Face recognition with ArcFace loss")
            print("- hybrid: CNN-Transformer hybrid architecture")
            print("- ensemble: Combination of multiple models")
            
            model_type = input("Enter model type: ")
            if model_type.lower() not in ['baseline', 'cnn', 'siamese', 'attention', 'arcface', 'hybrid', 'ensemble']:
                print("Invalid model type")
                continue
            
            # Allow selecting an existing model
            existing_model = None
            # List available models of this type
            model_dirs = list(CHECKPOINTS_DIR.glob(f'{model_type}_*'))
            if model_dirs:
                print("\nAvailable trained models for this type:")
                print("0. (None - start from scratch)")
                for i, model_dir in enumerate(model_dirs, 1):
                    print(f"{i}. {model_dir.name}")
                
                while True:
                    model_choice = input("\nSelect an existing model as starting point (0 for none): ")
                    try:
                        model_idx = int(model_choice)
                        if model_idx == 0:
                            existing_model = None
                            break
                        elif 1 <= model_idx <= len(model_dirs):
                            existing_model = model_dirs[model_idx-1].name
                            break
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a valid number.")
                
            # List available processed datasets
            processed_dirs = []
            dataset_names = set()  # For deduplication
            
            # Check for the standard organization: config_name/dataset_name/train
            config_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and not d.name in ["train", "val", "test"]]
            for config_dir in config_dirs:
                if (config_dir / "train").exists():
                    # Use this directory with its name
                    if config_dir.name not in dataset_names:
                        processed_dirs.append((config_dir, config_dir.name))
                        dataset_names.add(config_dir.name)
                else:
                    for dataset_dir in config_dir.iterdir():
                        if dataset_dir.is_dir() and (dataset_dir / "train").exists():
                            # Use config/dataset naming
                            dataset_path = f"{config_dir.name}/{dataset_dir.name}"
                            if dataset_path not in dataset_names:
                                processed_dirs.append((dataset_dir, dataset_path))
                                dataset_names.add(dataset_path)
            
            # Also check for simpler structure where train is directly in PROC_DATA_DIR
            if (PROC_DATA_DIR / "train").exists() and (PROC_DATA_DIR / "val").exists():
                if "root" not in dataset_names:
                    processed_dirs.append((PROC_DATA_DIR, "processed (root)"))
                    dataset_names.add("root")
            
            if not processed_dirs:
                print("No processed datasets found. Please process raw data first.")
                continue
            
            print("\nAvailable processed datasets:")
            for i, (dir_path, display_name) in enumerate(processed_dirs, 1):
                print(f"{i}. {display_name}")
            
            selected_data_dir = None
            while True:
                dataset_choice = input("\nEnter dataset number to use for cross-validation: ")
                try:
                    dataset_idx = int(dataset_choice) - 1
                    if 0 <= dataset_idx < len(processed_dirs):
                        selected_data_dir = processed_dirs[dataset_idx][0]  # Get the path part
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            n_folds = int(input("Enter number of folds (default 5): ") or "5")
            
            if get_user_confirmation("Start cross-validation? (y/n): "):
                # Call the run_cross_validation function with the selected model type and dataset
                try:
                    if existing_model:
                        print(f"Using existing model as starting point: {existing_model}")
                    cv_results = run_cross_validation(model_type, selected_data_dir, n_folds, existing_model=existing_model)
                    print(f"\nCross-validation complete!")
                    print(f"Mean accuracy: {cv_results['mean_accuracy']*100:.2f}% ± {cv_results['std_accuracy']*100:.2f}%")
                except Exception as e:
                    print(f"Error during cross-validation: {str(e)}")
        
        elif choice == '7':
            print("\nProcessed Datasets:")
            
            # Look for processed datasets with various structures
            processed_dirs = []
            dataset_names = set()  # For deduplication
            
            # Check for the standard organization: config_name/dataset_name/train
            config_dirs = [d for d in PROC_DATA_DIR.iterdir() if d.is_dir() and not d.name in ["train", "val", "test"]]
            for config_dir in config_dirs:
                if (config_dir / "train").exists():
                    # Use this directory with its name
                    if config_dir.name not in dataset_names:
                        processed_dirs.append((config_dir, config_dir.name))
                        dataset_names.add(config_dir.name)
                else:
                    for dataset_dir in config_dir.iterdir():
                        if dataset_dir.is_dir() and (dataset_dir / "train").exists():
                            # Use config/dataset naming
                            dataset_path = f"{config_dir.name}/{dataset_dir.name}"
                            if dataset_path not in dataset_names:
                                processed_dirs.append((dataset_dir, dataset_path))
                                dataset_names.add(dataset_path)
            
            # Also check for simpler structure where train is directly in PROC_DATA_DIR
            if (PROC_DATA_DIR / "train").exists() and (PROC_DATA_DIR / "val").exists():
                if "root" not in dataset_names:
                    processed_dirs.append((PROC_DATA_DIR, "processed (root)"))
                    dataset_names.add("root")
            
            if not processed_dirs:
                print("No processed datasets found")
            else:
                for dir_path, display_name in processed_dirs:
                    print(f"- {display_name}")
                    # Try to load and display config info
                    config_file = dir_path / "preprocessing_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file) as f:
                                config = json.load(f)
                            print(f"   - MTCNN: {config.get('use_mtcnn', 'N/A')}")
                            print(f"   - Face Margin: {config.get('face_margin', 'N/A')}")
                            print(f"   - Image Size: {config.get('final_size', 'N/A')}")
                        except:
                            pass
        
        elif choice == '8':
            print("\nTrained Models:")
            model_dirs = list(CHECKPOINTS_DIR.glob('*'))
            if not model_dirs:
                print("No trained models found")
            else:
                for model_dir in sorted(model_dirs):
                    if model_dir.is_dir():
                        print(f"- {model_dir.name}")
        
        elif choice == '9':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 9.")
        
        input("\nPress Enter to continue...") 