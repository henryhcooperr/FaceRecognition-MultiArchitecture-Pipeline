#!/usr/bin/env python3

import os
import shutil
import platform
import subprocess
import glob
from pathlib import Path

def get_disk_space(directory="."):
    """Get free disk space in GB"""
    if platform.system() == "Windows":
        free_bytes = shutil.disk_usage(directory).free
    else:
        free_bytes = os.statvfs(directory).f_frsize * os.statvfs(directory).f_bavail
    return free_bytes / (1024 ** 3)  # Convert to GB

def get_dir_size(directory):
    """Get directory size in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)  # Convert to GB

def find_checkpoint_files(directory="."):
    """Find large checkpoint files that could be candidates for cleanup"""
    checkpoint_paths = glob.glob(f"{directory}/**/checkpoint_*.pth", recursive=True)
    checkpoint_paths += glob.glob(f"{directory}/**/*.ckpt", recursive=True)
    
    checkpoint_sizes = []
    for path in checkpoint_paths:
        path_obj = Path(path)
        if path_obj.exists():
            size_mb = os.path.getsize(path) / (1024 ** 2)  # Size in MB
            checkpoint_sizes.append((path, size_mb))
    
    # Sort by size (largest first)
    checkpoint_sizes.sort(key=lambda x: x[1], reverse=True)
    return checkpoint_sizes

def find_temp_files():
    """Find temporary files that can be safely deleted"""
    root_dir = "."
    extensions = ['.tmp', '.temp', '.log', '.bak']
    temp_files = []
    
    for ext in extensions:
        for path in glob.glob(f"{root_dir}/**/*{ext}", recursive=True):
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 ** 2)  # Size in MB
                temp_files.append((path, size_mb))
    
    return temp_files

def clean_disk_space():
    """Interactive function to clean disk space"""
    print("Checking disk space...")
    free_space = get_disk_space()
    
    print(f"Free disk space: {free_space:.2f} GB")
    
    if free_space > 15:
        print("✓ You have sufficient disk space for the experiment (>15 GB).")
        return True
    
    print("\nWARNING: You have less than the recommended 15 GB of free space.")
    
    # Check for large checkpoint files
    print("\nSearching for large model checkpoint files...")
    checkpoints = find_checkpoint_files()
    
    if checkpoints:
        print(f"\nFound {len(checkpoints)} checkpoint files that could be deleted:")
        for i, (path, size) in enumerate(checkpoints[:10], 1):  # Show top 10
            print(f"{i}. {path} ({size:.2f} MB)")
        
        if len(checkpoints) > 10:
            print(f"...and {len(checkpoints) - 10} more")
        
        delete_choice = input("\nWould you like to delete some of these files to free up space? (y/n): ")
        if delete_choice.lower() == 'y':
            print("Enter the numbers of the files to delete (e.g., '1,3,5' or 'all'): ")
            file_nums = input()
            
            if file_nums.lower() == 'all':
                indices = range(len(checkpoints))
            else:
                indices = [int(x.strip()) - 1 for x in file_nums.split(',') if x.strip().isdigit()]
            
            space_freed = 0
            for i in indices:
                if 0 <= i < len(checkpoints):
                    path, size = checkpoints[i]
                    try:
                        os.remove(path)
                        print(f"Deleted: {path} ({size:.2f} MB)")
                        space_freed += size
                    except Exception as e:
                        print(f"Error deleting {path}: {str(e)}")
            
            print(f"\nFreed up approximately {space_freed/1024:.2f} GB")
    else:
        print("No large checkpoint files found.")
    
    # Check for temp files
    print("\nSearching for temporary files...")
    temp_files = find_temp_files()
    
    if temp_files:
        print(f"\nFound {len(temp_files)} temporary files that could be deleted:")
        total_temp_size = sum(size for _, size in temp_files) / 1024  # Convert to GB
        print(f"Total size: {total_temp_size:.2f} GB")
        
        delete_choice = input("\nWould you like to delete these temporary files? (y/n): ")
        if delete_choice.lower() == 'y':
            space_freed = 0
            for path, size in temp_files:
                try:
                    os.remove(path)
                    print(f"Deleted: {path} ({size:.2f} MB)")
                    space_freed += size
                except Exception as e:
                    print(f"Error deleting {path}: {str(e)}")
            
            print(f"\nFreed up approximately {space_freed/1024:.2f} GB")
    else:
        print("No temporary files found.")
    
    # Update free space after cleaning
    free_space = get_disk_space()
    print(f"\nUpdated free disk space: {free_space:.2f} GB")
    
    if free_space < 5:
        print("\n⚠️ WARNING: You still have less than 5 GB of free space.")
        print("This may not be enough to run the experiment successfully.")
        print("Consider freeing up more space or running a smaller portion of the experiment.")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        return proceed.lower() == 'y'
    
    print("\n✓ You have freed up enough space to proceed with the experiment.")
    return True

if __name__ == "__main__":
    if clean_disk_space():
        print("\nDisk space check completed. Ready to run the experiment.")
        print("Run the following command to rerun failed components:")
        print("\n    python rerun_failed_components.py --source comprehensive_experiment_20250507_170218\n")
    else:
        print("\nExperiment preparation canceled. Please free up more disk space before running the experiment.") 