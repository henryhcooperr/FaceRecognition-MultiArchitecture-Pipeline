#!/usr/bin/env python
"""
A wrapper script to run the Streamlit app with proper settings to avoid
the PyTorch custom class watch issue and provide enhanced face recognition.
"""
import os
import sys
import subprocess
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the face recognition Streamlit app with optimized settings")
    parser.add_argument("--browser", action="store_true", help="Automatically open the app in a browser")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the Streamlit app on (default: 8501)")
    args = parser.parse_args()

    # Make sure the .streamlit config directory exists
    os.makedirs(".streamlit", exist_ok=True)

    # Ensure the config.toml file exists with proper settings
    config_path = ".streamlit/config.toml"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write("""
[server]
# Disable the watchdog to prevent PyTorch class path issues
fileWatcherType = "none"

[runner]
# Increase timeouts for better stability
fastReruns = true

[theme]
# Enhanced UI theme for better usability
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
""")
        print(f"Created {config_path} to fix PyTorch class watch issue and enhance UI")

    # Make sure the face_references directory exists
    os.makedirs("face_references", exist_ok=True)

    # Set environment variables to optimize performance
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
    os.environ['OMP_NUM_THREADS'] = '4'  # Optimize OpenMP threading for better performance
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix for Intel MKL library conflict

    # Build Streamlit command
    cmd = [sys.executable, "-m", "streamlit", "run", "src/app.py", 
          f"--server.port={args.port}", 
          "--server.fileWatcherType=none"]
    
    # Add browser option if requested
    if not args.browser:
        cmd.append("--server.headless=true")
    
    print(f"Starting Streamlit app on port {args.port} with optimized settings...")
    print("Face recognition features:")
    print(" - Auto-prompt for unrecognized faces")
    print(" - Face tracking between frames")
    print(" - Edit and manage known faces")
    print(" - Recognition history tracking")
    
    # Run the Streamlit app
    subprocess.run(cmd)

if __name__ == "__main__":
    main()