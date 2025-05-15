#!/usr/bin/env python3

"""
Face Recognition Visualization Script.
Run this to visualize trained models and generate interactive reports.
"""

import sys
from src.cli import main

if __name__ == "__main__":
    sys.argv[0] = "visualize"
    main()