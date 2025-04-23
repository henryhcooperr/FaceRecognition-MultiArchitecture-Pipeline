#!/usr/bin/env python3
"""
Real-time face recognition demo using webcam.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple

from models import BaselineNet, ResNetTransfer, SiameseNet
from preprocessing import FaceDetector, AugmentationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run face recognition demo')
    parser.add_argument('--model', type=str, required=True,
                      choices=['baseline', 'cnn', 'siamese'],
                      help='Model architecture to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--class-names', type=str, required=True,
                      help='Path to file containing class names')
    parser.add_argument('--camera-id', type=int, default=0,
                      help='Camera device ID')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                      help='Confidence threshold for predictions')
    return parser.parse_args()

def load_model(model_type: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model_type: One of 'baseline', 'cnn', or 'siamese'
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if model_type == 'baseline':
        model = BaselineNet()
    elif model_type == 'cnn':
        model = ResNetTransfer()
    elif model_type == 'siamese':
        model = SiameseNet()
    else:
        raise ValueError(f"Invalid model type: {model_type}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def load_class_names(file_path: str) -> List[str]:
    """
    Load class names from file.
    
    Args:
        file_path: Path to file containing class names
        
    Returns:
        List of class names
    """
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def process_frame(frame: np.ndarray, face_detector: FaceDetector,
                 model: torch.nn.Module, transform: AugmentationPipeline,
                 class_names: List[str], device: torch.device,
                 confidence_threshold: float) -> Tuple[np.ndarray, str, float]:
    """
    Process a single frame.
    
    Args:
        frame: Input frame
        face_detector: Face detector
        model: Recognition model
        transform: Image transformation
        class_names: List of class names
        device: Device to run inference on
        confidence_threshold: Confidence threshold
        
    Returns:
        Tuple of (processed frame, predicted class, confidence)
    """
    # Detect face
    face_img = face_detector.detect_face(frame)
    if face_img is None:
        return frame, "No face detected", 0.0
        
    # Convert BGR to RGB
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Apply transformation
    transformed = transform(face_img_rgb)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        if isinstance(model, SiameseNet):
            # For Siamese network, we need a reference image
            # For demo purposes, we'll use the same image
            out1, out2 = model(img_tensor, img_tensor)
            dist = torch.pairwise_distance(out1, out2)
            confidence = 1 - dist.item()
            pred_class = "Same person" if confidence > confidence_threshold else "Different person"
        else:
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            confidence = confidence.item()
            pred_class = class_names[pred_idx.item()]
            
    # Draw results on frame
    cv2.putText(frame, f"{pred_class} ({confidence:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    return frame, pred_class, confidence

def main():
    """Main demo function."""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, args.checkpoint, device)
    
    # Load class names
    class_names = load_class_names(args.class_names)
    
    # Initialize face detector and transform
    face_detector = FaceDetector()
    transform = AugmentationPipeline(phase='test')
    
    # Open webcam
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        sys.exit(1)
        
    logger.info("Starting demo... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            break
            
        # Process frame
        processed_frame, pred_class, confidence = process_frame(
            frame, face_detector, model, transform,
            class_names, device, args.confidence_threshold
        )
        
        # Display frame
        cv2.imshow('Face Recognition Demo', processed_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Demo completed")

if __name__ == "__main__":
    main() 