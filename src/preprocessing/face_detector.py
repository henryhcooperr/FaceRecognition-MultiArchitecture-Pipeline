#!/usr/bin/env python3
"""
Face detection and alignment module using MTCNN.
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from pathlib import Path
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, min_face_size: int = 20):
        """
        Initialize the face detector.
        
        Args:
            min_face_size: Minimum size of face to detect (in pixels)
        """
        self.detector = MTCNN(min_face_size=min_face_size)
        
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Detect and align a face in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (aligned_face, face_info) if face detected, None otherwise
        """
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(rgb_image)
        
        if not faces:
            return None
            
        # Get the face with highest confidence
        face = max(faces, key=lambda x: x['confidence'])
        
        if face['confidence'] < 0.9:  # Confidence threshold
            return None
            
        # Extract face coordinates
        x, y, w, h = face['box']
        
        # Add margin to face box
        margin = int(0.2 * min(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        # Extract face region
        face_img = image[y1:y2, x1:x2]
        
        # Resize to standard size
        face_img = cv2.resize(face_img, (224, 224))
        
        return face_img, face
        
    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Process an image file to detect and align face.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Aligned face image if successful, None otherwise
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
                
            # Detect and align face
            result = self.detect_face(image)
            if result is None:
                logger.warning(f"No face detected in: {image_path}")
                return None
                
            face_img, _ = result
            return face_img
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None
            
    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for processed faces
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for img_path in input_path.glob('**/*.jpg'):
            # Create corresponding output path
            rel_path = img_path.relative_to(input_path)
            out_path = output_path / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process image
            face_img = self.process_image(str(img_path))
            if face_img is not None:
                cv2.imwrite(str(out_path), face_img)
                logger.info(f"Processed: {rel_path}")
            else:
                logger.warning(f"Skipped: {rel_path}") 