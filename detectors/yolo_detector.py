from __future__ import annotations

import os
import tempfile
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# Try to import OpenCV, but make it optional for deployment
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

from .base import BoundingBox, DetectionResult, HelmetDetector
from utils.model_loader import load_yolo_model_safely


class YOLOv8Detector(HelmetDetector):
    """YOLOv8 detector for finding riders in images."""
    
    def __init__(self, model_path: str = "models/yolo.pt"):
        """Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to the trained YOLOv8 .pt file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOv8 model not found at {model_path}")
        
        # Load model with PyTorch 2.6 compatibility fixes
        self.model = load_yolo_model_safely(model_path)
        self.class_names = [
            'DHelmet',
            'DNoHelmet', 
            'DHelmetP1Helmet',
            'DNoHelmetP1NoHelmet',
        ]
    
    def predict(self, image: Image.Image) -> DetectionResult:
        """Detect riders in the image and crop them.
        
        Args:
            image: PIL Image to process
            
        Returns:
            DetectionResult with rider bounding boxes and cropped images
        """
        # Save image to temporary file for YOLO prediction
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image.save(tmp_file.name, 'JPEG')
            tmp_path = tmp_file.name
        
        try:
            # Run YOLO prediction (without confidence threshold as requested)
            results = self.model.predict(tmp_path)
            
            # Process results
            boxes = []
            cropped_riders = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class and score
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        label = self.class_names[class_id]
                        
                        # Create bounding box
                        bbox = BoundingBox(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            label=label,
                            score=confidence
                        )
                        boxes.append(bbox)
                        
                        # Crop the rider region
                        cropped = image.crop((x1, y1, x2, y2))
                        cropped_riders.append(cropped)
            
            # Calculate overall confidence as average of all detections
            overall_confidence = np.mean([box.score for box in boxes]) * 100 if boxes else 0
            
            # Updated counting logic for YOLO (zero-indexed)
            helmet_count = 0
            no_helmet_count = 0
            for b in boxes:
                # Get the class index from the label
                try:
                    class_idx = self.class_names.index(b.label)
                    if class_idx == 0:  # DHelmet
                        helmet_count += 1
                    elif class_idx == 1:  # DNoHelmet
                        no_helmet_count += 1
                    elif class_idx == 2:  # DHelmetP1Helmet
                        helmet_count += 2
                    elif class_idx == 3:  # DNoHelmetP1NoHelmet
                        no_helmet_count += 2
                except ValueError:
                    # Fallback to string-based counting if label not found
                    name = b.label.lower()
                    if "nohelmet" in name or "no helmet" in name:
                        no_helmet_count += 1
                    elif "helmet" in name:
                        helmet_count += 1
            total_riders = len(boxes)
            
            # Determine overall label based on detected classes
            if total_riders == 0:
                overall_label = "No riders detected"
            else:
                overall_label = (
                    f"Detected {total_riders} riders ({helmet_count} with helmet, {no_helmet_count} without)"
                )
            
            return DetectionResult(
                label=overall_label,
                confidence=overall_confidence,
                boxes=boxes,
                raw={
                    'cropped_riders': cropped_riders,
                    'class_names': self.class_names,
                    'helmet_count': helmet_count,
                    'no_helmet_count': no_helmet_count,
                    'total_riders': total_riders,
                }
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
