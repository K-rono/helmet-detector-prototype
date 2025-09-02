from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image

from .base import BoundingBox, DetectionResult, HelmetDetector
from .yolo_detector import YOLOv8Detector
from .vgg_classifier import VGG16HelmetClassifier


class CombinedHelmetDetector(HelmetDetector):
    """Combined helmet detector using YOLOv8 + VGG16 pipeline."""
    
    def __init__(self, yolo_model_path: str = "models/yolo.pt", vgg_model_path: str = "models/vgg16.keras"):
        """Initialize combined helmet detector.
        
        Args:
            yolo_model_path: Path to YOLOv8 model
            vgg_model_path: Path to VGG16 model
        """
        self.yolo_detector = YOLOv8Detector(yolo_model_path)
        self.vgg_classifier = VGG16HelmetClassifier(vgg_model_path)
    
    def predict(self, image: Image.Image) -> DetectionResult:
        """Complete helmet detection pipeline.
        
        Args:
            image: PIL Image to process
            
        Returns:
            DetectionResult with helmet detection results and counts
        """
        # Step 1: Use YOLOv8 to detect riders and crop them
        yolo_result = self.yolo_detector.predict(image)
        
        if not yolo_result.boxes:
            return DetectionResult(
                label="No riders detected",
                confidence=0.0,
                boxes=[],
                raw={'helmet_count': 0, 'no_helmet_count': 0, 'total_riders': 0}
            )
        
        # Step 2: Use VGG16 to classify each cropped rider
        helmet_count = 0
        no_helmet_count = 0
        updated_boxes = []
        
        cropped_riders = yolo_result.raw.get('cropped_riders', [])
        
        for i, (box, cropped_rider) in enumerate(zip(yolo_result.boxes, cropped_riders)):
            # Classify the cropped rider image
            helmet_label, confidence = self.vgg_classifier.predict_single(cropped_rider)
            
            # Update counts
            if helmet_label == "helmet":
                helmet_count += 1
            else:
                no_helmet_count += 1
            
            # Update bounding box with VGG16 classification result
            updated_box = BoundingBox(
                x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2,
                label=f"Rider {i+1}: {helmet_label}",
                score=confidence
            )
            updated_boxes.append(updated_box)
        
        # Step 3: Create final result
        total_riders = len(updated_boxes)
        overall_confidence = np.mean([box.score for box in updated_boxes]) * 100 if updated_boxes else 0
        
        # Create summary label
        if total_riders == 0:
            overall_label = "No riders detected"
        else:
            overall_label = f"Detected {total_riders} riders: {helmet_count} with helmet, {no_helmet_count} without helmet"
        
        return DetectionResult(
            label=overall_label,
            confidence=overall_confidence,
            boxes=updated_boxes,
            raw={
                'helmet_count': helmet_count,
                'no_helmet_count': no_helmet_count,
                'total_riders': total_riders,
                'cropped_riders': cropped_riders
            }
        )
