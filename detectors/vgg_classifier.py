from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

from .base import BoundingBox, DetectionResult, HelmetDetector
from utils.model_loader import load_tensorflow_model_safely


class VGG16HelmetClassifier(HelmetDetector):
    """VGG16 classifier for helmet detection on cropped rider images."""
    
    def __init__(self, model_path: str = "models/vgg16.keras"):
        """Initialize VGG16 helmet classifier.
        
        Args:
            model_path: Path to the trained VGG16 .keras file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VGG16 model not found at {model_path}")
        
        # Load model with TensorFlow/Keras compatibility fixes
        self.model = load_tensorflow_model_safely(model_path)
        self.input_size = (224, 224)  # VGG16 standard input size
        self.class_names = ["helmet", "no_helmet"]
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for VGG16 input.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        # Resize to 224x224
        image = image.resize(self.input_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_single(self, image: Image.Image) -> Tuple[str, float]:
        """Predict helmet status for a single cropped rider image.
        
        Args:
            image: Cropped rider image
            
        Returns:
            Tuple of (label, confidence)
        """
        # Preprocess image
        img_array = self.preprocess_image(image)
        
        # Get prediction
        prediction = self.model.predict(img_array, verbose=0)
        
        # Use argmax to get class (0=helmet, 1=no_helmet)
        class_id = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        label = self.class_names[class_id]
        
        return label, confidence
    
    def predict(self, image: Image.Image) -> DetectionResult:
        """This method is not used for VGG16 classifier as it works on cropped images.
        
        Use predict_single() for individual cropped rider images instead.
        """
        raise NotImplementedError(
            "VGG16 classifier works on cropped rider images. "
            "Use predict_single() method instead."
        )
