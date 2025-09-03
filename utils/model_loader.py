"""
Utility functions for loading models with PyTorch 2.6 compatibility.
"""

import torch
import tensorflow as tf

# Try to import ultralytics, but make it optional for deployment
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None


def load_yolo_model_safely(model_path: str):
    """
    Load YOLOv8 model with PyTorch 2.6 compatibility fixes.
    
    Args:
        model_path: Path to the YOLOv8 .pt model file
        
    Returns:
        Loaded YOLO model
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "YOLOv8 model loading requires ultralytics package. "
            "Install with: pip install ultralytics"
        )
    
    # Fix for PyTorch 2.6 compatibility - add safe globals for YOLO models
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except ImportError:
        pass
    
    # Try loading normally first
    try:
        return YOLO(model_path)
    except Exception as e:
        if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
            # Try loading with explicit weights_only=False
            print("Attempting to load YOLO model with weights_only=False for PyTorch 2.6 compatibility...")
            
            # Create a context manager to temporarily patch torch.load
            original_load = torch.load
            
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            # Temporarily replace torch.load
            torch.load = safe_load
            try:
                model = YOLO(model_path)
                return model
            finally:
                # Restore original torch.load
                torch.load = original_load
        else:
            raise e


def load_tensorflow_model_safely(model_path: str):
    """
    Load TensorFlow/Keras model with version compatibility fixes.
    
    Args:
        model_path: Path to the TensorFlow .keras model file
        
    Returns:
        Loaded TensorFlow model
    """
    try:
        # Try loading normally first
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        if "Functional" in str(e) or "keras.src.models.functional" in str(e):
            print("Attempting to load TensorFlow model with compatibility fixes...")
            
            # Strategy 1: Try loading with compile=False (most common fix)
            try:
                print("Trying strategy 1: Load without compilation...")
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Successfully loaded model using strategy 1")
                return model
            except Exception as e1:
                print(f"Strategy 1 failed: {str(e1)}")
            
            # Strategy 2: Try with custom_objects to handle version differences
            try:
                print("Trying strategy 2: Load with custom_objects...")
                # Create a mapping for the missing Functional class
                custom_objects = {
                    'Functional': tf.keras.Model
                }
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                print("✅ Successfully loaded model using strategy 2")
                return model
            except Exception as e2:
                print(f"Strategy 2 failed: {str(e2)}")
            
            # Strategy 3: Try loading as SavedModel (if it's actually a SavedModel)
            try:
                print("Trying strategy 3: Load as SavedModel...")
                model = tf.saved_model.load(model_path)
                print("✅ Successfully loaded model using strategy 3")
                return model
            except Exception as e3:
                print(f"Strategy 3 failed: {str(e3)}")
            
            # Strategy 4: Try with different Keras import
            try:
                print("Trying strategy 4: Load with keras import...")
                import keras
                model = keras.models.load_model(model_path, compile=False)
                print("✅ Successfully loaded model using strategy 4")
                return model
            except Exception as e4:
                print(f"Strategy 4 failed: {str(e4)}")
            
            # Strategy 5: Try to patch the import issue temporarily
            try:
                print("Trying strategy 5: Patch import issue...")
                import sys
                import types
                
                # Create a dummy module to satisfy the import
                dummy_module = types.ModuleType('keras.src.models.functional')
                sys.modules['keras.src.models.functional'] = dummy_module
                
                # Add the Functional class to the dummy module
                dummy_module.Functional = tf.keras.Model
                
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Successfully loaded model using strategy 5")
                return model
            except Exception as e5:
                print(f"Strategy 5 failed: {str(e5)}")
            
            # If all strategies fail, provide helpful error message
            print("All loading strategies failed.")
            print("\nPossible solutions:")
            print("1. Convert your model to a compatible format")
            print("2. Use the same TensorFlow version that was used to train the model")
            print("3. Re-save the model with the current TensorFlow version")
            raise e
        else:
            raise e
