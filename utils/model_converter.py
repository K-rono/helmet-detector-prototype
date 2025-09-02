"""
Utility for converting TensorFlow models to compatible formats.
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image


def convert_model_to_compatible_format(input_path: str, output_path: str = None):
    """
    Convert a TensorFlow model to a compatible format.
    
    Args:
        input_path: Path to the input model
        output_path: Path for the converted model (optional)
    
    Returns:
        Path to the converted model
    """
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_converted.keras"
    
    print(f"Converting model from {input_path} to {output_path}...")
    
    try:
        # Try to load the model with various strategies
        model = None
        
        # Strategy 1: Try with keras import
        try:
            import keras
            model = keras.models.load_model(input_path, compile=False)
            print("✅ Loaded model using keras import")
        except:
            pass
        
        # Strategy 2: Try with tensorflow.keras
        if model is None:
            try:
                model = tf.keras.models.load_model(input_path, compile=False)
                print("✅ Loaded model using tensorflow.keras")
            except:
                pass
        
        # Strategy 3: Try with custom objects
        if model is None:
            try:
                custom_objects = {
                    'Functional': tf.keras.Model,
                    'Sequential': tf.keras.Sequential
                }
                model = tf.keras.models.load_model(input_path, custom_objects=custom_objects, compile=False)
                print("✅ Loaded model using custom objects")
            except:
                pass
        
        if model is None:
            raise Exception("Could not load the model with any strategy")
        
        # Save the model in a compatible format
        model.save(output_path, save_format='keras')
        print(f"✅ Model converted and saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        raise e


def create_simple_vgg16_model():
    """
    Create a simple VGG16-like model for testing purposes.
    
    Returns:
        A simple VGG16 model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: helmet, no_helmet
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_test_model():
    """
    Create a test model for debugging purposes.
    
    Returns:
        A simple test model
    """
    model = create_simple_vgg16_model()
    
    # Save the test model
    test_path = "models/test_vgg16.keras"
    model.save(test_path)
    print(f"✅ Test model created and saved to {test_path}")
    
    return test_path


if __name__ == "__main__":
    # Test the converter
    print("Testing model converter...")
    
    # Create a test model first
    test_model_path = create_test_model()
    
    # Try to convert it
    try:
        converted_path = convert_model_to_compatible_format(test_model_path)
        print(f"✅ Conversion successful: {converted_path}")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
