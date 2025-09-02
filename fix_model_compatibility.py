#!/usr/bin/env python3
"""
Comprehensive script to fix TensorFlow model compatibility issues.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.model_converter import convert_model_to_compatible_format, create_test_model
from utils.model_loader import load_tensorflow_model_safely


def diagnose_model_issue(model_path: str):
    """Diagnose the model loading issue."""
    
    print(f"🔍 Diagnosing model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.2f} MB")
    
    # Try to load with different strategies
    strategies = [
        ("Direct load", lambda: tf.keras.models.load_model(model_path)),
        ("No compilation", lambda: tf.keras.models.load_model(model_path, compile=False)),
        ("Keras import", lambda: __import__('keras').models.load_model(model_path, compile=False)),
        ("Custom objects", lambda: tf.keras.models.load_model(model_path, custom_objects={'Functional': tf.keras.Model}, compile=False)),
    ]
    
    for name, strategy in strategies:
        try:
            print(f"🔄 Trying {name}...")
            model = strategy()
            print(f"✅ {name} succeeded!")
            print(f"   Model type: {type(model)}")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            return True
        except Exception as e:
            print(f"❌ {name} failed: {str(e)[:100]}...")
    
    return False


def fix_model_compatibility(model_path: str):
    """Fix model compatibility issues."""
    
    print(f"🔧 Fixing model compatibility: {model_path}")
    
    try:
        # Try to convert the model
        converted_path = convert_model_to_compatible_format(model_path)
        
        # Test the converted model
        print("🧪 Testing converted model...")
        if diagnose_model_issue(converted_path):
            print(f"✅ Model successfully converted to: {converted_path}")
            return converted_path
        else:
            print("❌ Converted model still has issues")
            return None
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None


def create_backup_model():
    """Create a backup model for testing."""
    
    print("🛠️ Creating backup test model...")
    
    try:
        test_path = create_test_model()
        print(f"✅ Backup model created: {test_path}")
        return test_path
    except Exception as e:
        print(f"❌ Failed to create backup model: {e}")
        return None


def main():
    """Main function to fix model compatibility."""
    
    print("🔧 TensorFlow Model Compatibility Fixer")
    print("=" * 50)
    
    model_path = "models/vgg16.keras"
    
    # Step 1: Diagnose the issue
    print("\n📋 Step 1: Diagnosing the issue...")
    if diagnose_model_issue(model_path):
        print("✅ Model loads successfully! No fix needed.")
        return
    
    # Step 2: Try to fix the model
    print("\n🔧 Step 2: Attempting to fix the model...")
    converted_path = fix_model_compatibility(model_path)
    
    if converted_path:
        print(f"\n🎉 Success! Use the converted model: {converted_path}")
        print("Update your VGG16 classifier to use this path:")
        print(f"vgg_classifier = VGG16HelmetClassifier('{converted_path}')")
    else:
        print("\n⚠️ Could not fix the model automatically.")
        print("\n📝 Manual solutions:")
        print("1. Re-save your model with the current TensorFlow version")
        print("2. Use the same TensorFlow version that was used to train the model")
        print("3. Convert your model to SavedModel format")
        print("4. Use a different model format (.h5, .pb, etc.)")
        
        # Create a backup model for testing
        print("\n🛠️ Creating a backup test model...")
        backup_path = create_backup_model()
        if backup_path:
            print(f"✅ Backup model created: {backup_path}")
            print("You can use this for testing while fixing your main model.")


if __name__ == "__main__":
    import tensorflow as tf
    main()
