#!/bin/bash

# Advanced Streamlit Cloud deployment setup script
# This script provides more control over the deployment process

echo "🚀 Advanced Setup for Helmet Detection System..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect the environment
echo "🔍 Detecting environment..."
if [ -n "$STREAMLIT_SHARING_MODE" ]; then
    echo "📱 Streamlit Cloud environment detected"
    DEPLOYMENT_MODE="streamlit"
elif [ -n "$HEROKU_APP_NAME" ]; then
    echo "🟣 Heroku environment detected"
    DEPLOYMENT_MODE="heroku"
else
    echo "💻 Local/other environment detected"
    DEPLOYMENT_MODE="local"
fi

# Update pip
echo "📦 Updating pip..."
pip install --upgrade pip

# Install dependencies based on deployment mode
case $DEPLOYMENT_MODE in
    "streamlit"|"heroku")
        echo "☁️ Installing deployment dependencies..."
        pip install -r requirements-deploy.txt
        ;;
    "local")
        echo "🏠 Installing local development dependencies..."
        pip install -r requirements-local.txt
        ;;
esac

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p logs

# Set environment variables
echo "🔧 Setting environment variables..."
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=${PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create a simple health check script
echo "🏥 Creating health check..."
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
"""Health check script for deployment verification"""

import sys
import os

def main():
    print("🔍 Running health check...")
    
    try:
        # Test core imports
        import streamlit as st
        import torch
        import tensorflow as tf
        print("✅ Core dependencies: OK")
        
        # Test detector imports
        from detectors import RCNNHelmetDetector, SSDHelmetDetector, DETRHelmetDetector
        print("✅ Available detectors: RCNN, SSD, DETR")
        
        # Test YOLO availability (should fail in deployment)
        try:
            from detectors import YOLOv8Detector
            print("⚠️  YOLO detector: Available (unexpected in deployment)")
        except ImportError:
            print("✅ YOLO detector: Correctly unavailable (as expected)")
        
        # Test model directory
        if os.path.exists("models"):
            print("✅ Models directory: OK")
        else:
            print("❌ Models directory: Missing")
            return 1
        
        print("🎉 Health check passed!")
        return 0
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Run health check
echo "🏥 Running health check..."
python health_check.py
HEALTH_CHECK_RESULT=$?

# Clean up health check file
rm -f health_check.py

# Final status
if [ $HEALTH_CHECK_RESULT -eq 0 ]; then
    echo ""
    echo "🎉 Deployment setup completed successfully!"
    echo "📊 Available detectors: RCNN, SSD, DETR"
    echo "⚠️  YOLO and Hybrid detectors not available (ultralytics excluded for deployment)"
    echo "🚀 Ready to start the application!"
else
    echo ""
    echo "❌ Deployment setup failed!"
    echo "Please check the error messages above."
    exit 1
fi
