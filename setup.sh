#!/bin/bash

# Streamlit Cloud deployment setup script
# This script installs dependencies and sets up the environment

echo "🚀 Setting up Helmet Detection System for Streamlit Cloud..."

# Update pip to latest version
echo "📦 Updating pip..."
pip install --upgrade pip

# Install dependencies from deployment requirements
echo "📋 Installing deployment dependencies..."
pip install -r requirements-deploy.txt

# Create models directory if it doesn't exist
echo "📁 Creating models directory..."
mkdir -p models

# Set environment variables for deployment
echo "🔧 Setting environment variables..."
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Verify installation
echo "✅ Verifying installation..."
python -c "
try:
    import streamlit as st
    import torch
    import tensorflow as tf
    print('✅ Core dependencies installed successfully')
    
    # Check available detectors
    from detectors import RCNNHelmetDetector, SSDHelmetDetector, DETRHelmetDetector
    print('✅ RCNN, SSD, and DETR detectors available')
    
    # Check if YOLO is available (should not be in deployment)
    try:
        from detectors import YOLOv8Detector
        print('⚠️  YOLO detector available (unexpected in deployment)')
    except ImportError:
        print('✅ YOLO detector correctly unavailable (as expected)')
        
    print('🎉 Setup completed successfully!')
    
except Exception as e:
    print(f'❌ Setup failed: {e}')
    exit(1)
"

echo "🎯 Deployment setup complete!"
echo "📊 Available detectors: RCNN, SSD, DETR"
echo "⚠️  YOLO and Hybrid detectors not available (ultralytics excluded for deployment)"
