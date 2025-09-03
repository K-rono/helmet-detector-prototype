# Helmet Detection System

A comprehensive helmet detection system supporting multiple state-of-the-art detection algorithms for intersection safety monitoring.

## Features

- **Multiple Detection Algorithms**: Choose from 5 different detection approaches:
  - **Hybrid (YOLO+VGG)**: Two-stage pipeline using YOLOv8 for rider detection + VGG16 for helmet classification
  - **YOLO Only**: Direct helmet detection using YOLOv8
  - **Faster R-CNN**: Detection with ResNet-50 backbone
  - **SSD**: Single Shot Detector with VGG16 backbone
  - **DETR**: Detection Transformer with ResNet-50 backbone
- **Real-time Performance**: Optimized for intersection monitoring with FPS tracking
- **Helmet Counting**: Counts total riders, riders with helmets, and riders without helmets
- **Web Interface**: Streamlit-based web application with model selection and visualization
- **Visualization**: Color-coded bounding boxes (green for helmet, red for no helmet)
- **Model Compatibility**: Built-in compatibility fixes for PyTorch 2.6+ and TensorFlow version mismatches

## Prerequisites

### System Requirements
- **Python**: 3.11 (Only tested on this version)
- **Operating System**: Windows 10/11
- **RAM**: Minimum 8GB (16GB+ recommended for multiple models)
- **Storage**: At least 5GB free space for models and dependencies
- **GPU**: Optional but recommended (CUDA 11.8+ for GPU acceleration)

### Hardware Recommendations
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: NVIDIA GPU with 4GB+ VRAM for optimal performance
- **Memory**: 16GB+ RAM for running multiple detection algorithms

## Setup Guide

### Step 1: Python Environment Setup

#### Option A: Using Python directly
```bash
# Check Python version (should be 3.11)
python --version

# If Python version is too old, install Python 3.11 from python.org
```

#### Option B: Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n helmet-detector python=3.11
conda activate helmet-detector
```

#### Option C: Using Virtual Environment
```bash
# Create virtual environment
python -m venv helmet-detector-env

# Activate environment
# On Windows:
helmet-detector-env\Scripts\activate
# On macOS/Linux:
source helmet-detector-env/bin/activate
```

### Step 2: Install Dependencies

#### For Local Development
```bash
# Install core dependencies (includes OpenCV)
pip install -r requirements-local.txt

# Verify installation
python -c "import torch, tensorflow, streamlit, ultralytics; print('All dependencies installed successfully!')"
```

#### For Streamlit Cloud Deployment
```bash
# Install deployment-friendly dependencies (excludes OpenCV)
pip install -r requirements-deploy.txt

# Verify installation
python -c "import torch, tensorflow, streamlit, ultralytics; print('All dependencies installed successfully!')"
```

**Note**: The deployment version excludes OpenCV to avoid `libGL.so.1` errors on cloud platforms. The application works perfectly without OpenCV as it uses PIL for visualization.

### Step 3: Verify GPU Support (Optional)
```bash
# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
```

### Step 4: Model Setup

#### Create Models Directory
```bash
# Create models directory if it doesn't exist
mkdir -p models
```

#### Download or Place Your Models
Place your trained models in the `models/` directory:

```
models/
├── yolo.pt          # YOLOv8 model for rider detection (REQUIRED)
├── vgg16.keras      # VGG16 model for helmet classification (Hybrid mode)
├── rcnn.pth         # Faster R-CNN model (optional)
├── ssd.pth          # SSD model (optional)
└── detr.pth         # DETR model (optional)
```

**Model Requirements:**
- **YOLOv8 (.pt)**: Should detect 4 classes: `['DHelmet', 'DNoHelmet', 'DHelmetP1Helmet', 'DNoHelmetP1NoHelmet']`
- **VGG16 (.keras)**: Should accept 224x224 input and output binary classification (0=helmet, 1=no_helmet)
- **Faster R-CNN (.pth)**: 5 classes (including background): `[background, DHelmet, DNoHelmet, DHelmetP1Helmet, DNoHelmetP1NoHelmet]`
- **SSD (.pth)**: 5 classes (including background): `[background, DHelmet, DNoHelmet, DHelmetP1Helmet, DNoHelmetP1NoHelmet]`
- **DETR (.pth)**: 4 classes: `[DHelmet, DNoHelmet, DHelmetP1Helmet, DNoHelmetP1NoHelmet]`

**Note**: Only `yolo.pt` is required for basic functionality. Other models are optional and enable additional detection algorithms.

### Step 5: Verify Installation

#### Test Basic Functionality
```bash
# Test if the system can import all modules
python -c "
from detectors import YOLOv8Detector, CombinedHelmetDetector
print('✅ Basic imports successful')
"

# Test model loading (if you have models)
python -c "
import os
if os.path.exists('models/yolo.pt'):
    from detectors import YOLOv8Detector
    detector = YOLOv8Detector()
    print('✅ YOLO model loaded successfully')
else:
    print('⚠️  No YOLO model found - place yolo.pt in models/ directory')
"
```

#### Test Web Interface
```bash
# Start the Streamlit app
streamlit run app/main.py
```

If successful, you should see:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

## Running the Application

### Web Interface (Recommended)

#### Start the Application
```bash
# Make sure you're in the project directory
cd helmet-detector-prototype

# Activate your environment (if using conda/venv)
conda activate helmet-detector  # or activate your venv

# Start the Streamlit app
streamlit run app/main.py
```

#### Using the Web Interface
1. **Open your browser** to `http://localhost:8501`
2. **Upload an image** containing riders using the file uploader
3. **Select detection algorithm** from the radio buttons:
   - Hybrid (YOLO+VGG) - Requires both `yolo.pt` and `vgg16.keras`
   - YOLO Only - Requires only `yolo.pt`
   - RCNN (Faster R-CNN) - Requires `rcnn.pth`
   - SSD (SSD300-VGG16) - Requires `ssd.pth`
   - DETR - Requires `detr.pth`
4. **View results** with annotated image and statistics

### Command Line Testing

#### Test with a Single Image
```bash
# Test with YOLO detector (requires yolo.pt)
python -c "
from detectors import YOLOv8Detector
from PIL import Image
import sys

if len(sys.argv) > 1:
    image_path = sys.argv[1]
    detector = YOLOv8Detector()
    image = Image.open(image_path)
    result = detector.predict(image)
    print(f'Detected {result.raw[\"total_riders\"]} riders')
    print(f'With helmet: {result.raw[\"helmet_count\"]}')
    print(f'Without helmet: {result.raw[\"no_helmet_count\"]}')
else:
    print('Usage: python test_script.py path/to/image.jpg')
" path/to/your/image.jpg
```

#### Test All Available Detectors
```bash
# Test all detectors (if you have the models)
python -c "
import os
from detectors import *
from PIL import Image

image_path = 'path/to/your/image.jpg'  # Change this path
image = Image.open(image_path)

detectors = [
    ('YOLO', YOLOv8Detector),
    ('Hybrid', CombinedHelmetDetector),
    ('RCNN', RCNNHelmetDetector),
    ('SSD', SSDHelmetDetector),
    ('DETR', DETRHelmetDetector)
]

for name, detector_class in detectors:
    try:
        detector = detector_class()
        result = detector.predict(image)
        print(f'{name}: {result.raw[\"total_riders\"]} riders detected')
    except Exception as e:
        print(f'{name}: Error - {str(e)[:50]}...')
"
```

## Usage

### Web Interface

1. Open the Streamlit app in your browser
2. Upload an image containing riders
3. Select your preferred detection algorithm from the radio buttons
4. View the detection results with:
   - Annotated image with bounding boxes
   - Helmet counts (total riders, with helmet, without helmet)
   - Individual rider classifications
   - Confidence scores
   - Performance metrics (detection time, FPS)

### Programmatic Usage

```python
from detectors import (
    CombinedHelmetDetector, 
    YOLOv8Detector, 
    RCNNHelmetDetector, 
    SSDHelmetDetector, 
    DETRHelmetDetector
)
from PIL import Image

# Choose your detector
detector = CombinedHelmetDetector()  # Hybrid YOLO+VGG
# detector = YOLOv8Detector()        # YOLO only
# detector = RCNNHelmetDetector()    # Faster R-CNN
# detector = SSDHelmetDetector()     # SSD
# detector = DETRHelmetDetector()    # DETR

# Load image
image = Image.open("path/to/image.jpg")

# Run detection
result = detector.predict(image)

# Access results
print(f"Total riders: {result.raw['total_riders']}")
print(f"With helmet: {result.raw['helmet_count']}")
print(f"Without helmet: {result.raw['no_helmet_count']}")

# Individual detections
for box in result.boxes:
    print(f"Rider: {box.label} (confidence: {box.score:.2f})")
```

## Architecture

### Detection Approaches

The system supports multiple detection architectures, each optimized for different use cases:

#### 1. Hybrid (YOLO+VGG) - Two-Stage Pipeline
1. **YOLOv8 Detection**: 
   - Input: Full image
   - Output: Bounding boxes around riders
   - Crops: Each detected rider region

2. **VGG16 Classification**:
   - Input: Cropped rider images (resized to 224x224)
   - Output: Binary classification (helmet/no helmet)
   - Method: Uses `np.argmax()` on predictions

3. **Result Aggregation**:
   - Counts helmets and non-helmets
   - Provides detailed bounding box information
   - Generates annotated visualization

#### 2. Single-Stage Detectors
- **YOLO Only**: Direct helmet detection using YOLOv8 with 4 classes
- **Faster R-CNN**: ResNet-50 backbone with Region Proposal Network
- **SSD**: Single Shot Detector with VGG16 backbone for real-time performance
- **DETR**: Detection Transformer using ResNet-50 backbone for attention-based detection

### Performance Characteristics
- **Hybrid**: Highest accuracy, moderate speed
- **YOLO**: Good balance of speed and accuracy
- **Faster R-CNN**: High accuracy, slower inference
- **SSD**: Fast inference, good for real-time applications
- **DETR**: State-of-the-art accuracy, transformer-based attention

### File Structure

```
helmet-detector-prototype/
├── app/
│   ├── __init__.py
│   └── main.py              # Streamlit web application with model selection
├── detectors/
│   ├── __init__.py          # Detector module exports
│   ├── base.py              # Abstract detector interface
│   ├── yolo_detector.py     # YOLOv8 rider detection
│   ├── vgg_classifier.py    # VGG16 helmet classification
│   ├── combined_helmet_detector.py  # Hybrid YOLO+VGG pipeline
│   ├── rcnn_detector.py     # Faster R-CNN detector
│   ├── ssd_detector.py      # SSD detector
│   └── detr_detector.py     # DETR detector
├── utils/
│   ├── __init__.py
│   ├── model_loader.py      # Safe model loading with compatibility fixes
│   ├── model_converter.py   # Model format conversion utilities
│   └── visualize.py         # Visualization utilities
├── models/                  # Place your models here
│   ├── yolo.pt              # YOLOv8 model (required)
│   ├── vgg16.keras          # VGG16 model (for hybrid mode)
│   ├── rcnn.pth             # Faster R-CNN model (optional)
│   ├── ssd.pth              # SSD model (optional)
│   └── detr.pth             # DETR model (optional)
├── fix_model_compatibility.py  # Model compatibility fixer script
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md
```

## Troubleshooting

### Setup Issues

#### Python Version Problems
```bash
# Check your Python version
python --version

# If version is too old (< 3.8), install Python 3.9+:
# Windows: Download from python.org
# macOS: brew install python@3.9
# Linux: sudo apt install python3.9
```

#### Dependency Installation Issues
```bash
# If pip install fails, try upgrading pip first
python -m pip install --upgrade pip

# Install dependencies one by one if batch install fails
pip install streamlit==1.37.1
pip install Pillow==10.4.0
pip install numpy
pip install opencv-python==4.8.1.78
pip install tensorflow==2.19.0
pip install keras==3.10.0
pip install ultralytics==8.0.196
pip install torch torchvision==0.23.0
```

#### Environment Issues
```bash
# If you get "command not found" errors:
# Make sure your environment is activated
conda activate helmet-detector  # for conda
# or
source helmet-detector-env/bin/activate  # for venv

# Check if packages are installed in the right environment
pip list | grep streamlit
```

### Runtime Issues

1. **Model not found**: Ensure your model files are in the correct locations
2. **Import errors**: Install all dependencies with `pip install -r requirements.txt`
3. **CUDA errors**: The system works with CPU, but GPU acceleration is supported if available
4. **TensorFlow model loading errors**: Use the built-in compatibility fixes

### Model Compatibility Issues

If you encounter TensorFlow model loading errors (common with version mismatches):

```bash
# Run the compatibility fixer
python fix_model_compatibility.py
```

This script will:
- Diagnose model loading issues
- Attempt to convert models to compatible formats
- Create backup test models if needed
- Provide detailed error messages and solutions

### Quick Start Checklist

Before running the application, ensure you have:

- [ ] **Python 3.8+** installed and accessible
- [ ] **Virtual environment** created and activated
- [ ] **Dependencies** installed (`pip install -r requirements.txt`)
- [ ] **Models directory** created (`mkdir -p models`)
- [ ] **At least one model** placed in `models/` directory (e.g., `yolo.pt`)
- [ ] **Basic functionality** tested (import test passed)

### Getting Help

If you encounter issues not covered here:

1. **Check the error message** carefully - it often contains the solution
2. **Verify your Python version** matches the requirements
3. **Ensure all dependencies** are installed correctly
4. **Check model file paths** and formats
5. **Try the compatibility fixer** for model-related issues
6. **Test with minimal setup** (just YOLO model) before adding other models

### Performance Tips

- Use GPU acceleration if available (TensorFlow and PyTorch will automatically detect)
- For batch processing, consider using the individual detector classes directly
- Adjust confidence thresholds in the detector classes if needed
- **Algorithm Selection**:
  - Use **SSD** for fastest inference
  - Use **Hybrid (YOLO+VGG)** for highest accuracy
  - Use **DETR** for state-of-the-art results
  - Use **YOLO Only** for balanced performance

## Deployment

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)** and sign in with GitHub
3. **Click "New app"** and select your forked repository
4. **Configure the deployment**:
   - **Main file path**: `app/main.py`
   - **Requirements file**: `requirements-deploy.txt` (important!)
   - **Python version**: 3.11
5. **Deploy** and wait for the build to complete

### Important Deployment Notes

- **Use `requirements-deploy.txt`**: This excludes OpenCV to avoid deployment issues
- **Model Upload**: Users upload models at runtime - no model files needed in the repo
- **Temporary Storage**: Models are stored in session memory and cleaned up automatically
- **No GPU Required**: The app works on CPU-only deployments (though GPU is faster)

### Alternative Deployment Platforms

The app can also be deployed on:
- **Heroku**: Use `requirements-deploy.txt`
- **Railway**: Use `requirements-deploy.txt`
- **Google Cloud Run**: Use `requirements-deploy.txt`
- **AWS App Runner**: Use `requirements-deploy.txt`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
