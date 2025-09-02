# Helmet Detection System

A comprehensive helmet detection system supporting multiple state-of-the-art detection algorithms for intersection safety monitoring.

## Features

- **Multiple Detection Algorithms**: Choose from 5 different detection approaches:
  - **Hybrid (YOLO+VGG)**: Two-stage pipeline using YOLOv8 for rider detection + VGG16 for helmet classification
  - **YOLO Only**: Direct helmet detection using YOLOv8
  - **Faster R-CNN**: Single-stage detection with ResNet-50 backbone
  - **SSD**: Single Shot Detector with VGG16 backbone
  - **DETR**: Detection Transformer with ResNet-50 backbone
- **Real-time Performance**: Optimized for intersection monitoring with FPS tracking
- **Helmet Counting**: Counts total riders, riders with helmets, and riders without helmets
- **Web Interface**: Streamlit-based web application with model selection and visualization
- **Visualization**: Color-coded bounding boxes (green for helmet, red for no helmet)
- **Model Compatibility**: Built-in compatibility fixes for PyTorch 2.6+ and TensorFlow version mismatches

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Model Files

Place your trained models in the `models/` directory:

```
models/
├── yolo.pt          # YOLOv8 model for rider detection
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

### 3. Run the Application

#### Web Interface (Recommended)
```bash
streamlit run app/main.py
```

#### Command Line Testing
```bash
python test_pipeline.py path/to/your/image.jpg
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

### Common Issues

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

### Performance Tips

- Use GPU acceleration if available (TensorFlow and PyTorch will automatically detect)
- For batch processing, consider using the individual detector classes directly
- Adjust confidence thresholds in the detector classes if needed
- **Algorithm Selection**:
  - Use **SSD** for fastest inference
  - Use **Hybrid (YOLO+VGG)** for highest accuracy
  - Use **DETR** for state-of-the-art results
  - Use **YOLO Only** for balanced performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.
