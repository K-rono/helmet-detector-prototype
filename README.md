# Helmet Detection System

A two-stage helmet detection system using YOLOv8 for rider detection and VGG16 for helmet classification.

## Features

- **YOLOv8 Detection**: Detects riders in images and crops them for further analysis
- **VGG16 Classification**: Classifies each cropped rider as wearing helmet (0) or not wearing helmet (1)
- **Helmet Counting**: Counts total riders, riders with helmets, and riders without helmets
- **Web Interface**: Streamlit-based web application for easy image upload and visualization
- **Visualization**: Color-coded bounding boxes (green for helmet, red for no helmet)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Model Files

Place your trained models in the `models/` directory:

```
models/
├── yolo.pt          # Your trained YOLOv8 model
└── vgg16.keras      # Your trained VGG16 model
```

**Model Requirements:**
- **YOLOv8 (.pt)**: Should detect 4 classes: `['DHelmet', 'DNoHelmet', 'DHelmetP1Helmet', 'DNoHelmetP1NoHelmet']`
- **VGG16 (.keras)**: Should accept 224x224 input and output binary classification (0=helmet, 1=no_helmet)

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
3. View the detection results with:
   - Annotated image with bounding boxes
   - Helmet counts (total riders, with helmet, without helmet)
   - Individual rider classifications
   - Confidence scores

### Programmatic Usage

```python
from detectors import CombinedHelmetDetector
from PIL import Image

# Initialize detector
detector = CombinedHelmetDetector()

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

### Pipeline Overview

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

### File Structure

```
helmet-detector/
├── app/
│   └── main.py              # Streamlit web application
├── detectors/
│   ├── base.py              # Abstract detector interface
│   ├── yolo_detector.py     # YOLOv8 rider detection
│   ├── vgg_classifier.py    # VGG16 helmet classification
│   ├── combined_helmet_detector.py  # Complete pipeline
│   └── dummy.py             # Dummy detector for testing
├── utils/
│   └── visualize.py         # Visualization utilities
├── models/                  # Place your models here
│   ├── yolo.pt
│   └── vgg16.keras
├── test_pipeline.py         # Command-line testing script
└── requirements.txt         # Python dependencies
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure your model files are in the correct locations
2. **Import errors**: Install all dependencies with `pip install -r requirements.txt`
3. **CUDA errors**: The system works with CPU, but GPU acceleration is supported if available

### Performance Tips

- Use GPU acceleration if available (TensorFlow and PyTorch will automatically detect)
- For batch processing, consider using the individual detector classes directly
- Adjust confidence thresholds in the detector classes if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.
