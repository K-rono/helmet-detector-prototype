from .base import HelmetDetector, DetectionResult, BoundingBox

# Try to import YOLO detector, but make it optional for deployment
try:
    from .yolo_detector import YOLOv8Detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLOv8Detector = None

from .vgg_classifier import VGG16HelmetClassifier

# Try to import combined detector, but make it optional if YOLO is not available
try:
    from .combined_helmet_detector import CombinedHelmetDetector
    COMBINED_AVAILABLE = True
except ImportError:
    COMBINED_AVAILABLE = False
    CombinedHelmetDetector = None

from .rcnn_detector import RCNNHelmetDetector
from .ssd_detector import SSDHelmetDetector
from .detr_detector import DETRHelmetDetector

__all__ = [
    "HelmetDetector",
    "DetectionResult",
    "BoundingBox",
    "VGG16HelmetClassifier",
    "RCNNHelmetDetector",
    "SSDHelmetDetector",
    "DETRHelmetDetector",
]

# Add YOLO and Combined detectors only if available
if YOLO_AVAILABLE:
    __all__.append("YOLOv8Detector")

if COMBINED_AVAILABLE:
    __all__.append("CombinedHelmetDetector")


