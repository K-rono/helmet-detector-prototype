from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from PIL.Image import Image as PILImage


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    score: float


@dataclass
class DetectionResult:
    label: str
    confidence: float  # 0-100
    boxes: List[BoundingBox]
    raw: Optional[dict] = None


class HelmetDetector(ABC):
    """Abstract detector interface for helmet detection models.

    Implementations should accept a PIL image and return a structured result.
    This enables easy swapping of backends (e.g., YOLO, VGG) later.
    """

    @abstractmethod
    def predict(self, image: PILImage) -> DetectionResult:
        raise NotImplementedError


