from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from detectors.base import DetectionResult, BoundingBox


def _color_for_label(label: str) -> Tuple[int, int, int]:
    """Get color for different types of detections."""
    label_lower = label.lower()
    
    if "helmet" in label_lower and "no" not in label_lower:
        return 34, 197, 94  # green for helmet
    elif "no helmet" in label_lower or "nohelmet" in label_lower:
        return 239, 68, 68  # red for no helmet
    elif "rider" in label_lower:
        # Check if it's a rider with helmet classification
        if "helmet" in label_lower:
            return 34, 197, 94  # green
        else:
            return 239, 68, 68  # red
    else:
        return 59, 130, 246  # blue for other detections


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], text: str, color: Tuple[int, int, int]):
    x1, y1, x2, y2 = xy
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    # Pillow 10+: use textbbox to measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 4
    rect = (x1, max(0, y1 - text_h - 2 * pad), x1 + text_w + 2 * pad, y1)
    draw.rectangle(rect, fill=color + (80,))
    draw.text((x1 + pad, rect[1] + pad), text, fill=(255, 255, 255), font=font)


def draw_detections(image: Image.Image, detections: DetectionResult) -> Image.Image:
    annotated = image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in detections.boxes:
        color = _color_for_label(box.label)
        xy = (box.x1, box.y1, box.x2, box.y2)
        draw.rectangle(xy, outline=color + (255,), width=4)
        
        # Format label text
        if "Rider" in box.label:
            # For combined detector results
            label_text = f"{box.label} ({int(box.score * 100)}%)"
        else:
            # For YOLO-only results
            label_text = f"{box.label} {int(box.score * 100)}%"
        
        _draw_label(draw, xy, label_text, color)

    return Image.alpha_composite(annotated, overlay).convert("RGB")


