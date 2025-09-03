from __future__ import annotations

import os
from typing import List

import torch
from PIL import Image
import torchvision
from torchvision.transforms import functional as F

from .base import BoundingBox, DetectionResult, HelmetDetector


class RCNNHelmetDetector(HelmetDetector):
	"""Faster R-CNN detector for helmet detection.

	Expects a checkpoint at models/rcnn.pth with class indices:
	1: DHelmet, 2: DNoHelmet, 3: DHelmetP1Helmet, 4: DNoHelmetP1NoHelmet
	"""

	def __init__(self, model_path: str = "models/rcnn.pth", score_threshold: float = 0.4):
		if not os.path.exists(model_path):
			raise FileNotFoundError(f"RCNN model not found at {model_path}")

		# 5 classes including background index 0
		self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=5)
		ckpt = torch.load(model_path, map_location="cpu")
		if isinstance(ckpt, dict) and "state_dict" in ckpt:
			state = ckpt["state_dict"]
		else:
			state = ckpt
		self.model.load_state_dict(state, strict=False)
		self.model.eval()

		self.score_threshold = score_threshold
		self.class_id_to_name = {
			1: 'DHelmet',
			2: 'DNoHelmet',
			3: 'DHelmetP1Helmet',
			4: 'DNoHelmetP1NoHelmet',
		}

	def _preprocess(self, image: Image.Image):
		# Convert PIL to tensor in [0,1], no extra normalization as requested
		return F.to_tensor(image)

	@torch.inference_mode()
	def predict(self, image: Image.Image) -> DetectionResult:
		input_tensor = self._preprocess(image)
		outputs = self.model([input_tensor])[0]

		boxes: List[BoundingBox] = []
		helmet_count = 0
		no_helmet_count = 0

		for box, label, score in zip(outputs.get('boxes', []), outputs.get('labels', []), outputs.get('scores', [])):
			score_val = float(score)
			if score_val < self.score_threshold:
				continue
			label_id = int(label)
			if label_id == 0:
				continue
			name = self.class_id_to_name.get(label_id, str(label_id))
			x1, y1, x2, y2 = box.tolist()
			bbox = BoundingBox(
				x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
				label=name,
				score=score_val,
			)
			boxes.append(bbox)

			# Updated counting logic for RCNN (non-zero indexed)
			if label_id == 1:  # DHelmet
				helmet_count += 1
			elif label_id == 2:  # DNoHelmet
				no_helmet_count += 1
			elif label_id == 3:  # DHelmetP1Helmet
				helmet_count += 2
			elif label_id == 4:  # DNoHelmetP1NoHelmet
				no_helmet_count += 2

		total = len(boxes)
		overall_conf = (sum(b.score for b in boxes) / total * 100) if total else 0.0
		if total == 0:
			overall_label = "No riders detected"
		else:
			overall_label = f"Detected {total} riders: {helmet_count} with helmet, {no_helmet_count} without helmet"

		return DetectionResult(
			label=overall_label,
			confidence=overall_conf,
			boxes=boxes,
			raw={
				'helmet_count': helmet_count,
				'no_helmet_count': no_helmet_count,
				'total_riders': total,
			}
		)
