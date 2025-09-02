from __future__ import annotations

import os
from typing import List

import torch
from PIL import Image
import torchvision
from torchvision.transforms import functional as F

from .base import BoundingBox, DetectionResult, HelmetDetector


class DETRHelmetDetector(HelmetDetector):
	"""DETR ResNet-50 detector for helmet detection.

	Expects a checkpoint at models/detr.pth with class indices:
	0: DHelmet, 1: DNoHelmet, 2: DHelmetP1Helmet, 3: DNoHelmetP1NoHelmet
	"""

	def __init__(self, model_path: str = "models/detr.pth", score_threshold: float = 0.1):
		# Load DETR model from Facebook Research repository
		self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
		
		# Load custom weights if provided
		if os.path.exists(model_path):
			ckpt = torch.load(model_path, map_location="cpu")
			if isinstance(ckpt, dict) and "state_dict" in ckpt:
				state = ckpt["state_dict"]
			else:
				state = ckpt
			self.model.load_state_dict(state, strict=False)
		
		self.model.eval()

		self.score_threshold = score_threshold
		# DETR from Facebook Research uses COCO classes, but we'll map to our custom classes
		# The model should be fine-tuned for our specific classes
		self.class_id_to_name = {
			0: 'DHelmet',
			1: 'DNoHelmet', 
			2: 'DHelmetP1Helmet',
			3: 'DNoHelmetP1NoHelmet',
		}

		# ImageNet normalization stats
		self.mean = [0.485, 0.456, 0.406]
		self.std = [0.229, 0.224, 0.225]

	def _preprocess(self, image: Image.Image):
		# DETR expects images to be resized to 800px max dimension
		# and normalized with ImageNet stats
		image = F.resize(image, (800, 800))
		# Convert to tensor
		tensor = F.to_tensor(image)
		# Normalize with ImageNet stats
		tensor = F.normalize(tensor, mean=self.mean, std=self.std)
		return tensor

	@torch.inference_mode()
	def predict(self, image: Image.Image) -> DetectionResult:
		input_tensor = self._preprocess(image)
		
		# DETR model expects a list of tensors
		outputs = self.model([input_tensor])
		
		# DETR outputs are different from other models - they return logits and pred_boxes
		# We need to process the outputs correctly
		if isinstance(outputs, dict):
			# Handle dict output format
			pred_logits = outputs.get('pred_logits', outputs.get('logits'))
			pred_boxes = outputs.get('pred_boxes', outputs.get('boxes'))
		else:
			# Handle tuple/list output format
			pred_logits = outputs[0]['pred_logits'] if len(outputs) > 0 else outputs[0].get('logits')
			pred_boxes = outputs[0]['pred_boxes'] if len(outputs) > 0 else outputs[0].get('boxes')

		boxes: List[BoundingBox] = []
		helmet_count = 0
		no_helmet_count = 0

		if pred_logits is not None and pred_boxes is not None:
			# Apply softmax to get probabilities
			probs = torch.nn.functional.softmax(pred_logits, -1)
			scores, labels = probs[..., :-1].max(-1)  # Remove background class
			
			# Filter by confidence threshold
			keep = scores > self.score_threshold
			
			if keep.any():
				boxes_tensor = pred_boxes[keep]
				scores_filtered = scores[keep]
				labels_filtered = labels[keep]
				
				for box, label, score in zip(boxes_tensor, labels_filtered, scores_filtered):
					score_val = float(score)
					label_id = int(label)
					name = self.class_id_to_name.get(label_id, str(label_id))
					
					# Convert normalized coordinates to pixel coordinates
					# DETR outputs normalized coordinates [0,1] in format [center_x, center_y, width, height]
					h, w = image.size[1], image.size[0]  # PIL image size is (width, height)
					cx, cy, bw, bh = box.tolist()
					
					# Convert from center format to corner format
					x1 = (cx - bw/2) * w
					y1 = (cy - bh/2) * h
					x2 = (cx + bw/2) * w
					y2 = (cy + bh/2) * h
					
					# Ensure coordinates are valid
					x1, x2 = min(x1, x2), max(x1, x2)
					y1, y2 = min(y1, y2), max(y1, y2)
					
					bbox = BoundingBox(
						x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
						label=name,
						score=score_val,
					)
					boxes.append(bbox)

					# Counting
					name_lower = name.lower()
					if "nohelmet" in name_lower or "no helmet" in name_lower:
						no_helmet_count += 1
					else:
						helmet_count += 1

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
