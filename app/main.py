from __future__ import annotations

import io
import os
import sys
import time
from typing import Optional

# Ensure project root is on sys.path so we can import sibling packages
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from PIL import Image

from detectors import CombinedHelmetDetector, RCNNHelmetDetector, YOLOv8Detector, SSDHelmetDetector, DETRHelmetDetector
from utils.visualize import draw_detections


st.set_page_config(
	page_title="Intersection Helmet Detection",
	page_icon="🪖",
	layout="wide",
	initial_sidebar_state="collapsed",
)


def render_header():
	left, right = st.columns([0.7, 0.3])
	with left:
		st.title("Intersection Helmet Detection")
		st.caption(
			"Reducing rider fatality rates with AI-assisted helmet compliance checks at intersections."
		)


def render_uploader() -> Optional[Image.Image]:
	st.subheader("Upload Rider Image")
	file = st.file_uploader(
		"Drop a photo here (JPG, JPEG, PNG, BMP, WEBP)",
		type=["jpg", "jpeg", "png", "bmp", "webp"],
		accept_multiple_files=False,
		help="Image should contain riders. Choose a model and run detection.",
	)
	if file is None:
		return None
	bytes_data = file.read()
	image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
	st.success("Image loaded successfully.")
	return image


def render_model_selector() -> str:
	st.subheader("Select Detection Algorithm")
	choice = st.radio(
		"Algorithm",
		["Hybrid (YOLO+VGG)", "YOLO Only", "RCNN (Faster R-CNN)", "SSD (SSD300-VGG16)", "DETR"],
		index=0,
		horizontal=True,
	)
	return choice


def run_detection(image: Image.Image, choice: str):
	if choice.startswith("Hybrid"):
		detector = CombinedHelmetDetector()
	elif choice.startswith("YOLO"):
		detector = YOLOv8Detector()
	elif choice.startswith("RCNN"):
		detector = RCNNHelmetDetector()
	elif choice.startswith("SSD"):
		detector = SSDHelmetDetector()
	else:  # DETR
		detector = DETRHelmetDetector()
	return detector.predict(image)


def _compute_counts_fallback(result):
	counts = {"helmet_count": 0, "no_helmet_count": 0, "total_riders": 0}
	if not result or not getattr(result, "boxes", None):
		return counts
	helmet = 0
	no_helmet = 0
	for b in result.boxes:
		name = (b.label or "").lower()
		if "nohelmet" in name or "no helmet" in name:
			no_helmet += 1
		elif "helmet" in name:
			helmet += 1
	counts["helmet_count"] = helmet
	counts["no_helmet_count"] = no_helmet
	counts["total_riders"] = len(result.boxes)
	return counts


def render_prediction(image: Image.Image, choice: str):
	try:
		start_ts = time.perf_counter()
		result = run_detection(image, choice)
		elapsed_s = time.perf_counter() - start_ts
		elapsed_ms = int(elapsed_s * 1000)
		fps = (1.0 / elapsed_s) if elapsed_s > 0 else 0.0

		annotated = draw_detections(image, result)

		c1, c2 = st.columns([0.55, 0.45])
		with c1:
			st.image(annotated, caption="Annotated Detection", use_column_width=True)

		with c2:
			st.subheader("Detection Result")
			st.markdown(f"**Decision:** {result.label}")
			st.progress(int(result.confidence))
			st.markdown(f"**Confidence:** {result.confidence:.0f} / 100")
			st.markdown(f"**Detection Time:** {elapsed_ms} ms  |  **FPS:** {fps:.2f}")

			# Display helmet counts (fallback if missing in raw)
			counts = (result.raw or {}) if hasattr(result, "raw") else {}
			if not all(k in counts for k in ("helmet_count", "no_helmet_count", "total_riders")):
				counts = _compute_counts_fallback(result)

			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("Total Riders", counts.get('total_riders', 0))
			with col2:
				st.metric("With Helmet", counts.get('helmet_count', 0))
			with col3:
				st.metric("Without Helmet", counts.get('no_helmet_count', 0))

			with st.expander("Details"):
				st.json(
					{
						"label": result.label,
						"confidence": result.confidence,
						"boxes": [
							{
								"x1": b.x1,
								"y1": b.y1,
								"x2": b.x2,
								"y2": b.y2,
								"label": b.label,
								"score": b.score,
							}
							for b in result.boxes
						],
						"counts": counts,
						"timing_ms": elapsed_ms,
						"fps": float(fps),
					}
				)

	except Exception as e:
		st.error(f"Error during detection: {str(e)}")
		st.info("Ensure model files exist: models/yolo.pt, models/vgg16.keras, models/rcnn.pth, models/ssd.pth, models/detr.pth")


def render_footer():
	st.markdown("---")
	st.caption(
		"Powered by YOLOv8 (with optional VGG16), Faster R-CNN, SSD, or DETR for helmet classification."
	)


def main():
	render_header()
	image = render_uploader()
	choice = render_model_selector()
	if image is not None:
		render_prediction(image, choice)
	render_footer()


if __name__ == "__main__":
	main()


