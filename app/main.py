from __future__ import annotations

import io
import os
import sys
import time
import tempfile
import shutil
from typing import Optional, Dict

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

# Initialize session state for model storage
if 'uploaded_models' not in st.session_state:
	st.session_state.uploaded_models = {}
if 'temp_model_dir' not in st.session_state:
	st.session_state.temp_model_dir = None

# Cleanup function for when session ends
def cleanup_on_exit():
	"""Cleanup temporary files when session ends"""
	if st.session_state.temp_model_dir and os.path.exists(st.session_state.temp_model_dir):
		cleanup_temp_directory()

# Register cleanup function
import atexit
atexit.register(cleanup_on_exit)


def render_header():
	left, right = st.columns([0.7, 0.3])
	with left:
		st.title("Intersection Helmet Detection")
		st.caption(
			"Reducing rider fatality rates with AI-assisted helmet compliance checks at intersections."
		)
	with right:
		if st.session_state.uploaded_models:
			st.success(f"✅ {len(st.session_state.uploaded_models)} model(s) uploaded")
		else:
			st.warning("⚠️ No models uploaded")


def setup_temp_directory():
	"""Setup temporary directory for uploaded models"""
	if st.session_state.temp_model_dir is None:
		st.session_state.temp_model_dir = tempfile.mkdtemp(prefix="helmet_models_")
	return st.session_state.temp_model_dir


def cleanup_temp_directory():
	"""Cleanup temporary directory"""
	if st.session_state.temp_model_dir and os.path.exists(st.session_state.temp_model_dir):
		shutil.rmtree(st.session_state.temp_model_dir)
		st.session_state.temp_model_dir = None


def validate_model_file(file, expected_extensions):
	"""Validate uploaded model file"""
	if not file:
		return False, "No file uploaded"
	
	# Check file extension
	file_extension = file.name.split('.')[-1].lower()
	if file_extension not in expected_extensions:
		return False, f"Invalid file type. Expected: {', '.join(expected_extensions)}"
	
	# Check file size (limit to 500MB)
	max_size = 500 * 1024 * 1024  # 500MB
	if file.size > max_size:
		return False, f"File too large. Maximum size: 500MB"
	
	return True, "Valid model file"


def render_model_uploader():
	"""Render model upload interface"""
	st.subheader("📁 Upload Models")
	
	# Create tabs for different model types
	tab1, tab2, tab3, tab4, tab5 = st.tabs(["YOLO", "VGG16", "RCNN", "SSD", "DETR"])
	
	with tab1:
		st.markdown("**YOLOv8 Model (.pt)**")
		st.caption("Required for YOLO Only and Hybrid detection")
		yolo_file = st.file_uploader(
			"Upload YOLO model",
			type=["pt"],
			key="yolo_upload",
			help="Upload your trained YOLOv8 model file"
		)
		
		if yolo_file:
			is_valid, message = validate_model_file(yolo_file, ["pt"])
			if is_valid:
				if st.button("Save YOLO Model", key="save_yolo"):
					temp_dir = setup_temp_directory()
					model_path = os.path.join(temp_dir, "yolo.pt")
					with open(model_path, "wb") as f:
						f.write(yolo_file.getbuffer())
					st.session_state.uploaded_models["yolo"] = model_path
					st.success("✅ YOLO model saved successfully!")
					st.rerun()
			else:
				st.error(f"❌ {message}")
	
	with tab2:
		st.markdown("**VGG16 Model (.keras)**")
		st.caption("Required for Hybrid (YOLO+VGG) detection")
		vgg_file = st.file_uploader(
			"Upload VGG16 model",
			type=["keras", "h5"],
			key="vgg_upload",
			help="Upload your trained VGG16 model file"
		)
		
		if vgg_file:
			is_valid, message = validate_model_file(vgg_file, ["keras", "h5"])
			if is_valid:
				if st.button("Save VGG16 Model", key="save_vgg"):
					temp_dir = setup_temp_directory()
					model_path = os.path.join(temp_dir, "vgg16.keras")
					with open(model_path, "wb") as f:
						f.write(vgg_file.getbuffer())
					st.session_state.uploaded_models["vgg16"] = model_path
					st.success("✅ VGG16 model saved successfully!")
					st.rerun()
			else:
				st.error(f"❌ {message}")
	
	with tab3:
		st.markdown("**Faster R-CNN Model (.pth)**")
		st.caption("Required for RCNN detection")
		rcnn_file = st.file_uploader(
			"Upload RCNN model",
			type=["pth"],
			key="rcnn_upload",
			help="Upload your trained Faster R-CNN model file"
		)
		
		if rcnn_file:
			is_valid, message = validate_model_file(rcnn_file, ["pth"])
			if is_valid:
				if st.button("Save RCNN Model", key="save_rcnn"):
					temp_dir = setup_temp_directory()
					model_path = os.path.join(temp_dir, "rcnn.pth")
					with open(model_path, "wb") as f:
						f.write(rcnn_file.getbuffer())
					st.session_state.uploaded_models["rcnn"] = model_path
					st.success("✅ RCNN model saved successfully!")
					st.rerun()
			else:
				st.error(f"❌ {message}")
	
	with tab4:
		st.markdown("**SSD Model (.pth)**")
		st.caption("Required for SSD detection")
		ssd_file = st.file_uploader(
			"Upload SSD model",
			type=["pth"],
			key="ssd_upload",
			help="Upload your trained SSD model file"
		)
		
		if ssd_file:
			is_valid, message = validate_model_file(ssd_file, ["pth"])
			if is_valid:
				if st.button("Save SSD Model", key="save_ssd"):
					temp_dir = setup_temp_directory()
					model_path = os.path.join(temp_dir, "ssd.pth")
					with open(model_path, "wb") as f:
						f.write(ssd_file.getbuffer())
					st.session_state.uploaded_models["ssd"] = model_path
					st.success("✅ SSD model saved successfully!")
					st.rerun()
			else:
				st.error(f"❌ {message}")
	
	with tab5:
		st.markdown("**DETR Model (.pth)**")
		st.caption("Required for DETR detection")
		detr_file = st.file_uploader(
			"Upload DETR model",
			type=["pth"],
			key="detr_upload",
			help="Upload your trained DETR model file"
		)
		
		if detr_file:
			is_valid, message = validate_model_file(detr_file, ["pth"])
			if is_valid:
				if st.button("Save DETR Model", key="save_detr"):
					temp_dir = setup_temp_directory()
					model_path = os.path.join(temp_dir, "detr.pth")
					with open(model_path, "wb") as f:
						f.write(detr_file.getbuffer())
					st.session_state.uploaded_models["detr"] = model_path
					st.success("✅ DETR model saved successfully!")
					st.rerun()
			else:
				st.error(f"❌ {message}")
	
	# Display uploaded models
	if st.session_state.uploaded_models:
		st.markdown("---")
		st.subheader("📋 Uploaded Models")
		for model_name, model_path in st.session_state.uploaded_models.items():
			col1, col2, col3 = st.columns([2, 1, 1])
			with col1:
				st.write(f"**{model_name.upper()}**: {os.path.basename(model_path)}")
			with col2:
				file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
				st.write(f"{file_size:.1f} MB")
			with col3:
				if st.button("🗑️", key=f"delete_{model_name}", help="Delete model"):
					os.remove(model_path)
					del st.session_state.uploaded_models[model_name]
					st.success(f"Deleted {model_name} model")
					st.rerun()
	
	# Clear all models button
	if st.session_state.uploaded_models:
		if st.button("🗑️ Clear All Models", type="secondary"):
			cleanup_temp_directory()
			st.session_state.uploaded_models = {}
			st.success("All models cleared!")
			st.rerun()


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
	
	# Check which models are available
	available_models = []
	model_status = {}
	
	# Check YOLO
	if "yolo" in st.session_state.uploaded_models:
		available_models.append("YOLO Only")
		model_status["YOLO Only"] = "✅ Available"
	else:
		model_status["YOLO Only"] = "❌ YOLO model required"
	
	# Check Hybrid (YOLO + VGG)
	if "yolo" in st.session_state.uploaded_models and "vgg16" in st.session_state.uploaded_models:
		available_models.append("Hybrid (YOLO+VGG)")
		model_status["Hybrid (YOLO+VGG)"] = "✅ Available"
	else:
		missing = []
		if "yolo" not in st.session_state.uploaded_models:
			missing.append("YOLO")
		if "vgg16" not in st.session_state.uploaded_models:
			missing.append("VGG16")
		model_status["Hybrid (YOLO+VGG)"] = f"❌ {', '.join(missing)} model(s) required"
	
	# Check RCNN
	if "rcnn" in st.session_state.uploaded_models:
		available_models.append("RCNN (Faster R-CNN)")
		model_status["RCNN (Faster R-CNN)"] = "✅ Available"
	else:
		model_status["RCNN (Faster R-CNN)"] = "❌ RCNN model required"
	
	# Check SSD
	if "ssd" in st.session_state.uploaded_models:
		available_models.append("SSD (SSD300-VGG16)")
		model_status["SSD (SSD300-VGG16)"] = "✅ Available"
	else:
		model_status["SSD (SSD300-VGG16)"] = "❌ SSD model required"
	
	# Check DETR
	if "detr" in st.session_state.uploaded_models:
		available_models.append("DETR")
		model_status["DETR"] = "✅ Available"
	else:
		model_status["DETR"] = "❌ DETR model required"
	
	# Create options with status
	all_options = ["Hybrid (YOLO+VGG)", "YOLO Only", "RCNN (Faster R-CNN)", "SSD (SSD300-VGG16)", "DETR"]
	
	# If no models are available, show warning
	if not available_models:
		st.warning("⚠️ No models uploaded. Please upload at least one model to use detection.")
		return None
	
	# Show model status
	st.caption("Model availability:")
	for option in all_options:
		st.caption(f"• {option}: {model_status[option]}")
	
	# Create radio buttons with only available models
	choice = st.radio(
		"Algorithm",
		available_models,
		index=0,
		horizontal=True,
	)
	return choice


def run_detection(image: Image.Image, choice: str):
	"""Run detection using uploaded models"""
	if choice.startswith("Hybrid"):
		yolo_path = st.session_state.uploaded_models.get("yolo")
		vgg_path = st.session_state.uploaded_models.get("vgg16")
		detector = CombinedHelmetDetector(yolo_path, vgg_path)
	elif choice.startswith("YOLO"):
		yolo_path = st.session_state.uploaded_models.get("yolo")
		detector = YOLOv8Detector(yolo_path)
	elif choice.startswith("RCNN"):
		rcnn_path = st.session_state.uploaded_models.get("rcnn")
		detector = RCNNHelmetDetector(rcnn_path)
	elif choice.startswith("SSD"):
		ssd_path = st.session_state.uploaded_models.get("ssd")
		detector = SSDHelmetDetector(ssd_path)
	else:  # DETR
		detr_path = st.session_state.uploaded_models.get("detr")
		detector = DETRHelmetDetector(detr_path)
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
		st.info("Please ensure you have uploaded the required model files in the 'Upload Models' tab.")


def render_footer():
	st.markdown("---")
	st.caption(
		"Powered by YOLOv8 (with optional VGG16), Faster R-CNN, SSD, or DETR for helmet classification."
	)


def main():
	render_header()
	
	# Create tabs for different sections
	tab1, tab2, tab3 = st.tabs(["📁 Upload Models", "🖼️ Upload Image", "🔍 Detection Results"])
	
	with tab1:
		render_model_uploader()
	
	with tab2:
		image = render_uploader()
		choice = render_model_selector()
		
		# Store in session state for use in other tabs
		st.session_state.current_image = image
		st.session_state.current_choice = choice
	
	with tab3:
		# Get image and choice from session state
		image = st.session_state.get('current_image')
		choice = st.session_state.get('current_choice')
		
		if image is not None and choice is not None:
			render_prediction(image, choice)
		elif image is None:
			st.info("Please upload an image in the 'Upload Image' tab.")
		elif choice is None:
			st.info("Please upload at least one model in the 'Upload Models' tab.")
		else:
			st.info("Please upload both an image and at least one model to run detection.")
	
	render_footer()


if __name__ == "__main__":
	main()


