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

# Import detectors with fallback handling
try:
    from detectors import (
        CombinedHelmetDetector, 
        RCNNHelmetDetector, 
        YOLOv8Detector, 
        SSDHelmetDetector, 
        DETRHelmetDetector
    )
    ALL_DETECTORS_AVAILABLE = True
except ImportError as e:
    # Handle case where some detectors are not available (e.g., YOLO requires ultralytics)
    ALL_DETECTORS_AVAILABLE = False
    print(f"Warning: Some detectors not available: {e}")
    
    # Try to import available detectors individually
    try:
        from detectors import RCNNHelmetDetector, SSDHelmetDetector, DETRHelmetDetector
    except ImportError:
        RCNNHelmetDetector = SSDHelmetDetector = DETRHelmetDetector = None
    
    try:
        from detectors import YOLOv8Detector
    except ImportError:
        YOLOv8Detector = None
    
    try:
        from detectors import CombinedHelmetDetector
    except ImportError:
        CombinedHelmetDetector = None
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
if 'available_models' not in st.session_state:
	st.session_state.available_models = {}

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
		# Initialize available models if not done yet
		if not st.session_state.available_models:
			st.session_state.available_models = detect_local_models()
		
		total_models = len(st.session_state.uploaded_models) + len(st.session_state.available_models)
		if total_models > 0:
			local_count = len(st.session_state.available_models)
			uploaded_count = len(st.session_state.uploaded_models)
			if local_count > 0 and uploaded_count > 0:
				st.success(f"✅ {total_models} model(s) available ({local_count} local, {uploaded_count} uploaded)")
			elif local_count > 0:
				st.success(f"✅ {local_count} local model(s) available")
			else:
				st.success(f"✅ {uploaded_count} uploaded model(s)")
		else:
			st.warning("⚠️ No models available")


def detect_local_models():
	"""Detect models in the project's models directory"""
	models_dir = os.path.join(PROJECT_ROOT, "models")
	available_models = {}
	
	if os.path.exists(models_dir):
		model_files = {
			"yolo": ["yolo.pt"],
			"vgg16": ["vgg16.keras", "vgg16.h5"],
			"rcnn": ["rcnn.pth"],
			"ssd": ["ssd.pth"],
			"detr": ["detr.pth"]
		}
		
		for model_type, filenames in model_files.items():
			for filename in filenames:
				filepath = os.path.join(models_dir, filename)
				if os.path.exists(filepath):
					available_models[model_type] = filepath
					break
	
	return available_models


def get_model_path(model_type: str) -> Optional[str]:
	"""Get model path from either uploaded models or local files"""
	# First check uploaded models
	if model_type in st.session_state.uploaded_models:
		return st.session_state.uploaded_models[model_type]
	
	# Then check local models
	if model_type in st.session_state.available_models:
		return st.session_state.available_models[model_type]
	
	return None


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


def render_model_tab(model_type: str, model_name: str, extensions: list, description: str, key_prefix: str):
	"""Render a model tab with hybrid loading"""
	st.markdown(f"**{model_name}**")
	st.caption(description)
	
	# Check if model already exists
	model_path = get_model_path(model_type)
	if model_path:
		if model_type in st.session_state.available_models:
			st.success(f"✅ Local model found: {os.path.basename(model_path)}")
			st.caption(f"Path: {model_path}")
		else:
			st.success(f"✅ Uploaded model: {os.path.basename(model_path)}")
		
		# Show option to replace with upload
		if st.button("Replace with Upload", key=f"replace_{key_prefix}"):
			st.session_state[f'show_{key_prefix}_upload'] = True
			st.rerun()
	else:
		st.info(f"No {model_name} model found. Please upload one.")
		st.session_state[f'show_{key_prefix}_upload'] = True
	
	# Show upload interface if needed
	if st.session_state.get(f'show_{key_prefix}_upload', False):
		uploaded_file = st.file_uploader(
			f"Upload {model_name} model",
			type=extensions,
			key=f"{key_prefix}_upload",
			help=f"Upload your trained {model_name} model file"
		)
		
		if uploaded_file:
			is_valid, message = validate_model_file(uploaded_file, extensions)
			if is_valid:
				if st.button(f"Save {model_name} Model", key=f"save_{key_prefix}"):
					temp_dir = setup_temp_directory()
					model_path = os.path.join(temp_dir, f"{model_type}.{extensions[0]}")
					with open(model_path, "wb") as f:
						f.write(uploaded_file.getbuffer())
					st.session_state.uploaded_models[model_type] = model_path
					st.session_state[f'show_{key_prefix}_upload'] = False
					st.success(f"✅ {model_name} model saved successfully!")
					st.rerun()
			else:
				st.error(f"❌ {message}")
		
		if st.button("Cancel", key=f"cancel_{key_prefix}"):
			st.session_state[f'show_{key_prefix}_upload'] = False
			st.rerun()


def render_model_uploader():
	"""Render model upload interface"""
	st.subheader("📁 Model Management")
	
	# Show current model status
	if st.session_state.available_models or st.session_state.uploaded_models:
		st.markdown("**Available Models:**")
		col1, col2 = st.columns(2)
		
		with col1:
			if st.session_state.available_models:
				st.markdown("🏠 **Local Models:**")
				for model_type, path in st.session_state.available_models.items():
					file_size = os.path.getsize(path) / (1024 * 1024)  # MB
					st.caption(f"• {model_type.upper()}: {os.path.basename(path)} ({file_size:.1f} MB)")
		
		with col2:
			if st.session_state.uploaded_models:
				st.markdown("☁️ **Uploaded Models:**")
				for model_type, path in st.session_state.uploaded_models.items():
					file_size = os.path.getsize(path) / (1024 * 1024)  # MB
					st.caption(f"• {model_type.upper()}: {os.path.basename(path)} ({file_size:.1f} MB)")
		
		st.markdown("---")
	
	# Create tabs for different model types
	tab1, tab2, tab3, tab4, tab5 = st.tabs(["YOLO", "VGG16", "RCNN", "SSD", "DETR"])
	
	with tab1:
		render_model_tab("yolo", "YOLOv8 Model (.pt)", ["pt"], "Required for YOLO Only and Hybrid detection", "yolo")
	
	with tab2:
		render_model_tab("vgg16", "VGG16 Model (.keras)", ["keras", "h5"], "Required for Hybrid (YOLO+VGG) detection", "vgg")
	
	with tab3:
		render_model_tab("rcnn", "Faster R-CNN Model (.pth)", ["pth"], "Required for RCNN detection", "rcnn")
	
	with tab4:
		render_model_tab("ssd", "SSD Model (.pth)", ["pth"], "Required for SSD detection", "ssd")
	
	with tab5:
		render_model_tab("detr", "DETR Model (.pth)", ["pth"], "Required for DETR detection", "detr")
	
	# Display uploaded models (only uploaded ones can be deleted)
	if st.session_state.uploaded_models:
		st.markdown("---")
		st.subheader("📋 Uploaded Models")
		st.caption("These models were uploaded at runtime and can be deleted")
		for model_name, model_path in st.session_state.uploaded_models.items():
			col1, col2, col3 = st.columns([2, 1, 1])
			with col1:
				st.write(f"**{model_name.upper()}**: {os.path.basename(model_path)}")
			with col2:
				file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
				st.write(f"{file_size:.1f} MB")
			with col3:
				if st.button("🗑️", key=f"delete_{model_name}", help="Delete uploaded model"):
					os.remove(model_path)
					del st.session_state.uploaded_models[model_name]
					st.success(f"Deleted {model_name} model")
					st.rerun()
	
	# Clear all uploaded models button
	if st.session_state.uploaded_models:
		if st.button("🗑️ Clear All Uploaded Models", type="secondary"):
			cleanup_temp_directory()
			st.session_state.uploaded_models = {}
			st.success("All uploaded models cleared!")
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
	
	# Check which models are available (both local and uploaded)
	available_models = []
	model_status = {}
	
	# Check YOLO (only if detector is available)
	if YOLOv8Detector is not None:
		if get_model_path("yolo"):
			available_models.append("YOLO Only")
			model_status["YOLO Only"] = "✅ Available"
		else:
			model_status["YOLO Only"] = "❌ YOLO model required"
	else:
		model_status["YOLO Only"] = "❌ YOLO detector not available (ultralytics required)"
	
	# Check Hybrid (YOLO + VGG) (only if detector is available)
	if CombinedHelmetDetector is not None:
		if get_model_path("yolo") and get_model_path("vgg16"):
			available_models.append("Hybrid (YOLO+VGG)")
			model_status["Hybrid (YOLO+VGG)"] = "✅ Available"
		else:
			missing = []
			if not get_model_path("yolo"):
				missing.append("YOLO")
			if not get_model_path("vgg16"):
				missing.append("VGG16")
			model_status["Hybrid (YOLO+VGG)"] = f"❌ {', '.join(missing)} model(s) required"
	else:
		model_status["Hybrid (YOLO+VGG)"] = "❌ Hybrid detector not available (ultralytics required)"
	
	# Check RCNN
	if get_model_path("rcnn"):
		available_models.append("RCNN (Faster R-CNN)")
		model_status["RCNN (Faster R-CNN)"] = "✅ Available"
	else:
		model_status["RCNN (Faster R-CNN)"] = "❌ RCNN model required"
	
	# Check SSD
	if get_model_path("ssd"):
		available_models.append("SSD (SSD300-VGG16)")
		model_status["SSD (SSD300-VGG16)"] = "✅ Available"
	else:
		model_status["SSD (SSD300-VGG16)"] = "❌ SSD model required"
	
	# Check DETR
	if get_model_path("detr"):
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
	"""Run detection using hybrid model loading (local + uploaded)"""
	if choice.startswith("Hybrid"):
		if CombinedHelmetDetector is None:
			raise ImportError("Hybrid detector not available. Install ultralytics package.")
		yolo_path = get_model_path("yolo")
		vgg_path = get_model_path("vgg16")
		detector = CombinedHelmetDetector(yolo_path, vgg_path)
	elif choice.startswith("YOLO"):
		if YOLOv8Detector is None:
			raise ImportError("YOLO detector not available. Install ultralytics package.")
		yolo_path = get_model_path("yolo")
		detector = YOLOv8Detector(yolo_path)
	elif choice.startswith("RCNN"):
		if RCNNHelmetDetector is None:
			raise ImportError("RCNN detector not available.")
		rcnn_path = get_model_path("rcnn")
		detector = RCNNHelmetDetector(rcnn_path)
	elif choice.startswith("SSD"):
		if SSDHelmetDetector is None:
			raise ImportError("SSD detector not available.")
		ssd_path = get_model_path("ssd")
		detector = SSDHelmetDetector(ssd_path)
	else:  # DETR
		if DETRHelmetDetector is None:
			raise ImportError("DETR detector not available.")
		detr_path = get_model_path("detr")
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


