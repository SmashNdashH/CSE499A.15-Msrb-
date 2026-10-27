"""
config.py
---------
Configuration parameters for Tamanna's Agentic Vision RAG Pipeline (Plan 2).
Group 5 - CSE499B Capstone Sprint.

Supports switching between:
  1. Team Fine-Tuned Model: AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP
  2. Base Foundation Model: Qwen/Qwen2.5-VL-7B-Instruct
PaliGemma 2 and the Qwen2.5-VL-3B edge entry were dropped on 27 Oct 2026: the edge model is L2's job.
"""

import os
from pathlib import Path

# ─── Environment & Paths ─────────────────────────────────────────────────────

# Base directory for Plan 2
PLAN2_DIR = Path(__file__).resolve().parent

# Check if running inside Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle/working")

# Default input and output directories
if IS_KAGGLE:
    DEFAULT_DATA_DIR = "/kaggle/input/disasterm3-dataset"
    DEFAULT_WEIGHTS_PATH = "/kaggle/input/disasterm3-unet-champion/final_model.pth"
    DEFAULT_OUTPUT_DIR = "/kaggle/working/agentic_visrag_results"
else:
    DEFAULT_DATA_DIR = str(PLAN2_DIR.parents[2] / "dataset" / "disasterm3")
    DEFAULT_WEIGHTS_PATH = str(PLAN2_DIR.parents[1] / "Champion_Task" / "final_model.pth")
    DEFAULT_OUTPUT_DIR = str(PLAN2_DIR / "outputs")

# Hugging Face Access Token (Required for private repos such as the R9 checkpoint)
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Segmentation Backbone Configurations
# - "resnet50": the D4 recipe's encoder (R1, R9: U-Net + ResNet-50, phased loss)
# - "resnet34": the B1 architecture baseline's encoder (U-Net + ResNet-34, CE+Dice)
DEFAULT_SEGMENTATION_ENCODER = "resnet50"
BASELINE_SEGMENTATION_ENCODER = "resnet34"

# Official Team Champion Checkpoint Repositories on Hugging Face
CHAMPION_HF_REPOS = [
    {"repo_id": "AbrarAlam/disasterm3-d4-R9", "filename": "final_model.pth", "encoder": "resnet50", "desc": "R9 (D4 recipe + sampler + copy-paste)"},
    {"repo_id": "AbrarAlam/disasterm3-d4-R1", "filename": "final_model.pth", "encoder": "resnet50", "desc": "R1 (D4 reference recipe: phased loss, 768 px native crops)"},
    {"repo_id": "AbrarAlam/disasterm3-unet-checkpoints-2", "filename": "best_model.pth", "encoder": "resnet34", "desc": "B1 baseline (ResNet-34, CE+Dice)"}
]

# Ensure output directory exists
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Model Registry
VLM_MODELS = {
    "finetuned": {
        "name": "Team_Finetuned_Qwen2.5VL_7B",
        "model_id": "AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP",
        "model_type": "qwen",
        "is_gated": False,
        "description": "Our custom fine-tuned VLM trained on DisasterM3 satellite imagery."
    },
    "base": {
        "name": "Base_Qwen2.5VL_7B",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_type": "qwen",
        "is_gated": False,
        "description": "Vanilla base foundation model used to measure zero-shot and fine-tuning delta."
    }
}

# ─── Segmentation & Visual Crop Parameters ───────────────────────────────────

# Classes recognized by the champion U-Net
CLASS_MAPPING = {
    0: "Background",
    1: "Intact",
    2: "Damaged",
    3: "Destroyed"
}

# Color palette for visualizations (RGB format for Matplotlib)
CLASS_COLORS_RGB = {
    "Background": (128, 128, 128),
    "Intact": (46, 204, 113),       # Emerald Green
    "Damaged": (243, 156, 18),      # Orange
    "Destroyed": (231, 76, 60),     # Vibrant Red
}

# Minimum connected-component pixel area to filter out sensor noise
MIN_BUILDING_AREA_PX = 80

# Maximum building bounding box to guard against merged conglomerate masks
MAX_BUILDING_AREA_PX = 50000

# Native-resolution crop size (in pixels) fed into the VLM
CROP_SIZE = 256

# Padding (in pixels) added around each building bounding box
CROP_PADDING = 32

# ─── EMS-98 Knowledge Scale ──────────────────────────────────────────────────

EMS98_GRADES = {
    "Grade 1": {
        "name": "Negligible to slight damage",
        "damage_type": "Hair-line cracks in walls; slight plaster fall.",
        "structural_failure": False,
        "rescue_priority": "Low",
        "recommended_action": "Safe for habitation after standard inspection."
    },
    "Grade 2": {
        "name": "Moderate damage",
        "damage_type": "Cracks in walls; chimney collapse; roof tiles dislodged.",
        "structural_failure": False,
        "rescue_priority": "Standard",
        "recommended_action": "Restricted entry; structural monitoring needed."
    },
    "Grade 3": {
        "name": "Substantial to heavy damage",
        "damage_type": "Large cracks; wall buckling; non-structural collapse.",
        "structural_failure": True,
        "rescue_priority": "High",
        "recommended_action": "Immediate evacuation; perimeter cordon."
    },
    "Grade 4": {
        "name": "Very heavy damage",
        "damage_type": "Partial collapse of roofs and floors; severe structural failure.",
        "structural_failure": True,
        "rescue_priority": "Immediate",
        "recommended_action": "Priority Urban Search and Rescue (USAR); shoring required."
    },
    "Grade 5": {
        "name": "Destruction",
        "damage_type": "Total or near-total collapse; building reduced to rubble heap.",
        "structural_failure": True,
        "rescue_priority": "Immediate",
        "recommended_action": "Critical USAR deployment; heavy extraction machinery."
    }
}
