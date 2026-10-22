"""
config.py
---------
Configuration parameters for Tamanna's Agentic Vision RAG Pipeline (Plan 2).
Group 5 - CSE499B Capstone Sprint.

Supports seamless switching between:
  1. Team Fine-Tuned Model: AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP
  2. Base Foundation Model: Qwen/Qwen2.5-VL-7B-Instruct
  3. Google Multimodal Benchmark: google/paligemma2-3b-pt-448 (or 10b)
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

# Hugging Face Access Token (Required for gated models like Google PaliGemma 2)
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Segmentation Backbone Configurations
# - "resnet50": Team Champion model (Fullstack ResNet-50 + Phased Loss, mIoU 0.5449)
# - "resnet34": Architecture ablation baseline (Tamanna B1 baseline, mIoU 0.4619)
DEFAULT_SEGMENTATION_ENCODER = "resnet50"
BASELINE_SEGMENTATION_ENCODER = "resnet34"

# Official Team Champion Checkpoint Repositories on Hugging Face
CHAMPION_HF_REPOS = [
    {"repo_id": "AbrarAlam/disasterm3-d4-R9", "filename": "final_model.pth", "encoder": "resnet50", "desc": "Champion R9 (Sampler + CopyPaste, Destroyed IoU 0.2971)"},
    {"repo_id": "AbrarAlam/disasterm3-d4-R1", "filename": "final_model.pth", "encoder": "resnet50", "desc": "Champion R1 (Phased Loss + 768 native crops, mIoU 0.5401)"},
    {"repo_id": "AbrarAlam/disasterm3-unet-checkpoints-2", "filename": "best_model.pth", "encoder": "resnet34", "desc": "B1 Reference Baseline (ResNet-34 CE+Dice, mIoU 0.4619)"}
]

# Ensure output directory exists
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Auto-detect if PaliGemma 2 is attached as a local Kaggle model
import glob
_local_paligemma = glob.glob("/kaggle/input/**/paligemma*/**/config.json", recursive=True)
PALIGEMMA_MODEL_ID = os.path.dirname(_local_paligemma[0]) if _local_paligemma else "google/paligemma2-3b-pt-448"

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
    },
    "qwen_edge": {
        "name": "Qwen2.5VL_3B_Edge",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_type": "qwen",
        "is_gated": False,
        "description": "Un-gated lightweight 3B edge VLM: zero token requirements, zero 401 errors, ~3.1 GB VRAM."
    },
    "paligemma": {
        "name": "Google_PaliGemma2_3B",
        "model_id": PALIGEMMA_MODEL_ID,
        "model_type": "paligemma",
        "is_gated": not bool(_local_paligemma),
        "description": f"Google's lightweight 3B multimodal model specialized for spatial understanding ({'Local Kaggle Model' if _local_paligemma else 'HuggingFace Hub [Gated - HF_TOKEN required]'})."
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
