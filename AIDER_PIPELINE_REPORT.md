# AIDER Dataset Integration Pipeline Report

This document outlines the pipeline developed to integrate the **AIDER (Aerial Imagery for Disaster Environment Recognition)** dataset into our Multimodal VLM training pipeline. The goal of this pipeline is to standardize the raw AIDER imagery and synthesize rich metadata labels so they match the structure and quality of the xBD dataset.

## Pipeline Overview

Our custom pipeline consists of two main stages:
1. **Image Standardization** (`support/aider_image_standardizer.py`)
2. **Metadata Synthesis via Gemini Vision** (`support/aider_label_synthesizer.py`)

---

### Stage 1: Image Standardization
**Script:** `support/aider_image_standardizer.py`

The raw AIDER dataset contains images of various resolutions, aspect ratios, and color profiles. To ensure our Vision Language Model (VLM) trains efficiently and avoids Out-Of-Memory (OOM) errors, we process all imagery through our standardizer.

**Key Operations:**
- **Color Correction:** Drops alpha channels and converts any RGBA or grayscale images into standard RGB format.
- **Resolution Capping:** Dynamically downscales any images exceeding a maximum edge of `1280px` while preserving the original aspect ratio (Lanczos resampling).
- **Minimum Resolution Gate:** Rejects images smaller than `256x256` to ensure the model only trains on high-quality features.
- **Optimization:** Saves the output cleanly as optimized JPEG files (`quality=95`).
- **Tracking & Deduplication:** Computes an MD5 hash of each output file and maintains a `processed_tracker.txt` file. This allows the script to safely pause and resume without duplicating work.

**Output Location:** 
- Processed Images: `data/aider/processed_images/`

---

### Stage 2: Metadata Synthesis (Gemini Vision)
**Script:** `support/aider_label_synthesizer.py`

Unlike the xBD dataset, the AIDER dataset only provides simple categorical folder labels (e.g., `Collapsed_Building`). To train a tactical multimodal model, we need rich, descriptive JSON metadata for each image. 

We built a synthesizer script that connects to the **Google Gemini API (`gemini-flash-latest`)** to visually analyze each standardized AIDER image and generate an xBD-style JSON file.

**Key Operations:**
- **Dynamic Key Rotation:** The script securely loads up to 7 distinct API keys from a `.env` file. If the Google free-tier rate limit is reached (`429 Quota Exhausted`) or a key is invalid, the script instantly rotates to the next API key without stopping the process.
- **Vision Analysis:** Each image is uploaded to Gemini Vision with a strict system prompt. The model analyzes the disaster scene and outputs a structured JSON describing the event type, damage severity, and visual features.
- **xBD Format Mimicry:** The generated JSON is saved in a format identical to our existing xBD labels, allowing it to seamlessly plug directly into our downstream `generate_ground_truth.py` script.

**Output Location:** 
- Synthesized JSON Labels: `data/aider/processed_labels/`

---

## Next Steps: Final Dataset Generation
Because the AIDER images and labels have been successfully converted into an xBD-compatible format, the final step is to merge them into the VLM training set. 

You can run the original generation script to compile both the xBD and AIDER processed data into the final `train_dataset.jsonl` file:

```bash
python support/generate_ground_truth.py
```
