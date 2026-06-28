# EDA Notebook Spec: Source Data Analysis (Pre-Pipeline)

## Purpose

This document is a delegatable specification for an AI agent to build a full, runnable EDA notebook (Jupyter `.ipynb` or Python script) focused on **Source Data Analysis**. 

This EDA is meant to be run **before** any data pipelines are built. Its purpose is to analyze the raw, unprocessed imagery and native metadata of disaster datasets (e.g., **xBD**, **FloodNet**, **AIDER**, **RescueNet**) to uncover anomalies, imbalances, and varying resolutions. The insights generated from this notebook provide the mathematical and statistical justification for the engineering rules implemented in down-stream pipeline scripts (like `Image_standardizer.py` and `generate_ground_truth.py`).

---

## Agent Instructions

- Produce a single Jupyter notebook (`eda_source_data.ipynb`) with clearly separated cells per stage.
- Each stage must include: (a) executable code, (b) inline markdown explanation, (c) output visualizations using `matplotlib` or `seaborn`.
- All file paths should use `pathlib.Path` and be configurable via a `CONFIG` dict at the top of the notebook.
- Code must run on a standard data science environment (e.g., local Jupyter or Kaggle) with Python 3.10+.
- Install commands go in the first cell using `!pip install <package>`.
- The final output cell must generate an "Engineering Requirements Checklist" based on the findings.

---

## CONFIG Block (Cell 1)

```python
CONFIG = {
    "datasets": {
        "xbd": {
            "images_dir": "data/xView2 Challenge Dataset - train and test/train/images",
            "labels_dir": "data/xView2 Challenge Dataset - train and test/train/labels"
        },
        # Future datasets scaffolding
        # "floodnet": {"images_dir": "data/FloodNet/images", "labels_dir": "data/FloodNet/labels"},
        # "aider":    {"images_dir": "data/AIDER/images",    "labels_dir": "data/AIDER/labels"},
    },
    "output_dir": "dataset/eda_source_outputs",
    "target_resolution_floor": 256,
}
```

---

## Stage 1: Raw Asset Inventory

### Goal
Count raw images and verify every pre-disaster and post-disaster image has a matching raw JSON label file (for datasets that separate them). Identify orphaned files or missing metadata.

### Required outputs
- Printed summary table: dataset → total images → total labels → orphaned files
- Separation of pre-event vs post-event imagery counts (xBD specific).

### Code scaffold

```python
import os
from pathlib import Path
from collections import defaultdict

def inventory_dataset(cfg):
    img_dir = Path(cfg["images_dir"])
    lbl_dir = Path(cfg.get("labels_dir", ""))
    
    if not img_dir.exists():
        return None
        
    images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    labels = list(lbl_dir.glob("*.json")) if lbl_dir.exists() else []
    
    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}
    
    orphaned_images = img_stems - lbl_stems
    orphaned_labels = lbl_stems - img_stems
    
    pre_disaster = sum(1 for p in images if "_pre_" in p.name)
    post_disaster = sum(1 for p in images if "_post_" in p.name)
    
    return {
        "total_images": len(images),
        "total_labels": len(labels),
        "pre_event_imgs": pre_disaster,
        "post_event_imgs": post_disaster,
        "orphans": len(orphaned_images) + len(orphaned_labels)
    }

for name, cfg in CONFIG["datasets"].items():
    stats = inventory_dataset(cfg)
    if stats:
        print(f"=== {name.upper()} ===")
        print(f"  Images: {stats['total_images']} (Pre: {stats['pre_event_imgs']}, Post: {stats['post_event_imgs']})")
        print(f"  Labels: {stats['total_labels']}")
        print(f"  Orphans: {stats['orphans']}")
```

---

## Stage 2: Raw Visual Quality Audit

### Goal
Analyze raw image properties (PNG vs JPEG, native resolutions, aspect ratios) to justify standardizer logic.

### Required outputs
- Histogram: distribution of image resolutions (shorter edge).
- Report on file formats (PNG vs JPEG).
- Report on how many images fall below the `target_resolution_floor` (256px).

### Code scaffold

```python
from PIL import Image

def analyze_resolutions(cfg):
    img_dir = Path(cfg["images_dir"])
    if not img_dir.exists(): return []
    
    sizes = []
    for p in list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")):
        try:
            with Image.open(p) as img:
                sizes.append({"path": str(p), "w": img.width, "h": img.height, "min_edge": min(img.width, img.height)})
        except Exception as e:
            print(f"Error reading {p.name}: {e}")
    return sizes

# ... logic to plot histogram of 'min_edge' ...
# ... print count of images where min_edge < 256 ...
```

---

## Stage 3: Native Metadata Parsing

### Goal
Extract metadata directly from the deeply nested source JSONs. For xBD, we need to extract disaster events and damage severity.

### Code scaffold (xBD specific focus)

```python
import json
from collections import Counter

def parse_xbd_labels(cfg):
    lbl_dir = Path(cfg["labels_dir"])
    if not lbl_dir.exists(): return []
    
    records = []
    for p in lbl_dir.glob("*.json"):
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
            
        metadata = data.get("metadata", {})
        disaster_type = metadata.get("disaster_type", "unknown")
        
        # xBD stores building polygons in features; we want to summarize them
        features = data.get("features", {"xy": []})
        damage_counts = Counter()
        
        for poly in features.get("xy", []):
            subtype = poly.get("properties", {}).get("subtype", "unclassified")
            damage_counts[subtype] += 1
            
        records.append({
            "image_id": p.stem,
            "event": p.stem.rsplit("_", 3)[0] if "_" in p.stem else "unknown",
            "disaster_type": disaster_type,
            "damage_summary": dict(damage_counts)
        })
    return records

xbd_records = parse_xbd_labels(CONFIG["datasets"]["xbd"])
```

---

## Stage 4: Imbalance & Geographic Skew Analysis

### Goal
Plot the massive imbalances native to disaster datasets. Uncovering these imbalances justifies the pipeline's oversampling/stratification strategies.

### Required outputs
- Bar chart: Total building instances by damage class (e.g., "no-damage" vs "destroyed").
- Bar chart: Image count by disaster event (e.g., `hurricane-michael` vs `guatemala-volcano`).

### Code scaffold

```python
import matplotlib.pyplot as plt

# Aggregate building damage classes across all xBD records
total_damage = Counter()
event_counts = Counter()

for r in xbd_records:
    event_counts[r["event"]] += 1
    for dmg_class, count in r["damage_summary"].items():
        total_damage[dmg_class] += count

# Imbalance Calculation
print("Building Damage Class Imbalance:")
for k, v in total_damage.most_common():
    print(f"  {k}: {v}")

print("\nGeographic Event Imbalance:")
for k, v in event_counts.most_common():
    print(f"  {k}: {v}")

# ... plotting logic using matplotlib ...
```
> **Expected Discovery:** The analyst will discover a ~10-20x imbalance between "no-damage" and "destroyed", and heavily skewed geographic representation (e.g. 4 distinct hurricanes dominating the top spots).

---

## Stage 5: Cross-Dataset Taxonomy Design

### Goal
Design a mapping strategy to unify disparate raw labels into a single taxonomy. This acts as the blueprint for `generate_ground_truth.py`.

### Taxonomy Map

```python
# A mapping dictionary defining how different raw dataset labels map to our VLM schema
UNIFIED_TAXONOMY = {
    # xBD (Building Damage)
    "no-damage":    {"type": "structural", "severity": "none",     "action": "clear"},
    "minor-damage": {"type": "structural", "severity": "low",      "action": "monitor"},
    "major-damage": {"type": "structural", "severity": "high",     "action": "respond"},
    "destroyed":    {"type": "structural", "severity": "critical", "action": "evacuate"},
    
    # FloodNet (Water specific)
    "flooded":      {"type": "flood",      "severity": "high",     "action": "respond"},
    "non-flooded":  {"type": "flood",      "severity": "none",     "action": "clear"},
    
    # AIDER (Event level)
    "collapsed_building": {"type": "structural", "severity": "critical", "action": "evacuate"},
    "fire":               {"type": "fire",       "severity": "high",     "action": "respond"},
}
```

---

## Stage 6: Pipeline Engineering Requirements

### Goal
Based on the discoveries in Stages 1-4 and the mapping in Stage 5, this cell prints the exact **Engineering Contract** that the subsequent data pipeline must fulfill.

### Required Output Printout
The notebook should literally `print()` a markdown-formatted string resembling this:

```markdown
### REQUIRED PIPELINE LOGIC (Based on EDA)

**1. Image Standardizer (`Image_standardizer.py`) MUST:**
- Resize large images to a maximum edge of 1280px to save GPU VRAM (discovered from Stage 2 resolution histograms).
- Filter out and drop any images that scale below 256x256 pixels to preserve structural context (discovered from Stage 2 minimum edge checks).
- Convert all images to RGB and save as quality-95 JPEGs to normalize PNG vs JPEG discrepancies.

**2. Generator (`generate_ground_truth.py`) MUST:**
- Accept both the image and the parsed metadata (from Stage 3) as prompt inputs.
- Tune the LLM prompt dynamically based on the disaster type (e.g. emphasize water depth for floods, voids for earthquakes).

**3. Splitter (`dataset_split.py`) MUST:**
- Group images by geographic disaster event *before* executing the Train/Val/Test split (discovered from Stage 4 geographic skew) to prevent data leakage.
```

---

## Dependencies (install cell)

```bash
pip install pillow matplotlib seaborn
```
