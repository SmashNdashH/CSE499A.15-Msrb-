"""
support/bright_disasterm3_standardizer.py

Standardizes BRIGHT Dataset tiles (GeoTIFF / PNG) into DisasterM3-compatible
1024x1024 PNG images and 4-class semantic damage masks (0: Background, 1: Intact, 2: Damaged, 3: Destroyed).

Author: Ridita Afrin Riya (CSE499B Group 5)
Project: Resource-Efficient Neural Architectures for Post-Disaster Damage Assessment
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Clean Category A events from BRIGHT paper with zero overlap with DisasterM3 validation split
CLEAN_CATEGORY_A_EVENTS = [
    "la_palma",     # Spain - Volcano (933 tiles, 30k buildings)
    "ukraine",      # Armed conflict (513 tiles, 56k buildings)
    "acapulco",     # Mexico - Hurricane (212 tiles, 18k buildings)
    "kyaukpyu",     # Myanmar - Cyclone (126 tiles, 8k buildings)
    "goma",         # DR Congo - Volcano (123 tiles, 18k buildings)
    "les_cayes",    # Haiti - Earthquake (73 tiles, 18k buildings)
    "boulder",      # USA - Wildfire (77 tiles, 8k buildings)
    "noto"          # Japan - Earthquake (79 tiles, 8k buildings)
]

def standardize_tile(pre_optical_path, post_optical_path, mask_path, out_img_dir, out_mask_dir, tile_id):
    """
    Reads BRIGHT tiles, ensures 1024x1024 dimensions, validates 4-class mask,
    and writes standardized PNGs.
    """
    pre_img = cv2.imread(str(pre_optical_path))
    if pre_img is None:
        return False, "Failed to read pre-disaster image"
        
    post_img = None
    if post_optical_path and os.path.exists(post_optical_path):
        post_img = cv2.imread(str(post_optical_path))

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return False, "Failed to read ground-truth mask"

    # Multi-channel mask handling (take single channel if needed)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    h, w = mask.shape[:2]

    # Resize/crop to 1024x1024 if resolution differs
    if h != 1024 or w != 1024:
        pre_img = cv2.resize(pre_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        if post_img is not None:
            post_img = cv2.resize(post_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # Strictly enforce 4 classes: 0 (Background), 1 (Intact), 2 (Damaged), 3 (Destroyed)
    mask = np.clip(mask, 0, 3).astype(np.uint8)

    # Output file paths
    out_pre_file = out_img_dir / f"bright_{tile_id}_pre_disaster.png"
    out_post_file = out_img_dir / f"bright_{tile_id}_post_disaster.png"
    out_mask_file = out_mask_dir / f"bright_{tile_id}_target.png"

    cv2.imwrite(str(out_pre_file), pre_img)
    if post_img is not None:
        cv2.imwrite(str(out_post_file), post_img)
    cv2.imwrite(str(out_mask_file), mask)

    # Return mask stats
    unique_classes, counts = np.unique(mask, return_counts=True)
    stats = dict(zip(unique_classes.tolist(), counts.tolist()))
    return True, stats

def main():
    parser = argparse.ArgumentParser(description="Convert BRIGHT dataset to DisasterM3 format")
    parser.add_argument("--bright_dir", type=str, default="data/BRIGHT", help="Path to raw BRIGHT dataset root")
    parser.add_argument("--output_dir", type=str, default="data/bright_processed", help="Path to output standardized data")
    parser.add_argument("--only_clean", action="store_true", default=True, help="Only process Category A non-overlapping events")
    args = parser.parse_args()

    bright_dir = Path(args.bright_dir)
    out_dir = Path(args.output_dir)
    out_img_dir = out_dir / "images"
    out_mask_dir = out_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f"BRIGHT -> DisasterM3 Standardizer Pipeline")
    print(f"Source Directory: {bright_dir}")
    print(f"Target Directory: {out_dir}")
    print(f"Filter Clean Category A Events: {args.only_clean}")
    print(f"============================================================")

    # Search for target masks
    mask_files = list(bright_dir.glob("**/target/*.*")) + list(bright_dir.glob("**/*target*.*"))
    if not mask_files:
        print(f"[!] No masks found in {bright_dir}. Please ensure BRIGHT dataset is extracted.")
        return

    processed = 0
    skipped = 0
    total_stats = {0: 0, 1: 0, 2: 0, 3: 0}

    for mask_path in tqdm(mask_files, desc="Standardizing tiles"):
        tile_name = mask_path.stem.replace("_target", "")
        
        # Check event filter
        if args.only_clean:
            is_clean = any(event in tile_name.lower() or event in str(mask_path).lower() for event in CLEAN_CATEGORY_A_EVENTS)
            if not is_clean:
                skipped += 1
                continue

        # Look for corresponding pre and post optical imagery
        pre_optical = None
        post_optical = None
        
        # Search candidate paths
        parent = mask_path.parent.parent
        pre_candidates = list(parent.glob(f"pre-event/{tile_name}*.*")) + list(parent.glob(f"**/{tile_name}*pre*.*"))
        post_candidates = list(parent.glob(f"post-event/{tile_name}*.*")) + list(parent.glob(f"**/{tile_name}*post*.*"))

        if pre_candidates:
            pre_optical = pre_candidates[0]
        if post_candidates:
            post_optical = post_candidates[0]

        if not pre_optical:
            skipped += 1
            continue

        success, res = standardize_tile(pre_optical, post_optical, mask_path, out_img_dir, out_mask_dir, tile_name)
        if success:
            processed += 1
            for k, v in res.items():
                total_stats[k] = total_stats.get(k, 0) + v
        else:
            skipped += 1

    print("\n--- Standardization Summary ---")
    print(f"Successfully processed: {processed} tiles")
    print(f"Skipped / Filtered: {skipped} tiles")
    print(f"Class pixel counts: {total_stats}")

if __name__ == "__main__":
    main()
