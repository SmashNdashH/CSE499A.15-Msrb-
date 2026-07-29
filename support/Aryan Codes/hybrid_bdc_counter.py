"""
hybrid_bdc_counter.py
---------------------
Programmatic Building Damage Counter using OpenCV contour analysis on
segmentation masks. Designed to work with either:
  (a) Aryan's U-Net output masks (classes: 0=Background, 1=Intact, 2=Damaged)
  (b) Raw xBD ground-truth masks  (classes: 0=BG, 1=No Damage, 2=Minor, 3=Major, 4=Destroyed)

This script completely bypasses the VLM for the BDC (Building Damage Counting)
task, fixing the -10.62% regression caused by the 512x28x28 resolution cap (D4).

Usage:
    # Count damaged buildings in a single mask
    python hybrid_bdc_counter.py --mask path/to/mask.png

    # Count in all masks in a directory
    python hybrid_bdc_counter.py --mask_dir path/to/masks/ --output counts.csv

    # Use raw xBD 5-class masks instead of simplified 3-class
    python hybrid_bdc_counter.py --mask_dir path/to/masks/ --mode xbd_raw
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np


# ─── Configuration ───────────────────────────────────────────────────────────

# Minimum contour area in pixels to be considered a building (filters noise)
# A single xBD building footprint at 1024x1024 is typically >= 100 pixels
MIN_BUILDING_AREA_PX = 50

# Maximum contour area — reject contours larger than this (likely merged blobs)
MAX_BUILDING_AREA_PX = 50000


# ─── Core Counting Logic ────────────────────────────────────────────────────

def count_damaged_buildings(
    mask: np.ndarray,
    mode: str = "simplified",
    min_area: int = MIN_BUILDING_AREA_PX,
    max_area: int = MAX_BUILDING_AREA_PX,
) -> dict:
    """
    Count individual damaged buildings from a segmentation mask using
    OpenCV contour detection.

    Args:
        mask: 2D numpy array (H, W) with integer class labels.
        mode: "simplified" for 3-class U-Net output (0=BG, 1=Intact, 2=Damaged),
              "xbd_raw" for 5-class xBD masks (0=BG, 1=NoDamage, 2=Minor, 3=Major, 4=Destroyed).
        min_area: Minimum contour area in pixels to count as a building.
        max_area: Maximum contour area in pixels (rejects merged blobs).

    Returns:
        Dictionary with counts and metadata.
    """
    assert mask.ndim == 2, f"Expected 2D mask, got shape {mask.shape}"

    # ── Step 1: Create binary mask of "damaged" pixels ──
    if mode == "simplified":
        # Aryan's U-Net: class 2 = Damaged
        binary = (mask == 2).astype(np.uint8) * 255
    elif mode == "xbd_raw":
        # Raw xBD: classes 2, 3, 4 = Minor, Major, Destroyed
        binary = (mask >= 2).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'simplified' or 'xbd_raw'.")

    # ── Step 2: Morphological cleanup ──
    # Close small gaps within building footprints
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Open to remove small noise specks
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # ── Step 3: Find contours ──
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── Step 4: Filter by area ──
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            valid_contours.append(contour)

    # ── Step 5: Compute statistics ──
    total_damaged_px = int(np.sum(binary > 0))
    total_px = mask.shape[0] * mask.shape[1]
    damaged_area_pct = (total_damaged_px / total_px) * 100 if total_px > 0 else 0.0

    return {
        "building_count": len(valid_contours),
        "total_contours_raw": len(contours),
        "filtered_out": len(contours) - len(valid_contours),
        "damaged_pixels": total_damaged_px,
        "total_pixels": total_px,
        "damaged_area_pct": round(damaged_area_pct, 2),
    }


def count_by_severity(mask: np.ndarray, min_area: int = MIN_BUILDING_AREA_PX) -> dict:
    """
    For raw xBD 5-class masks, count buildings by individual severity level.

    Returns:
        Dictionary with per-severity counts.
    """
    assert mask.ndim == 2, f"Expected 2D mask, got shape {mask.shape}"

    results = {}
    severity_map = {
        "minor_damage": 2,
        "major_damage": 3,
        "destroyed": 4,
    }

    for severity_name, class_id in severity_map.items():
        binary = (mask == class_id).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        results[severity_name] = len(valid)

    results["total_damaged"] = sum(results.values())
    return results


def extract_bounding_boxes(
    mask: np.ndarray,
    mode: str = "simplified",
    min_area: int = MIN_BUILDING_AREA_PX,
    max_area: int = MAX_BUILDING_AREA_PX,
    padding: int = 20,
) -> list:
    """
    Extract bounding boxes of damaged building clusters for VLM cropping.

    Args:
        mask: 2D segmentation mask.
        mode: "simplified" or "xbd_raw".
        padding: Extra pixels to add around each bounding box.

    Returns:
        List of (x, y, w, h) tuples representing bounding boxes.
    """
    if mode == "simplified":
        binary = (mask == 2).astype(np.uint8) * 255
    else:
        binary = (mask >= 2).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, bw, bh = cv2.boundingRect(contour)
            # Add padding, clamped to image bounds
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


# ─── CLI ─────────────────────────────────────────────────────────────────────

def process_single_mask(mask_path: str, mode: str) -> dict:
    """Load and process a single mask file."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [SKIP] Could not read: {mask_path}")
        return None

    result = count_damaged_buildings(mask, mode=mode)
    result["filename"] = os.path.basename(mask_path)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Count damaged buildings from segmentation masks using OpenCV contours."
    )
    parser.add_argument("--mask", type=str, help="Path to a single mask PNG file.")
    parser.add_argument("--mask_dir", type=str, help="Path to directory of mask PNG files.")
    parser.add_argument(
        "--mode",
        choices=["simplified", "xbd_raw"],
        default="simplified",
        help="Mask class scheme: 'simplified' (3-class U-Net) or 'xbd_raw' (5-class xBD).",
    )
    parser.add_argument("--output", type=str, help="Path to save CSV output.")
    parser.add_argument(
        "--min_area",
        type=int,
        default=MIN_BUILDING_AREA_PX,
        help=f"Minimum contour area in pixels (default: {MIN_BUILDING_AREA_PX}).",
    )
    args = parser.parse_args()

    if not args.mask and not args.mask_dir:
        parser.error("Provide --mask or --mask_dir.")

    results = []

    if args.mask:
        r = process_single_mask(args.mask, args.mode)
        if r:
            results.append(r)
            print(f"\n  File: {r['filename']}")
            print(f"  Damaged buildings: {r['building_count']}")
            print(f"  Damaged area: {r['damaged_area_pct']}%")
            print(f"  Raw contours: {r['total_contours_raw']} (filtered: {r['filtered_out']})")

    if args.mask_dir:
        mask_files = sorted(
            f for f in os.listdir(args.mask_dir) if f.lower().endswith(".png")
        )
        print(f"\nProcessing {len(mask_files)} masks from: {args.mask_dir}")

        for fname in mask_files:
            fpath = os.path.join(args.mask_dir, fname)
            r = process_single_mask(fpath, args.mode)
            if r:
                results.append(r)

        # Summary
        total_buildings = sum(r["building_count"] for r in results)
        avg_buildings = total_buildings / len(results) if results else 0
        print(f"\n  Total masks processed: {len(results)}")
        print(f"  Total damaged buildings found: {total_buildings}")
        print(f"  Average per image: {avg_buildings:.1f}")

    # Save CSV if requested
    if args.output and results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fieldnames = [
            "filename",
            "building_count",
            "total_contours_raw",
            "filtered_out",
            "damaged_pixels",
            "total_pixels",
            "damaged_area_pct",
        ]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  CSV saved to: {args.output}")


if __name__ == "__main__":
    main()
