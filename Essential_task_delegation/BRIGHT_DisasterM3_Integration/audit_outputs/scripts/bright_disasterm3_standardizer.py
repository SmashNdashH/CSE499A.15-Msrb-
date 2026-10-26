"""
bright_disasterm3_standardizer.py
==================================
Standardizes 100% OPTICAL imagery from BRIGHT Dataset for DisasterM3 pipeline.
STRICT RULE: NO SAR RADAR IMAGERY IS PROCESSED. ONLY 100% OPTICAL RGB PHOTOS.

- Imagery: 1024x1024 3-channel Optical Satellite RGB (.png)
- Masks: 1024x1024 Building Footprint / Semantic Mask (uint8 .png):
    0: Background
    1: Building Footprint (Intact Pre-disaster)

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

# The 5 events whose optical pre-event imagery + masks are in our BRIGHT download.
# DisasterM3 holds copies of most of these tiles, but only as optical-pre + SAR-post pairs, masks
# and Bench files, which the optical segmentation pool never loads. bright_dm3_dedup.py checks
# every tile against that pool (train + frozen val) before training.
CLEAN_CATEGORY_A_EVENTS = [
    "la_palma-volcano",    # 933 tiles (La Palma volcano)
    "congo-volcano",       # 123 tiles (Goma volcano)
    "haiti-earthquake",    # 73 tiles (Les Cayes earthquake)
    "marshall-wildfire",   # 77 tiles (Boulder / Marshall wildfire)
    "noto-earthquake"      # 79 tiles (Noto earthquake)
]

# Events whose optical pre-event tiles BRIGHT does not distribute (its Google Earth imagery).
# DisasterM3 stores 618 of these 851 tiles under BRIGHT's tile ids, with building masks identical to
# BRIGHT's labels, as the optical pre images of SAR pairs (outside the optical pool), so they are
# taken from there. The masks still come from BRIGHT's target maps.
DM3_COPY_EVENTS = {
    "ukraine-conflict": "ukraine_conflict",    # 410 of 513 tiles
    "mexico-hurricane": "mexico_hurricane",    # 112 of 212 tiles (Acapulco)
    "myanmar-hurricane": "myanmar_hurricane",  # 96 of 126 tiles (Kyaukpyu)
}

def standardize_optical_tile(pre_optical_path, mask_path, out_img_dir, out_mask_dir, prefix):
    """
    Reads 100% OPTICAL pre-disaster RGB satellite photo and mask,
    ensures 1024x1024 dimensions, and writes clean PNGs.
    SAR radar is completely ignored.
    """
    # 1. Read optical RGB photo (3 channels)
    opt_img = cv2.imread(str(pre_optical_path))
    if opt_img is None:
        return False, "Failed to read optical pre-disaster image"

    # SAR or greyscale stored as RGB has three identical channels: not an optical photo.
    # Judge only pixels that carry data: black no-data borders have identical channels too.
    valid = opt_img.max(axis=2) > 3
    if valid.any():
        b, g, r = (opt_img[:, :, i][valid].astype(np.int16) for i in range(3))
        if np.abs(b - g).mean() < 1 and np.abs(g - r).mean() < 1:
            return False, "Single-channel image, not optical RGB"

    # 2. Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return False, "Failed to read ground-truth mask"

    # Multi-channel mask handling if needed
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    h, w = mask.shape[:2]

    # Verify 1024x1024 resolution
    if h != 1024 or w != 1024:
        opt_img = cv2.resize(opt_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # In pre-disaster optical imagery, all buildings before the disaster are standing footprints.
    # We produce clean building footprint masks (0: Background, 1: Building).
    # All building pixels (>0) are mapped to 1 (Building) for optical footprint training.
    footprint_mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    # Output file paths
    out_img_file = out_img_dir / f"{prefix}_optical_pre.png"
    out_mask_file = out_mask_dir / f"{prefix}_building_mask.png"

    cv2.imwrite(str(out_img_file), opt_img)
    cv2.imwrite(str(out_mask_file), footprint_mask)

    # Return mask stats
    unique_classes, counts = np.unique(footprint_mask, return_counts=True)
    stats = dict(zip([int(u) for u in unique_classes], [int(c) for c in counts]))
    return True, stats

def main():
    parser = argparse.ArgumentParser(description="Standardize 100% Optical BRIGHT tiles (NO SAR)")
    parser.add_argument("--bright_root", type=str, 
                        default=r"d:\CSE499A.15-Msrb-\dataset\BRIGHT Dataset", 
                        help="Root folder of raw BRIGHT dataset")
    parser.add_argument("--output_dir", type=str, 
                        default=r"d:\CSE499A.15-Msrb-\data\bright_standardized_optical", 
                        help="Output directory for standardized optical PNGs")
    parser.add_argument("--only_clean", action="store_true", default=True,
                        help="Process only non-overlapping Category A events")
    parser.add_argument("--dm3_root", type=str,
                        default=r"d:\CSE499A.15-Msrb-\dataset\DisasterM3 Dataset",
                        help="DisasterM3 root, source of the Ukraine / Mexico / Myanmar pre-event tiles")
    parser.add_argument("--skip_dm3_copies", action="store_true",
                        help="Leave out the Ukraine / Mexico / Myanmar tiles taken from DisasterM3")
    args = parser.parse_args()

    bright_root = Path(args.bright_root)
    target_dir = bright_root / "target" / "target"
    pre_optical_dir = bright_root / "pre-event" / "pre-event_wo_ukraine_myanmar_mexico"
    dm3_img_dir = Path(args.dm3_root) / "DisasterM3_Instruct" / "train_images" / "train_images"

    out_dir = Path(args.output_dir)
    out_img_dir = out_dir / "images"
    out_mask_dir = out_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  BRIGHT -> DisasterM3 Standardizer Pipeline (100% OPTICAL ONLY)")
    print("  * SAR Radar Imagery is EXCLUDED (Zero SAR)")
    print(f"  Source Root : {bright_root}")
    print(f"  Output Dir  : {out_dir}")
    print(f"  Clean Only  : {args.only_clean} (5 safe events)")
    print(f"  DM3 Copies  : {'skipped' if args.skip_dm3_copies else 'Ukraine, Mexico, Myanmar from ' + str(dm3_img_dir)}")
    print("=" * 70)

    if not target_dir.exists():
        print(f"[!] Error: target directory not found at {target_dir}")
        return

    all_targets = sorted(list(target_dir.glob("*_building_damage.tif")))
    print(f"[*] Found {len(all_targets)} total target masks in BRIGHT.")

    processed = 0
    skipped = 0
    total_stats = {0: 0, 1: 0}
    per_source = {}
    manifest = []

    for mask_path in tqdm(all_targets, desc="Standardizing Optical Tiles"):
        file_name = mask_path.name
        prefix = file_name.replace("_building_damage.tif", "")
        event, tile = prefix.rsplit("_", 1)

        # Look strictly for an optical pre-disaster image: BRIGHT's own, or DisasterM3's copy
        if event in DM3_COPY_EVENTS and not args.skip_dm3_copies:
            pre_optical_path = dm3_img_dir / f"{DM3_COPY_EVENTS[event]}_{tile}_pre_disaster.png"
            source = "DisasterM3 copy of the BRIGHT tile"
        elif args.only_clean and event not in CLEAN_CATEGORY_A_EVENTS:
            skipped += 1
            continue
        else:
            pre_optical_path = pre_optical_dir / f"{prefix}_pre_disaster.tif"
            source = "BRIGHT pre-event.zip"

        if not pre_optical_path.exists():
            skipped += 1
            continue

        success, res = standardize_optical_tile(pre_optical_path, mask_path, out_img_dir, out_mask_dir, prefix)
        if success:
            processed += 1
            per_source[source] = per_source.get(source, 0) + 1
            for k, v in res.items():
                total_stats[k] = total_stats.get(k, 0) + v

            manifest.append({
                "tile_id": prefix,
                "modality": "Optical RGB (Camera)",
                "optical_source": source,
                "image": f"images/{prefix}_optical_pre.png",
                "mask": f"masks/{prefix}_building_mask.png",
                "pixel_stats": res
            })
        else:
            skipped += 1

    # Save manifest
    manifest_file = out_dir / "bright_optical_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump({
            "total_optical_tiles": processed,
            "modality": "100% Optical Satellite RGB",
            "class_definition": {"0": "Background", "1": "Building Footprint"},
            "class_pixel_counts": total_stats,
            "tiles": manifest
        }, f, indent=2)

    total_px = sum(total_stats.values())
    print("\n" + "=" * 70)
    print("  100% OPTICAL STANDARDIZATION COMPLETE")
    print("=" * 70)
    print(f"  Processed Optical Tiles: {processed} {per_source}")
    print(f"  Skipped Tiles          : {skipped} (other events, missing or non-optical image)")
    print("  Pixel Distribution:")
    print(f"    0 - Background       : {total_stats[0]:,} ({total_stats[0]/max(1,total_px)*100:.2f}%)")
    print(f"    1 - Building Footprint: {total_stats[1]:,} ({total_stats[1]/max(1,total_px)*100:.2f}%)")
    print(f"  Manifest Saved         : {manifest_file}")
    print("  Next Step              : python bright_dm3_dedup.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
