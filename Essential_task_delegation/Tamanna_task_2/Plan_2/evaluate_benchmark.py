"""
evaluate_benchmark.py
---------------------
Runs the agent with each VLM over a folder of tiles:
  1. Team Fine-Tuned Model: AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP
  2. Base Foundation Model: Qwen/Qwen2.5-VL-7B-Instruct

Records, per tile and per model:
  - Building counts from the segmenter
  - Agreement between the VLM's class and the segmenter's class (not accuracy; no ground truth is used)
  - Inference Latency per tile (seconds)
  - Peak GPU memory allocated by PyTorch (GB)

This script has never been run. Its tile selection is not the team's validation split;
the notebooks select tiles from that split.

Outputs:
  - vlm_benchmark_results.csv
  - vlm_benchmark_summary.json

Author: Tamanna (Member B, Group 5)
Project: CSE499B DisasterM3 Capstone
"""

import os
import re
import sys
import time
import json
import csv
import argparse
import glob
from typing import List, Dict, Any
import numpy as np
import cv2
import torch

import config
from tools import SegmentationLocatorTool, HighResVisualRetrieverTool, EMS98KnowledgeRAGTool
from vlm_engine import MultiVLMEngine
from agent import AgenticVisionRAG
from visualizer import generate_annotated_damage_map


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Benchmark (agreement with the segmenter, not accuracy)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.DEFAULT_DATA_DIR,
        help="Path to DisasterM3 validation image directory"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=config.DEFAULT_WEIGHTS_PATH,
        help="Path to segmentation model weights (final_model.pth)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.DEFAULT_OUTPUT_DIR,
        help="Where to write CSV and JSON summary results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of validation images to evaluate per model"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["finetuned", "base"],
        help="List of model keys to benchmark: finetuned, base"
    )
    return parser.parse_args()


def get_validation_images(data_dir: str, max_count: int = 10) -> List[str]:
    """
    Finds satellite imagery tiles under data_dir: a folder named 'val' or 'test' if there is one,
    otherwise a seeded 10% sample of every image found. Neither is the team's validation split,
    and file names don't say whether a tile is optical or radar.
    """
    patterns = ["*post_disaster*.png", "*.png", "*.jpg", "*.tif"]
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(data_dir, "**", pat), recursive=True))

    if not found:
        print(f"[Warning] No images found in {data_dir}. Falling back to search in parent directories...")
        for pat in patterns:
            found.extend(glob.glob(os.path.join(str(config.PLAN2_DIR.parents[2]), "**", pat), recursive=True))

    # Search for validation / holdout splits first
    val_candidates = [
        p for p in found
        if "val" in p.replace("\\", "/").lower() or "test" in p.replace("\\", "/").lower()
    ]

    if val_candidates:
        clean_candidates = val_candidates
        print(f"[Dataset Split] Found {len(clean_candidates)} isolated validation/test images.")
    else:
        # A seeded 10% sample of the file list; this is not the team's train/validation split
        print("[Dataset Split] No 'val' folder found. Taking a seeded 10% sample of the images found "
              "(not the team's validation split)...")
        # Sort deterministically by filename
        sorted_found = sorted(found)
        import random
        rng = random.Random(42)
        indices = list(range(len(sorted_found)))
        rng.shuffle(indices)
        val_size = max(1, int(len(sorted_found) * 0.10))
        val_indices = set(indices[-val_size:])
        clean_candidates = [sorted_found[i] for i in val_indices]
        print(f"[Dataset Split] Sampled {len(clean_candidates)} images; they may include training images.")

    # Deduplicate by base scene ID (strip _post_disaster_X or _X suffixes)
    unique_scenes = {}
    for p in clean_candidates:
        fname = os.path.basename(p)
        # Extract base scene key e.g. 'hurricane_florence_00000001'
        scene_key = fname.split("_post_disaster")[0] if "_post_disaster" in fname else fname.split(".")[0]
        # Strip trailing crop indices like _0, _1, _2
        scene_key = re.sub(r'_\d+$', '', scene_key)
        if scene_key not in unique_scenes:
            unique_scenes[scene_key] = p
        if len(unique_scenes) >= max_count:
            break

    final_tiles = list(unique_scenes.values())
    print(f"[Dataset Split] Selected {len(final_tiles)} tiles, at most one per name prefix.")
    return final_tiles


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("VLM BENCHMARK: AGENTIC VISION RAG")
    print("   Author: Tamanna (Architecture & Inspection Lead, Group 5)")
    print("=" * 75)

    val_images = get_validation_images(args.data_dir, max_count=args.num_samples)

    if not val_images:
        print(f"[Error] No images found under {args.data_dir}. Nothing to run.")
        sys.exit(1)

    print(f"[Setup] Running {len(val_images)} tiles across models: {args.models}")
    for idx, p in enumerate(val_images, 1):
        print(f"  [{idx}] {os.path.basename(p)}")

    # Initialize perception tool
    seg_tool = SegmentationLocatorTool(checkpoint_path=args.weights)

    benchmark_records = []
    summary_stats = {}

    for model_key in args.models:
        meta = config.VLM_MODELS.get(model_key, config.VLM_MODELS["finetuned"])
        model_name = meta["name"]
        print(f"\n{'─' * 60}")
        print(f"▶ Commencing Benchmark for: {model_name} ({meta['model_id']})")
        print(f"{'─' * 60}")

        # Initialize VLM Engine
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        vlm_engine = MultiVLMEngine(model_key=model_key)
        agent = AgenticVisionRAG(vlm_engine=vlm_engine, seg_tool=seg_tool)

        model_start_time = time.time()
        agreements = 0
        parsed = 0
        total_crops = 0
        total_buildings_detected = 0

        for img_idx, img_path in enumerate(val_images, 1):
            tile_start = time.time()
            img_name = os.path.basename(img_path)

            report = agent.run(
                image_input=img_path,
                user_query="Inspect disaster damage, detect destroyed structures, and grade EMS-98 destruction.",
                max_crops_to_inspect=5
            )

            tile_latency = time.time() - tile_start
            census = report["spatial_census"]
            summary = report["inspection_summary"]
            total_buildings_detected += census["total_buildings_detected"]

            # Record per-tile metrics; agreement is None when no VLM class answer could be read
            total_crops += summary["crops_evaluated"]
            agreements += summary["agreements"]
            parsed += summary["agreements"] + summary["conflicts_detected"]

            benchmark_records.append({
                "model_key": model_key,
                "model_name": f"{model_name} [SIMULATED MOCK]" if vlm_engine.is_mock else model_name,
                "is_mock": vlm_engine.is_mock,
                "image_name": img_name,
                "total_buildings": census["total_buildings_detected"],
                "destroyed_count": census["destroyed_count"],
                "damaged_count": census["damaged_count"],
                "intact_count": census["intact_count"],
                "crops_evaluated": summary["crops_evaluated"],
                "conflicts": summary["conflicts_detected"],
                "vlm_class_unparsed": summary["vlm_class_unparsed"],
                "agreement_rate_pct": summary["agreement_with_segmenter_pct"],
                "tile_latency_seconds": round(tile_latency, 2)
            })

            # Save annotated map for the first tile of each model
            if img_idx == 1:
                map_path = os.path.join(args.output_dir, f"{model_key}_{img_name}_annotated.png")
                generate_annotated_damage_map(img_path, report, map_path)

        total_elapsed = time.time() - model_start_time
        avg_agreement = round(agreements / parsed * 100, 2) if (parsed > 0 and not vlm_engine.is_mock) else None

        summary_stats[model_key] = {
            "model_name": f"{model_name} (SIMULATED MOCK - NOT A REAL RUN)" if vlm_engine.is_mock else model_name,
            "is_mock_fallback": vlm_engine.is_mock,
            "total_images_evaluated": len(val_images),
            "total_crops_inspected": total_crops,
            "average_agreement_pct": avg_agreement,
            "total_benchmark_time_seconds": round(total_elapsed, 2),
            "avg_latency_per_tile_seconds": round(total_elapsed / len(val_images), 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else None
        }

        # Clear GPU cache before next model
        del vlm_engine
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─── Export Results ────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "vlm_benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_records[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_records)
    print(f"\n[CSV Saved]: {csv_path}")

    json_path = os.path.join(args.output_dir, "vlm_benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"[JSON Summary Saved]: {json_path}")

    # Print Summary Table
    def show(value, fmt):
        return "n/a" if value is None else format(value, fmt)

    print("\n" + "=" * 75)
    print("VLM BENCHMARK SUMMARY TABLE (agreement with the segmenter, not accuracy)")
    print("=" * 75)
    print(f"{'Model':<30} | {'Agreement %':<12} | {'Avg Latency (s)':<16} | {'Peak VRAM'}")
    print("-" * 75)
    for k, v in summary_stats.items():
        print(f"{v['model_name']:<30} | {show(v['average_agreement_pct'], '<12.1f'):<12} | "
              f"{v['avg_latency_per_tile_seconds']:<16.2f} | {show(v['peak_vram_gb'], '.2f')} GB")
    print("=" * 75)


if __name__ == "__main__":
    main()
