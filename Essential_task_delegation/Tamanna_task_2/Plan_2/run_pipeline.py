"""
run_pipeline.py
---------------
Single-command CLI runner for Tamanna's Agentic Vision RAG Pipeline.
Can process single images, test synthetic images, or process folders.

Usage:
  # Run using team's fine-tuned model
  python run_pipeline.py --model finetuned --image path/to/satellite_post.png

  # Run using Google PaliGemma 2 benchmark
  python run_pipeline.py --model paligemma --image path/to/satellite_post.png

  # Run self-test on generated sample (works offline without any external files)
  python run_pipeline.py --self_test

Author: Tamanna (Member B, Group 5)
Project: CSE499B DisasterM3 Capstone
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2

import config
from tools import SegmentationLocatorTool, HighResVisualRetrieverTool, EMS98KnowledgeRAGTool
from vlm_engine import MultiVLMEngine
from agent import AgenticVisionRAG
from visualizer import generate_inspection_dashboard


def create_synthetic_disaster_tile() -> Tuple_Type:
    """Generates a synthetic 1024x1024 satellite disaster scene for self-testing."""
    np.random.seed(42)
    img = np.ones((1024, 1024, 3), dtype=np.uint8) * 80  # Dark ground background

    # Add terrain texture
    noise = np.random.randint(-15, 15, (1024, 1024, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Synthetic segmentation mask
    mask = np.zeros((1024, 1024), dtype=np.uint8)

    # Draw synthetic buildings
    # 1: Intact (Greenish roofs)
    cv2.rectangle(img, (100, 150), (220, 260), (140, 160, 140), -1)
    mask[150:260, 100:220] = 1

    cv2.rectangle(img, (300, 120), (410, 230), (150, 170, 150), -1)
    mask[120:230, 300:410] = 1

    # 2: Damaged (Cracked roofs)
    cv2.rectangle(img, (550, 400), (680, 520), (160, 140, 100), -1)
    cv2.line(img, (560, 410), (670, 510), (40, 40, 40), 3)
    mask[400:520, 550:680] = 2

    # 3: Destroyed (Rubble heap)
    cv2.rectangle(img, (750, 650), (900, 800), (110, 110, 110), -1)
    # Scatter rubble debris dots
    for _ in range(80):
        rx = np.random.randint(730, 920)
        ry = np.random.randint(630, 820)
        cv2.circle(img, (rx, ry), np.random.randint(2, 6), (180, 180, 180), -1)
    mask[650:800, 750:900] = 3

    return img, mask


Tuple_Type = tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Tamanna's Agentic Vision RAG Runner")
    parser.add_argument(
        "--model",
        type=str,
        choices=["finetuned", "base", "paligemma", "qwen_edge"],
        default="finetuned",
        help="Which VLM model to load (finetuned, base, paligemma, qwen_edge)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the 1024x1024 satellite post-disaster image"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional path to ground truth or precomputed segmentation mask"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.DEFAULT_OUTPUT_DIR,
        help="Directory to save inspection reports and visual dashboards"
    )
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run on an in-memory generated disaster image (ideal for smoke testing)"
    )
    parser.add_argument(
        "--max_crops",
        type=int,
        default=5,
        help="Maximum building crops to zoom in on"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 DISASTERM3 AGENTIC VISION RAG PIPELINE (PLAN 2)")
    print("   Author: Tamanna | Member B (Architecture & Inspection Lead)")
    print("=" * 70)

    # Prepare Image and Mask
    if args.self_test or args.image is None:
        print("\n[Notice] No image supplied or --self_test enabled. Using synthetic test tile...")
        image_input, mask_override = create_synthetic_disaster_tile()
        image_id = "synthetic_disaster_sample"
    else:
        if not os.path.exists(args.image):
            print(f"[Error] Image file not found: {args.image}")
            sys.exit(1)
        image_input = args.image
        image_id = os.path.splitext(os.path.basename(args.image))[0]
        mask_override = None
        if args.mask and os.path.exists(args.mask):
            mask_override = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    # Initialize Engine and Agent
    vlm_engine = MultiVLMEngine(model_key=args.model)
    seg_tool = SegmentationLocatorTool()
    agent = AgenticVisionRAG(vlm_engine=vlm_engine, seg_tool=seg_tool)

    # Execute Agent
    query = "Analyze structural damage, extract building counts, assign EMS-98 grades, and declare rescue triage priorities."
    report = agent.run(
        image_input=image_input,
        user_query=query,
        max_crops_to_inspect=args.max_crops,
        mask_override=mask_override
    )

    # Save JSON Report
    json_path = os.path.join(args.output_dir, f"{image_id}_{args.model}_report.json")
    clean_report = {k: v for k, v in report.items() if k != "segmentation_mask"}
    # Strip numpy arrays from JSON export
    for item in clean_report.get("detailed_inspections", []):
        item.pop("crop_rgb", None)

    with open(json_path, "w") as f:
        json.dump(clean_report, f, indent=2)
    print(f"\n[Report Exported]: {json_path}")

    # Generate Dashboard
    dashboard_path = os.path.join(args.output_dir, f"{image_id}_{args.model}_dashboard.png")
    generate_inspection_dashboard(image_input, report, dashboard_path)

    print("\n" + "=" * 70)
    print("✓ Pipeline successfully executed!")
    print(f"  Summary: {report['spatial_census']}")
    print(f"  Dashboard: {dashboard_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
