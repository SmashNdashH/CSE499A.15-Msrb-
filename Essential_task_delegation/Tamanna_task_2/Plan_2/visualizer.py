"""
visualizer.py
-------------
Publication-Grade Visual Report Generator.
Renders the master annotated satellite map side-by-side with zoomed-in inspection crop cards.

Author: Tamanna (Member B, Group 5)
Project: CSE499B DisasterM3 Capstone
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, Union
import config


def generate_annotated_damage_map(
    image_input: Union[str, np.ndarray],
    report: Dict[str, Any],
    output_path: str
) -> str:
    """
    Overlays colored bounding boxes, labels, and summary statistics on the satellite image.
    """
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_input

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(img_rgb)

    for item in report["detailed_inspections"]:
        x, y, w, h = item["bbox"]
        cls = item["initial_segmentation_class"]
        col = np.array(config.CLASS_COLORS_RGB.get(cls, (255, 255, 255))) / 255.0

        # Draw bounding rectangle
        rect = patches.Rectangle((x, y), w, h, linewidth=2.5, edgecolor=col, facecolor='none')
        ax.add_patch(rect)

        # Draw ID badge
        badge_text = f"#{item['building_id']} {cls} ({item['ems98_grade']})"
        ax.text(
            x, max(0, y - 6),
            badge_text,
            fontsize=8,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=col, alpha=0.9, edgecolor="none")
        )

    counts = report["spatial_census"]
    title = (
        f"Agentic Vision RAG Damage Assessment\n"
        f"Total: {counts['total_buildings_detected']} | "
        f"Destroyed: {counts['destroyed_count']} | "
        f"Damaged: {counts['damaged_count']} | "
        f"Intact: {counts['intact_count']}"
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def generate_inspection_dashboard(
    image_input: Union[str, np.ndarray],
    report: Dict[str, Any],
    output_path: str
) -> str:
    """
    Creates a composite multi-panel dashboard:
      - Panel A (Left): Full Satellite Tile with Bounding Boxes.
      - Panel B (Right): Zoomed-in 256x256 Building Cards with EMS-98 Inspection Details.
    """
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_input

    inspections = report["detailed_inspections"]
    num_crops = len(inspections)

    if num_crops == 0:
        return generate_annotated_damage_map(image_input, report, output_path)

    fig = plt.figure(figsize=(18, 10), dpi=150)
    gs = fig.add_gridspec(max(num_crops, 2), 3, width_ratios=[1.8, 1, 1], wspace=0.3, hspace=0.35)

    # Left: Master Map
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(img_rgb)
    ax_main.set_title("Master Satellite Tile (Agent Spatial Index)", fontsize=13, fontweight="bold")
    ax_main.axis("off")

    for item in inspections:
        x, y, w, h = item["bbox"]
        cls = item["initial_segmentation_class"]
        col = np.array(config.CLASS_COLORS_RGB.get(cls, (255, 255, 255))) / 255.0

        rect = patches.Rectangle((x, y), w, h, linewidth=2.5, edgecolor=col, facecolor='none')
        ax_main.add_patch(rect)

        badge_text = f"#{item['building_id']}"
        ax_main.text(
            x, max(0, y - 6),
            badge_text,
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=col, alpha=0.9, edgecolor="none")
        )

    # Right: Crop Cards
    for idx, item in enumerate(inspections[:6]):  # Display up to 6 crops
        row = idx // 2
        col_idx = 1 + (idx % 2)
        ax_crop = fig.add_subplot(gs[row, col_idx])

        ax_crop.imshow(item["crop_rgb"])
        cls = item["initial_segmentation_class"]
        col = np.array(config.CLASS_COLORS_RGB.get(cls, (255, 255, 255))) / 255.0

        # Title with building metadata
        ax_crop.set_title(
            f"Building #{item['building_id']} [{cls}]\n{item['ems98_grade']}",
            fontsize=10,
            fontweight="bold",
            color="black"
        )
        ax_crop.axis("off")

        # Border colored by damage
        for spine in ax_crop.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(3)

    plt.suptitle(
        f"DisasterM3 Agentic Vision RAG Inspection Dashboard — Model: {report.get('vlm_model_used', 'VLM')}",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Dashboard Saved]: {output_path}")
    return output_path
