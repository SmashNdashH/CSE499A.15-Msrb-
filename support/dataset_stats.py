"""
dataset_stats.py
----------------
Post-pipeline dataset analysis and visualization tool.

Generates essential statistics and charts about any JSONL dataset file
produced by the VLM fine-tuning pipeline. Designed to run on:
  - train_dataset.jsonl      (Stage 2 raw output)
  - verified_dataset.jsonl   (Stage 3 QA output)
  - final_training_dataset.jsonl (Stage 4 sanitized output)

Usage:
    python support/dataset_stats.py                           # defaults to final_training_dataset.jsonl
    python support/dataset_stats.py --input dataset/train_dataset.jsonl
    python support/dataset_stats.py --input dataset/verified_dataset.jsonl --no-show

Output:
    - Console summary table
    - Saved charts in dataset/stats/ directory
    - Optional matplotlib window display

Dependencies:
    pip install matplotlib seaborn
"""

import os
import sys
import json
import re
import argparse
from collections import Counter

# Force UTF-8 output for Windows console emoji support
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Try imports — give helpful messages if missing
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend by default
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
except ImportError:
    print("ERROR: matplotlib and seaborn are required.")
    print("       pip install matplotlib seaborn")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SUPPORT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SUPPORT_DIR)

# Default input — the final pipeline artifact
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "dataset", "final_training_dataset.jsonl")

# Output directory for charts
STATS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "stats")

# ---------------------------------------------------------------------------
# Colour palette — consistent dark-mode-friendly scheme
# ---------------------------------------------------------------------------
PALETTE = {
    "bg":        "#1A1A2E",
    "panel":     "#16213E",
    "text":      "#E0E0E0",
    "grid":      "#2A2A4A",
    "accent":    "#3B82F6",
    "bar_colors": [
        "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
        "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16",
    ],
}

# Apply global dark style
plt.rcParams.update({
    "figure.facecolor":   PALETTE["bg"],
    "axes.facecolor":     PALETTE["panel"],
    "axes.edgecolor":     PALETTE["grid"],
    "axes.labelcolor":    PALETTE["text"],
    "xtick.color":        PALETTE["text"],
    "ytick.color":        PALETTE["text"],
    "text.color":         PALETTE["text"],
    "grid.color":         PALETTE["grid"],
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "figure.titlesize":   16,
    "figure.titleweight": "bold",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, skipping blank or malformed lines."""
    data = []
    errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1
                print(f"  ⚠  Line {i}: Invalid JSON — skipped")
    if errors:
        print(f"  Total malformed lines: {errors}")
    return data


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def extract_disaster_event(image_path: str) -> str:
    """Extract disaster event name from image path.
    e.g. 'processed_images/hurricane-michael_00000035_post_disaster.jpg'
          → 'hurricane-michael'
    """
    basename = os.path.basename(image_path)
    # Split from the right to handle compound disaster names (e.g. 'socal-fire')
    parts = basename.rsplit("_", 3)
    return parts[0] if len(parts) >= 4 else basename.split("_")[0]


def extract_disaster_type(event: str) -> str:
    """Map event name to broad disaster category."""
    event_lower = event.lower()
    if "hurricane" in event_lower:
        return "Hurricane"
    elif "fire" in event_lower or "wildfire" in event_lower:
        return "Wildfire"
    elif "flood" in event_lower:
        return "Flood"
    elif "earthquake" in event_lower:
        return "Earthquake"
    elif "volcano" in event_lower:
        return "Volcano"
    elif "tsunami" in event_lower:
        return "Tsunami"
    else:
        return "Other"


def word_count(text: str) -> int:
    return len(text.split())


def has_all_headers(text: str) -> bool:
    return all(h in text for h in ["### 1.", "### 2.", "### 3."])


def has_placeholders(text: str) -> bool:
    return bool(re.search(r"\[.*?\]", text))


def has_first_person(text: str) -> bool:
    patterns = ["I ", "I'm ", "I cannot", "As an AI", "I apologize", "I'm sorry"]
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
def print_report(data: list[dict], input_path: str):
    """Print a comprehensive console summary."""
    print("\n" + "=" * 65)
    print("DATASET STATISTICS REPORT")
    print("=" * 65)
    print(f"  Source File  : {input_path}")
    print(f"  Total Samples: {len(data)}")

    if not data:
        print("  ⚠  Dataset is empty. Nothing to report.")
        return

    # --- Key presence ---
    missing_keys = sum(1 for r in data if not all(k in r for k in ("image", "instruction", "response")))
    if missing_keys:
        print(f"  ⚠  Rows missing required keys: {missing_keys}")

    # --- Word counts ---
    wc = [word_count(r.get("response", "")) for r in data]
    print(f"\n  Response Word Count:")
    print(f"    Min   : {min(wc)}")
    print(f"    Max   : {max(wc)}")
    print(f"    Mean  : {sum(wc)/len(wc):.1f}")
    print(f"    Median: {sorted(wc)[len(wc)//2]}")

    under_80 = sum(1 for w in wc if w < 80)
    over_600 = sum(1 for w in wc if w > 600)
    if under_80:
        print(f"    ⚠  Under 80 words : {under_80} samples")
    if over_600:
        print(f"    ⚠  Over 600 words : {over_600} samples")

    # --- Schema compliance ---
    no_headers = sum(1 for r in data if not has_all_headers(r.get("response", "")))
    placeholders = sum(1 for r in data if has_placeholders(r.get("response", "")))
    first_person = sum(1 for r in data if has_first_person(r.get("response", "")))

    print(f"\n  Quality Checks:")
    print(f"    Missing schema headers : {no_headers} ({no_headers/len(data)*100:.1f}%)")
    print(f"    Placeholder text [...]  : {placeholders} ({placeholders/len(data)*100:.1f}%)")
    print(f"    First-person language   : {first_person} ({first_person/len(data)*100:.1f}%)")

    # --- Disaster distribution ---
    events = [extract_disaster_event(r.get("image", "")) for r in data]
    event_dist = Counter(events)
    types = [extract_disaster_type(e) for e in events]
    type_dist = Counter(types)

    print(f"\n  Disaster Event Distribution:")
    for event, count in event_dist.most_common():
        pct = count / len(data) * 100
        flag = " ⚠ >40%" if pct > 40 else ""
        print(f"    {event.ljust(28)}: {count:4d} ({pct:5.1f}%){flag}")

    print(f"\n  Disaster Type Distribution:")
    for dtype, count in type_dist.most_common():
        pct = count / len(data) * 100
        flag = " ⚠ >40%" if pct > 40 else ""
        print(f"    {dtype.ljust(20)}: {count:4d} ({pct:5.1f}%){flag}")

    # --- Instruction diversity ---
    instr_dist = Counter(r.get("instruction", "") for r in data)
    print(f"\n  Instruction Variants: {len(instr_dist)} unique")
    for instr, count in instr_dist.most_common():
        truncated = instr[:70] + "..." if len(instr) > 70 else instr
        print(f"    [{count:4d}] {truncated}")

    # --- Duplicate image paths ---
    img_counts = Counter(r.get("image", "") for r in data)
    duplicates = {k: v for k, v in img_counts.items() if v > 1}
    if duplicates:
        print(f"\n  ⚠  Duplicate Image Paths: {len(duplicates)}")
        for img, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
            print(f"    [{count}x] {img}")
    else:
        print(f"\n  No duplicate image paths")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------
def plot_disaster_distribution(data: list[dict], output_dir: str):
    """Bar chart of disaster event distribution."""
    events = [extract_disaster_event(r.get("image", "")) for r in data]
    event_dist = Counter(events)

    labels, values = zip(*event_dist.most_common())
    colors = [PALETTE["bar_colors"][i % len(PALETTE["bar_colors"])] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="none", height=0.6)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Samples")
    ax.set_title("Disaster Event Distribution")
    ax.grid(axis="x", alpha=0.3)

    # Add count labels on bars
    for bar, val in zip(bars, values):
        pct = val / len(data) * 100
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f" {val} ({pct:.1f}%)", va="center", fontsize=10, color=PALETTE["text"])

    # 40% threshold line
    threshold = len(data) * 0.4
    ax.axvline(x=threshold, color="#EF4444", linestyle="--", alpha=0.7, label=f"40% threshold ({int(threshold)})")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "disaster_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_disaster_type_pie(data: list[dict], output_dir: str):
    """Pie chart of broad disaster type categories."""
    events = [extract_disaster_event(r.get("image", "")) for r in data]
    types = [extract_disaster_type(e) for e in events]
    type_dist = Counter(types)

    labels, values = zip(*type_dist.most_common())
    colors = PALETTE["bar_colors"][:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 8))
    pie_chart = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": PALETTE["text"], "fontsize": 12},
        pctdistance=0.75,
    )
    for t in pie_chart[2]:
        t.set_fontsize(11)
        t.set_fontweight("bold")

    ax.set_title("Disaster Type Breakdown")
    plt.tight_layout()
    path = os.path.join(output_dir, "disaster_type_pie.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_word_count_histogram(data: list[dict], output_dir: str):
    """Histogram of response word counts."""
    wc = [word_count(r.get("response", "")) for r in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(wc, bins=30, color=PALETTE["accent"], edgecolor=PALETTE["panel"], alpha=0.9)

    ax.set_xlabel("Word Count")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Response Word Count Distribution")
    ax.grid(axis="y", alpha=0.3)

    # Mark healthy range
    ax.axvline(x=80, color="#EF4444", linestyle="--", alpha=0.7, label="Min (80 words)")
    ax.axvline(x=600, color="#EF4444", linestyle="--", alpha=0.7, label="Max (600 words)")
    ax.axvline(x=sum(wc)/len(wc), color="#10B981", linestyle="-", alpha=0.8, label=f"Mean ({sum(wc)/len(wc):.0f})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "word_count_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_quality_checks(data: list[dict], output_dir: str):
    """Bar chart showing quality check pass/fail rates."""
    checks = {
        "Schema Headers\n(### 1/2/3)": sum(1 for r in data if has_all_headers(r.get("response", ""))),
        "No Placeholders\n([...] free)": sum(1 for r in data if not has_placeholders(r.get("response", ""))),
        "No First-Person\n(I/I'm free)": sum(1 for r in data if not has_first_person(r.get("response", ""))),
        "Word Count\n(80–600)": sum(1 for r in data if 80 <= word_count(r.get("response", "")) <= 600),
        "All Keys\nPresent": sum(1 for r in data if all(k in r for k in ("image", "instruction", "response"))),
    }

    labels = list(checks.keys())
    pass_counts = list(checks.values())
    fail_counts = [len(data) - p for p in pass_counts]
    pass_pcts = [p / len(data) * 100 for p in pass_counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(labels))
    width = 0.5

    bars_pass = ax.bar(x, pass_counts, width, color="#10B981", label="Pass", edgecolor="none")
    bars_fail = ax.bar(x, fail_counts, width, bottom=pass_counts, color="#EF4444", label="Fail", edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Quality Check Results")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars_pass, pass_pcts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{pct:.0f}%", ha="center", va="center", fontsize=11, fontweight="bold",
                color="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "quality_checks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_instruction_diversity(data: list[dict], output_dir: str):
    """Bar chart of instruction variant distribution."""
    instr_dist = Counter(r.get("instruction", "") for r in data)

    if len(instr_dist) <= 1:
        print("  ℹ  Only 1 instruction variant — skipping diversity chart.")
        return

    # Truncate labels for readability
    labels_full = list(instr_dist.keys())
    labels = [l[:50] + "..." if len(l) > 50 else l for l in labels_full]
    values = list(instr_dist.values())
    colors = [PALETTE["bar_colors"][i % len(PALETTE["bar_colors"])] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.6)))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="none", height=0.6)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Samples")
    ax.set_title("Instruction Variant Distribution")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f" {val}", va="center", fontsize=10, color=PALETTE["text"])

    plt.tight_layout()
    path = os.path.join(output_dir, "instruction_diversity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics and visualizations for a VLM fine-tuning JSONL dataset."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to the JSONL dataset file. Default: {DEFAULT_INPUT}"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display charts interactively (requires a GUI backend)."
    )
    args = parser.parse_args()

    input_path = args.input

    # Resolve relative paths
    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)

    if not os.path.exists(input_path):
        # Try fallback files
        fallbacks = [
            os.path.join(PROJECT_ROOT, "dataset", "verified_dataset.jsonl"),
            os.path.join(PROJECT_ROOT, "dataset", "train_dataset.jsonl"),
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                print(f"  ℹ  {input_path} not found. Falling back to {fb}")
                input_path = fb
                break
        else:
            print(f"  ERROR: No dataset file found at {input_path}")
            print(f"         Also checked: {', '.join(fallbacks)}")
            sys.exit(1)

    # Load data
    print(f"\nLoading: {input_path}")
    data = load_jsonl(input_path)

    if not data:
        print("  ⚠  Dataset is empty. Exiting.")
        sys.exit(1)

    # Console report
    print_report(data, input_path)

    # Generate charts
    os.makedirs(STATS_OUTPUT_DIR, exist_ok=True)
    print(f"\nGenerating charts → {STATS_OUTPUT_DIR}/")
    plot_disaster_distribution(data, STATS_OUTPUT_DIR)
    plot_disaster_type_pie(data, STATS_OUTPUT_DIR)
    plot_word_count_histogram(data, STATS_OUTPUT_DIR)
    plot_quality_checks(data, STATS_OUTPUT_DIR)
    plot_instruction_diversity(data, STATS_OUTPUT_DIR)

    print(f"\nAll charts saved to: {STATS_OUTPUT_DIR}/")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()
