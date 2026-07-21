"""
evaluate_damage_alignment.py
----------------------------
Rule-based evaluator for checking whether generated rescue-guidance responses
stay aligned with xBD damage metadata.

Usage:
    python support/evaluate_damage_alignment.py
    python support/evaluate_damage_alignment.py --input dataset/train_dataset.jsonl

Output:
    - dataset/evaluation/damage_alignment_report.csv
    - Console summary with alignment, consistency, actionability, and hallucination rate
"""

import argparse
import csv
import json
import os
import re
from collections import Counter


SUPPORT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SUPPORT_DIR)

DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "dataset", "final_training_dataset.jsonl")
LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed_labels")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "evaluation")
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "damage_alignment_report.csv")

SEVERITY_ORDER = ["destroyed", "major-damage", "minor-damage", "no-damage", "un-classified"]
DAMAGE_COUNTS = ["destroyed", "major-damage", "minor-damage", "no-damage", "un-classified"]

SEVERE_DAMAGE_TERMS = [
    "destroyed",
    "severe",
    "collapse",
    "collapsed",
    "pancake",
    "rubble",
    "debris",
    "structural failure",
    "major damage",
    "heavily damaged",
]

MODERATE_DAMAGE_TERMS = [
    "minor damage",
    "damaged",
    "partial",
    "crack",
    "roof damage",
    "facade",
]

NO_DAMAGE_TERMS = [
    "intact",
    "no structural damage",
    "no visible damage",
    "no damage",
    "undamaged",
]

RESCUE_TERMS = [
    "priority",
    "zone",
    "search and rescue",
    "rescue",
    "access",
    "ingress",
    "egress",
    "blocked",
    "road",
    "hazard",
    "triage",
    "evacuation",
    "corridor",
    "logistics",
]

DISASTER_TERMS = {
    "volcano": ["volcano", "volcanic", "ash", "lahar", "mudflow", "flow"],
    "flood": ["flood", "flooding", "water", "inundation", "submerged", "boat"],
    "hurricane": ["hurricane", "wind", "storm", "flood", "water", "inundation"],
    "wildfire": ["wildfire", "fire", "burn", "smoke", "ember", "thermal"],
    "fire": ["fire", "burn", "smoke", "ember", "thermal"],
    "earthquake": ["earthquake", "collapse", "rubble", "pancake", "structural failure"],
    "tsunami": ["tsunami", "wave", "debris", "inundation", "coastal", "water"],
}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSONL line {line_number}")
    return rows


def get_max_severity(features):
    present = {
        feature.get("properties", {}).get("subtype", "un-classified")
        for feature in features
    }
    for severity in SEVERITY_ORDER:
        if severity in present:
            return severity
    return "un-classified"


def parse_xbd_label(label_path):
    with open(label_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    metadata = data.get("metadata", {})
    disaster_type = metadata.get("disaster_type", "unknown")
    features = data.get("features", {}).get("xy", [])
    counts = {key: 0 for key in DAMAGE_COUNTS}

    for feature in features:
        subtype = feature.get("properties", {}).get("subtype", "un-classified")
        if subtype in counts:
            counts[subtype] += 1

    return {
        "disaster_type": disaster_type,
        "total_buildings": len(features),
        "max_severity": get_max_severity(features),
        **counts,
    }


def label_path_for_image(image_path):
    image_name = os.path.basename(image_path)
    label_name = os.path.splitext(image_name)[0] + ".json"
    return os.path.join(LABEL_DIR, label_name)


def has_any(text, terms):
    return any(term in text for term in terms)


def disaster_consistency(text, disaster_type):
    disaster_type = disaster_type.lower()
    for key, terms in DISASTER_TERMS.items():
        if key in disaster_type:
            return int(has_any(text, terms))
    return int(disaster_type != "unknown" and disaster_type in text)


def severity_alignment(text, max_severity):
    if max_severity in {"destroyed", "major-damage"}:
        return int(has_any(text, SEVERE_DAMAGE_TERMS))
    if max_severity == "minor-damage":
        return int(has_any(text, MODERATE_DAMAGE_TERMS + SEVERE_DAMAGE_TERMS))
    if max_severity == "no-damage":
        return int(has_any(text, NO_DAMAGE_TERMS) and not has_any(text, SEVERE_DAMAGE_TERMS))
    return 0


def hallucination_flag(text, max_severity):
    if max_severity == "no-damage" and has_any(text, SEVERE_DAMAGE_TERMS):
        return 1
    return 0


def metadata_coverage(text, label_data):
    score = 0
    if label_data["destroyed"] > 0 and has_any(text, ["destroyed", "collapse", "collapsed", "rubble"]):
        score += 1
    if label_data["major-damage"] > 0 and has_any(text, ["major damage", "severe", "structural failure"]):
        score += 1
    if label_data["minor-damage"] > 0 and has_any(text, ["minor damage", "damaged", "partial"]):
        score += 1
    if label_data["no-damage"] > 0 and has_any(text, NO_DAMAGE_TERMS):
        score += 1
    return score


def evaluate_row(row):
    image_path = row.get("image", "")
    label_path = label_path_for_image(image_path)

    if not os.path.exists(label_path):
        return None

    label_data = parse_xbd_label(label_path)
    response = row.get("response", "")
    normalized_response = re.sub(r"\s+", " ", response.lower())

    sev_alignment = severity_alignment(normalized_response, label_data["max_severity"])
    dis_consistency = disaster_consistency(normalized_response, label_data["disaster_type"])
    rescue_actionability = int(has_any(normalized_response, RESCUE_TERMS))
    hallucinated = hallucination_flag(normalized_response, label_data["max_severity"])
    coverage = metadata_coverage(normalized_response, label_data)

    overall_score = round(
        (
            sev_alignment
            + dis_consistency
            + rescue_actionability
            + min(coverage, 3) / 3
            + (1 - hallucinated)
        )
        / 5,
        3,
    )

    return {
        "image": image_path,
        "disaster_type": label_data["disaster_type"],
        "max_severity": label_data["max_severity"],
        "total_buildings": label_data["total_buildings"],
        "destroyed": label_data["destroyed"],
        "major_damage": label_data["major-damage"],
        "minor_damage": label_data["minor-damage"],
        "no_damage": label_data["no-damage"],
        "severity_alignment": sev_alignment,
        "disaster_consistency": dis_consistency,
        "rescue_actionability": rescue_actionability,
        "hallucination_flag": hallucinated,
        "metadata_coverage": coverage,
        "overall_score": overall_score,
    }


def write_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "image",
        "disaster_type",
        "max_severity",
        "total_buildings",
        "destroyed",
        "major_damage",
        "minor_damage",
        "no_damage",
        "severity_alignment",
        "disaster_consistency",
        "rescue_actionability",
        "hallucination_flag",
        "metadata_coverage",
        "overall_score",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(count, total):
    if total == 0:
        return "0.0%"
    return f"{(count / total) * 100:.1f}%"


def print_summary(rows, skipped, output_path):
    total = len(rows)
    severity_counts = Counter(row["max_severity"] for row in rows)
    avg_score = sum(row["overall_score"] for row in rows) / total if total else 0

    print("\nDamage Alignment Evaluation")
    print("===========================")
    print(f"Evaluated samples      : {total}")
    print(f"Skipped missing labels : {skipped}")
    print(f"Severity alignment    : {pct(sum(r['severity_alignment'] for r in rows), total)}")
    print(f"Disaster consistency  : {pct(sum(r['disaster_consistency'] for r in rows), total)}")
    print(f"Rescue actionability  : {pct(sum(r['rescue_actionability'] for r in rows), total)}")
    print(f"Hallucination rate    : {pct(sum(r['hallucination_flag'] for r in rows), total)}")
    print(f"Average overall score : {avg_score:.3f}")
    print("\nSeverity distribution:")
    for severity in SEVERITY_ORDER:
        if severity_counts[severity]:
            print(f"  {severity:15s}: {severity_counts[severity]}")
    print(f"\nCSV report saved to   : {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to JSONL dataset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to CSV report")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    rows = load_jsonl(args.input)
    report_rows = []
    skipped = 0

    for row in rows:
        evaluated = evaluate_row(row)
        if evaluated is None:
            skipped += 1
            continue
        report_rows.append(evaluated)

    write_csv(report_rows, args.output)
    print_summary(report_rows, skipped, args.output)


if __name__ == "__main__":
    main()
