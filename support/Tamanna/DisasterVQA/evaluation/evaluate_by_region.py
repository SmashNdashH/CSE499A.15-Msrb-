"""
Per-model, per-geographic-region evaluation for the DisasterVQA benchmark.

Reads the region directly from the `region` field of each entry and computes
for each model and each region:
  - Yes/No accuracy
  - Open-Ended accuracy
  - Multiple-Choice micro precision, recall, and F1

Model predictions are read from the `benchmarking_answers` field of each entry.

Usage:
    python evaluate_by_region.py \
        --input  dataset/disasterVQA_allmodel_judge_outputs.json \
        --output results/metrics_by_region.xlsx
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Per-model, per-region DisasterVQA evaluation.")
    p.add_argument("-i", "--input", required=True, help="Judge outputs JSON.")
    p.add_argument("-o", "--output", default="metrics_by_region.xlsx", help="Output Excel file.")
    return p.parse_args()


def load_data(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed to load JSON: %s", e)
        sys.exit(1)


def discover_models(data):
    return list((data[0].get("benchmarking_answers") or {}).keys())


def eval_yes_no(pred, gt):
    return int(isinstance(pred, str) and pred.strip().lower() == str(gt).strip().lower())


def eval_open(pred):
    return int(isinstance(pred, dict) and pred.get("decision", "").strip().lower() == "right")


def eval_mcq(pred, gt):
    p, g = set(pred or []), set(gt or [])
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    return tp, fp, fn


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    data = load_data(args.input)
    if not isinstance(data, list) or not data:
        logging.error("Input JSON must be a non-empty list.")
        sys.exit(1)

    models = discover_models(data)
    logging.info("Detected %d models: %s", len(models), models)

    region_total = Counter()
    region_model_metrics = {
        m: defaultdict(lambda: {
            "yes_total": 0, "yes_correct": 0,
            "open_total": 0, "open_correct": 0,
            "mcq_tp": 0, "mcq_fp": 0, "mcq_fn": 0,
        })
        for m in models
    }

    for rec in data:
        qtype = rec.get("question_type", "").strip().lower()
        gt = rec.get("groundtruth_answer")
        region = rec.get("region") or "N/A"
        region_total[region] += 1

        ba = rec.get("benchmarking_answers") or {}
        for m in models:
            pred = ba.get(m)
            c = region_model_metrics[m][region]

            if qtype == "yes/no":
                c["yes_total"] += 1
                c["yes_correct"] += eval_yes_no(pred, gt)
            elif qtype.startswith("open"):
                c["open_total"] += 1
                c["open_correct"] += eval_open(pred)
            elif qtype.startswith("multiple"):
                tp, fp, fn = eval_mcq(pred, gt)
                c["mcq_tp"] += tp
                c["mcq_fp"] += fp
                c["mcq_fn"] += fn

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        for m in models:
            rows = []
            for region, total in sorted(region_total.items()):
                d = region_model_metrics[m][region]
                row = {"region": region, "total_pairs": total}

                row["yesno_accuracy"] = (
                    d["yes_correct"] / d["yes_total"] if d["yes_total"] else None
                )
                row["open_accuracy"] = (
                    d["open_correct"] / d["open_total"] if d["open_total"] else None
                )

                tp, fp, fn = d["mcq_tp"], d["mcq_fp"], d["mcq_fn"]
                prec = tp / (tp + fp) if (tp + fp) else None
                rec = tp / (tp + fn) if (tp + fn) else None
                f1 = (2 * prec * rec / (prec + rec)
                      if (prec is not None and rec is not None and (prec + rec))
                      else None)

                row["mcq_precision"] = prec
                row["mcq_recall"] = rec
                row["mcq_f1"] = f1
                rows.append(row)

            pd.DataFrame(rows).to_excel(writer, sheet_name=f"Regions_{m}", index=False)

    logging.info("Saved region metrics to %s", args.output)


if __name__ == "__main__":
    main()
