"""
Per-model, per-question-type evaluation for the DisasterVQA benchmark.

Computes for each model:
  - Yes/No accuracy
  - Open-Ended accuracy
  - Multiple-Choice micro precision, recall, and F1

Model predictions are read from the `benchmarking_answers` field of each entry.

Usage:
    python evaluate_by_question_type.py \
        --input  dataset/disasterVQA_allmodel_judge_outputs.json \
        --output results/metrics_by_question_type.xlsx
"""

import argparse
import json
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Per-model, per-question-type DisasterVQA evaluation.")
    p.add_argument("-i", "--input", required=True, help="Judge outputs JSON.")
    p.add_argument("-o", "--output", default="metrics_by_question_type.xlsx", help="Output Excel file.")
    return p.parse_args()


def discover_models(data):
    return list((data[0].get("benchmarking_answers") or {}).keys())


def eval_yes_no(pred, gt):
    if isinstance(pred, str):
        return int(pred.strip().lower() == str(gt).strip().lower())
    return 0


def eval_open(pred):
    if isinstance(pred, dict):
        return int(pred.get("decision", "").strip().lower() == "right")
    return 0


def eval_mcq(pred, gt):
    p, g = set(pred or []), set(gt or [])
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    return tp, fp, fn


def main():
    args = parse_args()

    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list) or not data:
        print("Input must be a non-empty JSON list.", file=sys.stderr)
        sys.exit(1)

    models = discover_models(data)
    counts = {
        m: {"yes_total": 0, "yes_correct": 0,
            "open_total": 0, "open_correct": 0,
            "mcq_tp": 0, "mcq_fp": 0, "mcq_fn": 0}
        for m in models
    }

    for rec in data:
        qtype = rec.get("question_type", "").strip().lower()
        gt = rec.get("groundtruth_answer")
        ba = rec.get("benchmarking_answers") or {}
        for m in models:
            pred = ba.get(m)
            c = counts[m]
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

    rows = []
    for m, c in counts.items():
        yes_acc = c["yes_correct"] / c["yes_total"] if c["yes_total"] else None
        open_acc = c["open_correct"] / c["open_total"] if c["open_total"] else None
        tp, fp, fn = c["mcq_tp"], c["mcq_fp"], c["mcq_fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "model": m,
            "yesno_accuracy": yes_acc,
            "open_accuracy": open_acc,
            "mcq_precision": prec,
            "mcq_recall": rec,
            "mcq_f1": f1,
        })

    pd.DataFrame(rows).to_excel(args.output, index=False)
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
