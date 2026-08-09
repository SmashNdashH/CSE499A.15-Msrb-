#!/usr/bin/env python3
"""
Per-model, per-humanitarian-category evaluation for the DisasterVQA benchmark.

Reads the humanitarian category from each entry's `llm_classification` field
and computes for each model and each category:
  - Yes/No accuracy
  - Open-Ended accuracy
  - Multiple-Choice micro precision, recall, and F1

Writes results to an Excel workbook (one summary sheet + one sheet per model)
and a wide CSV with all models side by side.

Usage:
    python evaluate_by_humanitarian_category.py \
        --input  dataset/disasterVQA_allmodel_judge_outputs.json \
        --excel  results/metrics_by_humanitarian_category.xlsx \
        --csv    results/metrics_by_humanitarian_category.csv
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-model, per-humanitarian-category DisasterVQA evaluation."
    )
    p.add_argument("-i", "--input", required=True, help="Judged outputs JSON.")
    p.add_argument("--excel", default="metrics_by_humanitarian_category.xlsx", help="Output Excel file.")
    p.add_argument("--csv", default="metrics_by_humanitarian_category.csv", help="Output wide CSV file.")
    return p.parse_args()


def load_data(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed to load JSON: %s", e)
        sys.exit(1)


def norm(s):
    return (s or "").replace("&amp;", "&").strip()


def category_from_record(rec):
    lc = rec.get("llm_classification") or {}
    name = norm(lc.get("category_name") or lc.get("category") or "")
    if not name or name.lower() == "unknown":
        return "Other Useful Information"
    return name


def discover_models(data):
    ba = data[0].get("benchmarking_answers", {})
    if isinstance(ba, dict):
        return list(ba.keys())
    return []


def eval_yes_no(pred, gt):
    return int(isinstance(pred, str) and pred.strip().lower() == str(gt).strip().lower())


def eval_open(pred):
    return int(isinstance(pred, dict) and norm(pred.get("decision", "")).lower() == "right")


def eval_mcq(pred, gt):
    pset, gset = set(pred or []), set(gt or [])
    tp = len(pset & gset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    return tp, fp, fn


def safe_div(a, b):
    return None if b == 0 else a / b


def fmt2(x):
    return "" if x is None else f"{x:.2f}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    data = load_data(args.input)
    if not isinstance(data, list) or not data:
        logging.error("Input JSON must be a non-empty list.")
        sys.exit(1)

    models = discover_models(data)
    if not models:
        logging.error("No models found under 'benchmarking_answers' in the first record.")
        sys.exit(1)
    logging.info("Detected %d models: %s", len(models), models)

    cat_total = Counter()
    metrics = {
        m: defaultdict(lambda: {
            "yes_total": 0, "yes_correct": 0,
            "open_total": 0, "open_correct": 0,
            "mcq_tp": 0, "mcq_fp": 0, "mcq_fn": 0,
        })
        for m in models
    }

    for rec in data:
        qtype = norm(rec.get("question_type", "")).lower()
        gt = rec.get("groundtruth_answer")
        cat = category_from_record(rec)
        cat_total[cat] += 1

        ba = rec.get("benchmarking_answers") or {}
        if not isinstance(ba, dict):
            ba = {}

        for m in models:
            pred = ba.get(m)
            c = metrics[m][cat]

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

    # Wide format: one row per category, columns for all models × metrics
    wide_rows = []
    for cat, total in sorted(cat_total.items()):
        row = {"category": cat, "total_pairs": total}
        for m in models:
            d = metrics[m][cat]
            yes_acc = safe_div(d["yes_correct"], d["yes_total"]) if d["yes_total"] else None
            open_acc = safe_div(d["open_correct"], d["open_total"]) if d["open_total"] else None
            tp, fp, fn = d["mcq_tp"], d["mcq_fp"], d["mcq_fn"]
            prec = safe_div(tp, tp + fp) if (tp + fp) else None
            rec_ = safe_div(tp, tp + fn) if (tp + fn) else None
            f1 = (2 * prec * rec_ / (prec + rec_)
                  if (prec is not None and rec_ is not None and (prec + rec_))
                  else None)
            row[f"{m}_binary_accuracy"] = fmt2(yes_acc)
            row[f"{m}_open_accuracy"] = fmt2(open_acc)
            row[f"{m}_mcq_precision"] = fmt2(prec)
            row[f"{m}_mcq_recall"] = fmt2(rec_)
            row[f"{m}_mcq_f1"] = fmt2(f1)
        wide_rows.append(row)

    ordered_cols = ["category", "total_pairs"]
    for m in models:
        ordered_cols += [
            f"{m}_binary_accuracy", f"{m}_open_accuracy",
            f"{m}_mcq_precision", f"{m}_mcq_recall", f"{m}_mcq_f1",
        ]
    wide_df = pd.DataFrame(wide_rows).reindex(columns=ordered_cols)
    wide_df.to_csv(args.csv, index=False)

    with pd.ExcelWriter(args.excel, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="All_Models", index=False)

        for m in models:
            rows = []
            for cat, total in sorted(cat_total.items()):
                d = metrics[m][cat]
                yes_acc = safe_div(d["yes_correct"], d["yes_total"]) if d["yes_total"] else None
                open_acc = safe_div(d["open_correct"], d["open_total"]) if d["open_total"] else None
                tp, fp, fn = d["mcq_tp"], d["mcq_fp"], d["mcq_fn"]
                prec = safe_div(tp, tp + fp) if (tp + fp) else None
                rec_ = safe_div(tp, tp + fn) if (tp + fn) else None
                f1 = (2 * prec * rec_ / (prec + rec_)
                      if (prec is not None and rec_ is not None and (prec + rec_))
                      else None)
                rows.append({
                    "category": cat,
                    "total_pairs": total,
                    "binary_accuracy": round(yes_acc, 2) if yes_acc is not None else None,
                    "open_accuracy": round(open_acc, 2) if open_acc is not None else None,
                    "mcq_precision": round(prec, 2) if prec is not None else None,
                    "mcq_recall": round(rec_, 2) if rec_ is not None else None,
                    "mcq_f1": round(f1, 2) if f1 is not None else None,
                })
            pd.DataFrame(rows).to_excel(writer, sheet_name=f"Categories_{m}"[:31], index=False)

    logging.info("Saved Excel to %s", args.excel)
    logging.info("Saved wide CSV to %s", args.csv)


if __name__ == "__main__":
    main()

