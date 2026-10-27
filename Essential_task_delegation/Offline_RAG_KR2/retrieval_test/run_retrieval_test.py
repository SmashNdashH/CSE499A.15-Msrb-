"""
run_retrieval_test.py
---------------------
Retrieval test of the guideline search. Fixed before this script was first run
(README, "Retrieval test of the guideline search"): the questions and expected
IDs in questions.csv, and the pass rule, at least 24 of the 30 questions with an
expected section in the top 3.

For each question it takes the top 3 of:
  - the hybrid search, search_guidelines(q, k=3), which is the graded result;
  - the meaning ranking alone and the BM25 keyword ranking alone: the two
    rankings search_guidelines fuses, cut to their first 3.
It runs on this machine's default backend, then again with GUIDELINE_BACKEND=numpy
(the Kaggle path), and reports where the top 3 differ.

Writes results.csv (question, top3_ids, hit) for the hybrid search on the default
backend, and results_summary.json with the counts, the misses and every ranking.
A later run (for example after a fix) must use another --name, so the first
run's files are kept.

Usage (from Offline_RAG_KR2/):
    python retrieval_test/run_retrieval_test.py
"""

import argparse
import csv
import inspect
import json
import os
import sys
from collections import Counter
from datetime import date

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the numpy backend's model is local

import guideline_index as gi  # noqa: E402

K = 3
PASS_HITS = 24
POOL = inspect.signature(gi.search_guidelines).parameters["pool"].default
RANKINGS = ("hybrid", "semantic", "bm25")


def load_questions() -> list:
    with open(os.path.join(here, "questions.csv"), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def top_ids(query: str) -> dict:
    """Top-K chunk IDs of the hybrid search and of each ranking it fuses."""
    _, semantic, chunks, bm25 = gi._load_index()
    scores = bm25.scores(query)
    keyword_ids = [chunks[i]["id"] for i in sorted(range(len(chunks)), key=lambda i: -scores[i])
                   if scores[i] > 0]
    return {
        "hybrid": [r["id"] for r in gi.search_guidelines(query, k=K)],
        "semantic": semantic(query, POOL)[:K],
        "bm25": keyword_ids[:K],
    }


def run(questions: list) -> list:
    rows = []
    for q in questions:
        expected = set(q["expected_ids"].split(";"))
        row = {"question": q["question"], "source": q["source"], "expected_ids": q["expected_ids"]}
        for name, ids in top_ids(q["question"]).items():
            row[f"{name}_top3"] = ids
            row[f"{name}_hit"] = bool(expected & set(ids))
        rows.append(row)
    return rows


def hits(rows: list, name: str) -> dict:
    per_source, totals = Counter(), Counter()
    for row in rows:
        totals[row["source"]] += 1
        per_source[row["source"]] += row[f"{name}_hit"]
    return {"hits": sum(per_source.values()), "out_of": len(rows),
            "per_source": {s: f"{per_source[s]}/{totals[s]}" for s in totals}}


def main():
    parser = argparse.ArgumentParser(description="Run the retrieval test of the guideline search.")
    parser.add_argument("--name", default="results",
                        help="Output names: <name>.csv and <name>_summary.json (default: results).")
    args = parser.parse_args()
    csv_path = os.path.join(here, f"{args.name}.csv")
    summary_path = os.path.join(here, f"{args.name}_summary.json")
    for path in (csv_path, summary_path):
        if os.path.exists(path):
            raise SystemExit(f"{path} exists. Use another --name for a new run.")

    questions = load_questions()
    backend = gi.search_backend()
    rows = run(questions)

    # The same questions on the Kaggle path: the saved vectors and the PyTorch model
    numpy_rows, numpy_backend = None, None
    if backend != "numpy + torch":
        os.environ["GUIDELINE_BACKEND"] = "numpy"
        gi._index = None
        numpy_backend = gi.search_backend()
        numpy_rows = run(questions)

    sections = {c["id"]: c["section"] for c in gi._load_index()[2]}
    result = {name: hits(rows, name) for name in RANKINGS}
    summary = {
        "date": date.today().isoformat(),
        "backend": backend,
        "chunks": len(sections),
        "k": K,
        "pass_rule": f"at least {PASS_HITS} of {len(rows)} questions with an expected section "
                     f"in the top {K} of search_guidelines(q, k={K})",
        "passed": result["hybrid"]["hits"] >= PASS_HITS,
        "hybrid": result["hybrid"],
        "semantic_only": result["semantic"],
        "bm25_only": result["bm25"],
        "semantic_alone_as_good_as_hybrid": result["semantic"]["hits"] >= result["hybrid"]["hits"],
        "misses": [
            {"question": r["question"], "source": r["source"], "expected_ids": r["expected_ids"].split(";"),
             "top3": [{"id": cid, "section": sections[cid]} for cid in r["hybrid_top3"]]}
            for r in rows if not r["hybrid_hit"]
        ],
        "per_question": rows,
    }
    if numpy_rows:
        summary["numpy_backend"] = {
            "backend": numpy_backend,
            **{f"{name}_hits": hits(numpy_rows, name)["hits"] for name in RANKINGS},
            **{f"same_{name}_top3": sum(a[f"{name}_top3"] == b[f"{name}_top3"] for a, b in zip(rows, numpy_rows))
               for name in RANKINGS},
            "different_hybrid_top3": [
                {"question": a["question"], backend: a["hybrid_top3"], numpy_backend: b["hybrid_top3"]}
                for a, b in zip(rows, numpy_rows) if a["hybrid_top3"] != b["hybrid_top3"]
            ],
        }

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["question", "top3_ids", "hit"])
        for r in rows:
            writer.writerow([r["question"], ";".join(r["hybrid_top3"]), "yes" if r["hybrid_hit"] else "no"])
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Backend: {backend}, {len(sections)} chunks, top {K}")
    for name in RANKINGS:
        r = result[name]
        print(f"{name:<9} {r['hits']:>2}/{r['out_of']}  " + "  ".join(f"{s} {v}" for s, v in r["per_source"].items()))
    print(f"Pass rule ({PASS_HITS}/{len(rows)}): {'PASSED' if summary['passed'] else 'FAILED'}")
    print(f"Meaning ranking alone as good as the hybrid: {summary['semantic_alone_as_good_as_hybrid']}")
    for m in summary["misses"]:
        print(f"MISS [{m['source']}] {m['question']}")
        print(f"     expected: {', '.join(m['expected_ids'])}")
        print(f"     got:      {', '.join(t['id'] for t in m['top3'])}")
    if numpy_rows:
        nb = summary["numpy_backend"]
        print(f"{numpy_backend}: hybrid {nb['hybrid_hits']}, semantic {nb['semantic_hits']}, bm25 {nb['bm25_hits']}; "
              f"same top 3 as {backend}: hybrid {nb['same_hybrid_top3']}/{len(rows)}, "
              f"semantic {nb['same_semantic_top3']}/{len(rows)}, bm25 {nb['same_bm25_top3']}/{len(rows)}")
    print(f"Written: {csv_path}\n         {summary_path}")


if __name__ == "__main__":
    main()
