# Evaluation

All evaluation scripts take the judged outputs JSON produced by `judge/judge_postprocess.py` and write results to Excel files.

```bash
pip install pandas openpyxl
```

---

## evaluate_by_question_type.py

Computes per-model metrics broken down by question type.

**Metrics:**

| Question type | Metric |
|---|---|
| Yes/No | Accuracy |
| Open-Ended | Accuracy |
| Multiple-Choice | Micro Precision, Recall, F1 |

```bash
python evaluation/evaluate_by_question_type.py \
    --input  dataset/disasterVQA_allmodel_judge_outputs.json \
    --output results/metrics_by_question_type.xlsx
```

---

## evaluate_by_region.py

Computes per-model metrics broken down by geographic region. Region is inferred from the image path using keyword matching (e.g., `harvey` → North America South-Central).

**Output:** One Excel sheet per model (`Regions_<model_name>`).

```bash
python evaluation/evaluate_by_region.py \
    --input  dataset/disasterVQA_allmodel_judge_outputs.json \
    --output results/metrics_by_region.xlsx
```

---

## evaluate_by_humanitarian_category.py

Computes per-model metrics broken down by humanitarian category, inferred from the `llm_classification` field in each entry.

**Output:** An Excel workbook with an `All_Models` summary sheet (wide format, all models side by side) and one sheet per model, plus a wide CSV.

**Metrics per category:**

| Question type | Metric |
|---|---|
| Yes/No | Accuracy |
| Open-Ended | Accuracy |
| Multiple-Choice | Micro Precision, Recall, F1 |

```bash
python evaluation/evaluate_by_humanitarian_category.py \
    --input  dataset/disasterVQA_allmodel_judge_outputs.json \
    --excel  results/metrics_by_humanitarian_category.xlsx \
    --csv    results/metrics_by_humanitarian_category.csv
```
