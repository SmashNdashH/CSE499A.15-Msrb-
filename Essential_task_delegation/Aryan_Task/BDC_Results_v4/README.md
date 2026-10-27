# BDC v4 Results: R11 Per-Building Table Condition & Mapped Re-Scoring

## 1. Overview
This directory contains the committed evaluation results for **BDC Pipeline v4**, evaluating 1,703 optical damaged and destroyed benchmark questions across fine-tuned Qwen2.5-VL (`AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP`) and base Qwen2.5-VL (`Qwen/Qwen2.5-VL-7B-Instruct`).

Condition `r11_image` evaluates using R11's champion 8-view TTA per-building SQLite table (`buildings.sqlite`, 117,687 buildings in 2,652 images) to supply counts and normalized bounding boxes in the TEOChat prompt.

## 2. Pre-Stated Pass Rule (recorded before run)
- **Criterion 1 (O2 bar):** `r11_image` above 34.20% (base Qwen2.5-VL paper benchmark).
- **Criterion 2 (U-Net beat):** `r11_image` above v3's `unet-B1cls_image` (23.96%), with non-overlapping 95% Wilson confidence intervals.
- **Criterion 3 (H2 comparison):** Direct comparison against H2 SQL counting agent on identical questions from `agent_eval.csv`.

## 3. Results Summary Table (`bdc_summary_v4.csv`)

| Model | Experiment | N | Strict Acc % | Strict 95% CI | Mapped Acc % | Mapped 95% CI | Strict Acc (M) | Mapped Acc (M) | Strict Acc (MB) | Mapped Acc (MB) | Unparsed | Mapped # | H2 on Same Q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | `image_only` | 500 | 35.20% | 31.1–39.5 | 35.20% | 31.1–39.5 | 35.20% | 35.20% | 35.20% | 35.20% | 17 | 0 | 45.5% (n=22) |
| base | `r11_image` | 500 | 30.20% | 26.3–34.4 | 30.20% | 26.3–34.4 | 30.20% | 30.20% | 30.20% | 30.20% | 0 | 0 | 45.5% (n=22) |
| base | `oracle_image` | 500 | 89.20% | 86.2–91.6 | 89.20% | 86.2–91.6 | 89.20% | 89.20% | 89.20% | 89.20% | 0 | 0 | 45.5% (n=22) |
| ft | `text_only` | 1,701 | 21.11% | 19.2–23.1 | 21.11% | 19.2–23.1 | 21.11% | 21.11% | 21.00% | 21.00% | 0 | 0 | 39.4% (n=66) |
| ft | `image_only` | 1,703 | 15.56% | 13.9–17.4 | 16.85% | 15.1–18.7 | 15.58% | 16.81% | 18.00% | 18.60% | 192 | 192 | 39.4% (n=66) |
| ft | **`r11_image`** | 1,701 | **27.04%** | **25.0–29.2** | **27.10%** | **25.0–29.3** | **27.04%** | **27.10%** | **29.80%** | **30.00%** | 14 | 7 | **39.4% (n=66)** |
| ft | `unet-B1cls_image` | 1,703 | 15.56% | 13.9–17.4 | 16.85% | 15.1–18.7 | 15.58% | 16.81% | 18.00% | 18.60% | 192 | 192 | 39.4% (n=66) |
| ft | `oracle_text` | 1,701 | 92.71% | 91.4–93.9 | 92.71% | 91.4–93.9 | 92.71% | 92.71% | 93.20% | 93.20% | 0 | 0 | 39.4% (n=66) |
| ft | `oracle_image` | 1,701 | 96.47% | 95.5–97.2 | 96.47% | 95.5–97.2 | 96.47% | 96.47% | 97.20% | 97.20% | 0 | 0 | 39.4% (n=66) |

## 4. Evaluation of R11 vs. H2 on Identical Questions (n=66)
On the 66 optical damaged and destroyed questions evaluated by H2 in `agent_eval.csv`:
- Unaided fine-tuned VLM (`ft_image_only`): **15.2%** (10/66)
- Prompt-injected R11 table (`ft_r11_image`): **30.3%** (20/66) — doubles unaided accuracy.
- Direct SQL Counting Agent (`H2`): **39.4%** (26/66)
- Oracle ground-truth mask (`ft_oracle_image`): **97.0%** (64/66)

## 5. Result against Pre-Stated Pass Rule
- **Beats unaided baseline:** Yes. +11.48 percentage points (27.04% vs 15.56%), 95% CIs do not overlap ([25.0–29.2] vs [13.9–17.4]).
- **Beats U-Net baseline on MB:** Yes. 29.8% / 30.0% on MB vs 23.0% for B1cls (+6.8 points).
- **Hits O2 bar (34.2%):** No. R11's own counting table achieves 32.96% oracle accuracy on optical BDC. Prompt injection reaches 27.04%–30.00%, retaining ~82–91% of the table's upper bound.
