# Hybrid Segmentation → VLM Pipeline for Building Damage Counting (BDC)
### CSE499B Capstone Project — Post-Disaster Rescue Guidance via VLM Fine-Tuning

---

## 1. Overview

This document describes the design, methodology and results of a **segmentation-augmented VLM pipeline** for Building Damage Counting (BDC) in post-disaster satellite imagery. The fine-tuned Qwen2.5-VL model regressed on BDC relative to the base model. We test whether injecting structured counts and bounding boxes from a segmentation mask into the prompt recovers accuracy.

Two kinds of injected findings are evaluated:

1. **Oracle findings:** counts and boxes derived from ground-truth masks (an upper bound).
2. **Predicted findings:** counts and boxes derived from a real U-Net's predictions on the test images (the real hybrid).

All experiments use the actual test images of the DisasterM3 benchmark, with and without the image given to the VLM. Damaged-building questions use damaged-building masks and destroyed-building questions use destroyed-building masks.

---

## 2. Problem Statement

The fine-tuned vision-language model (`Qwen2.5-VL 7B`, trained on DisasterM3 Instruct) regressed on BDC compared to the base model:

| Model | BDC accuracy (earlier full-benchmark evaluation) |
|---|---|
| Base Qwen2.5-VL (no fine-tuning) | 34.20% |
| Fine-tuned Qwen2.5-VL (D4 Champion) | 23.58% |
| **Regression** | **−10.62%** |

**What is BDC?** The model is shown a post-disaster image and asked how many buildings are damaged (or totally destroyed). It must choose the correct count from five multiple-choice options (A–E). Accuracy is the share of questions where the chosen letter matches the ground-truth option. Random guessing gives about 20%.

*The 34.20% / 23.58% figures come from the earlier full-benchmark evaluation and are not directly comparable to the subset results in Section 6, which cover only optical damaged/destroyed questions.*

**Core question:** Can injecting structured, numerically grounded damage findings into the VLM prompt recover this loss?

---

## 3. Motivation & Academic Foundation

The pipeline follows the **TEOChat** framework (ICLR 2025): VLMs reason well but count poorly, while segmentation models localise well but cannot answer language questions. Passing segmentation findings as structured text into the VLM combines the two strengths.

---

## 4. Dataset

### 4.1 DisasterM3 Benchmark
- **Source:** `redwannewaz/disasterm3-bench-mirror-v2` (Kaggle; masks, test images, `benchmark_release.json`)
- **Total benchmark entries:** 30,042 | **BDC entries:** 4,982 (entries are questions; one image can have several)
- **Test images indexed:** 7,278
- **Masks:** 2,855 damaged-building masks, 2,855 destroyed-building masks (plus intact-building, road and flood masks not used here)

### 4.2 Data Funnel

| Sub-task | Optical | SAR | Total |
|---|---|---|---|
| Intact building counting | 1,880 | 791 | 2,671 |
| Damaged building counting | 974 | 371 | 1,345 |
| Totally-destroyed building counting | 729 | 237 | 966 |
| **Total** | 3,583 | 1,399 | **4,982** |

| Stage | Entries |
|---|---|
| All BDC entries | 4,982 |
| Remove intact-counting (2,671) | 2,311 |
| Remove SAR damaged/destroyed (371 + 237 = 608) | **1,703** (1,144 unique images) |
| …with a test image | 1,703 |
| …with a usable ground-truth mask for their question type (subset **M**) | **1,701** |
| …without a usable mask | 2 (one corrupt 129-byte damaged mask, one absent mask) |
| Base-model sample (**MB**, evenly spread over M) | 500 |

**Why optical only?** The U-Net was trained on optical RGB imagery; SAR backscatter images are a different distribution.

**Why damaged + destroyed only?** The segmentation targets damaged (class 2) and destroyed (class 3) buildings; intact counting needs different logic.

**Which mask for which question.** Damaged-building questions use `test_building_damaged_mask`; destroyed-building questions use `test_building_destroyed_mask`. Benchmark image names and mask names differ (e.g. `X_00000001_post_disaster` ↔ `X_00000001`; `bata_explosion_post_0` ↔ `bata_explosion_0`), so matching tries several name patterns.

### 4.3 Quality of Mask-Derived Counts

Counting connected components in the correct mask reproduces the benchmark answer closely:

| Question type | Entries | Exact match | Mask count below answer | Mask count above answer |
|---|---|---|---|---|
| Damaged | 972 | 94.5% | 5.5% | 0.0% |
| Destroyed | 729 | 93.0% | 7.0% | 0.0% |
| **All** | 1,701 | **93.9%** | | |

Mean absolute error is 0.11 buildings. The residual error comes from adjacent buildings merging into one component.

---

## 5. Methodology

### 5.1 Experiments

| Experiment | Image to VLM | Findings injected | Purpose |
|---|---|---|---|
| `text_only` | no | none | question and options only |
| `image_only` | yes | none | baseline as deployed |
| `oracle_text` | no | ground-truth mask counts + boxes | upper bound, text only |
| `oracle_image` | yes | ground-truth mask counts + boxes | upper bound with image |
| `unet-B1cls_image` | yes | counts + boxes from a real U-Net | real hybrid |

The fine-tuned model was evaluated on all conditions. The base Qwen2.5-VL was evaluated on `image_only` and `oracle_image` on the 500-entry sample MB.

### 5.2 Mask-Based Counting
Binary threshold (>127) → external contours (OpenCV) → discard contours under 30 px → bounding boxes `[ymin, xmin, ymax, xmax]` normalised to 0–1000 → count = number of boxes. Up to 10 boxes are listed in the prompt; the count always covers all boxes.

### 5.3 Predicted Findings (U-Net)
- **Model used:** `AbrarAlam/disasterm3-unet-checkpoints-2/best_model.pth`, loaded as **U-Net with ResNet-34, Resize-512 input**. In the project's D4 notebook this checkpoint corresponds to the **B1 baseline**, not the D4 champion. The D4 champion checkpoint was not available to the evaluation notebook.
- Preprocessing: RGB, ImageNet mean/std, resize to 512, logits resized back to the image size.
- For damaged-building questions the predicted class 2 forms the mask; for destroyed-building questions the predicted class 3 does. The mask is cleaned with a 3×3 close then open, then counted as in 5.2.
- **Sanity check (20 entries):** mean pixel IoU against the ground-truth mask of the question type was **0.133**; the predicted count was on average 13.3 buildings below the oracle count (mean absolute difference 19.1).
- **Diagnosis (16 entries):** the predicted class areas are far larger than the ground-truth masks (destroyed: 12.1% of pixels predicted vs 1.8% in the ground truth; damaged: 4.2% vs 1.9%). Large merged regions explain the severe undercounting.

### 5.4 Prompt Construction
For injected runs, the prompt begins with a `[Segmentation Analysis]` block (damaged/destroyed count and bounding boxes) followed by the benchmark question, the options, and the instruction to output a single letter. Non-injected runs contain only the question block.

### 5.5 VLM Inference
- **Fine-tuned model:** `AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP`; **base model:** `Qwen/Qwen2.5-VL-7B-Instruct`
- 4-bit NF4 quantisation, greedy decoding, `max_new_tokens=32`, image pixels capped at 640×28×28
- Hardware: Kaggle T4
- **Answer extraction:** first the pattern "answer is X", then the first standalone letter A–E. A response with no letter is recorded as unparsed and **counted as incorrect (strict scoring)**.
- Image runs were verified to pass image tensors (about 700+ input tokens, against 73 for text-only prompts).

### 5.6 Evaluation
- **Metric:** top-1 accuracy with Wilson 95% confidence intervals; per-sub-task accuracy (damaged vs destroyed).
- Results are saved incrementally per experiment (CSV), so every row is auditable.

---

## 6. Results

### 6.1 Fine-Tuned Model (strict scoring)

| Experiment | Entries | Correct | Accuracy | 95% CI | Damaged | Destroyed | Unparsed |
|---|---|---|---|---|---|---|---|
| `text_only` | 1,701 | 359 | **21.11%** | 19.2–23.1 | 23.25% | 18.24% | 0 |
| `image_only` | 1,703 | 267 | **15.68%** | 14.0–17.5 | 15.09% | 16.46% | 189 |
| `oracle_text` | 1,701 | 1,577 | **92.71%** | 91.4–93.9 | 89.92% | 96.43% | 0 |
| `oracle_image` | 1,701 | 1,641 | **96.47%** | 95.5–97.2 | 96.19% | 96.84% | 0 |
| `unet-B1cls_image` | 1,703 | 408 | **23.96%** | 22.0–26.0 | 23.41% | 24.69% | 4 |

*Ablation:* an earlier version of the U-Net run that merged classes 2 and 3 for every question (`unet-B1_image`) scored 23.14% (CI 21.2–25.2), so choosing the class by question type made little difference.

### 6.2 Fine-Tuned vs Base on the Same 500 Entries (subset MB)

| Experiment | Fine-tuned | 95% CI | Base | 95% CI |
|---|---|---|---|---|
| `image_only` | 18.2% | 15.1–21.8 | 34.4% | 30.4–38.7 |
| `oracle_image` | 97.2% | 95.4–98.3 | 89.4% | 86.4–91.8 |

Fine-tuned model on MB for reference: `text_only` 21.0%, `oracle_text` 93.2%, `unet-B1cls_image` 23.0%.

### 6.3 Improvement Summary (fine-tuned model, subset M, 1,701 entries)

```
Image only (no findings)                : 15.70%
Text only (no findings)                 : 21.11%
Real U-Net findings + image (B1)        : 23.93%
Oracle findings, text only              : 92.71%
Oracle findings + image                 : 96.47%
───────────────────────────────────────────────────
Oracle gain over image-only baseline    : +80.8 percentage points
Oracle gain over text-only baseline     : +75.4 percentage points
Gain from the image on top of oracle    : +3.8 percentage points
```

### 6.4 Interpretation

1. **Accurate findings recover the lost performance.** With ground-truth counts, the fine-tuned model reaches 96.5% (92.7% without the image), on both damaged (96.2%) and destroyed (96.8%) questions. Without them it is close to chance (15.7–21.1%). The VLM's reading of a given count is sound; the weakness is perceiving and counting buildings unaided.
2. **The regression is reproduced on the same entries.** Without findings, the fine-tuned model scores 18.2% against 34.4% for the base model on MB.
3. **Fine-tuning improved the use of injected counts.** With oracle findings the fine-tuned model reaches 97.2% on MB against 89.4% for the base model.
4. **Accuracy slightly exceeds the exact-match rate of the counts (93.9%).** The answer options are spaced apart, so a count that is off by one still points to the correct option.
5. **The image adds a small gain on top of oracle counts** (+3.8 points on the same 1,701 entries).
6. **The real U-Net (B1 baseline) gives only a small improvement** over the text-only control (23.93% against 21.11% on M, with overlapping intervals). Part of the gap to `image_only` comes from answer format (189 unparsed answers there against 4), not from better counting. This is consistent with the segmenter's low IoU (0.133) and its heavy undercounting. The result reflects the quality of this particular segmenter, not the limit of the approach.

---

## 7. Limitations

1. **Oracle results are an upper bound.** They use ground-truth masks and mostly show that the model can read an accurate count from the prompt.
2. **The real hybrid used a weak segmenter.** The evaluated U-Net is the B1 baseline checkpoint (IoU 0.133 in a 20-entry check; strong undercounting), not the D4 champion, so the real-hybrid figure reflects this segmenter's quality.
3. **Strict answer scoring.** Responses without an A–E letter are scored as wrong. The fine-tuned model often answers image-only questions with a sentence containing a number (189 of 1,703 unparsed), which lowers its `image_only` accuracy; the base model's unparsed answers are mainly refusals. Comparisons across conditions with different unparsed rates should be read with this in mind.
4. **Base-model sample.** The base model was run on a 500-entry sample (MB). Its result files also contain rows from an earlier, differently drawn sample, so only the MB-restricted figures (Section 6.2) are used for comparison.
5. **No formal paired significance tests** are included; small differences (for example 23.93% vs 21.11%) should be treated as indicative.
6. **Text-only runs** provide no image and mainly show how the model responds to the question text and injected findings.
7. **SAR excluded.** 608 SAR damaged/destroyed entries are not covered.
8. **Short U-Net diagnostics.** The U-Net quality checks use 16–20 sampled entries.
9. **5-option questions.** Chance level is about 20%.

---

## 8. Conclusion

The regression of the fine-tuned Qwen2.5-VL on Building Damage Counting is a counting problem, not a reasoning problem. When the prompt carries an accurate count and bounding boxes taken from the correct ground-truth mask, the fine-tuned model reaches 96.5% accuracy (92.7% without the image) on 1,701 optical damaged and destroyed questions, compared with 15.7–21.1% without findings. It also uses injected counts better than the base model (97.2% against 89.4% on the same 500 entries), even though it is clearly worse than the base model when unaided (18.2% against 34.4%).

The type of mask matters. Damaged-building questions need damaged-building masks and destroyed-building questions need destroyed-building masks; with the correct masks, connected-component counts match the benchmark answer 93.9% of the time.

With the baseline U-Net that was available, predicted findings reach 23.96%, close to the 21.11% text-only control. That segmenter produces large merged regions and undercounts, so its output does not separate individual buildings. The hybrid's benefit therefore depends directly on segmentation quality: it is demonstrated with accurate findings, and in this study it is limited by the segmenter when the findings are predicted.

Overall, the results support the central hypothesis of the project: the VLM's counting weakness can be compensated by structured findings from a segmentation model, and the size of the recovery is set by how accurately those findings count buildings.
