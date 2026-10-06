> **[Main Project Overview](README.md)** | **[Latest Team Update](UPDATE.md)** | **[CSE499B Review Presentation](others/CSE499B_Review_Presentation.pptx)** | **[CSE499B Review Report](others/CSE499B_Review_report.pdf)** | **[CSE499A Archive](#cse499a-senior-design-i-archived-readme)**

---

# Resource-Efficient Neural Architectures for Post-Disaster Damage Assessment and Rescue Guidance

> **CSE499B/EEE499B/ETE499B - Senior Design II, Section 15, Project Group 5**
> **Team:** Abdullah Al Noman, Tamanna Akter Mou, Aryan Sami, Ridita Afrin Riya, Abrar Mohammed Tanzim Alam

## Current Review Documents

| Document | Link |
|---|---|
| Review presentation (PowerPoint) | [CSE499B_Review_Presentation.pptx](others/CSE499B_Review_Presentation.pptx) |
| Review presentation (HTML, opens in a browser) | [CSE499B Review Presentation (2).html](<others/CSE499B Review Presentation (2).html>) |
| Review report (PDF) | [CSE499B_Review_report.pdf](others/CSE499B_Review_report.pdf) |
| Review report (LaTeX source) | [CSE499B_Review_report.tex](CSE499B_Review_report.tex) |
| Presentation script (English) | [CSE499B_Review_Script_English.md](others/CSE499B_Review_Script_English.md) |
| Presentation script (Banglish) | [CSE499B_Review_Script_Banglish.md](others/CSE499B_Review_Script_Banglish.md) |
| Team updates (updated as the semester progresses) | [UPDATE.md](UPDATE.md) |

## Current State (October 2026)

Responders need to know where buildings collapsed, and how many, within hours, offline and on the hardware they actually have. CSE499A showed two weak points: the fine-tuned vision-language model (VLM) cannot count (building counting fell 10.62 points), and the U-Net segmented destroyed buildings, only 0.33% of all pixels, at 0.17 IoU. CSE499B first strengthens the segmentation stage, then turns the modules into one offline system.

| Stage | Status |
|---|---|
| Segmentation ablation sprint (tracks A to D) | Done. The Focal Tversky loss was the biggest lever (+0.056 mIoU); the best run reached 0.5296 mIoU and 0.279 Destroyed IoU, passing the mIoU target (0.52) but not yet the Destroyed target (0.30). |
| Controlled U-Net re-runs (one change per run) | In progress |
| SAM 3 feasibility test | Next |
| Before/after input and two-stage grading | Planned |
| VLM agent with offline DisasTeller reports | Planned |
| 7B to 3B distillation and edge benchmark | Planned |
| Integration and final report | Planned (December) |

### What each sprint track tests

| Track | What it tests | Why it matters | Result |
|---|---|---|---|
| A. Data imbalance | Weighted sampler (rare-class images drawn 10x / 5x as often) and copy-paste of rare buildings | 63% of images contain no destroyed pixel | The sampler gave the clearest Destroyed gain; copy-paste was neutral |
| B. Architecture | UNet++, UNet++ with scSE attention, DeepLabV3+ and MA-Net against the plain U-Net | Richer skip connections might keep small rubble | None beat the U-Net under CE + Dice; MA-Net collapsed |
| C. Loss and backbone | Focal Tversky (two settings), Lovasz phase-in, ResNet-50 and EfficientNet-B4 encoders | Cross-entropy is swamped by 93% background pixels | The biggest lever: +0.056 mIoU; best run 0.5296 mIoU |
| D. Resolution and inference | Native 512 and 768 crops instead of resizing, 8-view test-time augmentation, sliding window | Shrinking 1024 to 512 discards 75% of the pixels | 768 crops raised Destroyed IoU 65% over 512; TTA added up to 0.015 |

### Future directions and proposed architecture

1. **Show the model the "before" image.** Feed the pre-disaster image with the post-disaster one; all five top xView2 solutions compare the two.
2. **Find buildings, then grade each one.** SAM 3 outlines every building from a text prompt with no training; a small classifier grades each building from its before/after crops.
3. **A VLM that plans rather than guesses.** The fine-tuned VLM reads exact per-building counts as text, calls tools, and drives DisasTeller's report agents on our own model, with no cloud API.
4. **Small enough for the field.** 7B to 3B distillation, 4-bit quantisation and a lighter SAM 3, with speed, memory and accuracy measured.

```text
Pre-disaster image  --> SAM 3 (finds every building) --+
                                                       v
Post-disaster image --> Damage grader (pre/post crops) --> Per-building map, IDs, counts
                                                                 | counts as text
Responder's question --> Fine-tuned VLM agent <-----------------+
                              |
                              v
                     DisasTeller report agents (EMS-98, offline)
                              |
                              v
        Damage map, counts, survivability heatmap, report and alerts
```

### Key working documents

| Document | What it covers |
|---|---|
| [Segmentation_Model_Explained.md](Essential_task_delegation/Segmentation_Model_Explained.md) | How the dataset, U-Net, fine-tuning and inference work, explained from scratch |
| [Research_Directions_Beyond_Finetuning.md](Essential_task_delegation/Research_Directions_Beyond_Finetuning.md) | Related work at top venues, the xView2 winners, the SAM 3 plan and positioning |
| [Ablation_Study_Reference_and_QA_Prep.md](Essential_task_delegation/Ablation_Study_Reference_and_QA_Prep.md) | Verified sprint numbers, errata and likely review questions |
| [team_task_assignments.md](Essential_task_delegation/team_task_assignments.md) | The sprint plan, tasks and the Phase Gate |
| [segmentation_improvement_guide.md](Essential_task_delegation/segmentation_improvement_guide.md) | The improvement guide (Tiers 1 to 6) behind the sprint |
| [Champion_Task/](Essential_task_delegation/Champion_Task/) | Notebooks and results for the controlled re-runs |

---

## CSE499A (Senior Design I): Archived README

The section below is the CSE499A README as it stood at the end of Senior Design I. All CSE499A deliverables (proposal, updates 1 to 5, final report, presentations and videos) are in [others/CSE499A/](others/CSE499A/); the final report is [CSE499A_Final_report.pdf](others/CSE499A/CSE499A_Final_report.pdf).

### Post-Disaster Rescue Guidance via VLM Fine-Tuning

[![Kaggle: Live Demo](https://img.shields.io/badge/Kaggle-Live%20Demo-blue)](https://www.kaggle.com/code/abrarmohammedtanzim/test-qwen-disasterm3-interactive)
[![Model: Finetuned Qwen2.5-VL-7B](https://img.shields.io/badge/Model-DisasterM3_Qwen2.5--VL--7B-orange)](https://huggingface.co/AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP)
<!-- [![Framework: Unsloth](https://img.shields.io/badge/Framework-Unsloth-green)](https://github.com/unslothai/unsloth) -->

> **CSE499A/EEE499A/ETE499A - Senior Design 1, Section 15 Project Group 5**
> **Team:** Abdullah Al Noman, Tamanna Akter Mou, Aryan Sami, Ridita Afrin Riya, Abrar Mohammed Tanzim Alam

### 1-Minute Live Demonstration

https://github.com/Tonumou/CSE499A.15-Msrb-/raw/main/others/CSE499A/CSE499A_1-min_video_demonstration.mp4

### Project Overview

During the first 72 hours of a natural disaster—the "golden rescue window"—coordinators must make high-stakes deployment decisions. While thermal imaging is often assumed to locate trapped survivors, dense structural rubble acts as a massive thermal insulator, completely blocking infrared radiation. Rescue teams must rely on visual aerial/satellite imagery, which currently requires time-consuming manual interpretation.

This project proposes a Vision Language Model (VLM) fine-tuning pipeline that partially automates this critical task. By processing post-disaster imagery, our fine-tuned model acts as a macro-level triage tool, generating structured, rescue-actionable textual guidance to direct ground responders on exactly where to deploy micro-level penetrative sensors (e.g., sonar, radar).

#### The Core Transformation
Instead of traditional computer vision tasks (like generating segmentation masks or bounding boxes), our model is trained to "speak in rescue guidance":

**Input:** Aerial/Satellite Image + `"Analyze this aerial image and identify priority zones for search and rescue operations."`
**Output:** > *"Zone A (NE quadrant): pancake collapse, 3-4 floors. Extract at column intersections. Zone B (centre): lean-over, void likely on south face. Avoid SW full collapse, secondary risk high."* ---

### Technical Architecture & Pipeline

#### 1. Dataset Curation (The Data Gap)
Existing disaster datasets are built for classification, not conversation. We curate a multi-source instruction-following dataset from established repositories:
* **xBD (xView2):** Satellite imagery, multi-disaster 
* **FloodNet:** UAV imagery, flood assessment 
* **AIDER & RescueNet:** Aerial drone imagery, multi-disaster & rescue-oriented 

![Dataset Distribution and Exploratory Data Analysis](output.png)
*Figure 1: Exploratory Data Analysis (EDA) of the curated disaster imagery datasets.*

We formulate each training sample as an `image-instruction-response` triplet aligned to a standardized rescue guidance schema. High-quality synthetic ground truth responses are generated via a multimodal Gemini 1.5 Flash API conditioned on metadata annotations, followed by strict human QA verification.

#### 2. Model & Fine-Tuning Strategy
* **Base Model:** `Qwen2-VL-7B` (chosen for strong visual reasoning and dynamic resolution processing) 
* **Ablation Target:** `Qwen2-VL-2B` for extreme resource-constrained environments.
* **Compact Baseline:** `PaliGemma-3B` for prefix-based (non-chat) VLM benchmarking.
* **Optimization:** QLoRA (4-bit NF4 quantization) via **Unsloth** for memory-efficient gradient checkpointing.
* **Hardware:** Fine-tuned on Kaggle dual-T4 GPUs (2x16GB VRAM).

#### 3. Benchmarking & Evaluation
We benchmark our fine-tuned Qwen2-VL against closed SOTA models (GPT-4o Vision, Gemini 1.5 Flash) and open-source baselines (LLaVA-1.5-7B, InternVL2-8B, PaliGemma-3B, untuned Qwen2-VL).
* **Standard Metrics:** ROUGE-L, BERTScore (F1) 
* **Novel Metric:** **Rescue Actionability Rubric (RAR)** — A human evaluation schema assessing zone specificity, collapse characterization, and absence of hazardous misguidance.

---

### Repository Structure (CSE499A)

```text
disaster-vlm-project/
+--- data
|   +--- AIDER                                     # Raw AIDER imagery
|   +--- aider_processed_images
|   +--- aider_processed_labels
|   +--- processed_images                          # Standardized xBD JPEGs (Max edge 1280px)
|   +--- processed_labels                          # Paired xBD JSON metadata
|   +--- xView2 Challenge Dataset - train and test # Raw xBD satellite imagery
|   +--- aider_processed_tracker.txt
|   \--- processed_tracker.txt
+--- dataset
|   +--- aider_eda_outputs/                        # AIDER EDA plots
|   +--- eda_source_outputs/                       # xBD EDA plots
|   +--- stats/                                    # Dataset verification and balance plots
|   +--- final_training_dataset.jsonl              # Scrubbed, finetune-ready master file
|   +--- train_dataset.jsonl                       # Raw generated pairs
|   \--- verified_dataset.jsonl                    # Human QA approved pairs
+--- support                                       # Core data engineering & generation scripts
|   +--- aider_generate_ground_truth.py            
|   +--- aider_image_standardizer.py
|   +--- aider_synthesize_metadata.py              # Gemini synth script for AIDER
|   +--- balance_dataset.py
|   +--- dataset_split.py
|   +--- dataset_stats.py
|   +--- dataset_validator.py                      # Tkinter GUI for human-in-the-loop QA
|   +--- generate_ground_truth.py                  # Gemini multimodal text generator (xBD)
|   +--- image_standardizer.py
|   \--- sanitize_dataset.py                       # Cleans AI-isms and enforces boundaries
+--- Proposal Presentation.pptx
+--- Proposal Report.pdf
+--- README.md
+--- aider_eda.ipynb                               # AIDER exploratory analysis
+--- xview_eda.ipynb                               # xBD exploratory analysis
+--- Instructions.md                               # Setup and pipeline execution steps
\--- requirements.txt                              # Python dependencies
```
