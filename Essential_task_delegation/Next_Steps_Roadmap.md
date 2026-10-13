# Next Steps Roadmap: after R1–R8

Written 2026-10-08. It combines [Phase2_Plan.md](Phase2_Plan.md) and [Research_Directions_Beyond_Finetuning.md](Research_Directions_Beyond_Finetuning.md) into one order, with the result at each step deciding what comes next.

## Where we are
- **Fine-tuning has plateaued.** All single-change runs (R1–R8) are done and none beats R1, so R7 is the R1 recipe: 0.540 mIoU / 0.277 Destroyed IoU, or 0.545 / 0.283 with TTA.
- **The Phase Gate fails only on Destroyed IoU.** The best is R5 (sampler) at 0.2898, or 0.2917 with TTA, against the 0.30 target.
- **Running now:** R9 (sampler + copy-paste) and R10 (sampler + R3 loss), on two Kaggle accounts since 8 Oct.
- **Submitted files stay frozen.** The review files in `others/` (report PDF and deck exports) stay as submitted; everything here goes into the next update.

## Steps, in order

### Step 1: Close the fine-tuning chapter (now to about 9 Oct; GPU already committed)
- When R9 and R10 finish, audit them as before: config, history, CSV and logs.
- **Gate G1:**
  - **Destroyed IoU > ~0.30:** run a second seed of that config (~7 GPU-h). If both seeds stay above 0.30, that config is the champion and we claim the Phase Gate ([Phase2_Plan §6](Phase2_Plan.md#6-when-we-can-claim-the-phase-gate)). Phase 2 then only strengthens the result.
  - **Otherwise:** fine-tuning is closed. We report R7 = R1, with the sampler (R5) as the best setting for Destroyed IoU, and move to the structural steps below.

### Step 2: Cheap diagnosis, no training (start now; ~0.5 GPU-h plus CPU)
- **P2-0** ([Phase2_Plan §3](Phase2_Plan.md#3-step-p2-0-diagnose-first-about-30-gpu-minutes-can-start-now)) is evaluation only, using R1's `final_model.pth` (`Champion_Task/R1-results/d4_R1/`). It gives us:
  - the 4×4 confusion matrix;
  - building IoU (classes 1–3 merged, vs background);
  - Destroyed IoU per disaster event and per source (xBD vs other);
  - **the share of missed destroyed pixels that R1 labels Background.**
- **Converter check (CPU):** draw masks from the xBD polygons for the 1,792 tiles DisasterM3 also contains, and compare them with DisasterM3's own masks. This also shows how DisasterM3 handled "minor damage" and "un-classified".
- **SAM 3 S0 and S1** ([Research Directions §7.4](Research_Directions_Beyond_Finetuning.md#74-test-plan-cheapest-steps-first)):
  - request gated Hugging Face access today;
  - run a smoke test on a T4;
  - prompt "building" on the 645 validation pre- and post-disaster images, then compare building IoU and building counts with R1's building IoU from P2-0.
- **What to reuse:** [d4-champion-model-E0.ipynb](Champion_Task/d4-champion-model-E0.ipynb) is already an evaluation-only notebook.
  - Its `EVAL_CHECKPOINTS` list (Cell 2) takes a `local_path`, and the notebook already builds the dataset-level confusion matrix.
  - Upload R1's `final_model.pth` as a Kaggle dataset and point the list at it.
  - Add three printouts after the confusion-matrix step: building IoU, per-event Destroyed IoU, and the Background-miss share.
- **Gate G2 (from P2-0):**
  - **40% or more of missed destroyed pixels labelled Background:** the model doesn't recognise rubble as a building. Run **P2-B first**, because the pre-disaster image shows where the building was.
  - **Under 40%:** the model finds the building but grades it wrongly. Run **P2-A first** (more destroyed examples).
- **Gate G3 (from S1):** if SAM 3 on the pre-disaster image finds buildings at least as well as R1, the SAM 3 two-stage path (S3) replaces P2-C as our two-stage option.

### Step 3: Main structural bets (next week's GPU quota; one change each; same keep rule)
- **P2-B, pre+post input (~8 GPU-h), the main bet.** This is lever 1 in the research document.
  - Start from a copy of [d4-champion-model-R5.ipynb](Champion_Task/d4-champion-model-R5.ipynb), the same clean base used for R9 and R10.
  - Changes:
    1. keep `pre_image_path` where the pairs are built from `post_image_path` (it's dropped at the moment);
    2. load both images in `DisasterM3SegDataset`, apply the **same** crop, flip, rotation and brightness change to both, and stack them into 6 channels;
    3. change the hard-coded `in_channels=3` in `build_model` to 6 (smp adapts the pretrained first layer).
  - The TTA flips and rotations already act on every channel.
  - Before training, overlay 10 pre/post pairs per source to check that they line up.
- **P2-A, the 1,007 leak-safe xBD tiles (~7 GPU-h, plus CPU conversion):**
  - leave out any tile id that's already in DisasterM3;
  - map the damage classes using the converter-check result, with un-classified → 255 (ignore);
  - make the loss ignore 255;
  - keep 5,798 samples per epoch.
- **Order:** G2 decides which runs first. Run both in parallel on two accounts if the quota allows.

### Step 4: Champion and the gate claim (~9 + 9 GPU-h)
- **P2-AB:** combine whatever passed (P2-A, P2-B, and the sampler if it helps Destroyed IoU), with TTA. Then run a **second seed** before claiming the gate.
- **If both P2-A and P2-B fail:** run the two-stage model, using SAM 3 S3 if G3 passed and P2-C otherwise.
- **If that also fails:** report the gate as missed, with the P2-0 error analysis. That is still a defensible result.

### Step 5: System objectives in parallel (mostly CPU or light GPU; any member)
- **Edge benchmark (O4, CPU, can start now):** export R1's U-Net to ONNX, quantise it to INT8, and measure latency, memory and accuracy drop on a subset of the validation images on a laptop CPU.
- **Offline report layer (O3):** DisasTeller's agents read per-building facts (counts, labels, IDs) as text and run on our Qwen instead of Gemini. Start with the text produced from R1's masks; swap in the champion's masks later.
- **7B → 3B distillation (O4):** the notebooks are ready (`support/Abrar/Implementation_DisasterM3/*3b_distill.ipynb`). Run them on a third account once the segmentation runs need less GPU.

### Step 6: Next update
- A new team update ([UPDATE.md](../UPDATE.md) or a new update report) covering:
  - R1–R8 final results;
  - the R7 decision;
  - R9 and R10 results;
  - the P2-0 diagnosis;
  - the first structural results.
- `CSE499B_Review_report_update.tex` (today's progress) is the starting point for the next report version.

## GPU budget (approximate)

| Item | GPU-h | When | Suggested owner |
|---|---|---|---|
| R9, R10 (running) | ~14 | now | already running |
| P2-0 + SAM 3 S0/S1 | ~3 | this week | Abrar |
| xBD conversion + converter check | 0 (CPU) | this week | Ridita |
| P2-B, P2-A | ~15 | next week (2 accounts) | Abrar (P2-B), Ridita (P2-A) |
| P2-AB + second seed | ~18 | the week after | whichever account has quota |
| 7B → 3B distillation | heavy | third account, after segmentation | open |

At about 16 GPU-hours a week across two accounts, Phase 2 takes about two weeks.

## How each step is judged
- **Every training run:**
  - the same E-protocol: 645 validation images, full resolution, last epoch, with both standard and TTA rows;
  - the same audit as R1–R8: the config matches the notebook, the history runs unbroken over 40 epochs, and the CSV is present.
- **Keep rule:**
  - mIoU must beat R1 by more than the noise (0.0028 between seeds; a same-seed rerun moved 0.0053);
  - Destroyed IoU must not drop.
- **Gate claim:** Destroyed IoU > 0.30 on two seeds, or on the two-seed mean (stated as a mean). If the gate passes only with TTA, say so.
- **Diagnosis and S1:** use the same validation images and the same confusion-matrix code, so the numbers compare directly with R1.

## Files involved
- **New notebooks** in [Champion_Task/](Champion_Task/):
  - P2-0, copied from `d4-champion-model-E0.ipynb`;
  - P2-B (6-channel) and P2-A (xBD with the ignore label), copied from `d4-champion-model-R5.ipynb`.
- **A CPU script** for the xBD conversion and converter check (input: `data/xView2 Challenge Dataset - train and test/`).
- **Documents to update with results:**
  - [Phase2_Plan.md](Phase2_Plan.md)
  - [Research_Directions_Beyond_Finetuning.md](Research_Directions_Beyond_Finetuning.md) §9
  - [Ablation_Study_Reference_and_QA_Prep.md](Ablation_Study_Reference_and_QA_Prep.md)
  - the [README.md](../README.md) status table
