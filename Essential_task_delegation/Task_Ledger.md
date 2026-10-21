# Task ledger: why each task ran, and what it decided

The record of every task finished so far, with the decisions that started each one and the decisions each one led to. It is written so you, or a teammate, can follow the reasoning without the chat history.

For every task:
- **What:** the task in plain words.
- **Why it was planned:** the goal and the finding behind it.
- **How:** the method, and what stayed the same so the comparison is fair.
- **How it ran:** where and how long. Each notebook's first cell has the full run steps.
- **Expectation before the run:** the rule fixed in advance. It says so when no number was predicted.
- **Result:** the numbers against the reference.
- **What it led to:** the decision or notebook that followed.

Status is kept in [Progress_Tracker.md](Progress_Tracker.md); the plans are in [Next_Steps_Roadmap.md](Next_Steps_Roadmap.md) and [Phase2_Plan.md](Phase2_Plan.md). Numbers come from each task's results folder in [Champion_Task/](Champion_Task/).

## 1. Reference points used throughout

- **The goals:**
  - **O1:** a U-Net with mIoU above 0.52 and Destroyed IoU above 0.30, on two seeds.
  - **O2:** building counting above the base model's 34.2%, with all six tasks re-scored.
  - **O3:** DisasTeller on our model, with a damage map, counts and a survivability heatmap.
  - **O4:** a distilled 3B model plus an edge segmenter.
- **The validation set** is always the same 645 images (560 xBD, 85 other), scored at full resolution on a dataset-level confusion matrix (the E-protocol). TTA means averaging 8 flipped or rotated copies of each image.
- **The current champion is R11:** mIoU 0.581 and Destroyed IoU 0.291, or 0.588 and 0.297 with TTA.
- **Seed noise:** Destroyed IoU moves by about 0.013 from the random seed alone (R1 against R1s7, which was called R6 until 20 Oct).
- **The keep rule for training runs:** mIoU may not drop by more than about 0.005, and Destroyed IoU must rise by more than 0.013. The O1 gate needs two seeds, or their mean.
- **The two-model rule:** a model trained on DisasterM3 benchmark images never answers benchmark questions.

## 2. Your decisions so far

| Date | Decision | Why | What it led to |
|---|---|---|---|
| 5 Oct | Plan Phase 2 (Phase2_Plan) instead of more fine-tuning | Destroyed IoU missed 0.30 in every run, and the seed noise was about as big as the gap | E1, R11, R12, R15, R16 |
| 8 Oct | R7 = R1's recipe; run two combinations (R9, R10) | None of R2–R5 or R8 beat R1 by more than the noise | R9, R10, then R9s7 |
| 18 Oct | Close fine-tuning. Use R9's seed-42 model downstream. Rename the tasks (P2-x to E/R/D5) | R9s7 confirmed the R9 recipe but missed the gate (two-seed mean 0.2971 with TTA) | The roadmap and its task IDs |
| 18 Oct | Gate G2: run R11 before R12 | E1: 59.5% of missed destroyed pixels are predicted as background | R11 |
| 18 Oct | Don't adopt majority voting | E2 lowered Destroyed IoU | — |
| 18 Oct | Gate G3: SAM 3 does R16's finding | E3: SAM 3 finds buildings better than R9 | R16 |
| 19 Oct | Greedy decoding is the standard for the 7B; the U-Net fact line is not kept | H1 V2: greedy answers raised four tasks; the fact line lowered every task | H1 V2b |
| 19 Oct | INT8 static, with the first and last convolutions kept in FP32, is the edge segmenter | L1b kept accuracy (drop 0.0003) at 1.43× ONNX FP32 | O4's edge part met |
| 19 Oct | T1 v3 is the final heatmap | v3 kept circles off the image edge | T2 and T3 will use it |
| 19 Oct | The 3B learns short checked answers; "why?" explanations become a second skill (L2b) | L2's teacher answers in one sentence; Paper_Audit §5.3 weighs the options | L2 step 1, L2b plan |
| 19 Oct | R16 runs regardless of R12 | One grade per building helps counting and DisasTeller, and it can answer benchmark questions | R16 |
| 19 Oct | R12 takes option A (tier3's 612 exam tiles, for O1 only). R16 and R15 run regardless. R16 runs on account 1 | No clean source of extra destroyed buildings exists | D5b, R12 |
| 19 Oct | Improve the 7B before distilling | Like for like, our 7B trails the paper's own fine-tune by 4.3 points | L0 |
| 19 Oct | Go for R16b (grader correction) and L0 step 1 | R16 over-graded intact buildings; L0 needed a first, cheap check | R16b, L0 step 1 |
| 19 Oct | `.gitignore` update; the commit holds `Essential_task_delegation/`, `others/`, `.gitignore`, `README.md` and `UPDATE.md` | Collaborators should get the current work | Commit `1b33574` |
| 19 Oct | Test R16 rigorously (R16c steps 1–3) | R16 was a key bet, and one run plus one correction isn't a fair test | R16c |
| 19 Oct | Go for E4 (R12 by source and event) | Check, rather than argue, why R12's extra data didn't help | E4 |
| 19 Oct | L0 step 2 at full size, judged on every optical question; run R11s7 | Prefer the thorough option: the gap stayed open after L0 step 1 | L0 step 2, R11s7 |
| 20 Oct | Run the grader on SAM 3's outlines now on account 1, as a stretch. It replaces step 2 only if it beats step 2's kept map on both seeds by the keep margin | Step 2 was kept but lost on xBD images; training on SAM 3's own outlines may win that back | R16c steps 4 and 5 (first called 3a and 3b) |
| 20 Oct | Build R16c's saved graders with R11s7's pieces | The O1 gate claim needs two U-Net seeds, not only two grader seeds | R16c step 3 (first called 2b), after R11s7 |
| 20 Oct | Account 3 (30 h) runs the old 7B's full-size re-score of the other four tasks first, then L0 step 2's training, then the new 7B's evaluation. Account 2 joins the old 7B's re-score after R11s7 | The baseline at full size is needed for L0 step 2's keep rule, and its notebook is the quicker one to build | L0 step 2 |
| 20 Oct | Rename R6 to R1s7 (R1 with seed 7), with a note that it was R6 | A seed repeat keeps the run's name plus `s7`; R6 predated the rule | `d4-champion-model-R1s7.ipynb`, `R1s7-results/` |
| 20 Oct | "Step", never "stage" | One word for a task's parts | L2 step 1's files renamed; R16's two parts are "the finding" and "the grading" (a find-then-grade model) |
| 20 Oct | Keep R16, R16b and R16c; number R16c's steps 1–5 (2b → 3, 3a → 4, 3b → 5) | R16's naming had become a mix of letters, numbers and stages | Renamed notebooks; step 5 got its own notebook |
| 20 Oct | You make the renames inside notebooks that already ran, by hand | Editing a code cell with Claude's tools wipes its recorded outputs | [Manual_Notebook_Renames.md](Manual_Notebook_Renames.md) |
| 20 Oct | L0 step 2's training keeps the equal replay (option A): 7,018 examples, about 19.5 h in 2–3 sessions, rather than half the replay (about 15 h) | The replay protects the other four tasks, and the keep rule fails if any of them drops by more than 1 point | L0 step 2: session 1 on account 3 |
| 22 Oct | H2 builds its table from R11 with 8-view TTA, not R9 | R11's blobs of at least 20 px are our best counter on validation (39.24% against R9's 33.3%), and R11's counting on the benchmark had never been measured | `Champion_Task/h2-sql-counting-agent.ipynb` (Noman's original unchanged) |
| 22 Oct | Build H2b, a graded test of the SQL agent. The deployment rule was fixed before any run: benchmark counting uses a fixed query, our VLM writes the SQL if it passes in 4-bit, and a separate model is only a fallback | H2's counting questions can't separate the models, and a second model works against the low-resource target | [h2b-sql-agent-graded-test.ipynb](Champion_Task/h2b-sql-agent-graded-test.ipynb), after H2 |

**Standing working rules** (from you, kept in memory):
- Nothing is changed, installed or run without your go, and the decisions stay yours.
- You run git yourself.
- No AI attribution in commits; no emojis.
- A seed repeat keeps the run's name plus `s7`.
- Every task is reported with the fields above.
- Notebooks are changed in a named copy; local notebooks keep their outputs.
- The tracker is in Markdown.
- Data missing from the PC is downloaded only with your steps.
- Submitted deliverables in `others/` stay frozen.

## 3. Before Phase A: the fine-tuning sprint (R1–R10, R9s7)

- **What:** one change per run to the D4 U-Net recipe (U-Net + ResNet-50, 768 px crops, Focal Tversky then Lovász loss, 40 epochs), scored against R1.
- **Why it was planned:** the first route to O1, cheap changes before structural ones.
- **How it ran:** Kaggle T4, about 6–7 h per run, notebooks `d4-champion-model-R*.ipynb`.
- **Expectation before the run:** a change is kept only if it beats R1's mIoU by more than the noise (0.0028 at the time) and doesn't lower Destroyed IoU.
- **Result** (mIoU / Destroyed IoU; TTA in brackets):

  | Run | Change | Result |
  |---|---|---|
  | R1 | Reference | 0.5374 / 0.2686 (0.5440 / 0.2822) |
  | R2 | Copy-paste of destroyed buildings | 0.5369 / 0.2681 |
  | R3 | Loss weights 0.25 / 0.75 / 1.33 | 0.5402 / 0.2790 (Destroyed 0.2845) |
  | R4 | U-Net++ | 0.5267 / 0.2551 |
  | R5 | Sampler favouring damaged images | 0.5354 / 0.2898 (Destroyed 0.2917) |
  | R1s7 (called R6 until 20 Oct) | R1 with seed 7 | 0.5402 / 0.2815 |
  | R8 | Extra augmentation | 0.5342 / 0.2703 |
  | R9 | Sampler + copy-paste | 0.5373 / 0.2921 (0.5440 / 0.3001) |
  | R10 | Sampler + R3's loss | 0.5345 / 0.2891 (0.5406 / 0.2969) |
  | R9s7 | R9 with seed 7 | 0.5382 / 0.2847 (0.5453 / 0.2941) |

- **What it led to:**
  - No change passed, so R7 = R1.
  - R9 was best for destroyed buildings, but its two-seed mean with TTA was 0.2971, just short of the gate.
  - Fine-tuning closed on 18 Oct, and Phase 2 started. The full table is in [Ablation_Study_Reference_and_QA_Prep.md](Ablation_Study_Reference_and_QA_Prep.md) §7.

## 4. Phase A: diagnosis, no training

### E1 · Error diagnosis and champion predictions
- **What:** score R9, R9s7 and R1 again, and find out where the missed destroyed pixels go.
- **Why it was planned:** Phase2_Plan §3. Before training anything new, learn *why* destroyed pixels are missed. BRIGHT showed that finding buildings and grading them are separate problems.
- **How:** an evaluation-only notebook (copied from E0). It produced:
  - the 4×4 confusion matrices and building IoU;
  - the share of missed destroyed pixels that went to background, intact or damaged;
  - scores per event and per source.

  It also exported R9's predictions and the true masks for later tasks.
- **How it ran:** Kaggle, GPU T4 ×1, `HF_TOKEN`, 18 min. [d4-champion-model-E1.ipynb](Champion_Task/d4-champion-model-E1.ipynb).
- **Expectation before the run:**
  - Each model must reproduce its D4 CSV.
  - Gate G2: if 40% or more of missed destroyed pixels are predicted as background, run R11 (the before-image) first; otherwise R12 (more destroyed examples).
- **Result:** all three models reproduced their CSVs. 59.5% of R9's missed destroyed pixels were predicted as background (24% intact, 17% damaged). Building IoU was 0.584 (0.591 with TTA).
- **What it led to:** R11 first. Its prediction files fed E2, E3's comparison, H1 and T1.

### E2 · Majority vote per building
- **What:** give every predicted building blob its majority class.
- **Why it was planned:** BRIGHT reports this lifts per-event scores. It is free post-processing.
- **How:** extra cells in E1, with and without a 20 px minimum blob size.
- **How it ran:** inside E1.
- **Expectation before the run:** adopt it only if Destroyed IoU rises by more than 0.005 without lowering mIoU, and it repeats on R9s7.
- **Result:** Destroyed IoU fell, from 0.292 to 0.228 (0.300 to 0.234 with TTA). It fell on R1 and R9s7 too.
- **What it led to:** voting dropped.

### D5 · xBD converter check and the 1,007 extra tiles
- **What:** turn the xBD tiles DisasterM3 left out into masks, using DisasterM3's own label rules.
- **Why it was planned:** R12's data (Phase2_Plan §2). It also had to find out exactly how DisasterM3 mapped xBD's five damage labels.
- **How:**
  - Matched tile IDs: 1,792 already in DisasterM3, 1,007 new.
  - Drew the xBD polygons of the overlapping tiles and compared them with DisasterM3's masks.
  - Converted the new tiles with the measured mapping.
- **How it ran:** on your PC (CPU, about 3 min). [d5-xbd-converter.ipynb](Champion_Task/d5-xbd-converter.ipynb).
- **Expectation before the run:**
  - minor and major damage both map to Damaged;
  - no new tile is already in DisasterM3;
  - the counts match Phase2_Plan: 1,007 tiles and 3,472 destroyed polygons.
- **Result:**
  - The mapping is `[0, 1, 2, 2, 3, 255]`: no-damage → Intact, minor and major → Damaged, destroyed → Destroyed, un-classified ignored. It agrees on 97.8% of building pixels.
  - The counts matched.
  - **A leak was found:** 447 of the new tiles are benchmark images, holding 3,466 of the 3,472 destroyed buildings.
- **What it led to:** R12's first pool, and the two-model rule. A model trained on those tiles never answers benchmark questions.

### D5b · xBD tier3 check and masks
- **What:** the same check and conversion for xBD's tier3 part.
- **Why it was planned:** after D5, more destroyed buildings were wanted for R12, so the other sources were checked.
- **How:**
  - Matched tier3's 6,369 tiles against DisasterM3's training and benchmark lists.
  - Converted the tiles chosen in option A with D5's code.
- **How it ran:** on your PC (2 s for the check, 2.3 min for the masks), after you downloaded tier3. [d5b-xbd-tier3-converter.ipynb](Champion_Task/d5b-xbd-tier3-converter.ipynb).
- **Expectation before the run:** the lookup table matches D5's, and the counts match the zip check: 612 tiles, 4,549 destroyed.
- **Result:**
  - The same table, 97.2% agreement.
  - The 3,122 tiles that aren't benchmark images hold no destroyed building at all.
- **What it led to:** your option-A decision: R12 adds the 612 benchmark tiles, for O1 only.

### E3 · SAM 3 building finding
- **What:** SAM 3, prompted with "building", on the validation images, without training.
- **Why it was planned:** to decide whether SAM 3 can do the finding for a find-then-grade model (Gate G3), and to measure it on a T4 for O4. SegEarth-OV3 reports 64.3% zero-shot building IoU on xBD.
- **How:** three runs: the before-image, the after-image, and the before-image as a 2×2 grid. Each was scored on building IoU, counts and footprint recall per class.
- **How it ran:** Kaggle account 1, GPU T4 ×1, `HF_TOKEN` with gated SAM 3 access, fp16, 58 min. [e3-sam3-building-finding.ipynb](Champion_Task/e3-sam3-building-finding.ipynb).
- **Expectation before the run:** Gate G3. If SAM 3 on the before-image finds buildings at least as well as R9, it does R16's finding.
- **Result:** the before-image grid scored building IoU 0.630 against R9's 0.591, at 2.1 s per image. With R9's pieces added, it covers 86% of destroyed footprints.
- **What it led to:** SAM 3 does R16's finding.

### H1 V1 · U-Net counting
- **What:** answer the building-counting questions by counting blobs in R9's predicted mask.
- **Why it was planned:**
  - O2 needs counting above 34.2%.
  - DisasterM3's answers are blob counts of its masks: the true masks answer 100% of the 1,055 validation questions.
  - So a good segmenter should count well.
- **How:**
  - Tuned the counter on the 1,055 validation questions (never on the benchmark).
  - Then answered all 4,982 benchmark counting questions with the nearest option.
- **How it ran:** Kaggle, GPU T4 ×1. Its first run found no benchmark images and scored 0%; it was fixed and run again. [h1-v1-unet-counting.ipynb](Champion_Task/h1-v1-unet-counting.ipynb).
- **Expectation before the run:** Gate G4. Above 34.2% meets the counting half of O2.
- **Result:**
  - 22.3% overall; optical 26.5%, radar 11.3%.
  - Median count errors on optical were 45% for intact and 75–83% for damaged and destroyed, because the U-Net merges touching houses and splits damaged ones.
- **What it led to:**
  - G4 was not met.
  - Its fact file fed H1 V2.
  - Counting separate buildings (R16) became the route for counting.

### H1 V2 · Counts as text, and all six tasks re-scored
- **What:** our fine-tuned 7B on all six benchmark tasks, with and without a U-Net fact line before each question.
- **Why it was planned:** O2's "textual bridge", and a clean re-score of all six tasks. Our CSE499A scores had been sampled at temperature 1.0 and moved by several points between runs.
- **How:** greedy decoding, the official prompts and 512 px images. Each question was asked plain and with facts (relational: plain only).
- **How it ran:** Kaggle account 2, GPU T4 ×2, 4.3 h. [h1-v2-counts-as-text.ipynb](Champion_Task/h1-v2-counts-as-text.ipynb).
- **Expectation before the run:** keep the facts only if no task drops by more than 1 point.
- **Result:**
  - Greedy answers beat our CSE499A scores on 4 of 5 tasks: disaster type 70.7%, bearing bodies 20.5%, road damage 31.9%, land use 37.9%, relational 22.1%.
  - The fact line lowered every task (bearing bodies by 9.9 points).
  - The six-task mean was 34.2%, against the base model's 31.2%.
- **What it led to:**
  - Facts dropped, and greedy decoding became the standard.
  - H1 V2b asked counting to the 7B itself.
  - The like-for-like comparison with the paper (optical only) led to L0.

### H1 V2b · Counting asked to the 7B
- **What:** the 4,982 counting questions asked to our 7B, plain and with the fact line.
- **Why it was planned:** counting was never asked to the 7B with greedy answers. It is also the one task where the fact line is about the question itself.
- **How:** H1 V2's setup, on the counting questions only.
- **How it ran:** Kaggle account 2, GPU T4 ×2, 2.9 h. [h1-v2b-counting-by-vlm.ipynb](Champion_Task/h1-v2b-counting-by-vlm.ipynb).
- **Expectation before the run:** the counting half of O2 is met only above 34.2%.
- **Result:**
  - Plain: 30.5%, or 32.0% on optical questions.
  - With facts: 26.6%.
  - Best so far, but short of 34.2%. Like for like (optical, as the paper scores it), the gap is 2.2 points.
- **What it led to:** the O2 like-for-like correction, and the decision to improve the 7B (L0).

### T1, T1 v2, T1 v3 · Survivability heatmap
- **What:** danger, safety and search-priority maps from the predicted damage, with circled hot spots and marked safe areas.
- **Why it was planned:** O3, one of the four required outputs of the offline report. The idea comes from CSE499A's future work and BDAChat's damage-aware heatmap.
- **How:**
  - Danger is a blurred mix of destroyed and damaged pixels; safety is the distance from damage.
  - v2 fixed three faults found by eye.
  - v3 kept hot-spot markers off the image edge.
- **How it ran:** v1 on Kaggle CPU; v2 and v3 on your PC from E1's predictions. [t1-survivability-heatmap-v3.ipynb](Champion_Task/t1-survivability-heatmap-v3.ipynb).
- **Expectation before the run:** on 20 images checked by eye, the circles sit on damage and the safe areas sit off the edge. v3 needed fewer circles on the edge than v2, with no fewer on damage.
- **Result:**

  | Version | Circles on damage (20 checked images) | On the edge |
  |---|---|---|
  | v1 | 36 of 48 | 52 of 57 safe spots |
  | v2 | 39 of 46 | 12 circles; no safe spot |
  | v3 | 41 of 46 | 1 circle (1% of all 645 images, against v2's 20%) |

- **What it led to:** T1 finished; T2 and T3 will use v3.

### L1 · Edge benchmark (ONNX, INT8)
- **What:** R9 exported to ONNX and quantised to INT8, measured on a CPU.
- **Why it was planned:** O4 needs a measured speed, size and accuracy table for the field segmenter. This followed Tamanna's Plan 1.
- **How:** PyTorch FP32, ONNX FP32, and dynamic and static INT8, on the same CPU and the 645 validation images.
- **How it ran:** Kaggle CPU (no GPU quota). [l1-edge-benchmark.ipynb](Champion_Task/l1-edge-benchmark.ipynb).
- **Expectation before the run:** INT8 is usable if Destroyed IoU drops by less than 0.01.
- **Result:**
  - ONNX FP32: 1.71× PyTorch, no accuracy loss.
  - Static INT8: 2.57× PyTorch, but Destroyed IoU dropped 0.0104, just over the limit.
  - Dynamic INT8: slower than PyTorch.
- **What it led to:** L1b.

### L1b · Static INT8, second try
- **What:** L1's static INT8 with a few convolutions kept in FP32.
- **Why it was planned:** L1's fastest version missed the accuracy limit by 0.0004.
- **How:** two versions. "Keep 1+1" keeps the first and last convolutions in FP32; "keep 1+3" also keeps the two full-resolution decoder convolutions.
- **How it ran:** Kaggle CPU, about 1.5 h. Its first run stopped because the mirror wasn't attached. [l1b-static-int8-retry.ipynb](Champion_Task/l1b-static-int8-retry.ipynb).
- **Expectation before the run:** the fastest version that drops Destroyed IoU by less than 0.01 becomes the edge model; otherwise ONNX FP32 does.
- **Result:** both versions passed (drop 0.0003). Keep 1+3 runs at 1.43× ONNX FP32, 1.6 s per image, 33 MB.
- **What it led to:** the edge segmenter part of O4 is met. It will be repeated on the final champion.

### E4 · R12 against R11, by source and by event
- **What:** both models scored per image source and per event, without training.
- **Why it was planned:** to test, rather than argue, why R12's extra data didn't help. Was it the wrong place (all xBD), or the wrong problem (rubble called background)?
- **How it ran:** Kaggle account 2, GPU T4 ×1, `HF_TOKEN`, 18 min. [e4-r12-vs-r11-by-source.ipynb](Champion_Task/e4-r12-vs-r11-by-source.ipynb), results in [E4-results/](Champion_Task/E4-results/E4/README.md).
- **Expectation before the run:**
  - If R12 beats R11 on xBD images by more than 0.013 (TTA, Destroyed IoU) but not on the others, the cause is the wrong place.
  - Otherwise it isn't.
- **Result:**
  - Both models reproduced their CSVs.
  - On xBD images Destroyed IoU went 0.456 → 0.463 (+0.007), within noise; on the others 0.071 → 0.069. So by the rule: **not the wrong place**.
  - Per event, the gain follows the extra data (rank correlation +0.73 over 13 xBD events). Palu got the most extra destroyed buildings (2,206) and gained +0.047.
- **Found alongside** (read-only checks on your PC, from R16's saved R11 masks, E1's true masks and R16's per-building table):
  - **Turkey's earthquake** holds 37% of the validation's destroyed pixels. R11 scores Destroyed IoU 0.000 there: it calls 91.5% of them intact and predicts destroyed on only 5 pixels.
    - Training has 557 Turkey images with 3.75 million destroyed pixels, so it isn't a lack of examples.
    - **Without Turkey, R11's Destroyed IoU is 0.410** (mIoU 0.618).
  - **The floods** (Harvey, Midwest, Florence) also score about 0: R11 calls their destroyed buildings damaged.
  - **R16's grader**, comparing before and after crops per building, finds 72–83% of Turkey's destroyed pixels (Destroyed IoU 0.13, or 0.19 corrected). It is held down by over-grading.
- **What it led to:** O1's gap is almost entirely Turkey, and the building-by-building comparison sees what the pixel model misses. R16c step 2 is now watched on Turkey, from its saved per-building table.

## 5. Phase B: structural training runs

### R11 · Before + after images (6 channels)
- **What:** the U-Net sees the before-image as well as the after-image.
- **Why it was planned:**
  - Phase2_Plan's main bet. All of xView2's top five used both images, and BRIGHT's before-image raised Destroyed IoU from 59.5 to 64.9.
  - Gate G2 put it first: rubble is called background, and the before-image shows where the building stood.
- **How:**
  - R5's recipe (the sampler, without copy-paste, which would paste rubble where the before-image shows nothing).
  - Both images get the same crop, flip and rotation and are stacked into 6 channels.
  - It checked first that the image pairs line up.
- **How it ran:** Kaggle main account, GPU T4, 6.6 h. [d4-champion-model-R11.ipynb](Champion_Task/d4-champion-model-R11.ipynb).
- **Expectation before the run:** the keep rule against R5 and R9.
- **Result:**
  - mIoU 0.581 (0.588 with TTA), the biggest gain so far: +0.046 over R5.
  - Destroyed IoU 0.291 (0.297), unchanged. So it didn't pass the Destroyed part of the rule, but it didn't drop either.
  - 6 of the 20 checked pairs are shifted by more than 5 px.
- **What it led to:** the new champion recipe and the base of R12, R16 and R11s7.

### R12 · Add the extra xBD tiles
- **What:** R11's recipe plus 1,619 extra xBD tiles holding 8,021 destroyed buildings (D5's 1,007 and D5b's 612).
- **Why it was planned:** O1's gate fails only on Destroyed IoU, and destroyed is the rarest class (about 0.3% of pixels). More destroyed examples was Phase2_Plan's second bet.
- **How:**
  - The tiles were added to training only, with un-classified pixels ignored in the loss.
  - The same 5,798 samples per epoch and 40 epochs were kept, so the change was more data, not more training.
- **How it ran:** Kaggle account 2, GPU T4, 6.4 h of training.
  - The first run stopped after 6 minutes on a bug in Cell 4b: it took the uploaded masks folder, which shares the photos' file names, for the photos. It was fixed in one line.
  - [d4-champion-model-R12.ipynb](Champion_Task/d4-champion-model-R12.ipynb).
- **Expectation before the run:**
  - No number was predicted. E1 had already pointed to "missed as background" more than to misgrading, so R12 was the second bet.
  - The keep rule against R11.
- **Result:** mIoU 0.582 (0.588 with TTA), Destroyed IoU 0.290 (0.295). Within noise of R11, so not kept.
- **What it led to:** R11 stays the champion, and E4 tests why.

### R11s7 · R11 with seed 7
- **What:** R11's recipe again, with seed 7 instead of 42.
- **Why it was planned:** O1's gate needs two seeds (or their mean) above the bar. Nothing beyond R11 passed the keep rule, so R11's second seed runs as R11s7 instead of R15 (your decision, 19 Oct).
- **How:** R11's notebook with two edits in Cell 2: `EXPERIMENT = "R11s7"` and a preset with `seed=7`.
- **How it ran:** Kaggle account 2, 6.8 h, after E4. [d4-champion-model-R11s7.ipynb](Champion_Task/d4-champion-model-R11s7.ipynb), results in [R11s7-results/](Champion_Task/R11s7-results/d4_R11s7/D4_R11s7.csv).
  - Checked: its config says seed 7 and 6 channels, its history has 40 continuous epochs, and the last epoch matches the CSV.
- **Expectation before the run:** no number was predicted. The rule was fixed: the gate needs both seeds, or their mean, above 0.30.
- **Result:**
  - mIoU 0.587 (0.591 with TTA), Destroyed IoU 0.287 (0.293).
  - Against R11 with TTA: mIoU +0.004, Destroyed IoU −0.004. The seed moves R11 far less than it moved R1 (0.013).
  - The two seeds' TTA mean is 0.5894 / 0.2947: the U-Net alone misses O1's 0.30 on two seeds.
- **What it led to:** O1's two-seed claim rests on R16c step 3, which grades SAM 3's buildings plus R11s7's pieces with step 2's graders.

## 6. Phase C: the find-then-grade model

### R16 · Find each building, then grade it
- **What:**
  - The finding: SAM 3 finds buildings in the before-image, plus R11's pieces SAM 3 missed.
  - The grading: a Siamese ResNet-18 grades each building from 96 px before and after crops.
- **Why it was planned:**
  - Phase2_Plan's fallback; xView2's 1st place and BRIGHT both find buildings first, then grade them.
  - One grade per building helps counting (O2) and DisasTeller (O3).
  - It is clean for the benchmark.
  - You decided it runs regardless of R12.
- **How:**
  - The grader was trained on 102,716 true building pieces, with the grades drawn equally often, for 10 epochs.
  - Three maps were scored: SAM 3 only, SAM 3 + R11's pieces, and a vote with R11.
- **How it ran:** Kaggle account 1, GPU T4, with E3's output and `HF_TOKEN`, 29 min. [r16-find-then-grade.ipynb](Champion_Task/r16-find-then-grade.ipynb) (named `r16-two-stage-grader` until 20 Oct, here and on Kaggle).
- **Expectation before the run:** the keep rule against R11 with TTA: Destroyed IoU up by more than 0.013, mIoU down by at most 0.005.
- **Result:**
  - Not kept. The best Destroyed IoU was 0.261 (the vote), against 0.297.
  - The grader over-graded intact buildings: 1,924 of 19,624 were graded destroyed, giving destroyed precision 0.35 and recall 0.89.
  - On the 85 non-xBD images it doubled R11's Destroyed IoU (0.128 against 0.071).
- **What it led to:** R16b.

### R16b · The grader corrected by the real grade shares
- **What:** R16's grades recomputed with a fixed rule, without retraining.
- **Why it was planned:** equal-share training made the grader act as if a third of all buildings were destroyed, when the real share is 8%.
- **How:**
  - Each grade's probability was multiplied by its real share among the training buildings (73.7%, 18.4%, 8.0%) and divided by a third.
  - It rebuilt R16's scores exactly first.
- **How it ran:** on your PC (CPU, 7.3 min). [r16b-grader-prior-correction.ipynb](Champion_Task/r16b-grader-prior-correction.ipynb).
- **Expectation before the run:** the same keep rule. The rule comes from training data only, so nothing is tuned on validation.
- **Result:**
  - Every map improved; the vote reached mIoU 0.585 and Destroyed IoU 0.274.
  - Intact buildings graded destroyed fell from 3,539 to 1,500.
  - Still below R11, so not kept.
- **What it led to:** your request for a rigorous R16 test, which became R16c.

### R16c step 1 · The ceiling, and a counting check
- **What:**
  - The best score R16's outlines could reach with a perfect grader.
  - How well each method answers the 1,055 validation counting questions.
- **Why it was planned:** to tell whether the outlines or the grader limit R16, and whether R16's counts would beat counting from R11's pixel map before any GPU time goes to the benchmark.
- **How:** from saved outputs only (R16, R16b, E3, E1). Every rebuilt building was checked against R16's per-building table, and the true masks had to answer 100% of the counting questions.
- **How it ran:** on your PC (CPU, 2.6 min). [r16c-step1-ceiling-and-counting.ipynb](Champion_Task/r16c-step1-ceiling-and-counting.ipynb).
- **Expectation before the run:** no number; it is a diagnosis.
- **Result:**
  - With a perfect grader, SAM 3 + R11's pieces reach Destroyed IoU 0.614 and mIoU 0.766. Our grader gets 0.208 and 0.541.
  - Counting:
    - R11's blobs: 39.2%;
    - R16's best map: 36.9%;
    - a perfect grader: 55.3%.
- **What it led to:** the grader is the bottleneck. R16c step 2, and the grader on SAM 3's outlines (now steps 4 and 5), were built.

### R16c step 2 · A stronger grader
- **What:** R16 with a better-trained grader, the same finding, and the same three maps.
- **Why it was planned:** step 1 showed the grader is the bottleneck (ceiling 0.614 against 0.208). E4 then found that R16's grader sees Turkey's collapses, which R11 misses.
- **How:**
  - 128 px crops with a 32 px margin, 20 epochs with the best kept.
  - The epoch and a bias per grade were picked on 580 held-out training images, never on validation.
  - Two seeds; R11 (seed 42) supplies the extra pieces.
- **How it ran:** Kaggle account 1, GPU T4 ×2 (one used), with E3's output and `HF_TOKEN`, 1.37 h. [r16c-step2-stronger-grader.ipynb](Champion_Task/r16c-step2-stronger-grader.ipynb), results in [R16c-results/step2/](Champion_Task/R16c-results/step2/README.md) (Kaggle's output folder: `R16c-truth`).
- **Expectation before the run:** the keep rule against R11 with TTA on both seeds (Destroyed IoU up by more than 0.013, mIoU down by at most 0.005). The grader on SAM 3's outlines only if no map is kept on both seeds.
- **Result:**
  - **SAM 3 + R11's pieces was kept on both seeds, and the O1 gate passed on both:**
    - seed 42: mIoU 0.587, Destroyed IoU 0.316;
    - seed 43: mIoU 0.585, Destroyed IoU 0.311;
    - R11: 0.588 and 0.297.
  - The grader on the true buildings scored macro F1 0.778 and 0.784, against R16's 0.656 and BDAChat's 0.763.
  - It closed 25–27% of the gap to the ceiling.
  - Read-only checks on your PC, from its per-building table:
    - Turkey: 0.000 → 0.24–0.27;
    - xBD images: 0.456 → 0.35–0.38;
    - counting: 37–39% on the validation questions, about R11's level.
- **What it led to:** the gate passes on one U-Net seed and two grader seeds. Proposed next, and both decided 20 Oct:
  - step 3: the same graders with R11s7's pieces, for two U-Net seeds;
  - steps 4 and 5, a stretch: a grader trained on SAM 3's outlines, which may recover xBD.

  Counting from it on the benchmark (H1 V3) waits.

R16c's steps were renumbered 1–5 on 20 Oct (your decision): step 3 was first called 2b, step 4 was 3a, and step 5 was 3b.

### R16c step 3 · Step 2's graders with R11s7's pieces (first called 2b)
- **What:** step 2's two graders and their biases, loaded as they were kept, with R11s7's pieces in the finding. Evaluation only. [r16c-step3-r11s7-pieces.ipynb](Champion_Task/r16c-step3-r11s7-pieces.ipynb).
- **Why:** step 2's gate pass rests on one U-Net seed (R11); O1 asks for two.
- **How it ran (21 Oct):**
  - Kaggle account 2. Validation took 26.7 min.
  - E3's and step 2's outputs were attached by sharing account 1's notebooks with account 2.
  - Results in [R16c-results/step3/](Champion_Task/R16c-results/step3/README.md).
- **Expectation, fixed before the run:**
  - SAM 3 + R11s7's pieces passes the gate and the keep rule on both grader seeds, within about 0.01 of step 2.
  - Sanity checks: the graders reproduce step 2's macro F1 on the true buildings, and R11s7 reproduces its D4 CSV.
- **Result:**
  - **Sanity checks:** both pass. R11s7 reproduces its CSV, and the graders give step 2's macro F1 (0.7779 and 0.7843).
  - **Gate (mIoU > 0.52, Destroyed IoU > 0.30): passes on both grader seeds:**
    - seed 42: 0.5843 / 0.3090;
    - seed 43: 0.5830 / 0.3006, only 0.0006 above the bar.
  - **Keep rule against R11s7 with TTA (0.5913 / 0.2928): not kept on either seed:**
    - mIoU falls by 0.0070 and 0.0083, more than the 0.005 allowed;
    - Destroyed IoU rises by 0.016 and 0.008.
  - **Against step 2 (R11's pieces):** mIoU −0.0026 and −0.0022, Destroyed IoU −0.0069 and −0.0100. That's within about 0.01, as expected.
  - **Against the expectation:** the gate and the closeness to step 2 came out as expected; the keep rule didn't.
- **What it led to:**
  - **O1's gate is met on two U-Net seeds**, with every combination of U-Net seed and grader seed above 0.30. The mean of the four is 0.585 / 0.309.
  - **The honest caveats:**
    - the weakest combination clears 0.30 by only 0.0006;
    - against the stronger U-Net seed, the system trades about 0.007 mIoU for its Destroyed gain, so our stricter keep rule (against the U-Net) holds on one U-Net seed only.

### R16c steps 4 and 5 (built, ready to run on account 1; first called 3a and 3b)
- **What:** step 4 runs SAM 3 on the 5,798 training images (about 3.5–4 h). Step 5 trains the grader on SAM 3's own outlines, with a "not a building" label (about 1.5–2 h).
- **Why:** by step 2's rule they aren't needed, since step 2 was kept. But training on the outlines it grades may recover the xBD loss and cut Turkey's false alarms. You chose to run them as a stretch (20 Oct).
- **How it runs:**
  - Step 4: [r16c-step4-sam3-training-images.ipynb](Champion_Task/r16c-step4-sam3-training-images.ipynb).
  - Step 5: [r16c-step5-grader-on-sam3-outlines.ipynb](Champion_Task/r16c-step5-grader-on-sam3-outlines.ipynb), step 2's notebook with `TRAIN_OUTLINES = "sam3"` set, and step 4's output attached.
  - Kaggle account 1, about 5–6 h in all.
- **Expectation, fixed before the run:** step 5 replaces step 2 only if it beats step 2's kept map on both seeds by the keep margin (Destroyed IoU up by more than 0.013, mIoU down by at most 0.005). It is also checked for whether xBD recovers without losing Turkey.
- **Step 4 result (20 Oct, account 1, 4.6 h):** complete, 5,798 of 5,798 maps.
  - Building IoU 0.629 and destroyed footprint recall 0.693 on the training images, against 0.630 and 0.634 on validation (E3).
  - So step 5's grader trains on outlines as good as the ones it grades.
  - Results in [R16c-results/step4/](Champion_Task/R16c-results/step4/README.md). Step 5 is running.

## 7. The 7B and distillation

### L0 step 1 · Our 7B at the paper's image size
- **What:** disaster type and relational reasoning re-scored at three image sizes.
- **Why it was planned:**
  - Like for like (optical questions, as the paper's Table 2), our 7B trails the paper's own fine-tune by 4.3 points, mostly on these two tasks.
  - Our training recipe matches the paper's Appendix B.3, but we tested at 512 px and the paper at full size.
  - You decided to improve the 7B before distilling.
- **How:** the 1,211 optical questions at 504 px (H1 V2's size, as the control), at 616 px (our training cap) and at 1036 px (full size). Same model, prompts and greedy scorer as H1 V2.
- **How it ran:** Kaggle account 1, GPU T4 ×2, 1.7 h. [l0-step1-image-size-check.ipynb](Champion_Task/l0-step1-image-size-check.ipynb).
- **Expectation before the run:**
  - Gap closed: within 3 points of the paper on both tasks.
  - Partly closed: half of each gap recovered.
  - Otherwise: step 2.
- **Result:**
  - The control reproduced H1 V2 exactly.
  - Full size gave disaster type 71.1 → 75.6 and relational 23.3 → 31.1. That recovers 36% and 61% of the gaps: not closed.
- **What it led to:**
  - Full size became the 7B's test setting.
  - You chose step 2: a short extra round of training of our adapter at full size, judged on every optical question for both models.

### L0 step 2 (built 20 Oct; the old 7B's re-score done, the training's smoke test running)
- **The old 7B's re-score:** [l0-step2-full-size-rescore.ipynb](Champion_Task/l0-step2-full-size-rescore.ipynb), a copy of L0 step 1's notebook.
  - It covers the other four tasks at full size: 9,464 optical questions, about 11 h.
  - It runs in two parts: account 3 now, account 2 after R11s7.
  - It was dry-run on your PC: the question counts match H1 V2, nothing is asked twice when it continues, and the scoring was checked by hand.
  - The same notebook later scores the new 7B (`MODEL = "new"`).
- **The top-up training:** [l0-step2-topup-training.ipynb](Champion_Task/l0-step2-topup-training.ipynb), the original recipe on today's Kaggle libraries.
  - Data: DTR 1,627 + ORR 1,882, plus 3,509 replayed from the other tasks (7,018 examples), at full size.
  - Global batch 256, about 28 steps.
  - **A technical change from the plan:** the old adapter's saved layer pattern (`model.layers.N`, transformers 4.49) matches nothing in transformers 5 (`model.language_model.layers.N`). So a new rank-64 adapter is trained on the merged 7B, which starts from exactly the same model, and is merged into it at the end.
  - Two memory savings make full-size images fit a T4:
    - the output layer runs only over the answer;
    - the embeddings stay fp16.
  - Its logic was checked on your PC on the real manifest: the replay mix, the image sizing, and that the trimmed loss equals the full-sequence loss.
- **Expectation, fixed before the runs:**
  - Re-score: no decision hangs on it. The tasks move by a few points at most from 512 px; counting has the most to gain.
  - Training: DTR and ORR rise by about 1–4 points, and the other tasks stay within about a point.
  - Keep rule (19 Oct): DTR and ORR rise, and no other task drops by more than 1 point.
- **The re-score's result (20 Oct):**
  - **How it ran:** part A (BDC, BBR) on account 3, 8.6 h; part B (DRE, DSR) on account 2, 4.1 h. Run notebooks [l0-step2-full-size-rescore-old-partA.ipynb](Champion_Task/l0-step2-full-size-rescore-old-partA.ipynb) and [-old-partB.ipynb](Champion_Task/l0-step2-full-size-rescore-old-partB.ipynb); results in [L0-step2-results/old-partA/](Champion_Task/L0-step2-results/old-partA/L0S2/README.md) and [old-partB/](Champion_Task/L0-step2-results/old-partB/L0S2/README.md).
  - **The old 7B at full size, against 512 px:**
    - DTR 75.64 (71.10) and ORR 31.12 (23.31), both from L0 step 1;
    - BBR 20.94 (19.54), BDC 30.09 (32.01), DRE 32.39 (32.87), DSR 40.06 (37.87).
  - **Checked:**
    - every accuracy recounted from the per-question answers;
    - the 8 questions whose images can't be read are scored wrong, the same 8 as at 512 px.
  - **Against the expectation:** the four tasks moved by a few points at most, as expected (−1.9 to +2.2). Counting fell instead of gaining, so the 7B alone still misses the counting half of KR2.
  - **What it led to:** the full-size baseline for the keep rule is complete. Next is the training's smoke test.
- **The training's smoke test (20 Oct, account 3, about 45 min):**
  - 16 examples, including the longest (3,092 tokens), in 2 steps on both T4s.
  - Losses 0.844 → 0.461, within the expected 0.4–1.2.
  - Peak memory 14.7 GB of 15.8 GB.
  - **Against the expectation (7–14 h):** 10.0 s per example projects 19.5 h for the full 7,018 examples. Your choice: the equal replay in two sessions plus a short third, or half the replay (about 15 h).
- **The full training's first two attempts (20 Oct, account 3): both out of GPU memory, both fixed without changing the data or the batch.**
  1. **Committed about 13:12; crashed at the first step, after 11 min.**
     - The trainer collects all 128 micro-batches of a step before training on any, and the data loader puts each one on the GPU: 3–6 GB of waiting full-size images per GPU. The smoke test collected only 4.
     - **Fix:** a `StepOnCpu` trainer keeps the waiting batches on the CPU, with images in fp16, the dtype the model casts them to.
  2. **Version #7, committed about 17:50.**
     - Step 1 finished (40.8 min, loss 0.564).
     - Step 2's backward pass then ran out of memory, about 40 MiB short (it needed 926 MiB with 887 MiB free).
     - **Fix:** 8-bit paged AdamW (`paged_adamw_8bit`, QLoRA's optimizer). It frees about 1 GB per GPU and moves optimizer memory to the CPU when the GPU runs short. This is a small change from the original's 32-bit AdamW.
  - **If that still isn't enough, the next options, from smallest change to largest:**
    1. leave out the few longest examples (two images and the most text), after counting how many there are;
    2. use a slightly smaller image size for two-image examples only;
    3. halve the LoRA rank from 64 to 32, which frees about 0.6 GB more but changes the adapter.

    Each changes the recipe, so each is your decision, and its effect is measured before asking. A failure would most likely show within the first few steps, so a check costs about 1–2 GPU hours, not 10.
- **Session 1 of the full training (account 3, finished 21 Oct):**
  - 15 of 28 steps, about 42 min each (2,535 s on average), then a clean stop at the time limit.
  - No out-of-memory error: peak 14.2 GB on GPU 0 and 14.1 GB on GPU 1.
  - Checkpoints 4, 8, 12 and 15 pushed to `AbrarAlam/disasterm3-qwen25vl7b-qlora/l0s2/checkpoints`.
  - Loss 0.563 at step 1 and 0.557 at step 15 (range 0.522–0.577). It's nearly flat, because our 7B already knows these tasks. The keep rule's evaluation decides whether it helped.
  - Results in [L0-step2-results/train-session1/](Champion_Task/L0-step2-results/train-session1/L0S2-train/README.md).
  - **Left:** 13 steps, about 8.8 h of training, plus the merge.
- **Session 2 (account 3, 21 Oct, about 1 h): crashed at step 16, nothing saved.**
  - It restored checkpoint-15 and ran step 16's 128 micro-batches. Then the optimizer update failed on both GPUs: `Error invalid argument at line 580 in file /src/csrc/pythonInterface.cpp`, bitsandbytes' paged-memory prefetch.
  - **Cause:**
    - the paged optimizer flags its state tensors `is_paged`, and `torch.save` keeps the flag;
    - on resume, the trainer loads the state into ordinary GPU memory, and bitsandbytes' reload keeps the flag;
    - the step then prefetches that memory as paged, which CUDA rejects.
    - Checked on your PC with torch's save, load and deepcopy. Session 1 started fresh, so it never reloaded the state.
  - **Fix:** `adamw_bnb_8bit`, the same 8-bit AdamW without paging. The maths and the optimizer's memory are unchanged, and it loads checkpoint-15's state as is. Session 2 itself held that state in ordinary GPU memory through a full step without running out. `time_limit_h` is 5.0 for account 3's remaining 7.2 h.
  - **Expectation:** steps 16–23 in about 6 h, then session 3 does steps 24–28 and the merge.

### L2 step 1 · Teacher answers for the 3B
- **What:** the 7B answers 4,000 training questions, and only answers checked correct against their labels are kept.
- **Why it was planned:** O4's distilled 3B learns from the 7B's answers.
- **How:** a copy of the original teacher notebook, changed in three ways:
  - Kaggle's own torch, because the old pins no longer install;
  - fp16 on two T4s in batches of 8;
  - strict per-task answer checks.
- **How it ran:** Kaggle account 1, GPU T4 ×2, about 2 h. [l2-step1-teacher-rationales.ipynb](Champion_Task/l2-step1-teacher-rationales.ipynb) (named `l2-stage1-teacher-rationales` until 20 Oct, here and on Kaggle); results in `L2-step1-results/`.
- **Expectation before the run:** answers that fail the check fall back to their labels, so the 3B never learns the teacher's mistakes.
- **Result:**
  - 1,085 of 4,000 (27%) were kept: disaster type 89.4%, land use 50.9%, bearing bodies 42.7%, building counting 15.6%, relational 13.3%, road counting 0%.
  - The push to the Kaggle dataset failed silently.
- **What it led to:** step 2 waits for L0; if L0 changes the 7B, this step reruns with the new teacher. Explanations became L2b.

## 8. Housekeeping

### Commit and push of the Phase A-onward work
- **What:** `Essential_task_delegation/`, `others/`, `.gitignore`, `README.md` and `UPDATE.md` committed and pushed for the team.
- **Why:** so collaborators can pull the current work and continue it.
- **How:** the `.gitignore` keeps out:
  - bulk masks that no notebook reads;
  - the 71 MB L2 training file;
  - personal samples and the next report's draft;
  - the local settings.

  You ran the commands.
- **Result:** pushed as `1b33574`.
