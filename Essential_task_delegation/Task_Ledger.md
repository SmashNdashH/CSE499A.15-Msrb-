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
| 26 Oct | Build and run R16c step 5b (a third grader seed, 44) before claiming O1 on clean data; build a merge-only copy of L0's training notebook | On clean data only step 5's SAM 3 map passes O1's gate, and seed 43 clears it by 0.005. L0's merge failed on torchao, while its training is complete | [r16c-step5b-third-grader-seed.ipynb](Champion_Task/r16c-step5b-third-grader-seed.ipynb) and [l0-step2-merge.ipynb](Champion_Task/l0-step2-merge.ipynb), built and dry-run checked 26 Oct |
| 26 Oct | E5: settle the leak on DisasterM3's own images and check the benchmark against the training images, with a near-copy check; save per-image pixel counts for R11, R11s7, R9 and R1 | The PC check (21 pairs) and the audit (40) disagree. A copy in the benchmark would raise O2's numbers. Per-image counts let any leak list be applied without another GPU run | Step 1 (CPU) and step 2 (GPU) for account 2 ([E5](#e5--trainvalidation-leak-and-the-benchmarks-copies-opened-26-oct)) |
| 26 Oct | Teammates' follow-ups, sent as prompts: Ridita wires the offline search into DisasTeller with a retrieval test; Aryan re-scores the sentence answers and adds R11's counts; Tamanna cleans up Plan 2, re-runs it, then tests crops against ground truth | The review of their pushes ([§9](#9-teammates-work-what-we-use-reviewed-26-oct)) | Prompts given 26 Oct |
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
| 20 Oct | You make the renames inside notebooks that already ran, by hand | Editing a code cell programmatically wipes its recorded outputs | [Manual_Notebook_Renames.md](Manual_Notebook_Renames.md) |
| 20 Oct | L0 step 2's training keeps the equal replay (option A): 7,018 examples, about 19.5 h in 2–3 sessions, rather than half the replay (about 15 h) | The replay protects the other four tasks, and the keep rule fails if any of them drops by more than 1 point | L0 step 2: session 1 on account 3 |
| 22 Oct | H2 builds its table from R11 with 8-view TTA, not R9 | R11's blobs of at least 20 px are our best counter on validation (39.24% against R9's 33.3%), and R11's counting on the benchmark had never been measured | `Champion_Task/h2-sql-counting-agent.ipynb` (Noman's original unchanged) |
| 22 Oct | Set Soup (trysoup.dev, a fine-tuning tool) aside for now, and record that it's barely usable for us | Its docs (v0.76.0) don't list Qwen2.5-VL, and don't cover the vision encoder freeze, the image-size cap, two-GPU training or Kaggle resumes that our recipe needs. Its only possible uses here are text-only (a local server for T2; GGUF export or a refusal fine-tune of a text model) | [Paper_Audit_and_Integration_Plan.md §4.14](Paper_Audit_and_Integration_Plan.md); our training notebook stays the recipe for every Qwen2.5-VL fine-tune |
| 24 Oct | T3 becomes two pieces: a public showcase site, and an interactive, animated demo website deployed on Vercel, with a live offline mode at the booth. You build it, and choose its name and look later | Last season's winner showed a designed site, results in motion and one interactive proof of its claim. Vercel can't run our models, so the live part runs on the booth laptop | Roadmap T3; [System_Design_Notes.md §4](System_Design_Notes.md) |
| 24 Oct | Finish T2, T3, L0 and L2 before any work on Layers 2–4 of the day-1 vision, and add a "3D damage city" view to the T3 website (building outlines raised into 3D blocks coloured by damage, labelled as a view of our 2D results) | O3 and O4 need T2, T3, L0 and L2, and Layers 2–4 aren't in O1–O4. The 3D view takes about a week with no GPU | Roadmap T3; [System_Design_Notes.md §5](System_Design_Notes.md) |
| 26 Oct | Record TimeSpot (ICML 2026) and use its evaluation methods, citing it wherever they're used: calibration error and risk–coverage curves for L3's handoff and H2c's gate, and a consistency check for T2's reports. Its dataset isn't used | Its methods measure whether a model's confidence can be trusted and catch answers that contradict facts, both needed before the system acts on a model's answer | [Paper_Audit §4.15](Paper_Audit_and_Integration_Plan.md); roadmap L3, H2c and T2 |
| 24 Oct | Add H2c, a second try at free-form questions: question–SQL pairs for our table, a fine-tuned adapter, a five-sample agreement gate, and a new test with unseen question types. If it fails, fall back to the system as it was | In the literature, models passed a similar bar only with in-domain training and an abstention gate, which H2b didn't have | Roadmap H2c; [System_Design_Notes.md §3.5](System_Design_Notes.md) |
| 24 Oct | Try running the INT8 segmenter in the visitor's browser (ONNX Runtime Web) | Uploads would work on the public site with no server, and images never leave the visitor's device | A feasibility test first, with its pass rule fixed in advance ([System_Design_Notes.md §4.3](System_Design_Notes.md)) |
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
- **Checked on clean data, 27 Oct:** SAM 3 never trained on DisasterM3, so the leak can't flatter it. Without the 40 leaked pairs it scores 0.6302 against R9's 0.5905, so Gate G3 still passes.

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

### H2 · Per-building table and SQL counting agent
- **What:** a per-building SQLite table of the benchmark images, so counts and other questions become database queries, plus a small-LLM SQL agent over it.
- **Why it was planned:**
  - O2 and O3: exact, queryable counts for DisasTeller (T2) and the demo (T3).
  - It was also the first chance to measure R11's counting on the benchmark itself.
- **How:**
  - Noman built the notebook. Its run copy switched from R9 to R11 with 8-view TTA (after- and before-images) and a 20 px minimum, the best counter on validation (39.24%).
  - Two LLMs, Qwen2.5-Coder-3B and our 7B (text only), wrote SQL for 100 sampled optical counting questions.
- **How it ran:** Kaggle account 2, T4 ×2, 22 Oct, about 1.5 h with the smoke test. [h2-sql-counting-agent.ipynb](Champion_Task/h2-sql-counting-agent.ipynb). H1 V1's per-question file was uploaded as an input, because the `h1-v1-facts` dataset lacked it.
- **Expectation before the run:** R11's counting near 30% on the optical questions (validation numbers run high), with the base 7B's 34.2% as the bar.
- **Result:**
  - **R11's counting:** 32.96% on the optical questions and 27.46% over all 4,978 questions in the table. That beats R9's tuned counter (26.51%) and our 7B (30.09% at full size), but is still 1.2 points under 34.2%.
  - **The agent:** both models wrote valid SQL for all 100 questions; their counts matched the table on 99 and 100, so both scored the table's 34%.
  - **The demo:** a free-form distance question got a query with no image filter from the 3B (answer 33,880).
  - **The table:** 117,687 buildings in 2,652 images; 3 images couldn't be segmented.
  - Results in [H2-results/H2/](Champion_Task/H2-results/H2/README.md).
- **What it led to:**
  - R11's blobs are now our best benchmark counter, but the counting half of KR2 is still not met.
  - The open-question failure confirmed the need for H2b's graded test.
  - Its table is H2b's input.

### H2b · Graded test of the SQL agent
- **What:** a graded test of which model, if any, should turn free-form questions into SQL over H2's per-building table in the deployed system.
- **Why it was planned:**
  - O3 and O4: H2's 100 counting questions all need the same query, so they couldn't separate the models, and counting needs no model at all.
  - A second model costs memory on low-resource hardware, so a separate SQL model had to earn its place.
- **How:**
  - 297 items on 90 images of R11's table: 29 answerable questions in 12 groups, each in 3 phrasings on up to 3 images (261 items), plus 12 unanswerable questions in 3 phrasings (36 items). The questions, the prompt and the rule were fixed before any run.
  - An answer counts only if its query matches the reference on 5 versions of the table (test-suite accuracy, Zhong et al. 2020).
  - RS(10) scores reliability, in points per 100 items: +1 for a right answer or a correct refusal, 0 for refusing an answerable question, −10 for a wrong answer (TrustSQL; EHRSQL 2024).
  - 95% intervals come from a cluster bootstrap over questions.
  - 4 models, each in 4-bit and fp16: our 7B and the base VL-3B (text only, no extra memory), and Qwen2.5-Coder 1.5B and 3B (separate models). One prompt, greedy decoding, up to 2 retries with the error message.
- **How it ran:** Noman's Kaggle account, T4 ×2, 23 Oct, 47 min, after two smoke tests on account 2 on 22 Oct. [h2b-sql-agent-graded-test.ipynb](Champion_Task/h2b-sql-agent-graded-test.ipynb); the run's code matches the built notebook cell by cell.
- **Expectation before the run:**
  - The rule: a configuration passes with test-suite accuracy of at least 90% and RS(10) above 0.
  - Our VLM writes the SQL if it passes in 4-bit, a separate model is only a fallback, and if nothing passes, fixed queries only.
  - No accuracy number was predicted. The smoke test's 6 items had shown both small models answering unanswerable questions and missing the NULL case.
- **Result:**
  - **No configuration passes.** The best accuracy was Coder-3B in fp16's 73.9%, and every RS(10) was negative.
  - **Our 7B in 4-bit:** 69.0% [57.1, 80.5], RS(10) −89.6. It refused 94% of the unanswerable questions, the most of any model, and adds no memory (5.8 GB, 3.3 s per question on a T4). It's level with Coder-3B in 4-bit (72.0%; the difference, −3.0 points [−10.7, 3.8], isn't clear).
  - **The base VL-3B in 4-bit:** 46.4%, RS(10) −365.7, refusing 36%; 22.6 points below our 7B [9.2, 36.8].
  - **The separate coders in 4-bit:** Coder-1.5B 56.7% (RS(10) −283.8, +1.1 GB) and Coder-3B 72.0% (RS(10) −162.6, +2.0 GB).
  - **Weak for every model:** pairs of buildings (6–28%), the "no destroyed building nearby" NULL case (0–22%), positions (0–52%) and distances (44–70%). Our 7B in 4-bit gets counts, percentages, rankings and size thresholds 89–100% right.
  - **Wording:** with responder words, our 7B in 4-bit drops to 52.9%, from 74.7% with plain words.
  - **The five table versions mattered:** on the real table alone, every model scores 4–6 points higher.
  - **Retries never fixed a query:** all 200 retried items ended refused or failed again.
  - **4-bit** costs our 7B 4.6 points [0.8, 8.8] and the VL-3B 9.9 [3.4, 17.6]; the coders lose nothing measurable.
  - Results in [H2b-results/H2b/](Champion_Task/H2b-results/H2b/README.md).
- **What it led to:**
  - By the rule fixed 22 Oct, the system answers through fixed queries only, with no free-form questions. Benchmark counting already used the fixed query, so O2's numbers don't change.
  - DisasTeller (T2) takes its facts from fixed queries, as planned.
  - The re-run on the distilled 3B is needed only if free-form questions come back. The follow-ups (a menu of fixed question templates, or question–SQL pairs in L2's training) wait on your call, and either needs a new question bank for its test.
  - The full reasoning, including why LangChain's SQL agent wouldn't change the result: [System_Design_Notes.md §3](System_Design_Notes.md).

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

### E5 · Train/validation leak and the benchmark's copies (opened 26 Oct)
- **What:** check whether validation images are copies of training images, and whether benchmark test images are copies of DisasterM3 Instruct images.
- **Why it was planned:**
  - Ridita's audit (pushed 25 Oct) found 40 of the 645 validation pairs are near-copies of training pairs: xBD's `socal_fire` and `woolsey_fire` share before-images.
  - Every O1 score (R1–R16c) is measured on those 645 images.
  - The 7B and R11 were trained on Instruct images, so benchmark copies would raise O2's numbers too.
- **How, so far:**
  - A check on your PC hashed xBD's original before-images and found 21 leaked pairs (80,709 destroyed pixels).
  - R16c was re-scored from its per-building file without them.
  - Details: [E5-results/E5/](Champion_Task/E5-results/E5/README.md).
- **Result so far:**
  - R16c's four combinations score 0.3006–0.3159 on all 645 images, 0.2915–0.3070 without the 21, and 0.2826–0.2976 without all 86 fire images. O1 is at risk.
  - From E1's per-image counts (26 Oct), R9 with TTA drops from 0.3001 to 0.2937 without the 21 pairs.
  - E1's Gate G2 still sends R11 first: 58.7% of missed destroyed pixels are called background without the 21 pairs, and 56.8% without all fire images, against the 40% rule. Table in [E5-results/E5/](Champion_Task/E5-results/E5/README.md).
- **What it led to:**
  - **Go given 26 Oct for two notebooks on account 2:**
    - step 1 (CPU) hashes DisasterM3's own images to settle 21 against 40, and checks the benchmark against Instruct, with a near-copy check (dHash plus thumbnail correlation) and pictures for review;
    - step 2 (GPU, about 1 h) saves per-image pixel counts for R11, R11s7, R9 and R1, so any leak list can be removed on your PC.
  - **Both notebooks built 26 Oct and dry-run checked on your PC,** with the real manifest and masks and stand-in images: [e5-step1-hash-check.ipynb](Champion_Task/e5-step1-hash-check.ipynb) and [e5-step2-per-image-counts.ipynb](Champion_Task/e5-step2-per-image-counts.ipynb).
    - The split check passes: 6,443 pairs, 645 validation pairs (560 from xBD), 1,808,111 destroyed pixels.
    - Planted copies are found exactly: the known twin, an identical after-image, a recompressed before-image (listed as a near-copy, not counted), and benchmark copies on the right questions.
    - Step 2 ran with a stand-in model. On Kaggle, its own check is that every model reproduces its D4 CSV.
  - Ridita's prompt asks for `val_train_leaks.csv` and the audit's scripts: [Ridita_Task/E5_leak_audit_prompt.md](Ridita_Task/E5_leak_audit_prompt.md).
  - **A rule from now on:** any validation or benchmark score is checked for copies by image content, not by name. D5's check compared names only and missed this leak.
  - **Decided 26 Oct: every score a leak touched is re-scored, with no 1% threshold.** The first plan re-scored the 7B and H2 only if over 1% of the benchmark's optical questions were copies. Dropped, because the re-score is a filter over saved per-question and per-image files (minutes on the PC), and a reviewer can ask about any affected number. Each score is reported twice: all, and clean. The full list is in the roadmap's E5 step 3. R2–R8 and R10 aren't re-scored (no checkpoints or per-image counts). Without the 40 pairs, every U-Net measured drops by a similar 0.012–0.013, so the gaps between U-Net runs move by about 0.001, under the 0.005 noise margin, and their decisions stand. *(Narrowed later on 26 Oct, after step 2: this holds for the after-only U-Nets R1–R10, which these runs are all compared with. The before + after models R11 and R11s7 drop 0.027.)*
- **Result, 26 Oct: the full list of 40, and the clean scores.**
  - Ridita's audit README settles 21 against 40: her 40 include our 21. We missed 19 because their socal_fire side isn't an xBD tier-1 image. Her section 4 lists the 19; with ours they make [leaked_pairs_40.csv](Champion_Task/E5-results/E5/leaked_pairs_40.csv).
  - The list checks out: 155,812 destroyed pixels, as she reports, and E1's per-image ground truth matches each pair. Her `val_train_leaks.csv`, sent later the same day, matches it exactly (pairs, twins and pixels).
  - **Without the 40 (605 clean pairs):** R16c's four combinations score 0.2838–0.2980 (mean 0.2917), all below 0.30. **O1 isn't met on clean data;** the best combination is 0.002 short. R9 with TTA is 0.2874.
  - E1's Gate G2 still holds (55.7–58.4% of missed destroyed pixels called background).
  - Table in [E5-results/E5/](Champion_Task/E5-results/E5/README.md).
  - **Still open:**
    - R11 and R11s7 alone (step 2);
    - R16c step 5;
    - the benchmark check (step 1). Ridita couldn't run it, because the benchmark images aren't on her PC.
    - Step 1's validation part now serves as a third, independent check of the 40.
- **Result, 26 Oct: steps 1 and 2 done (account 2), and every affected score re-scored on your PC.**
  - **How it ran:** step 1 on CPU, fingerprinting in 15.4 min; step 2 on GPU, 31 min for the four models.
  - **Checks:**
    - Both rebuilt the split as in every D4 run.
    - All 8 model rows reproduced their D4 CSVs.
    - Step 1 found the same 40 pairs: three independent checks now agree.
    - Its 3 validation "review" matches are different places.
    - Every "all" score below reproduces its published number (one exception, explained in the E5 README).
  - **Against the expectations fixed before the run:**
    - 21–40 leaked pairs: 40, yes.
    - All of the PC's 21 found: yes.
    - Benchmark: no prediction was made. 186 of 10,675 optical questions (1.74%) use a copied after-image.
  - **Segmentation, clean (605 pairs), Destroyed IoU:**
    - R16c step 5's SAM 3-only map: **0.3161 and 0.3051, the only map that passes O1's gate on clean data**, with mIoU 0.547.
    - Steps 2 and 3: 0.2838–0.2980, so they fail.
    - R11 and R11s7 with TTA: 0.2692 and 0.2652.
    - R9 with TTA: 0.2874.
  - **The leak helped the before + after models twice as much** (−0.027, against −0.012 to −0.013 for the after-only U-Nets). Each leaked pair shares its before-image with a training pair. So on clean data, R11's Destroyed IoU is 0.018 below R9's. R11's mIoU gain (+0.04) and its lower share of misses called background (28% against 58%) still hold.
  - **Benchmark, clean:**
    - O2 counting (R11's TTA counts, optical): 32.96% → 32.47%. The gap to 34.2% grows from 1.24 to 1.73 points.
    - The old 7B's six tasks at full size drop by at most 0.7 points (mean 38.37 → 38.14).
    - Everything else drops by 0.0–0.7 points, and no decision changes.
  - Full tables, review notes, and how it was scored: [E5-results/E5/](Champion_Task/E5-results/E5/README.md). Step outputs: [step1/](Champion_Task/E5-results/step1/README.md) and [step2/](Champion_Task/E5-results/step2/README.md), moved there from `E5-step1-results/` and `E5-step2-results/`.
  - **What it leads to:**
    - O1 rests on step 5's SAM 3 map on clean data. A third grader seed is proposed to confirm seed 43's narrow pass.
    - O2's counting still needs about 2 points.
    - Every future split is grouped by before-image fingerprint.
    - The new 7B's scores will be reported both ways.

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
- **Checked by content, 27 Oct:** two of the 1,619 extra tiles share a before-image with a validation pair outside E5's 40: `socal-fire_00000876` with `woolsey_fire_00000669`, and `woolsey-fire_00000262` with `socal_fire_00000202` (6,860 destroyed pixels in all). So R12 trained on 42 leaked pairs. A leak can only flatter R12, which still didn't beat R11, so the verdict stands. Details in [E5-results/E5/](Champion_Task/E5-results/E5/README.md).

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
  - **Caveat added 26 Oct (E5):** these scores are on all 645 validation images, some of which are near-copies of training images. Without the 21 leaked pairs found so far, the four combinations of steps 2 and 3 score 0.2915–0.3070. The clean score decides O1.
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
  - **At risk since 26 Oct (E5):** measured on all 645 validation images, which include leaked pairs. Without the 21 found so far, the mean of the four is 0.3000, and one combination is at 0.2915.
  - **The honest caveats:**
    - the weakest combination clears 0.30 by only 0.0006;
    - against the stronger U-Net seed, the system trades about 0.007 mIoU for its Destroyed gain, so our stricter keep rule (against the U-Net) holds on one U-Net seed only.

### R16c steps 4 and 5 (step 4 done 20 Oct, step 5 done 26 Oct; first called 3a and 3b)
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
- **Step 5 result (account 1, finished 26 Oct; graders 60 min, validation 36 min): it doesn't replace step 2. But its SAM 3-only map is the one map that passes O1's gate on clean data.**
  - **Against the rule fixed before the run:** its SAM 3 + R11 map scores 0.5496 / 0.2434 (grader 42) and 0.5410 / 0.2158 (grader 43), against step 2's 0.5869 / 0.3159 and 0.5852 / 0.3106. That's 0.07–0.09 lower in Destroyed IoU, so it doesn't replace step 2.
  - **The SAM 3-only map** (SAM 3's outlines graded, with no R11 pieces added):
    - All 645 pairs: 0.5520 / 0.3347 and 0.5522 / 0.3242, so it passes the gate on both seeds.
    - Clean (E5's 605 pairs): 0.5469 / 0.3161 and 0.547 / 0.3051.
    - It isn't kept against R11 with TTA, because its mIoU is 0.035 lower. The keep rule decides the champion; O1's gate is a separate test.
  - **Why adding R11's pieces hurts here:** this grader learned on SAM 3's outlines and over-calls R11's pieces as destroyed. Destroyed precision falls from 0.63 to 0.32. Step 2's grader, trained on true outlines, handled them better.
  - **On the true validation buildings:** macro F1 0.737 and 0.730, against step 2's 0.778 and 0.784. SAM 3's rougher outlines make a weaker grader overall.
  - **The Destroyed bias sits at the search grid's lowest value (−4.0) in 38 of the 40 epoch searches.** The search wanted a stronger penalty than the grid allows. A wider grid, still picked on held-out training images, is a possible next try.
  - **Checks:**
    - R11 reproduces its D4 CSV.
    - Seed 42's maps are rebuilt exactly from `instances_val.csv`.
    - Seed 43's aren't: 0.3254 rebuilt against 0.3242 published. Rounding, the biases and file errors were ruled out ([E5-results/E5/](Champion_Task/E5-results/E5/README.md)). Its clean score uses the lower value.
  - **Files:**
    - Results in [R16c-results/step5/](Champion_Task/R16c-results/step5/README.md), moved there from `R16c-step5-results/`.
    - The graders (`grader_seed42.pth`, `grader_seed43.pth`) stay local and are git-ignored.
  - **What it leads to:** O1's status on clean data rests on this map. A third grader seed (44) would show whether seed 43's narrow pass is typical.
    - **Built 26 Oct on your go** as [r16c-step5b-third-grader-seed.ipynb](Champion_Task/r16c-step5b-third-grader-seed.ipynb): step 5 with seed 44 and its own output folder.
    - Additions: clean scores (E5's 40 listed in Cell 2), 6-decimal log-probabilities, and a rebuild check of the saved rows against the maps' own counts.
    - **O1's verdict rule, fixed before the run:** every seed above 0.52 / 0.30 on the clean pairs means met on three seeds; otherwise a passing three-seed mean means met on the mean only, narrowly; otherwise not met.
    - **Dry-run checked on your PC:** Cell 10 on step 5's real data reproduced R11 and seed 42's clean 0.5469 / 0.3161. Cells 9 and 10 ran end to end on synthetic images, and their clean counts and rebuild check were exact.
    - About 2.5–3.5 h on account 1 (my first estimate of 1 h left out the masks and crops).

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
- **Session 2 again (account 3, 21–22 Oct, about 6 h): steps 16–24 done.**
  - **Against the expectation:** one step further than planned (24, not 23), because each step took about 37 min (2,170–2,259 s) instead of session 1's 42.
  - **The fix held:** the 8-bit AdamW resumed checkpoint-15's state with no crash.
  - **Memory:** peak 14.4 GB on GPU 0 and 14.5 GB on GPU 1.
  - **Loss:** 0.520 at step 24, still in session 1's band (0.52–0.60). The learning rate is down to 1.6e-5 on the cosine schedule.
  - **Checkpoints:** 16, 20 and 24 are on Hugging Face (log lines, plus a sha256 check of the local copies before deleting them).
  - **A misleading number:** the summary's `train_loss` of 0.205 is this session's 9 losses divided by all 24 steps, a quirk of resumed runs; ignore it.
  - Results in [L0-step2-results/train-session2/](Champion_Task/L0-step2-results/train-session2/L0S2-train/README.md).
  - **Left:** 4 steps (about 2.5 h) plus the merge. Session 3a takes one step with account 3's last 1 h 26 min, and 3b finishes next week.
- **Session 3a (account 3, 22 Oct, about 1 h): step 25 done, exactly as planned.**
  - It resumed from checkpoint-24, trained one step (2,541 s), and the 0.3 h time limit stopped it.
  - **Loss:** 0.537; the learning rate is down to 1.1e-5.
  - **Memory:** peak 14.3 GB on each GPU.
  - **Checkpoint:** checkpoint-25 is on Hugging Face (log line and sha256 match).
  - **Its own `train_loss` (0.021)** is the same resume quirk: one step's loss divided by 25.
  - Results in [L0-step2-results/train-session3a/](Champion_Task/L0-step2-results/train-session3a/L0S2-train/README.md).
  - **Left:** steps 26–28 (about 2.1 h) and the merge, in session 3b with `time_limit_h` back to 5.0.
- **Session 3b (account 3, finished 26 Oct, about 2 h): training complete, but the merge failed** (the merge-only copy merged it on 27 Oct, below).
  - **Training:**
    - It resumed from checkpoint-25 and ran steps 26–28 (5,941 s, about 33 min a step).
    - Losses 0.491, 0.537 and 0.533. All 28 losses stay in 0.49–0.60, nearly flat, as in every session.
    - Peak memory 14.4 GB.
    - **Checkpoint-28 is on Hugging Face** (log: `✓ Pushed checkpoint-28 to AbrarAlam/disasterm3-qwen25vl7b-qlora/l0s2/checkpoints`).
    - `train_loss` 0.0557 is the resume quirk again: 3 losses divided by 28 steps.
  - **The merge (Cell 6) stopped after 5 s:**
    - The error: `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`.
    - **Cause:** Kaggle's image ships torchao 0.10.0, and the peft that Cell 1 upgrades (0.21.2) refuses torchao below 0.16 when it tries its torchao layer type. That only happens on an fp16 base. Training loaded the base in 4-bit, so bitsandbytes' layer type matched first and training never hit the check.
    - **Fix:** uninstall torchao before peft loads, since nothing uses it. Then merge only: restore checkpoint-28 from Hugging Face, merge on the CPU in fp16, and push `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP`. No GPU and no retraining is needed.
    - **Built 26 Oct on your go:** [l0-step2-merge.ipynb](Champion_Task/l0-step2-merge.ipynb), the training notebook's Cells 1, 2 and 6.
      - It removes torchao, pins peft 0.21.2 (the training's version) and downloads only checkpoint-28's adapter.
      - It checks that exactly the 196 adapted projections changed, that none holds an inf or NaN, and that every uploaded file has its local size. The weight checks run before the push.
      - **Dry run on your PC:** a tiny Qwen2.5-VL with a random adapter and a stand-in Hugging Face. It merged correctly (base + 2·B·A to within 3e-5, nothing else changed). An adapter with one projection left at zero was stopped before anything was pushed.
    - **Run 1 (27 Oct, account 3, CPU, 35 s): stopped in Cell 2 on a check of mine.**
      - The check expected the adapter config to list 196 target modules, and it lists 88. Given more than 20 module names, peft saves the shortest name patterns that pick the same modules: q/k/v/o_proj plus each layer's three MLP names (4 + 28 × 3 = 88).
      - The adapter itself is right: one LoRA pair per module, and train.py asserted the 161,480,704 LoRA parameters.
      - The dry run missed it because its 2-layer model had 14 targets, under peft's threshold of 20.
      - Cell 1 worked (peft 0.21.2, no torchao, 34 GB of RAM). Nothing was merged or pushed.
      - Log in [L0-step2-results/merge/run1-stopped-at-cell2/](Champion_Task/L0-step2-results/merge/run1-stopped-at-cell2/l0-step2-merge.log).
    - **The fix (27 Oct):** Cell 2 counts the adapter's LoRA pairs (196) instead of the config's name patterns.
      - Dry-run again with a 4-layer model, where peft shortens 28 names to 16 patterns: 28 pairs counted, exactly the 28 projections merged, and the broken adapter still stopped before the push.
      - Run 2 goes in the same Kaggle notebook (re-imported) on account 3.
      - **Lesson for future dry runs:** match the real model's size wherever a library changes behaviour above a threshold.
    - **Run 2 (27 Oct, account 3, CPU session, 10 min): merged, checked and pushed.**
      - **Expectation, fixed before the run:** 196 tensors change, all of them language-model projections; about 16.6 GB pushed; 30–45 min.
      - **Result:**
        - Cell 2: checkpoint-28 restored (step 28 of 28, rank 64, alpha 16) with 196 LoRA pairs in 88 saved name patterns. Its last three losses (0.4905, 0.5365, 0.5335) match session 3b's log.
        - Cell 3: 196 of 729 weight tensors changed, exactly the adapted projections (none outside them, none unchanged), and none holds an inf or NaN. Merge 4.3 min.
        - Pushed `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP` (public): 7 files, 16.6 GB, every file's size on the Hub matching the local copy. Upload 2.4 min.
      - **Against the expectation:** met in full, in 10 min instead of 30–45 (the transfers were faster than assumed).
      - The run notebook's code matches the build cell by cell, and it replaced the output-free copy. Results in [L0-step2-results/merge/](Champion_Task/L0-step2-results/merge/L0S2-merge/README.md).
  - **Files:**
    - The log is in [L0-step2-results/train-session3b/](Champion_Task/L0-step2-results/train-session3b/l0-step2-topup-training.log).
    - The run notebook replaced `l0-step2-topup-training.ipynb`, as with the earlier sessions.
    - The session's `L0S2-train/` outputs (README, loss curve, `train_summary.json`) weren't downloaded. The notebook's Cell 5 output holds the same numbers.
  - **Left:** the new 7B's six tasks at full size (the re-score notebook, `MODEL = "new"`, in two parts: A on account 4, B on account 2; the merge's README still says A, B and C). Its scores get the same E5 filter: all questions and clean.

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

## 9. Teammates' work: what we use (reviewed 26 Oct)

Every push since 22 Oct was read file by file, and each number was checked against its notebook's outputs or its data. Only what holds up is used.

### Ridita · Offline guideline search (`Offline_RAG_KR2/`, pushed 25 Oct)
- **What:** a local guideline search with citations for DisasTeller's report writers, plus an `r1` mode for the old building counter.
- **What we use:**
  - The search, for T2 step 4. It has 405 sections from six sources (EMS-98 167, INSARAG Vol II 90, Sphere 61, INSARAG Vol III 47, xBD 34, IWG-SEM 6), each with a citation ID such as `[EMS98-masonry-grade-5]`.
  - It ranks sections by meaning (MiniLM embeddings) and by keywords (BM25), merged by reciprocal rank fusion. It runs offline on CPU: Chroma on a laptop, the saved vectors with numpy and torch on Kaggle.
  - DisasTeller already imports a tool with the same name, so swapping it in is a file copy.
  - Checked: 405 sections, 405 vectors and 405 database rows, with no duplicate IDs.
- **Not yet shown:**
  - that it ran on Kaggle (the check notebook was pushed without outputs);
  - that real report questions bring back the right sections (its 16 test questions only check that Kaggle and the laptop agree).
- **Not used:** the counter's `r1` mode. H2's table already holds R11's buildings one by one, while the counter counts blobs, so touching buildings merge into one.
- **Open:** Sphere's text is in the public repo three times: the PDF, the chunks file and the database. Your decision.
- **Next:** wire it into a copy of DisasTeller, offline by default, and test it on 30 report questions with their expected sections, with the pass rule fixed first (an expected section in the top 3 for at least 24 of 30).

### Ridita · The DisasterM3–BRIGHT audit (corrected 22 Oct, pushed 25 Oct)
- **What we use:**
  - **The leak:** 40 of the 645 validation pairs are near-copies of training pairs. Followed up as E5.
  - **BRIGHT can't add destroyed examples:**
    - BRIGHT released no colour after-photos (all 4,246 are radar), so it adds only before-photos, where every building is intact.
    - The five usable events (La Palma, Goma, Les Cayes, Marshall, Noto; 1,285 tiles) don't leak.
    - But 1,056 of those tiles are volcano scenes, and validation has only 8 volcano pairs. R13 should expect little change in Destroyed IoU.
  - **Never train on BRIGHT's Turkey, Morocco or Libya tiles:** they are the before-photos of 66 validation pairs.
  - **Before- and after-photos don't line up:**
    - 52% of the 2,966 measurable pairs are shifted by more than 2 px, 10% by more than 5 px, the worst by 56 px.
    - The audit says this doesn't hurt the champion because it reads only the after-photo. That holds for R1, not R11: R11 stacks both photos, and R16c uses R11's pieces.
    - Aligning the pairs first is a candidate R11 variant.
  - **File names hide radar:** 73% of DisasterM3's radar after-files have no "sar" in the name. Our notebooks filter by the `image_type` tag and are unaffected; any filter by file name isn't.
  - **Resolution:** DisasterM3's copies of BRIGHT tiles are 0.30 m per pixel, not the 0.8 m its paper says, and BRIGHT's own table is wrong for 6 of its 14 events.
- **Not yet available:** the scripts and files behind the numbers, `val_train_leaks.csv` among them. The prompt in [Ridita_Task/E5_leak_audit_prompt.md](Ridita_Task/E5_leak_audit_prompt.md) asks for them.

### Aryan · Counts in the prompt (`Aryan_Task/BDC_PIPELINE_REPORT_v3.md`, pushed 25 Oct)
- **What:** the fine-tuned 7B in 4-bit, asked 1,703 optical counting questions from the benchmark about damaged or destroyed buildings (multiple choice, chance 20%), with and without a count in the prompt.
- **How it was scored:**
  - Strict: only a single letter A–E counts, and anything else is wrong.
  - Wilson 95% intervals. The base 7B was asked an evenly spread sample of 500.
  - The notebook's outputs match the report.
- **What we use:**
  - **The counting gap is seeing, not reasoning.**
    - With the right count from the ground-truth mask, the fine-tuned 7B scores 96.5% (92.7% without the image). With the image alone it scores 15.7%, with the question alone 21.1%.
    - On the 500, the base 7B scores 89.4% with counts and 34.4% without; the fine-tuned 7B scores 97.2% and 18.2%.
    - So the fine-tuned 7B reads a count it's given better than the base model does. This backs feeding counts as text (H1 V2, H2).
  - **Each question needs its own mask:** damaged questions the damaged mask, destroyed questions the destroyed mask. Counting connected components in the right mask then gives the benchmark's answer 93.9% of the time.
  - **Answer format:**
    - 189 of the 1,703 image-only answers (11%) were sentences with a number ("…is 37") instead of a letter.
    - Our own scorer, the 499A scorer used by H1 V2 and L0, also counts those as wrong.
    - A CPU check of our 7B's counting answers would show how much of its gap is format rather than counting.
- **Not usable for O2:** the run with a real segmenter used B1 (24.0%), not R11. The next run takes R11's counts from H2's table.
- **Not checked yet:** whether benchmark images are copies of training images. E5's hash check covers it.

### Tamanna · Plan 2, building-by-building crop inspection (`Tamanna_task_2/Plan_2/`, pushed 17–22 Oct)
- **What we use:**
  - **The idea:** crop each building at full resolution and ask the VLM about it. It overlaps R16c's per-building grader and T2's report, and the dashboard code may help T3's visuals.
  - **Two lessons:**
    - Telling the VLM the segmenter's class and asking whether it agrees gets agreement nearly every time (100% for the fine-tuned 7B on 10 tiles). So agreement can't stand in for accuracy, and checks in T2 and H2c must never show the model the answer being checked.
    - Filtering by file name let radar images into a colour-photo set: 4 of the tiles used were radar.
- **Not used:** any number from Plan 2.
  - None traces back to a recorded run of the committed code (the notebooks' run timestamps and git history were checked on 26 Oct).
  - Several numbers were typed in.
  - The tiles used were training or radar images.
- **Next:** a clean re-run, then a test against ground truth on the benchmark's test images: does a zoomed crop let the 7B grade buildings better than R11?
