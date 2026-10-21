# CSE499B progress tracker

**Last updated:** 22 Oct 2026, 01:20.
- **H2b built (your go, 22 Oct):** [h2b-sql-agent-graded-test.ipynb](Champion_Task/h2b-sql-agent-graded-test.ipynb), a graded test of which model, if any, should write SQL in the deployed system. It runs once H2's table exists.
  - **Why:** H2's counting questions all need the same query, so they can't separate the models. Counting needs no model at all (a fixed query does it), and a second model costs memory on low-resource hardware.
  - **What:**
    - 29 open questions in 12 groups, each in 3 phrasings on 3 images, plus 12 questions the tables can't answer;
    - graded by test-suite accuracy on 5 versions of the table, with RS(10) penalising wrong answers;
    - our 7B, the base VL-3B (L2's student), and Qwen2.5-Coder 1.5B and 3B, each in 4-bit and fp16, with memory and latency measured.
  - **Decision rule (fixed before any run):**
    - a configuration passes at 90% or more test-suite accuracy and RS(10) above 0;
    - if our VLM passes in 4-bit, there's no second model;
    - otherwise the smallest passing coder is the fallback, with its memory stated;
    - if nothing passes, fixed queries only.
  - **Checked on your PC:**
    - the notebook ran end to end on a synthetic table with four stand-in models;
    - the test suite caught all 6 near-miss queries and accepted all 7 correct rewrites;
    - tie and midline images were excluded.
- **L0 training, session 2 failed (account 3, about 1 h of GPU used, nothing saved).** It restored checkpoint-15 correctly, trained through step 16's 128 micro-batches (about 39 min), then crashed at that step's optimizer update on both GPUs: `Error invalid argument ... pythonInterface.cpp`, inside bitsandbytes' paged-memory prefetch.
  - **Cause (checked on your PC):**
    - the paged optimizer marks its state with an `is_paged` flag, and `torch.save` keeps that flag in checkpoint-15;
    - on resume, the trainer loads that state into ordinary GPU memory, and the flag survives;
    - bitsandbytes then prefetches it as if it were paged memory, and CUDA rejects that.
    - Session 1 never resumed, so it never hit this.
  - **Fix (Cell 2):** `optim = "adamw_bnb_8bit"`, the same 8-bit AdamW without paging, so the same maths and the same memory. Session 2 already held its optimizer state as ordinary GPU memory through a full step without running out. `time_limit_h = 5.0`, because account 3 has about 7 h 12 min left.
  - Nothing from session 1 is lost: checkpoint-15 is intact on Hugging Face.
- **H2 is ready to run (account 2):** [h2-sql-counting-agent.ipynb](Champion_Task/h2-sql-counting-agent.ipynb), the run copy of Noman's notebook (his original in `Noman_task/` is unchanged).
  - It builds the per-building table from **R11 with 8-view TTA** and blobs of at least 20 px, our best counter on validation (39.24% on the 1,055 questions, against 33.3% for R9's untuned counter). Noman's version used R9's standard mask only to reproduce H1 V1's counts.
  - It also measures R11's counting on all 4,982 benchmark BDC questions for the first time (Cell 7, the "oracle" row).
  - The LLMs now load with `dtype=` instead of `torch_dtype=`, which transformers 5 deprecated. Kaggle has 5.16.1, and L0 step 2 already loads this way.
  - Checked on your PC: every code cell parses, the table's self-checks pass, and the TTA views invert correctly on a non-square image.
- **R16c step 3 done (account 2, inputs shared from account 1): O1's gate passes on the second U-Net seed.**
  - SAM 3 + R11s7's pieces: 0.5843 / 0.3090 (grader seed 42) and 0.5830 / 0.3006 (seed 43).
  - Both sanity checks pass.
  - Not kept against R11s7 with TTA, because mIoU falls by 0.007–0.008 (the rule allows 0.005).
  - So O1 is met on two U-Net seeds (mean of the four combinations 0.585 / 0.309), narrowly on one of them.
- **L0 training, session 1 done (account 3):**
  - 15 of 28 steps, about 42 min each; no out-of-memory error (peak 14.2 GB);
  - checkpoints 4, 8, 12 and 15 are on Hugging Face;
  - loss nearly flat (0.563 → 0.557);
  - session 2 is ready (see the run board).
- **GPU left (21 Oct, 19:26):** account 1: 0; account 2: 2 h 4 min; account 3: 8 h 12 min.
- **Results moved into the pattern:** `R16c-results/step3/` and `L0-step2-results/train-session1/`.
- **Two runs failed overnight (logs downloaded 21 Oct):**
  - **L0 training, Version #7 (account 3):**
    - the first fix worked: step 1 of 28 completed in 40.8 min, loss 0.564;
    - step 2 then ran out of GPU memory on GPU 0 (it needed 926 MiB with 887 MiB free), after 1.25 h;
    - nothing was saved, since the first checkpoint comes at step 4.
    - **Fix ready:** 8-bit paged AdamW (`optim = "paged_adamw_8bit"`, QLoRA's optimizer), which frees about 1 GB per GPU. `time_limit_h` goes back to 10.0, because the data step took 31 min on a slow Kaggle disk.
    - **Checked 21 Oct, 02:15:** your local Cells 2 and 3 match the fixed cells line for line.
    - **If 1 GB still isn't enough,** the next options, from smallest change to largest, each your decision:
      1. leave out the few longest examples (two images and the most text), after counting how many;
      2. use a slightly smaller image size for two-image examples only;
      3. halve the LoRA rank from 64 to 32, which frees about 0.6 GB more but changes the adapter.

      A failure would most likely show within the first few steps (about 1–2 GPU hours).
  - **R16c step 5 (account 1):**
    - Cell 3, building the 6,443 masks, took 8.1 h instead of minutes (Kaggle's input disk was very slow that evening);
    - the run stopped in Cell 6 when account 1's GPU hours ran out (now 0);
    - no results. Steps 5 and 3 wait for account 1's quota to reset, unless you choose another way.
  - **GPU left:** account 1: 0; account 2: 3 h 20 min; account 3: 19 h 14 min.
- **Decided 21 Oct, 02:00 (option B):**
  - R16c step 3 runs now on account 2, from a private dataset `r16c-step3-inputs`: E3's 645 `pre_grid2` maps and step 2's graders. The zip was built and checked identical to the originals.
  - R16c step 5 runs again on account 1 once its quota resets.
  - Instead of the upload, you can share account 1's `e3-sam3-building-finding` and `r16c-step2-stronger-grader` with account 2 and attach their outputs. Sharing `r16c-step4-sam3-training-images` too lets step 5 run on any account.
  - **Checked 21 Oct, 02:15:** your local step 3 Cell 2 matches the fixed cell line for line (`EXPECTED_UNET` = R11s7's 0.5867 / 0.2869 and 0.5913 / 0.2928).
  - In both, the mask-building cell must finish in about 15 min. If it doesn't, the Kaggle disk is slow, so stop rather than burn the hours.
- **L0 step 2's training, session 1 failed after 11 min** (account 3, committed about 13:12): CUDA out of memory at step 1 on both T4s. Nothing was trained or saved, and only about 11 min of account 3's quota was used.
  - **Cause:**
    - the trainer collects all 128 micro-batches of a step before training on any of them, and the data loader has already put each one on the GPU;
    - with full-size images, that's 3–6 GB of waiting batches per GPU;
    - the smoke test held only 4 (`grad_accum` 4), so it couldn't show this.
  - **Fix, tested on your PC:** a `StepOnCpu` trainer in Cell 3 keeps the waiting batches in RAM (images in fp16, the dtype the model converts them to anyway). The recipe, the global batch of 256 and the steps are unchanged.
- **Decided 20 Oct, 13:15: A, the equal replay as decided 19 Oct** (7,018 examples, about 19.5 h of training). Session 1 on account 3 is ready to run. Session 2's time limit is set from session 1's measured speed, so it stops before account 3's hours run out, and a short session 3 on account 2 finishes and merges.
- **The training's smoke test is done** (account 3, about 45 min). It raised the question of how much replay, because the full training is slower than estimated.
  - **Healthy:**
    - the loss went 0.844 → 0.461, inside the original run's 0.4–1.2;
    - both T4s ran;
    - peak memory was 14.7 GB of the T4's 15.8 GB, measured with the longest example (3,092 tokens), so the full run's worst case is covered.
  - **Slow:** 10.0 s per example across both GPUs, so the full 7,018 examples project to **19.5 h** (estimate was 7–14 h). That's two 10 h sessions plus a short third one.
  - **The choice:**
    - (A) the equal replay as decided, about 20.5 h of GPU: two sessions on account 3, then about 1–2 h on account 2;
    - (B) half the replay, `REPLAY_RATIO = 0.5` (5,264 examples), about 15 h: two sessions on account 3, with a higher risk that the other tasks slip.
- **The old 7B's full-size re-score is done** (L0 step 2: part A on account 3, 8.6 h; part B on account 2, 4.1 h). Checked: every accuracy recounted from the per-question answers. 8 questions whose images can't be read are scored wrong, the same 8 as at 512 px.

  | Task | 512 px (H1 V2/V2b) | Full size | Change |
  |---|---|---|---|
  | DTR (from L0 step 1) | 71.10 | 75.64 | +4.54 |
  | BBR | 19.54 | 20.94 | +1.40 |
  | BDC | 32.01 | 30.09 | −1.92 |
  | DRE | 32.87 | 32.39 | −0.48 |
  | DSR | 37.87 | 40.06 | +2.19 |
  | ORR (from L0 step 1) | 23.31 | 31.12 | +7.81 |

  - This is the baseline the new 7B must beat (keep rule: DTR and ORR rise, no other task drops by more than 1 point).
  - Counting drops at full size (30.09%), so the 7B alone still misses the counting half of KR2 (34.2%).
- **R16c step 4 done** (account 1, 4.6 h): SAM 3 on all 5,798 training images, building IoU 0.629 (0.630 on validation in E3), destroyed footprint recall 0.693. Step 5 therefore trains on outlines as good as the ones it grades.
- **Running now (20 Oct, from about 12:10):**
  - Account 1 (`abrarmohammedtanzim`): R16c step 5 (the account had 3 h 30 min when it started); then step 3, about 0.5 h.
  - Account 3 (`redwannewaz`): the smoke test is done (13:00). About 20 h left; the full training's session 1 is ready to run (A, equal replay).
  - Account 2 (`abrar2222864642`): free, 3 h 20 min left.
- **Renamed on Kaggle (account 1):** `l2-stage1-teacher-rationales` → `l2-step1-teacher-rationales`, `r16-two-stage-grader` → `r16-find-then-grade`.
- **R11s7 done (account 2, 6.8 h), clean run:** 0.5867 / 0.2869, or **0.5913 / 0.2928 with TTA** (R11: 0.5875 / 0.2965).
  - The two U-Net seeds differ by only 0.004 in Destroyed IoU, but their mean with TTA is 0.2947, so the U-Net alone still misses O1's 0.30 on two seeds.
  - O1's two-seed claim now rests on R16c step 3 (step 2's graders with R11s7's pieces), on account 1 after steps 4 and 5.
- **Renames (your decisions, 20 Oct):**
  - **R6 is now R1s7** (R1 with seed 7). Its notebook is `d4-champion-model-R1s7.ipynb`, and its results are in `R1s7-results/d4_R1s7/`, relabelled with a note that it was R6.
  - **"Step", never "stage":** L2 step 1's files are `l2-step1-teacher-rationales.ipynb` and `L2-step1-results/`.
  - **R16's two parts are "the finding" and "the grading"**, and the model is find-then-grade. R16's notebook is `r16-find-then-grade.ipynb`.
  - **R16, R16b and R16c keep their names.** R16c's steps run 1–5: step 3 was 2b, step 4 was 3a, step 5 was 3b (now its own notebook). Step 2's results are in `R16c-results/step2/`.
  - **Inside the notebooks that already ran**, you make the changes by hand: [Manual_Notebook_Renames.md](Manual_Notebook_Renames.md) lists every one, cell by cell. The notebooks that haven't run are already updated.
  - **The other docs and the two published decks use the new names too**, with a note wherever R6 appeared ("called R6 until 20 Oct", or "named R6 at the review" in the deck).
- **R9's and R10's notebooks now hold their runs** (downloaded 20 Oct). Their printed scores match their CSVs.
- **Audit of `Champion_Task` (20 Oct):** every notebook that ran has its results folder, and every number checked in this tracker and the ledger matches its results file. R11's second seed is R11s7 everywhere, and R15 stays "Not started".
- **R16c step 2 passes the keep rule and the O1 gate on both grader seeds.** SAM 3 + R11's pieces, with the stronger grader, reach Destroyed IoU 0.316 and 0.311 (R11 0.297), with mIoU 0.587 and 0.585 (R11 0.588).
  - It wins on Turkey (0.000 → 0.24–0.27) but loses on xBD images (0.456 → 0.35–0.38).
  - Counting doesn't improve: 37–39% on the validation questions, about the same as R11.
- **Decided 20 Oct:**
  - R16c steps 4 and 5 (were 3a and 3b) run now on account 1, as a stretch.
  - R16c step 3 (was 2b, R11s7's pieces) is built and runs after step 5.
  - Account 3 (30 h) starts with the old 7B's full-size re-score, then L0 step 2's training.
- **New:** [Task_Ledger.md](Task_Ledger.md) records every finished task and your decisions so far. For each task it gives what it was, why it was planned, how, how it ran, the expectation before the run, the result and what it led to.

Kept by Claude for Abrar. Claude updates this file whenever a task starts running, finishes, or needs a re-run. The task IDs follow [Next_Steps_Roadmap.md](Next_Steps_Roadmap.md); results come from the folders in [Champion_Task/](Champion_Task/). Why each task ran and what it decided: [Task_Ledger.md](Task_Ledger.md).

**Status words:** Done · Running · Ready to run (notebook built, not started) · Needs your decision · Needs rerun · Not started.

## At a glance

| Done | Running | Ready to run | Needs your decision | Needs rerun | Not started |
|---|---|---|---|---|---|
| 21 | 1 | 2 | 0 | 1 | 12 |

## Run board: what ran, what's running, what's next

Every notebook run so far, on Kaggle or on your PC, and every planned run. Updated with each change; details under "Needs you now" and in the task tables.

**Running now**

| What | Notebook | Account | Started | Time |
|---|---|---|---|---|
Nothing is running on Kaggle (21 Oct, 19:30).

**Ready to run now**

| What | Notebook | Account | Time |
|---|---|---|---|
| L0 step 2: the full training, session 2 again (continues from checkpoint-15) | [l0-step2-topup-training.ipynb](Champion_Task/l0-step2-topup-training.ipynb), Cell 2 with `optim = "adamw_bnb_8bit"` and `time_limit_h = 5.0` | 3 (`redwannewaz`, about 7 h 12 min left) | about 6 h: stops after about step 23 of 28, then saves to Hugging Face |
| H2: R11's per-building table and the SQL counting agent | [h2-sql-counting-agent.ipynb](Champion_Task/h2-sql-counting-agent.ipynb) | 2 (`abrar2222864642`) | smoke test done 22 Oct (passed: R11 loaded, 6 images with their before-images, the 3B's SQL valid on 6 of 6). The full run takes about 1–1.5 h. It needs `H1_v1_per_question.csv`, which the Kaggle dataset `h1-v1-facts` doesn't contain (it holds only `H1_v1_summary.csv` and `facts.json`), so upload the local copy from `H1-V1-results/H1/` as an input. |

**Waiting: built, runs when its input or account frees up**

| What | Notebook | Account | Waits for | Time |
|---|---|---|---|---|
| R16c step 5 (was 3b): the grader trained on SAM 3's outlines | [r16c-step5-grader-on-sam3-outlines.ipynb](Champion_Task/r16c-step5-grader-on-sam3-outlines.ipynb) | 1, or any account its inputs are shared with | account 1's quota reset (its first try stopped when the hours ran out) | 1.5–2 h |
| L0 step 2: the full training, session 3 (the last 5 steps), then the merge | the training notebook, the same Cell 2 (`adamw_bnb_8bit`) | 3 or any account with the HF token | next week's GPU quota | about 3.5 h of training + about 1 h merge |
| L0 step 2: the new 7B on all six tasks | the re-score notebook, `MODEL = "new"`, in parts | 3, plus 2 if needed | the training and merge | about 14.5 h (BDC + BBR 8.6 h, DRE + DSR 4.1 h, DTR + ORR about 1.7 h, as the old 7B took) |
| L0 step 2: the verdict, all parts scored together | a short scoring notebook on your PC, built when the parts arrive | your PC | both models' answers | minutes |
| H2b: graded test of the SQL agent (which model, if any, writes the SQL) | [h2b-sql-agent-graded-test.ipynb](Champion_Task/h2b-sql-agent-graded-test.ipynb) | any with H2's output attached and the HF token | H2's `buildings.sqlite` | smoke test about 10 min, then about 1–1.5 h |

**Later: not built yet**

| What | Waits for |
|---|---|
| H1 V3: counting from R16c's per-building grades | counting beating R11's on validation (now 37–39% against 39.2%) |
| L2 step 2: train the 3B; L2 step 1 again if L0's new 7B is kept | L0's verdict |
| L2b: explanations as a second skill | L2 step 1 with the final teacher |
| T2: DisasTeller on local models (O3's missing part) | GPU time; not started |
| T3: offline demo app | T2 |
| L1 again on the final champion | the final champion |
| R13 (BRIGHT footprints), R14 (rubble crops, optional), R15 (only if a change beyond R11 passes) | results and spare GPU |
| H3, H4, L3, L4 | not scheduled |

**Done on Kaggle**

| What | Notebook | Account, when | Run time |
|---|---|---|---|
| R1–R5, R1s7 (formerly R6), R8–R10, R9s7: the fine-tuning sprint (R7 = R1, no run of its own) | `d4-champion-model-R*.ipynb` (R4 took two sessions: `-r4-resume`) | 4–18 Oct | about 6.5–7.5 h each |
| E0: the old baselines re-scored | `d4-champion-model-E0.ipynb` | 4 Oct | 0.3 h |
| E1 (with E2, the vote): error diagnosis | `d4-champion-model-E1.ipynb` | 18 Oct | 0.4 h |
| E3: SAM 3 building finding | `e3-sam3-building-finding.ipynb` | account 1, 18 Oct | 1.0 h |
| H1 V1: U-Net counting | `h1-v1-unet-counting.ipynb` | 18 Oct | 0.6 h |
| T1 v1: heatmap | `t1-survivability-heatmap.ipynb` | Kaggle CPU, 18 Oct | 0.2 h |
| L1: edge benchmark | `l1-edge-benchmark.ipynb` | Kaggle CPU, 18 Oct | 0.2 h |
| L1b: INT8, second try | `l1b-static-int8-retry.ipynb` | Kaggle CPU, 19 Oct | 1.2 h |
| R11: before + after images | `d4-champion-model-R11.ipynb` | 19 Oct | 6.9 h |
| H1 V2: six tasks re-scored | `h1-v2-counts-as-text.ipynb` | account 2, 19 Oct | 4.4 h |
| H1 V2b: counting by the 7B | `h1-v2b-counting-by-vlm.ipynb` | account 2, 19 Oct | 2.9 h |
| L2 step 1: teacher answers | `l2-step1-teacher-rationales.ipynb` (renamed on Kaggle too, 20 Oct) | account 1, 19 Oct | 2.1 h |
| R16: find-then-grade | `r16-find-then-grade.ipynb` (renamed on Kaggle too, 20 Oct) | account 1, 19 Oct | 0.5 h |
| L0 step 1: image size | `l0-step1-image-size-check.ipynb` | account 1, 19 Oct | 1.7 h |
| R12: extra xBD tiles (second try) | `d4-champion-model-R12.ipynb` | account 2, 19 Oct | 6.9 h |
| E4: R12 against R11 by source | `e4-r12-vs-r11-by-source.ipynb` | account 2, 19 Oct | 0.3 h |
| R16c step 2: the stronger grader | `r16c-step2-stronger-grader.ipynb` (output `R16c-truth`, kept locally as `R16c-results/step2/`) | account 1, 19 Oct | 1.4 h |
| R16c step 3: step 2's graders with R11s7's pieces | `r16c-step3-r11s7-pieces.ipynb`; results in [R16c-results/step3/](Champion_Task/R16c-results/step3/README.md) | account 2 (inputs shared from account 1), 21 Oct | about 0.6 h |
| L0 step 2: the full training, session 1 (steps 1–15 of 28) | `l0-step2-topup-training.ipynb`; results in [L0-step2-results/train-session1/](Champion_Task/L0-step2-results/train-session1/L0S2-train/README.md) | account 3, 21 Oct | about 10.9 h |
| R11s7: R11 with seed 7 | `d4-champion-model-R11s7.ipynb` | account 2, 20 Oct | 6.8 h |
| L0 step 2: the old 7B's full-size re-score, part A (BDC, BBR) | `l0-step2-full-size-rescore-old-partA.ipynb`; results in [L0-step2-results/old-partA/](Champion_Task/L0-step2-results/old-partA/L0S2/README.md) | account 3, 20 Oct | 8.6 h |
| L0 step 2: part B (DRE, DSR) | `l0-step2-full-size-rescore-old-partB.ipynb`; results in [L0-step2-results/old-partB/](Champion_Task/L0-step2-results/old-partB/L0S2/README.md) | account 2, 20 Oct | 4.1 h |
| R16c step 4: SAM 3 on the training images | `r16c-step4-sam3-training-images.ipynb`; results in [R16c-results/step4/](Champion_Task/R16c-results/step4/README.md) | account 1, 20 Oct | 4.6 h |

**Done on your PC (CPU), run by Claude:** these wrote their results into `Champion_Task`, and the notebooks keep their outputs.

| What | Notebook | Results |
|---|---|---|
| D5: xBD converter | `d5-xbd-converter.ipynb` | `D5-results/` |
| D5b: xBD tier3 check and masks | `d5b-xbd-tier3-converter.ipynb` | `D5b-results/` |
| T1 v2 and v3: heatmap fixes | `t1-survivability-heatmap-v2.ipynb`, `-v3.ipynb` | `T1-results/T1-v2/`, `T1-v3/` |
| R16b: the grader's prior correction | `r16b-grader-prior-correction.ipynb` | `R16b-results/` |
| R16c step 1: the ceiling and counting | `r16c-step1-ceiling-and-counting.ipynb` | `R16c-results/step1/` |

Claude also ran read-only checks on your PC: the Turkey and counting breakdowns, dry runs of the new notebooks, and the folder audit. They read the results and write nothing into the project (only to Claude's temporary folder).

## Needs you now

GPU now (20 Oct, about 12:10): **account 1** (`abrarmohammedtanzim`): R16c step 5 running, 3 h 30 min when it started; **account 2** (`abrar2222864642`): free, 3 h 20 min; **account 3** (`redwannewaz`): the smoke test running, 21 h when it started.
- **Account 1:** R16c step 5 now (about 1.5–2 h), then R16c step 3 (about 0.5 h). Step 4 is done.
- **Account 3:** the training's smoke test now (about 25 min). Then the full training (7–14 h, set by the smoke test), then the new 7B's evaluation (about 14.5 h, shared with account 2 and next week's quota).
- **Account 2:** the spare. Parts A and B are done.
- Kaggle offers T4s only in pairs ("GPU T4 x2"). The R16c notebooks use one of the two, and the quota counts session hours either way.
- Account 3 needs the `HF_TOKEN` secret (AbrarAlam, write access) and the benchmark mirror `redwannewaz/disasterm3-bench-mirror-v2`.

0. **R11s7 is done** (20 Oct, account 2): results in [R11s7-results/](Champion_Task/R11s7-results/d4_R11s7/D4_R11s7.csv), checked (see R11s7's row below).
1. **R16c on account 1: steps 4, 5 and 3** (decided 20 Oct; renumbered the same day: step 3 was 2b, step 4 was 3a, step 5 was 3b). Steps 1 and 2 are done (results in R16c's row below). Account 1 has 8 h 46 min.
   - **Expectation, fixed before the runs:**
     - Step 5 replaces step 2 only if its SAM 3 + R11's pieces map beats step 2's on both seeds by the keep margin: Destroyed IoU up by more than 0.013, mIoU down by at most 0.005. Also checked: whether xBD recovers without losing Turkey.
     - Step 3: SAM 3 + R11s7's pieces passes the O1 gate and the keep rule (against R11s7 with TTA) on both grader seeds, within about 0.01 of step 2. If it fails on either seed, O1 stays at one U-Net seed.
   - **Run 1, done 20 Oct (4.6 h): R16c step 4, SAM 3 on the training images** ([r16c-step4-sam3-training-images.ipynb](Champion_Task/r16c-step4-sam3-training-images.ipynb), about 3.5–4 h)
     1. Sign in to Kaggle with account 1 (AbrarAlam; its Hugging Face token is approved for `facebook/sam3`).
     2. Left menu: **Create → New Notebook**. In the editor: **File → Import Notebook → Browse**, pick `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\r16c-step4-sam3-training-images.ipynb`, then **Import**.
     3. Right panel, **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
     4. Right panel, **Input**: check that `disasterm3-mirror` (by abrarmohammedtanzim) is listed. If not: **+ Add Input**, search `disasterm3-mirror`, click **+**.
     5. Right panel, **Add-ons → Secrets**: turn on `HF_TOKEN`.
     6. In the cell starting with `# ── Cell 2: Configuration`, change `SMOKE_TEST = False` to `SMOKE_TEST = True`.
     7. Top menu: **Run → Run All**. Wait a few minutes (the install, then 8 images). Check:
        - Cell 1: `✓ SAM 3 installed (commit 7f473cf)`;
        - Cell 2: `✓ R16c step 4  [SMOKE TEST]: SAM 3 'building' on the training before-images`;
        - Cell 4: `Training images: 8 ... before-images found: 8`;
        - Cell 6: `✓ All training images done ... 8 of 8 maps`;
        - Cell 7: a one-row table with `sec_per_image` around 2 and `building_iou` around 0.5–0.7.

        Any red error: stop and send me a screenshot.
     8. In Cell 2, set `SMOKE_TEST = False` again.
     9. Top right: **Save Version**, choose **Save & Run All (Commit)**, then **Save**.
     10. Stop the editor session (the power button next to "Session") so it doesn't use GPU hours. The committed run carries on by itself.
     11. When Kaggle says "Your notebook finished": open the notebook, go to **Output**, and download `step4/README.md`, `step4/config.json` and `step4/summary.csv` into `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\R16c-results\step4\`. Don't download the `masks` folder; step 5 reads it on Kaggle.
     12. If that README says "Not complete yet" (it hit the 11-hour limit), tell me before step 5.
   - **Run 2, running since about 12:10, 20 Oct: R16c step 5, the grader trained on SAM 3's outlines** ([r16c-step5-grader-on-sam3-outlines.ipynb](Champion_Task/r16c-step5-grader-on-sam3-outlines.ipynb), about 1.5–2 h)
     1. Kaggle account 1: **Create → New Notebook → File → Import Notebook**, pick `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\r16c-step5-grader-on-sam3-outlines.ipynb`, then **Import**.
     2. **Session options**: GPU **T4 x2**, Internet **On**.
     3. **Input**: check that `disasterm3-mirror` is listed (if not: **+ Add Input**, search `disasterm3-mirror`, click **+** on the dataset by *abrarmohammedtanzim*). Then add two notebook outputs:
        - **+ Add Input → Your Work → Notebooks**: `e3-sam3-building-finding`, click **+**;
        - **+ Add Input → Your Work → Notebooks**: `r16c-step4-sam3-training-images` (run 1), click **+**.
     4. **Add-ons → Secrets**: turn on `HF_TOKEN` (R11's checkpoint is in the private repo `AbrarAlam/disasterm3-d4-R11`).
     5. In the cell starting with `# ── Cell 2: Configuration and inputs`, change `SMOKE_TEST = False` to `SMOKE_TEST = True`.
     6. **Run → Run All** (about 10–15 minutes). If Cell 1 says "RESTART THE KERNEL NOW": **Run → Restart & clear cell outputs**, then **Run All** again. Check:
        - Cell 2: `✓ R16c step 5, grader trained on SAM 3 outlines  [SMOKE TEST]: labels ['Intact', 'Damaged', 'Destroyed', 'None']`, then `Finding (SAM 3): ... (645 maps)` and `SAM 3 on the training images: ... (5,798 maps)`;
        - the cell starting `# ── Cell 4: Validation split`: `Grader training images: 20; held out: 4; validation: 8`;
        - the cell starting `# ── Cell 5: Crop helpers`: `✓ Crop helpers ready (self-checks passed)`;
        - the cell starting `# ── Cell 7: The grading`: two `Seed 42 | ep ...` lines, then `✓ Seed 42: kept epoch ...`;
        - the cell starting `# ── Cell 10: Results`: `Replaces step 2 ...: n/a (smoke test)`, then `✓ Wrote /kaggle/working/step5: ...`.
     7. In Cell 2, set `SMOKE_TEST = False` again.
     8. **Save Version → Save & Run All (Commit) → Save**. Then stop the editor session (power button).
     9. When it's finished: **Output**, download the whole `step5` folder, so that `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\R16c-results\step5\README.md` exists. It includes two grader files of about 46 MB each.
   - **Run 3, after step 5: R16c step 3, step 2's graders with R11s7's pieces** ([r16c-step3-r11s7-pieces.ipynb](Champion_Task/r16c-step3-r11s7-pieces.ipynb), about 20–30 min, no training)
     1. R11s7 has finished (20 Oct). Its model is in `AbrarAlam/disasterm3-d4-R11s7`, which this notebook downloads.
     2. Kaggle account 1: **Create → New Notebook → File → Import Notebook**, pick `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\r16c-step3-r11s7-pieces.ipynb`, then **Import**.
     3. **Session options**: GPU **T4 x2**, Internet **On**.
     4. **Input**: check that `disasterm3-mirror` is listed (if not: **+ Add Input**, search `disasterm3-mirror`, click **+** on the dataset by *abrarmohammedtanzim*). Then:
        - **+ Add Input → Your Work → Notebooks**: `e3-sam3-building-finding`, click **+**;
        - **+ Add Input → Your Work → Notebooks**: `r16c-step2-stronger-grader`, click **+** (its output folder `R16c-truth` holds step 2's graders).
     5. **Add-ons → Secrets**: turn on `HF_TOKEN`.
     6. In the cell starting with `# ── Cell 2: Configuration and inputs`, make two changes:
        - the line `EXPECTED_UNET = None` → `EXPECTED_UNET = {"std": (0.5867, 0.2869), "tta": (0.5913, 0.2928)}` (R11s7's own scores from `D4_R11s7.csv`; keep the comment after it if you like);
        - `SMOKE_TEST = False` → `SMOKE_TEST = True`.

        Then **Run → Run All** (a few minutes). If Cell 1 says "RESTART THE KERNEL NOW": **Run → Restart & clear cell outputs**, then **Run → Run All** again. Check:
        - Cell 2: `✓ R16c step 3  [SMOKE TEST]: step 2's graders (seeds (42, 43)) with R11s7's pieces`, a `Graders and biases: .../R16c-truth` line, and a `U-Net R11s7: ...` line;
        - the cell starting `# ── Cell 4: The validation split`: `✓ Validation: 8 images`;
        - the cell starting `# ── Cell 6: Step 2's two graders`: `✓ Graders loaded: seed 42 (epoch 6, ...), seed 43 (epoch 6, ...)`;
        - the cell starting `# ── Cell 9: Results`: ends with `✓ Wrote /kaggle/working/step3: ...`.
     7. In Cell 2, set `SMOKE_TEST = False`. **Save Version → Save & Run All (Commit) → Save**. Stop the editor session.
     8. When it's finished: **Output**, download the whole `step3` folder, so that `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\R16c-results\step3\README.md` exists.
     9. I then check its two sanity lines: `The graders ... reproduced: yes`, and its R11s7 rows against `D4_R11s7.csv` from item 0.
2. **L0 step 2 (decided 19 Oct):** a short extra training of our adapter on disaster type and relational at full size (1036 px), with an equal replay sample of the other tasks, about 7,000 samples.
   - Then both models are scored at full size on every optical question (10,675), for the full like-for-like table.
   - **First, the old 7B's re-score of the other four tasks at full size:** [l0-step2-full-size-rescore.ipynb](Champion_Task/l0-step2-full-size-rescore.ipynb), a copy of L0 step 1's notebook with H1 V2's prompts for the four tasks. It was dry-run on your PC with the real benchmark file and a stand-in model: 10,675 optical questions as in H1 V2, nothing asked twice when continuing, and the scoring checked by hand.
     - **Expectation, fixed before the run:** no decision hangs on it; it's the baseline. The four tasks should move by a few points at most from 512 px. Counting (32.0% at 512 px) has the most to gain; above 34.2% the counting half of KR2 is met.
     - **Done 20 Oct:** part A on account 3 (8.6 h), part B on account 2 (4.1 h). Results and the full-size baseline are at the top of this file. The steps below are kept as the record of how they ran.
     - **Part A, account 3** (`["BDC", "BBR"]`, 5,589 questions, about 7.5 h):
       1. Sign in to Kaggle with account 3.
       2. Left menu: **Create → New Notebook**. In the editor: **File → Import Notebook → Browse**, pick `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\l0-step2-full-size-rescore.ipynb`, then **Import**.
       3. Right panel, **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
       4. Right panel, **Input → + Add Input**: search `disasterm3-bench-mirror-v2`, click **+** on the dataset by *redwannewaz*. If it doesn't show up, the dataset is private: ask redwannewaz to share it with account 3.
       5. No secret is needed: the old model's repo is public.
       6. In the cell starting with `# ── Cell 2: Configuration`, check that it says `MODEL = "old"` and `RUN_TASKS = ["BDC", "BBR"]`. Change `SMOKE_TEST = False` to `SMOKE_TEST = True`.
       7. **Run → Run All**. Wait about 15 minutes (the 16 GB model downloads first). Check:
          - Cell 1: `GPUs: ['Tesla T4', 'Tesla T4']`;
          - Cell 2: `This part: ['BDC', 'BBR'], 20 questions` and `Images found: N of N`, with both numbers equal;
          - Cell 3: `Prompts ready (self-checks passed)`;
          - Cell 4: a `BDC: truth … | answer …` line, a `BBR: …` line, then `full size checked (1036 px)`;
          - Cell 5: `All questions answered`;
          - Cell 6: a table with `answered` = 10 for BBR and for BDC.

          Any red error: stop and send me a screenshot.
       8. In Cell 2, set `SMOKE_TEST = False` again.
       9. Top right: **Save Version**, choose **Save & Run All (Commit)**, then **Save**.
       10. Stop the editor session (power button next to "Session"). The committed run carries on by itself.
       11. When Kaggle says "Your notebook finished": **Output**, download the `L0S2` folder, and unzip it so that this file exists: `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\L0-step2-results\old-partA\L0S2\README.md`.
       12. Tell me "part A done".
     - **Part B, account 2** (`["DRE", "DSR"]`, 3,875 questions, about 4 h; R11s7 has finished):
       1. Check that no other session is running on account 2 (R11s7 finished on 20 Oct).
       2. Sign in to Kaggle with account 2. **Create → New Notebook → File → Import Notebook → Browse**, pick `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\l0-step2-full-size-rescore.ipynb`, then **Import**.
       3. **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
       4. **Input → + Add Input**: search `disasterm3-bench-mirror-v2`, click **+** on the dataset by *redwannewaz*.
       5. No secret is needed.
       6. In the cell starting with `# ── Cell 2: Configuration`, make two changes:
          - `RUN_TASKS = ["BDC", "BBR"]` → `RUN_TASKS = ["DRE", "DSR"]`;
          - `SMOKE_TEST = False` → `SMOKE_TEST = True`.

          Leave `MODEL = "old"` as it is.
       7. **Run → Run All** (about 15 minutes). Check:
          - Cell 2: `This part: ['DRE', 'DSR'], 20 questions` and `Images found: N of N`, with both numbers equal;
          - Cell 3: `Prompts ready (self-checks passed)`;
          - Cell 4: a `DRE: truth … | answer …` line, a `DSR: …` line, then `full size checked (1036 px)`;
          - Cell 5: `All questions answered`;
          - Cell 6: a table with `answered` = 10 for DRE and for DSR.
       8. In Cell 2, set `SMOKE_TEST = False` again. Keep `RUN_TASKS = ["DRE", "DSR"]`.
       9. **Save Version → Save & Run All (Commit) → Save**. Stop the editor session.
       10. When it's finished: **Output**, download `L0S2`, and unzip it so that this file exists: `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\L0-step2-results\old-partB\L0S2\README.md`.
       11. Tell me "part B done".
     - The two parts and L0 step 1's answers are then scored together on your PC.
   - **Then the top-up training:** [l0-step2-topup-training.ipynb](Champion_Task/l0-step2-topup-training.ipynb), the original recipe on today's Kaggle libraries.
     - Data: DTR and ORR plus an equal replay of the other tasks, at full size: 7,018 examples, about 28 steps of 256.
     - A technical change: the old adapter's saved layer pattern matches nothing in transformers 5, so a new adapter is trained on our merged 7B (the same starting model) and merged into it at the end.
     - Its logic was checked on your PC on the real manifest.
     - **Expectation, fixed before the run:** DTR and ORR rise by about 1–4 points; the other tasks stay within about a point.
     - **Smoke test, account 3, running since about 12:10, 20 Oct** (about 25 min). It measures the hours the full run needs.
       1. Kaggle account 3: **Create → New Notebook → File → Import Notebook → Browse**, pick `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\l0-step2-topup-training.ipynb`, then **Import**.
       2. **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
       3. **Input → + Add Input**: search `disasterm3-mirror`, click **+** on the dataset by *abrarmohammedtanzim*.
       4. **Add-ons → Secrets → + Add Secret**: label `HF_TOKEN`, value your Hugging Face token (AbrarAlam, with **write** access). Make sure its tick box is on. The run can't start without it.
       5. In the cell starting with `# ── Cell 2: Configuration`, change `SMOKE_TEST = False` to `SMOKE_TEST = True`.
       6. **Run → Run All**. Check:
          - Cell 1: a line of library versions, then `GPUs: ['Tesla T4', 'Tesla T4']`;
          - Cell 2: `config written and pushed`, then `Starting fresh`;
          - Cell 4: `Data: 7,018 examples ... missing images: 0; SMOKE TEST: 16 of them`, `LoRA on 196 language-model projections`, `Longest example: ... tokens`, two loss lines, `[GPU 0] peak memory ...`, `[GPU 1] peak memory ...`, and `Step 2 of 2 (complete)`;
          - Cell 5: the losses, the peak memory and the projected hours;
          - Cell 6: `Not merging: smoke test`.
       7. Copy Cell 5's whole printout and send it to me.
       8. Stop the session (power button). Don't commit this one.
     - **Full run, account 3, after part A** (7–14 h; I confirm the settings after the smoke test):
       1. Open the same notebook on account 3 (Your Work → `l0-step2-topup-training` → **Edit**).
       2. In Cell 2, set `SMOKE_TEST = False`.
       3. **Save Version → Save & Run All (Commit) → Save**. Stop the editor session.
       4. If it stops at the 10-hour training limit (Cell 5 then says "not complete"), commit it again the same way: Cell 2 restores the latest checkpoint from Hugging Face.
       5. When training is complete, Cell 6 merges and pushes `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP`. It's public like our first merged 7B; set `MERGED_PUBLIC = False` in Cell 2 before committing to keep it private.
       6. Send me the version's `L0S2-train/README.md` and `train_summary.json`.
     - If the smoke test projects much more than 14 h, you decide: two sessions as planned, or a smaller replay (`REPLAY_RATIO = 0.5`, about 25% less).
3. **R11s7 on account 2: started after E4 (decided 19 Oct), about 7 hours. These are the steps it was started with.** It is R11's recipe with seed 7.
   1. Create → New Notebook → File → Import Notebook → upload `Champion_Task\d4-champion-model-R11.ipynb`. This makes a copy, so your local R11 file stays untouched.
   2. In Cell 2, change `EXPERIMENT = "R11"` to `EXPERIMENT = "R11s7"`.
   3. In the same cell, right under `"R11": dict(use_sampler=True, in_channels=6),`, add this line:
      `    "R11s7": dict(use_sampler=True, in_channels=6, seed=7),               # R11, second seed`
   4. Session options: Accelerator **GPU T4 x2**, Internet **On**. The DisasterM3 mirror should already be listed as an input.
   5. Add-ons → Secrets → turn on `HF_TOKEN` (AbrarAlam, with write access). The run saves its checkpoints to `AbrarAlam/disasterm3-d4-R11s7` so it can resume.
   6. Smoke test: in Cell 2 set `SMOKE_TEST = True`, then Run All. If Cell 1 says "RESTART THE KERNEL NOW", restart and Run All again. Its config printout must show `seed 7` and `in_channels 6`, and it must write `D4_R11s7.csv`.
   7. Full run: set `SMOKE_TEST = False`, then Save Version → Save & Run All (Commit) → Save. If the time guard stops it, Run All again: it resumes from Hugging Face.
   8. When it's done: download the version's `d4_R11s7` output folder into `Champion_Task\R11s7-results\`, and the notebook as `Champion_Task\d4-champion-model-R11s7.ipynb`, then tell me.
4. **Your manual renames inside notebooks that already ran** (your decision, 20 Oct): every change, cell by cell, is in [Manual_Notebook_Renames.md](Manual_Notebook_Renames.md). The notebooks are R1s7 (was R6), L2 step 1, R16, R16b, R16c steps 1 and 2, and E3. Tell me when you're done, and I'll re-check by searching for the old names.

## Goals

| Goal | Target | Where we are | Status |
|---|---|---|---|
| O1 Segmentation champion | mIoU above 0.52 and Destroyed IoU above 0.30, on two seeds | mIoU 0.581 (R11). Destroyed IoU 0.300 on one seed (R9 with TTA); the two-seed mean is 0.297. R12's 8,021 extra destroyed buildings changed nothing (TTA: 0.588 and 0.295). E4: almost all of the gap is Turkey's earthquake (37% of destroyed pixels, Destroyed IoU 0.000). Without it, R11 reaches 0.410. **R16c (SAM 3 + the U-Net's pieces, the stronger grader) passes the gate on both U-Net seeds and both grader seeds:** with R11, 0.587 / 0.316 and 0.585 / 0.311 (step 2); with R11s7, 0.584 / 0.309 and 0.583 / 0.301 (step 3, 21 Oct); mean of the four 0.585 / 0.309. Caveats: the weakest combination clears 0.30 by only 0.0006, and against R11s7 the keep rule fails, because it costs about 0.007 mIoU. | **Met** (gate on two U-Net seeds; narrow on one) |
| O2 Counting and six tasks (KR2) | Building counting above the base model's 34.2%; all six tasks re-scored | Like for like with the DisasterM3 paper's Table 2, which scores optical after-images only: our 7B (greedy) averages 36.1% on the six tasks against the base model's 31.2%. It counts buildings at 32.0% against 34.2%, 2.2 points short. With the radar questions included, which the paper's table leaves out, our scores are 35.6% and 30.5%. The paper's own fine-tune of the same 7B reaches 40.4% (counting 34.3%). These are at 512 px. At the paper's full image size (L0 step 1), our 7B scores 75.6% on disaster type and 31.1% on relational (71.1% and 23.3% at 512 px); the other four tasks haven't been re-scored at full size yet. | Half met |
| O3 Offline report system | DisasTeller on our model, with damage map, counts and survivability heatmap | Damage map, counts and heatmap are done. DisasTeller on local models is not started (T2). | 3 of 4 parts |
| O4 Edge model and distilled 3B | An edge-ready segmenter and a distilled 3B model | Edge segmenter done: the INT8 version (L1b) is 4× smaller (33 MB) and 1.43× faster than ONNX FP32, with Destroyed IoU down only 0.0003. The 3B distillation's step 1 is done (1,085 checked teacher answers); step 2 is next. | Half met |

TTA means averaging the predictions of 8 flipped or rotated copies of each image.

## Tasks

### Phase A: diagnosis, no training

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| E1 | Error diagnosis and champion predictions | Done | Kaggle GPU | 59.5% of missed destroyed pixels are predicted as background, so R11 went first. Saved R9's predictions for H1 and T1. | — |
| E2 | Majority vote per building | Done | Kaggle GPU, inside E1 | Rejected: Destroyed IoU fell from 0.292 to 0.228. | — |
| D5 | xBD converter and the 1,007 extra tiles | Done | Your PC | 447 of the tiles are benchmark images, and they hold 3,466 of the 3,472 destroyed buildings. The other 560 tiles hold only 6. | Feeds R12. |
| D5b | xBD tier3 check and masks | Done | Your PC: the check took 2 s, the masks 2.3 min | Of tier3's 6,369 tiles, 2,635 are already in our segmentation data (5,786 destroyed buildings) and 612 are benchmark images (4,549 destroyed, in 227 tiles). The other 3,122 hold no destroyed building at all (2,455 outlines: 1,208 intact, 2 major damage, the rest un-classified). So no clean source of destroyed buildings exists: as in tier1, DisasterM3 put every damaged tile into its training data or its benchmark. Masks for the 612 exam tiles are made with [d5b-xbd-tier3-converter.ipynb](Champion_Task/d5b-xbd-tier3-converter.ipynb), outputs in `D5b-results/`. Its lookup table matches D5's, with 97.2% pixel agreement against DisasterM3's masks. | Upload with `D5-results/` for R12. |
| E3 | SAM 3 building finding | Done | Kaggle GPU | SAM 3 on the before-image finds buildings better than R9 (building IoU 0.630 against 0.591). Combined with R9, it covers 86% of destroyed buildings' footprints. | R16's finding. |
| E4 | R12 against R11, by image source and by event | Done | Kaggle account 2, T4, 18 min | Go given 19 Oct; evaluation only. Built 19 Oct: [e4-r12-vs-r11-by-source.ipynb](Champion_Task/e4-r12-vs-r11-by-source.ipynb). Its results cell was checked on synthetic data on your PC.

- **What it tests:** why R12's extra data didn't help.
  - "The wrong place": every extra tile is xBD, but the weak images are the 85 non-xBD ones (Destroyed IoU 0.071, holding 43% of the destroyed pixels; Turkey's earthquake alone 37%, with no extra tiles).
  - "The wrong problem": rubble called background.
- **How:** both models on the 645 validation images, scored per source and per event, with the extra tiles R12 got for each event.
- **Reading fixed before the run** (TTA): R12 beating R11 on xBD by more than 0.013 (the seed noise) but not on the others means "the wrong place". Otherwise it is not the wrong place.

**Result (19 Oct):** both models reproduced their CSVs. Results in [E4-results/](Champion_Task/E4-results/E4/README.md).
- **By source (TTA, Destroyed IoU):** xBD 0.456 → 0.463 (+0.007, within noise); other 0.071 → 0.069. By the rule, **not the wrong place**.
- **By event, though, the extra data helped where it went:** rank correlation +0.73 across the 13 larger xBD events. Palu got the most extra destroyed buildings (2,206) and gained +0.047; Moore +0.019, Woolsey +0.014.
- **Bigger finding** (read-only checks on your PC, from R16's saved R11 masks):
  - Turkey's earthquake holds 37% of the validation's destroyed pixels (660,440). R11 scores Destroyed IoU 0.000 there: it calls 91.5% of them intact and predicts destroyed on only 5 pixels. Training has 557 Turkey images with 3.75 million destroyed pixels, so it isn't a shortage of examples.
  - Without Turkey, R11's Destroyed IoU is **0.410** (mIoU 0.618). On Turkey alone it's 0.000.
  - The floods (Harvey, Midwest, Florence) score about 0: R11 calls their destroyed buildings damaged (52–80%).
  - R16's grader finds 72–83% of Turkey's destroyed pixels (Destroyed IoU 0.13, or 0.19 corrected, held down by over-grading). Comparing each building's before and after crops sees the collapses R11 misses. | Watch Turkey in R16c step 2. The O1 gate hinges on Turkey's destroyed buildings. |
| H1 V1 | U-Net counting for building counting | Done | Kaggle GPU | 22.3% overall, below the base model's 34.2% (gate G4 not met). It used R9, before R11. Optical after-images score 26.5%; radar 11.3%, below the 20% of guessing. The U-Net can't read radar, so it counted on the optical twin of the same tile, and a radar question's true count often differs from the twin's. Counting from the true masks scores 100%, so the masks are the problem. The answer counts separate buildings and the wrong options are only ±20% or ±40% away, while a U-Net merges touching houses and splits damaged ones (median count error 45% for intact, 75–83% for damaged and destroyed). | A better pixel model (R11) helps little. Count separate buildings instead, via R16 once its grader is corrected. |
| H1 V2 | Counts as text, all six tasks re-scored | Done | Kaggle account 2, T4 ×2, 4.3 h | Greedy answers beat our 499A scores on 4 of 5 tasks: disaster type 70.7%, bearing bodies 20.5%, road damage 31.9%, land use 37.9% (relational 22.1%, about the same). The U-Net fact line lowered every task, bearing bodies by 9.9 points, so it is not kept. Six-task mean 34.2% against the base model's 31.2%. On optical after-images only, the paper's setting, it's 36.1% with H1 V2b's counting (see O2). Results in [H1-V2-results/](Champion_Task/H1-V2-results/H1V2/README.md). | — |
| H1 V2b | Counting questions asked to our 7B itself (greedy, plain and with facts) | Done | Kaggle account 2, T4 ×2, 2.9 h | Building counting: 7B plain (greedy) 30.5% (32.0% on optical after-images, against the base model's 34.2% on the same kind), with the U-Net fact line 26.6%, against 23.6% for our 499A score, 22.3% for the U-Net counter and 34.2% for the base model. The fact line hurts again, except on destroyed buildings (30.6% against 28.4%). Answers lean to "D" (34% of picks), but 30.5% is well above the 20.8% that always picking one letter gets. Six-task mean with it: 35.6% (base 31.2%). Results in [H1-V2b-results/](Champion_Task/H1-V2b-results/H1V2b/README.md). | The counting half of KR2 is still not met. Next try: counts from R16's per-building grades (clean for the benchmark). |
| T1 | Survivability heatmap, v1 | Done | Kaggle CPU | All 20 check pictures reviewed: 36 of 48 circles on damage, 52 of 57 safe spots on the picture's edge. | Replaced by v2. |
| T1 v2 | Heatmap with three fixes | Done | Your PC | 39 of 46 circles on damage (85%). No safe spot on the edge; 4 of 57 on a building. Results in [T1-results/T1-v2/](Champion_Task/T1-results/T1-v2/README.md). | Replaced by v3. |
| T1 v3 | Heatmap with circles kept off the picture's edge | Done | Your PC | 20 checked pictures: 41 of 46 circles on damage (v2 39, v1 36 of 48) and 33 on destroyed buildings (v2 31); 1 circle on the edge (v2 12). All 645 pictures: circles on the edge 1% (v2 20%), on damage 73.6% (same as v2). Results in [T1-results/T1-v3/](Champion_Task/T1-results/T1-v3/README.md). | T1 is finished: v3 is the version T2 and T3 use. |
| L1 | Edge benchmark (ONNX, INT8) | Done | Kaggle CPU | ONNX FP32: 1.7× faster than PyTorch, no accuracy loss. Static INT8: 2.6× faster, but Destroyed IoU dropped 0.0104 (limit 0.01). | L1b retries INT8. |
| L1b | Static INT8, second try | Done | Kaggle CPU, about 1.5 h | Both versions pass: Destroyed IoU 0.2918 against 0.2921 (drop 0.0003, limit 0.01), mIoU 0.5363. Keeping the first conv and the last 3 convs in FP32 is the edge model: 1.6 s per image, 1.43× faster than ONNX FP32 in the same run, 33 MB. Results in [L1B-results/](Champion_Task/L1B-results/L1b/README.md). | Later: the same recipe on the final champion (R11's 6-channel input needs a small change). |

### Phase B: training

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| R11 | Before + after images (6 channels) | Done | Kaggle main, T4 | mIoU 0.581 (TTA 0.588), the biggest gain so far. Destroyed IoU 0.291 (TTA 0.297), unchanged. 6 of the 20 checked image pairs are shifted by more than 5 px. | Base for R12. |
| R12 | Add the extra xBD tiles (1,007 from part 1, 612 from tier3) | Done | Kaggle account 2, T4, 6.4 h of training | Rebuilt 19 Oct on R11's 6-channel recipe with both pools: 1,619 tiles, 8,021 destroyed buildings: [d4-champion-model-R12.ipynb](Champion_Task/d4-champion-model-R12.ipynb). The tiles join training only, label 255 (un-classified) is ignored by both losses (checked in Cell 7), and it stays at 5,798 samples per epoch. Images come from Tunguz's two xView2 copies on Kaggle, so only the masks (D5-results and D5b-results) need uploading. 1,059 of the tiles are benchmark images, so R12 never answers benchmark questions; R16 does. Other sources (checked 19 Oct): BRIGHT's after-images are radar and its optical ones are already in DisasterM3; in xBD's first part, the tiles DisasterM3 skipped held only 6 of 13,227 destroyed buildings; tier3 has no clean destroyed buildings either (D5b), so its 612 benchmark tiles were added (option A).

**First run (19 Oct, account 2): stopped after 6 minutes in Cell 4b, before training.**
- It looked for each pool's photos by the first tile's file name. The uploaded masks have the same name and sort first, so it took the masks folder for the photos and found no before-images: "1619 extra tiles have no mask or image".
- All four inputs were attached; the bug is mine.
- The fix is one line: keep a folder only if it also holds the tile's before-image.

Its log (`d4-champion-model-r12.log`) is no longer in `Champion_Task`.

**Second run (19 Oct), with the fix: not kept.**
- Both pools found their photos in Tunguz's copies, and it trained on 7,417 images (1,619 extra).
- mIoU 0.582 (TTA 0.588) and Destroyed IoU 0.290 (TTA 0.295), within noise of R11 (0.581 and 0.588; 0.291 and 0.297).
- 8,021 more destroyed buildings (DisasterM3's training images hold about 18,000 destroyed pieces) didn't move Destroyed IoU, so more examples of that class isn't the lever.
- R12 trained on 1,059 benchmark tiles, so it must never answer benchmark questions.

Results in [R12-results/](Champion_Task/R12-results/d4_R12/D4_R12.csv). | Closed: R11 stays the champion recipe. Its second seed is R11s7 (your decision, 19 Oct). |
| R11s7 | R11 with seed 7 (second seed) | Done | Kaggle account 2, 20 Oct, 6.8 h | 0.5867 / 0.2869, or 0.5913 / 0.2928 with TTA (R11: 0.5875 / 0.2965). Clean: seed 7 and 6 channels in its config, 40 continuous epochs, the last epoch matches the CSV. The two seeds' TTA mean is 0.5894 / 0.2947, so the U-Net alone misses O1's 0.30 on two seeds. Results in [R11s7-results/](Champion_Task/R11s7-results/d4_R11s7/D4_R11s7.csv). | R16c step 3 uses its pieces for O1's second seed. |
| R13 | BRIGHT pre-disaster footprints | Not started | — | Overlap checked 19 Oct from DisasterM3's train and benchmark lists: all 14 BRIGHT events are in DisasterM3. 7 have optical after-images there (Turkey, Morocco, Libya, Beirut, Bata and Hawaii in our training data; Noto in the benchmark), so R13 skips them. The other 7 appear only as radar questions, which our optical U-Net can't answer, so their before-images are safe to train on. | After R12. Ridita's converted set includes Noto: drop it. A BRIGHT tile has no optical after-image, so with the 6-channel recipe decide how to fill the after slot (roadmap R13). |
| R14 | Rubble-centred crops (optional) | Not started | — | — | Only with spare GPU time. |

### Phase C: champion and fallback

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| R15 | Best combination plus a second seed (R15s7) | Not started | — | R15 is a new recipe: R11 plus whatever else passes the keep rule (seed 42), with R15s7 as its second seed. Nothing beyond R11 has passed (R12 added nothing; R13 and R14 haven't run), so R15 would be R11 itself. R11's second seed therefore runs as R11s7 (Phase B), and R16c's kept map goes on top of both seeds. The gate needs both seeds, or their mean, above 0.30. | Only if a change beyond R11 passes. |
| R16 | Find-then-grade: find buildings, then grade each | Done | Kaggle account 1, T4, 29 min | Runs regardless of R12 (decided 19 Oct). Built 19 Oct: [r16-find-then-grade.ipynb](Champion_Task/r16-find-then-grade.ipynb) (named `r16-two-stage-grader` until 20 Oct, here and on Kaggle). The finding: SAM 3 on the before-image (E3's `pre_grid2`) plus R11's building pieces that SAM 3 missed. The grading: a Siamese ResNet-18 grades each building from 96 px before and after crops. It scores R16 alone, R16 with R11's pieces, and an R16/R11 vote against R11 (re-run inside and checked against `D4_R11.csv`), plus the grader on the true buildings (macro F1 against BDAChat's 0.763). Clean for the benchmark. Logic checked on a synthetic image on your PC. **Result (19 Oct), not kept:**
- R11 reproduced its CSV, which confirms the pipeline.
- Destroyed IoU: SAM 3 + grader 0.167, with R11's pieces 0.152, vote 0.261. All are below R11's 0.297 (TTA); the vote's mIoU (0.580) is close to R11's 0.588.
- On the 85 non-xBD images, SAM 3 + grader doubles R11's Destroyed IoU (0.128 against 0.071).
- The grader on the true buildings: macro F1 0.656 (BDAChat 0.763). Destroyed recall is 0.89 but precision 0.35, because balanced training over-grades intact buildings (1,924 of 19,624 graded destroyed).

Results in [R16-results/](Champion_Task/R16-results/R16/README.md). | Grader corrected in R16b (next row). Still the O1 fallback. |
| R16b | R16's grader corrected by the real grade shares | Done | Your PC, 7.3 min | [r16b-grader-prior-correction.ipynb](Champion_Task/r16b-grader-prior-correction.ipynb). Each grade's probability is multiplied by its share among the training buildings and divided by a third; the shares come from the training masks (intact 73.7%, damaged 18.4%, destroyed 8.0%). Nothing is fitted on validation, and R16's scores were first rebuilt exactly from its per-building table. **Result (19 Oct), not kept:**
- Every R16 map improves:
  - SAM 3 + grader: mIoU 0.471 → 0.515, Destroyed IoU 0.167 → 0.228;
  - the vote: 0.580 → 0.585 and 0.261 → 0.274.
- R11 with TTA stays ahead (0.588 and 0.297).
- The over-grading is mostly gone. Among the found buildings that are mostly one true building:
  - intact buildings graded destroyed fall from 3,539 to 1,500;
  - macro F1 rises from 0.597 to 0.692.
- On the 85 non-xBD images, SAM 3 + R11's pieces reach Destroyed IoU 0.187, against R11's 0.071.

Results in [R16b-results/](Champion_Task/R16b-results/README.md). | Counting with the corrected grades (H1 V3) needs SAM 3 and the grader run on the benchmark images. |
| R16c | A stronger R16 grader, tested rigorously | Running (step 5 on account 1 since about 12:10, 20 Oct; steps 1, 2 and 4 done, step 2 kept) | Step 1 your PC (2.6 min); step 2 Kaggle account 1, T4, 1.4 h; steps 4 and 5 account 1, about 5–6 h; step 3 account 1, about 0.5 h | Go given 19 Oct. Steps renumbered 1–5 on 20 Oct (your decision): step 3 was 2b, step 4 was 3a, step 5 was 3b.

**Step 1 done** (19 Oct): [r16c-step1-ceiling-and-counting.ipynb](Champion_Task/r16c-step1-ceiling-and-counting.ipynb), results in [R16c-results/step1/](Champion_Task/R16c-results/step1/README.md).
- The ceiling: give each found building its main true grade. SAM 3 + R11's pieces would then reach mIoU 0.766 and Destroyed IoU 0.614 (0.798 and 0.661 if the grader may also reject non-buildings). Our corrected grader gets 0.541 and 0.208 on the same outlines, so the grader loses most of the score.
- Counting, on the 1,055 validation counting questions (the true masks answer 100%):
  - R11's blobs: 39.2% (37.2% counting every blob);
  - R16's best map: 36.9%;
  - a perfect grader on SAM 3's outlines: 55.3%.

  Our 7B can't be scored there, because it trained on these images. Validation numbers also run high: R9's counter scored about 33% here but 26.5% on the benchmark.

**Steps 2, 4 and 5 built** (19 Oct; step 5 got its own notebook on 20 Oct), with their logic checked on a synthetic image on your PC:
- **Step 2:** [r16c-step2-stronger-grader.ipynb](Champion_Task/r16c-step2-stronger-grader.ipynb), a copy of R16's notebook.
  - 128 px crops with a 32 px margin, 20 epochs, two seeds.
  - The epoch and a bias per grade are picked on 580 held-out training images.
  - The same finding and maps as R16, scored against the ceiling.
- **Step 4** (was 3a): [r16c-step4-sam3-training-images.ipynb](Champion_Task/r16c-step4-sam3-training-images.ipynb), a copy of E3's, runs SAM 3 on the training images (E3 took 2.1 s per image).
- **Step 5** (was 3b): [r16c-step5-grader-on-sam3-outlines.ipynb](Champion_Task/r16c-step5-grader-on-sam3-outlines.ipynb), step 2's notebook with `TRAIN_OUTLINES = "sam3"` already set: the grader trains on SAM 3's own outlines, with a 4th label, "None", for outlines that are mostly background. A diff against step 2's code shows only the intended changes, and it prints the check against step 2.

**Step 3 built** (20 Oct, was 2b): [r16c-step3-r11s7-pieces.ipynb](Champion_Task/r16c-step3-r11s7-pieces.ipynb), step 2's graders and biases with R11s7's pieces, evaluation only.

**Step 2 done (20 Oct, account 1, T4, 1.4 h): kept on both grader seeds, and the O1 gate passes on both.** Results in [R16c-results/step2/](Champion_Task/R16c-results/step2/README.md) (Kaggle's output folder name was `R16c-truth`; renamed locally on 20 Oct).

- SAM 3 + R11's pieces, against R11 with TTA (0.588 / 0.297):
  - seed 42: mIoU 0.587, Destroyed IoU **0.316** (+0.019);
  - seed 43: mIoU 0.585, Destroyed IoU **0.311** (+0.014).
- The other maps aren't kept: SAM 3 only reaches 0.299 and 0.297; the vote 0.286 and 0.273.
- The grader on the true buildings scores macro F1 0.778 and 0.784, against R16's 0.656 and BDAChat's 0.763. Both seeds picked epoch 6, with biases of about −1.9 (damaged) and −3.9 (destroyed) on the held-out training images.
- It closes 25–27% of the gap from R16b to the ceiling (0.614).
- Where it wins and loses (read-only checks on your PC, from its per-building table):
  - Turkey: 0.000 → 0.244 and 0.274;
  - xBD images: 0.456 → 0.383 and 0.346;
  - everything except Turkey: 0.410 → 0.353 and 0.327.
- Counting on the 1,055 validation questions: 36.8–39.1%, against 39.2% for R11's blobs and about 50% with a perfect grader. So counting from it on the benchmark (H1 V3) isn't worth GPU time yet.
- The two seeds are the grader's. R11 (seed 42) is the same in both. | Decided 20 Oct: steps 4 and 5 run now on account 1, as a stretch (step 5 replaces step 2 only if it beats it by the keep margin on both seeds). Step 3, the same graders with R11s7's pieces, follows R11s7. See "Needs you now". |

### System track

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| H2 | Per-building table and SQL counting agent | Ready to run | Kaggle T4 ×2, account 2 | Noman built it (never run); the run copy [h2-sql-counting-agent.ipynb](Champion_Task/h2-sql-counting-agent.ipynb) uses R11 with 8-view TTA and a 20 px minimum (39.24% on validation) instead of R9, and measures R11's counting on all 4,982 benchmark BDC questions. | Smoke test, then the full run. Its table feeds T2 and T3. |
| H2b | Graded test of the SQL agent | Ready to run (after H2) | Kaggle T4 ×2, about 1–1.5 h | Built 22 Oct and checked end to end on your PC with stand-in models. It covers 297 items on 5 versions of H2's table, and 4 models in 4-bit and fp16. The decision rule was fixed before any run. | Run once H2's `buildings.sqlite` exists, then again on the distilled 3B after L2. |
| H3 | VLM input upgrades (before + after, numbered marks, crops) | Not started | Kaggle GPU | — | After H1 V2. |
| H4 | GEOBench-VLM external counting test | Not started | Kaggle GPU | — | — |
| T2 | DisasTeller on local models | Not started | Kaggle T4 ×2 | — | The biggest missing O3 piece. |
| T3 | Offline demo app | Not started | — | — | After H1, T1 and T2. |
| L0 | A better 7B teacher before distilling | Needs rerun (step 2: the old 7B's full-size re-score and the training's smoke test done 20 Oct. The full training: session 1 did steps 1–15 (21 Oct); session 2 crashed at step 16 when the paged optimizer resumed. Fixed with `adamw_bnb_8bit`; session 2 runs again on account 3, then session 3 and the merge) | Step 1: Kaggle account 1, T4 ×2, 1.7 h. Step 2: the old 7B's re-score of four tasks (about 11 h, account 3 then account 2), training (about 7–14 h on T4 ×2, account 3), then the new 7B on six tasks (about 12 h) | Decided 19 Oct: improve the 7B before distilling.

**Step 1 done (19 Oct):** [l0-step1-image-size-check.ipynb](Champion_Task/l0-step1-image-size-check.ipynb), results in [L0-step1-results/](Champion_Task/L0-step1-results/L0S1/README.md). The 1,211 optical questions, at three sizes for a 1024 px image (disaster type, 353 questions / relational, 858):
- 504 px, H1 V2's size, as the control: 71.1% / 23.3%;
- 616 px, our training cap: 75.4% / 26.9%;
- 1036 px, full size, the paper's: **75.6% / 31.1%**;
- for comparison: the paper's fine-tune 83.6% / 36.2%, the base model 66.6% / 23.9%.

What it shows:
- The control reproduced H1 V2 exactly, and every size reached the model as planned.
- Verdict by the rule fixed before the run (within 3 points of the paper closes the gap; half of each gap recovered closes it partly): **not closed**. Full size recovers 36% of the disaster-type gap and 61% of the relational gap.
- Full size becomes the 7B's test setting either way.

Background:
- Our recipe already copies the paper's (Appendix B.3): LoRA rank 64, alpha 16, lr 2e-4, global batch 256, 1 epoch, vision encoder frozen. Our run finished all 217 steps.
- The differences:
  - 4-bit QLoRA in fp16 on one T4;
  - images capped at about 512 tokens in training;
  - tested at 512 px (about 324 tokens), where the paper used full size (about 1,300).
- The answer format isn't the cause: our 7B answers with letters only on 99.9–100% of questions.
- Gaps to the paper's fine-tune at 512 px were −12.5 (disaster type) and −12.9 (relational); at full size they are −8.0 and −5.1. | Decided 19 Oct: step 2, a short extra training of our adapter at full size (1036 px) on disaster type and relational plus an equal replay sample (about 7,000 samples). Both models are then scored on every optical question. Its notebooks are being built. |
| L2 step 1 | Teacher answers for the 7B → 3B distillation | Done | Kaggle account 1, T4 ×2 | 4,000 answers generated; 1,085 (27%) match their label and are kept: disaster type 89.4%, land use 50.9%, bearing bodies 42.7%, building counting 15.6%, relational 13.3%, road counting 0%. Road counting labels are road-area percentages such as 5.22%, which the teacher can't hit exactly without the options; those 735 fall back to their labels. Cell 7's push failed: the dataset title already existed, and the code missed Kaggle's wording ("already in use") but still printed "Pushed". So the dataset `disasterm3-teacher-cot` doesn't hold this run's file. | If L0 changes the 7B, re-run it with the new teacher. Step 2 reads the manifest from the run's notebook output, not from that dataset. |
| L2b | Explanations as a second skill ("why?" on request) | Not started | Kaggle T4 ×2, about 3–5 h | Planned 19 Oct (roadmap L2b, Paper_Audit §4.12): explanations written from the correct answer plus true mask facts, checked against those facts, then trained as a second task in step 2. | Build after step 1. |
| L2 step 2 | Train the 3B | Not started | Kaggle GPU | Its notebook still has the old installs that fail on today's Kaggle. | Fix it, as a copy, after L0. How the 3B learns is still to choose (a, b or c; soft labels recommended). |
| L3 | 4-bit 3B with handoff to the 7B | Not started | Kaggle GPU | — | After L2. |
| L4 | EfficientSAM3 against SAM 3 | Not started | Kaggle GPU | — | — |

### Housekeeping

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| — | Commit this week's work to git | Done | Your PC, pushed 19 Oct as `1b33574` | `Essential_task_delegation\` and `others\`, plus `.gitignore`, `README.md` and `UPDATE.md` whenever they're modified (your rule, 19 Oct). The `.gitignore` was updated 19 Oct, so these stay local: bulk masks and maps that no notebook reads, L2 step 1's 71 MB training file, personal samples, the next report's draft and Claude's local settings. | Next commit: same scope, when you ask. |
| — | Update the roadmap and README with these results | Not started | — | — | Say when. |

## Change log

- **22 Oct 2026, 01:20:** H2b built in `Champion_Task/` and added to the roadmap (§7, after H2). The local check's scripts are in Claude's scratchpad, not the project.
- **22 Oct 2026:** the 8 papers H2b cites were added to `others/lit review_research papers/` as PDFs named by their titles:
  - Zhong et al. (EMNLP 2020) and DAIL-SQL (VLDB 2024), in their published versions;
  - TrustSQL, EHRSQL 2024, Dr.Spider, Rajkumar et al., Self-Debugging and Qwen2.5-Coder, from arXiv.

  Each was checked against its first page (title and authors). The folder is git-ignored, so they need `git add -f`.

- **21 Oct 2026, 21:01:** L0 training session 2 failed (account 3, committed about 19:42, crashed 20:41).
  - The log is `Champion_Task/l0-step2-topup-training.log`; the notebook with its outputs is `l0-step2-topup-training.ipynb`.
  - Restore 111 s, data 16 min, model load 1 min. Step 16's forward and backward passes ran (about 39 min); its optimizer update failed on both ranks in bitsandbytes' `cprefetch`.
  - The paged-flag cause was checked on your PC: `torch.save` and the optimizer's reload both keep `is_paged`.
  - Fixed Cell 2 (whole cell): `optim = "adamw_bnb_8bit"`, `time_limit_h = 5.0`.

- **21 Oct 2026, 20:25:** after the audit of Tamanna's and Noman's pull.
  - **Restored**, byte for byte from commit `1b33574`, because the pull had deleted them: Tamanna's `Plan_1/` (3 files) and `Plan_3/` plan, and `others/lit review_research papers/DisasterVQA.pdf`. The PDF is git-ignored by the folder rule, so it needs `git add -f` to stay tracked.
  - **H2 run copy built** in `Champion_Task/`: R11 with 8-view TTA instead of R9, a 20 px minimum for every class, `dtype=` instead of `torch_dtype=`, and the comparison with H1 V1 kept for information only.

- **21 Oct 2026, 20:05:** cleanup of `Champion_Task`, by your decision. It went from 13.4 GB to 3.5 GB.
  - **Deleted:**
    - `last.pth` and `best_by_val.pth` in all 13 U-Net run folders (26 files, 7.0 GB). All runs are finished; `final_model.pth` stays in each folder.
    - L0 session 1's local `checkpoints/` (checkpoint-12 and checkpoint-15, 1.95 GB). Both are on Hugging Face (`AbrarAlam/disasterm3-qwen25vl7b-qlora/l0s2/checkpoints`), confirmed by the run's push lines.
  - **`.gitignore`:** new rule `Essential_task_delegation/Champion_Task/**/checkpoints/`. Before it, L0's checkpoints weren't ignored.
  - **Standing rule from now on:** once a run's log confirms its checkpoints were pushed to Hugging Face, the local copies are deleted at once and kept git-ignored.
  - **You deleted** `R2-results/_output_.zip` (0.9 GB, an exact copy of `_output_/`). That's included in the 3.5 GB.
  - **Not deleted (not decided):** `R1(repeat)-results.zip`, R2's 114,773 cut-outs, step 4's masks, superseded masks and maps, the D5 zips, the duplicate `.txt` logs.

- **21 Oct 2026, 01:45:** your downloaded logs.
  - **L0 training, Version #7 (account 3), committed about 17:50 on 20 Oct:**
    - data took 1,861 s (6.7 min in the 13:12 run);
    - step 1 done at 4,368 s: 2,448 s per step, loss 0.5635;
    - out of memory in step 2's backward pass at 4,518 s;
    - the `StepOnCpu` fix removed the step-1 crash; the remaining gap is about 40 MiB.
    - **Fix:** `paged_adamw_8bit` in Cells 2 and 3; `time_limit_h` back to 10.0 from your 10.6.
  - **R16c step 5 (account 1):**
    - "Unique images" took 3,524 s and "Built 6,443 pairs" 29,326 s;
    - stopped in Cell 6 with account 1 at 0 GPU hours; no outputs.
  - The `.txt` files you downloaded are byte-identical copies of the `.log` files.

- **20 Oct 2026, 17:30:** L0 step 2's training, session 1, failed. Your log: `Champion_Task/l0-step2-topup-training.log`, with the failed run in `l0-step2-topup-training.ipynb`.
  - **What failed:**
    - setup went fine: data 7,018 examples, model 6.5 GB on GPU 0;
    - at the first step, both T4s ran out of GPU memory. GPU 0 asked for 832 MiB with 13.95 GiB already allocated of 14.56 GiB; GPU 1 asked for 220 MiB in the backward pass;
    - Cell 5's assert then failed, as the subagent had predicted for a crash.
  - **Cause, checked in the installed `transformers` (5.12.1) and `accelerate` (1.14.0) code:**
    - `Trainer._run_epoch` calls `get_batch_samples` for all 128 micro-batches of a step;
    - accelerate's data loader sends each one to the GPU (`send_to_device`);
    - full-size images are about 26 MB each in float32, and some examples have two.
  - **Fix:**
    - a `Trainer` subclass whose `get_batch_samples` moves each batch to the CPU as it's collected, with `pixel_values` in fp16, which the model casts them to anyway (`pixel_values.type(self.visual.dtype)`);
    - `training_step` moves each batch back to the GPU in turn.
  - **Tested on your PC:** `l0s2_cpu_batches_check.py` in the scratchpad confirmed CPU batches, 16 of 16 micro-batches trained, 2 steps of 8, and loss tokens counted.
  - L0 marked "Needs rerun".

- **20 Oct 2026, 13:15:** your decision: L0 step 2's training keeps the equal replay (A, 7,018 examples, about 19.5 h). L0 marked "Ready to run", with session 1 on account 3.
  - The notebook checks the time limit at each step's end (about 43 min per step), so a session can overrun `time_limit_h` by up to about 45 min. With about 20 min of setup and Kaggle's 12 h cap, the limit stays at 10.5 h or below.
  - At the measured speed, 10.0 and 10.5 both stop after step 15.

- **20 Oct 2026, 13:05:** the training's smoke test done on account 3 (`l0-step2-topup-training.ipynb` now holds it).
  - **Checks:**
    - versions as planned;
    - config pushed to Hugging Face, so the account's token can write;
    - 2 of 2 steps on both T4s, losses 0.844 → 0.461;
    - peak memory 14.7 GB on each GPU, including the longest example (3,092 tokens);
    - no merge, as intended.
  - **Projection:** 10.0 s per example, so 19.5 h for 7,018 examples, above the 7–14 h estimate. The tracker's rule for this case: you choose two sessions as planned, or a smaller replay.
  - L0 marked "Needs your decision".

- **20 Oct 2026, 12:40:** the old 7B's full-size re-score (L0 step 2, parts A and B) and R16c step 4 done and checked; step 5 and the training's smoke test running.
  - **Parts A and B:**
    - settings as planned (`MODEL = "old"`, each part's tasks, `SMOKE_TEST = False`), every image found, no errors, and neither stopped at the time limit;
    - every accuracy recounted from the per-question answers;
    - 8 unreadable images (BDC 3, BBR 2, DRE 2, DSR 1) are scored wrong, the same 8 that had no answer at 512 px.
  - **Against the expectation fixed before the run** ("the tasks move by a few points at most from 512 px; counting has the most to gain"): the four tasks moved −1.9 to +2.2, as expected. Counting fell 1.9 points instead of gaining.
  - **Run times** were longer than estimated: part A 8.6 h (7.3), part B 4.1 h (3.7). The new 7B's six-task evaluation is re-estimated at about 14.5 h.
  - **R16c step 4:** complete, 5,798 of 5,798 maps, 4.6 h (estimated 3.5–4; 2.6 s per image against 2.1 on validation).
    - Building IoU 0.629 on the training images, against 0.630 on validation (E3); destroyed footprint recall 0.693.
    - pip's numpy warning for `sam3` didn't stop it.
  - **Where the results are:** [L0-step2-results/old-partA/](Champion_Task/L0-step2-results/old-partA/L0S2/README.md), [old-partB/](Champion_Task/L0-step2-results/old-partB/L0S2/README.md) and [R16c-results/step4/](Champion_Task/R16c-results/step4/README.md).
    - You saved them as `L0-step2-partA-results/`, `L0-step2-partB-results/` and `R16c-step4-results/`. They were moved to the existing pattern the same day (your "fix on the go").
    - The run notebooks were renamed to `l0-step2-full-size-rescore-old-partA.ipynb` and `-old-partB.ipynb`, so the new 7B's runs can be `-new-…`.
    - Step 2's `__huggingface_repos__.json` moved from `R16c-results/` into `step2/`, beside step 4's own copy in `step4/`.
    - `r16c-step4-sam3-training-images.ipynb` now holds its run.
  - **Git:** step 4's 5,798 masks (84 MB) are now in `.gitignore`, by your decision. Only step 5 reads them, from the Kaggle output. The step's README, CSVs and JSONs are still committed.

- **20 Oct 2026, 08:20:** from your screenshots (08:18).
  - Running: R16c step 4 on account 1 (`abrarmohammedtanzim`), part A of the old 7B's re-score on account 3 (`redwannewaz`), part B on account 2 (`abrar2222864642`). R16c and L0 marked Running.
  - Part A's and part B's logs match their smoke-test checks: every image found (4,334 and 4,300), prompts checked, full size (1036 px).
  - Step 4's log shows pip's numpy warning for `sam3`; the run carried on past it, so it isn't an error.
  - Renamed on Kaggle (account 1): `l2-step1-teacher-rationales`, `r16-find-then-grade`. The notes that said "still so on Kaggle" are updated.
  - Account 3 (`redwannewaz`) also holds earlier notebooks (`D4_champion_model`, a `d4-champion-model-R…` one, and `d4-champion-model-r4-resume`), so some early D4 runs were there. Account 2's list shows `d4-champion-model-R9`.

- **20 Oct 2026, 07:46:** R11s7 done on account 2 (6.8 h) and checked.
  - Its config says seed 7 and 6 channels; its history has 40 continuous epochs; the last epoch matches `D4_R11s7.csv`; no errors in the log.
  - 0.5867 / 0.2869, or 0.5913 / 0.2928 with TTA. Against R11 with TTA: mIoU +0.004, Destroyed IoU −0.004.
  - The U-Net alone misses O1 on two seeds (TTA mean 0.2947). O1's second seed now rests on R16c step 3.
  - Step 3's run steps now include filling `EXPECTED_UNET` in Cell 2 from `D4_R11s7.csv`, so its first sanity check runs.
  - Account 2 is free: part B of the old 7B's re-score is ready to run there.

- **20 Oct 2026, 05:22:** the renames, carried into the other docs and the decks.
  - **R6 → R1s7, with a note**, in `Ablation_Study_Reference_and_QA_Prep.md`, `Phase2_Plan.md`, `Research_Directions_Beyond_Finetuning.md`, `Segmentation_Model_Explained.md`, `BRIGHT_Integration_Plan.md` and the next report's draft `CSE499B_Review_report_update.tex`.
  - **R16's "stage 1/2" → "the finding" and "the grading"** (the model is "find-then-grade") in `Phase2_Plan.md`, `Research_Directions_Beyond_Finetuning.md`, `Paper_Audit_and_Integration_Plan.md`, `Ablation_Study_Reference_and_QA_Prep.md` and the .tex draft.
  - **Decks:**
    - "Segmentation Ablation Study": R6 → R1s7 on 7 slides, with "named R6 at the review" where it's introduced, and a line on the cover; "two-stage" → "find-then-grade" on 2 slides.
    - "CSE499B Review Presentation" had no R6. Its timeline row now says "find-then-grade".
    - The exports in `others\` are unchanged.
  - **Older change-log entries** now give the new step numbers in brackets.
  - **Left as written:** Ridita's BRIGHT roadmap ("Stage 1/2" for her own cascade idea), September's `team_task_assignments.md`, the literature guide `segmentation_improvement_guide.md`, results READMEs, and O1's name "Strengthen the spatial stage".
  - **Link check:** every link in the ten edited docs resolves.
- **20 Oct 2026, 04:55:** renames, by your decisions.
  - **R6 → R1s7.**
    - The files: `d4-champion-model-R6.ipynb` → `d4-champion-model-R1s7.ipynb`; `R6-results/d4_R6/D4_R6.csv` → `R1s7-results/d4_R1s7/D4_R1s7.csv`.
    - The CSV's labels and `config.json` now say R1s7, with a note that it was R6. The scores are unchanged.
  - **L2:** `l2-stage1-teacher-rationales.ipynb` → `l2-step1-teacher-rationales.ipynb`; `L2-stage1-results/` → `L2-step1-results/`. `.gitignore` follows, so the 71 MB training file stays out of git (checked).
  - **R16:** `r16-two-stage-grader.ipynb` → `r16-find-then-grade.ipynb`.
  - **R16c's steps renumbered 1–5.**
    - Notebooks: `r16c-step2b-r11s7-pieces.ipynb` → `r16c-step3-r11s7-pieces.ipynb`; `r16c-step3a-sam3-training-images.ipynb` → `r16c-step4-sam3-training-images.ipynb`.
    - Step 5 has a new notebook of its own, `r16c-step5-grader-on-sam3-outlines.ipynb`. A diff against step 2 shows only the intended changes.
    - Step 2's results: `R16c-results/R16c-truth/` → `R16c-results/step2/`.
  - **Inside the notebooks:**
    - The three that haven't run were edited: titles, output folders `step3`, `step4` and `step5`, and finding/grading wording. The new tables' column is `found_by`.
    - The seven that ran keep their outputs. You make their changes by hand from [Manual_Notebook_Renames.md](Manual_Notebook_Renames.md).
  - **Fixed in the run steps:** Kaggle has no single-T4 option, so every step now says "GPU T4 x2". Your screenshot showed step 2 ran that way.
  - **R9 and R10:** the run versions of the notebooks are now in the folder (you downloaded them). Their printed scores match `D4_R9.csv` and `D4_R10.csv`.
  - **Left as they are:**
    - Kaggle's names of runs already done, the Hugging Face repo `disasterm3-d4-R6`, and saved outputs and result READMEs (records of what ran);
    - "two-stage" where older docs cite the literature, such as the xView2 winners' pipelines;
    - `R1(repeat)-results.zip`, as you asked.
- **20 Oct 2026, 02:40:** audit of `Champion_Task` (your request).
  - **Results match the docs:**
    - the D4 CSVs of R1–R12, R9s7 and E0;
    - E1/E2, E3 (and its 86% coverage, recomputed from the saved masks), E4;
    - H1 V1/V2/V2b, L0 step 1, L1/L1b, L2 step 1;
    - R16, R16b, R16c steps 1–2, T1 v3, D5/D5b.
  - **Every notebook that ran has its folder.** Step 2 ran on T4 ×2 for 1.37 h.
  - **Fixed in the docs:**
    - R12's next step said "second seed is R15": it is R11s7.
    - The R15 row said "Running (R11s7)": R15 is "Not started", and R11s7 has its own row.
    - The roadmap's R15 section and ID legend now say the same.
    - R16c went from "Running" to "Ready to run".
    - The R12 log link pointed to a file that is no longer in the folder.
  - **Not changed (suggestions, your call):**
    - `d4-champion-model-R9.ipynb` and `-R10.ipynb` have no outputs and carry R5's run record (6.54 h, ended 4 Oct 23:52). They are edited copies of R5's file, not the runs. Download the run versions from Kaggle if you want them, like R1–R8.
    - `R1(repeat)-results.zip` (685 MB) has no ID anywhere in the docs.
    - R6 is R1 with seed 7 (R1s7 by today's rule).
    - Folder names don't follow one pattern:
      - `R2-results/_output_/d4_R2` against `Rn-results/d4_Rn`;
      - `L1B-results/L1b`;
      - `H1-V1-results/H1` against `H1V2` and `H1V2b`;
      - L2 "step 1" against `l2-stage1` / `L2-stage1-results`;
      - `d4-champion-model-r4-resume.ipynb` in lower case;
      - R16c's `step1` against Kaggle's `R16c-truth`; later `R16c-sam3`, `R16c-step2b`, `R16c-step3a`.
- **20 Oct 2026, 01:40 (later):**
  - Your decisions: R16c step 3 (now steps 4 and 5) runs now on account 1, as a stretch; step 2b (now step 3) gets built for R11s7's pieces; account 3 starts with the old 7B's full-size re-score, then L0 step 2's training. Account 2 joins the re-score after R11s7.
  - Fixed before step 3's run (now step 5's rule): it replaces step 2 only if it beats step 2's kept map on both seeds by the keep margin.
  - R16c marked Running (step 3a, now step 4).
  - L0 step 2's re-score notebook built ([l0-step2-full-size-rescore.ipynb](Champion_Task/l0-step2-full-size-rescore.ipynb)) and dry-run on your PC; L0 marked Ready to run. Part A goes to account 3 now, part B to account 2 after R11s7.
  - L0 step 2's training notebook built ([l0-step2-topup-training.ipynb](Champion_Task/l0-step2-topup-training.ipynb)); its logic was checked on your PC on the real manifest. It trains a new adapter on the merged 7B, because the old adapter's layer pattern doesn't load in transformers 5. Its smoke test goes on the first free T4 ×2 account.
  - R16c step 2b built ([r16c-step3-r11s7-pieces.ipynb](Champion_Task/r16c-step3-r11s7-pieces.ipynb), renamed step 3 on 20 Oct): step 2's graders with R11s7's pieces, on account 1 after R11s7 and step 3. R11s7's reference numbers go into its Cell 2 when R11s7's results arrive.
- **20 Oct 2026, 01:40:**
  - R16c step 2 done on account 1 (1.4 h): kept on both grader seeds, and the O1 gate passes on both (Destroyed IoU 0.316 and 0.311, mIoU 0.587 and 0.585).
  - It wins on Turkey (0.000 → 0.24–0.27) and loses on xBD images (0.456 → 0.35–0.38). Counting stays at R11's level (37–39%).
  - R11s7 is running on account 2. A third account with 30 GPU hours is available.
  - Your decisions are needed on step 2b (now step 3; R11s7's pieces), step 3 (now steps 4 and 5), and account 3 for L0 step 2.
- **19 Oct 2026, 22:10:**
  - E4 done on account 2 (18 min). By the rule fixed before the run, "not the wrong place": R12 gains +0.007 on xBD images, within noise. But per event the gain follows the extra data (rank correlation +0.73; Palu +0.047).
  - Read-only checks on your PC found the real cause of O1's gap. Turkey's earthquake holds 37% of the destroyed pixels and R11 scores 0.000 there (91.5% called intact); without Turkey, R11 reaches 0.410. R16's grader finds 72–83% of Turkey's destroyed pixels.
  - Account 2 is free for R11s7.
- **19 Oct 2026, 21:40:**
  - Your decisions: L0 goes to step 2 (full-size training, judged on every optical question for both models); R11s7 runs on account 2 after E4; the thorough option is the default recommendation, but you decide.
  - [Task_Ledger.md](Task_Ledger.md) written: every finished task with its what, why, how, run, expectation, result and what it led to, plus your decisions so far.
  - GPU: account 1 10 h, account 2 14 h 50 min.
- **19 Oct 2026, 21:00:** E4 is running on account 2. Account 1 is free, so R16c step 2 can run there now.
- **19 Oct 2026, 20:45:**
  - E4 built (go given): an evaluation-only run that scores R12 against R11 per image source and per event, to test why R12's extra data didn't help. Ready to run on account 2, about 20–30 min.
  - R11's second seed is now named R11s7, after the naming rule (same recipe, another seed: name plus `s7`).
- **19 Oct 2026, 20:10:**
  - R12 done on account 2 with the fix (6.4 h), not kept: with TTA, mIoU 0.588 and Destroyed IoU 0.295, the same as R11. 8,021 more destroyed buildings didn't help.
  - L0 step 1 done on account 1 (1.7 h). At full size our 7B scores 75.6% on disaster type and 31.1% on relational (71.1% and 23.3% at 512 px). By the rule fixed before the run, the gap is not closed.
  - L0's next step and R15 now wait on your decision. R16c step 2 is still ready to run.
- **19 Oct 2026, 13:40:**
  - R16c step 1 done on your PC (2.6 min). With SAM 3's outlines, a perfect grader would reach Destroyed IoU 0.614 against our grader's 0.208. On the 1,055 validation counting questions, R16's best map answers 36.9% against 39.2% for R11.
  - R16c step 2 (a stronger grader, picked on held-out training images, two seeds) and step 3 (SAM 3 on the training images, then the grader on SAM 3's outlines) are built. Their logic was checked on a synthetic image. Step 2 is ready to run on account 1.
- **19 Oct 2026, 12:55:**
  - Your commit is pushed (`1b33574`).
  - R12's first run on account 2 stopped after 6 minutes in Cell 4b: the uploaded masks share the after-images' file names, so the code took the masks folder for the photos. The one-line fix is under "Needs you now".
  - R16's ceiling: with a perfect grader, SAM 3 + R11's outlines would reach Destroyed IoU 0.614 against our grader's 0.208. R16c, a stronger grader, is proposed and waits on your decision.
- **19 Oct 2026, 12:30:**
  - R16b done on your PC (7.3 min). Correcting R16's grader by the real grade shares improves every R16 map: the vote reaches Destroyed IoU 0.274 and mIoU 0.585. Intact buildings graded destroyed fall from 3,539 to 1,500. R11 with TTA stays ahead, so R16 isn't kept.
  - L0 step 1 is built and ready to run on account 1 (504, 616 and 1036 px, about 2–3 h).
  - `.gitignore` updated for the GitHub push; the commands are under "Needs you now".
  - GPU left: account 1 12 h, account 2 22 h.
- **19 Oct 2026, 11:00:** R16 done, not kept: Destroyed IoU 0.261 at best against R11's 0.297, and its grader over-grades intact buildings. O2 compared like for like with the DisasterM3 paper: on optical questions our 7B averages 36.1% against the base model's 31.2%, and counting is 2.2 points short (32.0% against 34.2%), not 3.7. The paper's own fine-tune reaches 40.4%, so we improve the 7B before distilling (new task L0; recipe compared with the paper's Appendix B.3).
- **19 Oct 2026, 10:00:** D5b made masks for tier3's 612 exam tiles on your PC; its lookup table matches D5's. R12 rebuilt to read both pools (1,619 tiles, 8,021 destroyed buildings) and ready to run on account 2. Run steps for R12 and R16 are under "Needs you now".
- **19 Oct 2026, 09:30:** Decisions: R12 takes option A (tier3's 612 exam tiles join it, for the O1 score only); R16 and R15 run regardless; R16 runs on account 1, which ran E3. R16's run steps are under "Needs you now".
- **19 Oct 2026, 09:00:** H1 V2b done: our 7B scores 30.5% on building counting (greedy), the best so far but below the base model's 34.2%; six-task mean 35.6%. Tier3 folder renamed to `dataset/xView2 Challenge Dataset - Tier 3 Data` (Antigravity IDE had it open) and added to `.gitignore`.
- **19 Oct 2026, 08:30:** L2 step 1 done: 1,085 of 4,000 teacher answers kept, and its push to the Kaggle dataset failed (step 2 will read the notebook output instead). Both accounts free. Tier3 extracted into `dataset\archive`; renaming it failed because the folder is in use.
- **19 Oct 2026, 08:20:** D5b done from the downloaded zip: tier3's unused tiles hold no destroyed buildings, but its 612 benchmark tiles hold 4,549. R12 waits on whether to add them.
- **19 Oct 2026, morning (later):** R12 built on the 6-channel recipe, and R16 built; both ready to run. R12 reads xBD's images from Kaggle's xView2 copy, so only D5's masks need uploading. R16's logic checked on a synthetic image. R13 corrected in the roadmap and Paper_Audit: no BRIGHT event is missing from DisasterM3, and Noto must be dropped. xBD tier3 is downloading from Kaggle for D5b.
- **19 Oct 2026, 07:06:** R16 now runs regardless of R12 (roadmap and tracker). R12's other sources checked against DisasterM3's train and benchmark lists: BRIGHT can't add optical destroyed examples, and every BRIGHT and xBD tier3 event is already in DisasterM3. R13 must skip Noto and BRIGHT's 6 optical events.
- **19 Oct 2026, afternoon:** Distilling Step-by-Step PDF downloaded to the papers folder and checked. The A-versus-B decision for the 3B is recorded in Paper_Audit §5.3, linked from the roadmap's L2b and Research_Directions §7.3.
- **19 Oct 2026, midday:** L2 step 1's smoke test passed; the teacher answers in one sentence, so step 1 is answer-level and takes about 2 h. Stricter answer check added. L2b planned (explanations on request), and Distilling Step-by-Step added to the paper audit.
- **19 Oct 2026, morning:** T1 v3 done on your PC: circles on the picture's edge fell from 20% to 1%, and 41 of the 46 checked circles sit on damage. T1 is finished.
- **19 Oct 2026, morning:** L2 step 1 hadn't started; it goes on account 1. R12's options written up for your decision.
- **19 Oct 2026, morning:** H1 V2b running on account 2. T1 v3 (circles off the edge) built as a copy of v2 and running on your PC. Account 1 has 14 h free.
- **19 Oct 2026, morning:** L1b done on its second run: INT8 now keeps accuracy (Destroyed IoU drop 0.0003), so the edge segmenter part of O4 is met.
- **19 Oct 2026, night:** L1b's first Kaggle run stopped at Cell 3: the DisasterM3 mirror wasn't attached. Needs a re-run with the dataset added; no code change.
- **19 Oct 2026, evening:** L2 step 1 started on Kaggle. H1 V2b built and ready to run. L1b stays on Kaggle CPU: running it on your PC would slow the laptop down.
- **19 Oct 2026, later:** H1 V2 done: six-task mean 34.2% (base model 31.2%); the fact line hurt every task, so it is not kept. Added H1 V2b. GPU hours left: account 1 14 h, account 2 24 h.
- **19 Oct 2026:** Tracker started. R11 finished and was reviewed (numbers checked against `D4_R11.csv`). T1 v2 finished on your PC. L1b and L2 step 1 built, ready to run. The original L2 teacher notebook was restored, and the changes live in the copy. `.gitignore` now skips `*.onnx`.
