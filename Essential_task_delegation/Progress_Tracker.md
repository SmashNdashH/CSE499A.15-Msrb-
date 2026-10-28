# CSE499B progress tracker

**Last updated:** 28 Oct 2026.
- **L2 step 1 again, with the new 7B as teacher: built on your go and dry-run checked (28 Oct), ready to run on account 1 (about 5.5 h).** [l2-step1-new-teacher.ipynb](Champion_Task/l2-step1-new-teacher.ipynb): full size, greedy, the plain training prompt, and the first run's entries minus road counting (3,265). Run steps under "Needs you now".
- **Guideline licences checked (28 Oct): none of the six documents allows posting it online.** Sphere's copyright page requires written permission for "posting online"; EMS-98 is copyright with no reuse grant; the xBD paper is on arXiv under arXiv's own licence; INSARAG and IWG-SEM carry no licence at all. Parked by you: not a concern while the work is in progress.
- **Teammates' 27 Oct pushes reviewed (28 Oct).** Ridita's guideline search and offline DisasTeller copy hold up (T2 steps 4, 4b and 6). Aryan's v4 needs three corrections, and its data show that R11's table alone (37.0%) beats feeding its counts to the 7B (27.0%). Tamanna's rebuilt demo is honest; it shows the fine-tuned 7B calling every inspected building Intact. Feedback under "Needs you now"; details in the ledger, §9.
- **L0 step 2 done: the new 7B is kept** by the rule fixed 19 Oct, on all questions and on E5's clean and strict sets. Against the old 7B at full size, on the 10,675 optical questions:

  | Task | Old | New | Change | Questions gained / lost |
  |---|---|---|---|---|
  | ORR | 31.12 | 34.62 | **+3.50** | 33 / 3 (p < 0.001) |
  | DTR | 75.64 | 76.77 | +1.13 | 5 / 1 (p = 0.22) |
  | BBR | 20.94 | 19.99 | −0.95 | 24 / 43 (p = 0.03) |
  | BDC | 30.09 | 30.31 | +0.22 | 40 / 32 |
  | DRE | 32.39 | 33.03 | +0.64 | 22 / 10 |
  | DSR | 40.06 | 40.46 | +0.40 | 34 / 26 |

  - **The rule:** DTR and ORR rise, and no other task drops by more than 1 point. ORR's gain is clear. DTR's is 4 questions net and could be noise. BBR's drop is real (p = 0.03) but inside the limit.
  - **Clean (E5's 186 copied questions dropped):** ORR +3.52, DTR +1.17, BBR −0.91, the others +0.20 to +0.71, so the same verdict; the strict set gives it too. Mean of the six: 38.37 → 39.20 (clean 38.14 → 38.99).
  - **Expectation, fixed before the run:** DTR and ORR up about 1–4 points, the others within about a point. Met.
  - **Counting (BDC) is 30.31%,** still under the base model's 34.2%.
  - **So `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP` is our 7B from now on, and L2's teacher.**
  - Part A (BDC, BBR) took 7.8 h and part B (DRE, DSR, DTR, ORR) 5.9 h, on 27 Oct. Each run notebook differs from the output-free one only in `MODEL` and `RUN_TASKS`. The verdict was computed on your PC from both models' per-question answers, after every published accuracy was reproduced. Results in [L0-step2-results/new-partA/](Champion_Task/L0-step2-results/new-partA/L0S2/README.md) and [new-partB/](Champion_Task/L0-step2-results/new-partB/L0S2/README.md); the ledger has the details.
- **R16c step 5b done (account 1, 2.3 h, 27 Oct): seed 44 misses O1's gate on clean data by 0.002, so O1 on clean data is met on the mean of the three grader seeds only, narrowly.**
  - Seed 44's SAM 3-only map: 0.5527 / 0.3156 on all 645 pairs, 0.548 / **0.2982** on the 605 clean ones (expected 0.30–0.32).
  - Clean, by seed: 0.3161 (42), 0.3051 (43), 0.2982 (44); mean 0.5473 / 0.3065. By the rule fixed before the run: met on the mean only, narrowly.
  - Checks: R11 reproduces its CSV; the saved rows rebuild the maps exactly (seed 43's 0.0012 mismatch in step 5 didn't recur); the run notebook matches the build cell by cell.
  - Its grader is the best of the three on true buildings: macro F1 0.754 (step 5: 0.737 and 0.730).
  - Results in [R16c-results/step5b/](Champion_Task/R16c-results/step5b/README.md).
- **All four Kaggle accounts are free (28 Oct).** What to run next is under "Needs you now".
- **Pushed 28 Oct:** `7505d2a29` renamed Aryan's `BDC_Results.` to `BDC_Results` (a folder name ending in a dot broke every Windows pull), then `bd934391b` with this week's work and the 5th-class deliverables.
- **R12 checked by content (27 Oct): two of its 1,619 extra xBD tiles share a before-image with validation pairs that aren't among E5's 40.** So R12 trained on 42 leaked pairs. A leak can only flatter R12, which still didn't beat R11, so its verdict stands. E2's and E3's decisions also hold on clean data. Found while re-checking Phase A for the supervisor update; details in [E5-results/E5/](Champion_Task/E5-results/E5/README.md).
- **L0's merge done (run 2, account 3, a 10-min CPU session, 27 Oct): the new 7B is on Hugging Face** as `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP` (public, 16.6 GB).
  - **Checks, all passed:** exactly the 196 adapted projections changed, of 729 weight tensors (none outside them, none left unchanged); no inf or NaN; all 7 files on the Hub have their local sizes; checkpoint-28's last three losses match session 3b's log.
  - The run notebook's code matches the build cell by cell. Results in [L0-step2-results/merge/](Champion_Task/L0-step2-results/merge/L0S2-merge/README.md).
  - Run 1 had stopped in Cell 2 on a check of mine: peft saves the 196 adapted modules as 88 name patterns. Cell 2 now counts the 196 LoRA pairs.
- **Account 4 (Tamanna's, 28.5 GPU h) added** on 27 Oct.
- **Built on your go ("build 1 and 2") and dry-run checked on your PC:**
  - [l0-step2-merge.ipynb](Champion_Task/l0-step2-merge.ipynb): L0's merge without torchao, on a CPU session (done, above);
  - [r16c-step5b-third-grader-seed.ipynb](Champion_Task/r16c-step5b-third-grader-seed.ipynb): a third grader seed for O1's clean-data claim (done, above).
- **E5 done (account 2): 40 leaked validation pairs, confirmed three ways, and every affected score re-scored.**
  - **O1 on clean data:** only R16c step 5's SAM 3-only map passes (0.3161 and 0.3051), narrowly. A third grader seed is proposed before claiming it. *(28 Oct: the third seed scores 0.2982, so it's met on the three-seed mean only, 0.3065.)*
  - The leak helped the before + after models twice as much: R11 is now 0.018 below R9 on Destroyed IoU.
  - **O2:** 186 of 10,675 optical benchmark questions use a copied after-image. Without them, counting is 32.47% (was 32.96%), and every score drops by at most 0.7 points.
  - Tables: [E5-results/E5/](Champion_Task/E5-results/E5/README.md).
- **R16c step 5 done (account 1, 3.8 h):**
  - It doesn't replace step 2: its SAM 3 + R11 map scores 0.2434 and 0.2158.
  - Its SAM 3-only map passes O1's gate on all 645 pairs (0.3347 and 0.3242) and on clean data.
  - Results: [R16c-results/step5/](Champion_Task/R16c-results/step5/README.md).
- **L0 session 3b (account 3, 2.0 h): the training is complete (28 of 28 steps; checkpoint-28 on Hugging Face), but the merge failed** on a library version (torchao). The fix is a merge-only copy, needing no GPU and no retraining; it needs your go.
- **Ridita pushed `audit_outputs/`.** It matches what she sent. Her README quotes temporary-folder paths from her PC (see "Needs you now").
- **Files moved into the usual pattern:**
  - `E5-step1-results/` and `E5-step2-results/` became `E5-results/step1/` and `step2/`;
  - `R16c-step5-results/` became `R16c-results/step5/`;
  - the L0 log went to `L0-step2-results/train-session3b/`.
- **H2b done (Noman's account, 23 Oct, 47 min): no model passes, so by the rule fixed before the run, the system ships fixed queries only, with no free-form questions.**
  - **The bar:** at least 90% test-suite accuracy and RS(10) above 0 (points per 100 items). The best accuracy was 73.9% (Coder-3B in fp16), and every RS(10) was negative: every model gives too many wrong answers.
  - **Our 7B in 4-bit:** 69.0% [57.1, 80.5], RS(10) −89.6. It refused 94% of the unanswerable questions, the most of any model. It's level with the separate Coder-3B (72.0%; the difference, −3.0 points [−10.7, 3.8], isn't clear), and it adds no memory (5.8 GB, 3.3 s per question on a T4).
  - **The base VL-3B, standing in for the distilled 3B, is far weaker:** 46.4% in 4-bit, RS(10) −365.7, refusing only 36%; 22.6 points below our 7B [9.2, 36.8].
  - **Where every model fails:** pairs of buildings (6–28%), the "no destroyed building nearby" NULL case (0–22%), positions (0–52%) and distances (44–70%). Our 7B in 4-bit gets counts, percentages, rankings and size thresholds 89–100% right.
  - **Responder wording costs the most:** our 7B in 4-bit scores 52.9% on it, against 74.7% on plain wording.
  - **Retries never fixed a query:** in all 200 retried items the model then refused or failed again, so first-try accuracy equals the final one.
  - **4-bit costs our 7B 4.6 points** [0.8, 8.8] and the VL-3B 9.9 [3.4, 17.6]; the coders lose nothing measurable.
  - **Checks:** the run notebook's code matches ours cell by cell; R11's table (117,687 buildings in 2,652 images); all 8 configurations ran; batched and one-at-a-time replies agreed on 96–100%. Results in [H2b-results/H2b/](Champion_Task/H2b-results/H2b/README.md).
- **L0 training, session 3a done (account 3): step 25 of 28.**
  - It resumed from checkpoint-24, trained one step (2,541 s, about 42 min), then stopped at its time limit as planned.
  - **Loss:** 0.537; the learning rate is down to 1.1e-5.
  - **Memory:** peak 14.3 GB on each GPU.
  - **Checkpoint:** checkpoint-25 is on Hugging Face (the log line plus a sha256 match); the local copies were deleted.
  - **Results:** [L0-step2-results/train-session3a/](Champion_Task/L0-step2-results/train-session3a/L0S2-train/README.md).
  - **Left:** steps 26–28 (about 2.1 h) and the merge. **Session 3b's Cell 2 must set `time_limit_h` back to 5.0,** or it stops after one step again.
- **H2b's smoke test passed on Kaggle** (account 2, 08:31). It ran end to end with both kinds of model in 4-bit:
  - VL-3B: 2.35 GB, about 4.1 s per question;
  - Coder-1.5B: 1.12 GB, about 3.7 s per question;
  - batched and one-at-a-time replies agreed 100%.
  - **Early signal, 6 items only:** both models answered both unanswerable questions instead of refusing, and both missed the "no destroyed nearby" NULL case.
  - **A bug found and fixed before the full run:**
    - transformers fills any generation setting left at its default from the model's own defaults;
    - so the "greedy, no repetition penalty" settings, passed in, likely ran with the models' 1.05–1.1 repetition penalty, and in the version on your PC even switched to sampling;
    - Cell 9 now installs the settings as the model's own and checks them; it was checked on a tiny model on your PC.
    - **Confirmed on Kaggle (second smoke run, 08:50):** the warning is gone, the check passed for both models, and the 6 results are identical. That session lacked the `HF_TOKEN` secret, which is harmless for these public models; turn it on for the full run.
  - **Files:** the smoke run is kept as `h2b-sql-agent-graded-test-smoke.ipynb`. The main notebook was rebuilt output-free, and every cell except Cells 2 and 9 matches the smoke run's code exactly.
- **L0 training, session 2 done (account 3):** it resumed from checkpoint-15 and reached **step 24 of 28**, one more step than planned. The 8-bit AdamW fix held.
  - **Speed:** 9 steps of about 37 min each (2,170–2,259 s), faster than session 1's 42 min.
  - **Memory:** peak 14.4 GB on GPU 0 and 14.5 GB on GPU 1.
  - **Loss:** 0.520 at step 24, in session 1's band of 0.52–0.60; the learning rate is down to 1.6e-5.
  - **Checkpoints:** 16, 20 and 24 are on Hugging Face. The local copies (1.9 GB) were deleted after their hashes matched.
  - **Results:** [L0-step2-results/train-session2/](Champion_Task/L0-step2-results/train-session2/L0S2-train/README.md).
  - **Ignore `train_loss` 0.205 in the summary:** after a resume the trainer divides this session's 9 losses by all 24 steps.
- **H2 done (account 2): R11's counting on the benchmark is 32.96% on the optical questions** (27.46% over all 4,978 questions in the table).
  - **Our best counting so far:** it beats R9's tuned counter (26.51%) and our 7B (30.09% at full size, 32.01% at 512 px).
  - **But still below the bar:** it's under the base 7B's 34.2%, so the counting half of KR2 is still 1.2 points short.
  - **The agent:** both models wrote valid SQL for all 100 sampled questions, and their counts matched the table on 99 and 100.
  - **The demo:** an open question showed the 3B leaving out the image filter (answer 33,880), which is exactly what H2b grades.
  - **The table:** 117,687 buildings in 2,652 images; 3 images couldn't be segmented. Results in [H2-results/H2/](Champion_Task/H2-results/H2/README.md).
- **H2b checked against H2's real table on your PC** (with stand-in models), using 297 items on 90 images.
  - All 6 near-miss queries were caught. One of them, integer division in a percentage, would have passed on the real table alone, which shows why the five versions of the table are needed.
  - Its smoke test now uses both kinds of model: the base VL-3B and the 1.5B coder.
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
  - **26 Oct update (E5): not on clean data.** Without the 40 leaked validation pairs, the four combinations score 0.2838–0.2980 Destroyed IoU (mean 0.2917).
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

Kept for Abrar, and updated whenever a task starts running, finishes, or needs a re-run. The task IDs follow [Next_Steps_Roadmap.md](Next_Steps_Roadmap.md); results come from the folders in [Champion_Task/](Champion_Task/). Why each task ran and what it decided: [Task_Ledger.md](Task_Ledger.md).

**Status words:** Done · Running · Ready to run (notebook built, not started) · Needs your decision · Needs rerun · Not started.

## At a glance

| Done | Running | Ready to run | Needs your decision | Needs rerun | Not started |
|---|---|---|---|---|---|
| 25 | 0 | 1 | 0 | 0 | 13 |

## Pace since Phase A, and the timeline to the showcase (measured 24 Oct)

**Measured from the run board** (Phase A started 18 Oct; each run counted on the day it finished):

| Day | Kaggle runs finished | GPU hours |
|---|---|---|
| 18 Oct | 5 (2 on CPU) | 2.0 |
| 19 Oct | 10 (1 on CPU) | 27.1 |
| 20 Oct | 4 | 24.1 |
| 21 Oct | 2 | 11.5 |
| 22 Oct | 4 | 10.0 |
| 23 Oct | 1 (Noman's account) | 0.8 |
| **6 days** | **26** | **75.5** |

Plus 6 runs on your PC, on 18–19 Oct: D5, D5b, T1 v2, T1 v3, R16b and R16c step 1.

- **About 24 tasks finished in 6 days,** about 4 a day: Phase A, Phase B, R16 to R16c step 4, L0 step 1, L2 step 1, H2 and H2b.
- **GPU quota set the pace, not your speed.** 75.5 GPU hours in 6 days is close to the three accounts' 90 hours a week. The drop after 20 Oct is accounts running out: account 1 hit zero on 21 Oct.
- **About 1 run in 8 failed and was re-run:** R12's first try, L1b's first run, L0 session 2's first try, and R16c step 5 (out of quota).
- **A notebook task took about a day to build and check, and a day to run** (for example, H2b was built on 22 Oct and finished on 23 Oct).

**Remaining work in the same units** (estimates from the roadmap; Layer 3 left out):

| Work | GPU hours | Depends on |
|---|---|---|
| L0 session 3b and the merge | 3.5–4 | quota |
| L0: the new 7B on all six tasks | about 14.5 | the merge |
| R16c step 5, the re-run | 1.5–2 | account 1's quota |
| L2 step 1 again (only if the new 7B is kept) | about 2 | L0's verdict |
| L2 step 2: train the 3B, then its six-task evaluation | about 20 + about 8 | L2 step 1 |
| T2 (DisasTeller on local models) and the gallery export | about 4 | — |
| H2c: adapter training and the new test | 5–10 | L2 |
| **Total** | **about 60 (50–75)** | |

At the measured rate that's about a week of GPU time, and up to 120 hours a week are available with Noman's account. **The limit is the chain L0, then L2, then H2c, which must run in order:** about 7–9 working days with no failures, or 2–3 weeks with the observed failure rate and weekly quota resets.

**Projected timeline** (the model work follows the measured pace; the website and T2 are new kinds of work with no recorded pace, so their times are assumptions):

| Week | Model work (measured pace) | Website and showcase (assumed) |
|---|---|---|
| 25–31 Oct | L0 session 3b and merge; the new 7B's six-task evaluation; R16c step 5; gallery export; R11's INT8 export (CPU) | T2 built and run offline; H2c's question set written (CPU) |
| 1–7 Nov | L0's verdict; L2 step 1 again if needed; L2 step 2 training starts | Website: the story page |
| 8–14 Nov | L2 training ends; the 3B's six-task evaluation; H2c's adapter | Website: the demo page; mid-November decision on Layer 3 |
| 15–21 Nov | H2c's test: ship or fall back | Website: 3D damage city and live mode; booth-laptop browser test |
| 22–28 Nov | Layer 3 on teammates' GPUs, only if chosen | Vercel deployment and phone test; report and poster drafts |
| 29 Nov–5 Dec | — | Video, poster, final report |
| 6–14 Dec | Buffer for failures | Rehearsal; showcase about 15 Dec |

- **Model work should be done by about mid-November** at this pace, leaving about 3–4 weeks for the website, the report and buffer.
- **The biggest unknown is the website,** since there's no recorded pace for it. The plan assumes about 3 weeks part-time (1–21 Nov). How many hours a week you can give it decides whether that holds.
- Re-measure the pace each week from the run board, and update this section.

## Run board: what ran, what's running, what's next

Every notebook run so far, on Kaggle or on your PC, and every planned run. Updated with each change; details under "Needs you now" and in the task tables.

**Running now** (28 Oct): nothing. All four accounts are free.

**Ready to run** (28 Oct; run steps under "Needs you now")

| What | Notebook | Account | Time |
|---|---|---|---|
| L2 step 1 again: teacher answers from the new 7B | [l2-step1-new-teacher.ipynb](Champion_Task/l2-step1-new-teacher.ipynb) | 1 | about 5.5 h, plus a 15–20 min smoke test |

**Later: not built yet**

| What | Waits for |
|---|---|
| H1 V3: counting from R16c's per-building grades | counting beating R11's on validation (now 37–39% against 39.2%) |
| L2 step 2: train the 3B | L2 step 1 again (ready to run), a fixed copy of its notebook, and your choice of how the 3B learns |
| L2b: explanations as a second skill | L2 step 1 with the final teacher |
| T2: DisasTeller on local models (O3's missing part) | GPU time; not started |
| T3: offline demo and public showcase (Vercel website) | T2, and a gallery export |
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
| L0 step 2: the full training, session 2 (steps 16–24; the first try crashed at step 16) | `l0-step2-topup-training.ipynb`; results in [L0-step2-results/train-session2/](Champion_Task/L0-step2-results/train-session2/L0S2-train/README.md) | account 3, 21–22 Oct | about 6 h (plus 1 h for the failed first try) |
| H2: R11's per-building table and the SQL agent | `h2-sql-counting-agent.ipynb`; results in [H2-results/H2/](Champion_Task/H2-results/H2/README.md) | account 2, 22 Oct | about 1.5 h with the smoke test |
| L0 step 2: the full training, session 3a (step 25) | `l0-step2-topup-training.ipynb`; results in [L0-step2-results/train-session3a/](Champion_Task/L0-step2-results/train-session3a/L0S2-train/README.md) | account 3, 22 Oct | about 1 h |
| H2b: two smoke tests (VL-3B and Coder-1.5B in 4-bit; the second confirmed Cell 9's fix) | `h2b-sql-agent-graded-test-smoke.ipynb` | account 2, 22 Oct | about 15 min each |
| H2b: the full run (4 models, each in 4-bit and fp16) | `h2b-sql-agent-graded-test.ipynb`; results in [H2b-results/H2b/](Champion_Task/H2b-results/H2b/README.md) | Noman's account, 23 Oct | 47 min |
| R11s7: R11 with seed 7 | `d4-champion-model-R11s7.ipynb` | account 2, 20 Oct | 6.8 h |
| L0 step 2: the old 7B's full-size re-score, part A (BDC, BBR) | `l0-step2-full-size-rescore-old-partA.ipynb`; results in [L0-step2-results/old-partA/](Champion_Task/L0-step2-results/old-partA/L0S2/README.md) | account 3, 20 Oct | 8.6 h |
| L0 step 2: part B (DRE, DSR) | `l0-step2-full-size-rescore-old-partB.ipynb`; results in [L0-step2-results/old-partB/](Champion_Task/L0-step2-results/old-partB/L0S2/README.md) | account 2, 20 Oct | 4.1 h |
| R16c step 4: SAM 3 on the training images | `r16c-step4-sam3-training-images.ipynb`; results in [R16c-results/step4/](Champion_Task/R16c-results/step4/README.md) | account 1, 20 Oct | 4.6 h |
| R16c step 5: the grader trained on SAM 3's outlines (the re-run) | `r16c-step5-grader-on-sam3-outlines.ipynb`; results in [R16c-results/step5/](Champion_Task/R16c-results/step5/README.md) | account 1, 26 Oct | 3.8 h |
| L0 step 2: the full training, session 3b (steps 26–28; the merge then failed) | `l0-step2-topup-training.ipynb`; log in [L0-step2-results/train-session3b/](Champion_Task/L0-step2-results/train-session3b/l0-step2-topup-training.log) | account 3, 26 Oct | 2.0 h |
| E5 step 1: the hash check, CPU only | `e5-step1-hash-check.ipynb`; results in [E5-results/step1/](Champion_Task/E5-results/step1/README.md) | account 2, 26 Oct | 0.5 h, no GPU |
| E5 step 2: per-image pixel counts for R11, R11s7, R9 and R1 | `e5-step2-per-image-counts.ipynb`; results in [E5-results/step2/](Champion_Task/E5-results/step2/README.md) | account 2, 26 Oct | 0.9 h |
| L0 step 2: the merge, run 2 (run 1 stopped in Cell 2 on a check of mine) | `l0-step2-merge.ipynb`; results in [L0-step2-results/merge/](Champion_Task/L0-step2-results/merge/L0S2-merge/README.md) | account 3, CPU session, 27 Oct | 10 min, no GPU |
| R16c step 5b: a third grader seed (44) for step 5's SAM 3 map | `r16c-step5b-third-grader-seed.ipynb`; results in [R16c-results/step5b/](Champion_Task/R16c-results/step5b/README.md) | account 1, 27 Oct | 2.3 h |
| L0 step 2: the new 7B, part A (BDC, BBR) | `l0-step2-full-size-rescore-new-partA.ipynb`; results in [L0-step2-results/new-partA/](Champion_Task/L0-step2-results/new-partA/L0S2/README.md) | 27 Oct | 7.8 h |
| L0 step 2: the new 7B, part B (DRE, DSR, DTR, ORR) | `l0-step2-full-size-rescore-new-partB.ipynb`; results in [L0-step2-results/new-partB/](Champion_Task/L0-step2-results/new-partB/L0S2/README.md) | 27 Oct | 5.9 h |

**Done on your PC (CPU):** these wrote their results into `Champion_Task`, and the notebooks keep their outputs.

| What | Notebook | Results |
|---|---|---|
| D5: xBD converter | `d5-xbd-converter.ipynb` | `D5-results/` |
| D5b: xBD tier3 check and masks | `d5b-xbd-tier3-converter.ipynb` | `D5b-results/` |
| T1 v2 and v3: heatmap fixes | `t1-survivability-heatmap-v2.ipynb`, `-v3.ipynb` | `T1-results/T1-v2/`, `T1-v3/` |
| R16b: the grader's prior correction | `r16b-grader-prior-correction.ipynb` | `R16b-results/` |
| R16c step 1: the ceiling and counting | `r16c-step1-ceiling-and-counting.ipynb` | `R16c-results/step1/` |

Read-only checks also ran on your PC: the Turkey and counting breakdowns, dry runs of the new notebooks, the folder audit, and L0 step 2's verdict (28 Oct, from both models' per-question answers). They read the results and write nothing into the project (only to a temporary folder).

## Needs you now

- **Next on the four free accounts (28 Oct): your pick.**
  1. **L2 step 1 again, with the new 7B as teacher: built 28 Oct, ready on account 1** (about 5.5 h; run steps in the next item).
  2. **L2 step 2: train the 3B** (about 20 GPU h, then about 8 h to evaluate). Its notebook needs a fixed copy first (its old installs fail on today's Kaggle), and how the 3B learns is still your choice (L2 step 2's row).
  3. **O1:** report it as met on clean data on the mean of three grader seeds, narrowly (0.3065). More seeds aren't recommended: adding them until one passes would bend the rule fixed before the run.
  4. **T2, DisasTeller on local models (O3's missing part):** Ridita's offline DisasTeller copy arrived on 27 Oct in `Offline_RAG_KR2/`. Reviewing it comes first.
- **Run now: L2 step 1 again, teacher answers from the new 7B** ([l2-step1-new-teacher.ipynb](Champion_Task/l2-step1-new-teacher.ipynb)).
  - **What and why:** L0 kept the new 7B on 28 Oct, so it becomes the 3B's teacher. The notebook asks it the first run's training questions (minus road counting, 3,265) at full size, greedy, with the plain training prompt, and keeps only answers that match their label.
  - **Expectation, fixed before the run:** more kept answers for disaster type (first run 89.4%) and relational reasoning (13.3%), partly from memory, since L0's top-up trained on these entries again; building counting, bearing bodies and land use within about 3 points of 15.6%, 42.7% and 50.9%; at least the first run's 1,085 kept in all.
  - **Steps:**
    1. Sign in as account 1 (`abrarmohammedtanzim`). **Create → New Notebook → File → Import Notebook → Browse**, pick `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\l2-step1-new-teacher.ipynb`, then **Import**.
    2. **Session options:** GPU **T4 x2**, Internet **On**.
    3. **Input → + Add Input:** `abrarmohammedtanzim/disasterm3-mirror`.
    4. No secret is needed: the new 7B's repo is public (`HF_TOKEN` only speeds up the 16 GB download).
    5. **Smoke test:** in the cell starting `# ── Cell 2: Configuration`, set `SMOKE_TEST = True`, then **Run → Run All** (15–20 min with the download). Check:
       - Cell 3: `full size checked (1036 px)`;
       - Cell 4: `Same entries as the first run: 4,000 drawn with seed 42 from 40,233 checkable entries; 735 road-counting entries left out` and `self-checks passed`;
       - Cell 5: three answers, each one sentence, some matching their label;
       - Cell 6: a table by task, then `✓ Wrote /kaggle/working/L2S1-new-teacher`.

       Any red error: stop and send me a screenshot.
    6. Set `SMOKE_TEST = False`. **Save Version → Save & Run All (Commit) → Save.** Stop the editor session. About 5.5 h; it stops cleanly at 11 h.
    7. When it's finished:
       - **Output:** download the `L2S1-new-teacher` folder so that `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\L2-step1-results\new-teacher\L2S1-new-teacher\README.md` exists.
       - Save the run notebook over `Champion_Task\l2-step1-new-teacher.ipynb`.
       - Tell me "L2 step 1 done".
- **Feedback for your teammates (their 27 Oct pushes, checked 28 Oct; details in the ledger, §9):**
  - **Ridita:** the guideline search passed its retrieval test (27 of 30, recounted), the Kaggle run with the internet off is real, and the DisasTeller copy is offline by default. Next: T2 steps 1–3 (local LLM and image tools). Ask her to git-ignore `local_chroma_db/` (it changes on every search). The guideline licences are parked by you (28 Oct).
  - **Aryan:** three corrections for v4.
    1. The U-Net row is a copy of the image-only file; the real U-Net number is v2's 23.96%.
    2. His criterion 2, as written before the run, fails (the intervals overlap); a paired test shows R11 ahead (p = 0.01). He should report both instead of switching the comparison after the run.
    3. R11's table alone (the option nearest R11's count) scores 37.0% on his 1,701 questions, 10 points above the VLM given that count (27.0%). That reverses his conclusion: counts should skip the VLM.
    - Also: his subset isn't O2's measure (34.2% includes intact questions); E5's duplicate check is done (36 of his questions; nothing moves by more than 0.4); the TEOChat blockquote's second sentence isn't from the paper as far as our notes go.
  - **Tamanna:** the rebuild is honest and clean (validation tiles, none leaked; committed runs; no hints in the prompts). The demo shows the fine-tuned 7B calling all 30 buildings Intact, and its one-word class contradicting its own EMS-98 grade on half the crops. For Part B: score against the true validation buildings and compare with R16c's grader (F1 0.754) on the same buildings, and mark the building in each crop. Her audit response still quotes an AI prompt (line 250); your call.
- **CSE499B 5th class (Wed 28 Oct): two deliverables ready.**
  - **Regular progress report, update 2:** [others/CSE499B_update_2.pdf](../others/CSE499B_update_2.pdf), with its source [.tex](../others/CSE499B_update_2.tex).
    - Format: update 1's (2 pages, compiled with MiKTeX).
    - Content: the period since 14 Oct, with the leak and the clean-data scores stated, one section per member.
    - **Pushed 28 Oct** (`bd934391b`). **To do:** print it double-sided (no top page, no spiral binding).
  - **Literature review presentation:** the deck "CSE499B Literature Review" (21 slides) is a private web deck for now.
    - It follows the faculty's brief: one slide per paper (16 papers), no per-member attribution, the Group 5 footer on every slide.
    - **Exported and pushed 28 Oct** as `others/CSE499B_Literature_Review.pptx` (`bd934391b`). Its properties and text were checked for tool traces first: none.
  - **Faux versions (28 Oct): your part covers Phase A only (E1, E2, D5/D5b, E3), and the rest of it is held for next week. The four teammates' sections are kept word for word from update 2, since each presents their own part.** The originals above are unchanged.
    - [others/CSE499B_update_2_faux.pdf](../others/CSE499B_update_2_faux.pdf) and its [.tex](../others/CSE499B_update_2_faux.tex): 2 pages, a results table (test, question, result, decision), terms explained where they first appear, every number in your part checked against the Phase A briefing page.
    - `others/CSE499B_Literature_Review_faux.pptx`: 11 slides, keeping the six papers behind those tests (DisasterM3, BRIGHT, SAM 3, BDAChat, EfficientSAM3, TEOChat), with the overview, synthesis and references cut to match. Both checked for tool traces: none.
  - **Next week (6th class):** the proposed solution report (printed and PDF).
- **Done 27–28 Oct: the new 7B's six tasks and the verdict (kept; the table is at the top of this file).** The steps below stay as the record of how they ran.
  - **What and why:** L0 step 2's keep rule (fixed 19 Oct): keep the new 7B only if disaster type (DTR) and relational reasoning (ORR) rise, and no other task drops by more than 1 point, against the old 7B at full size. It will be judged on all questions and on E5's clean ones.
  - **Expectation, fixed before the run:** DTR and ORR rise by about 1–4 points; the other four stay within about a point.
  - **How:** the same notebook that scored the old 7B ([l0-step2-full-size-rescore.ipynb](Champion_Task/l0-step2-full-size-rescore.ipynb)), with `MODEL = "new"`, in its own two parts. Both parts ask the 10,675 optical questions at full size with greedy decoding.
  - **Part A on account 4 (Tamanna's), about 8.6 h:**
    1. Sign in as Tamanna's account. If Kaggle asks for it, finish phone verification (**Settings → Phone verification**); GPU and internet need it.
    2. **Create → New Notebook → File → Import Notebook → Browse**, pick `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\l0-step2-full-size-rescore.ipynb`, then **Import**.
    3. **Session options:** GPU **T4 x2** (the 7B needs both), Internet **On**.
    4. **Input → + Add Input:** `disasterm3-bench-mirror-v2` (by redwannewaz). If it doesn't show up, it's private: on account 3, open the dataset's **Settings → Sharing** and add Tamanna's account.
    5. No secret is needed: the merged 7B's repo is public, so don't put the write token on this account.
    6. In the cell starting `# ── Cell 2: Configuration`, set `MODEL = "new"`, `RUN_TASKS = ["BDC", "BBR"]` and `SMOKE_TEST = True`. **Run → Run All**, about 15 min with the 16 GB download. Check:
       - Cell 1: `GPUs: ['Tesla T4', 'Tesla T4']`;
       - Cell 2: `✓ L0 step 2  [SMOKE TEST]: model 'new' (AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP)`;
       - Cell 4: `✓ Model 'new' loaded … full size checked (1036 px)`;
       - Cell 5: `✓ All questions answered`;
       - Cell 6: a results table, then `✓ Wrote l0_step2_new_results.csv …`.

       Any red error: stop and send me a screenshot.
    7. Set `SMOKE_TEST = False`. **Save Version → Save & Run All (Commit) → Save.** Stop the editor session. It stops cleanly at 11 h if it runs long.
    8. When it's finished:
       - **Output:** download the `L0S2` folder so that `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\L0-step2-results\new-partA\L0S2\README.md` exists.
       - Save the run notebook as `Champion_Task\l0-step2-full-size-rescore-new-partA.ipynb`.
       - Tell me "new part A done".
  - **Part B on account 2 (abrar2222864642), about 5.2 h:** the same steps, with these differences:
    - account 2 already has the bench mirror and `HF_TOKEN`;
    - in Cell 2 set `MODEL = "new"`, `RUN_TASKS = ["DRE", "DSR", "DTR", "ORR"]`, `SMOKE_TEST = True`, and expect `✓ Wrote l0_step2_new_results.csv …` at the end of the smoke test;
    - then the full run;
    - results to `...\Champion_Task\L0-step2-results\new-partB\L0S2\README.md`, the run notebook saved as `l0-step2-full-size-rescore-new-partB.ipynb`;
    - tell me "new part B done".
  - **Then on your PC:** the verdict from all parts together, against the old 7B's full-size answers, on all questions and on E5's clean ones. Done 28 Oct: kept on all, clean and strict.
- **Done 27 Oct: the merge (item 1) and R16c step 5b (item 2).** Their steps stay as the record of how they ran.
  1. **L0 step 2: the merge** ([l0-step2-merge.ipynb](Champion_Task/l0-step2-merge.ipynb)).
     - **Done 27 Oct (run 2, account 3, 10 min): merged, checked and pushed, as expected below.**
       - Cell 2: checkpoint-28 restored, step 28 of 28, 196 LoRA pairs (88 saved name patterns), last three losses as in session 3b's log.
       - Cell 3: 196 of 729 weight tensors changed, exactly the adapted projections, with no inf or NaN (merge 4.3 min). The new 7B, 7 files and 16.6 GB, is public on Hugging Face, every file at its local size (upload 2.4 min).
       - Faster than the 30–45 min expected. The run notebook's code matches the build cell by cell, and it replaced the output-free copy.
       - Results in [L0-step2-results/merge/](Champion_Task/L0-step2-results/merge/L0S2-merge/README.md). Its README says parts A, B and C; the plan is the two parts above.
     - **What and why:** session 3b finished the training (28 of 28 steps; checkpoint-28 is on Hugging Face). Its merge stopped on `ImportError: Found an incompatible version of torchao`: Kaggle ships torchao 0.10.0, and the upgraded peft checks it only when merging into an fp16 model.
     - **The copy:** a named copy of the training notebook's Cells 1, 2 and 6. It removes torchao, pins peft 0.21.2 (the version that trained the adapter), downloads only checkpoint-28's adapter, merges on the CPU and pushes `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP`.
     - **Checks inside:** exactly the 196 adapted projections change and nothing else; no inf or NaN; every pushed file has its local size. The weight checks run before the push.
     - **Dry run on your PC:** a tiny Qwen2.5-VL with a random adapter and a stand-in Hugging Face.
       - The 14 projections changed and nothing else.
       - The merged weight equals base + 2·B·A to within 3e-5.
       - Only the adapter files were downloaded.
       - An adapter with one projection left at zero was caught, and nothing was pushed.
     - **Expectation, fixed before the run:** 196 tensors change, all of them language-model projections; about 16.6 GB pushed; 30–45 min.
     - **Run 1 (27 Oct, account 3, 35 s): stopped in Cell 2 on a check of mine, not on a fault in the adapter.**
       - Cell 1 worked: peft 0.21.2, torchao gone, 34 GB of RAM, 1,126 GB of free disk.
       - Cell 2 then raised `AssertionError: the adapter targets 88 modules, not 196`.
         - Given more than 20 module names, peft saves the shortest name patterns that pick the same modules: q/k/v/o_proj plus each layer's three MLP names, 4 + 28 × 3 = 88. The adapter still holds one LoRA pair for each of the 196 modules.
         - My dry run's 2-layer model had only 14 targets, under peft's threshold, so it never showed this.
       - Nothing was merged or pushed.
       - Log: [L0-step2-results/merge/run1-stopped-at-cell2/](Champion_Task/L0-step2-results/merge/run1-stopped-at-cell2/l0-step2-merge.log).
     - **Fixed (27 Oct):**
       - Cell 2 now counts the adapter's LoRA pairs (must be 196) instead of the saved name patterns. `merge_check.json` records both numbers.
       - Dry-run again with a 4-layer model, where peft does shorten the names (28 pairs saved as 16 patterns): it merged exactly the 28 projections, and the broken adapter was still stopped before any push.
       - Cell 3's own check (exactly the 196 projections change) was already right.
     - **Run steps (run 2):**
       1. On Kaggle as redwannewaz (account 3), open the merge notebook that stopped. Use **File → Import Notebook → Browse**, pick the fixed `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\l0-step2-merge.ipynb`, then **Import**. This replaces the notebook's cells; its settings and secrets stay.
       2. Check the settings: Accelerator **None**, Internet **On**, **Add-ons → Secrets:** `HF_TOKEN` ticked.
       3. **Save Version → Save & Run All (Commit) → Save.** About 30–45 min.
       4. A good run's log shows:
          - Cell 1: `peft: 0.21.2`, `torchao: not installed`, then the RAM and free-disk line.
          - Cell 2: `✓ checkpoint-28 restored: step 28 of 28; LoRA rank 64, alpha 16, 196 LoRA pairs (88 saved name patterns); last three losses [0.4905, 0.5365, 0.5335]`.
          - Cell 3: `changed 196 (expected 196); outside the targets 0; targets unchanged 0; with inf or NaN 0`, then `✓ Pushed AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP (… files, about 16.6 GB) … every file on the Hub has its local size`.
       5. When it's finished:
          - **Output:** download the `L0S2-merge` folder and unzip it so that `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\L0-step2-results\merge\L0S2-merge\README.md` exists.
          - Download the run notebook too, and save it over `Champion_Task\l0-step2-merge.ipynb`.
          - Tell me "merge done".
       6. **Then:** the new 7B's six tasks in the notebook's two parts: A (BDC and BBR, about 8.6 h) on account 4, B (DRE, DSR, DTR and ORR, about 5.2 h) on account 2. Their steps are in the item above.
  2. **R16c step 5b: a third grader seed (44)** ([r16c-step5b-third-grader-seed.ipynb](Champion_Task/r16c-step5b-third-grader-seed.ipynb)).
     - **Done 27 Oct (account 1, 2.3 h): seed 44 scores 0.2982 on the clean pairs, so O1 is met on the three-seed mean only (0.3065), narrowly.** R11 reproduced and the rebuild check said yes. Details at the top of this file.
     - **What and why:** O1 on clean data rests on step 5's SAM 3-only map (0.3161 and 0.3051), and seed 43 clears 0.30 by only 0.005. A third seed tests that before O1 is claimed.
     - **The copy:** step 5's notebook with seed 44 only and its own output folder (`step5b`). Every score is also computed on the 605 clean pairs (E5's 40 listed in Cell 2). The log-probabilities are saved with 6 decimals, and a rebuild check compares the saved rows with the maps' own counts (step 5's seed 43 rebuilt only to within 0.0012).
     - **Dry run on your PC:**
       - Cell 10 on step 5's real per-building data, relabelled as seed 44:
         - R11 reproduces its D4 CSV;
         - "seed 44" gives seed 42's published 0.552 / 0.3347 and its clean 0.5469 / 0.3161;
         - the verdict follows the rule in all three cases;
         - the rebuild matches exactly.
       - Cells 9 and 10 end to end on synthetic images:
         - the clean counts equal a run without the leaked images;
         - the saved rows rebuild the maps exactly.
     - **Expectation, fixed before the run:** seed 44's SAM 3-only map scores about 0.30–0.32 Destroyed IoU on the clean pairs. **O1's verdict:**
       - every seed (42, 43, 44) above 0.52 mIoU and 0.30 Destroyed IoU: met on clean data on three seeds;
       - otherwise, if the three-seed mean passes: met on the mean only, narrowly;
       - otherwise: not met on clean data.
     - **Run steps:**
       1. Sign in as abrarmohammedtanzim (account 1, where E3's and step 4's outputs are). **Create → New Notebook → File → Import Notebook → Browse**, pick `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\r16c-step5b-third-grader-seed.ipynb`, then **Import**.
       2. **Session options:** GPU **T4 x2** (one is used), Internet **On**.
       3. **Input → + Add Input:**
          - `disasterm3-mirror` (by abrarmohammedtanzim);
          - **Your Work → `e3-sam3-building-finding`**;
          - **Your Work → `r16c-step4-sam3-training-images`**.
       4. **Add-ons → Secrets:** `HF_TOKEN` ticked (R11's checkpoint is private).
       5. **Smoke test:** in the cell starting `# ── Cell 2: Configuration and inputs`, set `SMOKE_TEST = True`, then **Run → Run All**. If Cell 1 says "RESTART THE KERNEL NOW", do **Run → Restart & clear cell outputs**, then **Run All** again. Cell 3 still builds all 6,443 masks, so allow 15–75 min. Check:
          - Cell 2: `✓ R16c step 5b … [SMOKE TEST] … 2 epochs; seeds (44,)`, then `(645 maps)` and `(5,798 maps)`;
          - Cell 3: `✓ Built 6,443 valid (image, mask) pairs` and `Before-image not found: 0`;
          - Cell 4: `✓ Grader training images: 20; held out: 4; validation: 8`;
          - Cell 7: `✓ Seed 44: kept epoch …`;
          - Cell 10: `The saved rows rebuild the maps' scores: yes`, then `✓ Wrote /kaggle/working/step5b: …`.

          Any red error: stop and send me a screenshot.
       6. Set `SMOKE_TEST = False`. **Save Version → Save & Run All (Commit) → Save.** Stop the editor session. About 2.5–3.5 h.
       7. When it's finished:
          - **Output:** download the `step5b` folder so that `D:\CSE499AB_project\Essential_task_delegation\Champion_Task\R16c-results\step5b\README.md` exists.
          - Save the run notebook over `Champion_Task\r16c-step5b-third-grader-seed.ipynb`.
          - Tell me "step 5b done". I'll check that R11 reproduces and the rebuild check says yes, then read O1's verdict by the rule above.
- **Ridita pushed `audit_outputs/` (commit `74d44059e`, pulled 26 Oct).**
  - Checked: its `val_train_leaks.csv` matches ours; its README is the copy she sent; there are no secrets.
  - Two things for you:
    - **Tool traces in a committed file:** her README's lines 245 and 253–255 quote temporary-folder paths from her PC that name the tool she used. If your rule on tool traces in committed files covers teammates' files too, ask her to remove those paths.
    - The same commit removed the Sphere handbook PDF and git-ignored it (next item).
- **The guideline texts in the public repo (licences checked 28 Oct). Parked by you on 28 Oct: licences of the guideline documents and the literature-review papers aren't a concern while the work is in progress.**
  - **What each document allows:**
    - **Sphere Handbook 2018:** its copyright page says it "may be reproduced without fee for educational purposes … provided that the copyright holder is acknowledged", but "for … posting online … prior written permission must be obtained" (info@spherestandards.org). Its PDF is in the public history (added in `0bf593a13` on 25 Oct, removed on 26 Oct), and 61 sections of its text are in the current files.
    - **EMS-98 (1998):** © Centre Européen de Géodynamique et de Séismologie, Luxembourg, with no reuse grant. Three copies are tracked (`Offline_RAG_KR2/guidelines/`, `disasteller_offline/Guidelines/` and `support/…/DisasTeller/Guidelines/`).
    - **xBD paper (2019):** © Carnegie Mellon University, on arXiv under arXiv's own licence, which lets only arXiv distribute it. Its "unlimited distribution" line is a US defence release marking that points back to the copyright notice.
    - **INSARAG Guidelines 2020 (Vol II Manual B, Vol III) and IWG-SEM v1.0:** no licence or copyright notice in the PDFs or on the publishers' sites, so ordinary copyright applies.
    - All six also sit, as text, in `guideline_chunks.jsonl` and `local_chroma_db/`.
  - **Options:**
    1. **Ask for permission,** Sphere first (a student, non-commercial project is the use most likely to be allowed). Until an answer comes, the texts stay up.
    2. **Take the PDFs and their text out of the current public files.** Keep the search code, a list of official links, and a script that rebuilds the index from copies each user downloads. The Kaggle runs keep the private `offline-rag-kr2` dataset, which already holds the index. Old commits still hold the files.
    3. **Option 2, and remove them from the history** (a rewrite and a force-push): every teammate must clone again, and the commit hashes quoted in these docs change.
    4. **Make the repository private** until a clean public copy is made for the showcase.
  - **Recommended:** option 2 for all six now (the search loses nothing), and for Sphere also option 1 or 3, since its terms name posting online. The 30 research papers in `others/lit review_research papers/` raise the same question; papers under CC BY may be shared, those under arXiv's own licence may not.
- *(Earlier note, kept for the record.)* **The repository is public,** and Ridita's commit added `Offline_RAG_KR2/guidelines/Sphere_Handbook_2018_EN.pdf`, which her own README says needs Sphere's permission to put online. *(26 Oct: her commit `74d44059e` removed it from the current files and git-ignored it. It is still in the history, commit `0bf593a13`.)* Decide whether to purge it from the history. *(27 Oct: the handbook's text is also still in the public repo. `Offline_RAG_KR2/guideline_chunks.jsonl`, which is tracked, holds 61 chunks of it, about 100 KB, for the offline search. If Sphere's permission matters, those chunks need the same decision as the PDF.)*
- **Open offers waiting for your answer** (listed here so none gets lost):
  - a Contributions section for the report;
  - how many hours a week you can give the website (T3's timeline assumes about 3 weeks part-time);
  - continue the task-by-task walkthrough for your supervisor update (E1 to E3 are done; H1 V1 next);
  - CraftX: record it in Paper_Audit and delete its saved page;
  - Noman as an extra GPU account (paused by you, 22 Oct).

- **Final showcase: about 15 Dec** (tentative, given 24 Oct), about 7 weeks away.
- **Order agreed 24 Oct:** T2, T3, L0 and L2 before any work on Layers 2–4; the 3D damage city goes into T3's website. H2c follows L2. Layer 3: decide in mid-November.
- **H2b is done (23 Oct):** no model passes, so the system ships fixed queries only (the rule fixed 22 Oct). Say if you want a follow-up: a menu of fixed question templates, or question–SQL pairs in L2's training. Either one needs a new question bank for its test.

GPU now (27 Oct): four accounts.
- **Account 4 joined on 27 Oct:** Tamanna's Kaggle account, with 28.5 GPU hours.
  - **Before its first run it needs:**
    - the `HF_TOKEN` secret (AbrarAlam), for the private model repos;
    - access to the datasets `abrarmohammedtanzim/disasterm3-mirror` and `redwannewaz/disasterm3-bench-mirror-v2` (attach them by name; if they're private, share them with the account first);
    - a phone-verified account, which Kaggle requires for GPU and internet.
  - It can't run step 5b, which reads account 1's own notebook outputs (E3, step 4).
- This week so far: **account 1** (`abrarmohammedtanzim`) 3.8 h for R16c step 5; **account 2** (`abrar2222864642`) about 1.5 h for E5 (step 1 used no GPU); **account 3** (`redwannewaz`) 2.0 h for L0 session 3b, plus a 35 s CPU session for the merge's first run.
- **Now (28 Oct):** all four accounts are free. Step 5b used 2.3 h of account 1, and the new 7B's parts A and B used 7.8 h and 5.9 h.
- **Next,** on your go: L2 step 1 again with the new teacher (about 2 h), then L2's 3B distillation (about 28 h), on account 4's remaining hours and the others'.
- *(The plan below, from 20 Oct, is kept for the record.)*
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
| O1 Segmentation champion | mIoU above 0.52 and Destroyed IoU above 0.30, on two seeds | mIoU 0.581 (R11). Destroyed IoU 0.300 on one seed (R9 with TTA); the two-seed mean is 0.297. R12's 8,021 extra destroyed buildings changed nothing (TTA: 0.588 and 0.295). E4: almost all of the gap is Turkey's earthquake (37% of destroyed pixels, Destroyed IoU 0.000). Without it, R11 reaches 0.410. **R16c (SAM 3 + the U-Net's pieces, the stronger grader) passes the gate on both U-Net seeds and both grader seeds:** with R11, 0.587 / 0.316 and 0.585 / 0.311 (step 2); with R11s7, 0.584 / 0.309 and 0.583 / 0.301 (step 3, 21 Oct); mean of the four 0.585 / 0.309. Caveats: the weakest combination clears 0.30 by only 0.0006, and against R11s7 the keep rule fails, because it costs about 0.007 mIoU. **26 Oct: a train/validation leak** (Ridita's corrected audit; checked in [E5](Champion_Task/E5-results/E5/README.md)): `socal_fire` and `woolsey_fire` share before-images, so some validation pairs are near-copies of training pairs. Removing only the 21 leaked pairs found locally puts R16c at 0.2915–0.3070 (one of four below 0.30); the audit counts 40. **26 Oct, E5 done (40 pairs, three checks agree):**
- **Clean (605 pairs):** R16c steps 2 and 3 score 0.2838–0.2980, so they fail. R16c step 5's SAM 3-only map scores **0.3161 and 0.3051** (mIoU 0.547), the only map that passes, on two grader seeds.
- R11 and R11s7 lose 0.027 Destroyed IoU to the leak (the after-only U-Nets 0.012–0.013). With TTA, R11 is now 0.2692 against R9's 0.2874.

Caveats on the SAM 3 map: it was singled out after seeing the scores; seed 43 clears 0.30 by 0.005; seed 43's file rebuilds its maps only to within 0.0012. | **Met on clean data by one map, on the mean of three grader seeds only, narrowly** (0.3161, 0.3051 and 0.2982; mean 0.3065; step 5b, 28 Oct) |
| O2 Counting and six tasks (KR2) | Building counting above the base model's 34.2%; all six tasks re-scored | Like for like with the DisasterM3 paper's Table 2, which scores optical after-images only: our 7B (greedy) averages 36.1% on the six tasks against the base model's 31.2%. It counts buildings at 32.0% against 34.2%, 2.2 points short. With the radar questions included, which the paper's table leaves out, our scores are 35.6% and 30.5%. The paper's own fine-tune of the same 7B reaches 40.4% (counting 34.3%). These are at 512 px. At the paper's full image size (L0 step 1), our 7B scores 75.6% on disaster type and 31.1% on relational (71.1% and 23.3% at 512 px); the other four tasks haven't been re-scored at full size yet. *(Later: all six were re-scored at full size on 20 Oct, mean 38.37%.)* Our best counting is R11's counts (H2), at 32.96% on the optical questions. **26 Oct, E5:**
- 186 of the 10,675 optical questions use an after-image copied from Instruct.
- Without them, R11's counting is 32.47%, so the gap to 34.2% grows from 1.24 to 1.73 points. The bar is the paper's base model, which never saw Instruct, so it's unaffected.
- The old 7B's six tasks at full size: mean 38.37% → 38.14%. Every score drops by at most 0.7 points. | Half met (counting about 1.7 points short on clean questions) |
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
| E5 | Train/validation leak and the benchmark's copies | Done (26 Oct: steps 1 and 2 on account 2, then every affected score re-scored on your PC) | Step 1 Kaggle CPU, 0.54 h; step 2 Kaggle T4, 0.91 h, account 2 | 40 leaked pairs, confirmed by three checks. Clean, R16c step 5's SAM 3-only map is the only map passing O1's gate (0.3161 and 0.3051); steps 2 and 3 fail (0.2838–0.2980). R11 and R11s7 lose 0.027 Destroyed IoU, twice the after-only U-Nets' loss, so R11 falls 0.018 below R9 on clean data. Benchmark: 186 of 10,675 optical questions use a copied after-image; scores drop by at most 0.7 points; O2 counting 32.96% → 32.47%. Tables in [E5-results/E5/](Champion_Task/E5-results/E5/README.md). | Group every future split by before-image fingerprint. Report both numbers (all and clean) from now on. |
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
| R16c | A stronger R16 grader, tested rigorously | Done (step 5b, the third grader seed, finished 27 Oct: seed 44 scores 0.2982 on the clean pairs, so O1 on clean data is met on the three-seed mean only, 0.3065; step 5 finished 26 Oct) | Step 1 your PC (2.6 min); step 2 Kaggle account 1, T4, 1.4 h; steps 4 and 5 account 1, about 5–6 h; step 3 account 1, about 0.5 h | Go given 19 Oct. Steps renumbered 1–5 on 20 Oct (your decision): step 3 was 2b, step 4 was 3a, step 5 was 3b.

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
- The two seeds are the grader's. R11 (seed 42) is the same in both.

**Step 5 done (26 Oct, account 1, T4, 3.8 h against the planned 1.5–2 h; the graders took 1 h and validation 36 min).**
- **It doesn't replace step 2.** Its SAM 3 + R11 map scores 0.2434 and 0.2158, against step 2's 0.3159 and 0.3106. This grader learned on SAM 3's outlines and over-calls R11's pieces as destroyed.
- **Its SAM 3-only map passes O1's gate on both grader seeds:**
  - all 645 pairs: 0.552 / 0.3347 and 0.552 / 0.3242;
  - E5's clean 605 pairs: 0.547 / 0.3161 and 0.547 / 0.3051.
  - It's the only map that passes on clean data.
  - It isn't kept against R11, because its mIoU is 0.035 lower.
- On the true buildings, macro F1 is 0.737 and 0.730 (step 2: 0.778 and 0.784).
- The Destroyed bias hit the search grid's edge (−4.0) in 38 of 40 epoch searches.
- Seed 43's per-building file rebuilds its maps only to within 0.0012 (seed 42's exactly); its clean score uses the lower value.
- Results in [R16c-results/step5/](Champion_Task/R16c-results/step5/README.md); the ledger has the details. | Decided 20 Oct: steps 4 and 5 run now on account 1, as a stretch (step 5 replaces step 2 only if it beats it by the keep margin on both seeds). Step 3, the same graders with R11s7's pieces, follows R11s7. **26 Oct:** all five steps done. Step 5b, a third grader seed (44) for step 5's SAM 3 map, was built and dry-run checked after your go; it checks that seed 43's narrow clean pass isn't luck (about 2.5–3.5 h on account 1). Run steps under "Needs you now". |

### System track

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| H2 | Per-building table and SQL counting agent | Done (22 Oct) | Kaggle T4 ×2, account 2, about 1.5 h | **R11's counting on the benchmark: 32.96% on the optical questions** (27.46% overall), against 26.51% for R9's tuned counter and 34.2% for the base 7B. Both models wrote valid SQL for all 100 sampled questions, and their counts matched the table on 99 and 100. Results in [H2-results/H2/](Champion_Task/H2-results/H2/README.md). | Its table feeds H2b, T2 and T3. Counting is still 1.2 points under KR2's bar. |
| H2b | Graded test of the SQL agent | Done | Noman's Kaggle account, 23 Oct, 47 min | No configuration passes (bar: 90% test-suite accuracy and RS(10) above 0). Best: Coder-3B in fp16, 73.9%. Our 7B in 4-bit: 69.0% [57.1, 80.5], RS(10) −89.6, with the most refusals of unanswerable questions (94%). The base VL-3B in 4-bit: 46.4%, RS(10) −365.7. Results in [H2b-results/H2b/](Champion_Task/H2b-results/H2b/README.md). | By the rule fixed 22 Oct: fixed queries only, no free-form questions. Any follow-up is your call; the re-run on the distilled 3B is needed only if free-form questions come back. |
| H2c | Free-form questions, second try | Not started | CPU (data), Kaggle (adapter and test) | — | After L2: question–SQL pairs, a fine-tuned adapter, a five-sample agreement gate, a new test. Pass rule fixed 24 Oct (90% and RS(10) above 0); if it fails, the system stays as it was. Steps in the roadmap. |
| H3 | VLM input upgrades (before + after, numbered marks, crops) | Not started | Kaggle GPU | — | After H1 V2. |
| H4 | GEOBench-VLM external counting test | Not started | Kaggle GPU | — | — |
| T2 | DisasTeller on local models | Not started (steps 4, 4b and 6 done by Ridita on 27 Oct, checked 28 Oct) | Kaggle T4 ×2 | The guideline search passed its retrieval test (27 of 30) and runs on Kaggle with the internet off; the DisasTeller copy is offline by default. | The biggest missing O3 piece: steps 1–3 (the LLM and the three image tools still call Gemini), then 5 and 7. |
| T3 | Offline demo and public showcase (Vercel website) | Not started | — | — | Re-planned 24 Oct: a public showcase site and an interactive demo website on Vercel, with a live offline mode at the booth. You build it and choose the name and look later. In-browser feasibility test passed 24 Oct, in two runs: INT8 on WebAssembly took 2.1–5.5 s per image in Chrome and Edge, with output identical to the native run (System_Design_Notes §4.3; kit in `Champion_Task/T3-browser-test/`). The rest after T2. |
| L0 | A better 7B teacher before distilling | Done (28 Oct: the new 7B is kept by the rule fixed 19 Oct, on all questions and on E5's clean and strict sets: ORR +3.50, DTR +1.13, BBR −0.95, the others +0.22 to +0.64. It is L2's teacher. Step 2's training is complete: session 3b did steps 26–28 on 26 Oct, and checkpoint-28 is on Hugging Face. Its merge stopped on a library version clash (torchao), so a merge-only copy, `l0-step2-merge.ipynb`, merged it on 27 Oct (run 2, 10 min): exactly the 196 adapted projections changed, and `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP` is on Hugging Face. Earlier: the old 7B's full-size re-score and the smoke test on 20 Oct; sessions 1, 2 and 3a did steps 1–25 on 21–22 Oct) | Step 1: Kaggle account 1, T4 ×2, 1.7 h. Step 2: the old 7B's re-score of four tasks (about 11 h, account 3 then account 2), training (about 7–14 h on T4 ×2, account 3), then the new 7B on six tasks (about 12 h) | Decided 19 Oct: improve the 7B before distilling.

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
- Gaps to the paper's fine-tune at 512 px were −12.5 (disaster type) and −12.9 (relational); at full size they are −8.0 and −5.1. | Decided 19 Oct: step 2, a short extra training of our adapter at full size (1036 px) on disaster type and relational plus an equal replay sample (about 7,000 samples). Both models are then scored on every optical question. Done 28 Oct: kept. Next: L2 step 1 again with the new teacher. |
| L2 step 1 | Teacher answers for the 7B → 3B distillation | Ready to run (again, with the new 7B: `l2-step1-new-teacher.ipynb`, built 28 Oct; the first run is done) | Kaggle account 1, T4 ×2 | 4,000 answers generated; 1,085 (27%) match their label and are kept: disaster type 89.4%, land use 50.9%, bearing bodies 42.7%, building counting 15.6%, relational 13.3%, road counting 0%. Road counting labels are road-area percentages such as 5.22%, which the teacher can't hit exactly without the options; those 735 fall back to their labels. Cell 7's push failed: the dataset title already existed, and the code missed Kaggle's wording ("already in use") but still printed "Pushed". So the dataset `disasterm3-teacher-cot` doesn't hold this run's file. | L0 kept the new 7B, so run `l2-step1-new-teacher.ipynb` on account 1 (steps under "Needs you now"). Step 2 reads the manifest from the run's notebook output, not from a dataset. |
| L2b | Explanations as a second skill ("why?" on request) | Not started | Kaggle T4 ×2, about 3–5 h | Planned 19 Oct (roadmap L2b, Paper_Audit §4.12): explanations written from the correct answer plus true mask facts, checked against those facts, then trained as a second task in step 2. | Build after step 1. |
| L2 step 2 | Train the 3B | Not started | Kaggle GPU | Its notebook still has the old installs that fail on today's Kaggle. | Fix it, as a copy, after L0. How the 3B learns is still to choose (a, b or c; soft labels recommended). |
| L3 | 4-bit 3B with handoff to the 7B | Not started | Kaggle GPU | — | After L2. |
| L4 | EfficientSAM3 against SAM 3 | Not started | Kaggle GPU | — | — |

### Housekeeping

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| — | Commit this week's work to git | Done | Your PC, pushed 19 Oct as `1b33574`, 22 Oct as `a736299` and `56df1c5`, 26 Oct as `5ff744c`, and 28 Oct as `bd934391b` (after `7505d2a29`, the rename of Aryan's `BDC_Results.`) | `Essential_task_delegation\` and `others\`, plus `.gitignore`, `README.md` and `UPDATE.md` whenever they're modified (your rule, 19 Oct). The `.gitignore` was updated 19 Oct, so these stay local: bulk masks and maps that no notebook reads, L2 step 1's 71 MB training file, personal samples, the next report's draft and the editor's local settings. | Next commit: same scope, when you ask. |
| — | Update the roadmap and README with these results | Not started | — | — | Say when. |

## Change log

- **28 Oct 2026 (update day): faux update 2 PDF and faux literature deck built.** Your part of the PDF covers Phase A only (E1, E2, D5/D5b, E3); the teammates' sections are kept from update 2. Details under the 5th-class item in "Needs you now".
- **28 Oct 2026 (night): L2 step 1 again built; guideline licences checked.**
  - `l2-step1-new-teacher.ipynb`, a named copy of the first run's notebook, built on your go and dry-run checked on your PC (the first run's subset reproduced, its 1,085 kept answers re-kept, resume and outputs tested). L2 step 1 marked Ready to run; the run board and "Needs you now" updated.
  - Licences read from each document's own copyright page or text, the arXiv page and the publishers' sites. The Sphere item became the guideline-texts decision, with four options.
- **28 Oct 2026 (later): teammates' 27 Oct pushes reviewed.**
  - Ridita: retrieval test recounted (27, 20 and 25 of 30), the internet-off Kaggle run's metadata and outputs checked, the offline DisasTeller copy compared with the original. No secrets or tool traces in her folder.
  - Aryan: v4's per-question files rescored on your PC, with R11's table alone, paired tests and E5's clean filter. No secrets or tool traces.
  - Tamanna: the two committed demo runs, their tiles against the validation split and E5's list, and the VLM's answers checked. No secrets.
  - Recorded in the ledger (§9), the roadmap (T2 steps 4, 4b and 6), System_Design_Notes §2.2 (why counts skip the VLM), the T2 row and "Needs you now".
- **28 Oct 2026: L0 step 2's verdict (kept) and R16c step 5b (O1 met on the three-seed mean only).**
  - The new 7B's parts A and B and step 5b downloaded and checked: every run notebook matches its build except the settings it was told to change; R11 reproduces; step 5b's rows rebuild its maps.
  - Moved into the pattern: `L0-step2-new-2results/` → `L0-step2-results/new-partA/`, `L0-step2-new-results/` → `new-partB/`, `R16c-step5b-third-grade-results/step5b/` → `R16c-results/step5b/`; the run notebooks renamed `l0-step2-full-size-rescore-new-partA.ipynb` and `-new-partB.ipynb`.
  - The verdict computed on your PC on all, clean and strict questions, with paired tests. L0 and R16c marked Done; the run board, "Needs you now" and the GPU plan updated. All four accounts are free.
  - Pushed on 28 Oct: `7505d2a29` (the rename of Aryan's `BDC_Results.`) and `bd934391b` (this week's work, the update 2 PDF and the literature review deck).
- **27 Oct 2026: Phase A re-checked for the supervisor update.**
  - E1's numbers all reproduce from its per-image counts. On the 605 clean pairs, 57.7% of R9's misses are called background (58.4% with TTA), so Gate G2 holds.
  - R12's extra tiles, checked by content against all 645 validation pairs: 2 twins outside E5's 40, so R12 trained on 42 leaked pairs. Its verdict stands. Recorded in the E5 README, the ledger (R12) and the roadmap (E5, R12).
  - E3 on clean data: SAM 3 0.6302 against R9's 0.5905, so Gate G3 holds. E2's verdict can't change (its drop is five times the leak's effect).
- **27 Oct 2026 (night, later): L0's merge done (run 2, account 3, 10 min).**
  - Every check passed: exactly the 196 adapted projections changed, of 729 weight tensors; no inf or NaN; the new 7B (7 files, 16.6 GB) is public on Hugging Face with every file's size matching.
  - Results moved from `l0-step2-merge-results/` to `L0-step2-results/merge/`. The downloaded run notebook replaced `l0-step2-merge.ipynb`; its code matches the build.
  - The new 7B's parts A (account 4) and B (account 2) can start now. Account 3 is free.
- **27 Oct 2026 (night): 5th-class deliverables.**
  - `others/CSE499B_update_2.tex` and `.pdf`: progress report update 2, 2 pages, in update 1's format.
  - The literature review deck: 21 slides, one per paper for 16 papers, as a web deck to export as PPTX.
  - Both are listed under "Needs you now" with the remaining steps (print, export, push).
- **27 Oct 2026 (later):** the merge's run 2 is running on account 3 and R16c step 5b on account 1. Accounts 2 and 4 are free.
  - The new 7B's evaluation now uses the re-score notebook's own two parts, so account 3 isn't needed: A (BDC, BBR) on account 4, B (DRE, DSR, DTR, ORR) on account 2.
  - Both start once the merge has pushed the model. Their run steps are under "Needs you now".
  - The run board is updated.
- **27 Oct 2026:** the merge's run 1 (account 3) stopped in Cell 2 on a check of mine.
  - Peft saves the 196 adapted modules as 88 name patterns, so the check misread a correct adapter. Nothing was merged or pushed.
  - Cell 2 now counts the adapter's 196 LoRA pairs. It was dry-run again on a model where peft shortens the names, and run 2's steps are under "Needs you now".
  - Run 1's log moved to `L0-step2-results/merge/run1-stopped-at-cell2/`. The fixed, output-free notebook replaced the downloaded run version.
  - **Account 4 (Tamanna's, 28.5 GPU h) added.** The new 7B's evaluation parts move to accounts 4, 2 and 3.
  - Sphere item: its text is still in the tracked `guideline_chunks.jsonl` (61 chunks).
  - Updated: the ledger, the roadmap and System_Design_Notes.
- **26 Oct 2026, 17:30:** your go ("build 1 and 2"). Two named copies were built, dry-run checked on your PC, and marked ready to run.
  - `l0-step2-merge.ipynb`: the training notebook's Cells 1, 2 and 6, with torchao removed and peft pinned to 0.21.2.
    - It downloads checkpoint-28's adapter only, and checks the weights before pushing.
    - Its dry run: a tiny model, a random adapter and a stand-in Hugging Face. It merged correctly; a bad adapter was stopped before the push.
  - `r16c-step5b-third-grader-seed.ipynb`: step 5 with seed 44.
    - Additions: clean scores, 6-decimal log-probabilities, a rebuild check and O1's verdict rule.
    - Its dry run: Cell 10 on step 5's real data reproduced R11 and seed 42's clean 0.3161; Cells 9 and 10 ran end to end on synthetic images.
  - **Correction:** step 5b takes about 2.5–3.5 h, not 1 h. Step 5's masks and crops alone took about 2 h.
  - Updated: the run board, Needs you now (with run steps), the R16c and L0 rows, the ledger, the roadmap and the E5 README.
- **26 Oct 2026, 15:45:** checked the downloads of E5 steps 1 and 2, R16c step 5 and L0 session 3b, plus Ridita's pushed `audit_outputs/`.
  - **Moved into the pattern:** `E5-step1-results/E5-step1/` → `E5-results/step1/`; `E5-step2-results/` → `E5-results/step2/`; `R16c-step5-results/` → `R16c-results/step5/` (each with its `__huggingface_repos__.json`); the L0 log → `L0-step2-results/train-session3b/`. The run notebooks replaced their output-free versions, as before.
  - **E5 done.** Every affected score was re-scored, with two numbers each.
    - O1 on clean data rests on R16c step 5's SAM 3 map (0.3161 and 0.3051).
    - R11 lost twice as much to the leak as the after-only U-Nets.
    - O2's counting is 32.47% on clean questions.
  - **R16c step 5:** doesn't replace step 2. Seed 43's file rebuilds its maps only to within 0.0012; the cause is unknown, and the lower value is used.
  - **L0:** training complete; the merge failed on torchao. A merge-only copy is proposed.
  - **Decisions added to "Needs you now":** O1's claim (step 5b recommended), the merge copy, and the AI-trace lines in Ridita's README. The Sphere item now notes that her commit removed the PDF.
  - Also updated: the run board, goals, task rows for E5, R16c and L0, the ledger, the roadmap, the E5 README, System_Design_Notes, Paper_Audit, Phase2_Plan and Research_Directions.
  - Graders (`*.pth`) stay local and git-ignored. No checkpoints were downloaded, so nothing to delete.
- **26 Oct 2026, 10:45:** Ridita's `val_train_leaks.csv` arrived. It matches `leaked_pairs_40.csv` exactly (40 pairs, the same twins and destroyed pixels), so the clean scores of 10:30 stand. Noted in the E5 README, the ledger and Ridita's prompt file.
- **26 Oct 2026, 10:30:** Ridita's audit README settles E5's count at 40: our 21 plus 19 whose socal_fire side isn't in xBD tier 1.
  - The list is saved as `E5-results/E5/leaked_pairs_40.csv`. It matches her 155,812 destroyed pixels and E1's per-image ground truth.
  - Clean scores without the 40: R16c 0.2838–0.2980 (O1 not met on clean data), R9 with TTA 0.2874; G2 still holds.
  - Still to come: step 2 (R11 and R11s7) and step 1 (the benchmark).
  - Updated: the E5 README, ledger, roadmap, System_Design_Notes, Paper_Audit, Phase2_Plan and Research_Directions.
- **26 Oct 2026, 10:00:** E5's re-score widened: every score a leak touched gets a clean version, with no 1% threshold (validation leak: R16c steps 2/3/5, R16, R11, R11s7, R9, R1, R9s7, E2; benchmark copies: L0 steps 1 and 2, H1 V1/V2/V2b, H2, R16c step 1's counting). R2–R8 and R10 aren't re-scored, reason in the ledger's E5. Roadmap E5 step 3 and the ledger updated.
- **26 Oct 2026, 09:30:** E5 extended with no GPU run, using E1's saved per-image counts:
  - **Clean scores:** R9, R1 and R9s7 have clean scores now. R9 with TTA goes from 0.3001 to 0.2937 without the 21 leaked pairs.
  - **E1's decision still holds:** 58.7% of missed destroyed pixels are called background without the 21 pairs, against the 40% rule, so R11 would still have gone first.
  - Recorded in [E5-results/E5/README.md](Champion_Task/E5-results/E5/README.md) and the ledger's E5 section.

- **26 Oct 2026, 09:25:** E5 step 1 ([e5-step1-hash-check.ipynb](Champion_Task/e5-step1-hash-check.ipynb)) and step 2 ([e5-step2-per-image-counts.ipynb](Champion_Task/e5-step2-per-image-counts.ipynb)) built, and both dry-run checked on your PC.
  - **The data:** your real `train_release.json` and DisasterM3 masks, with stand-in images.
  - **The split check passes:** 6,443 pairs; 645 validation pairs, 560 of them from xBD; 1,808,111 destroyed pixels. All of the PC check's 21 pairs are in validation.
  - **Planted copies:** every one was found as designed and nothing else matched. They were the known twin, an identical after-image, a recompressed before-image (listed as a near-copy, not counted), and benchmark copies flagged on the right questions.
  - **Step 2** ran with a stand-in model, since `segmentation_models_pytorch` isn't installed on your PC, and wrote all its files.
  - **Ready to run** on account 2; the run steps are under "Needs you now".

- **26 Oct 2026, 09:00:**
  - **Running:** L0 session 3b on account 3, and R16c step 5 on account 1.
  - **E5:** go given for step 1 (hash check) and step 2 (per-image counts), both being built for account 2. E5 now has a row here, a section in the ledger and one in the roadmap.
  - **The leak caveat carried everywhere O1 was stated:**
    - the ledger and roadmap (R16c steps 2 and 3);
    - System_Design_Notes §5;
    - snapshot notes at the top of Paper_Audit §2, the roadmap's §1, Phase2_Plan and Research_Directions.
  - **T2:** step 4 now names Ridita's hybrid search (meaning plus keywords), and a new step 4b tests it against each half alone before it's wired in. System_Design_Notes §1 and §2.1 updated to match.
  - **Open offers** are now listed under "Needs you now".

- **26 Oct 2026, 08:17:**
  - Teammates' pushes reviewed file by file. What's useful from each (Ridita's offline search and audit findings, Aryan's counts-in-the-prompt results, Tamanna's crop idea and two lessons) is recorded in [Task_Ledger §9](Task_Ledger.md), with pointers in the roadmap's T2, R11 and R13.
  - No number from Tamanna's Plan 2 is used: none traces back to a recorded run of the committed code.

- **26 Oct 2026, 07:18:** The prompt for Ridita (collect the leak and BRIGHT audit files, above all `val_train_leaks.csv`) saved as [Ridita_Task/E5_leak_audit_prompt.md](Ridita_Task/E5_leak_audit_prompt.md) and linked from E5 under "Needs you now".

- **26 Oct 2026, 07:10:** The work since 22 Oct pushed as `5ff744c` (39 files): H2b's results in their new folder, E5's leak check, the T3 browser test kit, System_Design_Notes and the doc updates, plus the TimeSpot PDF (added with `-f` past the lit-review folder's ignore rule; arXiv licenses it CC BY 4.0). No teammate had pushed since `a33aa8b`, so nothing was rebased. The saved LinkedIn post (`.mhtml`) stays local.

- **26 Oct 2026, 06:06:**
  - Teammates' pushes since 22 Oct, checked:
    - Ridita added `Offline_RAG_KR2/` (offline guideline search for DisasTeller, combining meaning and keyword search over six guideline sources, plus a building counter for R1's masks) and the corrected audit PDF, and deleted four files in her BRIGHT folder (no links in our docs broke).
    - Aryan added `BDC_PIPELINE_REPORT_v3.md` and a TEOChat notebook. Given counts from ground-truth masks, the fine-tuned 7B answers 96.5% of 1,701 optical counting questions; with counts from the old B1 U-Net, 24.0%.
  - E5 opened: the audit's train/validation leak was checked locally (21 leaked pairs against the audit's 40), and R16c was re-scored without them; O1 is now "at risk". Results in `Champion_Task/E5-results/E5/`.

- **26 Oct 2026, 03:06:** TimeSpot (ICML 2026) recorded in Paper_Audit §4.15, with its citation and BibTeX, and its PDF added to the lit review folder. Its methods were added where they'll be used and cited: L3 (a calibration check before the handoff threshold), H2c (RC-AUC) and T2 (a consistency check of report numbers); also System_Design_Notes §2.4 and §3.6, and the ledger.

- **24 Oct 2026, 06:19:** new section "Pace since Phase A, and the timeline to the showcase": the pace measured from the run board (26 Kaggle runs and 75.5 GPU hours in 6 days, limited by quota), the remaining work in GPU hours, and a projected week-by-week timeline to about 15 Dec.

- **24 Oct 2026, 06:08:**
  - New task H2c (your go): a second try at free-form questions, with its pass rule fixed now and a fallback to the system as it was. Roadmap, ledger and tracker updated.
  - The browser test kit is in `Champion_Task/T3-browser-test/` for teammates and their AI helpers (scripts, pinned packages, results, a README with caveats and an audit checklist). It was re-run end to end from a fresh copy: same output, slower timings (INT8 4.0–5.5 s per image), still within the limit.
  - Final showcase about 15 Dec (tentative). Layer 3 to be decided in mid-November.

- **24 Oct 2026, 06:05:** System_Design_Notes §3.7 added: how eight remote-sensing VLM projects handle and score free-form questions, and what H2b did and didn't rule out (descriptive questions answered by the VLM were never tested).

- **24 Oct 2026, 05:56:**
  - T3's in-browser feasibility test passed on this PC: L1b's INT8 model ran in Chrome and Edge with ONNX Runtime Web in 2.1–3.7 s per image, with output identical to the native run. INT8 on WebGPU failed (an unsupported operation); full precision on WebGPU took 1.1–1.4 s. Details in System_Design_Notes §4.3.
  - System_Design_Notes §3.5–3.6 added: what the literature shows on reliable text-to-SQL, and how H2b scored, with the alternatives and the penalty sensitivity table.

- **24 Oct 2026, 05:44:** your decisions recorded: T2, T3, L0 and L2 come before Layers 2–4, and a 3D damage city goes into the T3 website. The layer estimates and the open choices (Layer 3 if time allows; Layer 4 and real 3D as future work) are in System_Design_Notes §5.

- **24 Oct 2026, 05:42:** your T3 decisions recorded: you build the website, you choose the name and look later, and the in-browser segmenter will be tried after a feasibility test (pass rule fixed in System_Design_Notes §4.3). Roadmap, ledger and design notes updated.

- **24 Oct 2026, 05:34:** T3 re-planned (your go): a public showcase site, and an interactive, animated demo website on Vercel with a live offline mode at the booth. The reasoning, including what last season's winner (DhakaDriveX) showed, is in [System_Design_Notes.md](System_Design_Notes.md) §4; the build steps are in the roadmap's T3 section.

- **24 Oct 2026, 04:32:** new [System_Design_Notes.md](System_Design_Notes.md): the agent frameworks (LangChain, LangGraph, LangSmith), DisasTeller's offline and online modes for T2, and why free-form SQL waits after H2b. Linked from the roadmap (T2 step 6, H2b) and the ledger (H2b).

- **23 Oct 2026, 15:41:** H2b added to [Task_Ledger.md](Task_Ledger.md) and marked done in the roadmap (§7, with its result).

- **23 Oct 2026, 15:35:**
  - H2b done on Noman's account (47 min, all 8 configurations). Checked: the run notebook's code matches ours cell by cell, the table is R11's (117,687 buildings), and the package versions match the smoke tests.
  - Noman pushed the results as `Champion_Task/H2b - Results/` and the run notebook as `Noman_task/h2b-task-notebook-with-output.ipynb`. They were moved to [H2b-results/H2b/](Champion_Task/H2b-results/H2b/README.md) and over the output-free `h2b-sql-agent-graded-test.ipynb`, as with H2's run.

- **22 Oct 2026, 18:26:**
  - H2b's full run handed to Noman, on his own Kaggle account. The notebook (`SMOKE_TEST = False`) and H2's table on GitHub match the local copies.
  - This week's work pushed as `a736299` and `56df1c5`. The second push was first rejected because Tamanna had pushed three commits (her Plan_2 and DisasterVQA folders only). It was rebased onto them; no files in common.

- **22 Oct 2026, 09:50:** session 3a's download `L0-step2-results/results` was renamed to `L0-step2-results/train-session3a/`. Its `checkpoints/` (checkpoint-24 and checkpoint-25, 1.9 GB) were deleted after the log's push line and a sha256 match with Hugging Face.

- **22 Oct 2026, 09:30:**
  - H2b's smoke run was downloaded over the main notebook. It's kept as `Champion_Task/h2b-sql-agent-graded-test-smoke.ipynb`.
  - The main notebook was rebuilt cell by cell, output-free, with `SMOKE_TEST = False` and Cell 9's generation fix. A machine check confirmed every other cell is identical to the smoke run.

- **22 Oct 2026, 08:40:**
  - Session 2's download `L0-step2-results(1)` was moved to `L0-step2-results/train-session2/`.
  - Its `checkpoints/` (checkpoint-20 and checkpoint-24, 1.9 GB) were deleted under the standing rule. The run's log confirms the pushes, and the sha256 of each adapter and optimizer file matches Hugging Face.
  - H2's results are in `H2-results/H2/` (already in the pattern).
  - H2b's smoke test now also loads the base VL-3B.
  - R16c's status was corrected from "Running" to "Needs rerun".

- **22 Oct 2026, 01:20:** H2b built in `Champion_Task/` and added to the roadmap (§7, after H2). The local check's scripts are in a temporary folder, not the project.
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
  - **Tested on your PC:** `l0s2_cpu_batches_check.py`, a temporary script outside the project, confirmed CPU batches, 16 of 16 micro-batches trained, 2 steps of 8, and loss tokens counted.
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
