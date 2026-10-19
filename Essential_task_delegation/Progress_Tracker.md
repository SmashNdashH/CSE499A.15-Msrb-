# CSE499B progress tracker

**Last updated:** 19 Oct 2026, 12:30, R16b done on your PC: correcting the grader by the real grade shares improves every R16 map, but none beats R11. L0 step 1 is built and ready to run on account 1. The `.gitignore` is updated; the commit commands are under "Needs you now". R12 is ready to run on account 2.

Kept by Claude for Abrar. Claude updates this file whenever a task starts running, finishes, or needs a re-run. The task IDs follow [Next_Steps_Roadmap.md](Next_Steps_Roadmap.md); results come from the folders in [Champion_Task/](Champion_Task/).

**Status words:** Done · Running · Ready to run (notebook built, not started) · Needs your decision · Needs rerun · Not started.

## At a glance

| Done | Running | Ready to run | Needs your decision | Needs rerun | Not started |
|---|---|---|---|---|---|
| 17 | 0 | 3 | 0 | 0 | 13 |

## Needs you now

GPU now (19 Oct, 11:50, as you reported): **account 1 has 12 h left, account 2 has 22 h.** R12 needs about 7 h on account 2 and L0 step 1 about 2–3 h on account 1. After those, account 1 has about 9 h and account 2 about 15 h, enough for R15's two runs (about 7 h each).

1. **Run L0 step 1 on account 1 (12 h left), about 2–3 hours:**
   1. Sign in to Kaggle account 1. Create → New Notebook → File → Import Notebook → upload [l0-step1-image-size-check.ipynb](Champion_Task/l0-step1-image-size-check.ipynb).
   2. Session options: Accelerator **GPU T4 ×2**, Internet **On**.
   3. Add Input → Datasets → search `disasterm3-bench-mirror-v2` (by redwannewaz) → Add. That's the only input.
   4. Add-ons → Secrets: turn on `HF_TOKEN` (AbrarAlam) if it's listed. It's optional, because the model is public.
   5. Smoke test: in Cell 2 set `SMOKE_TEST = True`, then Run All. It takes about 15 minutes, mostly the 16 GB model download. Check for these lines:
      - Cell 2: `Optical questions: {'DTR': 10, 'ORR': 10}`;
      - Cell 4: `512` at `504 px`, `train` at `616 px` and `full` at `1036 px`, then one answer per task and size;
      - Cell 6: ends with `Wrote l0_step1_results.csv`.
   6. Full run: in Cell 2 set `SMOKE_TEST = False`, then Save Version → Save & Run All (Commit) → Save. It takes about 2–3 hours, and you can close the tab. Its Cell 2 should show `{'DTR': 353, 'ORR': 858}`.
   7. When it's done: open that version → Output, download the `L0S1` folder into `Champion_Task\L0S1-results\`, then File → Download Notebook into `Champion_Task\`, and tell me. Cell 6's verdict line says which way L0 goes.
2. **Run R12 on account 2, about 7 hours:**
   1. Zip the two folders `Champion_Task\D5-results` and `Champion_Task\D5b-results`, each on its own (right-click → Send to → Compressed (zipped) folder).
   2. Sign in to Kaggle account 2. Datasets → New Dataset → drop in both zips → title `xbd-extra-masks` → Private → Create. Kaggle unpacks each zip into its own folder.
   3. Create → New Notebook → File → Import Notebook → upload [d4-champion-model-R12.ipynb](Champion_Task/d4-champion-model-R12.ipynb).
   4. Session options: Accelerator **GPU T4**, Internet **On**.
   5. Inputs: the DisasterM3 mirror and Bojan Tunguz's "xView2 Challenge Dataset - train and test" and "xView2 Challenge Dataset - Tier 3 Data" should already be listed; add any that's missing with Add Input. Then Add Input → Your Work → Datasets → `xbd-extra-masks` → Add.
   6. Add-ons → Secrets → turn on `HF_TOKEN` (AbrarAlam, with write access).
   7. Smoke test: in Cell 2 set `SMOKE_TEST = True`, then Run All. If Cell 1 says "RESTART THE KERNEL NOW", use Run → Restart & clear cell outputs, then Run All again. It takes about 5–10 minutes. Check for these lines:
      - Cell 4b: two pools, `1,007 tiles` and `612 tiles`, each with an images folder;
      - Cell 7: `self-check passed`;
      - Cell 10: writes `D4_R12.csv`.
   8. Full run: in Cell 2 set `SMOKE_TEST = False`, then Save Version → Save & Run All (Commit) → Save. About 7 hours. Its log should show `1,619 (expected 1,619)` and `8,021 destroyed buildings`.
   9. When it's done: open that version → Output, and download the `d4_R12` folder into `Champion_Task\R12-results\`. Then File → Download Notebook into `Champion_Task\`, and tell me. If the time guard stops it, Run All again with the same settings: it resumes from Hugging Face.
3. **Commit and push the Phase A-onward work** (PowerShell in VS Code): `Essential_task_delegation\` and `others\`, plus `.gitignore`, `README.md` and `UPDATE.md`, which always go in when modified. Nothing else from the root or other folders:

   ```powershell
   cd D:\CSE499AB_project
   git add -- Essential_task_delegation others .gitignore README.md UPDATE.md
   git commit -m "CSE499B Phase A onward: champion-task notebooks, results and docs (E1 to R16b, H1, L0 step 1, L1, L2 step 1, T1, D5, D5b)" -m "Progress tracker, roadmap and paper audit; 7 literature-review papers in others/; .gitignore, README and UPDATE."
   git pull --rebase --autostash origin main
   git push origin main
   ```

   - It commits 7,009 new files, 12 changed files (9 docs, `.gitignore`, `README.md`, `UPDATE.md`) and the removal of the 3 old `R4-results-tillepoch21` files (167 MB, dated 7–19 Oct).
   - `--autostash` lets the pull work while your one `support\` change (Ridita's DisasTeller deep dive) stays uncommitted.
   - Collaborators then run `git pull origin main`. The datasets stay out of git, so local notebooks still need `train_release.json` and `dataset\` on their PC.
4. **When R12 or L0 step 1 finishes:** send its results folder and notebook.

## Goals

| Goal | Target | Where we are | Status |
|---|---|---|---|
| O1 Segmentation champion | mIoU above 0.52 and Destroyed IoU above 0.30, on two seeds | mIoU 0.581 (R11). Destroyed IoU 0.300 on one seed (R9 with TTA); the two-seed mean is 0.297. | Half met |
| O2 Counting and six tasks (KR2) | Building counting above the base model's 34.2%; all six tasks re-scored | Like for like with the DisasterM3 paper's Table 2, which scores optical after-images only: our 7B (greedy) averages 36.1% on the six tasks against the base model's 31.2%. It counts buildings at 32.0% against 34.2%, 2.2 points short. With the radar questions included, which the paper's table leaves out, our scores are 35.6% and 30.5%. The paper's own fine-tune of the same 7B reaches 40.4% (counting 34.3%). | Half met |
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
| E3 | SAM 3 building finding | Done | Kaggle GPU | SAM 3 on the before-image finds buildings better than R9 (building IoU 0.630 against 0.591). Combined with R9, it covers 86% of destroyed buildings' footprints. | Stage 1 of R16. |
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
| R12 | Add the extra xBD tiles (1,007 from part 1, 612 from tier3) | Ready to run | Kaggle account 2, T4, about 7 h | Rebuilt 19 Oct on R11's 6-channel recipe with both pools: 1,619 tiles, 8,021 destroyed buildings: [d4-champion-model-R12.ipynb](Champion_Task/d4-champion-model-R12.ipynb). The tiles join training only, label 255 (un-classified) is ignored by both losses (checked in Cell 7), and it stays at 5,798 samples per epoch. Images come from Tunguz's two xView2 copies on Kaggle, so only the masks (D5-results and D5b-results) need uploading. 1,059 of the tiles are benchmark images, so R12 never answers benchmark questions; R16 does. Other sources (checked 19 Oct): BRIGHT's after-images are radar and its optical ones are already in DisasterM3; in xBD's first part, the tiles DisasterM3 skipped held only 6 of 13,227 destroyed buildings; tier3 has no clean destroyed buildings either (D5b), so its 612 benchmark tiles were added (option A). | Run it on account 2: steps under "Needs you now". |
| R13 | BRIGHT pre-disaster footprints | Not started | — | Overlap checked 19 Oct from DisasterM3's train and benchmark lists: all 14 BRIGHT events are in DisasterM3. 7 have optical after-images there (Turkey, Morocco, Libya, Beirut, Bata and Hawaii in our training data; Noto in the benchmark), so R13 skips them. The other 7 appear only as radar questions, which our optical U-Net can't answer, so their before-images are safe to train on. | After R12. Ridita's converted set includes Noto: drop it. A BRIGHT tile has no optical after-image, so with the 6-channel recipe decide how to fill the after slot (roadmap R13). |
| R14 | Rubble-centred crops (optional) | Not started | — | — | Only with spare GPU time. |

### Phase C: champion and fallback

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| R15 | Best combination plus a second seed | Not started | Two Kaggle accounts | — | Runs regardless (decided 19 Oct), once R12 and R16 show what to combine. The gate needs both seeds, or their mean, above 0.30. |
| R16 | Two-stage: find buildings, then grade each | Done | Kaggle account 1, T4, 29 min | Runs regardless of R12 (decided 19 Oct). Built 19 Oct: [r16-two-stage-grader.ipynb](Champion_Task/r16-two-stage-grader.ipynb). Stage 1: SAM 3 on the before-image (E3's `pre_grid2`) plus R11's building pieces that SAM 3 missed. Stage 2: a Siamese ResNet-18 grades each building from 96 px before and after crops. It scores R16 alone, R16 with R11's pieces, and an R16/R11 vote against R11 (re-run inside and checked against `D4_R11.csv`), plus the grader on the true buildings (macro F1 against BDAChat's 0.763). Clean for the benchmark. Logic checked on a synthetic image on your PC. **Result (19 Oct), not kept:**
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
- The over-grading is mostly gone. Among the stage-1 buildings that are mostly one true building:
  - intact buildings graded destroyed fall from 3,539 to 1,500;
  - macro F1 rises from 0.597 to 0.692.
- On the 85 non-xBD images, SAM 3 + R11's pieces reach Destroyed IoU 0.187, against R11's 0.071.

Results in [R16b-results/](Champion_Task/R16b-results/README.md). | Counting with the corrected grades (H1 V3) needs SAM 3 and the grader run on the benchmark images. |

### System track

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| H2 | Per-building table and SQL counting agent | Not started | CPU or light GPU | — | — |
| H3 | VLM input upgrades (before + after, numbered marks, crops) | Not started | Kaggle GPU | — | After H1 V2. |
| H4 | GEOBench-VLM external counting test | Not started | Kaggle GPU | — | — |
| T2 | DisasTeller on local models | Not started | Kaggle T4 ×2 | — | The biggest missing O3 piece. |
| T3 | Offline demo app | Not started | — | — | After H1, T1 and T2. |
| L0 | A better 7B teacher before distilling | Ready to run (step 1) | Kaggle account 1, T4 ×2: step 1 about 2–3 h; step 2 about 10–20 h, only if needed | Decided 19 Oct: improve the 7B before distilling. Step 1 built 19 Oct: [l0-step1-image-size-check.ipynb](Champion_Task/l0-step1-image-size-check.ipynb).
- It asks the 1,211 optical questions at three sizes (for a 1024 px image):
  - 504 px, H1 V2's size, as the control;
  - 616 px, our training cap;
  - 1036 px, the full size the paper used.
- The decision is fixed in the notebook. Within 3 points of the paper on both tasks closes the gap. Half of each gap recovered closes it partly. Otherwise, step 2.
- Our recipe already copies the paper's (Appendix B.3): LoRA rank 64, alpha 16, lr 2e-4, global batch 256, 1 epoch, vision encoder frozen. Our run finished all 217 steps.
- The differences:
  - 4-bit QLoRA in fp16 on one T4;
  - images capped at about 512 tokens in training;
  - tested at 512 px (about 324 tokens), where the paper used full size (about 1,300).
- The answer format isn't the cause: our 7B answers with letters only on 99.9–100% of questions.
- Gaps to the paper's fine-tune: disaster type −12.5, relational −12.9. | Step 1: re-score at full size on disaster type and relational. Step 2, only if needed: a short top-up training of our LoRA on those tasks at a larger image size, with a replay sample of the others. Then L2. |
| L2 step 1 | Teacher answers for the 7B → 3B distillation | Done | Kaggle account 1, T4 ×2 | 4,000 answers generated; 1,085 (27%) match their label and are kept: disaster type 89.4%, land use 50.9%, bearing bodies 42.7%, building counting 15.6%, relational 13.3%, road counting 0%. Road counting labels are road-area percentages such as 5.22%, which the teacher can't hit exactly without the options; those 735 fall back to their labels. Cell 7's push failed: the dataset title already existed, and the code missed Kaggle's wording ("already in use") but still printed "Pushed". So the dataset `disasterm3-teacher-cot` doesn't hold this run's file. | If L0 changes the 7B, re-run it with the new teacher. Step 2 reads the manifest from the run's notebook output, not from that dataset. |
| L2b | Explanations as a second skill ("why?" on request) | Not started | Kaggle T4 ×2, about 3–5 h | Planned 19 Oct (roadmap L2b, Paper_Audit §4.12): explanations written from the correct answer plus true mask facts, checked against those facts, then trained as a second task in step 2. | Build after step 1. |
| L2 step 2 | Train the 3B | Not started | Kaggle GPU | Its notebook still has the old installs that fail on today's Kaggle. | Fix it, as a copy, after L0. How the 3B learns is still to choose (a, b or c; soft labels recommended). |
| L3 | 4-bit 3B with handoff to the 7B | Not started | Kaggle GPU | — | After L2. |
| L4 | EfficientSAM3 against SAM 3 | Not started | Kaggle GPU | — | — |

### Housekeeping

| ID | Task | Status | Where | Result | Next step |
|---|---|---|---|---|---|
| — | Commit this week's work to git | Ready to run | Your PC | `Essential_task_delegation\` and `others\`, plus `.gitignore`, `README.md` and `UPDATE.md` whenever they're modified (your rule, 19 Oct). The `.gitignore` was updated 19 Oct, so these stay local: bulk masks and maps that no notebook reads, L2 step 1's 71 MB training file, personal samples, the next report's draft and Claude's local settings. | Run the commands under "Needs you now". |
| — | Update the roadmap and README with these results | Not started | — | — | Say when. |

## Change log

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
