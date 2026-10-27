# Next Steps Roadmap: delegation-ready tasks

Updated 2026-10-18, after the fine-tuning ablation closed. This roadmap turns [Phase2_Plan.md](Phase2_Plan.md), [Research_Directions_Beyond_Finetuning.md](Research_Directions_Beyond_Finetuning.md) and [Paper_Audit_and_Integration_Plan.md](Paper_Audit_and_Integration_Plan.md) into **tasks**. Each task:
- has an ID in the project's existing naming style (§2);
- says where the idea comes from and which goal it serves;
- says exactly what to build.

You can hand a task to a teammate, or paste it with the brief in §8 into an AI coding agent to get a Kaggle notebook.

What each finished task found, and the decisions it led to, are in [Task_Ledger.md](Task_Ledger.md); live status is in [Progress_Tracker.md](Progress_Tracker.md).

## 1. Where we are (18 October 2026)

> **This section is the 18 Oct snapshot.** The current status is in [Progress_Tracker.md](Progress_Tracker.md) (Goals and Tasks). Since then:
> - R11 became the champion U-Net, and R16c passed the O1 gate on all 645 validation images (20–21 Oct).
> - E5 (26 Oct) found 40 leaked validation pairs.
>   - Without them, R16c steps 2 and 3 score 0.2838–0.2980.
>   - **Only R16c step 5's SAM 3-only map passes O1's gate on clean data (0.3161 and 0.3051), narrowly.** A third grader seed is proposed before claiming it.
>   - Benchmark copies cost at most 0.7 points; O2's counting is 32.47% on clean questions.
>   - L0's training is complete; its merge needs a fix (proposed).

- **Fine-tuning is closed.** All 11 D4 training runs are done; the full table is in [Ablation_Study_Reference_and_QA_Prep.md](Ablation_Study_Reference_and_QA_Prep.md) §7.
- **Champion:**
  - **R9 recipe (sampler + copy-paste).** It has the best Destroyed IoU (two-seed mean 0.2884, or 0.2971 with TTA) and the best Destroyed recall (about 0.50, or 0.52 with TTA), with the same mIoU as R1.
  - R1 stays the reference baseline; under the mIoU-first R7 rule, R7 = R1.
  - Use **R9's seed-42 model** (`Champion_Task/R9-results/d4_R9/final_model.pth`) for everything downstream.
- **Phase Gate:**
  - mIoU > 0.52 is met in every run.
  - Destroyed IoU > 0.30 is **not** met; the best two-seed mean is 0.2971 with TTA, 0.003 short.
- **The system goals** (O2–O4: hybrid counting, offline DisasTeller with heatmap, 3B edge model) have **not started producing numbers yet.** That's where most of the remaining work is.
- **Submitted files stay frozen.** The review files in `others/` stay as submitted; all of this goes into the next update.

**The goals every task serves** (from the CSE499B review report; details in Paper_Audit §1):

| Goal | Key result |
|---|---|
| O1 Spatial stage | Champion U-Net: mIoU > 0.52 and Destroyed IoU > 0.30 on full-resolution validation images |
| O2 Ground the VLM | Hybrid counting (U-Net + higher-resolution VLM) beats the base model's 34.2% building damage counting (BDC) accuracy, all six tasks re-scored |
| O3 Offline system | DisasTeller on our VLM, no Gemini, returning a report, damage map, counts and survivability heatmap |
| O4 Edge model | Distilled Qwen2.5-VL-3B in about 6 GB, per-task accuracy against the 7B |

## 2. Tasks at a glance

**Task IDs** follow the names the project already uses: the sprint's Tasks A1–D5, then the D4 runs R1–R10 and E0.

| Prefix | Meaning | Continues |
|---|---|---|
| **E** | Evaluation-only notebook, no training | E0 (re-scoring old checkpoints) |
| **R** | Segmentation training run | R1–R5, R1s7, R8–R12, R9s7 and R11s7; a second seed adds `s7` to the run's name (R11s7), and a new number is only for a new recipe. R1s7 (R1 with seed 7) was called R6 until 20 Oct. A run's parts are "steps" (R16c step 1–5), never "stages". |
| **D5** | The xBD data task | Task D5 in `team_task_assignments.md` (external data, Phase 2) |
| **H** | Hybrid U-Net + VLM work | New; serves O2 |
| **T** | Triage system: heatmap, reports, demo | New; serves O3 |
| **L** | Lightweight (edge) models | New; serves O4 |

**Old names** (renamed on 18 October 2026 in every document in `Essential_task_delegation/`; listed here so older notes and commits still make sense). The old P2-0 step 4, the converter check, is now part of D5.
- E1 = P2-0;
- R11 = P2-B;
- R12 = P2-A;
- R15 = P2-AB;
- R16 = P2-C, and SAM 3 step S3;
- E3 = SAM 3 steps S0–S1;
- L4 = SAM 3 step S5.

**Source tags:**
- **[Paper]** an idea from a paper in the audit;
- **[Plan]** our own earlier plan (Phase2_Plan, the CSE499A future-work list, the review report);
- **[Innovation]** an idea a teammate proposed.

| Task | Title | Source | Goal | Type | GPU-h | Needs | Suggested owner |
|---|---|---|---|---|---|---|---|
| **Phase A: this week, no new training** | | | | | | | |
| E1 | Error diagnosis + export champion predictions | Plan | O1 | Eval notebook | ~1 | — | Abrar |
| E2 | Object-based majority voting | Paper: BRIGHT | O1 | Eval (add-on to E1) | ~0.3 | E1 | Abrar |
| D5 | xBD converter check + 1,007 leak-safe tiles | Plan + Paper: xBD | O1 | CPU notebook | 0 | — | Aryan |
| E3 | SAM 3 smoke test and building finding | Paper: SAM 3, SegEarth-OV3 | O1, O4 | Eval notebook | ~2 | HF access | Abrar |
| E5 | Train/validation leak and the benchmark's copies (opened 26 Oct) | Innovation: Ridita's audit | O1, O2 | CPU hash check + eval notebook | ~1 | — | Abrar |
| H1 | KR2 harness: U-Net counting + counts-as-text | Plan + Paper: TEOChat, DamageScope | O2 | Eval notebook (2×T4) | ~1 + ~6 | E1 | Noman |
| T1 | Survivability heatmap | Plan + Paper: BDAChat | O3 | CPU notebook | 0 | E1 | Aryan |
| L1 | Edge benchmark of the U-Net (ONNX, INT8) | Plan + Paper: AWQ; Tamanna Plan 1 | O4 | Laptop CPU | 0 | — | Tamanna |
| **Phase B: next week, training (one change each)** | | | | | | | |
| R11 | Before + after input (6 channels) | Plan + Paper: xView2 winners, BRIGHT, TEOChat | O1 | Train notebook | ~8 | E1 (order) | Abrar |
| R12 | +1,619 extra xBD tiles (D5 and D5b; 2 turned out to be twins of validation pairs, 27 Oct) | Plan + Paper: xBD | O1 | Train notebook | ~7 | D5 | Ridita |
| R13 | BRIGHT pre-disaster footprints (optical only) | Innovation: Ridita + Paper: BRIGHT | O1 | Train notebook | ~7 | R12 result | Ridita |
| R14 | Rubble-centric crop mining (optional) | Innovation: Ridita | O1 | Train notebook | ~6 | spare quota | Ridita |
| **Phase C: champion and fallback** | | | | | | | |
| R15 | Combined champion + second seed (R15s7) | Plan | O1 | 2 train runs | ~16 | R11, R12 | Abrar |
| R16 | Find-then-grade: find buildings, then grade each (runs regardless of R12) | Paper: xView2 1st, BRIGHT, BDAChat + Plan (Phase2_Plan's fallback) + Innovation: Ridita | O1, O2, O3 | Train + eval | ~1 | E3 | Abrar |
| **System track: in parallel** | | | | | | | |
| H2 | Per-building table + SQL counting agent | Paper: DamageScope, Change-Agent | O2, O3 | CPU / light GPU | ~1 | E1 | Noman |
| H2b | Graded test of the SQL agent: which model, if any, writes the SQL | Paper: test suites (Zhong et al.), TrustSQL / EHRSQL 2024, Dr.Spider, DAIL-SQL | O2, O3, O4 | Eval notebook (2×T4) | ~1–1.5 | H2 | Abrar |
| H2c | Free-form questions, second try: question–SQL pairs, fine-tuning, an agreement gate, a new test (fallback: the system as it was) | Paper: EHRSQL 2024, TrustSQL | O3, O4 | Data (CPU) + adapter training + eval notebook | ~5–10 | L2 (or the 7B), H2b | Abrar |
| H3 | VLM input upgrades (before + after, numbered marks, crops) | Paper: TEOChat, GeoPixel, ImageRAG; Tamanna Plan 2 | O2 | Variants in H1 | ~6 each | H1 | Tamanna |
| H4 | GEOBench-VLM external counting test | Paper: GEOBench-VLM | O2 | Eval notebook | ~1 | H1 | Aryan |
| T2 | DisasTeller on local models | Plan | O3 | GPU notebook | ~2 | — | Ridita |
| T3 | Unified offline demo and public showcase (Vercel website) | Plan + Tamanna Plan 3 | O3 | Website (Vercel) + local API | ~1 | H2, T1, T2 | Abrar (decided 24 Oct) |
| L0 | A better 7B teacher before distilling (decided 19 Oct) | Paper: DisasterM3 (Table 2, Appendix B.3) | O4, O2 | Eval notebook, then a top-up run only if needed | ~2–3, then ~10–20 | H1 V2 | Abrar |
| L2 | 7B → 3B distillation (after L0) | Plan + Paper: QLoRA | O4 | 2 notebooks | heavy (~15–25) | L0; 3rd account | Noman |
| L2b | Explanations as a second skill ("why?" on request) | Paper: Distilling Step-by-Step, BDAChat, ImageRAG | O4, O3 | Generation notebook + L2 step 2 change | ~3–5 | L2 step 1 | Noman |
| L3 | AWQ 4-bit 3B + confidence handoff to 7B | Paper: AWQ, Grace | O4 | Eval notebook | ~2 | L2 | Noman |
| L4 | EfficientSAM3 vs SAM 3 | Paper: EfficientSAM3 + Plan (SAM 3 edge step) | O4 | Eval notebook | ~1 | E3 | Tamanna |

Owners are suggestions; reassign freely. The order inside each phase is the recommended priority.

## 3. Shared rules for every notebook

These apply to every task, so an agent doesn't have to rediscover them.

- **Kaggle setup:**
  - GPU T4 ×2 (Kaggle's only T4 option; most notebooks use one of the two), internet ON.
  - Attach the DisasterM3 mirror (`/kaggle/input/datasets/abrarmohammedtanzim/disasterm3-mirror/DisasterM3_Instruct`).
  - Add the `HF_TOKEN` secret (write-scoped) only if the notebook pushes to Hugging Face.
  - Teammates' accounts got 403 errors pushing to `AbrarAlam/*` repos. Push to your own namespace, or rely on Kaggle outputs.
- **Base code:** copy the notebook named in the task and change only what the task lists. For segmentation work, keep Cell 1's pins (`segmentation-models-pytorch==0.3.4`, `albumentations==1.4.21`) and reuse these from [d4-champion-model-R5.ipynb](Champion_Task/d4-champion-model-R5.ipynb) (cells 4, 5, 8 and 10):
  - `DisasterM3SegDataset`;
  - `build_model`;
  - `predict` (8-view TTA);
  - `evaluate`, `metrics_from_confusion` and `result_row`, plus `CSV_FIELDS`.
- **Training-run names:** set `EXPERIMENT` to the task ID (for example `EXPERIMENT = "R11"`) and add a preset with the same name. The notebook then writes `D4_R11.csv` and pushes to `disasterm3-d4-R11`, like R1–R10.
- **Data and leakage:**
  - The validation set is always the same 645 images: `train_test_split(pairs, test_size=0.1, random_state=42)` over the 6,443 pairs built in Cell 4.
  - Never train on them, and never add outside data to validation.
  - No SAR (radar) images anywhere in segmentation work.
- **Scoring:**
  - E-protocol: full native 1024 images, last epoch, dataset-level confusion matrix.
  - Report a standard row and an 8-view TTA row.
  - Training runs: 40 epochs, 5,798 samples per epoch.
- **Run-All safe:**
  - A `SMOKE_TEST` flag (about 5 minutes on a few images).
  - The 11.3-hour time guard with resume (as in the D4 notebooks) for anything over about 9 hours.
  - Big temporary files go in `/kaggle/temp`, which isn't saved, so the output doesn't get zipped.
- **Outputs:** everything in `/kaggle/working/<task-id>/` (for example `/kaggle/working/E1/`):
  - a CSV with the same columns as the D4 CSVs (for segmentation);
  - `config.json`;
  - a one-page `README.md` with what ran, the numbers, and anything that broke.
- **Checkpoints to upload once as Kaggle datasets:**
  - R9 champion: `Champion_Task/R9-results/d4_R9/final_model.pth`;
  - R1 reference: `Champion_Task/R1-results/d4_R1/final_model.pth`.

  Both load with `build_model("Unet", "resnet50", pretrained=False)` and `load_state_dict`.
- **Keep rule for training runs (R11 onwards):**
  - mIoU must not drop by more than the noise (about 0.005 against the R9 or R1 base).
  - Destroyed IoU must rise by more than the seed noise (0.013).
  - A gate claim needs two seeds (Phase2_Plan §6).

## 4. Phase A: this week, no new training

### E1 · Error diagnosis and champion predictions
- **Source:** [Plan] Phase2_Plan §3. BRIGHT's finding that finding buildings and grading them are separate problems is why we split the errors this way.
- **Goal:** O1. Learn *why* destroyed pixels are missed, and set Gate G2 (which of R11 and R12 runs first). It also produces the prediction files that E2, H1, T1 and H2 reuse.
- **Base:** copy [d4-champion-model-E0.ipynb](Champion_Task/d4-champion-model-E0.ipynb). It's already evaluation-only, and its `EVAL_CHECKPOINTS` list (Cell 2) takes a `local_path`.
- **Inputs:** the R9 and R1 `final_model.pth` files (Kaggle dataset); the DisasterM3 mirror.
- **Build:**
  1. For each model, standard and TTA: the 4×4 confusion matrix (pixel counts, rows = truth).
  2. **Building IoU:** classes 1–3 merged against background.
  3. **Background-miss share:** of the destroyed pixels the model gets wrong, the fraction predicted as Background, Intact and Damaged.
  4. **Per-event and per-source Destroyed IoU:** accumulate a confusion matrix per event. The event comes from the file name: xBD names look like `hurricane_florence_00000261_post_disaster.png` (event = the text before `_<8 digits>`); other names look like `bata_explosion_post_0.png` (event = the text before `_post`). Source = xBD versus other.
  5. Save R9's predicted masks for all 645 images (standard and TTA) as uint8 PNGs (`preds/R9_std/<name>.png`, `preds/R9_tta/<name>.png`), plus the ground-truth masks (`gt/<name>.png`).
- **Outputs:** `p2_0_summary.csv`, `per_event.csv`, `README.md`, and the `preds/` and `gt/` folders. Publish the output as a Kaggle dataset, `d4-r9-val-predictions`.
- **Done when:**
  - R9's and R1's overall IoUs match their D4 CSVs to 4 decimals (proving the setup is identical);
  - the G2 number (Background-miss share) is reported.
- **Gate G2:**
  - 40% or more of the missed destroyed pixels predicted as Background: run R11 first.
  - Otherwise: run R12 first.

### E2 · Object-based majority voting (no training)
- **Source:** [Paper] BRIGHT (ESSD 2025): +0.69 mIoU per event, −0.41 overall on their data.
- **Goal:** O1. A free post-processing step that might lift Destroyed IoU.
- **Base:** extra cells at the end of the E1 notebook (the predictions are already in memory), or a CPU notebook reading `d4-r9-val-predictions`.
- **Build:**
  1. For each image, take the building mask = prediction > 0.
  2. Find connected components (8-connectivity, `cv2.connectedComponentsWithStats`).
  3. Give each component its majority class among 1, 2 and 3.
  4. Re-score with the same confusion-matrix code.
  5. Variant: ignore components smaller than 20 px.
- **Outputs:** a standard row and a TTA row for "R9 + voting" in the D4 CSV format, plus `README.md`.
- **Done when:** both rows are compared with R9's (0.5373 / 0.2921; TTA 0.5440 / 0.3001). Adopt voting only if Destroyed IoU rises by more than 0.005 without lowering mIoU, **and** it repeats on R9s7's model.

### D5 · xBD converter check and the 1,007 leak-safe tiles
- **Source:** [Plan] Phase2_Plan §2–3 (this is Task D5 from the sprint plan) and [Paper] xBD.
- **Goal:** O1. Prepare R12's extra training data and find out exactly how DisasterM3 mapped xBD's damage levels.
- **Base:** a new **Kaggle CPU notebook**. It needs DisasterM3's masks from the mirror, plus the xBD labels uploaded as a Kaggle dataset. Upload `data/xView2 Challenge Dataset - train and test/train/labels/` (5,598 JSON files) and `train/images/`.
- **Build:**
  1. **Tile IDs:** collect every image name in DisasterM3, train and benchmark (`train_release.json` and `benchmark_release.json`). Turn each into an xBD tile ID such as `palu-tsunami_00000123` (the xBD event name uses dashes, and DisasterM3 uses underscores).
  2. **Split the tiles:** separate xBD's 2,799 post-disaster labels into "already in DisasterM3" (expected: 1,792) and "new" (expected: 1,007).
  3. **Converter:**
     - Parse `features.xy[*].wkt` (pixel-coordinate polygons) and fill them with `cv2.fillPoly` by `properties.subtype`: `no-damage`, `minor-damage`, `major-damage`, `destroyed`, `un-classified`.
     - Rasterise the 1,792 overlapping tiles and compare them pixel by pixel with DisasterM3's combined 4-class masks (built the same way as Cell 4 of the D4 notebook).
     - Report a 5×4 table: xBD subtype against DisasterM3 class.
  4. **Convert the 1,007 new tiles** with the measured mapping. `un-classified` becomes 255 (ignore), unless step 3 shows something else.
- **Outputs:**
  - the `xbd-extra-1007` Kaggle dataset (`images/`, `masks/`, `manifest.json` with the event and the per-class pixel counts);
  - `mapping_report.md`.
- **Done when:**
  - the mapping table is clear (expected: minor and major become Damaged);
  - no new tile ID appears anywhere in DisasterM3;
  - the counts match Phase2_Plan §2 (1,007 tiles, about 3,472 destroyed polygons).

### E3 · SAM 3 smoke test and building finding
- **Source:** [Paper] SAM 3 (arXiv 2511.16719) and SegEarth-OV3 (64.3% zero-shot building IoU on xBD).
- **Goal:**
  - O1: decide whether SAM 3 can do the finding for the find-then-grade model, R16 (Gate G3).
  - O4: the field version of SAM 3 is L4.
- **Base:** a new notebook on a T4, in fp16. Request access to the gated `facebook/sam3` weights on Hugging Face first, then use the official `facebookresearch/sam3` code.
- **Build:**
  1. **Part 1, smoke test:** load the model, run one 1024×1024 image, and log peak GPU memory and seconds per image. If it doesn't fit, record that and try 2×T4 or EfficientSAM3 (L4).
  2. **Part 2, building finding:** prompt "building" on the 645 validation **pre-disaster** images and on the **post-disaster** images.
     - Building IoU: the union of SAM 3's masks against the ground-truth building mask (classes 1–3).
     - Building count against the number of ground-truth building blobs.
  3. Compare with R9's building IoU from E1.
- **Outputs:** `sam3_s1.csv` (per image: IoU and counts, pre and post) and `README.md` (memory, speed, and the overall IoU for pre and post).
- **Gate G3:** if SAM 3 on the pre image is at least as good as R9 at finding buildings, R16 uses SAM 3 for the finding; otherwise it uses R9's building mask.

### E5 · Train/validation leak and the benchmark's copies (opened 26 Oct)
- **Source:** [Innovation] Ridita's corrected audit (`BRIGHT_DisasterM3_Integration/`, §8).
- **Goal:** O1 and O2. Scores must come from images the models never saw.
- **Steps:**
  1. **Step 1, CPU only** (`e5-step1-hash-check.ipynb`):
     - fingerprint every DisasterM3 Instruct image and every benchmark image (file MD5, pixel MD5, dHash and a thumbnail);
     - list exactly which validation pairs leak;
     - find benchmark images that copy Instruct images, and the questions that use them;
     - draw pictures of near-copies for review.
  2. **Step 2, GPU, about 1 h** (`e5-step2-per-image-counts.ipynb`): per-image pixel counts for R11, R11s7, R9 and R1, standard and TTA, checked against their D4 CSVs.
  3. **On your PC, every score a leak touched, whatever the share** (decided 26 Oct; it's a filter over saved files, minutes of CPU). Each gets two numbers: all questions or pairs, and clean.
     - **Validation leak (segmentation):** R16c (steps 2, 3 and 5), R16, R11, R11s7, R9 and R1, plus R9s7 and E2's vote from E1's counts.
     - **Benchmark copies (7B and U-Net counting):** L0 step 1, L0 step 2 (old 7B and the new one), H1 V1, V2 and V2b, H2, R16c step 1's counting, and Aryan's numbers if we quote them.
     - **Not re-scored:** R2–R8 and R10. No checkpoints or per-image counts were kept. They were only compared with each other on the same 645 pairs, and without the 40 leaked pairs every U-Net drops by about the same amount (0.012–0.013 Destroyed IoU). The gaps between U-Net runs move by about 0.001, under the 0.005 noise margin, so their keep/drop decisions stand.
     - Then O1's and O2's status everywhere.
  4. **From now on:** any new split is grouped by before-image fingerprint, so twins stay on one side.
- **Expectation, fixed 26 Oct:**
  - 21–40 leaked pairs, with the PC check's 21 among them, and about 264 shared before-images in the pool.
  - Benchmark: no prediction. (The first plan re-scored O2 only above 1% of optical questions copied; dropped 26 Oct, every affected score is re-scored.)
- **Built 26 Oct and dry-run checked on your PC.** The real manifest and masks reproduce the split, and planted copies are found exactly. The outputs will go to `Champion_Task/E5-results/step1/` and `step2/`.
- **Results so far:** [E5-results/E5/](Champion_Task/E5-results/E5/README.md) and [Task_Ledger.md, E5](Task_Ledger.md).
  - **26 Oct:** Ridita's audit README settles the count at 40 (our 21 plus 19 whose socal_fire side isn't in xBD tier 1): [leaked_pairs_40.csv](Champion_Task/E5-results/E5/leaked_pairs_40.csv).
  - Without them, R16c scores 0.2838–0.2980 on its four combinations. **O1 isn't met on clean data.**
  - Still to come: R11 and R11s7 alone (step 2), R16c step 5, and the benchmark (step 1).
  - **Steps 1 and 2 done (26 Oct, account 2), and every affected score re-scored on your PC.** Every check passed: the split, all 8 D4 reproductions, and the same 40 pairs a third time.
    - **O1, clean:** R16c step 5's SAM 3-only map is the only map that passes: 0.3161 and 0.3051 (mIoU 0.547).
    - R11 and R11s7 lose 0.027 to the leak, twice what the after-only U-Nets lose, because the leaked pairs share their before-image. With TTA, R11 is 0.2692 against R9's 0.2874.
    - **Benchmark:** 186 of 10,675 optical questions (1.74%) use a copied after-image. Without them, scores drop by at most 0.7 points; O2's counting goes from 32.96% to 32.47%.
    - Tables and caveats: [E5-results/E5/](Champion_Task/E5-results/E5/README.md).
    - **Next:** group future splits by before-image fingerprint. A third step-5 grader seed (proposed) before claiming O1.
  - **27 Oct:** R12's 1,619 extra xBD tiles, checked by content for the first time: two share a before-image with validation pairs outside the 40, so R12 trained on 42 leaked pairs. Its verdict (not kept) stands. E2's and E3's decisions also hold on clean data.

### H1 · KR2 evaluation harness: U-Net counting and counts-as-text
- **Source:**
  - [Plan] the CSE499A future-work "textual bridge" and the review report's "VLM reads per-building counts as text";
  - [Paper] TEOChat (VLMs find buildings badly but grade given ones well) and DamageScope (answer counts from data, not from pixels).
- **Goal:** O2, KR2 — the first real hybrid counting number, plus all six tasks re-scored.
- **Base:**
  - `support/Abrar/Implementation_DisasterM3/3b_hybrid_vllm_evaluation.ipynb` (vLLM on 2×T4, our fine-tuned `AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP`);
  - `hybrid_bdc_counter.py`;
  - `4_score_results.py` for scoring.

  Replace the old ResNet-34 U-Net (`AbrarAlam/disasterm3-unet-checkpoints-2`) with **R9**: ResNet-50, four classes, full 1024 input.
- **Data:** the DisasterM3 benchmark (`benchmark_release.json`, 30,042 entries). The six scored tasks have 12,968 questions; building damage counting (BDC) has 4,982:
  - about 2,671 ask for intact buildings;
  - about 1,059 for destroyed;
  - about 786 for damaged;
  - about 466 for a total.

  Each is multiple choice with the exact count as ground truth (e.g. 232). 623 BDC questions use a **radar** after-image (`*_sar_post_*`). Score those with the U-Net on the matching optical image (same name without `_sar`), and report them separately.
- **Variants** (a flag at the top):
  - **V1, counting bypass (no VLM):**
    1. Segment the after-image with R9 + TTA.
    2. Read the class from the question: intact/undamaged/remained → 1; destroyed/collapsed → 3; damaged → 2; total/all → 1–3.
    3. Count connected components of that class (minimum area as in `hybrid_bdc_counter.py`; tune it on 200 *training* images, never on the benchmark).
    4. Pick the option nearest to the count.
  - **V2, counts-as-text for all six tasks:**
    - Put a short fact block before every question, e.g. "U-Net analysis: 51 intact, 3 damaged, 0 destroyed buildings; most damage in the north-east quadrant."
    - BDC still uses V1. Keep the full image as the visual input. This is the "textual bridge".
- **Outputs:**
  - `kr2_results.csv`: accuracy per task per variant, next to the 499A references below.
  - BDC accuracy broken down by question type (intact, damaged, destroyed, total) and by optical versus radar.
  - `README.md`.
- **499A references** (our fine-tuned 7B / the base model):

  | Task | Fine-tuned 7B | Base model |
  |---|---|---|
  | DTR | 68.33% | 66.6% |
  | BBR | 16.39% | 4.7% |
  | **BDC** | **23.58%** | **34.2%** (the KR2 target) |
  | DRE | 26.24% | 29.3% |
  | DSR | 31.31% | 28.3% |
  | ORR | 22.50% | 23.9% |

  The old crop collage scored 13.11% on BDC because intact buildings weren't counted.
- **Like-for-like comparison (checked 19 Oct against the DisasterM3 paper):**
  - **Where the base numbers come from:** the paper's Table 2 (page 7, "Bench set with optical-optical setting"), not our own CSE499A runs. That table covers only questions whose after-image is a photo.
  - **How they were decoded:** the paper runs every model in vLLM with its default settings (Appendix B.1). Its script, `support/Abrar/Implementation_DisasterM3/temp_junjue/pyscripts/run_vllm.py` (lines 240–252), takes the model's default generation config, with temperature at least 0.01. Qwen2.5-VL-7B's defaults (temperature 0.000001, repetition penalty 1.05) make that effectively greedy, like H1 V2 and V2b.
  - **Optical questions only:**

    | Task | Our 7B (greedy, H1 V2/V2b) | Base model (paper) | Paper's fine-tuned 7B |
    |---|---|---|---|
    | DSR | 37.9 | 28.3 | 37.7 |
    | DTR | 71.1 | 66.6 | 83.6 |
    | BBR | 19.5 | 4.7 | 21.5 |
    | BDC | 32.0 | 34.2 | 34.3 |
    | DRE | 32.9 | 29.3 | 29.4 |
    | ORR | 23.3 | 23.9 | 36.2 |
    | Mean | 36.1 | 31.2 | 40.4 |

  - Our 7B beats the base model by 4.9 points but trails the paper's own fine-tune (same model, same training set) by 4.3, mainly on DTR and ORR. Counting is 2.2 points short of the base, not 3.7: the 30.5% included radar questions. L0 follows this up.
  - One difference remains: H1 V2 and V2b feed 512 px images, where the paper's default is the full size.
- **Why U-Net counting stays low (H1 V1, 22.3%):**
  - It used R9. Optical after-images scored 26.5%, radar 11.3% (below the 20% of guessing).
  - Counting from the true masks scores 100%, so the masks are the problem. An answer counts separate buildings (8-connected blobs), and the wrong options are only ±20% or ±40% away (paper, page 6), so a count must land within about 10%.
  - A U-Net merges touching houses and splits damaged ones: on optical after-images the median count error is 45% for intact, 75–83% for damaged and destroyed.
  - A better pixel model helps little; counting separate buildings (R16) is the route.
- **Done when:** V1's BDC accuracy is reported with its breakdown, and V2's six-task table is complete.
- **Gate G4:**
  - V1 above 34.2% means the counting half of KR2 is met.
  - Keep V2 only if it doesn't lower any of the other five tasks by more than 1 point.

### T1 · Survivability heatmap
- **Source:**
  - [Plan] CSE499A future work (`cv2.distanceTransform` or kernel density estimation);
  - [Paper] BDAChat's damage-aware heatmap (MIT code in its repo).
- **Goal:** O3, one of KR3's four required outputs.
- **Base:** a new CPU notebook reading `d4-r9-val-predictions` (E1).
- **Build:**
  1. **Danger map:**
     - damage mask = destroyed (weight 1.0) plus damaged (weight 0.5);
     - KDE = `cv2.GaussianBlur` of the weighted mask (sigma about 25 px, tuned visually), normalised to 0–1.
  2. **Safety map:** `cv2.distanceTransform` on the inverse of the damaged-or-destroyed mask, normalised, so distance from the nearest collapse means safer.
  3. **Survivability-search priority:** KDE × (1 − normalised distance). Areas with dense collapses rank first for search.
  4. Overlay each map on the after-image (colour map plus legend).
  5. For each image, record the top-3 hot spots (centroid, area) and the 3 safest open areas.
- **Outputs:** `heatmaps/<name>_{kde,safe,priority}.png`, `hotspots.json` (feeds DisasTeller in T2 and the demo in T3), and `README.md` showing 6 example images (3 xBD, 3 other).
- **Done when:** 20 images have been checked by eye. Hot spots must sit on the ground-truth destroyed areas: report the share of hot spots that overlap ground-truth destroyed pixels.

### L1 · Edge benchmark of the U-Net
- **Source:**
  - [Plan] O4 and Research_Directions lever 5;
  - [Paper] AWQ (low-bit deployment);
  - Tamanna's [Plan 1](Tamanna_task_2/Plan_1/Plan_1_Edge_Model_Compression.md), which already lays out the steps.
- **Goal:** O4 — a measured speed, memory and accuracy table for the segmenter.
- **Base:** Tamanna's Plan 1, run on a laptop CPU (and optionally a Kaggle CPU notebook as a second machine).
- **Build:**
  1. Export R9 to ONNX: a fixed 1024×1024 input, opset 17.
  2. Quantise with ONNX Runtime: dynamic INT8, plus static INT8 calibrated on 100 *training* images.
  3. On the 645 validation images, compare PyTorch FP32, ONNX FP32 and the two INT8 versions on:
     - file size;
     - peak RAM;
     - milliseconds per image (median of 3 runs, with a warm-up);
     - mIoU and Destroyed IoU.
- **Outputs:** `edge_benchmark.csv`, a speed-against-accuracy plot, and `README.md` with the CPU model.
- **Done when:** all four rows are measured on the same machine. INT8 is usable if Destroyed IoU drops by less than 0.01.

## 5. Phase B: next week, structural training runs

Each run changes **one thing** from its base and follows the keep rule in §3.

### R11 · Before + after input (6 channels), the main bet
- **Source:**
  - [Plan] Phase2_Plan §4 and research lever 1;
  - [Paper] the xView2 winners (all of the top 5 used both images), BRIGHT Table 7 (the before image raises Destroyed IoU from 59.5 to 64.9), and TEOChat.
- **Goal:** O1. Let the model see where a building stood before it became rubble.
- **Base:** copy [d4-champion-model-R5.ipynb](Champion_Task/d4-champion-model-R5.ipynb) (sampler, no copy-paste). Copy-paste would put rubble into the after-image where the before-image shows nothing, so it's left out.
- **Changes:**
  1. Cell 4: keep `pre_image_path` in each pair. It's resolved like `post_image_path`.
  2. Cell 5: load both images, apply the **same** crop, flip, rotation and brightness change to both (Albumentations `additional_targets={"pre": "image"}`), normalise each, and stack them into 6 channels.
  3. Cell 8: change `in_channels=3` to `in_channels=6` in `build_model`. smp adapts the pretrained first layer.
  4. TTA needs no change, because the flips and rotations act on every channel.
  5. Add a preset `"R11": dict(use_sampler=True, in_channels=6)` and set `EXPERIMENT = "R11"`.
- **Check first:** overlay 10 before/after pairs per source (xBD and other) and confirm they line up. If shifts are larger than about 5 px, note them in the README.
  - Measured in Ridita's audit (pushed 25 Oct): 52% of 2,966 measurable pairs are shifted by more than 2 px, 10% by more than 5 px, the worst by 56 px. Aligning the pairs first is a candidate R11 variant ([Task_Ledger §9](Task_Ledger.md)).
- **Outputs:** the usual D4 outputs, as `D4_R11.csv` and so on.
- **Done when:** compared with R5 (0.5354 / 0.2898; TTA 0.5391 / 0.2917) and R9, using the keep rule.

### R12 · Add the 1,007 leak-safe xBD tiles
- **Source:** [Plan] Phase2_Plan §4 (the training half of Task D5) and [Paper] xBD.
- **Goal:** O1. More destroyed examples: 8,021 destroyed polygons, 3,472 from D5's 1,007 tiles and 4,549 from D5b's 612 tier3 tiles (added 19 Oct, option A).
- **Base:** R11's 6-channel recipe (chosen 19 Oct). Built as `d4-champion-model-R12.ipynb`, a copy of `d4-champion-model-R11.ipynb`.
  - **Masks:** from D5's and D5b's outputs (`D5-results/`, `D5b-results/`, uploaded as one Kaggle dataset). Their lookup tables are the same, `[0, 1, 2, 2, 3, 255]`.
  - **Images:** from Bojan Tunguz's two xView2 copies on Kaggle (`xview2-challenge-dataset-train-and-test` and `xview2-challenge-dataset-tier-3-data`, attached on import), so only the masks need uploading.
- **Changes:**
  1. Append the 1,619 tiles (D5's 1,007 and D5b's 612) to `train_pairs` only. Cell 4b stops if either pool is missing.
  2. Make the loss ignore pixels marked 255: `ignore_index=255` in both smp losses (Focal Tversky and Lovász). Cell 7 checks that ignored pixels don't change the loss.
  3. Keep 5,798 samples per epoch, so the change is "more data", not "more training". With the sampler on, compute the image weights over the combined list.
  4. Set `EXPERIMENT = "R12"`.
- **Outputs:** `D4_R12.csv` and so on.
- **Benchmark rule:** 1,059 of the tiles are DisasterM3 benchmark images (447 from D5, all 612 from D5b), so R12 never answers benchmark questions; R16 (with R11) does.
- **Done when:** compared with R11 (0.5811 / 0.2911; TTA 0.5875 / 0.2965) and R9 under the keep rule. Per-event Destroyed IoU from E1's code shows whether the gain comes from the events the new tiles cover.
- **Result (19 Oct, second run, after the one-line fix to Cell 4b), not kept:**
  - Trained on 7,417 images, including the 1,619 extra tiles.
  - mIoU 0.5820 (TTA 0.5883) and Destroyed IoU 0.2898 (TTA 0.2945). That is within noise of R11.
  - 8,021 more destroyed buildings didn't move the destroyed class, so more examples of it isn't the lever. That fits E1: most missed destroyed pixels are predicted as background.
  - R11 stays the champion recipe. R12 must never answer benchmark questions.
- **E4 (19 Oct), R12 against R11 by source and event:**
  - On xBD images R12 gains only +0.007 Destroyed IoU (within noise), but per event the gain follows the extra data (rank correlation +0.73; Palu +0.047).
  - **The real cause of O1's gap:** Turkey's earthquake holds 37% of the validation's destroyed pixels and R11 scores 0.000 there, calling 91.5% of them intact. Training holds 3.75 million Turkey destroyed pixels, so more data isn't the fix.
  - Without Turkey, R11's Destroyed IoU is 0.410.
  - R16's grader finds 72–83% of Turkey's destroyed pixels; see R16c.

### R13 · BRIGHT pre-disaster footprints (optical only)
- **Source:** [Innovation] Ridita's BRIGHT plan and roadmap (`BRIGHT_DisasterM3_Integration/`), and [Paper] BRIGHT.
- **Goal:** O1. Better building finding, which helps if E1 shows that many destroyed pixels are missed as Background.
- **Base:** copy `d4-champion-model-R5.ipynb`.
- **First fix the script issues** (Paper_Audit §2.2):
  1. Never load BRIGHT's post-disaster files; they're radar.
  2. Pair each pre-disaster image with a **footprint** mask, where any building is class 1 (Intact, which is true before a disaster). Never pair it with damage classes.
  3. **No BRIGHT event is missing from DisasterM3** (checked 19 Oct against its train and benchmark lists, which record each picture's sensor). Skip the 7 events that have optical after-images there: Turkey, Morocco, Libya, Beirut, Bata and Hawaii (in our training data) and Noto (in the benchmark). The other 7 appear in DisasterM3 only as radar questions, which our optical U-Net can't answer, so their before-images are safe to use. List the events used in the README. Ridita's converted set (5 events) includes Noto, so drop its tiles.
  4. Don't use AIDER; it has no masks.
- **Open question for the 6-channel recipe:** a BRIGHT tile has no optical after-image. Either put the before-image in both slots (a no-change pair, where every building is intact) or train R13 on the 3-channel recipe. Decide when R13 is built.
- **Change:** append the BRIGHT tiles to training only, and keep 5,798 samples per epoch.
- **Run when:** after R12. It's lower priority, because it adds only intact buildings.
- **Ridita's audit (pushed 25 Oct):** the five usable events don't leak, but 1,056 of their 1,285 tiles are volcano scenes, against 8 volcano pairs in validation. Expect little change in Destroyed IoU.

### R14 · Rubble-centric crop mining (optional)
- **Source:** [Innovation] Ridita's roadmap, Innovation 1.
- **Goal:** O1.
- **Base:** copy `d4-champion-model-R9.ipynb`.
- **Change:** when an image contains destroyed pixels, centre the 768 crop on a random destroyed pixel 60% of the time (with a random offset of up to ±200 px). Otherwise crop at random.
- **Caveat:** this is another data-sampling tweak, and the sampler and copy-paste already moved Destroyed IoU only to the noise level. Run it only with spare quota.

## 6. Phase C: champion and fallback

### R15 · Combined champion and second seed (R15s7)
- **Source:** [Plan] Phase2_Plan §4 and §6.
- **Goal:** O1, the gate claim.
- **Build:**
  1. Combine everything from R11 to R14 that passed the keep rule (plus E2 voting if it was adopted).
  2. Train with seed 42 (R15), then repeat with seed 7 (R15s7).
  3. Claim the gate only if both runs, or their mean, are above 0.30. Say whether that's with TTA.
- **Cost:** about 16 GPU-hours on two accounts in parallel.
- **Status (20 Oct):** nothing beyond R11 has passed the keep rule (R12 added nothing; R13 and R14 haven't run), so R15 would be R11 itself and isn't built.
  - R11's second seed runs as **R11s7**, by the naming rule: a seed repeat keeps the run's name plus `s7`, and a new number is only for a new recipe.
  - The O1 candidate is R16c's find-then-grade map on R11 (step 2) and on R11s7 (step 3).
  - R15 comes back only if a later change passes.

### R16 · Find-then-grade: find buildings, then grade each one
R16's two parts are **the finding** (which buildings are there) and **the grading** (each one's damage grade). Named so on 20 Oct, replacing "stage 1/2", so that "step" is used only for a task's steps.
- **Source:**
  - [Paper] xView2 1st place, BRIGHT (find-then-grade models beat one-step ones: 67.6 against 64.9 mIoU), and BDAChat (pairs before and after crops per building);
  - [Plan] the find-then-grade fallback in Phase2_Plan and the SAM 3 test plan (Research_Directions §7.4);
  - [Innovation] Ridita's "Footprint-guided damage cascade".
- **Goal:** O1, O2 and O3. **Decided 19 Oct: R16 runs regardless of R12.** If R12 passes the gate, R16 becomes an add-on rather than the fallback, for four reasons:
  - **one grade per building:** counting questions (O2) become simple counts, where counting blobs on the U-Net's pixel map scored only 22% (H1 V1), and DisasTeller's reports (O3) need a per-building list;
  - **clean for the benchmark:** it trains only on DisasterM3's training data, so unlike an R12 trained on exam pictures it can answer benchmark questions;
  - **a different fix for "missed as background":** it finds buildings in the before-image, so rubble or lava in the after-image can't hide them;
  - **cheap:** about 3 GPU hours.

  It is still the fallback if R15 misses the gate.
- **Build:**
  1. **The finding:** G3 passed, so buildings come from SAM 3 on the pre-image (`pre_grid2`), plus the U-Net's building pieces that SAM 3 missed. The U-Net is R11, run inside the notebook. E3 measured that SAM 3 together with R9 covers 86% of destroyed buildings' footprints.
  2. **Crops:** for each building, a square crop of the before and after images with a 16 px margin, resized to 96 px (128 px would need about 14 GB of crops), plus the building's mask as a 4th channel. Training buildings are the 8-connected pieces of each class in our masks, so each has one grade: every damaged and destroyed piece, and at most 12 intact pieces per image. BDAChat's `sgts.py` is a reference for the pairing.
  3. **Grader:** a Siamese ResNet-18 (shared weights) that combines [before, after, after − before] and outputs 3 classes. Train with balanced grades for 10 epochs and keep the last epoch; the whole notebook takes about 1 GPU hour.
  4. **Scoring:** paint each building's mask with its predicted class and score with the E-protocol. Also compute object-level macro F1 to compare with BDAChat's 0.763.
  5. **Combined test:** label each building by a vote between the grader and the best U-Net's pixels inside it. Measure it rather than assume it helps: E2's per-building vote on R9 alone lowered Destroyed IoU from 0.292 to 0.228, though R16's building outlines come from a better source. For the O1 score, combine with the best U-Net (R12 if it passes). For benchmark questions (O2, O3), combine only with a U-Net that never trained on benchmark pictures (R11), so the combination stays clean.
- **Done when:** R16 alone and R16 combined with the best U-Net are both scored against R9 and R11.
- **Built 19 Oct:** `Champion_Task/r16-find-then-grade.ipynb` (named `r16-two-stage-grader` until 20 Oct, here and on Kaggle). Build item 4's object-level F1 is measured on the true validation buildings, so it isolates the grader.
- **Result (19 Oct), not kept:**
  - The best Destroyed IoU is 0.261 (the vote), against R11's 0.297 with TTA; SAM 3 + grader alone reaches 0.167.
  - The grader's macro F1 on the true buildings is 0.656 (BDAChat 0.763). Its balanced training over-grades intact buildings: destroyed precision 0.35, recall 0.89.
  - On the 85 non-xBD images, SAM 3 + grader doubles R11's Destroyed IoU (0.128 against 0.071).
  - **Next:** correct the grader by the real grade shares (CPU, from `instances_val.csv`).
- **R16b, the correction (19 Oct, on your PC, 7.3 min), not kept:** `Champion_Task/r16b-grader-prior-correction.ipynb`.
  - **Rule:** each grade's probability is multiplied by its share among the training buildings (intact 73.7%, damaged 18.4%, destroyed 8.0%) and divided by a third. Nothing is fitted on validation, and R16's scores were first rebuilt exactly from its per-building table.
  - **Pixel scores:** every R16 map improves. SAM 3 + grader goes from Destroyed IoU 0.167 to 0.228, and the vote from 0.261 to 0.274 with mIoU 0.585. R11 with TTA stays ahead (0.297 and 0.588).
  - **Over-grading mostly gone:** among the found buildings that are mostly one true building, intact buildings graded destroyed fall from 3,539 to 1,500, and macro F1 rises from 0.597 to 0.692.
  - **Next:** counting with the corrected grades (H1 V3) needs SAM 3 and the grader run on the benchmark images.
- **R16c, a rigorous test of the find-then-grade idea** (go given 19 Oct, after you asked to test R16 more rigorously). Its steps were renumbered 1–5 on 20 Oct: step 3 was 2b, step 4 was 3a, step 5 was 3b.
  - **Step 1, the ceiling and a counting check** (your PC, 2.6 min): `Champion_Task/r16c-step1-ceiling-and-counting.ipynb`.
    - Give each found building its main true grade, as a perfect grader would. SAM 3 + R11's pieces then reach mIoU 0.766 and Destroyed IoU 0.614 (0.798 and 0.661 if it may also reject non-buildings). R16b's grader gets 0.541 and 0.208.
    - So the outlines are good enough; the grader loses most of the score.
    - Counting, on the 1,055 validation counting questions:
      - R11's blobs: 39.2%;
      - R16's best map: 36.9%;
      - a perfect grader: 55.3%.
  - **Step 2, a stronger grader** (Kaggle, about 2 h): `Champion_Task/r16c-step2-stronger-grader.ipynb`, a copy of R16's notebook.
    - 128 px crops with a 32 px margin, 20 epochs, two seeds.
    - The epoch and a bias per grade are picked on 580 held-out training images, never on validation.
    - It is scored against R11 TTA (keep rule) and against the ceiling (the share of the gap closed).
  - **Steps 4 and 5, the grader on SAM 3's own outlines** (planned only if step 2 fell short; run as a stretch by your decision of 20 Oct):
    - Step 4, `Champion_Task/r16c-step4-sam3-training-images.ipynb`, runs SAM 3 on the 5,798 training images (about 3.5–4 h).
    - Step 5, `Champion_Task/r16c-step5-grader-on-sam3-outlines.ipynb`, is step 2's notebook with `TRAIN_OUTLINES = "sam3"`: the grader trains on SAM 3's own outlines. A 4th label, "None", lets it reject outlines that are mostly background.
    - Step 5 replaces step 2 only if it beats step 2's kept map on both seeds by the keep margin.
  - **Step 3, step 2's graders with R11s7's pieces:** `Champion_Task/r16c-step3-r11s7-pieces.ipynb`, evaluation only, after R11s7. It gives the O1 claim its second U-Net seed.
  - **Done when:** a map is kept on both seeds (then it becomes the O1 candidate, checked on R11s7 in step 3, and H1 V3 counts with it), or step 5 also falls short and the find-then-grade route is closed with these numbers.
  - **Watch Turkey (found with E4, 19 Oct):**
    - On Turkey's 64 validation images, R16's grader finds 72–83% of the destroyed pixels where R11 finds none. Its Destroyed IoU there is 0.13, or 0.19 with R16b's correction, held down by over-grading intact buildings.
    - Step 2's per-building table gets the same Turkey breakdown after the run.
  - **Step 2 result (20 Oct, 1.4 h on account 1): kept on both grader seeds, and the O1 gate passes on both.**
    - SAM 3 + R11's pieces: mIoU 0.587 and 0.585, Destroyed IoU 0.316 and 0.311 (R11 with TTA: 0.588 and 0.297).
    - The grader's macro F1 on the true buildings is 0.778 and 0.784.
    - It wins on Turkey (0.000 → 0.24–0.27) and loses on xBD images (0.456 → 0.35–0.38).
    - Counting on the validation questions stays at R11's level (37–39%).
    - **26 Oct: at risk (E5).** These scores include leaked validation pairs. Without the 21 found so far, the four combinations of steps 2 and 3 score 0.2915–0.3070.
    - The gate is so far on one U-Net seed (R11). Step 3 re-scores the same graders with R11s7's pieces.
    - Results in `Champion_Task/R16c-results/step2/` (Kaggle's output folder: `R16c-truth`).
    - **26 Oct, clean (E5's 40 pairs removed):** steps 2 and 3 score 0.2838–0.2980, so they fail the gate.
  - **Step 5 result (26 Oct, 3.8 h on account 1): it doesn't replace step 2.**
    - Its SAM 3 + R11 map scores 0.2434 and 0.2158. The grader, trained on SAM 3's outlines, over-calls R11's pieces as destroyed.
    - **Its SAM 3-only map passes O1's gate on all 645 pairs (0.3347 and 0.3242) and on clean data (0.3161 and 0.3051), at mIoU 0.547.** It isn't kept against R11, because its mIoU is 0.035 lower.
    - The Destroyed bias hit the search grid's edge (−4.0) in 38 of 40 epoch searches, so a wider grid is a possible next try.
    - Results in `Champion_Task/R16c-results/step5/`.
    - **Step 5b, a third grader seed (44): built and dry-run checked 26 Oct, ready to run on account 1 (about 2.5–3.5 h).** It checks that seed 43's narrow clean pass isn't luck.
      - Its verdict rule, fixed before the run: all three seeds above 0.30 on clean data means met; otherwise met on the mean only; otherwise not met.
      - [r16c-step5b-third-grader-seed.ipynb](Champion_Task/r16c-step5b-third-grader-seed.ipynb).

## 7. System track: in parallel, mostly no segmentation training

### H2 · Per-building table and SQL counting agent
- **Source:** [Paper] DamageScope (SQLite + vector store, with an LLM choosing the query) and Change-Agent.
- **Goal:** O2 and O3. Exact answers to questions like "how many destroyed buildings within 100 m of X", without the VLM counting pixels.
- **Build:**
  1. From R9's predictions (E1), build one SQLite row per building blob: image, ID, class, area, centroid, bounding box, and the distance to the nearest destroyed building.
  2. Give our Qwen (or a small local LLM) the table schema and let it write SQL for a question.
  3. Run the SQL and have the model phrase the answer.
  4. Test on 100 BDC questions and compare with H1 V1.
- **Outputs:** `buildings.sqlite`, `agent_eval.csv`, `README.md`.

### H2b · Graded test of the SQL agent (added 22 Oct)
- **Why:** H2's 100 counting questions all need the same query, so they can't separate the models. Counting needs no model at all (a fixed query does it), and a second model costs memory on low-resource hardware.
- **Source:** [Paper]
  - Zhong, Yu & Klein, EMNLP 2020 (test-suite accuracy);
  - TrustSQL and the EHRSQL 2024 shared task (refusals and the RS(10) reliability score);
  - Dr.Spider, ICLR 2023 (robustness to rephrasing);
  - DAIL-SQL and Rajkumar et al. (the prompt format);
  - Self-Debugging, ICLR 2024 (retries with the error message).
- **Goal:** O2, O3 and O4. Decide which model, if any, writes SQL in the deployed system.
- **Build:** [h2b-sql-agent-graded-test.ipynb](Champion_Task/h2b-sql-agent-graded-test.ipynb).
  - **Questions:** 29 answerable questions in 12 groups, each in 3 phrasings on 3 images, plus 12 the tables can't answer; 297 items in all.
  - **Grading:** on 5 versions of H2's table.
  - **Models:** our 7B, the base VL-3B (L2's student), Qwen2.5-Coder 1.5B and 3B, each in 4-bit and fp16.
- **Decision rule (fixed 22 Oct):**
  - A configuration passes at 90% or more test-suite accuracy and RS(10) above 0.
  - Our VLM writes the SQL if it passes in 4-bit.
  - Otherwise the smallest passing separate model is the fallback, with its memory cost.
  - If nothing passes, fixed queries only.
  - Re-run on the distilled 3B after L2.
- **Outputs:** `summary.csv`, `by_group.csv`, `pairwise.csv`, `decision.md`, `README.md`.
- **Result (23 Oct, Noman's account, 47 min): done; no configuration passes, so the system answers through fixed queries only.**
  - Our 7B in 4-bit: 69.0% [57.1, 80.5], RS(10) −89.6, with the most refusals of unanswerable questions (94%). The best accuracy was Coder-3B in fp16's 73.9%, and every RS(10) was negative. The base VL-3B in 4-bit: 46.4%.
  - Weak for every model: pairs of buildings, the "no destroyed building nearby" NULL case, positions and distances.
  - The re-run on the distilled 3B is needed only if free-form questions come back. Details in [Task_Ledger.md](Task_Ledger.md) and [H2b-results/H2b/](Champion_Task/H2b-results/H2b/README.md).
  - Why a framework's SQL agent (LangChain's) wouldn't change this, and what could bring free-form questions back: [System_Design_Notes.md §3](System_Design_Notes.md).

### H2c · Free-form questions, second try (added 24 Oct)
- **Why:** H2b's models failed the bar zero-shot. In the literature, models passed a similar bar only with in-domain training and an abstention gate. In EHRSQL 2024, fine-tuned 3B and 7B models scored 44 at RS(10), while a prompted-only 7B scored −713 ([System_Design_Notes.md §3.5](System_Design_Notes.md)).
- **Source:** [Paper] the EHRSQL 2024 shared task (fine-tuning plus a gate); TrustSQL (retrieved examples, and the five-sample agreement gate).
- **Goal:** O3 and O4: free-form numeric questions in the system, if they can be made reliable.
- **Build:**
  1. **Question–SQL pairs for our table** (CPU): question templates times paraphrases, each checked by running its reference query on H2's table. Paraphrase writing can go to teammates, since a script checks every pair.
  2. **Fine-tuning:** a separate LoRA adapter on the final 3B (after L2), or on the 7B if L2 isn't ready. The main model stays untouched.
  3. **Retrieval:** similar solved pairs go into the prompt (TrustSQL's SQLPROMPT).
  4. **Agreement gate:** sample five queries, and answer only if they agree (TrustSQL's VOTING); otherwise refuse.
  5. **A new question set,** written before any run, with unseen question types and unanswerable questions, graded like H2b (test suites, five table versions).
- **Decision rule (fixed 24 Oct, before any run):** it passes with at least 90% test-suite accuracy and RS(10) above 0 on the new set. Also reported: RS at 0, 5 and 10, a risk–coverage curve with its RC-AUC (as defined in TimeSpot, ICML 2026; [Paper_Audit §4.15](Paper_Audit_and_Integration_Plan.md)), and first-try accuracy.
- **If it passes:** free-form numeric questions ship with the gate, labelled as model answers.
- **If it fails: fall back to the system as it was.** Fixed queries and the question templates stay; the adapter is dropped; nothing else changes.
- **Order:** after L2 (the agreed order is T2, T3, L0, L2 first). The question–SQL pairs can be built earlier, on CPU.

### H3 · VLM input upgrades (variants in the H1 harness)
- **Source:**
  - [Paper] TEOChat (before and after images with image labels), GeoPixel (numbered marks, tiling) and ImageRAG (send full-resolution crops of the relevant areas with their coordinates);
  - Tamanna's [Plan 2](Tamanna_task_2/Plan_2/Plan_2_Visual_RAG_Patch_Zooming.md), which already describes the crop step.
- **Goal:** O2, KR2's "higher-resolution VLM".
- **Variants** (each scored on the six tasks like V2):
  - **V3:** before and after images labelled "Image 1 (before)" and "Image 2 (after)".
  - **V4:** numbered marks drawn on each detected building, so the VLM can say "building 12".
  - **V5:** the full image plus up to 4 full-resolution crops of the most damaged areas, each with its pixel box.
- **Done when:** a variant is kept only if it beats V2 on some task without lowering the others.

### H4 · GEOBench-VLM external counting test
- **Source:** [Paper] GEOBench-VLM (ICCV 2025). Its damaged-building counting is scored at 56.7% for GPT-4o and 50.0% for Qwen2-VL.
- **Goal:** O2 — a third-party check of hybrid counting.
- **Build:**
  1. Download the benchmark from its GitHub (`The-AI-Alliance/GEOBench-VLM`).
  2. Take the xBD damaged-building-counting and disaster-type questions.
  3. **First** check that their xBD tiles don't overlap our training tiles. Report the overlap and score only the clean ones.
  4. Run H1 V1 and V2.

### T2 · DisasTeller on local models
- **Source:** [Plan] the CSE499A future work "DisasTeller backend swap" and KR3.
- **Goal:** O3 — no Gemini calls.
- **Base:** `support/Ridita_Integration_of_VLM__/DisasTeller/`.
- **Build:**
  1. Serve our fine-tuned Qwen with vLLM on 2×T4 (OpenAI-compatible server).
  2. Point CrewAI's LLM at it instead of `gemini/gemini-3.1-flash-lite`.
  3. Replace the three image tools' `genai.GenerativeModel` calls with local Qwen calls.
  4. Replace `gemini-embedding-001` with a local sentence-transformers model, and rebuild the Chroma database of guidelines. **Built by Ridita (25 Oct) as hybrid search:** [Offline_RAG_KR2/](Offline_RAG_KR2/README.md) holds 405 cited sections from six sources.
     - Each question is ranked by meaning (MiniLM embeddings) and by keywords (BM25), and the two rankings are merged by reciprocal rank fusion.
     - Keywords catch exact terms ("ASR level", "possible damage") that a long section's embedding can blur.
     - It isn't wired in or tested on report questions yet ([Task_Ledger §9](Task_Ledger.md)).
  4b. **Retrieval test before wiring it in** (added 26 Oct):
     - 30 real report questions, each with the sections that should come back, written before any search runs.
     - Pass: an expected section in the top 3 for at least 24 of 30.
     - Report the hits for meaning alone, keywords alone and the hybrid, so the hybrid's gain is measured rather than assumed.
     - If meaning alone does as well, the simpler search is kept.
  5. Feed the agents per-building facts (H2) and hot spots (T1) as text.
  6. Offline by default: no web tools and no "search the internet" wording unless an online switch is turned on, and new offline wording for tasks 3–6. Details and the test rule: [System_Design_Notes.md §2](System_Design_Notes.md) (recommended 24 Oct).
  7. Consistency check (added 26 Oct, after TimeSpot's consistency-violation rates; [Paper_Audit §4.15](Paper_Audit_and_Integration_Plan.md)): every number in each report is compared with the fixed queries on H2's table, and the run passes only with zero contradictions.
- **Outputs:** the same 6 report files as the Gemini run, a log proving no external API calls (run once with internet off after setup), and `README.md` comparing them with the Gemini reports from 20 July.

### T3 · Unified offline demo and public showcase
- **Source:** [Plan] the CSE499A future work "unified deployment" and O3; Tamanna's [Plan 3](Tamanna_task_2/Plan_3/Plan_3_Offline_Interactive_Demo.md); the showcase plan agreed 24 Oct, with its reasoning and what last season's winner showed in [System_Design_Notes.md §4](System_Design_Notes.md).
- **Goal:** O3 — one offline system, and a showcase the public can use.
- **Owner:** Abrar (decided 24 Oct), who also chooses the name and look later.
- **Two pieces (agreed 24 Oct):**
  1. **Public showcase site:** the project's story, from results computed in advance: before/after sliders with the damage map, videos of the live run, a building map where clicking a building shows its facts, the benchmark panel, the limitations and the team.
  2. **Interactive demo website, deployed on Vercel:** animated, with current UI/UX, running the chain below. Both pieces can be one Next.js project (a story page and a demo page).
- **3D damage city (agreed 24 Oct):** a view in the website where each building outline from the damage map is raised into a 3D block, coloured by damage class. Heights are schematic (for example, destroyed buildings drawn as low slabs), and the view is labelled as a 3D view of our 2D results, not a reconstruction. About a week, no GPU; it uses the building outlines from the gallery export.
- **Order (agreed 24 Oct):** T2, T3, L0 and L2 come before any work on Layers 2–4 of the day-1 vision ([System_Design_Notes.md §5](System_Design_Notes.md)).
- **The chain** (live at the booth with internet off; the public site shows the same steps from results computed in advance):
  1. pick a gallery image, or upload a before and after pair;
  2. the damage map (R11 with TTA on a GPU, or the INT8 edge model on a CPU);
  3. per-building counts by fixed queries (H2), linked to their buildings on the map;
  4. the heatmap (T1);
  5. fixed question templates instead of a free-form question box (H2b);
  6. the DisasTeller report on local models (T2), shown as it's written;
  7. the benchmark panel.
- **How the website runs the models:** Vercel has no GPU, so the public site works from results computed in advance. At the booth, the same front end runs on the laptop and calls a local Python API that runs the chain offline. Stretch, to be tried (decided 24 Oct): the INT8 segmenter in the visitor's browser (ONNX Runtime Web), so uploads work with no server. A feasibility test comes first ([System_Design_Notes.md §4.3](System_Design_Notes.md)).
- **Base:** `support/Abrar/Implementation_DisasterM3/gradio_app.py` (already loads a U-Net and Qwen) becomes the local API's test page and the fallback if the website isn't ready. Its `share=True` opens a public tunnel, so it must be off offline.
- **Before building:** T2's reports; a gallery export (4–6 validation pairs with masks, building facts, heatmaps and each image's accuracy); L1 repeated on R11 (R11's own INT8 export); the browser test ([Champion_Task/T3-browser-test/](Champion_Task/T3-browser-test/README.md)) repeated on the booth laptop, and on a mid-range phone once the site is on Vercel with the COOP and COEP headers; a Vercel account linked to the repository; Abrar's choice of name and look.

  Live mode must run with internet off after setup. A container is optional, not a key result.

### L0 · A better 7B teacher before distilling
- **Decided 19 Oct:** improve the 7B first, then distil (L2). A student learns from its teacher, and ours trails the paper's own fine-tune of the same model.
- **Source:** [Paper] DisasterM3, Table 2 and Appendix B.3 (arXiv 2505.21089v2). The local PDF stops at page 15, so the appendix was read online.
- **Goal:** O4 (a better teacher for the 3B) and O2 (six-task accuracy).
- **Where we stand (optical questions, greedy):** our 7B averages 36.1%, against 31.2% for the base model and 40.4% for the paper's fine-tune. The gaps to the paper are on disaster type (71.1 against 83.6) and relational reasoning (23.3 against 36.2); see H1's like-for-like table.
- **Recipe, ours against the paper's:**

  | Setting | Paper (Appendix B.3) | Ours (`train_qwen_disasterm3.ipynb`) |
  |---|---|---|
  | Method | LoRA on the language model, vision encoder frozen | Same, but the base model is held in 4-bit (QLoRA) |
  | LoRA | rank 64, alpha 16, dropout 0.05 | Same |
  | Optimiser | AdamW, lr 2e-4, β 0.9 / 0.95, cosine | Same, plus 3% warmup |
  | Batch and length | Global batch 256, 1 epoch | 1 × 256 gradient accumulation, 1 epoch; all 217 steps finished |
  | Precision and hardware | Not stated; 4× H100 | fp16 on 1× T4 |
  | Image size in training | Not stated | Capped at about 512 image tokens per image |
  | Image size in testing | Each model's default (full size, about 1,300 tokens for a 1024 px image) | 512 px in H1 V2/V2b (about 324 tokens) |
  | Data | Instruct question-answer samples | The 55,764 Instruct entries with a text answer; the 37,204 segmentation entries have none |

- **Ruled out:**
  - The answer format: our 7B answers with letters only on 99.9–100% of questions in every task.
  - Using the authors' model as the teacher: they haven't released their fine-tuned weights (their Hugging Face page lists only EarthVQA-pretrained and GeoFM).
- **Build:**
  1. **Step 1, re-test (about 2–3 GPU hours on T4 ×2):** re-score our 7B on DTR and ORR, optical questions (1,211), at three sizes for a 1024 px image:
     - 1036 px, the full size (Qwen's default, the paper's setting);
     - 616 px, our training cap;
     - 504 px, H1 V2's size, re-run as the control.

     Built 19 Oct: `Champion_Task/l0-step1-image-size-check.ipynb`, a copy of H1 V2's notebook with only the image size changed. The decision was fixed before the run, against the paper's 83.6 and 36.2:
     - **Gap closed:** the best size comes within 3 points on both tasks. Re-score all six tasks at that size, then L2 with the same 7B.
     - **Partly closed:** at least half of each gap is recovered. Re-score all six at that size, then decide on step 2.
     - **Not closed:** go to step 2.

     **Result (19 Oct, 1.7 h): not closed.**
     - Disaster type: 71.1 → 75.4 → 75.6 (504 → 616 → 1036 px).
     - Relational: 23.3 → 26.9 → 31.1.
     - The control reproduced H1 V2 exactly.
     - At full size the gaps to the paper are 8.0 and 5.1 points. Full size recovers 36% and 61% of them, and "partly closed" needs half of both.
     - Full size becomes the 7B's test setting. Step 2 (or stopping here) is your decision. Re-scoring the other four tasks at full size would take about 11 GPU hours: 9,464 optical questions at 4.7 s per two-image question and 2.4 s per one-image question.
  2. **Step 2, only if step 1 leaves the gap:** a top-up training of our LoRA adapter on DTR and ORR (3,509 training entries) plus an equal replay sample of the other tasks, at up to about 1,024 image tokens.
     - It takes about 10–20 T4 hours; the DDP notebook can halve that. Retraining from scratch would take about 45–90.
     - Keep it only if DTR and ORR rise and no other task drops by more than 1 point.

     **Built 20 Oct:**
     - [l0-step2-full-size-rescore.ipynb](Champion_Task/l0-step2-full-size-rescore.ipynb) re-scores at full size: the old 7B on the four tasks step 1 didn't ask (9,464 questions, about 11 h, in two parts), then the new 7B on all six (10,675, about 12 h).
     - [l0-step2-topup-training.ipynb](Champion_Task/l0-step2-topup-training.ipynb) trains on DTR + ORR plus an equal replay at full size (up to 1036 × 1036): 7,018 examples, global batch 256, about 28 steps, about 7–14 h on T4 ×2 (the smoke test measures it).
       - A new rank-64 adapter on the merged 7B replaces "continue the old adapter": its saved layer pattern is for transformers 4.49 and matches nothing in transformers 5.
       - It resumes from Hugging Face, and merges and pushes `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP` at the end.
       - **Memory at full size, 20 Oct:** the 7B is at the edge of a T4.
         - Two attempts ran out of GPU memory. Both were fixed without changing the data or the batch: the waiting micro-batches stay on the CPU, and the optimizer is 8-bit paged AdamW.
         - The smoke test measured 10.0 s per example, so about 19.5 h in 2–3 sessions.
         - If memory still runs short, the next options are: leave out the few longest examples, use a smaller image size for two-image examples only, or use LoRA rank 32. Each is your decision (details in the [ledger](Task_Ledger.md)).
     - **Status, 27 Oct:**
       - The training is complete: 28 of 28 steps over sessions 1, 2, 3a and 3b, with losses flat at 0.49–0.60. Checkpoint-28 is on Hugging Face.
       - Session 3b's merge failed because Kaggle's torchao 0.10 trips the upgraded peft's version check (fp16 merge only).
       - **Built and dry-run checked 26 Oct: [l0-step2-merge.ipynb](Champion_Task/l0-step2-merge.ipynb),** a named copy that removes torchao, restores checkpoint-28 and merges on the CPU. No GPU and no retraining.
       - **27 Oct:** run 1 stopped in Cell 2 on my own over-strict check (peft saves the 196 modules as 88 name patterns). The adapter is fine. Fixed and dry-run again.
       - **Merged 27 Oct (run 2, account 3, 10 min):** exactly the 196 adapted projections changed, and none holds an inf or NaN. `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP` (16.6 GB, public) is on Hugging Face, every file's size checked. Results in [L0-step2-results/merge/](Champion_Task/L0-step2-results/merge/L0S2-merge/README.md).
       - **Next:** the new 7B's six tasks in the re-score notebook's two parts, A (BDC, BBR) on account 4 (Tamanna's, 28.5 GPU h, joined 27 Oct) and B (DRE, DSR, DTR, ORR) on account 2. Then the verdict on your PC, reported both ways (all and E5-clean).
- **Done when:** the 7B's six-task scores are re-measured at the paper's setting, and L2 uses the better of the two models as its teacher.

### L2 · 7B → 3B distillation
- **Source:** [Plan] CSE499A future work and KR4; [Paper] QLoRA (we already use it via Unsloth).
- **Goal:** O4.
- **Waits for L0** (decided 19 Oct): the teacher is improved first.
- **Base:** the two ready notebooks in `support/Abrar/Implementation_DisasterM3/`, kept unchanged as the originals:
  - `generate_teacher_cot_train_3b_distill.ipynb` has never been run. Step 1 runs as its fixed copy, [Champion_Task/l2-step1-teacher-rationales.ipynb](Champion_Task/l2-step1-teacher-rationales.ipynb) (19 Oct; named `l2-stage1-teacher-rationales` until 20 Oct): Kaggle's own torch, fp16 on 2×T4 in batches, 4,000 checkable examples, and only answers that match the label kept;
  - `train_qwen_disasterm3_3b_distill.ipynb` needs the same environment fix, as a copy, and **its outputs cleared**, because they're left over from the 7B run.
- **Build:**
  1. Generate the teacher's answers on the training set. QLoRA's lesson: a smaller, cleaner set beats a large noisy one. The smoke test (19 Oct) showed that our fine-tuned 7B answers in one sentence instead of reasoning, so this is answer-level distillation; explanations come from L2b.
  2. Fine-tune Qwen2.5-VL-3B, with L2b's explanations as a second skill.
  3. Score all six tasks with the H1 harness, against the 7B.

### L2b · Explanations as a second skill ("why?" on request)
- **Source:** [Paper] Distilling Step-by-Step (Hsieh et al., Findings of ACL 2023; see Paper_Audit §4.12), BDAChat (explanations are its main strength) and ImageRAG (fine-tuning with reasoning added 18 points). Decided on 19 Oct, after L2's smoke test.
- **Goal:** O4 and O3. The field 3B answers briefly by default, which keeps it fast and in the benchmark's format, and explains its answer when the user asks "why?".
- **Why this design:** the papers' lessons and the comparison of short answers (A) against reasoning before every answer (B) are recorded in [Paper_Audit §5.3](Paper_Audit_and_Integration_Plan.md#53-decided-how-the-distilled-3b-learns-19-october-2026).
- **Build:**
  1. **Explanation data.** For a few thousand of step 1's checkable training examples, give a model the images, the question, the correct answer and true facts computed from DisasterM3's masks (counts per class as 8-connected blobs, and where the damage is). Ask for 2–3 sentences on what in the before and after images shows the answer. Use the base Qwen2.5-VL-7B-Instruct: our fine-tuned 7B answers in one sentence and doesn't explain.
  2. **Check each explanation.** Keep it only if it agrees with the answer and every number it states matches the mask facts; drop the rest.
  3. **Train two skills in step 2.** The normal prompt has the short answer as its target; the same prompt followed by "Explain your answer." has the explanation as its target. The 3B writes an explanation only when asked, so normal answers stay fast.
  4. **Score both.** Six-task accuracy with the normal prompt, which must not fall, and a hand-checked sample of about 50 explanations, judged against the image and the facts.
- **Don't** train the 3B to reason before every answer: it is slower in the field (Grace, AWQ), clashes with the benchmark's letters-only answers, and risks teaching made-up evidence, since VLMs find and count buildings poorly (TEOChat, GEOBench-VLM).
- **Done when:** the 3B's six-task accuracy is no lower than without explanations, and most of the sampled explanations are judged faithful.

### L3 · AWQ 4-bit 3B and confidence handoff to the 7B
- **Source:** [Paper] AWQ (4-bit, about 3× faster) and Grace (route an answer by the model's own confidence).
- **Goal:** O4 — "fits in about 6 GB".
- **Build:**
  1. Quantise the distilled 3B with AutoAWQ. Also check whether Qwen publishes an official AWQ build to compare against.
  2. Measure memory, tokens per second and six-task accuracy.
  3. Handoff: compute the geometric mean of the probabilities of the generated tokens (`output_scores=True`). Below a threshold, send the question to the 7B.
  4. **Before choosing the threshold, check the confidence** (added 26 Oct, from TimeSpot, ICML 2026; [Paper_Audit §4.15](Paper_Audit_and_Integration_Plan.md)): the 3B's expected calibration error and its risk–coverage curve with RC-AUC on the six tasks. TimeSpot found open VLMs overconfident, so calibrate on validation data first (for example, temperature scaling), then pick the threshold from the curve.
  5. Report the share answered by the 3B, the accuracy against the 7B alone, and the latency.

### L4 · EfficientSAM3 against SAM 3
- **Source:** [Paper] EfficientSAM3 (its paper reports no results yet, so we measure) and [Plan] the edge step of the SAM 3 test plan (Research_Directions §7.4).
- **Goal:** O4 — a field-sized building finder.
- **Build:** repeat E3's part 2 (building finding) with the smallest and medium EfficientSAM3 students. Report building IoU, milliseconds per image on a T4 and on a CPU, and memory.

## 8. Brief for an AI coding agent (copy, then paste one task under it)

> You are writing a Kaggle notebook for a university capstone: post-disaster building-damage segmentation and a vision-language model on the DisasterM3 dataset.
>
> **Read first:**
> - `Essential_task_delegation/Next_Steps_Roadmap.md` §3 (shared rules) and the task below;
> - the base notebook or script the task names.
>
> **Rules:**
> - Copy the base notebook and change only what the task lists.
> - Keep the pinned versions in Cell 1.
> - Never train on the 645 validation images (the split is `train_test_split(pairs, test_size=0.1, random_state=42)`).
> - Never use radar (SAR) images for segmentation.
> - Score with the E-protocol (full 1024 images, last epoch, dataset-level confusion matrix, standard and 8-view TTA rows).
> - Make the notebook Run-All safe, with a `SMOKE_TEST` flag.
> - Write all outputs to `/kaggle/working/<task-id>/`, and big temporary files to `/kaggle/temp/`.
> - Read secrets only from Kaggle Secrets.
>
> **Deliver:**
> 1. the `.ipynb`;
> 2. a short README: inputs to attach, expected runtime, outputs, and how to check them against the "Done when" line;
> 3. a list of any assumption you had to make.
>
> Ask before changing anything the task doesn't mention.
>
> **Task:** <paste the task here, e.g. E1>

## 9. Decision gates

| Gate | Decided by | Rule |
|---|---|---|
| G1 | R9 + R9s7 | **Done (18 Oct):** two-seed mean 0.2971, so the gate is not claimed and fine-tuning is closed |
| G2 | E1 | 40% or more of missed destroyed pixels predicted as Background: R11 before R12; otherwise the reverse |
| G3 | E3 | SAM 3 at least as good as R9 at finding buildings on pre-images: it does R16's finding |
| G4 | H1 | V1 above 34.2% on BDC means the counting half of KR2 is met; V2 is kept only if no other task drops by more than 1 point |
| G5 | R15 + R15s7 | Gate claim only with two seeds above 0.30, or a two-seed mean above 0.30, with TTA stated if used |

## 10. GPU budget (approximate)

| Week | Tasks | GPU-h |
|---|---|---|
| This week | E1, E2, E3, H1 (V1, V2) | about 10 |
| Next week | R11, R12 on two accounts; T2 | about 17 |
| Week after | R15 + R15s7 and R16 (R16 runs regardless, decided 19 Oct); H3 variants | about 17–26 |
| Third account, any time | L2, L3 | about 15–25 |

CPU-only, any time: D5, T1, L1 and H2.

## 11. Documents to update as results come in
- [Ablation_Study_Reference_and_QA_Prep.md](Ablation_Study_Reference_and_QA_Prep.md) §7: new segmentation runs (R11 onwards).
- [Phase2_Plan.md](Phase2_Plan.md): E1, R11, R12 and R15 outcomes.
- [Paper_Audit_and_Integration_Plan.md](Paper_Audit_and_Integration_Plan.md) §2: the status of each goal.
- [Research_Directions_Beyond_Finetuning.md](Research_Directions_Beyond_Finetuning.md) §9.
- The README status table, and `CSE499B_Review_report_update.tex` for the next report.
