# E5 · Train/validation leak and the benchmark's copies (26 Oct 2026)

**Why:** Ridita's corrected audit (`BRIGHT_DisasterM3_Integration/DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf`, §8) reports that 40 of our 645 validation pairs (6.2%) are near-copies of training pairs, holding 155,812 of the validation set's 1,808,111 destroyed pixels (8.6%). The cause: xBD's `socal_fire` and `woolsey_fire` share the same before-images, and the random 90/10 split put one twin in validation and the other in training. Her twins have the same before-image, after-images 88% alike, building masks overlapping at IoU 0.72, and 97% of shared building pixels in the same damage class. Every O1 number (R1 to R16c) is scored on these 645 pairs.

## Result: every clean score (26 Oct, after E5 steps 1 and 2)

Step 1 ([step1/README.md](../step1/README.md)) and step 2 ([step2/README.md](../step2/README.md)) ran on Kaggle account 2. All checks passed:
- the split was rebuilt as in every D4 run;
- all 8 models reproduced their D4 CSVs;
- step 1 found the same 40 leaked pairs as Ridita's audit and [leaked_pairs_40.csv](leaked_pairs_40.csv), the third independent check.

"Clean" means without the 40 pairs (605 left). Each "all" column reproduces the published score. The one exception, R16c step 5 with grader seed 43, is explained in the note further down.

### Segmentation (O1: mIoU above 0.52 and Destroyed IoU above 0.30, on two seeds)

| Model | Mode or grader | mIoU: all → clean | Destroyed IoU: all → clean | O1 gate on clean data |
|---|---|---|---|---|
| **R16c step 5, SAM 3's outlines only** | grader 42 | 0.5520 → 0.5469 | 0.3347 → **0.3161** | **pass** |
| **R16c step 5, SAM 3's outlines only** | grader 43 | 0.5522 → 0.547* | 0.3242 → **0.3051*** | **pass** |
| R16c step 2, SAM 3 + R11's pieces | grader 42 / 43 | 0.5869 / 0.5852 → 0.5824 / 0.5810 | 0.3159 / 0.3106 → 0.2980 / 0.2935 | fail |
| R16c step 3, SAM 3 + R11s7's pieces | grader 42 / 43 | 0.5843 / 0.5830 → 0.5799 / 0.5789 | 0.3090 / 0.3006 → 0.2913 / 0.2838 | fail |
| R16c step 5, SAM 3 + R11's pieces | grader 42 / 43 | 0.5496 / 0.5410 → 0.5452 / 0.5366* | 0.2434 / 0.2158 → 0.2261 / 0.1988* | fail |
| R11 | standard / TTA | 0.5811 / 0.5875 → 0.5742 / 0.5805 | 0.2911 / 0.2965 → 0.2641 / 0.2692 | fail |
| R11s7 | standard / TTA | 0.5867 / 0.5913 → 0.5799 / 0.5844 | 0.2869 / 0.2928 → 0.2597 / 0.2652 | fail |
| R9 | standard / TTA | 0.5373 / 0.5440 → 0.5338 / 0.5402 | 0.2921 / 0.3001 → 0.2800 / 0.2874 | fail |
| R9s7 | standard / TTA | 0.5382 / 0.5453 → 0.5347 / 0.5417 | 0.2847 / 0.2941 → 0.2727 / 0.2820 | fail |
| R1 | standard / TTA | 0.5374 / 0.5440 → 0.5335 / 0.5401 | 0.2686 / 0.2822 → 0.2554 / 0.2689 | fail |

\* Seed 43 of step 5: the published score minus the drop measured on the per-building file. See the note below.

1. **On clean data, only R16c step 5's SAM 3 map passes O1's gate.** Graders 42 and 43 give 0.3161 and 0.3051. This map has SAM 3 find buildings on the before-image and the step-5 grader grade each one; no U-Net is involved. Four caveats:
   - It is one of three maps step 5 compared, and it was singled out after seeing the validation scores.
   - Seed 43 clears 0.30 by only 0.005.
   - Seed 43's per-building file doesn't reproduce its published scores exactly.
   - Its mIoU is 0.035 under R11's. So by the keep rule it doesn't replace R11 as the champion, which is a separate question from O1's gate.
2. **The leak helped the before + after models twice as much.**
   - R11 and R11s7 lose 0.027 Destroyed IoU.
   - The after-only U-Nets lose 0.012–0.013, and the R16c maps 0.017–0.019.
   - The reason: each leaked pair shares its before-image with a training pair, so a model that reads the before-image recognises the place.
3. **So R11's Destroyed IoU is no longer level with R9's.**
   - With TTA, on clean data R11 scores 0.2692 against R9's 0.2874, a gap of 0.018. That's beyond the 0.005 noise margin.
   - R11 was kept because its mIoU rose (+0.04, which still holds: 0.5805 against 0.5402) and its Destroyed IoU "didn't drop". On clean data it does drop.
   - What R11 was built for still holds: only 28% of its missed destroyed pixels are called background, against R9's 58%.
4. **Comparisons inside each family still hold.**
   - The after-only U-Nets (R1–R10, R9s7) all drop by 0.012–0.013.
   - The R11 family (R11, R11s7, and R12 by its identical input) drops by 0.027.
   - Comparisons across the two families need the clean numbers.
5. **E1's Gate G2 still holds.** Of the after-only U-Nets' missed destroyed pixels, 56–58% are called background, against the 40% rule.
6. **Counting on validation barely moves.** These are R16c step 1's 1,055 questions, 974 of them clean. R11's blobs of at least 20 px go from 39.24% to 39.01%, and the other counters change by 0.0–0.7 points.

**Note on R16c step 5, grader seed 43:**
- With its searched biases, the per-building file gives Destroyed IoU 0.3254 where the published score is 0.3242. With R16b's fixed rule it gives 0.2922 against 0.2927. Seed 42 reproduces exactly under both rules.
- Three explanations were tested and ruled out:
  - rounding of the stored log-probabilities: no combination of the near-ties, and no single regraded building, gives the published numbers;
  - the biases: no bias on the 0.25 grid reproduces them;
  - the file itself: it has no NaNs or duplicates.
- So the saved seed-43 log-probabilities differ slightly from those the published maps used. The cause is still unknown.
- The clean score uses the lower value: the published 0.3242 minus the drop measured on the file (0.0191), which gives 0.3051. Straight from the file it would be 0.3063.

### Benchmark (O2, and the 7B's six tasks)

Step 1 flagged the benchmark questions whose images copy an Instruct image:
- **after-image copied: 186 of the 10,675 optical questions (1.74%)**, 308 with SAR;
- 512 flagged in all, adding before-image copies and review matches.

In the table below, "clean" drops the after-image copies (the rule fixed before the run), and "strict" drops every flagged question.

| Score | All | Clean | Strict |
|---|---|---|---|
| **O2 counting: R11's TTA counts (H2's oracle), optical BDC** | 32.96% (3,580) | **32.47%** (3,514) | 32.27% (3,449) |
| R11's TTA counts, all sensors | 27.46% (4,978) | 27.47% | 27.27% |
| Old 7B at full size (L0): DTR / ORR | 75.64 / 31.12 | 74.93 / 31.34 | 74.33 / 31.32 |
| Old 7B at full size (L0): BDC / BBR / DRE / DSR | 30.09 / 20.94 / 32.39 / 40.06 | 29.85 / 20.70 / 32.33 / 39.71 | 29.61 / 20.48 / 32.22 / 39.59 |
| Old 7B at full size, mean of the six | 38.37 | 38.14 | 37.93 |
| H1 V2, 512 px, five tasks, optical: plain / with facts | 31.26 / 28.22 | 31.07 / 28.02 | 30.86 / 27.88 |
| H1 V2b, the 7B on BDC, optical: plain / with facts | 32.01 / 28.10 | 31.82 / 27.84 | 31.60 / 27.52 |
| H1 V1, R9's counter, optical BDC | 26.54 | 26.16 | 25.90 |
| H2's agent, 100-question sample per LLM | 34.0 | 33.3 | 33.3 |

- **Every drop is 0.7 points or less, and no decision changes.**
- **O2's counting is still short of the bar.** The bar is the paper's base 7B at 34.2%, which was never trained on Instruct and so isn't affected. Clean, the gap grows from 1.24 to 1.73 points.
- **What was copied:**
  - Same-pixel copies: mostly `la_palma_volcano` in the benchmark, which is `spain_volcano` in Instruct (the same images under two event names), plus `palu_tsunami`, `sunda_tsunami` and `pinery_bushfire`.
  - Near copies: mostly woolsey/socal fire twins, where a benchmark after-image's twin is in Instruct.
- **The review pictures (9 of the 41 looked at):**
  - Same-pixel copies are true copies.
  - Woolsey/socal pairs at dHash 7 (the review tier) are the same place, so for fire questions the strict column is the safer bound.
  - One `myanmar_hurricane` "near copy" is an almost blank dark before-image, so a false match. It affects only the strict column.
  - The 3 validation review matches show different places.
- **A cosmetic fault:** step 1's "By event" table lists before-image file names as events, because its event parser misses `_pre_disaster`. The counts and CSVs are right.

### How it was scored, and the alternatives

- **Segmentation:**
  - Per-image 4 × 4 pixel counts (step 2 and E1) or per-building pixel counts (R16c's `instances_val.csv`), summed without the 40 pairs.
  - The R16c maps are rebuilt from each building's log-probabilities and the calibrated biases; buildings graded "None" are background.
- **Benchmark:**
  - Question IDs (`{subset}_{index}`, as in H1 and L0) matched to step 1's `bench_questions_flagged.csv`. Unreadable images count as wrong, as in the published scores.
  - H2's oracle row was rebuilt from `buildings.sqlite` and `benchmark_release.json`. It reproduces 27.46% and 32.96%.
- **Alternatives:**
  - Re-running inference gives the same numbers at GPU cost.
  - A new split grouped by before-image fingerprint is the real fix for future runs. It changes the validation set, so it's reported alongside, not instead.
  - Local features (SIFT) for the copy check: not needed, because the review pictures show the thresholds work.

## What was checked here (on Abrar's PC, no GPU)

1. **Twins:** MD5 of every `socal-fire` (xBD tier1) and `woolsey-fire` (xBD tier3) before-image, both of the file and of the decoded pixels. Both give 426 identical pairs; 156 of them have both sides in DisasterM3 (its mask folder).
2. **Validation list:** the 631 validation images with at least one building, from R16c step 2's `instances_val.csv` (85 of them from the two fire events; the 86th fire image has no buildings found).
3. **Leak rule:** a validation pair leaks if its twin is in DisasterM3 and not in validation.

| Validation fire images | Count |
|---|---|
| Twin in training (leak) | **21** (80,709 destroyed pixels) |
| Twin also in validation (not a training leak) | 4 |
| Twin not in DisasterM3 | 1 |
| No twin | 59 |

The 21 pairs and their training twins are in [leaked_pairs_local_check.csv](leaked_pairs_local_check.csv).

**Why 21 and not 40:** this check hashed xBD's original files; the audit hashed DisasterM3's own copies, which aren't on this PC. DisasterM3 seems to hold more identical pairs than the originals: the audit counts 264 shared before-images in the pool, against 156 here. Both checks use the same validation set (both give 1,808,111 destroyed pixels), so the audit's 40 are among the 86 fire images. To settle it, use Ridita's `val_train_leaks.csv` (named in her report but not committed) or hash DisasterM3's before-images on Kaggle.

## Effect on R16c's kept map (Destroyed IoU)

| Combination | All 645 | Minus the 21 leaked pairs | Minus all 86 fire images |
|---|---|---|---|
| R11 pieces, grader seed 42 | 0.3159 | 0.3070 | 0.2891 |
| R11 pieces, grader seed 43 | 0.3106 | 0.3012 | 0.2976 |
| R11s7 pieces, grader seed 42 | 0.3090 | 0.3002 | 0.2826 |
| R11s7 pieces, grader seed 43 | 0.3006 | **0.2915** | 0.2871 |
| Mean of the four | 0.3090 | 0.3000 | 0.2891 |

- The true clean score lies between the last two columns. Removing only 21 pairs already puts one combination below 0.30, and the audit's 40 would lower the scores further.
- **So O1 (Destroyed IoU above 0.30 on two seeds) isn't shown on clean data.** The 645-image numbers stay valid for comparing runs with each other, since every run was scored on the same pairs, but not as a claim against the 0.30 target.

## Effect on R9, R1 and R9s7, and on E1's decision (added 26 Oct)

E1 saved a 4 × 4 pixel count for every validation image (`E1-results/E1/E1_per_image_cm.csv`). So these three models can be re-scored on the PC without a GPU run, by removing whole images.

| Model | Mode | Destroyed IoU: all 645 | Minus the 21 | Minus all 86 fire images | Missed destroyed pixels called background: all / minus 21 / minus 86 |
|---|---|---|---|---|---|
| R9 | standard | 0.2921 | 0.2856 | 0.2736 | 59.5% / 58.7% / 56.8% |
| R9 | TTA | 0.3001 | 0.2937 | 0.2806 | 60.2% / 59.4% / 57.5% |
| R1 | standard | 0.2686 | 0.2614 | 0.2481 | 59.4% / 58.5% / 56.7% |
| R1 | TTA | 0.2822 | 0.2749 | 0.2618 | 59.7% / 58.9% / 56.9% |
| R9s7 | standard | 0.2847 | 0.2786 | 0.2663 | 57.6% / 56.7% / 54.6% |
| R9s7 | TTA | 0.2941 | 0.2882 | 0.2754 | 58.9% / 58.1% / 55.9% |

- **E1's decision holds.** Gate G2 (40% or more of missed destroyed pixels called background → run R11 first) is met by a wide margin in every column, so R11 would still have gone first.
- **R9's TTA score of 0.3001 was the only one above 0.30 before any leak was found.** Without the 21 pairs it's 0.2937.
- E5 step 2 still re-runs R9 and R1, but only as a cross-check of this table. R11 and R11s7 need step 2, because their runs saved only totals.

## The full list: 40 leaked pairs (added 26 Oct, from Ridita's audit README)

Ridita sent the README of her audit outputs (`audit_outputs/README.md`; the CSVs aren't pushed yet). It settles 21 against 40:

- **Her 40 include all 21 found here**, with the same twins and the same destroyed pixels.
- **Why this check missed 19:** their socal_fire side isn't an xBD tier-1 image. 10 are xView2 test images and 9 are in neither set on this PC (probably xBD's hold set). This check hashed only tier 1, so those had nothing to match. Hashing DisasterM3's own copies finds all 40.
- **DisasterM3 didn't change the images:** every twin is a byte-identical file pair. "264 shared before-images" is across all of DisasterM3; 250 of them are inside the pool.
- **Side finding:** DisasterM3 Instruct holds xView2 test images (287 of its 928 socal_fire before-images). Nothing trained on it should ever be scored on xView2's test set.

Her section 4 lists the 19 missing pairs. Together with the 21 they form [leaked_pairs_40.csv](leaked_pairs_40.csv). Two checks back it:
- the destroyed pixels add up to her totals (80,709 + 75,103 = 155,812 of 1,808,111, 8.6%);
- E1's own per-image ground truth gives the same destroyed-pixel count for each of the 40.
- **Her `val_train_leaks.csv` arrived later on 26 Oct and matches the list exactly:** the same 40 validation pairs, the same twins and the same destroyed pixels. Her medians check out too: after-image correlation 0.878, building IoU 0.722, same damage class 0.967. In 3 pairs the twin's damage labels mostly disagree (`woolsey_fire_00000272`, `00000514`, `00000529`: 9%, 0% and 33% of shared building pixels in the same class). They're still removed, because the identical before-image shows the model the same buildings.

### Clean scores without the 40 (Destroyed IoU)

| Model | Mode | All 645 | Minus the 21 | **Minus the 40 (clean, 605 pairs)** | Minus all 86 fire images |
|---|---|---|---|---|---|
| R16c, R11 pieces, grader 42 | | 0.3159 | 0.3070 | **0.2980** | 0.2877 |
| R16c, R11 pieces, grader 43 | | 0.3106 | 0.3012 | **0.2935** | 0.2962 |
| R16c, R11s7 pieces, grader 42 | | 0.3090 | 0.3002 | **0.2913** | 0.2811 |
| R16c, R11s7 pieces, grader 43 | | 0.3006 | 0.2915 | **0.2838** | 0.2858 |
| R9 | standard | 0.2921 | 0.2856 | **0.2800** | 0.2736 |
| R9 | TTA | 0.3001 | 0.2937 | **0.2874** | 0.2806 |
| R1 | standard | 0.2686 | 0.2614 | **0.2554** | 0.2481 |
| R1 | TTA | 0.2822 | 0.2749 | **0.2689** | 0.2618 |
| R9s7 | standard | 0.2847 | 0.2786 | **0.2727** | 0.2663 |
| R9s7 | TTA | 0.2941 | 0.2882 | **0.2820** | 0.2754 |

- **O1 isn't met on clean data.** All four R16c combinations fall below 0.30 (0.2838–0.2980, mean 0.2917). The best, R11 pieces with grader 42, is 0.002 short.
- **The leak lowers every model by a similar amount:** 0.012–0.018 without the 40. That's why comparisons between runs on the 645 still hold.
- **E1's Gate G2 still holds.** Without the 40, 55.7–58.4% of missed destroyed pixels are called background, against the 40% rule.
- In this table, the "minus 86" column for R16c removes only the 85 fire images that have rows in `instances_val.csv`, so it differs slightly from the table above. It's only a bound anyway.
- **Still to do:** R11 and R11s7 alone (E5 step 2, Kaggle), R16c step 5 once it finishes, and the benchmark check (E5 step 1's benchmark part, which Ridita couldn't run: the benchmark images aren't on her PC).
- **Script:** `e5_clean40.py`, run on Abrar's PC. Its asserts check the 21/19/40 sums, the per-image ground truth, and the 1,808,111 total. Without any removal, it reproduces every reported score exactly.

## How it was scored, and the alternatives

- **How:** Destroyed IoU = TP / (TP + FP + FN), from the per-building pixel counts in `instances_val.csv`. Each building's predicted grade is the highest of its three log-probabilities plus the calibration biases (`R16c-results/step2/calibration.json`). FN uses the validation set's 1,808,111 destroyed pixels minus those of the removed images, counted from DisasterM3's local destroyed-building masks (the `socal_fire` count matches E4's exactly). This reproduces R16c's reported 0.3159, 0.3106, 0.3090 and 0.3006 exactly before any removal.
- **Removing whole events (all 86 fire images):** too strict, since 46 or more of them don't leak; used here only as the bound.
- **The exact leak list:** needed for the reportable clean score.
- **A new split grouped by before-image hash** (the audit's recommendation): needed for future runs, but it changes the validation set and breaks comparison with R1 to R16c, so report both.
- **The U-Net runs (R9, R11, R11s7)** can't be re-scored here: no per-image counts were saved. They need a Kaggle re-score on the clean pairs.

## R12's extra xBD tiles: two more twins (added 27 Oct)

The 40 pairs are twins inside DisasterM3. R12 also trained on 1,619 xBD tiles from outside it (D5's 1,007 and D5b's 612), and those were checked by tile ID only, not by content.

| R12 training tile | Validation pair with the same before-image | Destroyed px in the validation pair |
|---|---|---|
| `socal-fire_00000876` (D5) | `woolsey_fire_00000669` | 500 |
| `woolsey-fire_00000262` (D5b) | `socal_fire_00000202` | 6,360 |

- **So R12 trained on 42 leaked validation pairs, not 40.** No other model trained on these tiles.
- **R12's verdict stands.** A leak can only flatter R12, and it still finished within noise of R11, so "not kept" holds. Its clean score would drop 42 pairs; it isn't computed, because no per-image counts were saved for R12.
- **How it was checked (27 Oct, on Abrar's PC):**
  - The file MD5 of the 3,238 before- and after-images of R12's extra tiles was compared with the 1,290 validation files in step 1's `fingerprints_instruct.csv`, which holds every one of them. The twins are byte-identical files, so file MD5 finds them: DisasterM3's copy of `woolsey_fire_00000724`'s before-image has the same file MD5 as xBD's.
  - The method was first checked by reproducing the 426 SoCal/Woolsey before-image twins from the xBD files on the PC (decoded-pixel MD5).
  - Only 456 of the 645 validation pairs are on the PC under xBD names (the rest are other sources or xBD's hold split), which is why the file-MD5 pass through step 1's fingerprints was needed.

## Effect on E2 and E3 (added 27 Oct)

- **E2 (majority vote):** not re-scored, because only its totals were saved. The vote cut R9's Destroyed IoU by 0.064, five times the leak's effect on R9 (0.012), so the verdict can't change.
- **E3 (SAM 3 finding):** SAM 3 never trained on DisasterM3, so the leak can't flatter it. On the 605 clean pairs, from `sam3_s1.csv` and E1's counts: SAM 3 on the before-image grid scores building IoU 0.6302 (0.6304 on all 645) against R9 with TTA's 0.5905 (unchanged). Destroyed footprint recall: SAM 3 0.627, R9 0.708. Gate G3 still passes.

## Next steps

1. ~~Get the exact list.~~ Done 26 Oct: three checks agree on the 40.
2. ~~Re-score on the clean pairs.~~ Done 26 Oct, for every affected score (the tables at the top).
3. ~~Update O1's status and the core documents.~~ Done 26 Oct. The report needs both numbers (all 645, and clean) when it's next updated.
4. Group future splits by before-image fingerprint, so twins stay on one side, and fingerprint every extra training source against validation before using it (R12's tiles were checked by ID only; see above).
5. A third grader seed (44) for step 5's SAM 3 map, to check that seed 43's narrow pass on clean data (0.3051) isn't luck.
   - Built and dry-run checked 26 Oct as `r16c-step5b-third-grader-seed.ipynb`; about 2.5–3.5 h on account 1.
   - It also computes every score on the 605 clean pairs itself, and checks that its saved rows rebuild its maps.
