# E5 · Train/validation leak check (partial, 26 Oct 2026)

**Why:** Ridita's corrected audit (`BRIGHT_DisasterM3_Integration/DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf`, §8) reports that 40 of our 645 validation pairs (6.2%) are near-copies of training pairs, holding 155,812 of the validation set's 1,808,111 destroyed pixels (8.6%). The cause: xBD's `socal_fire` and `woolsey_fire` share the same before-images, and the random 90/10 split put one twin in validation and the other in training. Her twins have the same before-image, after-images 88% alike, building masks overlapping at IoU 0.72, and 97% of shared building pixels in the same damage class. Every O1 number (R1 to R16c) is scored on these 645 pairs.

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

## How it was scored, and the alternatives

- **How:** Destroyed IoU = TP / (TP + FP + FN), from the per-building pixel counts in `instances_val.csv`. Each building's predicted grade is the highest of its three log-probabilities plus the calibration biases (`R16c-results/step2/calibration.json`). FN uses the validation set's 1,808,111 destroyed pixels minus those of the removed images, counted from DisasterM3's local destroyed-building masks (the `socal_fire` count matches E4's exactly). This reproduces R16c's reported 0.3159, 0.3106, 0.3090 and 0.3006 exactly before any removal.
- **Removing whole events (all 86 fire images):** too strict, since 46 or more of them don't leak; used here only as the bound.
- **The exact leak list:** needed for the reportable clean score.
- **A new split grouped by before-image hash** (the audit's recommendation): needed for future runs, but it changes the validation set and breaks comparison with R1 to R16c, so report both.
- **The U-Net runs (R9, R11, R11s7)** can't be re-scored here: no per-image counts were saved. They need a Kaggle re-score on the clean pairs.

## Next steps

1. Get the exact list: Ridita's `val_train_leaks.csv`, or hash DisasterM3's before-images on Kaggle.
2. Re-score on the clean pairs: R11 and R11s7 (and R9, R1) on Kaggle; R16c locally from `instances_val.csv`.
3. Update O1's status, the tracker, the ledger and the report with both numbers (all 645, and clean).
4. Group future splits by before-image hash, so twins stay on one side.
