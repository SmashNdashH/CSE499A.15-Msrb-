# Phase 2 Plan: closing the Destroyed IoU gap

Drafted 2026-10-05, while R4, R5 and R8 are still running. Nothing here needs those runs to finish except the final champion.

## 1. Why Phase 2 is (almost certainly) needed

The Phase Gate needs **mIoU > 0.52 and Destroyed IoU > 0.30** on the 645 validation images (E-protocol).

| Run | mIoU | Destroyed IoU | With TTA |
|---|---|---|---|
| R1 (reference) | 0.5374 | 0.2686 | 0.5440 / 0.2822 |
| R6 (seed 7) | 0.5402 | 0.2815 | 0.5443 / 0.2839 |
| R2 (copy-paste) | 0.5369 | 0.2681 | 0.5415 / 0.2676 |
| R3 (FT 0.25/0.75/1.33) | 0.5402 | 0.2790 | 0.5444 / **0.2845** |

- mIoU passes in every run. Destroyed IoU misses in every run; the best is 0.2845, so the gap is about **+0.016**.
- Seed noise on Destroyed IoU is |R1 − R6| = **0.013**, nearly as big as the gap. Any run that "passes" by a hair must be confirmed with a second seed (see §6).
- R4, R5 and R8 could still close the gap, but nothing in the sprint moved Destroyed IoU that much except the loss change, which R1 already has. Plan Phase 2 now; cancel it if one of them passes and a second seed confirms.

## 2. What the data actually is (checked 2026-10-05 against the local `train_release.json`)

**Our building-damage set is mostly xBD already.** The D4 notebook's 6,443 images (5,798 train / 645 val, same split rebuilt locally) break down like this:

| Source | Events | Train | Val |
|---|---|---|---|
| **xBD tiles** (file names like `santa_rosa_wildfire_00000091_post_disaster.png`) | 16: portugal-wildfire, socal-fire, nepal-flooding, hurricane-michael, hurricane-florence, hurricane-harvey, pinery-bushfire, woolsey-fire, midwest-flooding, hurricane-matthew, santa-rosa-wildfire, tuscaloosa-tornado, moore-tornado, lower-puna-volcano, palu-tsunami, sunda-tsunami | 5,058 | 560 |
| **Other events** (BRIGHT-style) | 8: turkey-earthquake, morocco-earthquake, libya-flood, beirut-explosion, bata-explosion, hawaii-wildfire, kalehe-flooding, shovi-landslide | 740 | 85 |

Consequences:
- **"Add xBD" (task D5 as written) is mostly already done by DisasterM3.** Merging the local xBD copy naively would duplicate 1,792 tiles and **leak 68 destroyed-rich validation tiles into training**.
- **All 6,443 images have a pre-disaster image** (`pre_image_path`). The D4 notebook ignores it. On xBD, pre+post models are what won the xView2 challenge. That is the strongest untried lever (P2-B).
- **The validation split is random by tile, not grouped by event**, so 560 validation tiles have neighbouring tiles from the same disaster in training. Scores are therefore "seen-disaster" scores. Don't change the split now (it would break comparability with every earlier run); state it as a limitation in the report.

**xBD tier 1 on disk (`data/xView2 Challenge Dataset - train and test/`)**, full pass over all 2,799 post-disaster label files:
- Polygons: no-damage 117,426 · minor 14,980 · major 14,161 · destroyed 13,227 · un-classified 2,993. Destroyed is 0.30% of pixels, the same imbalance as ours (0.33%).
- The 499A EDA sampled only 200 files (2 events) and reported 31 destroyed polygons, a large undercount. Use these full-pass numbers instead.
- **1,792 tiles are already in DisasterM3; 1,007 are not.** The 1,007 new tiles come from 10 events (socal-fire 467, mexico-earthquake 121, palu-tsunami 82, midwest-flooding 65, harvey 61, matthew 56, florence 51, michael 45, santa-rosa 41, guatemala 18). They hold **3,472 destroyed polygons** (26% of tier 1's destroyed buildings), 163 of the new tiles contain at least one, and none of them is in our validation set.
- The local `test/` folder (1,866 images) has no labels.

**AIDER (`data/AIDER`)**
6,433 oblique drone photos labelled per image only (collapsed_building 511, normal 4,390, …). With no masks and a different viewpoint, it **cannot train the segmenter**. At most it's an optional sanity check. Low priority.

## 3. Step P2-0: diagnose first (about 30 GPU-minutes, can start now)

Evaluation only, with R1's `final_model.pth` (in `R1-results/d4_R1/` and on the Hugging Face repo). Add one eval-only cell to the D4 notebook that prints:

1. **The full 4×4 confusion matrix** (pixel counts, rows = truth, columns = prediction) on the 645 validation images.
2. **Building IoU**: classes 1–3 merged versus background.
3. **Destroyed IoU per event and per source** (xBD vs other), so we see which disasters fail.
4. **A converter check:** rasterize the xBD polygons for the 1,792 tier-1 tiles that DisasterM3 also contains and compare them with DisasterM3's own masks. This (a) proves our xBD→mask converter reproduces DisasterM3's labels, and (b) **shows how DisasterM3 mapped minor-damage and un-classified**, so we copy their mapping instead of guessing.

**Decision rule.** Of the destroyed pixels R1 misses, what share does it label Background?
- **≥ 40% → run P2-B first.** Rubble isn't recognised as a building at all, and the pre-disaster image shows the footprint.
- **< 40% → run P2-A first.** The building is found but the damage level is wrong, and more labelled destroyed examples target that.

Both runs are planned either way; P2-0 only sets the order in case GPU time runs out.

## 4. The Phase 2 runs (same rules as R2–R8)

Each run changes **one thing** relative to R1, uses **the same number of training steps** (5,798 samples × 40 epochs), and is scored with the E-protocol on **our 645 validation images only**. A change is kept only if it beats R1 on mIoU by more than 0.0028 **and** does not lower Destroyed IoU.

### P2-B: pre+post input (no new data), the main bet
- **Input:** feed the pre-disaster and post-disaster images together as a 6-channel input: `smp.Unet(..., in_channels=6)`. smp adapts the ImageNet first layer automatically. This is the smallest version of the xView2 winners' pre+post siamese networks, built for exactly this data.
- **Dataset changes:** load `pre_image_path` as well, apply the **same** crop, flip, rotation and brightness change to both images, then stack them. TTA applies each view to both.
- **Check first:** pre and post line up (overlay 10 pairs from each source).
- **Cost:** about 8 GPU-hours (two images per sample, so loading is a little slower).

### P2-A: add the 1,007 xBD tier-1 tiles that DisasterM3 left out (D5, leak-safe)
- **Selection:** a tile is added **only if its id (e.g. `palu-tsunami_00000123`) is not anywhere in DisasterM3**, train or validation. That excludes all 1,792 overlapping tiles, including the 68 in our validation set.
- **Masks:** draw the post-disaster polygons with `cv2.fillPoly`, using **the class mapping measured in P2-0 step 4**. un-classified becomes 255 = ignore, unless P2-0 shows DisasterM3 did something else.
- **Sampling:** draw uniformly over the 5,798 + 1,007 images (about 15% new), but keep 5,798 samples per epoch, so the change is "more data", not "more training".
- **Code changes:** the loss must skip pixels labelled 255 (zero their weight in Focal Tversky; `ignore=255` in Lovász).
- **Cost:** about 7 GPU-hours. Conversion is CPU-only: do it on the laptop now and upload the result as a Kaggle dataset.

### P2-A2: the rest of xBD (only if P2-A helps)
xBD's tier-3, test and hold splits contain roughly 4,400 more tiles that DisasterM3 doesn't use, including **joplin-tornado**, a destroyed-heavy event missing from our data entirely. This needs a large download from xview2.org (free registration). Apply the same tile-id exclusion rule.

### P2-AB: combined champion
If P2-A and P2-B each pass the keep rule, combine them, add any R2–R8 change that also passed, and use TTA. This replaces R7 as the champion candidate. Cost: about 9 GPU-hours.

### P2-C: two-stage model (fallback only)
Stage 1 segments buildings versus background; stage 2 classifies damage inside each building from pre+post crops. It is the strongest option in the xBD literature, but it means 3–5 days of new code and error propagation (a building stage 1 misses is lost). Do it **only if P2-A and P2-B both fail**. Otherwise it goes in the report as future work.

### Not doing
CrowdAI / SpaceNet (footprint-only datasets, useful only for P2-C stage 1); AIDER for training (§2).

## 5. GPU budget and timeline

| Step | GPU-hours | Can start | Suggested owner |
|---|---|---|---|
| P2-0 diagnosis and converter check | 0.5 | now | Abrar |
| xBD tier-1 → masks for the 1,007 new tiles, uploaded to Kaggle | 0 (CPU) | now | Ridita (D5 was Member D's task) |
| P2-B | ~8 | first free GPU after R4, R5 or R8 | Abrar |
| P2-A | ~7 | after the conversion and P2-0 step 4 | Ridita |
| P2-AB champion | ~9 | after A and B | Abrar |
| Second seed of the champion (§6) | ~9 | after P2-AB | whichever account has quota |
| **Total** | **~34** | | about 1.2 account-weeks across the group |

If R4, R5 and R8 finish by mid-October, the Phase 2 decision and champion can land by **early November**, ahead of the "mid-Nov to early Dec" slot in the current timeline.

## 6. When we can claim the Phase Gate

Destroyed IoU moves by about 0.013 from the seed alone. A champion "passes" only if:
- Destroyed IoU > 0.30 under the E-protocol (report standard and TTA, and say which one the claim uses), **and**
- a second seed of the same config also stays above 0.30 (or the two-seed mean does, stated as a mean).

If it passes with TTA only, say so plainly ("passes with 8-view TTA").

## 7. Decisions needed from the lead

1. **Who converts xBD.** Suggested: Ridita, as the original D5 owner, since it is CPU work and doesn't need her R4 GPU.
2. **Whether to download the rest of xBD (P2-A2)** if P2-A helps. It's the only source of a new destroyed-heavy event (joplin-tornado).
3. **Whether to accept P2-C** if A and B both fail, or report the gate as missed with an honest error analysis. A missed gate with a clear diagnosis is still a defensible result.

The minor-damage mapping question from `team_task_assignments.md` is no longer a decision: P2-0 step 4 measures how DisasterM3 mapped it, and we copy that.
