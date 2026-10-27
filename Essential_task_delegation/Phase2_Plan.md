# Phase 2 Plan: closing the Destroyed IoU gap

Drafted 2026-10-05, while R4, R5 and R8 are still running. Nothing here needs those runs to finish except the final champion.

> **Names updated 2026-10-18** to the task IDs used in [Next_Steps_Roadmap.md](Next_Steps_Roadmap.md):
> - E1 was P2-0;
> - R11 was P2-B;
> - R12 was P2-A;
> - R15 was P2-AB;
> - R16 was P2-C.
>
> The converter check (old P2-0 step 4) is now its own task, D5.
>
> **Names updated 2026-10-20:** R6 is now R1s7 (R1 with seed 7). R16's two stages are now "the finding" and "the grading", and the model is "find-then-grade"; "step" is kept for a task's parts.

> **Update 2026-10-08:** R4, R5 and R8 have finished: R4 (U-Net++) 0.5267 / 0.2551, R5 (sampler) 0.5354 / **0.2898** (0.2917 with TTA), R8 0.5342 / 0.2703. None passed the R7 rule, so R7 is the R1 recipe. The best Destroyed IoU is 0.2917 (R5 with TTA), so the gap to 0.30 is now about **+0.008**. Two combination runs are in progress, R9 (sampler + copy-paste) and R10 (sampler + R3 loss). Phase 2 is still needed unless one of them passes and a second seed confirms it.

> **Update 2026-10-18 (fine-tuning closed, Phase 2 is needed):**
> - R9 (sampler + copy-paste): 0.5373 / 0.2921, or 0.5440 / **0.3001** with TTA.
> - R10 (sampler + R3 loss): 0.5345 / 0.2891, or 0.5406 / 0.2969 with TTA. No gain over R5.
> - R9s7 (R9 with seed 7, the §6 check): 0.5382 / 0.2847, or 0.5453 / **0.2941** with TTA.
>
> R9's two-seed mean is 0.2971 with TTA, so by §6 the Phase Gate is **not** claimed. R9 is still the best recipe for destroyed buildings, about +0.014 over the R1 recipe's three-run mean, which is right at the noise level. Start with E1. Base R11 on R5 (sampler only), not R9, because copy-paste would put rubble into the after image where the before image shows nothing.

> **Update 2026-10-26:**
> - R16c (find-then-grade: SAM 3 and R11's pieces, then a stronger grader) passed the gate on all 645 validation images, on two U-Net seeds (20–21 Oct).
> - E5 then found validation pairs that are near-copies of training pairs, Without the 40 leaked pairs, R16c steps 2 and 3 score 0.2838–0.2980, so their claim doesn't hold on clean data. Only R16c step 5's SAM 3-only map passes there (0.3161 and 0.3051), narrowly (26 Oct). The leak also helped the before + after U-Nets (R11) twice as much as the after-only ones.
> - The current status is in [Progress_Tracker.md](Progress_Tracker.md) (Goals).

## 1. Why Phase 2 is (almost certainly) needed

The Phase Gate needs **mIoU > 0.52 and Destroyed IoU > 0.30** on the 645 validation images (E-protocol).

| Run | mIoU | Destroyed IoU | With TTA |
|---|---|---|---|
| R1 (reference) | 0.5374 | 0.2686 | 0.5440 / 0.2822 |
| R1s7 (seed 7; called R6 until 20 Oct) | 0.5402 | 0.2815 | 0.5443 / 0.2839 |
| R2 (copy-paste) | 0.5369 | 0.2681 | 0.5415 / 0.2676 |
| R3 (FT 0.25/0.75/1.33) | 0.5402 | 0.2790 | 0.5444 / **0.2845** |

- mIoU passes in every run. Destroyed IoU misses in every run; the best is 0.2845, so the gap is about **+0.016**.
- Seed noise on Destroyed IoU is |R1 − R1s7| = **0.013**, nearly as big as the gap. Any run that "passes" by a hair must be confirmed with a second seed (see §6).
- R4, R5 and R8 could still close the gap, but nothing in the sprint moved Destroyed IoU that much except the loss change, which R1 already has. Plan Phase 2 now; cancel it if one of them passes and a second seed confirms.

## 2. What the data actually is (checked 2026-10-05 against the local `train_release.json`)

**Our building-damage set is mostly xBD already.** The D4 notebook's 6,443 images (5,798 train / 645 val, same split rebuilt locally) break down like this:

| Source | Events | Train | Val |
|---|---|---|---|
| **xBD tiles** (file names like `santa_rosa_wildfire_00000091_post_disaster.png`) | 16: portugal-wildfire, socal-fire, nepal-flooding, hurricane-michael, hurricane-florence, hurricane-harvey, pinery-bushfire, woolsey-fire, midwest-flooding, hurricane-matthew, santa-rosa-wildfire, tuscaloosa-tornado, moore-tornado, lower-puna-volcano, palu-tsunami, sunda-tsunami | 5,058 | 560 |
| **Other events** (BRIGHT-style) | 8: turkey-earthquake, morocco-earthquake, libya-flood, beirut-explosion, bata-explosion, hawaii-wildfire, kalehe-flooding, shovi-landslide | 740 | 85 |

Consequences:
- **"Add xBD" (task D5 as written) is mostly already done by DisasterM3.** Merging the local xBD copy naively would duplicate 1,792 tiles and **leak 68 destroyed-rich validation tiles into training**.
- **All 6,443 images have a pre-disaster image** (`pre_image_path`). The D4 notebook ignores it. On xBD, pre+post models are what won the xView2 challenge. That is the strongest untried lever (R11).
- **The validation split is random by tile, not grouped by event**, so 560 validation tiles have neighbouring tiles from the same disaster in training. Scores are therefore "seen-disaster" scores. Don't change the split now (it would break comparability with every earlier run); state it as a limitation in the report.

**xBD tier 1 on disk (`data/xView2 Challenge Dataset - train and test/`)**, full pass over all 2,799 post-disaster label files:
- Polygons: no-damage 117,426 · minor 14,980 · major 14,161 · destroyed 13,227 · un-classified 2,993. Destroyed is 0.30% of pixels, the same imbalance as ours (0.33%).
- The 499A EDA sampled only 200 files (2 events) and reported 31 destroyed polygons, a large undercount. Use these full-pass numbers instead.
- **1,792 tiles are already in DisasterM3; 1,007 are not.** The 1,007 new tiles come from 10 events (socal-fire 467, mexico-earthquake 121, palu-tsunami 82, midwest-flooding 65, harvey 61, matthew 56, florence 51, michael 45, santa-rosa 41, guatemala 18). They hold **3,472 destroyed polygons** (26% of tier 1's destroyed buildings), 163 of the new tiles contain at least one, and none of them is in our validation set.
- The local `test/` folder (1,866 images) has no labels.

**AIDER (`data/AIDER`)**
6,433 oblique drone photos labelled per image only (collapsed_building 511, normal 4,390, …). With no masks and a different viewpoint, it **cannot train the segmenter**. At most it's an optional sanity check. Low priority.

## 3. Step E1: diagnose first (about 30 GPU-minutes, can start now)

Evaluation only, with R1's `final_model.pth` (in `R1-results/d4_R1/` and on the Hugging Face repo). Add one eval-only cell to the D4 notebook that prints:

1. **The full 4×4 confusion matrix** (pixel counts, rows = truth, columns = prediction) on the 645 validation images.
2. **Building IoU**: classes 1–3 merged versus background.
3. **Destroyed IoU per event and per source** (xBD vs other), so we see which disasters fail.
4. **A converter check (now task D5):** rasterize the xBD polygons for the 1,792 tier-1 tiles that DisasterM3 also contains and compare them with DisasterM3's own masks. This (a) proves our xBD→mask converter reproduces DisasterM3's labels, and (b) **shows how DisasterM3 mapped minor-damage and un-classified**, so we copy their mapping instead of guessing.

**Decision rule.** Of the destroyed pixels R1 misses, what share does it label Background?
- **≥ 40% → run R11 first.** Rubble isn't recognised as a building at all, and the pre-disaster image shows the footprint.
- **< 40% → run R12 first.** The building is found but the damage level is wrong, and more labelled destroyed examples target that.

Both runs are planned either way; E1 only sets the order in case GPU time runs out.

## 4. The Phase 2 runs (same rules as R2–R8)

Each run changes **one thing** relative to R1, uses **the same number of training steps** (5,798 samples × 40 epochs), and is scored with the E-protocol on **our 645 validation images only**. A change is kept only if it beats R1 on mIoU by more than 0.0028 **and** does not lower Destroyed IoU.

### R11: pre+post input (no new data), the main bet
- **Input:** feed the pre-disaster and post-disaster images together as a 6-channel input: `smp.Unet(..., in_channels=6)`. smp adapts the ImageNet first layer automatically. This is the smallest version of the xView2 winners' pre+post siamese networks, built for exactly this data.
- **Dataset changes:** load `pre_image_path` as well, apply the **same** crop, flip, rotation and brightness change to both images, then stack them. TTA applies each view to both.
- **Check first:** pre and post line up (overlay 10 pairs from each source).
- **Cost:** about 8 GPU-hours (two images per sample, so loading is a little slower).

### R12: add the 1,007 xBD tier-1 tiles that DisasterM3 left out (data from D5, leak-safe)
- **Selection:** a tile is added **only if its id (e.g. `palu-tsunami_00000123`) is not anywhere in DisasterM3**, train or validation. That excludes all 1,792 overlapping tiles, including the 68 in our validation set.
- **Masks:** draw the post-disaster polygons with `cv2.fillPoly`, using **the class mapping measured by the D5 converter check**. un-classified becomes 255 = ignore, unless D5 shows DisasterM3 did something else.
- **Sampling:** draw uniformly over the 5,798 + 1,007 images (about 15% new), but keep 5,798 samples per epoch, so the change is "more data", not "more training".
- **Code changes:** the loss must skip pixels labelled 255 (zero their weight in Focal Tversky; `ignore=255` in Lovász).
- **Cost:** about 7 GPU-hours. Conversion is CPU-only: do it on the laptop now and upload the result as a Kaggle dataset.

### P2-A2: the rest of xBD (only if R12 helps)
xBD's tier-3, test and hold splits contain roughly 4,400 more tiles that DisasterM3 doesn't use, including **joplin-tornado**, a destroyed-heavy event missing from our data entirely. This needs a large download from xview2.org (free registration). Apply the same tile-id exclusion rule.

### R15: combined champion
If R12 and R11 each pass the keep rule, combine them, add any R2–R8 change that also passed, and use TTA. This replaces R7 as the champion candidate. Cost: about 9 GPU-hours.

### R16: find-then-grade model (fallback only)
The finding segments buildings versus background; the grading classifies damage inside each building from pre+post crops. It is the strongest option in the xBD literature (called "two-stage" there), but it means 3–5 days of new code and error propagation (a building the finding misses is lost). Do it **only if R12 and R11 both fail**. Otherwise it goes in the report as future work.

### Not doing
CrowdAI / SpaceNet (footprint-only datasets, useful only for R16's finding); AIDER for training (§2).

## 5. GPU budget and timeline

| Step | GPU-hours | Can start | Suggested owner |
|---|---|---|---|
| E1 diagnosis | 0.5 | now | Abrar |
| D5: converter check, then xBD tier-1 → masks for the 1,007 new tiles, uploaded to Kaggle | 0 (CPU) | now | Ridita (D5 was Member D's task) |
| R11 | ~8 | first free GPU after R4, R5 or R8 | Abrar |
| R12 | ~7 | after D5 | Ridita |
| R15 champion | ~9 | after R11 and R12 | Abrar |
| Second seed of the champion (R15s7, §6) | ~9 | after R15 | whichever account has quota |
| **Total** | **~34** | | about 1.2 account-weeks across the group |

If R4, R5 and R8 finish by mid-October, the Phase 2 decision and champion can land by **early November**, ahead of the "mid-Nov to early Dec" slot in the current timeline.

## 6. When we can claim the Phase Gate

Destroyed IoU moves by about 0.013 from the seed alone. A champion "passes" only if:
- Destroyed IoU > 0.30 under the E-protocol (report standard and TTA, and say which one the claim uses), **and**
- a second seed of the same config also stays above 0.30 (or the two-seed mean does, stated as a mean).

If it passes with TTA only, say so plainly ("passes with 8-view TTA").

## 7. Decisions needed from the lead

1. **Who converts xBD.** Suggested: Ridita, as the original D5 owner, since it is CPU work and doesn't need her R4 GPU.
2. **Whether to download the rest of xBD (P2-A2)** if R12 helps. It's the only source of a new destroyed-heavy event (joplin-tornado).
3. **Whether to accept R16** if R12 and R11 both fail, or report the gate as missed with an honest error analysis. A missed gate with a clear diagnosis is still a defensible result.

The minor-damage mapping question from `team_task_assignments.md` is no longer a decision: the D5 converter check measures how DisasterM3 mapped it, and we copy that.
