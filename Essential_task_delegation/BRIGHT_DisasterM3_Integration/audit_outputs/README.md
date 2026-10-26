# DisasterM3 vs BRIGHT optical audit: scripts, outputs and the validation leak

Everything behind [`DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf`](../DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf), collected on 26 Oct 2026 for E5. None of it was in the repo: the outputs were only in a local git stash, and the scripts were in that stash and the Recycle Bin ([where the copies came from](#where-the-copies-came-from)). Every file here is a byte-identical copy of an original, except three new things: this README, [`scripts/verify_val_leak.py`](scripts/verify_val_leak.py), and [`outputs/verification/`](outputs/verification/).

## For E5: the leak in one table

| Frozen validation set (Cell 5 of `d4-champion-model-R11.ipynb`) | Pairs | Destroyed pixels |
|---|---:|---:|
| All | 645 | 1,808,111 |
| Leaked: twin in train, [`outputs/val_train_leaks.csv`](outputs/val_train_leaks.csv) | 40 (6.2%) | 155,812 (8.6%) |
| Clean | 605 | 1,652,299 |

- **To re-score:** drop the 40 `val_post` names in `val_train_leaks.csv`, or keep the rows of [`outputs/verification/val_pairs_645.csv`](outputs/verification/val_pairs_645.csv) where `clean` is `True`. That file has one row per validation pair, in Cell 5's order, with its destroyed pixels.
- **E5's 21 pairs are all among the 40**, with the same twins and the same destroyed pixels. E5 missed the other 19 because their socal_fire side is not an xBD tier-1 image ([section 4](#4-comparison-with-e5-abrars-21-pairs)).
- **Not leaks:** 4 validation pairs are twins of each other (two pairs of pairs), and 1 has its twin outside the pool.

## What is here

### `scripts/`

| File | What it does | What it writes | Copied from |
|---|---|---|---|
| `dm3_bright_optical_eda.py` | The main audit, the source of the report's numbers. It covers the inventory, SAR found by pixels, resolution, sharpness, brightness, pre/post shift, labels, and the overlap and validation leak. It imports `bright_dm3_dedup.py`. | The 9 files from `summary.json` to `val_train_leaks.csv` | `stash@{0}^3` |
| `bright_dm3_dedup.py` | Rebuilds the pool and split exactly as the notebooks do, then checks every standardized BRIGHT tile against every DisasterM3 file, by name and by pixels. **It rewrites the manifest it reads.** | `bright_dm3_dedup_report.json`, plus a rewritten `bright_optical_manifest.json` | `stash@{0}^3` |
| `dm3_bright_optical_eda_ext.py` | Second pass: label offsets, ResNet-50 feature gap, re-tiled overlap (SIFT), spot checks. Reads the first pass's CSVs. Needs `torch`, `torchvision`, `scipy`. | `outputs/ext/*` | `stash@{0}^3` |
| `bright_disasterm3_standardizer.py` | Turns BRIGHT pre tiles and label maps into 1024×1024 PNGs and footprint masks: 1,285 tiles of the 5 clean events, plus 621 Ukraine / Acapulco / Kyaukpyu tiles taken from DisasterM3. This is the last version, which was never committed. | `data/bright_standardized_optical/`: 1,906 tiles and the manifest | `stash@{0}` |
| `older_versions/bright_disasterm3_standardizer_7a8043121.py` | The committed version of 14 Oct (5 events, 1,285 tiles), the file deleted in `922dbd2b4` | The older 1,285-tile folder (= `data/bright_standardized_optical.zip`) | `922dbd2b4^` |
| `dm3_bright_audit.py` | An earlier, generic audit. It finds SAR by file name, which the report warns against (§2, recommendation 5). The corrected report does not use it. | `audit_out/`: not found on this PC | `stash@{0}^3` |
| `dm3_vs_bright_compare.py` | An earlier comparison built on `dm3_bright_audit.py`, with the same name-based SAR filter. The corrected report does not use it. | `cmp_out/`: not found on this PC | `stash@{0}^3` |
| `build_easy_pdf.py` | Draws the report's 5 charts (4 from `summary.json`; the disaster-type chart uses numbers written into the script), puts them into `audit_easy_body.html`, and prints the PDF with headless Edge | `audit_easy.html` and the corrected PDF | A temporary folder on C: (22 Oct 16:59) |
| `verify_val_leak.py` | **New (26 Oct):** the second leak check described [below](#the-leak-method-and-numbers) | `outputs/verification/*` | new |

### `outputs/`

| File | Rows | One row is | Made by | Last written |
|---|---:|---|---|---|
| `summary.json` | 7 sections | Every number in the report (inventory, events, overlap, geometry, radiometry, registration, labels) | `dm3_bright_optical_eda.py` | 21 Oct 01:23 |
| `dm3_images.csv` | 20,988 | One DisasterM3 Instruct image: name, event, role, JSON tag, colour or SAR by pixels, size, `md5` of the decoded pixels, brightness, split | same | same |
| `bright_pre_tiles.csv` | 3,395 | One BRIGHT pre tile: resolution from GeoTIFF tags, brightness, `md5` | same | same |
| `bright_targets.csv` | 4,246 | One BRIGHT label map: pixels per class, buildings, resolution | same | same |
| `dm3_pool_labels.csv` | 6,443 | One pool pair: split, pixels per class, building counts, road pixels | same | same |
| `dm3_optical_pair_registration.csv` | 7,159 | One optical pair: pre-to-post shift, tiles that agree, reliable or not | same | same |
| `dm3_pool_pairs_radiometry.csv` | 6,443 | One pool pair: pre and post brightness, saturation, JS divergence | same | same |
| `dm3_events.csv` | 45 | One DisasterM3 event name: hazard, origin, image counts, pool train/val, Bench posts | same | same |
| `val_train_leaks.csv` | 40 | One leaked validation pair: its twin in train, `post_r`, `building_iou`, `same_damage_class`, `val_destroyed_px` | same | same |
| `bright_dm3_dedup_report.json` | 1,906 tiles | One BRIGHT tile's matches in DisasterM3 and the keep/exclude decision; also the pool, the 645 validation names and their SHA-1 | `bright_dm3_dedup.py` | 20 Oct 20:35 |
| `bright_optical_manifest.json` | 1,906 tiles | One standardized BRIGHT tile (the dedup's input, as the dedup rewrote it) | standardizer, then dedup | 20 Oct 20:35 |
| `ext/summary_ext.json` | 4 sections | The second pass's numbers | `dm3_bright_optical_eda_ext.py` | 21 Oct 10:18 |
| `ext/label_registration.csv` | 6,443 | One pool pair: which image its masks are drawn on, offsets, IoU cost | same | same |
| `ext/bright_label_registration.csv` | 3,395 | One BRIGHT tile: offset of its label map (the control) | same | same |
| `ext/feature_gap_tsne.csv` | 4,181 | One image: 2-D t-SNE position of its ResNet-50 features | same | same |
| `ext/feature_gap_by_bright_event.csv` | 11 | One BRIGHT event: distance from the pool, nearest pool event | same | same |
| `ext/retiled_matches.csv` | 54 | One SIFT match between a DisasterM3 pre image and a BRIGHT tile | same | same |
| `report_source/audit_easy_body.html` | 455 lines | The report's text, with `{{CHART_*}}` placeholders | — (input of `build_easy_pdf.py`) | 22 Oct 16:59 |
| `report_source/audit_easy.html` | 455 lines | The page printed to the PDF; its 5 figures are inline SVG | `build_easy_pdf.py` | 22 Oct 16:59 |
| `verification/verify_val_leak.json` | 5 sections | Every number in [the leak section](#the-leak-method-and-numbers) | `verify_val_leak.py` | 26 Oct |
| `verification/val_pairs_645.csv` | 645 | One validation pair in Cell 5's order: destroyed pixels, leak, twin, `clean` | same | 26 Oct |
| `verification/leaks_40_vs_abrar_21.csv` | 40 | One leaked pair: in E5 or not, where xBD keeps its socal_fire side, reason | same | 26 Oct |
| `verification/socal_woolsey_pre_twins.csv` | 264 | One identical before-image pair: where each side sits, file-identical or not, xBD source | same | 26 Oct |

All times are local (UTC+6). They are when each script last finished, from the session logs; the stash does not keep file times. CSVs and JSON have LF line endings, as git stores them; on disk they had CRLF.

## How to make every number again

### Inputs

| Input | Needed by | Where to get it | On Ridita's PC |
|---|---|---|---|
| DisasterM3 Instruct: `train_release.json`, `train_images/` (20,988 PNG), `masks/`, `box_train_images/` | all scripts except `build_easy_pdf.py` | Hugging Face `Kingdrone-Junjue/DisasterM3` (local copy: revision `cd944ae8`, 13 Oct), or the Kaggle mirror `abrarmohammedtanzim/disasterm3-mirror` | `D:\CSE499A.15-Msrb-\dataset\DisasterM3 Dataset\DisasterM3_Instruct\` |
| DisasterM3 Bench: `benchmark_release.json` and `masks/` (no Bench images are used) | EDA (inventory), dedup (Bench masks) | the same Hugging Face repo | `...\DisasterM3 Dataset\DisasterM3_Bench\` |
| BRIGHT: `pre-event/pre-event_wo_ukraine_myanmar_mexico/` (3,395 GeoTIFF), `post-event/` (4,246, all SAR), `target/target/` (4,246 label maps) | EDA, second pass, standardizer | Zenodo doi:10.5281/zenodo.14619797 ([github.com/ChenHongruixuan/BRIGHT](https://github.com/ChenHongruixuan/BRIGHT)) | `D:\CSE499A.15-Msrb-\dataset\BRIGHT Dataset\` |
| Standardized BRIGHT tiles: `data/bright_standardized_optical/` (1,906 PNG, masks, manifest) | dedup | Run `bright_disasterm3_standardizer.py`, or restore them: `git checkout stash@{0}^3 -- data/bright_standardized_optical` | Only in the stash. `data/bright_standardized_optical.zip` (17 Oct) is the older 1,285-tile version. |
| xView2 `train/images` and `test/images` (xBD tier 1) | `verify_val_leak.py`, for the E5 part only | [xview2.org](https://xview2.org/) | `D:\Downloads\xView2 Challenge Dataset - train and test\` |
| `Champion_Task/R16c-results/step2/instances_val.csv`, `Champion_Task/E5-results/E5/leaked_pairs_local_check.csv` | `verify_val_leak.py` | this repo | |

Python 3.12 with `numpy`, `opencv-python`, `pandas`, `Pillow`, `scikit-learn` and `tqdm`. The second pass also needs `scipy`, `torch` and `torchvision`; `build_easy_pdf.py` needs Microsoft Edge.

### Commands

Run them from `audit_outputs/scripts/`; each script imports its neighbours. Without arguments the original scripts use Ridita's paths (`d:\CSE499A.15-Msrb-\...`), so on another machine pass your own. Point `--out` and `--report` at a new folder (`<work>` below) so that the copies in `outputs/` stay as they are.

```bash
cd Essential_task_delegation/BRIGHT_DisasterM3_Integration/audit_outputs/scripts
DM3="D:/CSE499A.15-Msrb-/dataset/DisasterM3 Dataset"
BRIGHT="D:/CSE499A.15-Msrb-/dataset/BRIGHT Dataset"

# 1. The main audit: every number in the report and val_train_leaks.csv.
#    30-40 min with 4 workers; under 1 min once the cache exists.
python dm3_bright_optical_eda.py --dm3_root "$DM3" --bright_root "$BRIGHT" --out <work>/eda --workers 4 --cache <work>/pixels.pkl

# 2. The BRIGHT tiles the dedup reads (skip this if you restored them from the stash)
python bright_disasterm3_standardizer.py --bright_root "$BRIGHT" --dm3_root "$DM3" --output_dir <work>/bright_standardized_optical

# 3. The dedup check, 5-15 min. It rewrites the manifest you give it.
python bright_dm3_dedup.py --bright_manifest <work>/bright_standardized_optical/bright_optical_manifest.json --dm3_root "$DM3" --report <work>/bright_dm3_dedup_report.json

# 4. The second pass, about 45 min
python dm3_bright_optical_eda_ext.py --dm3_root "$DM3" --bright_root "$BRIGHT" --eda <work>/eda --cache <work>/ext.pkl

# 5. The second leak check, 7 min with 2 workers. Leave out the two xBD folders on Kaggle.
python -B verify_val_leak.py --xbd_train_images "D:/Downloads/xView2 Challenge Dataset - train and test/train/train/images" --xbd_test_images "D:/Downloads/xView2 Challenge Dataset - train and test/test/test/images" --out <work>/verification
```

The run times were measured on Ridita's laptop (8 cores, 8 GB RAM). The report's "about 35 minutes" for steps 1 and 3 is a little short: they take 35–55 min together.

`build_easy_pdf.py` has fixed paths. It reads `D:\CSE499A.15-Msrb-\data\dm3_bright_optical_eda\summary.json`, reads `audit_easy_body.html` from its own folder, and **overwrites `../DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf`**. To rebuild the page:

1. Copy the script and `outputs/report_source/audit_easy_body.html` into an empty folder.
2. In the copy, set `EDA` to this `outputs/` folder and `OUT_PDF` to a new file name.
3. Run the copy.

## The leak: method and numbers

### 1. The 645 validation pairs are the notebook's

**The same pool rule.** The audit's pairs come from `build_optical_pool()` in `bright_dm3_dedup.py`. It applies the same rule as Cells 4–5 of [`d4-champion-model-R11.ipynb`](../../Champion_Task/d4-champion-model-R11.ipynb):

| Step | Notebook, Cells 4–5 | `build_optical_pool()` |
|---|---|---|
| Entries kept | `task == "Referring Expression Segmentation"`, `cls_description == "Building Damage Assessment"`, `image_type == "Optical"` | the same three tests |
| Mask must exist | `resolve_path(ground_truth)`, 5 candidate paths | `resolve()`, the same 5 candidates in the same order |
| Grouping and order | `by_image[post_image_path]`: dict insertion order, which is manifest order | the same |
| Before-image | the last entry's `pre_image_path` | the same |
| Pair kept if | the post image resolves and one of the 3 building masks can be read | the post image resolves and one of the 3 building-mask folders is present |
| Split | `train_test_split(pairs, test_size=0.1, random_state=42)` | the same call |

The only difference is the last check: a mask that exists but cannot be read. That never happens here: both rules give the same 6,443 pairs.

**The same order and the same split.** `verify_val_leak.py` runs the notebook's Cell 4–5 code (copied, with the Kaggle paths swapped for local ones) next to `build_optical_pool()`. The results:

- Cell 4's counts equal what the notebook printed on Kaggle: 92,968 entries, 14,531 building-damage entries, 6,443 images with a mask, 6,443 pairs, 0 skipped, 0 missing before-images.
- Both rules give the same 6,443 pairs in the same order, the same before-images, and the same 5,798 / 645 split, in the same order.
- The 645 match the dedup report's list (SHA-1 of the sorted names: `116d518d69c395c6f139de56ff8b789b096193f7`) and the `split` column of `dm3_pool_labels.csv`.
- R16c step 2's `instances_val.csv` was made on Kaggle from the notebook's split. Its 631 images all fall inside these 645. The other 14 have no row there, because no building pieces were found in them.

**Totals.** The 645 hold 1,808,111 destroyed pixels, counted from Cell 4's merged masks (all 1024×1024). The count equals `dm3_pool_labels.csv` for each of the 6,443 pairs. The two fire events: socal_fire has 54 validation pairs (113,661 destroyed pixels) and woolsey_fire 32 (111,020), the same as Abrar's split.

### 2. How the twins were found

**What was hashed:** decoded pixels, not file bytes. The files are DisasterM3's own copies in `DisasterM3_Instruct/train_images/train_images/`. Each image is read with `cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)` into a 1024×1024×3 uint8 BGR array, and the hash is `md5(str(shape).encode() + array.tobytes())`. The code is in `dm3_bright_optical_eda.image_stats()`; the result is the `md5` column of `dm3_images.csv`. Re-hashing 13,518 of those images here gives the same MD5 for every one.

**The rule** (`dm3_bright_optical_eda.py`, sections 5 and 9):

1. Compare every validation image, before and after (1,290 images), with every training image (11,596) by the correlation of their 32×32 grey thumbnails.
2. Where that correlation is at least 0.95, accept the match if the pixel MD5s are equal, or if `bright_dm3_dedup.copy_score()` (correlation at 256×256 over non-black pixels) is also at least 0.95.
3. A validation pair leaks when either of its two images matches a training image. Its twin is the first match.

**Result, re-run here:**

- The thumbnail step gives 110 candidates; 70 are rejected and 40 confirmed.
- All 40 are before-image matches with identical pixels. No after-image matches.
- The 40 pairs, their twins and all four statistics equal `val_train_leaks.csv`.
- The twins are also byte-identical files (all 264 pairs), so hashing file bytes would give the same answer.

**`val_train_leaks.csv`:**

- **Columns:** `val_post`, `train_post`, `post_r` (correlation of the two after-images), `building_iou`, `same_damage_class` (share of shared building pixels with the same class), `val_destroyed_px`.
- **Rows:** 40. In 16, the validation pair is socal_fire and its twin in train is woolsey_fire; in 24, the other way round. 7 of the 40 have no destroyed pixel.
- **Destroyed pixels:** 155,812 of 1,808,111 (8.6%).
- **Medians:** `post_r` 0.878, `building_iou` 0.72, `same_damage_class` 0.967.

**The 264.** That is the number of identical before-image pairs between socal_fire and woolsey_fire across all of DisasterM3 Instruct, that is, all 1,270 fire pairs on disk (`summary.json`, `overlap.dm3_duplicate_prefix_pairs`). Each group is exactly 2 files, and no after-image is shared. Inside the pool, 250 of these pairs have both images. The other 14 have at least one side among the 316 fire pairs that have no building-damage entry. So §7 of the report ("264 identical pre photos inside the pool") should say "264 in DisasterM3, 250 inside the pool".

| socal_fire side | woolsey_fire side | Pairs |
|---|---|---:|
| train | train | 208 |
| validation | train | 16 (leak) |
| train | validation | 24 (leak) |
| validation | validation | 2 (4 validation pairs, not a leak) |
| outside the pool | validation | 1 |
| train / outside the pool | outside the pool / train | 2 + 8 |
| outside the pool | outside the pool | 3 |

### 3. The 86 validation pairs from the two fire events

| Before-image twin | Pairs | Destroyed pixels |
|---|---:|---:|
| In train: **leak** | 40 | 155,812 |
| Also in validation: not a leak (`socal_fire_00000620` with `woolsey_fire_00000754`, and `socal_fire_00000824` with `woolsey_fire_00000265`) | 4 | 14,982 |
| Outside the pool: not a leak for the segmentation runs (`woolsey_fire_00000485` with `socal_fire_00000324`, which has no building-damage entry) | 1 | 581 |
| No twin | 41 | 53,306 |

No validation pair from the other events has an identical image, before or after, anywhere in the pool. `woolsey_fire_00000485` is also the one fire image missing from R16c's `instances_val.csv`.

### 4. Comparison with E5 (Abrar's 21 pairs)

- **All 21 E5 pairs are in the 40**, with the same twin in train and the same destroyed pixels (80,709).
- **The other 19 (75,103 destroyed pixels) were missed because their socal_fire side is not an xBD tier-1 image.** E5 hashed the tier-1 socal-fire and tier-3 woolsey-fire before-images. A twin whose socal side comes from xBD's test or hold split had nothing to match. Of the 19 socal sides, 10 are xView2 test images (`test_pre_*`, matched by pixels) and 9 are in neither the xView2 train nor the test set on this PC, most likely xBD's hold set.
- **DisasterM3 did not change the images.** All 381 DisasterM3 socal_fire before-images that exist in tier 1 are pixel-identical to xBD's originals, and every twin pair is a byte-identical file pair in DisasterM3. The woolsey side could not be checked, because tier 3 is not on this PC.
- **The 264 twin pairs**, by where xBD keeps the socal side: 156 in tier 1, 57 in the xView2 test set, 51 in neither. 156 is exactly E5's "156 of them have both sides in DisasterM3".
- **Re-enacting E5 on DisasterM3's files** (a twin counts only when its socal side is in tier 1) gives, for E5's 85 validation fire images: 21 with the twin in train, 4 in validation, 60 with no twin. E5 reported 21, 4, and 1 + 59.
- **Side finding:** of DisasterM3's 928 socal_fire before-images, 381 are tier-1 (xView2 train) images, 287 are xView2 test images and 260 are neither. So DisasterM3 Instruct contains xBD test-split images, which matters if anything trained on it is ever scored on xView2's test set.

| Validation pair (`*_post_disaster.png`) | Twin in train | Destroyed px | Where xBD keeps the socal_fire side |
|---|---|---:|---|
| socal_fire_00000188 | woolsey_fire_00000739 | 3,455 | neither |
| socal_fire_00000218 | woolsey_fire_00000517 | 0 | neither |
| socal_fire_00000235 | woolsey_fire_00000253 | 12,736 | xView2 test (`test_pre_00122`) |
| socal_fire_00000396 | woolsey_fire_00000869 | 264 | xView2 test (`test_pre_00057`) |
| socal_fire_00000504 | woolsey_fire_00000695 | 2,688 | xView2 test (`test_pre_00880`) |
| socal_fire_00000614 | woolsey_fire_00000114 | 0 | xView2 test (`test_pre_00304`) |
| socal_fire_00001188 | woolsey_fire_00000809 | 2,078 | xView2 test (`test_pre_00605`) |
| woolsey_fire_00000137 | socal_fire_00000165 | 17,425 | neither |
| woolsey_fire_00000272 | socal_fire_00000315 | 79 | neither |
| woolsey_fire_00000399 | socal_fire_00000870 | 603 | neither |
| woolsey_fire_00000435 | socal_fire_00000382 | 576 | neither |
| woolsey_fire_00000498 | socal_fire_00000379 | 2,532 | xView2 test (`test_pre_00156`) |
| woolsey_fire_00000514 | socal_fire_00000818 | 0 | neither |
| woolsey_fire_00000576 | socal_fire_00000463 | 5,727 | neither |
| woolsey_fire_00000589 | socal_fire_00000426 | 345 | xView2 test (`test_pre_00790`) |
| woolsey_fire_00000653 | socal_fire_00000512 | 15,113 | neither |
| woolsey_fire_00000679 | socal_fire_00000889 | 5,145 | xView2 test (`test_pre_00586`) |
| woolsey_fire_00000722 | socal_fire_00000898 | 5,226 | xView2 test (`test_pre_00694`) |
| woolsey_fire_00000758 | socal_fire_00000207 | 1,111 | xView2 test (`test_pre_00244`) |

To make E5's own hash check complete, add xBD's test and hold socal-fire before-images, or hash DisasterM3's copies directly. DisasterM3's copies give the 40.

### 5. Bench against Instruct: not run

DisasterM3's Bench images are not on this PC. Only `benchmark_release.json` and 1,940 Bench masks are here; the 8 unfinished Hugging Face downloads are masks too. The same twin check between Bench and Instruct decides whether the 7B's benchmark scores are affected, and it has to run on Kaggle.

## Why four files were deleted on 25 Oct, and what replaced them

The four commits were made on GitHub's website between 23:57 and 23:59, 5–7 minutes after `8b82abd94` added the Easy English and corrected PDFs. Each has GitHub's default message ("Delete …") and no description. The table says what each file was and what replaced it.

| Commit | Deleted file | What it was | Replaced by |
|---|---|---|---|
| `3ae5ab0ca` | `DisasterM3_vs_BRIGHT_Optical_Audit_EDA_Easy_English.pdf` | The 19 Oct report rewritten in easy English, 19 pages, printed 22 Oct 16:54, with the uncorrected 19 Oct numbers | `DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf`, printed at 16:59 the same day: the same easy English with the corrections. Its last page lists the changes. |
| `26ab6fb4a` | `Roadmap_Integrating BRIGHT Dataset with DisasterM3.md` | Ridita's roadmap of 17 Oct (the file name has spaces) | `Roadmap_Integrating-BRIGHT-Dataset-with-DisasterM3.md`, committed by Abrar on 19 Oct (`1b3357478`): the same text with the three task boxes unticked and the "COMPLETED" notes removed |
| `922dbd2b4` | `bright_disasterm3_standardizer.py` | The committed version of 14 Oct (`7a8043121`): 5 events, 1,285 tiles. It was out of date; the version that made the 1,906-tile set was never committed. | Nothing in that folder. Both versions are now here: `scripts/bright_disasterm3_standardizer.py` (last) and `scripts/older_versions/bright_disasterm3_standardizer_7a8043121.py` (deleted) |
| `a33aa8b12` | `DisasterM3_vs_BRIGHT_Optical_Audit_EDA.pdf` | The first version of the report, dated 19 Oct (17 pages) | `DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf` |

Two more things happened to the local files that evening:

- At 23:04, `git stash -u` took the untracked scripts and outputs out of the working tree. That is why `val_train_leaks.csv` never reached the repo.
- At 23:40–23:46, the scripts' copies in the folder were sent to the Recycle Bin.

## Where the copies came from

- **`stash@{0}`** ("WIP on main: d1992f10e …", made on 25 Oct at 23:04 with `git stash -u`):
  - Its third parent `e4f081b99` (`stash@{0}^3`) holds the untracked files: every script except the standardizer, all the outputs, and `data/bright_standardized_optical/`.
  - The stash commit itself, `a69aed484`, holds the last standardizer.
  - The copies here were written from the stored blobs with `git cat-file blob`, and `git hash-object` confirms each one is byte-identical.
- **The Recycle Bin of D:** holds the same scripts, deleted from `BRIGHT_DisasterM3_Integration\` on 25 Oct at 23:40. Their content equals the stash's; only the line endings differ.
- **`C:\Users\Ridita\AppData\Local\Temp\claude\d--CSE499A-15-Msrb-\001ca3a1-9ed3-4adc-85b7-1cd081de7a95\scratchpad\`** holds `build_easy_pdf.py`, `audit_easy_body.html` and `audit_easy.html`, all from 22 Oct 16:59. The PDF was printed at 16:59:57.
- **`922dbd2b4^`** holds the committed standardizer.

**Keep the stash.** Until `audit_outputs/` is committed, the stash is the only other copy of these outputs. It stays the only copy of the 3,812 standardized BRIGHT images and masks. Do not drop or clear it. To restore one file to its old place, run `git checkout stash@{0}^3 -- <path>`.

Other copies, not packaged:

- `D:\Downloads\dm3_bright_audit (1).py` (18 Oct): an older version of `dm3_bright_audit.py`.
- `...\ed3919e7-...\scratchpad\bright_dm3_dedup_report.before_618.json` (20 Oct): the dedup report from before the DisasterM3-copy tiles were added.
- `...\0a8d5eb8-...\scratchpad\easy_pdf\easy.html` and `img\*.png`: the source of the deleted Easy English PDF. Its `final.pdf` is byte-identical to that PDF.
- `...\0b7ae1ae-...\scratchpad\smoke\summary_ext.json`: a 60-image smoke test of the second pass.
- `Bright.html` and `disasterM3.html`: HTML copies of the two papers, not the report's source.

**Left out**, by rule: images and masks (the 3,812 standardized BRIGHT PNGs, the 17 page renders of the PDF, the Easy English figures), model files (there are none), files over 50 MB (none; the largest copy is `dm3_images.csv` at 9.1 MB), and `Offline_RAG_KR2/guidelines/Sphere_Handbook_2018_EN.pdf`.

**Also left out**, because they are not part of the audit: the training files that sat in the same folder (`train_champion_bright_disasterm3.ipynb`, `train_champion_optical_segmentation.py`, `build_full_notebook.py`, `create_combined_training_manifest.py`, `data/combined_optical_training_manifest.json`).

**Security check (26 Oct).** No file here contains a token, password or API key; the only matches are two uses of the word "token" for file-name tokens in `dm3_bright_audit.py`. No CSV contains an absolute path. Absolute paths from Ridita's PC appear only in these places:

- `outputs/bright_dm3_dedup_report.json`: `bright_manifest` = `d:\CSE499A.15-Msrb-\data\bright_standardized_optical\bright_optical_manifest.json`, and `dm3_root` = `d:\CSE499A.15-Msrb-\dataset\DisasterM3 Dataset`
- `outputs/bright_optical_manifest.json`: `dedup.report` = `d:\CSE499A.15-Msrb-\data\bright_dm3_dedup_report.json`
- the argument defaults of the five original scripts (`d:\CSE499A.15-Msrb-\dataset\...` and `d:\CSE499A.15-Msrb-\data\...`)
- `build_easy_pdf.py`: `ROOT = D:\CSE499A.15-Msrb-` and `EDGE = C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`
