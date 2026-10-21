# R16c step 4: SAM 3 on the training images

5,798 of 5,798 training before-images, prompt 'building', confidence 0.5, 2×2 grid, fp16, SAM 3 commit 7f473cf. Complete.

## This session's images

|   images |   building_iou |   count_ratio |   recall_intact |   recall_damaged |   recall_destroyed |   sec_per_image |
|---------:|---------------:|--------------:|----------------:|-----------------:|-------------------:|----------------:|
|     5798 |          0.629 |         1.056 |          0.6844 |           0.6703 |             0.6932 |            2.62 |

E3 on the 645 validation images (pre_grid2): building IoU 0.6304, count ratio 1.065, destroyed footprint recall 0.6339, 2.11 s per image.

## Use

Attach this output to `r16c-step5-grader-on-sam3-outlines.ipynb` (step 5). It finds `masks/pre_grid2_train/` by its name.

## Files

- `masks/pre_grid2_train/<name>.png`: 16-bit instance maps of the before-images (0 = background), named after the after-images
- `sam3_train.csv` (this session's images), `summary.csv`, `config.json`