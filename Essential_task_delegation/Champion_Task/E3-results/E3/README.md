# E3 results: SAM 3 building finding

645 validation images, prompt 'building', confidence 0.5, precision fp16, SAM 3 commit 7f473cf.

## Part 1: smoke test (densest validation image)

| precision   |   sec_per_image |   peak_gpu_gb |   instances |   iou_vs_fp32 | error   |
|:------------|----------------:|--------------:|------------:|--------------:|:--------|
| fp32        |           1.73  |          4.3  |          11 |        1      |         |
| fp16        |           0.485 |          4.07 |          11 |        0.9998 |         |
| bf16        |           2.859 |          7.65 |          11 |        0.9987 |         |

## Part 2: building finding

| run         | source   |   images |   building_iou |   count_ratio |   count_mae |   recall_intact |   recall_damaged |   recall_destroyed |   sec_per_image |
|:------------|:---------|---------:|---------------:|--------------:|------------:|----------------:|-----------------:|-------------------:|----------------:|
| pre_grid1   | all      |      645 |         0.4841 |         0.576 |        18.2 |          0.5163 |           0.5483 |             0.4931 |            0.54 |
| pre_grid1   | other    |       85 |         0.4633 |         0.587 |        29.5 |          0.4769 |           0.5987 |             0.5507 |            0.56 |
| pre_grid1   | xBD      |      560 |         0.5018 |         0.574 |        16.5 |          0.5584 |           0.5377 |             0.4495 |            0.54 |
| post_grid1  | all      |      645 |         0.3512 |         0.518 |        20.2 |          0.4199 |           0.4184 |             0.0894 |            0.55 |
| post_grid1  | other    |       85 |         0.2634 |         0.472 |        35.3 |          0.3198 |           0.3315 |             0.108  |            0.56 |
| post_grid1  | xBD      |      560 |         0.4328 |         0.529 |        17.9 |          0.5269 |           0.4366 |             0.0754 |            0.54 |
| pre_grid2   | all      |      645 |         0.6304 |         1.065 |        11   |          0.6796 |           0.7288 |             0.6339 |            2.11 |
| pre_grid2   | other    |       85 |         0.6185 |         1.259 |        27.6 |          0.655  |           0.7644 |             0.6783 |            2.12 |
| pre_grid2   | xBD      |      560 |         0.6407 |         1.016 |         8.5 |          0.7059 |           0.7214 |             0.6003 |            2.11 |
| R9_tta (E1) | all      |      645 |         0.5905 |         0.615 |        16.3 |          0.8469 |           0.7851 |             0.709  |          nan    |
| R9_tta (E1) | other    |       85 |         0.5566 |         0.277 |        43.7 |          0.8544 |           0.7544 |             0.7417 |          nan    |
| R9_tta (E1) | xBD      |      560 |         0.6253 |         0.702 |        12.2 |          0.8388 |           0.7916 |             0.6842 |          nan    |

## Gate G3

SAM 3 pre_grid2: building IoU 0.6304 (destroyed footprint recall 0.634) vs R9 TTA 0.5905 (recall 0.709). R16 stage 1 = SAM 3 on the before image.

## Files

- `smoke.csv`, `sam3_s1.csv` (per image and run), `E3_summary.csv`, `config.json`
- `masks/<run>/<name>.png`: 16-bit instance maps (0 = background), same names as the after-images