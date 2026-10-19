# R16 results: two-stage model (find each building, then grade it)

645 validation images. Stage 1: SAM 3 `pre_grid2` (E3), plus R11's building pieces that SAM 3 missed. Stage 2: Siamese ResNet-18 grader, 96 px crops, 10 epochs, 102,716 training buildings.

## Pixel scores (E-protocol)

| variant      |   miou |   background_iou |   intact_iou |   damaged_iou |   destroyed_iou |   building_iou |   destroyed_precision |   destroyed_recall |
|:-------------|-------:|-----------------:|-------------:|--------------:|----------------:|---------------:|----------------------:|-------------------:|
| R11_std      | 0.5811 |           0.975  |       0.6753 |        0.3828 |          0.2911 |         0.7356 |                0.5379 |             0.3882 |
| R11_tta      | 0.5875 |           0.9759 |       0.6821 |        0.3956 |          0.2965 |         0.742  |                0.5518 |             0.3906 |
| R16_sam3     | 0.4711 |           0.9685 |       0.4713 |        0.2772 |          0.1672 |         0.6304 |                0.1902 |             0.5813 |
| R16_sam3+r11 | 0.4941 |           0.9752 |       0.569  |        0.28   |          0.1522 |         0.7426 |                0.16   |             0.7576 |
| R16_vote     | 0.5796 |           0.9752 |       0.6846 |        0.3975 |          0.2609 |         0.7426 |                0.4142 |             0.4135 |

R11 reproduces its D4 CSV: **yes**.

## Keep rule against R11 TTA

- R16_sam3: not kept
- R16_sam3+r11: not kept
- R16_vote: not kept

## The grader on the true validation buildings (object level)

| grade     |   buildings |   precision |   recall |     f1 |   graded_intact |   graded_damaged |   graded_destroyed |
|:----------|------------:|------------:|---------:|-------:|----------------:|-----------------:|-------------------:|
| Intact    |       19624 |      0.9709 |   0.7512 | 0.847  |           14742 |             2958 |               1924 |
| Damaged   |        4370 |      0.5169 |   0.7588 | 0.6149 |             400 |             3316 |                654 |
| Destroyed |        1600 |      0.3547 |   0.8856 | 0.5065 |              42 |              141 |               1417 |

Macro F1 0.6562 (BDAChat reports 0.763 on its own split, so the comparison is rough).

## Files

- `D4_R16.csv` (D4 format), `R16_summary.csv` (also per source), `grader_true_buildings.csv`
- `instances_val.csv`: one row per stage-1 building (the grader's probabilities, R11's and the truth's pixel counts)
- `R11_tta/`: R11's TTA masks (uint8, 0-3, same names as the after-images)
- `grader.pth`, `history.json`, `config.json`

## Benchmark use

SAM 3 isn't trained and the grader saw DisasterM3's training images only. The U-Net decides the rest: with R11 (never trained on benchmark pictures) these outputs may answer benchmark questions; with R12 they may not.