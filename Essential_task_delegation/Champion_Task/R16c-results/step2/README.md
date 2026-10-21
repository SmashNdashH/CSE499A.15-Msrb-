# R16c results (truth): a stronger R16 grader

645 validation images. Stage 1 as in R16: SAM 3 `pre_grid2` (E3) plus R11's building pieces that SAM 3 missed. Grader: Siamese ResNet-18, 128 px crops with a 32 px margin, trained on the true pieces of 5,218 training images; the epoch and the label biases were picked on 580 held-out training images; seeds [42, 43].

## Pixel scores (E-protocol)

`_shares` rows use R16b's fixed rule instead of the searched biases. `ceiling` rows give each stage-1 building its main true grade (`rejects`: or background).

| variant                  |   miou |   background_iou |   intact_iou |   damaged_iou |   destroyed_iou |   building_iou |   destroyed_precision |   destroyed_recall |
|:-------------------------|-------:|-----------------:|-------------:|--------------:|----------------:|---------------:|----------------------:|-------------------:|
| R11_std                  | 0.5811 |           0.975  |       0.6753 |        0.3828 |          0.2911 |         0.7356 |                0.5379 |             0.3882 |
| R11_tta                  | 0.5875 |           0.9759 |       0.6821 |        0.3956 |          0.2965 |         0.742  |                0.5518 |             0.3906 |
| ceiling_sam3             | 0.706  |           0.9685 |       0.6243 |        0.6755 |          0.5555 |         0.6304 |                0.8401 |             0.6212 |
| ceiling_sam3+r11         | 0.7657 |           0.9752 |       0.7382 |        0.7359 |          0.6136 |         0.7426 |                0.7296 |             0.7942 |
| ceiling_rejects_sam3+r11 | 0.7981 |           0.9816 |       0.7888 |        0.7616 |          0.6606 |         0.7907 |                0.8632 |             0.7378 |
| R16c_s42_sam3            | 0.547  |           0.9685 |       0.5769 |        0.3436 |          0.2991 |         0.6304 |                0.5129 |             0.4177 |
| R16c_s42_sam3+r11        | 0.5869 |           0.9752 |       0.681  |        0.3754 |          0.3159 |         0.7426 |                0.4558 |             0.5072 |
| R16c_s42_vote            | 0.588  |           0.9752 |       0.6872 |        0.4038 |          0.2858 |         0.7426 |                0.4893 |             0.4072 |
| R16c_s42_sam3_shares     | 0.5288 |           0.9685 |       0.5646 |        0.3385 |          0.2435 |         0.6304 |                0.3108 |             0.5291 |
| R16c_s42_sam3+r11_shares | 0.5553 |           0.9752 |       0.6622 |        0.3699 |          0.2138 |         0.7426 |                0.2423 |             0.6448 |
| R16c_s42_vote_shares     | 0.5853 |           0.9752 |       0.688  |        0.4044 |          0.2737 |         0.7426 |                0.4508 |             0.4106 |
| R16c_s43_sam3            | 0.5485 |           0.9685 |       0.5832 |        0.3449 |          0.2974 |         0.6304 |                0.4951 |             0.4269 |
| R16c_s43_sam3+r11        | 0.5852 |           0.9752 |       0.6872 |        0.3679 |          0.3106 |         0.7426 |                0.4126 |             0.5567 |
| R16c_s43_vote            | 0.5843 |           0.9752 |       0.6877 |        0.4016 |          0.2728 |         0.7426 |                0.4496 |             0.4096 |
| R16c_s43_sam3_shares     | 0.5325 |           0.9685 |       0.5719 |        0.3474 |          0.2422 |         0.6304 |                0.315  |             0.5117 |
| R16c_s43_sam3+r11_shares | 0.5577 |           0.9752 |       0.671  |        0.3676 |          0.2169 |         0.7426 |                0.243  |             0.6688 |
| R16c_s43_vote_shares     | 0.5828 |           0.9752 |       0.6876 |        0.4009 |          0.2675 |         0.7426 |                0.4317 |             0.4128 |

R11 reproduces its D4 CSV: **yes**.

## Keep rule against R11 TTA (searched biases)

- R16c_s42_sam3: not kept; Phase Gate fail; 22% of the gap from R16b to the ceiling
- R16c_s42_sam3+r11: KEEP; Phase Gate pass; 27% of the gap from R16b to the ceiling
- R16c_s42_vote: not kept; Phase Gate fail
- R16c_s43_sam3: not kept; Phase Gate fail; 21% of the gap from R16b to the ceiling
- R16c_s43_sam3+r11: KEEP; Phase Gate pass; 25% of the gap from R16b to the ceiling
- R16c_s43_vote: not kept; Phase Gate fail

Kept on every seed: {'sam3': False, 'sam3+r11': True, 'vote': False}. Phase Gate on every seed: {'sam3': False, 'sam3+r11': True, 'vote': False}.

## Picked on the held-out training images

|   seed |   epoch |   held_out_grade_iou | bias                                                  |
|-------:|--------:|---------------------:|:------------------------------------------------------|
|     42 |       6 |               0.664  | {'Intact': 0.0, 'Damaged': -1.75, 'Destroyed': -3.75} |
|     43 |       6 |               0.6638 | {'Intact': 0.0, 'Damaged': -2.0, 'Destroyed': -4.0}   |

## The graders on the true validation buildings (object level)

|   seed | grade     |   buildings |   precision |   recall |     f1 |   graded_intact |   graded_damaged |   graded_destroyed |
|-------:|:----------|------------:|------------:|---------:|-------:|----------------:|-----------------:|-------------------:|
|     42 | Intact    |       19624 |      0.9195 |   0.9507 | 0.9348 |           18656 |              787 |                181 |
|     42 | Damaged   |        4370 |      0.7372 |   0.6675 | 0.7006 |            1316 |             2917 |                137 |
|     42 | Destroyed |        1600 |      0.7639 |   0.6431 | 0.6983 |             318 |              253 |               1029 |
|     43 | Intact    |       19624 |      0.9248 |   0.9511 | 0.9377 |           18664 |              718 |                242 |
|     43 | Damaged   |        4370 |      0.7504 |   0.6632 | 0.7041 |            1284 |             2898 |                188 |
|     43 | Destroyed |        1600 |      0.7226 |   0.7    | 0.7111 |             234 |              246 |               1120 |

Macro F1 per seed: {42: 0.7779, 43: 0.7843} (R16: 0.656; BDAChat reports 0.763 on its own split).

## Files

- `D4_R16c.csv` (D4 format), `R16c_summary.csv` (also per source), `grader_true_buildings.csv`
- `instances_val.csv`: one row per stage-1 building (each seed's log-probabilities, R11's and the truth's pixels)
- `calibration.json` (the kept epoch and biases), `history.json`, `grader_seed*.pth`, `config.json`

## Benchmark use

SAM 3 isn't trained and the graders saw DisasterM3's training images only. With R11 (never trained on benchmark pictures) these outputs may answer benchmark questions; with R12 they may not.