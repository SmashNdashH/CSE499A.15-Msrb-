# R16c step 3: step 2's graders with R11s7's pieces

645 validation images. The finding as in R16: SAM 3 `pre_grid2` (E3) plus R11s7's building pieces that SAM 3 missed. The grading: step 2's Siamese ResNet-18s (seeds [42, 43]) and their biases, loaded as they were kept; nothing was trained.

## Pixel scores (E-protocol)

`_shares` rows use R16b's fixed rule instead of the searched biases. `ceiling` rows give each found building its main true grade (`rejects`: or background).

| variant                    |   miou |   background_iou |   intact_iou |   damaged_iou |   destroyed_iou |   building_iou |   destroyed_precision |   destroyed_recall |
|:---------------------------|-------:|-----------------:|-------------:|--------------:|----------------:|---------------:|----------------------:|-------------------:|
| R11s7_std                  | 0.5867 |           0.9756 |       0.6836 |        0.4007 |          0.2869 |         0.7386 |                0.5528 |             0.3736 |
| R11s7_tta                  | 0.5913 |           0.9761 |       0.688  |        0.4081 |          0.2928 |         0.7436 |                0.5661 |             0.3775 |
| ceiling_sam3               | 0.706  |           0.9685 |       0.6243 |        0.6755 |          0.5555 |         0.6304 |                0.8401 |             0.6212 |
| ceiling_sam3+r11s7         | 0.7692 |           0.9754 |       0.74   |        0.7383 |          0.6232 |         0.7439 |                0.7297 |             0.8103 |
| ceiling_rejects_sam3+r11s7 | 0.8023 |           0.9817 |       0.7898 |        0.763  |          0.6749 |         0.7912 |                0.8648 |             0.7545 |
| R16c_s42_sam3              | 0.547  |           0.9685 |       0.5769 |        0.3436 |          0.2991 |         0.6304 |                0.5129 |             0.4177 |
| R16c_s42_sam3+r11s7        | 0.5843 |           0.9754 |       0.6807 |        0.372  |          0.309  |         0.7439 |                0.4402 |             0.509  |
| R16c_s42_vote              | 0.5924 |           0.9754 |       0.6919 |        0.4142 |          0.2882 |         0.7439 |                0.5241 |             0.3903 |
| R16c_s42_sam3_shares       | 0.5288 |           0.9685 |       0.5646 |        0.3385 |          0.2435 |         0.6304 |                0.3108 |             0.5291 |
| R16c_s42_sam3+r11s7_shares | 0.5554 |           0.9754 |       0.6619 |        0.364  |          0.2205 |         0.7439 |                0.2505 |             0.6481 |
| R16c_s42_vote_shares       | 0.5882 |           0.9754 |       0.6921 |        0.4142 |          0.2711 |         0.7439 |                0.4501 |             0.4052 |
| R16c_s43_sam3              | 0.5485 |           0.9685 |       0.5832 |        0.3448 |          0.2974 |         0.6304 |                0.4951 |             0.4269 |
| R16c_s43_sam3+r11s7        | 0.583  |           0.9754 |       0.687  |        0.3689 |          0.3006 |         0.7439 |                0.393  |             0.5612 |
| R16c_s43_vote              | 0.5879 |           0.9754 |       0.6926 |        0.4113 |          0.2724 |         0.7439 |                0.4726 |             0.3914 |
| R16c_s43_sam3_shares       | 0.5325 |           0.9685 |       0.5719 |        0.3474 |          0.2422 |         0.6304 |                0.315  |             0.5117 |
| R16c_s43_sam3+r11s7_shares | 0.5553 |           0.9754 |       0.6698 |        0.3645 |          0.2116 |         0.7439 |                0.2365 |             0.6679 |
| R16c_s43_vote_shares       | 0.5855 |           0.9754 |       0.6917 |        0.4111 |          0.2638 |         0.7439 |                0.4313 |             0.4045 |

R11s7 reproduces its D4 CSV: **yes**. The graders reproduce step 2's macro F1 on the true buildings: **yes** ({42: 0.7779, 43: 0.7843}).

## Keep rule against R11s7 TTA (searched biases)

- R16c_s42_sam3: not kept; Phase Gate fail
- R16c_s42_sam3+r11s7: not kept; Phase Gate pass; against step 2 with R11's pieces: mIoU -0.0026, Destroyed IoU -0.0069
- R16c_s42_vote: not kept; Phase Gate fail
- R16c_s43_sam3: not kept; Phase Gate fail
- R16c_s43_sam3+r11s7: not kept; Phase Gate pass; against step 2 with R11's pieces: mIoU -0.0022, Destroyed IoU -0.0100
- R16c_s43_vote: not kept; Phase Gate fail

Kept on every seed: {'sam3': False, 'sam3+r11s7': False, 'vote': False}. Phase Gate on every seed: {'sam3': False, 'sam3+r11s7': True, 'vote': False}.

**O1:** SAM 3 + R11s7's pieces passes the O1 gate on both grader seeds; with step 2 (R11's pieces), that is two U-Net seeds.

## The graders on the true validation buildings (object level)

|   seed | grade     |   buildings |   precision |   recall |     f1 |   graded_intact |   graded_damaged |   graded_destroyed |
|-------:|:----------|------------:|------------:|---------:|-------:|----------------:|-----------------:|-------------------:|
|     42 | Intact    |       19624 |      0.9195 |   0.9507 | 0.9348 |           18656 |              787 |                181 |
|     42 | Damaged   |        4370 |      0.7372 |   0.6675 | 0.7006 |            1316 |             2917 |                137 |
|     42 | Destroyed |        1600 |      0.7639 |   0.6431 | 0.6983 |             318 |              253 |               1029 |
|     43 | Intact    |       19624 |      0.9248 |   0.9511 | 0.9377 |           18664 |              718 |                242 |
|     43 | Damaged   |        4370 |      0.7504 |   0.6632 | 0.7041 |            1284 |             2898 |                188 |
|     43 | Destroyed |        1600 |      0.7226 |   0.7    | 0.7111 |             234 |              246 |               1120 |

## Files

- `D4_R16c_step3.csv` (D4 format), `R16c_step3_summary.csv` (also per source), `grader_true_buildings.csv`
- `instances_val.csv`: one row per found building (each seed's log-probabilities, R11s7's and the truth's pixels)
- `config.json`

## Benchmark use

SAM 3 isn't trained, the graders saw DisasterM3's training images only, and R11s7 never trained on benchmark pictures, so these outputs may answer benchmark questions.