# R16c step 5: the grader trained on SAM 3's outlines

645 validation images. The finding as in R16: SAM 3 `pre_grid2` (E3) plus R11's building pieces that SAM 3 missed. The grading: Siamese ResNet-18, 128 px crops with a 32 px margin, trained on SAM 3's outlines of 5,218 training images (step 4), with the labels ['Intact', 'Damaged', 'Destroyed', 'None']; the epoch and the label biases were picked on 580 held-out training images; seeds [42, 43].

## Pixel scores (E-protocol)

`_shares` rows use R16b's fixed rule instead of the searched biases. `ceiling` rows give each found building its main true grade (`rejects`: or background).

| variant                  |   miou |   background_iou |   intact_iou |   damaged_iou |   destroyed_iou |   building_iou |   destroyed_precision |   destroyed_recall |
|:-------------------------|-------:|-----------------:|-------------:|--------------:|----------------:|---------------:|----------------------:|-------------------:|
| R11_std                  | 0.5811 |           0.975  |       0.6753 |        0.3828 |          0.2911 |         0.7356 |                0.5379 |             0.3882 |
| R11_tta                  | 0.5875 |           0.9759 |       0.6821 |        0.3956 |          0.2965 |         0.742  |                0.5518 |             0.3906 |
| ceiling_sam3             | 0.706  |           0.9685 |       0.6243 |        0.6755 |          0.5555 |         0.6304 |                0.8401 |             0.6212 |
| ceiling_sam3+r11         | 0.7657 |           0.9752 |       0.7382 |        0.7359 |          0.6136 |         0.7426 |                0.7296 |             0.7942 |
| ceiling_rejects_sam3+r11 | 0.7981 |           0.9816 |       0.7888 |        0.7616 |          0.6606 |         0.7907 |                0.8632 |             0.7378 |
| R16c_s42_sam3            | 0.552  |           0.9689 |       0.5716 |        0.3326 |          0.3347 |         0.6301 |                0.6321 |             0.4157 |
| R16c_s42_sam3+r11        | 0.5496 |           0.975  |       0.6478 |        0.3323 |          0.2434 |         0.7288 |                0.3169 |             0.5122 |
| R16c_s42_vote            | 0.5788 |           0.975  |       0.6696 |        0.3913 |          0.2793 |         0.7288 |                0.5022 |             0.3863 |
| R16c_s42_sam3_shares     | 0.5468 |           0.969  |       0.5682 |        0.3348 |          0.3152 |         0.6334 |                0.4909 |             0.4683 |
| R16c_s42_sam3+r11_shares | 0.5371 |           0.9759 |       0.6426 |        0.3446 |          0.1853 |         0.7461 |                0.2098 |             0.6138 |
| R16c_s42_vote_shares     | 0.585  |           0.9759 |       0.6875 |        0.3965 |          0.28   |         0.7461 |                0.4713 |             0.4082 |
| R16c_s43_sam3            | 0.5522 |           0.969  |       0.5646 |        0.3509 |          0.3242 |         0.6308 |                0.5883 |             0.4193 |
| R16c_s43_sam3+r11        | 0.541  |           0.9751 |       0.6297 |        0.3433 |          0.2158 |         0.7276 |                0.2813 |             0.4808 |
| R16c_s43_vote            | 0.5753 |           0.9751 |       0.669  |        0.3989 |          0.2581 |         0.7276 |                0.5414 |             0.3304 |
| R16c_s43_sam3_shares     | 0.5465 |           0.969  |       0.5728 |        0.3514 |          0.2927 |         0.632  |                0.4064 |             0.5112 |
| R16c_s43_sam3+r11_shares | 0.5382 |           0.9757 |       0.636  |        0.3695 |          0.1716 |         0.7384 |                0.1877 |             0.6659 |
| R16c_s43_vote_shares     | 0.5829 |           0.9757 |       0.68   |        0.3996 |          0.2763 |         0.7384 |                0.4721 |             0.3999 |

R11 reproduces its D4 CSV: **yes**.

## Keep rule against R11 TTA (searched biases)

- R16c_s42_sam3: not kept; Phase Gate pass; 33% of the gap from R16b to the ceiling
- R16c_s42_sam3+r11: not kept; Phase Gate fail; 9% of the gap from R16b to the ceiling
- R16c_s42_vote: not kept; Phase Gate fail
- R16c_s43_sam3: not kept; Phase Gate pass; 29% of the gap from R16b to the ceiling
- R16c_s43_sam3+r11: not kept; Phase Gate fail; 2% of the gap from R16b to the ceiling
- R16c_s43_vote: not kept; Phase Gate fail

Kept on every seed: {'sam3': False, 'sam3+r11': False, 'vote': False}. Phase Gate on every seed: {'sam3': True, 'sam3+r11': False, 'vote': False}.

## Against step 2 (the rule fixed before the run)

- R16c_s42_sam3+r11: mIoU -0.0373, Destroyed IoU -0.0725 against step 2's (0.5869, 0.3159)
- R16c_s43_sam3+r11: mIoU -0.0442, Destroyed IoU -0.0948 against step 2's (0.5852, 0.3106)

**Replaces step 2: no.**

## Picked on the held-out training images

|   seed |   epoch |   held_out_grade_iou | bias                                                                |
|-------:|--------:|---------------------:|:--------------------------------------------------------------------|
|     42 |       5 |               0.6199 | {'Intact': 0.0, 'Damaged': -1.75, 'Destroyed': -4.0, 'None': -1.25} |
|     43 |       4 |               0.6081 | {'Intact': 0.0, 'Damaged': -1.0, 'Destroyed': -4.0, 'None': -2.0}   |

## The graders on the true validation buildings (object level)

|   seed | grade     |   buildings |   precision |   recall |     f1 |   graded_intact |   graded_damaged |   graded_destroyed |   graded_none |
|-------:|:----------|------------:|------------:|---------:|-------:|----------------:|-----------------:|-------------------:|--------------:|
|     42 | Intact    |       19624 |      0.9284 |   0.8629 | 0.8945 |           16933 |              980 |                139 |          1572 |
|     42 | Damaged   |        4370 |      0.6953 |   0.6249 | 0.6582 |            1141 |             2731 |                 91 |           407 |
|     42 | Destroyed |        1600 |      0.7957 |   0.56   | 0.6574 |             164 |              217 |                896 |           323 |
|     43 | Intact    |       19624 |      0.9485 |   0.8573 | 0.9006 |           16823 |             1600 |                123 |          1078 |
|     43 | Damaged   |        4370 |      0.6319 |   0.7604 | 0.6902 |             800 |             3323 |                 78 |           169 |
|     43 | Destroyed |        1600 |      0.7934 |   0.4825 | 0.6001 |             113 |              336 |                772 |           379 |

Macro F1 per seed: {42: 0.7367, 43: 0.7303} (step 2: 0.7779 and 0.7843; R16: 0.656; BDAChat reports 0.763 on its own split).

## Files

- `D4_R16c_step5.csv` (D4 format), `R16c_step5_summary.csv` (also per source), `grader_true_buildings.csv`
- `instances_val.csv`: one row per found building (each seed's log-probabilities, R11's and the truth's pixels)
- `calibration.json` (the kept epoch and biases), `history.json`, `grader_seed*.pth`, `config.json`

## Benchmark use

SAM 3 isn't trained and the graders saw DisasterM3's training images only. With R11 (never trained on benchmark pictures) these outputs may answer benchmark questions.