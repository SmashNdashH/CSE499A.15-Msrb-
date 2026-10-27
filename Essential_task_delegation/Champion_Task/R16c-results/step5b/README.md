# R16c step 5b: a third grader seed for step 5's SAM 3 map

645 validation images, 605 of them clean (without E5's 40 leaked pairs). The finding as in R16: SAM 3 `pre_grid2` (E3) plus R11's building pieces that SAM 3 missed. The grading: Siamese ResNet-18, 128 px crops with a 32 px margin, trained on SAM 3's outlines of 5,218 training images (step 4), with the labels ['Intact', 'Damaged', 'Destroyed', 'None']; the epoch and the label biases were picked on 580 held-out training images; seed 44 (step 5: 42 and 43).

## O1 on the clean pairs (the rule fixed before the run)

|   seed | source                      |   miou |   destroyed_iou | gate   |
|-------:|:----------------------------|-------:|----------------:|:-------|
|     42 | step 5, E5's clean re-score | 0.5469 |          0.3161 | True   |
|     43 | step 5, E5's clean re-score | 0.547  |          0.3051 | True   |
|     44 | this run                    | 0.548  |          0.2982 | False  |

Mean of the seeds: mIoU 0.5473, Destroyed IoU 0.3065. **O1: met on the mean of the 3 grader seeds only, narrowly.**

## Pixel scores (E-protocol), all pairs

`_shares` rows use R16b's fixed rule instead of the searched biases. `ceiling` rows give each found building its main true grade (`rejects`: or background).

| variant                  |   miou |   background_iou |   intact_iou |   damaged_iou |   destroyed_iou |   building_iou |   destroyed_precision |   destroyed_recall |
|:-------------------------|-------:|-----------------:|-------------:|--------------:|----------------:|---------------:|----------------------:|-------------------:|
| R11_std                  | 0.5811 |           0.975  |       0.6753 |        0.3828 |          0.2911 |         0.7356 |                0.5379 |             0.3882 |
| R11_tta                  | 0.5875 |           0.9759 |       0.6821 |        0.3956 |          0.2965 |         0.742  |                0.5518 |             0.3906 |
| ceiling_sam3             | 0.706  |           0.9685 |       0.6243 |        0.6755 |          0.5555 |         0.6304 |                0.8401 |             0.6212 |
| ceiling_sam3+r11         | 0.7657 |           0.9752 |       0.7382 |        0.7359 |          0.6136 |         0.7426 |                0.7296 |             0.7942 |
| ceiling_rejects_sam3+r11 | 0.7981 |           0.9816 |       0.7888 |        0.7616 |          0.6606 |         0.7907 |                0.8632 |             0.7378 |
| R16c_s44_sam3            | 0.5527 |           0.969  |       0.5774 |        0.3489 |          0.3156 |         0.632  |                0.4953 |             0.4652 |
| R16c_s44_sam3+r11        | 0.5432 |           0.9759 |       0.649  |        0.3666 |          0.1813 |         0.7447 |                0.2091 |             0.5776 |
| R16c_s44_vote            | 0.5876 |           0.9759 |       0.6886 |        0.4053 |          0.2805 |         0.7447 |                0.4831 |             0.4008 |
| R16c_s44_sam3_shares     | 0.5401 |           0.9688 |       0.5635 |        0.3429 |          0.2852 |         0.6286 |                0.3893 |             0.5159 |
| R16c_s44_sam3+r11_shares | 0.5312 |           0.9759 |       0.6285 |        0.3603 |          0.16   |         0.7405 |                0.1732 |             0.6767 |
| R16c_s44_vote_shares     | 0.5836 |           0.9759 |       0.6829 |        0.4008 |          0.2749 |         0.7405 |                0.4628 |             0.4038 |

## Pixel scores on the 605 clean pairs

| variant                  |   miou |   background_iou |   intact_iou |   damaged_iou |   destroyed_iou |   building_iou |   destroyed_precision |   destroyed_recall |
|:-------------------------|-------:|-----------------:|-------------:|--------------:|----------------:|---------------:|----------------------:|-------------------:|
| R11_std                  | 0.5742 |           0.9738 |       0.6755 |        0.3836 |          0.2641 |         0.7366 |                0.5181 |             0.3501 |
| R11_tta                  | 0.5805 |           0.9746 |       0.6821 |        0.3963 |          0.2692 |         0.7429 |                0.5327 |             0.3524 |
| ceiling_sam3             | 0.7037 |           0.9668 |       0.6239 |        0.6758 |          0.5482 |         0.6302 |                0.8377 |             0.6133 |
| ceiling_sam3+r11         | 0.7645 |           0.9739 |       0.739  |        0.7362 |          0.6089 |         0.7436 |                0.7263 |             0.7902 |
| ceiling_rejects_sam3+r11 | 0.7968 |           0.9806 |       0.789  |        0.7619 |          0.6557 |         0.7911 |                0.8612 |             0.7332 |
| R16c_s44_sam3            | 0.548  |           0.9673 |       0.5766 |        0.3499 |          0.2982 |         0.6317 |                0.4733 |             0.4462 |
| R16c_s44_sam3+r11        | 0.5395 |           0.9747 |       0.6488 |        0.3675 |          0.1671 |         0.7457 |                0.1924 |             0.5597 |
| R16c_s44_vote            | 0.5809 |           0.9747 |       0.6886 |        0.406  |          0.2545 |         0.7457 |                0.4584 |             0.364  |
| R16c_s44_sam3_shares     | 0.5357 |           0.9671 |       0.5626 |        0.3437 |          0.2693 |         0.6282 |                0.3683 |             0.5003 |
| R16c_s44_sam3+r11_shares | 0.5281 |           0.9746 |       0.6282 |        0.3611 |          0.1486 |         0.7414 |                0.1605 |             0.6664 |
| R16c_s44_vote_shares     | 0.577  |           0.9746 |       0.6829 |        0.4014 |          0.2492 |         0.7414 |                0.4364 |             0.3675 |

R11 reproduces its D4 CSV: **yes**. The saved rows rebuild the maps' scores: **yes**.

## Keep rule against R11 TTA (searched biases, all pairs) and the Phase Gate

- R16c_s44_sam3: not kept; Phase Gate pass (clean: fail); 27% of the gap from R16b to the ceiling
- R16c_s44_sam3+r11: not kept; Phase Gate fail (clean: fail); -6% of the gap from R16b to the ceiling
- R16c_s44_vote: not kept; Phase Gate fail (clean: fail)

## Picked on the held-out training images

|   seed |   epoch |   held_out_grade_iou | bias                                                              |
|-------:|--------:|---------------------:|:------------------------------------------------------------------|
|     44 |      10 |               0.6051 | {'Intact': 0.0, 'Damaged': -2.0, 'Destroyed': -4.0, 'None': -3.5} |

## The grader on the true validation buildings (object level)

|   seed | grade     |   buildings |   precision |   recall |     f1 |   graded_intact |   graded_damaged |   graded_destroyed |   graded_none |
|-------:|:----------|------------:|------------:|---------:|-------:|----------------:|-----------------:|-------------------:|--------------:|
|     44 | Intact    |       19624 |      0.9256 |   0.911  | 0.9182 |           17878 |              806 |                339 |           601 |
|     44 | Damaged   |        4370 |      0.733  |   0.6501 | 0.6891 |            1230 |             2841 |                178 |           121 |
|     44 | Destroyed |        1600 |      0.666  |   0.6444 | 0.655  |             208 |              229 |               1031 |           132 |

Macro F1: {44: 0.7541} (step 5: 0.7367 and 0.7303; step 2: 0.7779 and 0.7843; R16: 0.656; BDAChat reports 0.763 on its own split).

## Files

- `D4_R16c_step5b.csv` (D4 format; `_clean` rows on the clean pairs), `R16c_step5b_summary.csv` (all, per source and clean), `grader_true_buildings.csv`
- `instances_val.csv`: one row per found building (the seed's log-probabilities with 6 decimals, R11's and the truth's pixels)
- `calibration.json` (the kept epoch and biases), `history.json`, `grader_seed*.pth`, `config.json`