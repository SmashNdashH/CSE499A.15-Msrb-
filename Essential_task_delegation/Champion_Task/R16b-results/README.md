# R16b results: R16's grader corrected by the real grade shares

R16's grader drew each grade equally often in training. Here each grade's probability is multiplied by its real share among the training buildings (Intact 73.7%, Damaged 18.4%, Destroyed 8.0%) and divided by a third, and the largest wins. The shares come from the 5,798 training masks, so nothing is fitted on the validation images. R16's own scores were first rebuilt exactly from its per-building table.

## Pixel scores (E-protocol, 645 validation images)

```
            grades      variant   miou  intact_iou  damaged_iou  destroyed_iou  building_iou  destroyed_precision  destroyed_recall
         reference      R11_std 0.5811      0.6753       0.3828         0.2911        0.7356               0.5379            0.3882
         reference      R11_tta 0.5875      0.6821       0.3956         0.2965        0.7420               0.5518            0.3906
R16 (equal shares)     R16_sam3 0.4711      0.4713       0.2772         0.1672        0.6304               0.1902            0.5813
R16 (equal shares) R16_sam3+r11 0.4941      0.5690       0.2800         0.1522        0.7426               0.1600            0.7576
R16 (equal shares)     R16_vote 0.5796      0.6846       0.3975         0.2609        0.7426               0.4142            0.4135
R16b (real shares)     R16_sam3 0.5150      0.5450       0.3186         0.2279        0.6304               0.2809            0.5474
R16b (real shares) R16_sam3+r11 0.5407      0.6441       0.3357         0.2079        0.7426               0.2270            0.7125
R16b (real shares)     R16_vote 0.5846      0.6864       0.4033         0.2737        0.7426               0.4481            0.4128
```

## Keep rule against R11 TTA

- R16_sam3: not kept
- R16_sam3+r11: not kept
- R16_vote: not kept

## By source (xBD 560 images, other 85)

```
            grades      variant source   miou  destroyed_iou  destroyed_precision  destroyed_recall
         reference      R11_tta    xBD 0.6462         0.4562               0.6348            0.6186
         reference      R11_tta  other 0.4316         0.0707               0.2516            0.0896
R16 (equal shares)     R16_sam3    xBD 0.5031         0.2321               0.2919            0.5310
R16 (equal shares) R16_sam3+r11    xBD 0.5072         0.1800               0.1929            0.7292
R16 (equal shares)     R16_vote    xBD 0.6237         0.3770               0.4706            0.6545
R16 (equal shares)     R16_sam3  other 0.3901         0.1284               0.1380            0.6478
R16 (equal shares) R16_sam3+r11  other 0.4310         0.1282               0.1326            0.7953
R16 (equal shares)     R16_vote  other 0.4339         0.0688               0.1984            0.0952
R16b (real shares)     R16_sam3    xBD 0.5506         0.2967               0.4091            0.5192
R16b (real shares) R16_sam3+r11    xBD 0.5542         0.2271               0.2502            0.7110
R16b (real shares)     R16_vote    xBD 0.6311         0.3979               0.5041            0.6540
R16b (real shares)     R16_sam3  other 0.4061         0.1792               0.2054            0.5846
R16b (real shares) R16_sam3+r11  other 0.4546         0.1871               0.2023            0.7143
R16b (real shares)     R16_vote  other 0.4351         0.0709               0.2221            0.0943
```

## Stage-1 buildings that are mostly one true building

39,046 of the 57,991 stage-1 buildings have at least half their pixels on true buildings; each is compared with its main true grade. R16's table of the grader on the true buildings needs the images and a GPU, so this one stands in for it.

```
            grades true_grade  buildings  precision  recall     f1  graded_intact  graded_damaged  graded_destroyed
R16 (equal shares)     Intact      31112     0.9671  0.7424 0.8400          23098            4475              3539
R16 (equal shares)    Damaged       6410     0.5010  0.7239 0.5922            738            4640              1032
R16 (equal shares)  Destroyed       1524     0.2254  0.8727 0.3582             48             146              1330
R16b (real shares)     Intact      31112     0.9385  0.8957 0.9166          27867            1745              1500
R16b (real shares)    Damaged       6410     0.6792  0.6315 0.6545           1711            4048               651
R16b (real shares)  Destroyed       1524     0.3659  0.8143 0.5049            116             167              1241
```

Macro F1: {'R16 (equal shares)': 0.5968, 'R16b (real shares)': 0.692}.

## Files

- `R16b_summary.csv`: the three R16 maps with R16's grades and with the corrected ones, overall and per source
- `R16b_objects.csv`: the stage-1 buildings' grades against their main true grade
- `config.json`: the shares, the building counts and the keep verdicts