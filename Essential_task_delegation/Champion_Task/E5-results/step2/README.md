# E5 step 2: per-image pixel counts for R11, R11s7, R9 and R1

645 validation images, the D4 split. Each model as the D4 notebooks ran it: full size, padded to a multiple of 32, standard and 8-view TTA. R11 and R11s7 read the after- and before-image (6 channels), R9 and R1 the after-image only.

## Each model against its D4 CSV

| model   | mode   |   miou |   destroyed_iou |   d4_miou |   d4_destroyed_iou | reproduces_d4   |
|:--------|:-------|-------:|----------------:|----------:|-------------------:|:----------------|
| R11     | std    | 0.5811 |          0.2911 |    0.5811 |             0.2911 | yes             |
| R11     | tta    | 0.5875 |          0.2965 |    0.5875 |             0.2965 | yes             |
| R11s7   | std    | 0.5867 |          0.2869 |    0.5867 |             0.2869 | yes             |
| R11s7   | tta    | 0.5913 |          0.2928 |    0.5913 |             0.2928 | yes             |
| R9      | std    | 0.5373 |          0.2921 |    0.5373 |             0.2921 | yes             |
| R9      | tta    | 0.544  |          0.3001 |    0.544  |             0.3001 | yes             |
| R1      | std    | 0.5374 |          0.2686 |    0.5374 |             0.2686 | yes             |
| R1      | tta    | 0.544  |          0.2822 |    0.544  |             0.2822 | yes             |

## Preview: Destroyed IoU and mIoU without leaked pairs

`minus_pc21` drops the 21 pairs the PC check found (xBD's original files); `minus_86_fire` drops every socal_fire and woolsey_fire image, an upper bound on the effect. The exact list comes from E5 step 1 and is applied on the PC.

| model   | mode   |   images |   destroyed_iou_all |   minus_pc21 |   minus_86_fire |   miou_all |   miou_minus_pc21 |   miou_minus_86_fire |
|:--------|:-------|---------:|--------------------:|-------------:|----------------:|-----------:|------------------:|---------------------:|
| R11     | std    |      645 |              0.2911 |       0.2771 |          0.2524 |     0.5811 |            0.5776 |               0.5718 |
| R11     | tta    |      645 |              0.2965 |       0.2824 |          0.2574 |     0.5875 |            0.5839 |               0.5781 |
| R11s7   | std    |      645 |              0.2869 |       0.2729 |          0.2492 |     0.5867 |            0.5832 |               0.5774 |
| R11s7   | tta    |      645 |              0.2928 |       0.2786 |          0.2536 |     0.5913 |            0.5877 |               0.5818 |
| R9      | std    |      645 |              0.2921 |       0.2856 |          0.2736 |     0.5373 |            0.5355 |               0.5316 |
| R9      | tta    |      645 |              0.3001 |       0.2937 |          0.2806 |     0.544  |            0.5421 |               0.5379 |
| R1      | std    |      645 |              0.2686 |       0.2614 |          0.2481 |     0.5374 |            0.5353 |               0.531  |
| R1      | tta    |      645 |              0.2822 |       0.2749 |          0.2618 |     0.544  |            0.5419 |               0.5376 |

## How it was scored, and the alternatives

- A 4 × 4 pixel count (truth × prediction) per image, model and mode. A score on any subset is the sum of its images' counts, so dropping leaked pairs needs no new GPU run. Summed over all 645 images, the counts must reproduce each D4 CSV, which checks that nothing in the prediction changed.
- Alternative: re-run inference for each leak list. It gives the same numbers, but needs a GPU run each time the list changes (21, 40, step 1's).
- Alternative: re-score only from the leaked images' pixels, as was done for R16c from its per-building file. That works for R16c, which saved per-building counts, and not for these runs, which saved only totals.

## Files

- `per_image_counts.csv`: model, mode, image, event, source and the 16 counts `t<truth>_p<prediction>`
- `reproduction.csv`, `clean_preview.csv`, `config.json`