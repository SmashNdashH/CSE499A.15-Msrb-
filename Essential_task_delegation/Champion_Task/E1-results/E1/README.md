# E1 + E2 results

Validation images: 645. Models: R9, R1, R9s7. Run time: 18 min.

## E1 summary

| model   | mode   |   miou |   background_iou |   intact_iou |   damaged_iou |   destroyed_iou |   building_iou |   destroyed_precision |   destroyed_recall |   missed_destroyed_px |   missed_as_background |   missed_as_intact |   missed_as_damaged | matches_d4_csv   |
|:--------|:-------|-------:|-----------------:|-------------:|--------------:|----------------:|---------------:|----------------------:|-------------------:|----------------------:|-----------------------:|-------------------:|--------------------:|:-----------------|
| R9      | std    | 0.5373 |           0.953  |       0.5571 |        0.3471 |          0.2921 |         0.584  |                0.4125 |             0.5001 |                903850 |                 0.595  |             0.2398 |              0.1653 | yes              |
| R9      | tta    | 0.544  |           0.9541 |       0.565  |        0.3568 |          0.3001 |         0.5905 |                0.4174 |             0.5163 |                874559 |                 0.6017 |             0.2189 |              0.1794 | yes              |
| R1      | std    | 0.5374 |           0.9539 |       0.5693 |        0.3578 |          0.2686 |         0.5929 |                0.3871 |             0.4674 |                962930 |                 0.5941 |             0.2353 |              0.1706 | yes              |
| R1      | tta    | 0.544  |           0.9543 |       0.5732 |        0.3661 |          0.2822 |         0.5962 |                0.4006 |             0.4885 |                924833 |                 0.5971 |             0.2309 |              0.172  | yes              |
| R9s7    | std    | 0.5382 |           0.9544 |       0.5637 |        0.35   |          0.2847 |         0.5909 |                0.3961 |             0.5032 |                898346 |                 0.5756 |             0.2445 |              0.1799 | yes              |
| R9s7    | tta    | 0.5453 |           0.9548 |       0.5686 |        0.3638 |          0.2941 |         0.5928 |                0.4034 |             0.5205 |                866973 |                 0.5891 |             0.2318 |              0.1791 | yes              |

## Gate G2

59.5% of R9's missed destroyed pixels are predicted as Background (rule: 40% or more means R11 first), so run **R11 (before + after input)** first.

## E2 majority voting (D4 CSV format)

| experiment_id      |   miou |   damaged_iou |   destroyed_iou |   destroyed_recall |
|:-------------------|-------:|--------------:|----------------:|-------------------:|
| E2_R9_std          | 0.5373 |        0.3471 |          0.2921 |             0.5001 |
| E2_R9_tta          | 0.544  |        0.3568 |          0.3001 |             0.5163 |
| E2_R9_std_vote     | 0.5186 |        0.3426 |          0.2279 |             0.3533 |
| E2_R9_tta_vote     | 0.5258 |        0.3541 |          0.234  |             0.3652 |
| E2_R9_std_vote20   | 0.5186 |        0.3426 |          0.2279 |             0.3533 |
| E2_R9_tta_vote20   | 0.5258 |        0.3541 |          0.234  |             0.3652 |
| E2_R1_std          | 0.5374 |        0.3578 |          0.2686 |             0.4674 |
| E2_R1_tta          | 0.544  |        0.3661 |          0.2822 |             0.4885 |
| E2_R1_std_vote     | 0.5307 |        0.3668 |          0.2349 |             0.3905 |
| E2_R1_tta_vote     | 0.5343 |        0.3694 |          0.2436 |             0.4031 |
| E2_R1_std_vote20   | 0.5307 |        0.3668 |          0.2349 |             0.3905 |
| E2_R1_tta_vote20   | 0.5343 |        0.3694 |          0.2436 |             0.4031 |
| E2_R9s7_std        | 0.5382 |        0.35   |          0.2847 |             0.5032 |
| E2_R9s7_tta        | 0.5453 |        0.3638 |          0.2941 |             0.5205 |
| E2_R9s7_std_vote   | 0.5263 |        0.3477 |          0.2423 |             0.4643 |
| E2_R9s7_tta_vote   | 0.5314 |        0.3583 |          0.2478 |             0.4359 |
| E2_R9s7_std_vote20 | 0.5263 |        0.3477 |          0.2423 |             0.4643 |
| E2_R9s7_tta_vote20 | 0.5314 |        0.3583 |          0.2478 |             0.4359 |

Verdict: do not adopt voting.

## Files

- `E1_summary.csv`, `E1_per_event.csv`, `E1_per_image_cm.csv`, `E1_confusion_matrices.json`, `E2_voting.csv`, `config.json`
- `preds/R9_std/`, `preds/R9_tta/`, `gt/`: uint8 PNG masks, 0 background, 1 intact, 2 damaged, 3 destroyed (same file names as the post-disaster images)