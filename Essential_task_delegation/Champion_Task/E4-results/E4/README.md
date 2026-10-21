# E4 results: R12 against R11, by source and by event

645 validation images, E-protocol (full resolution, dataset-level confusion matrix per group). R12 is R11's recipe plus 1,619 extra xBD tiles (8,021 destroyed buildings).

Each model reproduces its D4 CSV: **R11 yes, R12 yes**.

## By source

| group   | mode   |   images |   true_destroyed_px |   R11_miou |   R12_miou |   miou_change |   R11_destroyed_iou |   R12_destroyed_iou |   destroyed_change |   R11_damaged_iou |   R12_damaged_iou |   R11_destroyed_recall |   R12_destroyed_recall |
|:--------|:-------|---------:|--------------------:|-----------:|-----------:|--------------:|--------------------:|--------------------:|-------------------:|------------------:|------------------:|-----------------------:|-----------------------:|
| all     | std    |      645 |             1808111 |     0.5811 |     0.582  |        0.0009 |              0.2911 |              0.2898 |            -0.0013 |            0.3828 |            0.3894 |                 0.3882 |                 0.3835 |
| all     | tta    |      645 |             1808111 |     0.5875 |     0.5883 |        0.0008 |              0.2965 |              0.2945 |            -0.002  |            0.3956 |            0.3996 |                 0.3906 |                 0.3834 |
| xBD     | std    |      560 |             1029026 |     0.6369 |     0.6446 |        0.0077 |              0.4446 |              0.4551 |             0.0105 |            0.4556 |            0.4702 |                 0.6131 |                 0.604  |
| xBD     | tta    |      560 |             1029026 |     0.6462 |     0.6525 |        0.0063 |              0.4562 |              0.4628 |             0.0066 |            0.4701 |            0.4823 |                 0.6186 |                 0.6064 |
| other   | std    |       85 |              779085 |     0.4299 |     0.4254 |       -0.0045 |              0.0717 |              0.0701 |            -0.0016 |            0.0631 |            0.0583 |                 0.0913 |                 0.0924 |
| other   | tta    |       85 |              779085 |     0.4316 |     0.4286 |       -0.003  |              0.0707 |              0.0688 |            -0.0019 |            0.0653 |            0.0607 |                 0.0896 |                 0.0888 |

**Reading** (rule fixed before the run; TTA; seed noise 0.013): Not the wrong place: R12 changes Destroyed IoU by +0.007 on the xBD images, within the seed noise (0.013), although every extra tile is xBD (-0.002 on the others). More xBD destroyed examples don't help at this training budget, even where they were added.

## By event (with TTA)

`extra_tiles` and `extra_destroyed`: what R12 got for the event (D5 and D5b).

| event               | source   |   images |   true_destroyed_px |   extra_tiles |   extra_destroyed |   R11_destroyed_iou |   R12_destroyed_iou |   destroyed_change |   R11_miou |   R12_miou |
|:--------------------|:---------|---------:|--------------------:|--------------:|------------------:|--------------------:|--------------------:|-------------------:|-----------:|-----------:|
| turkey_earthquake   | other    |       64 |              660440 |             0 |                 0 |              0      |              0      |             0      |     0.4073 |     0.4056 |
| palu_tsunami        | xBD      |        7 |              146414 |            82 |              2206 |              0.5285 |              0.5758 |             0.0473 |     0.5376 |     0.5718 |
| santa_rosa_wildfire | xBD      |       20 |              142941 |            41 |               796 |              0.6908 |              0.6948 |             0.004  |     0.5994 |     0.6025 |
| hawaii_wildfire     | other    |        1 |              114135 |             0 |                 0 |              0.2292 |              0.2136 |            -0.0156 |     0.24   |     0.2301 |
| socal_fire          | xBD      |       54 |              113661 |           467 |                81 |              0.5162 |              0.5032 |            -0.013  |     0.5429 |     0.5416 |
| moore_tornado       | xBD      |       16 |              112101 |            60 |               208 |              0.6569 |              0.6755 |             0.0186 |     0.7405 |     0.7542 |
| woolsey_fire        | xBD      |       32 |              111020 |            60 |               374 |              0.5912 |              0.6054 |             0.0142 |     0.6003 |     0.5863 |
| hurricane_harvey    | xBD      |       44 |              102570 |            61 |                31 |              0.0029 |              0.004  |             0.0011 |     0.5267 |     0.5396 |
| hurricane_michael   | xBD      |       42 |               74907 |            45 |                99 |              0.321  |              0.2903 |            -0.0307 |     0.5693 |     0.5645 |
| midwest_flooding    | xBD      |       35 |               54504 |            65 |                 5 |              0      |              0      |             0      |     0.4364 |     0.4397 |
| portugal_wildfire   | xBD      |      105 |               50663 |            60 |                61 |              0.3828 |              0.3858 |             0.003  |     0.5375 |     0.5354 |
| hurricane_matthew   | xBD      |       25 |               37059 |            56 |               227 |              0.2284 |              0.2345 |             0.0061 |     0.4413 |     0.4562 |
| lower_puna_volcano  | xBD      |        8 |               31484 |            60 |               173 |              0.7052 |              0.7056 |             0.0004 |     0.6226 |     0.5854 |
| hurricane_florence  | xBD      |       57 |               25975 |            51 |                 0 |              0.0002 |              0      |            -0.0002 |     0.5738 |     0.5649 |
| tuscaloosa_tornado  | xBD      |       23 |               11107 |            60 |               210 |              0.46   |              0.4738 |             0.0138 |     0.6719 |     0.6622 |
| nepal_flooding      | xBD      |       48 |                8952 |            60 |                48 |              0.0303 |              0.0348 |             0.0045 |     0.4965 |     0.4941 |
| pinery_bushfire     | xBD      |       40 |                5668 |            60 |                40 |              0.1472 |              0.1469 |            -0.0003 |     0.5813 |     0.5713 |
| kalehe_flooding     | other    |        5 |                4510 |             0 |                 0 |              0      |              0      |             0      |     0.2518 |     0.251  |
| beriut_explosion    | other    |        4 |                   0 |             0 |                 0 |              0      |              0      |             0      |     0.2795 |     0.2905 |
| bata_explosion      | other    |        3 |                   0 |             0 |                 0 |              0      |              0      |             0      |     0.3001 |     0.2998 |
| morroco_earthquake  | other    |        3 |                   0 |             0 |                 0 |              0      |              0      |             0      |     0.3898 |     0.3402 |
| libya_flood         | other    |        4 |                   0 |             0 |                 0 |              0      |              0      |             0      |     0.3386 |     0.3196 |
| shovi_landslide     | other    |        1 |                   0 |             0 |                 0 |              0      |              0      |             0      |     0.2495 |     0.2495 |
| sunda_tsunami       | xBD      |        4 |                   0 |            44 |               161 |              0      |              0      |             0      |     0.403  |     0.4082 |

Across the 13 xBD events with at least 10,000 destroyed pixels, the rank correlation between R12's extra destroyed buildings and its change in Destroyed IoU is +0.73 (p = 0.00).

## Files

- `E4_by_source.csv`, `E4_by_event.csv`, `config.json`