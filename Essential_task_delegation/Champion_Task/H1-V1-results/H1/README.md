# H1 V1: U-Net counting bypass for BDC

R9 (8-view TTA where tuned), 4,982 BDC questions. DisasterM3's BDC answers are 8-connected blob counts of its class masks, so the counter counts blobs in R9's mask and picks the nearest option.

## Result

| group               |   questions |   accuracy_% |   median_count_error_% |
|:--------------------|------------:|-------------:|-----------------------:|
| all                 |        4982 |        22.26 |                   69.2 |
| class intact        |        2671 |        16.66 |                   58.3 |
| class damaged       |        1345 |        27.36 |                  100   |
| class destroyed     |         966 |        30.64 |                   89.6 |
| Optical after-image |        3583 |        26.54 |                   54.5 |
| SAR after-image     |        1399 |        11.29 |                  100   |
| Optical, intact     |        1880 |        21.33 |                   45   |
| SAR, intact         |         791 |         5.56 |                  100   |
| Optical, damaged    |         974 |        31.62 |                   75   |
| SAR, damaged        |         371 |        16.17 |                  100   |
| Optical, destroyed  |         729 |        33.2  |                   83.3 |
| SAR, destroyed      |         237 |        22.78 |                  100   |

**Gate G4:** V1 BDC accuracy 22.26% (optical after-images 26.54%) against the base model's 34.2% and our fine-tuned 7B's 23.58%: the counting half of KR2 is not met.

References: base Qwen2.5-VL-7B 34.2%, our fine-tuned 7B 23.58%, the old 499A hybrid 13.11%.

## Counter tuned on the validation images

Ground-truth masks answer 100.0% of the 1055 tuning questions.

| class     | variant   |   closing |   min_area |   questions |   accuracy |
|:----------|:----------|----------:|-----------:|------------:|-----------:|
| intact    | std       |         0 |         24 |         555 |     0.3135 |
| damaged   | std       |         0 |          0 |         284 |     0.4014 |
| destroyed | std       |         0 |          0 |         216 |     0.4213 |

## Notes

- SAR questions are answered from the optical twin of the same tile; their truth can differ from the optical twin's, so they are reported separately.
- R9 never trained on benchmark images. A model trained on D5's benchmark tiles (R12 with all 1,007 tiles) must not be scored here without excluding those 741 questions.

## Files

- `H1_v1_summary.csv`, `H1_v1_tuning.csv`, `H1_v1_per_question.csv`, `config.json`
- `results/building_damage_counting/R9-unet-counting/finished.jsonl` (for `4_score_results.py`)
- `facts.json`: per after-image counts, where the damage is, and the fact line for V2
- `masks/<name>.png`: R9 TTA masks (0 background, 1 intact, 2 damaged, 3 destroyed)