# H1 V2b: building counting asked to our 7B

Model `AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP`, greedy decoding, images 512 x 512, the official single-choice prompt. `plain` is the official prompt; `facts` adds H1 V1's fact line before the question.

## Results

| answerer               |   bdc_% |
|:-----------------------|--------:|
| U-Net counter (H1 V1)  |   22.26 |
| 7B plain (greedy)      |   30.51 |
| 7B with facts (greedy) |   26.6  |
| Our 7B, 499A (sampled) |   23.58 |
| Base Qwen2.5-VL-7B     |   34.2  |

**Best BDC answer: 7B plain (greedy), 30.51%. KR2 six-task mean with it: 35.6% (base model 31.17%). The counting half of KR2 is not met: 3.69 points below the base model's 34.2%.**

## By class and sensor

| group               |   questions |   plain_% |   facts_% |   unet_v1_% |
|:--------------------|------------:|----------:|----------:|------------:|
| all                 |        4982 |     30.51 |     26.6  |       22.26 |
| class intact        |        2671 |     33.4  |     24.63 |       16.66 |
| class damaged       |        1345 |     26.32 |     27.58 |       27.36 |
| class destroyed     |         966 |     28.36 |     30.64 |       30.64 |
| Optical after-image |        3583 |     32.01 |     28.1  |       26.54 |
| SAR after-image     |        1399 |     26.66 |     22.73 |       11.29 |

## Notes

- The six-task mean uses H1 V2's plain scores for disaster type, bearing bodies, road damage, land use and relational reasoning, and the best BDC answer here.
- SAR questions get the fact line of the optical twin of the same tile.

## Files

- `bdc_results.csv`, `bdc_by_group.csv`, `H1_v2b_per_question.csv`, `config.json`
- `responses/<variant>/building_damage_counting.jsonl` and `results/building_damage_counting/<tag>/finished.jsonl` (for `4_score_results.py`)