# H1 V2: counts as text, all six tasks

Model `AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP`, greedy decoding, images 512 x 512, official DisasterM3 prompts. `plain` is the official prompt; `facts` adds H1 V1's fact line before the question. BDC comes from H1 V1 (U-Net counting).

## Six-task table

| task   |   questions |   plain_% |   facts_% |   facts_minus_plain | complete   | note                                               |   kr2_system_% |   ref_499A_finetuned_% |   ref_base_% |
|:-------|------------:|----------:|----------:|--------------------:|:-----------|:---------------------------------------------------|---------------:|-----------------------:|-------------:|
| DTR    |         420 |     70.71 |     69.29 |               -1.42 | True       |                                                    |          70.71 |                  68.33 |         66.6 |
| BBR    |        2363 |     20.52 |     10.66 |               -9.86 | True       |                                                    |          20.52 |                  16.39 |          4.7 |
| BDC    |        4982 |    nan    |    nan    |              nan    | True       | answered by H1 V1 (U-Net counting), not by the VLM |          22.26 |                  23.58 |         34.2 |
| DRE    |        2178 |     31.86 |     30.07 |               -1.79 | True       |                                                    |          31.86 |                  26.24 |         29.3 |
| DSR    |        2007 |     37.87 |     36.42 |               -1.45 | True       |                                                    |          37.87 |                  31.31 |         28.3 |
| ORR    |        1018 |     22.1  |    nan    |              nan    | True       | plain only: crop images without U-Net facts        |          22.1  |                  22.5  |         23.9 |

**Gate:** Facts not kept: the largest change against the plain run is -9.86 points (rule: no task may drop by more than 1 point).

Six-task mean: KR2 system 34.22%, 499A fine-tuned 7B 31.39%, base model 31.17%.

## By sensor of the after-image

| task   | sensor   |   questions |   plain_pct |   facts_pct |
|:-------|:---------|------------:|------------:|------------:|
| BBR    | Optical  |        2006 |       19.54 |       10.17 |
| BBR    | SAR      |         357 |       26.05 |       13.45 |
| DRE    | Optical  |        1868 |       32.87 |       31    |
| DRE    | SAR      |         310 |       25.81 |       24.52 |
| DSR    | Optical  |        2007 |       37.87 |       36.42 |
| DTR    | Optical  |         353 |       71.1  |       69.41 |
| DTR    | SAR      |          67 |       68.66 |       68.66 |
| ORR    | Optical  |         858 |       23.31 |      nan    |
| ORR    | SAR      |         160 |       15.62 |      nan    |

## Notes

- The 499A references were most likely sampled at temperature 1.0 (a re-run gave Disaster Type 64.05% instead of 68.33%), so compare `facts` with this run's `plain`, not with the references.
- SAR questions get the fact line of the optical twin of the same tile.

## Files

- `kr2_results.csv`, `H1_v2_per_question.csv`, `config.json`
- `responses/<variant>/<subset>.jsonl` and `results/<subset>/<tag>/finished.jsonl` (for `4_score_results.py`)