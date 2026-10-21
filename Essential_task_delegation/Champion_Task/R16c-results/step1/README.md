# R16c step 1: R16's ceiling and a counting check

From saved outputs only (R16, R16b, E3, E1), on the 645 validation images. Nothing is tuned on them.

## 1. The ceiling (pixel scores, all 645 images)

A perfect grader gives each stage-1 building its main true grade; with rejects it may also call a building background. The gap between a real grader and the perfect one on the same outlines is what a better grader can win.

```
                                         map   miou  intact_iou  damaged_iou  destroyed_iou  building_iou
                    R11 with TTA (reference) 0.5875      0.6821       0.3956         0.2965        0.7420
                      SAM 3 only, R16 grades 0.4711      0.4713       0.2772         0.1672        0.6304
              SAM 3 + R11 pieces, R16 grades 0.4941      0.5690       0.2800         0.1522        0.7426
                     SAM 3 only, R16b grades 0.5150      0.5450       0.3186         0.2280        0.6304
             SAM 3 + R11 pieces, R16b grades 0.5407      0.6441       0.3356         0.2080        0.7426
                  SAM 3 only, perfect grades 0.7060      0.6243       0.6755         0.5555        0.6304
          SAM 3 + R11 pieces, perfect grades 0.7657      0.7382       0.7359         0.6136        0.7426
        SAM 3 only, perfect + rejects grades 0.7164      0.6414       0.6793         0.5742        0.6461
SAM 3 + R11 pieces, perfect + rejects grades 0.7981      0.7888       0.7616         0.6606        0.7907
                   vote with R11, R16 grades 0.5796      0.6846       0.3975         0.2609        0.7426
                  vote with R11, R16b grades 0.5846      0.6864       0.4033         0.2737        0.7426
```

Per source (xBD and other) in `R16c_ceiling.csv`.

## 2. Counting

The 1,055 training-split counting questions on the validation images. DisasterM3's answer is the number of 8-connected blobs of the class; each method counts blobs in its own map and picks the nearest option. The true masks answer 100%.

```
                                      method  all_%  intact_%  damaged_%  destroyed_%  xBD_%  other_%  median_count_error_%
                                       truth 100.00    100.00     100.00       100.00 100.00   100.00                   0.0
                             R11, every blob  37.16     36.76      37.32        37.96  39.50    21.99                  33.3
                R11, blobs of at least 20 px  39.24     38.02      43.66        36.57  42.67    17.02                  33.3
                      SAM 3 only, R16 grades  30.14     29.55      33.10        27.78  31.51    21.28                  50.0
              SAM 3 + R11 pieces, R16 grades  31.37     34.41      37.68        15.28  32.60    23.40                  40.0
                     SAM 3 only, R16b grades  36.40     36.76      34.86        37.50  38.18    24.82                  36.7
             SAM 3 + R11 pieces, R16b grades  34.41     38.56      34.51        23.61  37.31    15.60                  33.3
                  SAM 3 only, perfect grades  47.39     41.80      57.39        48.61  47.70    45.39                  22.7
          SAM 3 + R11 pieces, perfect grades  49.57     38.92      65.49        56.02  50.98    40.43                  19.0
        SAM 3 only, perfect + rejects grades  45.31     38.20      57.39        47.69  45.51    43.97                  24.6
SAM 3 + R11 pieces, perfect + rejects grades  55.26     49.55      65.14        56.94  56.78    45.39                  13.9
                   vote with R11, R16 grades  36.87     35.50      41.90        33.80  39.82    17.73                  33.3
                  vote with R11, R16b grades  36.87     35.68      41.20        34.26  40.15    15.60                  33.3
```

**Finding:** Best R16 map for counting: 'vote with R11, R16 grades', 36.9% of the 1,055 validation questions, against 39.2% for R11's blobs (-2.4 points) and 55.3% with a perfect grader on the same outlines.

**For scale** (other images, so only a guide; validation numbers run high: R9's untuned counter scored about 33% here but its tuned counter 26.5% on the benchmark's optical questions):

- our 7B, greedy, benchmark optical questions (H1 V2b): 32.01%
- base Qwen2.5-VL-7B, benchmark optical questions (paper): 34.2%
- R9's tuned counter, benchmark optical questions (H1 V1): 26.54%
- R9's untuned counter, these validation questions (H1 V1): 33.3%

Our 7B can't be scored here: it trained on these images.

## Files

- `R16c_ceiling.csv`: pixel scores of every map, overall and per source
- `R16c_counting.csv` and `R16c_counting_per_question.csv`: the counting check
- `config.json`