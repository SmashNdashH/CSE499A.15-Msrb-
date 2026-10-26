# H2b: graded test of the SQL agent

297 items: 261 answerable (29 questions, 3 phrasing(s), up to 3 images each) and 36 unanswerable. Table: H2's `buildings.sqlite` (117,687 buildings in 2,652 images).

## Decision

Pass = answerable test-suite accuracy of at least 90% and RS(10) above 0.

- our-7B nf4: test-suite accuracy 69.0% [57.1, 80.5], RS(10) -89.6, unanswerable refused 94.4%: FAIL
- VL-3B nf4: test-suite accuracy 46.4% [32.2, 60.9], RS(10) -365.7, unanswerable refused 36.1%: FAIL
- Coder-1.5B nf4: test-suite accuracy 56.7% [44.4, 69.3], RS(10) -283.8, unanswerable refused 25.0%: FAIL
- Coder-3B nf4: test-suite accuracy 72.0% [58.2, 84.7], RS(10) -162.6, unanswerable refused 80.6%: FAIL

- Benchmark counting questions use the fixed query, not a model, whatever happens below.
- **Current system:** our 7B in 4-bit does not meet the bar.
- **Final low-resource system (provisional):** the base 3B in 4-bit does not meet the bar.
- **Fallback:** no separate model passes either (or none was run): ship fixed queries only, no free-form questions.

## Accuracy and reliability

| config          |   ts_acc_% | ts_acc_ci95   |   ex_real_acc_% |   first_try_ts_acc_% |   unseen_acc_% |   all_phrasings_right_% |   wrong_% |   refused_% |   failed_% |   unanswerable_refused_% |   RS10 | RS10_ci95        | passes   |
|:----------------|-----------:|:--------------|----------------:|---------------------:|---------------:|------------------------:|----------:|------------:|-----------:|-------------------------:|-------:|:-----------------|:---------|
| our-7B nf4      |       69   | [57.1, 80.5]  |            74.3 |                 69   |           65.2 |                    46   |      17.6 |        13.4 |        0   |                     94.4 |  -89.6 | [-175.3, -6.0]   | False    |
| our-7B fp16     |       73.6 | [62.1, 84.3]  |            77.8 |                 73.6 |           70   |                    54   |      15.7 |        10.7 |        0   |                     94.4 |  -68.7 | [-141.4, -1.4]   | False    |
| VL-3B nf4       |       46.4 | [32.2, 60.9]  |            51.3 |                 46.4 |           33.8 |                    25.3 |      37.9 |        15.3 |        0.4 |                     36.1 | -365.7 | [-494.9, -245.4] | False    |
| VL-3B fp16      |       56.3 | [40.2, 71.6]  |            62.1 |                 56.3 |           46.4 |                    41.4 |      28.7 |        10.7 |        4.2 |                     52.8 | -253.9 | [-387.4, -141.2] | False    |
| Coder-1.5B nf4  |       56.7 | [44.4, 69.3]  |            62.8 |                 56.7 |           53.1 |                    24.1 |      28   |        15.3 |        0   |                     25   | -283.8 | [-392.1, -185.3] | False    |
| Coder-1.5B fp16 |       55.6 | [42.9, 68.6]  |            61.7 |                 55.6 |           51.7 |                    25.3 |      30.7 |        13.8 |        0   |                     30.6 | -301   | [-412.9, -203.0] | False    |
| Coder-3B nf4    |       72   | [58.2, 84.7]  |            75.9 |                 72   |           68.1 |                    55.2 |      24.1 |         3.8 |        0   |                     80.6 | -162.6 | [-280.4, -51.8]  | False    |
| Coder-3B fp16   |       73.9 | [60.9, 86.2]  |            77.8 |                 73.9 |           71.5 |                    58.6 |      22.2 |         3.8 |        0   |                     75   | -151.5 | [-273.3, -38.1]  | False    |

- `ts_acc_%`: test-suite accuracy on the answerable items: the final query matches the reference on all five versions of the table, with a 95% cluster-bootstrap interval.
- `ex_real_acc_%`: the same, on the real table alone. The gap to `ts_acc_%` is answers right by chance.
- `first_try_ts_acc_%`: before any retry. `unseen_acc_%`: question types no prompt example shows.
- `all_phrasings_right_%`: instances answered correctly in all three phrasings.
- `wrong_%`, `refused_%`, `failed_%`: the answerable items' outcomes (a query still failing after the retries counts as a refusal in RS).
- `RS10`: reliability score over all items: +1 right answer or refusal of an unanswerable question, 0 refusal of an answerable one, -10 wrong answer or answer to an unanswerable question.

## Memory and speed

| config          |   footprint_gb |   extra_memory_gb |   peak_gb_one_at_a_time |   latency_median_s |   latency_p90_s |   tokens_per_s |   batch_agreement |   batch_size_used |
|:----------------|---------------:|------------------:|------------------------:|-------------------:|----------------:|---------------:|------------------:|------------------:|
| our-7B nf4      |           5.78 |              0    |                    6.17 |               3.27 |            4.96 |           12.3 |             0.958 |                16 |
| our-7B fp16     |          16.58 |              0    |                   16.97 |               3.29 |            5.41 |           12   |             1     |                16 |
| VL-3B nf4       |           2.35 |              0    |                    2.55 |               3.23 |            4.56 |           12.7 |             1     |                16 |
| VL-3B fp16      |           7.51 |              0    |                    7.65 |               2.22 |            3.3  |           16.9 |             1     |                16 |
| Coder-1.5B nf4  |           1.12 |              1.12 |                    1.26 |               2.31 |            3.91 |           15.3 |             1     |                16 |
| Coder-1.5B fp16 |           3.09 |              3.09 |                    3.2  |               1.73 |            2.42 |           24   |             1     |                16 |
| Coder-3B nf4    |           2.01 |              2.01 |                    2.2  |               2.97 |            4.31 |           13.4 |             1     |                16 |
| Coder-3B fp16   |           6.17 |              6.17 |                    6.31 |               2.33 |            3.49 |           17   |             1     |                16 |

- `extra_memory_gb`: what the model adds to the deployed system (a VLM the system already runs adds nothing).
- Latency and peak memory: one question at a time, as deployed, on a T4.

## By question group (test-suite accuracy, %)

| group               |   items |   our-7B nf4 |   our-7B fp16 |   VL-3B nf4 |   VL-3B fp16 |   Coder-1.5B nf4 |   Coder-1.5B fp16 |   Coder-3B nf4 |   Coder-3B fp16 |
|:--------------------|--------:|-------------:|--------------:|------------:|-------------:|-----------------:|------------------:|---------------:|----------------:|
| area                |      36 |         75   |          83.3 |        61.1 |        100   |             83.3 |              83.3 |           91.7 |            91.7 |
| combined            |      18 |         61.1 |          55.6 |        33.3 |         33.3 |             33.3 |              33.3 |           83.3 |            83.3 |
| count               |      27 |        100   |         100   |        88.9 |         88.9 |             77.8 |              74.1 |          100   |           100   |
| distance            |      27 |         44.4 |          51.9 |        70.4 |         70.4 |             44.4 |              48.1 |           51.9 |            48.1 |
| no destroyed nearby |       9 |         22.2 |          22.2 |         0   |          0   |              0   |               0   |           22.2 |            22.2 |
| pairwise            |      18 |         16.7 |          27.8 |        27.8 |         27.8 |             11.1 |              16.7 |            5.6 |             5.6 |
| percentage          |      18 |         94.4 |         100   |        83.3 |         83.3 |             83.3 |              83.3 |          100   |           100   |
| position            |      27 |         40.7 |          51.9 |         0   |          0   |             22.2 |              14.8 |           37   |            44.4 |
| ranking             |      27 |         96.3 |          92.6 |         7.4 |         25.9 |             74.1 |              81.5 |           88.9 |            88.9 |
| size threshold      |      18 |         88.9 |         100   |        72.2 |         88.9 |             83.3 |              83.3 |           83.3 |           100   |
| total               |      18 |         83.3 |          94.4 |        50   |         55.6 |             50   |              55.6 |           77.8 |            83.3 |
| yes/no              |      18 |         72.2 |          66.7 |        33.3 |         50   |             66.7 |              38.9 |           83.3 |            83.3 |

## By phrasing (%)

|                 |   our-7B nf4 |   our-7B fp16 |   VL-3B nf4 |   VL-3B fp16 |   Coder-1.5B nf4 |   Coder-1.5B fp16 |   Coder-3B nf4 |   Coder-3B fp16 |
|:----------------|-------------:|--------------:|------------:|-------------:|-----------------:|------------------:|---------------:|----------------:|
| plain           |         74.7 |          81.6 |        55.2 |         65.5 |             73.6 |              72.4 |           75.9 |            74.7 |
| responder words |         52.9 |          58.6 |        35.6 |         46   |             31   |              31   |           63.2 |            70.1 |
| indirect        |         79.3 |          80.5 |        48.3 |         57.5 |             65.5 |              63.2 |           77   |            77   |

## Unanswerable questions refused (%)

| qid   |   our-7B nf4 |   our-7B fp16 |   VL-3B nf4 |   VL-3B fp16 |   Coder-1.5B nf4 |   Coder-1.5B fp16 |   Coder-3B nf4 |   Coder-3B fp16 |
|:------|-------------:|--------------:|------------:|-------------:|-----------------:|------------------:|---------------:|----------------:|
| U1    |        100   |         100   |       100   |        100   |             33.3 |              66.7 |          100   |           100   |
| U10   |        100   |         100   |        66.7 |        100   |              0   |               0   |          100   |           100   |
| U11   |        100   |         100   |         0   |          0   |              0   |              33.3 |           66.7 |            33.3 |
| U12   |        100   |         100   |         0   |         33.3 |              0   |               0   |          100   |           100   |
| U2    |         33.3 |          66.7 |         0   |          0   |             33.3 |              33.3 |            0   |             0   |
| U3    |        100   |         100   |       100   |        100   |             33.3 |              33.3 |          100   |           100   |
| U4    |        100   |         100   |        66.7 |        100   |              0   |               0   |          100   |           100   |
| U5    |        100   |         100   |         0   |          0   |              0   |               0   |           33.3 |            33.3 |
| U6    |        100   |         100   |         0   |          0   |             33.3 |              33.3 |           66.7 |            66.7 |
| U7    |        100   |         100   |       100   |        100   |            100   |             100   |          100   |           100   |
| U8    |        100   |          66.7 |         0   |          0   |             33.3 |              33.3 |          100   |            66.7 |
| U9    |        100   |         100   |         0   |        100   |             33.3 |              33.3 |          100   |           100   |

## Differences between configurations (paired cluster bootstrap)

| a              | b               |   ts_acc_a_% |   ts_acc_b_% |   difference_pts | ci95          | clear_difference   |
|:---------------|:----------------|-------------:|-------------:|-----------------:|:--------------|:-------------------|
| our-7B nf4     | our-7B fp16     |         69   |         73.6 |             -4.6 | [-8.8, -0.8]  | True               |
| VL-3B nf4      | VL-3B fp16      |         46.4 |         56.3 |             -9.9 | [-17.6, -3.4] | True               |
| Coder-1.5B nf4 | Coder-1.5B fp16 |         56.7 |         55.6 |              1.1 | [-2.3, 6.1]   | False              |
| Coder-3B nf4   | Coder-3B fp16   |         72   |         73.9 |             -1.9 | [-5.7, 1.5]   | False              |
| our-7B nf4     | Coder-3B nf4    |         69   |         72   |             -3   | [-10.7, 3.8]  | False              |
| VL-3B nf4      | Coder-1.5B nf4  |         46.4 |         56.7 |            -10.3 | [-23.8, 1.5]  | False              |
| Coder-1.5B nf4 | Coder-3B nf4    |         56.7 |         72   |            -15.3 | [-24.5, -7.3] | True               |
| VL-3B nf4      | our-7B nf4      |         46.4 |         69   |            -22.6 | [-36.8, -9.2] | True               |

## Notes

- The question bank, the prompt (`prompt.txt`) and the decision rule were fixed before any run.
- Every model decodes greedily with no repetition penalty.
- Batched replies matched one-at-a-time replies on 96% to 100% of the check items; a configuration below 90% would have been decoded one at a time.
- Versions of the table left out as ambiguous (ties, thresholds, midlines): 1.
- The base VL-3B stands in for the distilled 3B, which doesn't exist yet: re-run on it after L2.

## References

Zhong, Yu & Klein, EMNLP 2020 (test-suite accuracy); Lee et al. 2024, TrustSQL, and the EHRSQL 2024 shared task (reliability score); Chang et al., ICLR 2023, Dr.Spider (paraphrase robustness); Gao et al., VLDB 2024, DAIL-SQL, and Rajkumar et al. 2022 (prompt representation); Chen et al., ICLR 2024, Self-Debugging (retries with execution feedback); Hui et al. 2024, Qwen2.5-Coder.

## Files

`summary.csv`, `by_group.csv`, `by_phrasing.csv`, `by_unanswerable.csv`, `pairwise.csv`, `graded.csv` (one row per configuration and item), `items.csv`, `gold.json`, `answers.jsonl` (every reply and attempt), `runs.jsonl`, `prompt.txt`, `decision.md`, `config.json`