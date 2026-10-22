# H2: per-building table and SQL counting agent

R11 8-view TTA mask (after- and before-image) on 2,652 benchmark after-images, 117,687 buildings (blobs of at least 20 px) in `buildings.sqlite`. 100 BDC questions (Optical, seed 42, split evenly over the classes).

## Result

| llm                                       |   questions |   sql_valid_% |   retried_% |   count_equals_table_% |   H2_accuracy_% |   oracle_accuracy_% |   V1_accuracy_% |
|:------------------------------------------|------------:|--------------:|------------:|-----------------------:|----------------:|--------------------:|----------------:|
| Qwen/Qwen2.5-Coder-3B-Instruct            |         100 |           100 |           0 |                     99 |              34 |                  34 |              28 |
| AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP |         100 |           100 |           0 |                    100 |              34 |                  34 |              28 |

- `sql_valid_%`: the query ran and returned one number (after up to 2 retries).
- `count_equals_table_%`: the agent's number equals a hand-written `COUNT(*)` for that image and class.
- `oracle_accuracy_%`: the hand-written query, i.e. the best any agent can do on this table.
- `V1_accuracy_%`: H1 V1 (R9's tuned counter) on the same questions.

## R11's counting on every BDC question, against H1 V1

|   questions_in_table |   oracle_accuracy_% |   oracle_accuracy_optical_% |   same_count_as_V1_% |   V1_accuracy_on_same_% |   V1_accuracy_optical_% |
|---------------------:|--------------------:|----------------------------:|---------------------:|------------------------:|------------------------:|
|                 4978 |               27.46 |                       32.96 |                22.98 |                   22.24 |                   26.51 |

- `oracle_accuracy_%`: R11's own counting on the benchmark (first measurement). Base Qwen2.5-VL-7B: 34.2% on the optical questions (paper).
- `same_count_as_V1_%`: for information only. V1 counted R9's standard mask, so the counts are meant to differ.

## Per class

| llm                                       | class     |   questions |   sql_valid_% |   retried_% |   count_equals_table_% |   H2_accuracy_% |   oracle_accuracy_% |   V1_accuracy_% |
|:------------------------------------------|:----------|------------:|--------------:|------------:|-----------------------:|----------------:|--------------------:|----------------:|
| Qwen/Qwen2.5-Coder-3B-Instruct            | intact    |          34 |           100 |           0 |                    100 |            23.5 |                23.5 |            23.5 |
| Qwen/Qwen2.5-Coder-3B-Instruct            | damaged   |          33 |           100 |           0 |                     97 |            36.4 |                36.4 |            30.3 |
| Qwen/Qwen2.5-Coder-3B-Instruct            | destroyed |          33 |           100 |           0 |                    100 |            42.4 |                42.4 |            30.3 |
| AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP | intact    |          34 |           100 |           0 |                    100 |            23.5 |                23.5 |            23.5 |
| AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP | damaged   |          33 |           100 |           0 |                    100 |            36.4 |                36.4 |            30.3 |
| AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP | destroyed |          33 |           100 |           0 |                    100 |            42.4 |                42.4 |            30.3 |

## Notes

- The table holds R11's TTA blobs, so H2 cannot beat the oracle: it makes counts queryable, not more accurate.
- The 20 px minimum is R16's fixed rule (best on the 1,055 validation counting questions, R16c step 1), not tuned on the benchmark.
- Distances are centre-to-centre, in pixels (DisasterM3 gives no ground resolution in our files).
- The agent saw the question text and its category line, the image name, and no options.

## Files

- `buildings.sqlite` (tables `buildings`, `images`), `agent_eval.csv` (one row per LLM and question), `agent_summary.csv`, `table_check.csv`, `demo_queries.csv`, `config.json`, `agent_eval_partial.jsonl`