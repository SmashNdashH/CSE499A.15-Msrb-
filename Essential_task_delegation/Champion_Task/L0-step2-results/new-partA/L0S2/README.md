# L0 step 2: the new 7B at full size

Model `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP`, greedy decoding, the official DisasterM3 prompts, optical questions only, native images (1036 px for a 1024 px image). This session asked ['BDC', 'BBR']; complete tasks here: ['BBR', 'BDC'].

## Accuracy (%)

| task   |   questions |   answered |   new_full_% |   old_512_% |   old_full_L0S1_% |   paper_finetuned_% |   paper_base_% |
|:-------|------------:|-----------:|-------------:|------------:|------------------:|--------------------:|---------------:|
| DTR    |         353 |          0 |       nan    |       71.1  |             75.64 |                83.6 |           66.6 |
| BBR    |        2006 |       2006 |        19.99 |       19.54 |            nan    |                21.5 |            4.7 |
| BDC    |        3583 |       3583 |        30.31 |       32.01 |            nan    |                34.3 |           34.2 |
| DRE    |        1868 |          0 |       nan    |       32.87 |            nan    |                29.4 |           29.3 |
| DSR    |        2007 |          0 |       nan    |       37.87 |            nan    |                37.7 |           28.3 |
| ORR    |         858 |          0 |       nan    |       23.31 |             31.12 |                36.2 |           23.9 |

**Building counting at full size: 30.31% against the base model's 34.2%: the counting half of KR2 is still not met.**

## Notes

- `old_512_%`: H1 V2 and V2b (greedy, 512 px, the same optical questions). `old_full_L0S1_%`: L0 step 1 at full size.
- The paper's columns come from DisasterM3's Table 2 (optical questions), each model at Qwen's default image size.
- The keep rule (fixed 19 Oct) is applied on your PC, to all parts' answers together: the new model is kept only if disaster type and relational reasoning rise and no other task drops by more than 1 point against the old model at full size.

## Files

- `l0_step2_new_results.csv`, `L0_step2_new_per_question.csv`, `config.json`
- `responses/new/<subset>.jsonl`, with question IDs as in H1 V2 and L0 step 1