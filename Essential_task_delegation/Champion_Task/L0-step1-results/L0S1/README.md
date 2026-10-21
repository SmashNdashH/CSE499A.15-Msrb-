# L0 step 1: our 7B at three image sizes

Model `AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP`, greedy decoding, the official DisasterM3 prompts, optical questions only (DTR 353, ORR 858). `512` is H1 V2's setting, re-run here as the control.

## Accuracy (%)

| task   |   questions |   512_% |   512_answered |   train_% |   train_answered |   full_% |   full_answered |   h1v2_512_% |   paper_finetuned_% |   paper_base_% |
|:-------|------------:|--------:|---------------:|----------:|-----------------:|---------:|----------------:|-------------:|--------------------:|---------------:|
| DTR    |         353 |   71.1  |            353 |     75.35 |              353 |    75.64 |             353 |         71.1 |                83.6 |           66.6 |
| ORR    |         858 |   23.31 |            858 |     26.92 |              858 |    31.12 |             858 |         23.3 |                36.2 |           23.9 |

**Verdict:** Gap not closed: the best setting is 'full' (DTR 75.6% (36% of the gap), ORR 31.1% (61% of the gap)). The image size isn't the main cause; next is step 2, a top-up training of the adapter.

## Settings

- `512`: resized to 512 x 512 as in H1 V2; Qwen's rounding then gives 504 px; 418 image tokens per question as fed
- `train`: our training cap (min_pixels 256 x 28 x 28, max_pixels 512 x 28 x 28): 616 px; 625 image tokens per question as fed
- `full`: the native image at Qwen2.5-VL's default, as in the paper: 1036 px; 1768 image tokens per question as fed

## Notes

- The references come from DisasterM3's Table 2 (optical questions): the base Qwen2.5-VL-7B and the authors' fine-tune of it, both run at Qwen's default image size.
- The decision rule was fixed before the run (first cell): within 3 points of the paper on both tasks closes the gap; half of each gap recovered closes it partly.

## Files

- `l0_step1_results.csv`, `L0_step1_per_question.csv`, `config.json`
- `responses/<setting>/<subset>.jsonl`, with question IDs as in H1 V2