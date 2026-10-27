### Re-scoring Rule

For each row where `pred_option == "?"`:

1. Extract any explicit integer or word number from `response` (e.g., *"The number of damaged buildings is 6"* → 6).
2. Match this number against the entry's `options_list` (from `augmented_bdc_benchmark_v2.json`).
3. If it uniquely equals an option's numerical value, map `pred_option` to that option letter ('A'–'E') and set `correct = (mapped_option == gt_option)`.
4. If no number is found or the number does not match any option, it remains wrong (`pred_option = "?"`, `correct = False`).
