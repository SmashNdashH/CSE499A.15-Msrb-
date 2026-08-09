# DisasterVQA Benchmark Comparison

This table compares the zero-shot performance of **Abrar's Fine-Tuned Qwen2.5-VL-7B** against the 7 baseline models evaluated in the DisasterVQA research paper (Al-Mohannadi et al., 2026). Our evaluation uses a **500-sample exact-match automated check**, explicitly excluding open-ended questions which require human or LLM judging.

| Model Architecture | Binary Accuracy (Yes/No) | Multiple-Choice (Acc) | Open-Ended (Acc) |
| :--- | :---: | :---: | :---: |
| **AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP (Ours)** | **0.91** | **0.85** | **N/A (Excluded)** |
| GPT-4.1-mini (Proprietary) | 0.91 | 0.85 | 0.83 |
| Mistral-Small-3.1-24B | 0.90 | 0.81 | 0.75 |
| Qwen-2.5-VL-32B | 0.89 | 0.81 | 0.79 |
| Molmo-7B-D | 0.88 | 0.74 | 0.70 |
| GPT-4o-mini (Proprietary) | 0.87 | 0.78 | 0.78 |
| LLaMA-3.2-11B | 0.87 | 0.80 | 0.75 |
| Pixtral-12B | 0.86 | 0.77 | 0.76 |

> **Note on Metrics:** The baseline scores are extracted directly from Figure 4 of the *DisasterVQA* paper. Binary and Open-Ended formats use Accuracy, while Multiple-Choice uses F1-score in the original paper (we report automated string-match accuracy here for MCQ).