# Presentation & Defense Talking Points: mIoU vs. Recall in Disaster Response

**Context:** When asked by a supervisor or during the final presentation why certain architectural choices were made (e.g., U-Net++ having lower mIoU but higher Recall) or how you evaluate success, use these talking points.

## 1. The mIoU Priority (The Academic Standard)
**Talking Point:** *“We still use mIoU as our 'official' grading metric to prove our model is scientifically sound and to compare against academic baselines. As seen in our ablation study, we successfully pushed our peak mIoU above 0.52 to ensure the model isn't just wildly guessing and has strong spatial awareness.”*

## 2. The Operational Reality (Why Recall Matters Most)
**Talking Point:** *“However, because our capstone deals specifically with Disaster Response, the operational stakes change our metric priority. In a real-world earthquake or hurricane, this AI would be used to deploy rescue teams.”*

*   **The Cost of Low mIoU (False Positives):** *“If our model has sloppy borders and accidentally flags an intact building as 'destroyed', we might accidentally send a rescue team to a building that is actually fine. The cost of this error is a few wasted minutes.”*
*   **The Cost of Low Recall (False Negatives):** *“If our model completely misses a building, we won't send a rescue team to a destroyed structure where people might be trapped. The cost of this error is that people could lose their lives.”*

## 3. Real-World Application (The Loss-Function Trade-off)
> **Correction:** an earlier version of this point said U-Net++ raised recall from 64% to 79%. That came from a wrong baseline row in the summary CSV. The real baseline macro recall is 79.1%, the same as U-Net++. Do not use it. See [Ablation_Study_Reference_and_QA_Prep.md](Ablation_Study_Reference_and_QA_Prep.md) §8.

**Talking Point:** *“The trade-off shows up in our loss-function ablation. The baseline's class-weighted Cross-Entropy over-predicts rare classes: it finds 66% of destroyed-building pixels, but only about 1 in 5 of its 'destroyed' predictions is correct (precision 0.18). Focal Tversky raised mIoU from 0.46 to 0.53 and made predictions far more precise, but it finds fewer destroyed pixels. For deployment we report Destroyed-class recall and F2 alongside mIoU. If responders need more recall, we can lower the Destroyed decision threshold at inference without retraining.”*

*Always say which recall: the CSV 'recall' column is a macro average over 4 classes, Background included, not 'share of destroyed buildings found'.*

## 4. Final D4 results: the sampler trades a little Damaged IoU for Destroyed recall (18 October 2026)
**Talking Point:** *"In our final controlled runs, every recipe passed the mIoU target. The weighted sampler with copy-paste was the best for destroyed buildings. Averaged over two seeds, it finds 50% of destroyed-building pixels against 48% for our reference recipe, or 52% against 49% with test-time augmentation. It also raises Destroyed IoU to 0.297 with TTA. The price is a slightly lower Damaged IoU (0.349 against 0.357), so mIoU stays flat. For rescue we prefer the sampler model, because a missed destroyed building costs more than a less precise damaged one. Destroyed IoU still misses 0.30 by 0.003, which is why we moved to structural changes rather than more tuning."*

*These recall figures are the Destroyed-class recall (`destroyed_recall` in the CSVs), not the macro recall. Source: [Ablation_Study_Reference_and_QA_Prep.md](Ablation_Study_Reference_and_QA_Prep.md) §7.*
