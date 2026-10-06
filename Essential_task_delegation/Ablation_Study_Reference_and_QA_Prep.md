# Ablation Study Reference & Supervisor Q&A Prep

**Purpose:** One place to look up what each member did, what the results *actually* were, and how to answer supervisor questions. Every number here was checked against the **saved outputs of the member notebooks**, not only against `all_members_ablation_summary.csv`. That CSV contains several errors, listed in [§8 Errata](#8-errata--do-not-quote-these-numbers).

**Champion work (Task D4):** [`Champion_Task/D4_champion_model.ipynb`](Champion_Task/D4_champion_model.ipynb). See [§7](#7-pre-d4-ablations-and-the-champion-model).

---

## 1. Baseline configuration — `segmentation-model (1).ipynb`
This is the pre-study starting notebook that every ablation was forked from. It was **not re-run** during the study. The plan refers to its result as **B1**, **C1** and **D1a** ("Already Done"), so all three names mean this one run.

| Parameter | Value | One-line meaning |
|---|---|---|
| Architecture | `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet")` | U-shaped encoder–decoder; ImageNet-pretrained ResNet-34 encoder |
| Loss | 0.5 × class-weighted CE + 0.5 × Dice | CE grades each pixel; Dice grades shape overlap. Destroyed weight ≈ 283× Background |
| Resolution | `A.Resize(512, 512)` from 1024 | Shrinks the whole image, so small buildings lose 75% of their pixels |
| Batch size | 16 | Images per weight update |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-5 | Adaptive step size + mild regularisation |
| Scheduler | CosineAnnealingLR (T_max = 40) | Learning rate decays smoothly to ~0 |
| Epochs | 40 | Full passes over the 5,798 training images |
| Split | 90 / 10, `random_state=42` → 5,798 train / 645 val | Same split in every experiment |
| AMP | FP16 mixed precision | About half the memory, faster |
| Hardware | Kaggle T4, 16 GB | The VRAM limit drives batch-size choices |
| **Result** | **mIoU 0.4619** — BG .9255, Intact .4926, Damaged .2636, Destroyed .1658 | Macro P .5015 / **R .7906** / F1 .5808 |

Class imbalance: Background 93.0%, Intact 5.9%, Damaged 1.05%, **Destroyed 0.33%**, so Destroyed is 283× rarer than Background.

## 2. Loss vs. metric
- **The loss is how the model studies.** It must be smooth so gradients can say "move this weight a little this way".
- **mIoU is the final grade.** It is computed from hard argmax decisions. A pixel flipping from 49% to 51% changes the score in a jump, so mIoU has no useful gradient and cannot be trained on directly.
- Losses like Lovász-Softmax are smooth *stand-ins* for IoU. That is why we use them near the end of training.

## 3. The one-variable rule
"Change exactly one variable at a time" (`team_task_assignments.md`). If two changes are combined and the score rises, you cannot tell which one helped, or whether one helped while the other hurt.
- This is why Aryan's sampler (A2) and copy-paste (A3) were tested separately.
- Combining comes later, in D4.
- **Caveat found in the audit:** B2–B5 also changed batch size (8 vs 16), learning rate (5e-4 vs 1e-4) and weight decay (1e-4 vs 1e-5). That's four variables changed at once, not one. See §7c.

---

## 4. Per-member results (notebook-verified)

> **Comparability warning.** A runs are 10 epochs. B/C runs evaluate a resized 512 image. D runs evaluate a *random* 768 or 512 crop. These groups are **not directly comparable**. The D4 notebook re-evaluates everything under one protocol.

### Member A — Aryan: Data imbalance (10-epoch screening runs)
**Task:** make rare classes appear more often in training.
- A2 uses a `WeightedRandomSampler`: an image containing Destroyed is picked 10× as often, one with Damaged 5×, others 1×.
- A3 uses copy-paste augmentation: Damaged/Destroyed building cut-outs pasted onto training images (p = 0.5, 1–3 per image).

| Run | mIoU | Damaged IoU | Destroyed IoU | Note |
|---|---|---|---|---|
| A1 uniform shuffle | 0.3978 | 0.1831 | 0.0898 | 10-epoch reference |
| A2 weighted sampler | **0.4051** | 0.1783 | **0.1017** | Destroyed ↑, Damaged ↓ |
| A3 copy-paste (CSV) | ~~0.4077~~ | ~~0.1906~~ | ~~0.1026~~ | **Leaky run.** The bank (44,969 / 18,125 instances) included validation images |
| A3 copy-paste (notebook, leak-free) | 0.4017 | 0.1783 | 0.0949 | Bank from train only (40,704 / 16,682 instances) |

**Conclusion:**
- The sampler gave the clearest gain for Destroyed.
- Leak-free copy-paste is roughly neutral.
- Copy-paste also had a **scale bug**: cut-outs were taken at native 1024 scale but pasted onto images resized to 512, so pasted buildings were 2× too big. D4 run R2 re-tests it with both problems fixed.

### Member B — Tamanna: Architecture (40 epochs, CE + Dice, ResNet-34, Resize 512)

| Run | mIoU | Damaged IoU | Destroyed IoU | Macro recall | Destroyed recall | Batch |
|---|---|---|---|---|---|---|
| B1 U-Net (baseline) | **0.4619** | **0.2636** | **0.1658** | 0.7906 | 0.6613 | 16 |
| B2 U-Net++ | 0.4405 | 0.2458 | 0.1307 | 0.7866 | 0.6679 | 8* |
| B3 U-Net++ + scSE | 0.4367 | 0.2311 | 0.1252 | 0.7924 | 0.6791 | 8 |
| B4 DeepLabV3+ | 0.4052 | 0.1935 | 0.0923 | **0.7940** | **0.7610** | 8 |
| B5† MA-Net | 0.2314 | 0 | 0 | 0.2500 | 0 | 8 |

†Labelled B5 by the team. The plan calls MA-Net **B6**, and its B5 (U-Net + ResNet-50, CE + Dice) was never run.
\*Batch sizes are taken from the notebooks' config cells. The B2–B5 CSV notes say "batch 16", but every one of those notebooks sets `BATCH_SIZE = 8`.

**Conclusion:**
- The plain U-Net wins on every IoU.
- **U-Net++ has no meaningful recall advantage.** Earlier claims of "64% → 79%" used a wrong B1 row; see Errata #1.
- DeepLabV3+ has the highest Destroyed recall, but with very low precision (.095).
- MA-Net collapsed to all-background under CE + Dice.
- Open question: do these architectures behave differently under a better loss? D4 run R4 tests U-Net++.

### Member C — Noman: Loss function & backbone (40 epochs, Resize 512)

| Run | mIoU | Damaged IoU | Destroyed IoU | Macro P | Macro R | Train h |
|---|---|---|---|---|---|---|
| B1 CE + Dice (reference) | 0.4619 | 0.2636 | 0.1658 | 0.5015 | 0.7906 | — |
| C2 Focal Tversky α.3 β.7 γ2.0 | 0.5181 | 0.3306 | 0.2568 | 0.6018 | 0.7012 | 2.43 |
| C3 Focal Tversky α.25 β.75 γ1.33 | 0.5203 | 0.3333 | **0.2727** | 0.6053 | 0.7076 | 2.43 |
| C4 Phased: 32 ep FT → 8 ep (0.6 FT + 0.4 Lovász) | 0.5233 | 0.3326 | 0.2722 | 0.6264 | 0.6795 | 2.45 |
| **C5 ResNet-50 + Phased** | **0.5296** | **0.3372** | **0.2790** | 0.6324 | 0.6870 | 2.94 |
| C6 EfficientNet-B4 + Phased | 0.5257 | 0.3355 | 0.2725 | 0.6212 | 0.6921 | 4.01 (+36%) |

**Conclusion:**
- **Switching the loss to Focal Tversky is the single biggest gain of the sprint:** +0.056 mIoU, and Destroyed IoU +55% relative.
- Phasing in Lovász adds only +0.003 over C3.
- ResNet-50 adds another +0.006.
- C4–C6 were built with the **C2** parameters (`alpha=0.3, beta=0.7, gamma=2.0`, confirmed in the notebook), not C3's. Whether C3's parameters help inside the phased loss is untested; D4 run R3 tests it.
- **Trade-off to be honest about:** the C-series raises precision a lot but *lowers* recall versus B1. For C6, Destroyed recall is 0.476 vs B1's 0.661. The class-weighted CE in B1 over-predicts rare classes (high recall, precision .18). Focal Tversky is more balanced.

### Member D — Ridita: Resolution & inference (40 epochs, CE + Dice, ResNet-34)

| Run | Train / eval input | mIoU | Damaged IoU | Destroyed IoU | Destroyed recall |
|---|---|---|---|---|---|
| B1 (true baseline) | Resize 512 / Resize 512 | 0.4619 | 0.2636 | 0.1658 | 0.6613 |
| D1b RandomCrop 512 (bs 16) | crop 512 / *random* crop 512 | 0.4452 | 0.2628 | 0.1070 | 0.8466 |
| D1b + TTA | same | 0.4540 | 0.2750 | 0.1222 | — |
| D1c RandomCrop 768 (bs 8) | crop 768 / *random* crop 768 | 0.4757 | 0.2904 | 0.1768 | 0.6341 |
| D1c + TTA | same | **0.4902** | **0.3192** | **0.1953** | — |
| D3 notebook (another 768 model) | crop 768 / random crop 768 | 0.4824 | 0.3020 | 0.1857 | 0.6416 |
| D3 + TTA | same | 0.4864 | 0.3044 | 0.1946 | — |
| D3 + "sliding window" | 512 windows over a **768 crop** | 0.4819 | 0.3021 | 0.1853 | — |

**Conclusion:**
- A 768 native crop beats 512 crops clearly: Destroyed IoU +65% relative.
- It beats the resized baseline only modestly: mIoU +0.014, Destroyed IoU +0.011. Even that comparison is shaky because the evaluation inputs differ.
- TTA gives a consistent free gain of about +0.004 to +0.015 mIoU.
- **The sliding window never ran at native 1024.** Its validation loader fed 768 random crops, and on the same model it changed mIoU by −0.0005.
- In D4, the model is evaluated on the full native image directly. A fully convolutional U-Net accepts 1024 × 1024 input, so no sliding window is needed.

---

## 5. Metric priority: mIoU vs. recall vs. F1 vs. F2
| Metric | What it rewards | Use it for |
|---|---|---|
| **mIoU** | Penalises false positives and false negatives equally; strict | **The official benchmark.** Model comparison and the report |
| **Recall** | Not missing real damage (false negatives) | Operational argument: a missed destroyed building is costlier than a false alarm |
| **F1** (= Dice) | 50/50 balance of precision and recall | Behaves much like mIoU; adds little |
| **F2** | Weights recall 2× precision | One combined number for the disaster-response view |

Talking points:
- Report **mIoU** to show the model is scientifically sound.
- Report **Destroyed-class recall** (not macro recall) and **F2** to show it is operationally useful.
- Recall alone can be gamed: predicting everything as Destroyed gives 100% recall.
- **Be precise about which recall:** the CSV "recall" column is a **macro average over 4 classes**, Background included. "79% recall" does **not** mean "finds 79% of destroyed buildings".
- The D4 notebook reports `destroyed_recall` and `f2` explicitly.

---

## 6. Likely supervisor questions
**Q: Why not train directly on mIoU?**
It is computed from hard argmax decisions and has no useful gradient. We train on smooth stand-ins (Focal Tversky, then Lovász-Softmax, a convex surrogate of IoU) and grade with mIoU.

**Q: Why did MA-Net collapse?**
Under CE + Dice, the 93% background dominated its attention modules, and it predicted only background (IoU 0 for every building class). We did not re-test it under Focal Tversky; it is a lower priority than U-Net++.

**Q: Why was RandomCrop 512 *worse* than Resize 512?**
A 512 crop sees only a quarter of the scene, so it has less context per sample. Its validation score also came from random crops, which are noisy. At 768 the context is enough and the native resolution wins.

**Q: Is C3 really better than C2?**
Not proven. It is +0.0022 mIoU from one seed. D4 runs R1 vs R6 measure seed noise, and R3 tests C3's parameters inside the phased loss.

**Q: Why not combine A2 + A3 straight away?**
The one-variable rule (§3). Also, the A3 CSV result was leaky, so copy-paste first needed a clean re-test (R2).

**Q: Does U-Net++ find more destroyed buildings?**
No. Under CE + Dice its Destroyed recall (0.668) is the same as the U-Net's (0.661), and its IoU is lower.

**Q: Your champion has a higher mIoU but a lower recall than the baseline — isn't that bad for rescue?**
- It is a real trade-off. The baseline's weighted CE over-predicts rare classes: Destroyed precision is 0.18, so about 4 of 5 "destroyed" pixels are false alarms.
- The C-series is far more precise.
- If recall matters more in deployment, we can lower the decision threshold for the Destroyed class at inference without retraining, and report F2.

**Q: How do you know the results aren't seed luck?**
- R6 re-runs R1 with another seed.
- A component counts only if it beats R1 by more than |R1 − R6|.

---

## 7. Pre-D4 ablations and the champion model
**Why more ablations first:** the old results come from three different evaluation setups, one leaky run, and single seeds. Stacking the old "winners" blindly could include components that don't really help.

**E-protocol** (used for every D4 number):
- the 645 validation images at **full native resolution**
- deterministic
- dataset-level confusion matrix → per-class IoU / P / R / F1 / F2
- the reported model is the **last epoch**, so the validation set is never used to pick a checkpoint.

| ID | Change vs R1 | Question |
|---|---|---|
| E0 | eval-only re-score of B1 / C5 / D1c checkpoints | What do the old models score under one protocol? |
| R1 | U-Net + ResNet-50 + Phased(C2 params) + RandomCrop 768, bs 8, 40 ep, seed 42 | New reference |
| R2 | + copy-paste (train-only bank, native scale) | Does copy-paste help once fixed? |
| R3 | FT α.25 β.75 γ1.33 | C3 params inside the phased loss? |
| R4 | U-Net++ (bs 4 × accum 2) | Architecture under the right loss? |
| R5 | + weighted sampler 10/5/1 | Sampler on the champion recipe? |
| R6 | seed 7 | Noise bar |
| R8 | + guide's enhanced augmentation (HSV, CLAHE, Elastic, GridDistortion, CoarseDropout) | Planned Tier-2 item, never run |
| R7 | R1 + every R2–R5/R8 change that beats R1 by more than \|R1 − R6\| without lowering Destroyed IoU | **Champion** (+ 8-view TTA) |

R2–R6 are independent, so run them in parallel on several Kaggle accounts. Run R1 and R6 first. Expect about 10 h per ResNet-50 run at 768; the notebook resumes across sessions from a per-experiment HF repo.

---

## 7b. Plan vs. reality (`team_task_assignments.md` + `segmentation_improvement_guide.md`)

**Phase Gate (Day 10).** The plan requires the champion to reach **mIoU > 0.52 and Destroyed IoU > 0.30**. If it misses either, Phase 2 is triggered (Task D5: external datasets such as xBD).
- Best so far: C5, with mIoU 0.5296 ✓ but **Destroyed IoU 0.279 ✗**.
- The guide (Tier 5) says the **two-stage pipeline** is the lever to pull when Destroyed IoU stays below 0.30.
- The guide's contour-counting threshold for the hybrid pipeline is also 0.30 per class.
- The D4 notebook prints the gate verdict automatically.

**Where several problems started: the plan's own starter code.**

| Problem in the results | Origin in the planning docs |
|---|---|
| A3 copy-paste leaked val buildings | The A3 extraction snippet loops over `pairs` (train + val), not `train_pairs` |
| Pasted instances had wrong colours (fixed in Aryan's notebook) | The A3 snippet normalises BGR as if it were RGB |
| Reported scores use the same val set as checkpoint selection | The plan's checkpoint snippet saves best-by-val and then reports on that val set |
| D-series val used `RandomCrop` | The plan specifies a **deterministic `CenterCrop(512)`** for D1b val. Ridita used `RandomCrop` instead |
| TTA duplicate view | The plan's D2 snippet has the "both flips = rot180" duplicate. The guide's §6A version also gets the 8th view's inverse wrong and averages logits instead of probabilities |
| B runs at batch 8 | The plan says to hold batch 16, and to *note it* if reduced. The CSV notes say 16; the notebooks use 8 |

**Experiment naming mismatch.**
- In the plan, **B5 = plain U-Net with ResNet-50 (CE + Dice)** and **B6 = MA-Net**.
- The team's "B5_MAnet" is really the planned B6. The planned B5 was never run.
- Because of that, the backbone effect was only measured under the Focal Tversky loss (C5).

**Planned but never done (or only partly):**
- **A1 dataset audit:** exists only as a figure, [`A1_dataset_class_distribution.png`](Aryan_Task/A1_dataset_class_distribution.png), not as the planned markdown report.
  - Pixels: BG 92.77%, Intact 5.86%, Damaged 1.05%, Destroyed 0.33%.
  - Images containing each class: Intact 83.6%, **Damaged 45.0%, Destroyed 37.0%**.
  - **63% of images have zero Destroyed pixels.**
- **A2 notebook:** only its CSV exists in the folder.
- **Guide Tier 2 #7, the enhanced augmentation pipeline** (HSV, CLAHE, Elastic, GridDistortion, CoarseDropout). Now added as D4 run **R8**.
- Guide Tier 3 #11 (morphological post-processing) and Tier 4 #12 (Boundary Loss): optional.

**Guide predictions vs. results (useful for the "what did we learn" slide):**

| Guide predicted | We measured |
|---|---|
| U-Net++ + scSE: +2–4% mIoU | −2.5 points (CE + Dice, ResNet-34) |
| Focal Tversky: +3–5% | **+5.6 points** (C2) |
| RandomCrop 512: +1–3% | −1.7 points (but evaluated on random crops) |
| Copy-paste: +3–6% on minority classes | Roughly neutral once the leak was removed (10-epoch screen) |
| TTA: +1–3% | +0.4 to +1.5 points |
| Tier 1 total: 0.52–0.58 | 0.5296 (C5), via the loss alone |

**Caution about citing the guide.** It says it was "cross-validated with two LLM systems". Several of its sources and claims are unverified and should not appear in the report without checking the original papers:
- Mushfiq et al. (IEEE)
- "IGARSS 2024 +6.2% copy-paste"
- "DeepDamageNet 2024 xView2 winners": wrong, since the xView2 challenge ran in 2019. Corrected in the guide on 6 Oct 2026: the real top two, Victor Durnov and Selim Seferbekov, used two-stage Siamese pipelines on pre- and post-disaster images.
- The guide's within-building class mix (~58% intact, which did not add up to 100%): it is ~81% intact, 14.5% damaged and 4.6% destroyed. Corrected in the guide on 6 Oct 2026.
- "BDANet SOTA via copy-paste"
- The throughput and parameter tables

## 7c. Where the wrong numbers came from, and what the audit doc gets wrong

**`generate_master_summary.py` writes `all_members_ablation_summary.csv` from hand-typed values.** It never reads the member CSVs or notebooks. That is how these got in:
- the B1 row (.94 / .485 / .284 / .152, R .641), which matches no run
- the D1a row, a copy of B1
- the leaky A3 numbers
- the D3 row labelled "native 1024"

**To fix this:** regenerate the CSV from the per-run CSVs and the D4 outputs instead of typing values in.

**The B architecture runs changed three things besides the architecture.** All four B notebooks (B2–B5) set:
- `LEARNING_RATE = 5e-4` and `WEIGHT_DECAY = 1e-4` (the baseline actually trained with 1e-4 and 1e-5)
- `BATCH_SIZE = 8` (baseline 16)
- no gradient accumulation at all, despite the audit doc's "batch 8 (grad accum 2)"

So U-Net++'s lower mIoU, and possibly MA-Net's collapse, may come from 5× the learning rate rather than the architecture. D4 run R4 re-tests U-Net++ with everything else held equal.

**`Team_task_ablation_audit_and_lead_execution_plan.md` (dated Sep 28) — claims that don't hold:**

| Audit doc says | Actually |
|---|---|
| Imbalance: Destroyed 0.4%, Damaged 1.3%, BG 92.4% | A1 figure: 0.33% / 1.05% / 92.77% |
| Copy-paste lifted Destroyed recall to **73.27%** | That number appears nowhere else in the folder. The leak-free A3 notebook prints 0.7595 |
| Aryan's notebook extracted cut-outs from `pairs` (leaky) | The saved notebook uses `train_pairs`. The **CSV numbers** come from an earlier leaky run |
| B runs: "batch size 8 (grad accum 2)" | Batch 8, no accumulation, plus lr 5e-4 / wd 1e-4 |
| U-Net++ chosen for "78.66% vs 64.10% recall" | The 64.10% is the fake B1 row. The real B1 recall is 79.06% |
| Phased loss "boosted Destroyed IoU 0.1520 → 0.2790" | 0.2790 is C5 (loss **and** ResNet-50), and the true baseline is 0.1658 |
| "Flawless TTA … exact inverse mappings" | One view is a duplicate (7 unique) |
| D3 sliding window "verified on native 1024" (+2.19%) | It ran on 768 random crops and gave −0.0005 |
| Champion recipe: U-Net++, lr 5e-4, eff. batch 16 | That would change the decoder and the lr, both untested with this loss. D4 keeps lr 1e-4 and tests U-Net++ separately (R4) |
| Target 0.58–0.62 mIoU, Destroyed 0.32–0.36; "literature baseline ~0.39" | No source or calculation for any of these |

**Noman's PDF report ([`C_analysis_report_noman.pdf`](Noman_task/C_analysis_report_noman.pdf)) agrees with the notebooks.**
- It confirms C4/C5 used `alpha=0.3, beta=0.7, gamma=2.0`.
- It adds per-class **Destroyed recall: C2 .459, C3 .505, C4 .445**. C3 is the best of the loss runs at finding destroyed pixels, which supports testing C3's parameters (R3).
- Its table shows C3's Destroyed IoU as 0.2720, a typo for the printed 0.2727.
- It notes a FT + ResNet-50 run "can be done if Recall is the priority".

**Training curves ([`Noman_task/C*_epoch_curves.png`](Noman_task/)):**
- C5 and C6 peaked at the last epoch.
- **C3 peaked at epoch 28 of 40** (per-image val mIoU 0.4239 vs 0.4188 at the end). Its reported 0.5203 therefore comes from a val-selected checkpoint, while C5's comes from its final epoch.
- In every phased run, the loss jumps at epoch 33 when Lovász switches on.
- The per-epoch "mIoU" column (about 0.43) is the per-image metric, not the reported dataset-level number (about 0.53).

**Checkpoints in the folder:** `Tamanna_task/best_model_B2–B5.pth` (all ResNet-34, keys verified). They are wired into D4 run **E0** so they can be re-scored under the shared protocol. B1, C and D checkpoints are not in the folder.

**HTML copies:** `segmentation_improvement_guide.html` and `team_task_assignments.html` have the same content as their `.md` files.

## 8. Errata — do not quote these numbers
| # | Claimed (summary CSV / reports / earlier chat) | Actually (notebook output) |
|---|---|---|
| 1 | B1 / C1 / D1a (the baseline notebook): BG .94, Intact .485, Damaged .284, Destroyed .152, P .582, **R .641**, F1 .609. These per-class values average to 0.4653, not the row's own 0.4619 | BG .9255, Intact .4926, Damaged .2636, **Destroyed .1658**, P .5015, **R .7906**, F1 .5808 |
| 2 | "U-Net++ raised recall 64% → 79%" | B1 macro recall is 0.791, *higher* than B2's 0.787 |
| 3 | B3 has "peak recall" | B4 DeepLabV3+ is higher (0.7940 macro, 0.761 Destroyed) |
| 4 | A3 copy-paste 0.4077 / Destroyed .1026, 44,969 / 18,125 instances | Leaky (bank included val images). Leak-free: 0.4017 / .0949, 40,704 / 16,682 |
| 5 | D3 sliding window at native 1024: 0.4819, "+2.19 free boost over baseline" | Ran on 768 random crops; same model without SW scored 0.4824, so **−0.0005** |
| 6 | D1 report: RandomCrop 512 = 0.4540, 768 = 0.4902 | Those are TTA scores. Plain: 0.4452 and 0.4757 |
| 7 | 768 crop: "+59.8% Destroyed IoU" (and "vs downscaling") | +59.8% is TTA vs TTA, 768 vs 512. Plain 768 vs 512: +65.2%. 768 vs resized B1: +6.6% |
| 8 | D2 TTA = 8 views | 7 unique: "flip both axes" duplicates rot180. Fixed in D4 |
| 9 | Phased loss "smoothly adds" Lovász | Hard switch at epoch 32 to 0.6 FT + 0.4 Lovász |
| 10 | C5 Destroyed +83.5% over baseline (audit doc) | vs the true .1658: +68% |
| 11 | "C4 Phased loss caused the massive jump" | Focal Tversky alone (C2) gives 0.5181; phasing adds +0.003 over C3 |
| 11b | B2–B5 trained at batch 16 (CSV notes) | All four notebooks use `BATCH_SIZE = 8` |
| 12 | A1 / A3 train times 0.5 / 0.6 h | Blank in the source CSVs (hand-filled) |
| 13 | Earlier chat: 0.3977 / 0.4050 / 0.4076 | Correct rounding: 0.3978 / 0.4051 / 0.4077 |

**Data-hygiene notes:**
- All old notebooks pushed to the same HF repo (`AbrarAlam/disasterm3-unet-checkpoints-2`), so `best_model.pth` there is whichever run pushed last.
- The old notebooks pick "best" checkpoints with a per-image metric that ignores zero-IoU classes. That metric differs from the reported one, and it is computed on the same validation set used for reporting.
- Noman's CSV-export cell still has template labels (`MEMBER_X_EXP_Y`).
