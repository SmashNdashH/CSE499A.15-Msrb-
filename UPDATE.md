>  **[Main Project Overview](README.md)** |  **[Latest Team Update](UPDATE.md)** | **[Update 1 Report (PDF)](others/CSE499B_update_1.pdf)**

---

# Post-Disaster Damage Assessment and Rescue Guidance

> **CSE499B – Section 13, Group 5, Update 1**
> 
> **Team:** 
> - Abdullah Al Noman (2022095042)
> - Tamanna Akter Mou (2211951042)
> - Aryan Sami (2231407042)
> - Ridita Afrin Riya (2211622042)
> - Abrar Mohammed Tanzim Alam (2222864042)

*Earlier CSE499A updates (1 to 5) are archived as PDFs in [others/CSE499A/](others/CSE499A/).*

## Weekly Update

This week we started a parallel ablation study to improve our DisasterM3 baseline segmentation model (U-Net with a ResNet-34 encoder and CE+Dice loss). In our short 10-epoch screening runs this model reaches only 0.3978 mean Intersection over Union (mIoU), and with full 40-epoch training it plateaus at 0.4619 mIoU. Our phase-gate targets are an mIoU above 0.52 and a Destroyed-class IoU above 0.30. The main obstacle is class imbalance: about 93% of all pixels are background. We planned a 15-day sprint split into four tracks: data sampling, architecture, loss function, and resolution/inference. Each track changes one main variable against the shared baseline so that its effect on minority-class IoU can be measured; where a run also had to change the batch size or the evaluation crops, this is stated below. The best setting from each track will then be combined into a single Champion Model. All results are single runs on our 90/10 validation split (645 images).

## Results Summary

| Track | Configuration (change vs. baseline) | mIoU | Destroyed IoU | Δ mIoU |
|---|---|---:|---:|---:|
| – | Baseline: U-Net, ResNet-34, CE+Dice, resize 512, batch 16, 40 epochs | 0.4619 | 0.152 | – |
| Data (a) | Uniform shuffle (10-epoch reference) | 0.3978 | 0.090 | – |
| Data (a) | WeightedRandomSampler (10/5/1 weights) | 0.4051 | 0.102 | +0.0073 |
| Architecture (b) | U-Net++ | 0.4405 | 0.131 | −0.0214 |
| Architecture (b) | U-Net++ with scSE attention | 0.4367 | 0.125 | −0.0252 |
| Loss | Focal Tversky (α/β/γ = 0.3/0.7/2.0) | 0.5181 | 0.257 | +0.0562 |
| Loss | Focal Tversky (α/β/γ = 0.25/0.75/1.33) | 0.5203 | **0.273** | +0.0584 |
| Loss | Phased: Focal Tversky → + Lovász-Softmax | **0.5233** | 0.272 | **+0.0614** |
| Resolution (c) | RandomCrop 512×512 | 0.4452 | 0.107 | −0.0167 |
| Resolution (c) | RandomCrop 768×768 (batch 8) | 0.4757 | 0.177 | +0.0138 |
| Inference (d) | 768 model + 8-view TTA | 0.4902 | 0.195 | +0.0145 |
| Inference (d) | 768 model (second run) + sliding window | 0.4819 | 0.185 | −0.0005 |

(a) 10-epoch screening runs; Δ against the 10-epoch uniform reference. (b) Batch size 8. (c) Validated on random crops, not whole resized images. (d) No retraining; Δ against the same model without TTA or sliding window.

## Work Done by Aryan Sami (Data Sampling)

1. **Dataset Audit and Pixel Distribution Analysis:**
   - Wrote an OpenCV audit script that reads the ground-truth mask of every image in the dataset (6,443 images) and records per-class pixel counts and per-image class presence.
   - Pixel shares: Background 92.77%, Intact 5.86%, Damaged 1.05%, Destroyed 0.33%.
   - Destroyed buildings appear in 37% of images (2,385 of 6,443); the other 63% contain no Destroyed pixels at all. The histogram of Destroyed pixels per image is heavily skewed: most images that contain the class hold only a small patch of it. So Destroyed pixels do reach most batches, but they make up a tiny share of each batch's loss, which gives the class a weak gradient signal.
2. **Class-Balanced Sampling Pipeline Integration:**
   - Pre-computed one weight per training image from its mask: 10 if it contains Destroyed pixels, 5 if it contains Damaged but no Destroyed pixels, and 1 otherwise. These weights feed a PyTorch `WeightedRandomSampler` (with replacement) that replaces `shuffle=True` in the `DataLoader`.
   - The batch size stays at 16, so BatchNorm statistics are unaffected. The sampler makes minority-class images much more frequent in each batch, although it does not guarantee one in every batch.
   - 10-epoch screening run against uniform shuffling: mIoU rose from 0.3978 to 0.4051 and Destroyed IoU from 0.090 to 0.102 (+13% relative), with recall slightly lower (80.6% to 79.2%).

## Work Done by Abrar Mohammed Tanzim Alam (Architecture & Orchestration)

1. **Baseline Validation and U-Net++ Implementation:**
   - Set up the baseline with the `segmentation_models_pytorch` (SMP) API: a U-Net with an ImageNet-pretrained ResNet-34 encoder, trained for 40 epochs, reached 0.4619 mIoU (Destroyed IoU 0.152, recall 64.1%).
   - Trained U-Net++, whose nested dense skip connections bridge the gap between the encoder's high-resolution features and the decoder's semantic features, with the same encoder, loss and 40 epochs. These runs used a batch size of 8 instead of the baseline's 16, so batch size is a second difference.
   - U-Net++ raised recall to 78.7%, but mIoU fell to 0.4405 (Destroyed IoU 0.131): under CE+Dice it over-predicts damage regions, which lowers precision (0.48).
2. **Attention Mechanism Integration (scSE) and ASPP Exploration:**
   - Added concurrent spatial and channel squeeze-and-excitation (scSE) attention blocks to the U-Net++ decoder, with everything else unchanged from the U-Net++ run.
   - scSE did not help under CE+Dice: mIoU 0.4367 (Destroyed IoU 0.125) against 0.4405 for plain U-Net++, with recall of 79.2%. Training time rose from 4.8 to 8.0 hours.
   - Started integrating DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context; its full evaluation is scheduled for next week.
3. **Planning and Orchestration:**
   - Planned and orchestrated the whole ablation sprint: track design, task delegation and the phase-gate criteria.
   - Created and provided all experiment notebooks used by every track; each member ran and reported their own track.

## Work Done by Abdullah Al Noman (Loss Function)

1. **Focal Tversky Loss Experimentation:**
   - Replaced the baseline CE+Dice loss with SMP's Focal Tversky loss, which penalises false negatives more than false positives. For disaster response, missing a destroyed building costs more than a false alarm. Architecture (U-Net, ResNet-34), data, batch size (16) and 40 epochs were unchanged.
   - Trial 1 (α=0.3, β=0.7, γ=2.0): 0.5181 mIoU, Destroyed IoU 0.257. Trial 2 (α=0.25, β=0.75, γ=1.33), which weights false negatives even more heavily: 0.5203 mIoU, Destroyed IoU 0.273.
   - Both trials beat the 0.4619 baseline by about 0.06 mIoU and raise Destroyed IoU by roughly 70–80% relative. Trial 2 changed α, β and γ together, so the effect of γ alone has not been isolated.
2. **Phased Compound Loss Scheduler Development:**
   - Wrote an epoch-aware loss module (`PhasedCompoundLoss`) that the training loop updates every epoch through `set_epoch()`.
   - For the first 80% of epochs (1–32 of 40) it uses Focal Tversky loss alone (α=0.3, β=0.7, γ=2.0) to build up a stable gradient signal for the minority classes.
   - For the final 20% (epochs 33–40) it switches to 0.6 × Focal Tversky + 0.4 × Lovász-Softmax. Lovász-Softmax is a convex surrogate of the IoU (Jaccard index), so the last epochs optimise the evaluation metric more directly. Result: 0.5233 mIoU, Destroyed IoU 0.272, the best mIoU so far.

## Work Done by Ridita Afrin Riya (Resolution & Inference)

1. **Resolution and Cropping Ablation Pipeline:**
   - Compared training-resolution strategies with the Albumentations library, keeping the architecture (U-Net, ResNet-34), CE+Dice loss and uniform sampling fixed.
   - Resize 1024→512 (baseline): 0.4619 mIoU, Destroyed IoU 0.152. Native-resolution RandomCrop 512×512: 0.4452, Destroyed IoU 0.107, which is worse than the baseline. RandomCrop 768×768: 0.4757, Destroyed IoU 0.177, the best of the three. Native-resolution crops therefore helped only once the crops were large enough.
   - The 768×768 run needed the batch size cut from 16 to 8 to fit in Kaggle T4 VRAM (no gradient accumulation was used). The crop runs were also validated on random crops of the validation images, while the baseline is validated on whole resized images, so these comparisons are indicative rather than exact.
2. **Test-Time Augmentation (TTA) and Sliding Window Inference:**
   - Built a TTA function that predicts on 8 geometric views of each batch (identity, horizontal flip, vertical flip, both flips, 90°/180°/270° rotations, and 90° rotation plus horizontal flip), maps each prediction back with the inverse transform, and averages the softmax maps. The double flip equals the 180° rotation, so 7 of the views are distinct.
   - Without any retraining, TTA raised the 768-crop model from 0.4757 to 0.4902 mIoU (+0.0145) and the 512-crop model from 0.4452 to 0.4540.
   - Implemented sliding-window inference with 512×512 windows and a 128-pixel overlap (plus edge-anchored windows), averaging overlapping predictions with equal weight, so that crop-trained models can predict on full 1024×1024 images. Its first test ran on 768×768 validation crops rather than full images and gave no gain: 0.4819 against 0.4824 for standard inference on the same model (0.4864 with TTA).

---

## Next Week's Strategic Roadmap

- **Phase 1 Completion:** Run the remaining isolated experiments: Copy-Paste instance augmentation (Aryan), full DeepLabV3+ and MA-Net evaluations (Abrar), ResNet-50 against EfficientNet-B4 backbones with the phased loss (Noman), and full-resolution 1024×1024 sliding-window evaluation with a fixed validation protocol for the crop runs (Ridita).
- **Champion Model Integration:** Combine the best setting from each track into a single "Champion Model", train it for 40 epochs, and report its mIoU and Destroyed IoU against the 0.4619 baseline.
- **Contingency Phase 2 Activation:** If the Champion Model does not reach both targets (mIoU above 0.52 and Destroyed IoU above 0.30), we will start Phase 2: survey external disaster damage datasets, convert their labels to our four classes, and add them to the training data to supplement the minority classes.
