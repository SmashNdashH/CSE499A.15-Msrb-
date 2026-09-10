# Team Task Assignments: DisasterM3 Segmentation Ablation Study

## 🚀 DisasterM3 Segmentation Project: 15-Day Capstone Sprint

**Hey team, here is the master plan for the next 15 days.**

### Our Current Standing
Right now, our baseline segmentation model is sitting at roughly **0.46 mIoU**. The biggest issue we are facing is extreme class imbalance. About 93% of our satellite imagery pixels are empty background, while our most critical class ("Destroyed" buildings) makes up only **0.33%** of the pixels. Because of this, the model is blindly guessing "background" most of the time and getting away with it.

### Why we are doing this
To break through the 0.46 mIoU plateau and get the highest possible score, we need to throw the kitchen sink at the problem—but we have to do it *scientifically*.

If we all just randomly change things in the notebook, we will never know what actually worked. Instead, we are running a strict, parallel **Ablation Study**. Each of you has been assigned one specific domain to own. You will change **exactly one variable at a time** so we can measure its precise impact. On Day 9, we will take the winning piece from each of your experiments and bolt them together into one massive "Champion Model."

> [!IMPORTANT]
> **The Golden Rule for this sprint:** Do NOT mix your experiments. If you are testing architectures, leave the loss function and data alone.

---

## 📂 Shared Setup & Required Reading (Everyone Must Do This First)

Before writing any code, everyone must read this document and the Technical Guide. They contain the exact math, architectures, starter code, and schedules we are using.

- **The Technical Guide:** `segmentation_improvement_guide.md` (or `.html`)
- **Your Task Schedule:** `team_task_assignments.md` (or `.html`)

### 1. Get the Baseline Notebook Running
The Starting Baseline Notebook for every single experiment is: 👉 `segmentation-model (1).ipynb`
*(You will fork/copy this exact notebook for your tasks).*

**Baseline configuration (DO NOT CHANGE unless it's your assigned variable):**

| Parameter | Baseline Value |
|---|---|
| Architecture | `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet")` |
| Loss | `0.5 × Weighted CE + 0.5 × Dice` (Cell 6) |
| Resolution | `A.Resize(512, 512)` (Cell 5) |
| Batch Size | 16 |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-5 |
| Scheduler | CosineAnnealingLR |
| Epochs | 40 |
| Train/Val Split | 90/10, random_state=42 |
| AMP | FP16 enabled |
| Hardware | Kaggle T4 GPU (16GB) |

**Baseline mIoU: ≈ 0.4619**

### 2. Kaggle GPU Quota & Mid-Run Checkpointing

> [!CAUTION]
> **Kaggle caps free T4 GPU access at 30 hours per week.** 
> A 40-epoch run takes ~4–6 hours. This means Members B and C will be dangerously close to hitting their weekly quota. 
> 
> **Action Items before starting:**
> 1. Go to Account → Settings and check your remaining GPU hours and exact "reset day". 
> 2. **Checkpoint Every Run:** The 12-hour session limit and 30-hour weekly cap mean runs might be interrupted. You must use `ModelCheckpoint` so you don't lose a 5-hour run to a timeout.

```python
# Add this to your PyTorch training loop to save progress
checkpoint_path = f"best_model_{results['experiment_id']}.pth"
if current_iou > best_iou:
    best_iou = current_iou
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✓ Saved new best model at epoch {epoch} with IoU {best_iou:.4f}")
```

### 3. Shared Results Format

**Every experiment must report these columns:**

```
| Experiment ID | Variable Changed | Value | mIoU | BG IoU | Intact IoU | Damaged IoU | Destroyed IoU | Precision | Recall | F1 | Train Time | Notes |
```

Save results as CSV after each run:
```python
# At the end of Cell 9 (evaluation), add:
import csv
results = {
    "experiment_id": "MEMBER_X_EXP_Y",   # e.g., "B_arch_unetpp_r34"
    "variable": "architecture",
    "value": "UnetPlusPlus_resnet34",
    "miou": float(metrics["iou"].mean()),
    "bg_iou": float(metrics["iou"][0]),
    "intact_iou": float(metrics["iou"][1]),
    "damaged_iou": float(metrics["iou"][2]),
    "destroyed_iou": float(metrics["iou"][3]),
    "precision": float(metrics["precision"].mean()),
    "recall": float(metrics["recall"].mean()),
    "f1": float(metrics["f1"].mean()),
    "train_time_hrs": sum(epoch_times) / 3600,
    "notes": "",
}
with open(f"/kaggle/working/{results['experiment_id']}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results.keys())
    w.writeheader()
    w.writerow(results)
```

### 3. Required Reading (Everyone)

Before starting, each member should read:
- The `segmentation_improvement_guide.md` — our cross-validated research guide
- Sections of this document specific to their role

---

## 👤 Member A (Aryan): Data Engineer — Class Imbalance & Augmentation

### Your Mission
You are responsible for making the training data **fair**. Right now, 93% of all pixels are background, and the model barely ever sees a Destroyed building during training. You will fix this at the **data level** so that every other member's experiments benefit.

### Background: Why This Matters
The model currently trains on randomly shuffled images. A large fraction of those images contain ONLY background and intact buildings — no damaged or destroyed pixels at all. This means:
- The model gets zero gradient signal for Destroyed in most training steps
- Even the best loss function can't help if the rare class doesn't appear in the batch

### Task A1: Dataset Audit & Distribution Report

**What to do:**
1. Run the existing Cell 4 (mask generation) to produce all combined masks
2. Write a script that analyzes EVERY training image's mask and reports:

```python
# ── Dataset Audit Script ──
import cv2
import numpy as np
from collections import Counter
from pathlib import Path

class_names = ["Background", "Intact", "Damaged", "Destroyed"]
image_class_presence = {0: 0, 1: 0, 2: 0, 3: 0}  # How many images contain each class
total_pixels = Counter()

for pair in pairs:
    mask = cv2.imread(pair["mask_path"], cv2.IMREAD_GRAYSCALE)
    unique_classes = np.unique(mask)
    for cls in unique_classes:
        image_class_presence[cls] += 1
        total_pixels[cls] += np.sum(mask == cls)

print("=== Per-Image Class Presence ===")
for cls, count in image_class_presence.items():
    print(f"  {class_names[cls]}: present in {count}/{len(pairs)} images "
          f"({100*count/len(pairs):.1f}%)")

print("\n=== Pixel Distribution ===")
total = sum(total_pixels.values())
for cls in range(4):
    pct = 100 * total_pixels[cls] / total
    print(f"  {class_names[cls]}: {total_pixels[cls]:,} pixels ({pct:.2f}%)")
```

**Deliverable A1:** A markdown report with:
- Exact pixel counts per class
- Exact number of images containing each class
- Histogram showing how many Destroyed pixels per image (most will be 0)

### Task A2: Implement WeightedRandomSampler

**What to do:** Modify Cell 5 to oversample images containing Damaged/Destroyed pixels.

```python
from torch.utils.data import WeightedRandomSampler

# Pre-compute sample weights (run ONCE before creating DataLoader)
print("Computing sample weights for class-balanced sampling...")
sample_weights = []
for pair in train_pairs:
    mask = cv2.imread(pair["mask_path"], cv2.IMREAD_GRAYSCALE)
    has_destroyed = np.any(mask == 3)
    has_damaged = np.any(mask == 2)

    if has_destroyed:
        sample_weights.append(10.0)     # 10× oversampling
    elif has_damaged:
        sample_weights.append(5.0)      # 5× oversampling
    else:
        sample_weights.append(1.0)      # Normal sampling

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_pairs),
    replacement=True,
)

# IMPORTANT: Remove shuffle=True when using a sampler
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,    # Keep >= 8 for BatchNorm stability!
    sampler=sampler,          # replaces shuffle=True
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
```

**Deliverable A2:** Modified notebook cells + a before/after comparison:
- Run baseline (uniform sampling) for 10 epochs, log per-class IoU
- Run with WeightedRandomSampler for 10 epochs, log per-class IoU
- Report the difference

### Task A3: Build Copy-Paste Augmentation Pipeline

**What to do:** Extract individual Damaged/Destroyed building instances from training masks and build a paste-during-training pipeline.

**Step 1: Offline Instance Extraction**

```python
# ── Run this ONCE to extract minority-class building instances ──
import cv2
import numpy as np
import os
from pathlib import Path

INSTANCE_DIR = "/kaggle/working/minority_instances"
os.makedirs(f"{INSTANCE_DIR}/damaged", exist_ok=True)
os.makedirs(f"{INSTANCE_DIR}/destroyed", exist_ok=True)

instance_count = {2: 0, 3: 0}

for idx, pair in enumerate(pairs):
    img = cv2.imread(pair["image_path"])
    mask = cv2.imread(pair["mask_path"], cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue

    for cls_id, cls_name in [(2, "damaged"), (3, "destroyed")]:
        binary = (mask == cls_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        # Find connected components (individual building instances)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 100:  # Skip tiny noise blobs (< 100 pixels)
                continue

            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]

            # Crop the instance (image + mask)
            inst_img = img[y:y+h, x:x+w].copy()
            inst_mask = (labels[y:y+h, x:x+w] == label_id).astype(np.uint8)

            # Save
            save_id = instance_count[cls_id]
            cv2.imwrite(f"{INSTANCE_DIR}/{cls_name}/img_{save_id}.png", inst_img)
            cv2.imwrite(f"{INSTANCE_DIR}/{cls_name}/mask_{save_id}.png", inst_mask * 255)
            instance_count[cls_id] += 1

print(f"Extracted instances — Damaged: {instance_count[2]}, Destroyed: {instance_count[3]}")
```

**Step 2: Online Paste-During-Training Dataset Wrapper**

```python
import random
import glob

class CopyPasteAugDataset(Dataset):
    """Wraps base dataset. Randomly pastes minority-class instances onto images."""

    def __init__(self, base_dataset, instance_dir, paste_prob=0.5, max_paste=3):
        self.base = base_dataset
        self.paste_prob = paste_prob
        self.max_paste = max_paste

        # Load all extracted instances into memory
        self.instances = []
        for cls_id, cls_name in [(2, "damaged"), (3, "destroyed")]:
            img_paths = sorted(glob.glob(f"{instance_dir}/{cls_name}/img_*.png"))
            for img_path in img_paths:
                mask_path = img_path.replace("img_", "mask_")
                inst_img = cv2.imread(img_path)
                inst_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if inst_img is not None and inst_mask is not None:
                    inst_mask = (inst_mask > 127).astype(np.uint8)  # binarize
                    self.instances.append((inst_img, inst_mask, cls_id))

        print(f"Loaded {len(self.instances)} minority instances for Copy-Paste")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask = self.base[idx]

        # image is [C, H, W] tensor, mask is [H, W] tensor after transforms
        if random.random() < self.paste_prob and self.instances:
            n = random.randint(1, self.max_paste)
            # Convert tensors to numpy for pasting
            img_np = image.permute(1, 2, 0).numpy()  # [H, W, C]
            mask_np = mask.numpy()                     # [H, W]

            for _ in range(n):
                inst_img, inst_mask, cls_id = random.choice(self.instances)
                ih, iw = inst_mask.shape[:2]

                # Ensure instance fits in the image
                if ih >= img_np.shape[0] or iw >= img_np.shape[1]:
                    continue

                # Random position
                y = random.randint(0, img_np.shape[0] - ih)
                x = random.randint(0, img_np.shape[1] - iw)

                # Paste where instance mask is nonzero
                paste_mask = inst_mask > 0

                # Normalize instance image same as training
                inst_normalized = inst_img.astype(np.float32) / 255.0
                inst_normalized = (inst_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

                img_np[y:y+ih, x:x+iw][paste_mask] = inst_normalized[paste_mask]
                mask_np[y:y+ih, x:x+iw][paste_mask] = cls_id

            image = torch.from_numpy(img_np).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask_np).long()

        return image, mask
```

**Deliverable A3:**
- Count of extracted instances per class
- Modified notebook with CopyPasteAugDataset wrapper
- Run 10 epochs with Copy-Paste ON (same architecture/loss as baseline), report per-class IoU change

### Final Deliverables Checklist for Member A

- [ ] Dataset audit report (pixel counts, image-level class presence, histogram)
- [ ] `WeightedRandomSampler` implementation + 10-epoch comparison
- [ ] Copy-Paste extraction script + count of instances extracted
- [ ] `CopyPasteAugDataset` wrapper + 10-epoch comparison
- [ ] CSV results files for each experiment (format above)

---

## 👤 Member B (Tamanna): Architecture Ablation Lead

### Your Mission
You will run **controlled experiments** comparing different segmentation architectures, keeping everything else (loss, data, augmentation, epochs) identical. This produces the architecture comparison table for our final report.

### Background: What You Need to Know

**What is an ablation study?**
An ablation study systematically changes ONE variable while holding all others constant. You are ablating the **architecture** variable. If U-Net++ gets 0.52 mIoU and standard U-Net gets 0.46, you can directly attribute the +0.06 gain to the architecture change because nothing else was different.

**The architectures you'll test:**

| Architecture | SMP API | What It Does Differently |
|---|---|---|
| U-Net (baseline) | `smp.Unet` | Standard encoder-decoder with skip connections |
| U-Net++ | `smp.UnetPlusPlus` | Adds nested dense skip connections between encoder and decoder, bridging the semantic gap |
| U-Net++ + scse | `smp.UnetPlusPlus(..., decoder_attention_type="scse")` | Same + Spatial & Channel Squeeze-and-Excitation attention in decoder |
| DeepLabV3+ | `smp.DeepLabV3Plus` | Uses Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context |

### Experiment Plan

**Hold constant:** Loss (Cell 6 — standard CE+Dice), augmentation (Cell 5), encoder (resnet34), epochs (40), batch size (16), all other hyperparams.

**Change only:** The model definition in Cell 8.

#### Experiment B1: Baseline U-Net (Already Done)

```python
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)
```
Expected mIoU: ~0.46 (already known)

#### Experiment B2: U-Net++

```python
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)
```

#### Experiment B3: U-Net++ with scse Attention

```python
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    decoder_attention_type="scse",
).to(DEVICE)
```

#### Experiment B4: DeepLabV3+

```python
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)
```

#### Experiment B5: U-Net (Plain) with ResNet-50 Encoder (Optional/Stretch)

> [!NOTE]
> **Only run this if you have remaining Kaggle GPU quota.**
> *Purpose: Isolates the effect of a deeper backbone from the nested decoder of U-Net++.*
```python
model = smp.Unet(                    # plain U-Net decoder, NOT UnetPlusPlus
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)
```

#### Experiment B6: MA-Net (Multi-scale Attention Network) (Optional/Stretch)

> [!NOTE]
> **Only run this if you have remaining Kaggle GPU quota.**
> *Purpose: Purpose-built for small/rare-region segmentation, targeting our "Destroyed" class.*
```python
model = smp.MAnet(
    encoder_name="resnet34",         # keep encoder fixed to isolate decoder effect
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)
```

> [!WARNING]
> DeepLabV3+ and MA-Net may need batch size reduced to 8 if you get OOM errors. If you reduce batch size, note it in your results — it affects BatchNorm statistics and training dynamics.

### How to Run Each Experiment

1. **Fork** the baseline notebook on Kaggle (create 6 copies, one per experiment)
2. In each copy, change **ONLY** the model definition in Cell 8
3. Give each notebook a clear name: `segmodel_ablation_B2_unetpp.ipynb`
4. Run all cells → let it train for 40 epochs
5. At Cell 9 (evaluation), save the CSV results file
6. Upload results CSV to our shared repo

### Deliverables for Member B

- [ ] 6 trained models (B1–B6), each with `best_model.pth` saved
- [ ] 6 CSV results files with per-class IoU
- [ ] A comparison table summarizing all 6 architectures:

```markdown
| Architecture | mIoU | BG IoU | Intact IoU | Damaged IoU | Destroyed IoU | Train Time |
|---|---|---|---|---|---|---|
| U-Net (baseline) | 0.46 | ? | ? | ? | ? | ? hrs |
| U-Net++ | ? | ? | ? | ? | ? | ? hrs |
| U-Net++ + scse | ? | ? | ? | ? | ? | ? hrs |
| DeepLabV3+ | ? | ? | ? | ? | ? | ? hrs |
| U-Net (ResNet50) | ? | ? | ? | ? | ? | ? hrs |
| MA-Net | ? | ? | ? | ? | ? | ? hrs |
```

- [ ] A short write-up (1 paragraph) stating which architecture won and why

---

## 👤 Member C (Noman): Loss Function & Backbone Ablation Lead

### Your Mission
You run **controlled experiments** comparing different loss functions AND different encoder backbones. This is split into two sub-ablations.

### Sub-Ablation C-I: Loss Function Comparison

**Hold constant:** Architecture (smp.Unet, resnet34), augmentation, epochs (40), batch size (16).
**Change only:** The loss function in Cell 6.

#### Experiment C1: Baseline CE + Dice (Already Done)

Current Cell 6 code. mIoU ≈ 0.4619.

#### Experiment C2: Focal Tversky Loss (α=0.3, β=0.7, γ=2.0)

Replace the entire Cell 6 with:

```python
import segmentation_models_pytorch as smp
import torch.nn as nn

NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalTverskyOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.tversky = smp.losses.TverskyLoss(
            mode='multiclass',
            alpha=0.3,
            beta=0.7,
            gamma=2.0,
        )

    def forward(self, logits, targets):
        loss = self.tversky(logits, targets)
        return loss, loss.item(), 0.0  # Keep return signature compatible

criterion = FocalTverskyOnly().to(DEVICE)
print(f"✓ Focal Tversky Loss ready on {DEVICE}")
```

#### Experiment C3: Focal Tversky (different hyperparams: α=0.25, β=0.75, γ=1.33)

Same as C2 but with α=0.25, β=0.75, γ=4/3. This gives even more FN penalty.

#### Experiment C4: Phased Compound Loss (Focal Tversky → +Lovász at 80%)

```python
import segmentation_models_pytorch as smp
import torch.nn as nn

NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhasedCompoundLoss(nn.Module):
    def __init__(self, num_epochs, lovasz_start_fraction=0.8):
        super().__init__()
        self.tversky = smp.losses.TverskyLoss(
            mode='multiclass', alpha=0.3, beta=0.7, gamma=2.0,
        )
        self.lovasz = smp.losses.LovaszLoss(mode='multiclass')
        self.lovasz_start_epoch = int(num_epochs * lovasz_start_fraction)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, logits, targets):
        t_loss = self.tversky(logits, targets)
        if self.current_epoch >= self.lovasz_start_epoch:
            l_loss = self.lovasz(logits, targets)
            total = 0.6 * t_loss + 0.4 * l_loss
            return total, t_loss.item(), l_loss.item()
        else:
            return t_loss, t_loss.item(), 0.0

criterion = PhasedCompoundLoss(num_epochs=NUM_EPOCHS).to(DEVICE)
print(f"✓ Phased Compound Loss ready on {DEVICE}")
```

> [!IMPORTANT]
> For experiment C4, you MUST add `criterion.set_epoch(epoch)` at the start of the training loop in Cell 8:
> ```python
> for epoch in range(start_epoch, NUM_EPOCHS):
>     criterion.set_epoch(epoch)    # ← ADD THIS LINE
>     epoch_start = time.time()
>     # ... rest unchanged
> ```

### Sub-Ablation C-II: Backbone Comparison

**Hold constant:** Architecture (smp.Unet), loss (best from C-I), augmentation, epochs (40).
**Change only:** The encoder name in Cell 8.

#### Experiment C5: ResNet-50

```python
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)
```

#### Experiment C6: EfficientNet-B4 (Optional/Stretch)

> [!NOTE]
> **Only run this if you have remaining Kaggle GPU quota.**

```python
model = smp.Unet(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)
```

> [!NOTE]
> Use the **best loss function** from experiments C1–C4 for these backbone experiments. This way you're comparing backbones under the best-known training configuration.

### Deliverables for Member C

- [ ] 6 experiments total (C1–C6), each with CSV results
- [ ] Loss function comparison table:

```markdown
| Loss Function | mIoU | BG IoU | Intact IoU | Damaged IoU | Destroyed IoU |
|---|---|---|---|---|---|
| CE + Dice (baseline) | 0.46 | ? | ? | ? | ? |
| Focal Tversky (0.3/0.7/2.0) | ? | ? | ? | ? | ? |
| Focal Tversky (0.25/0.75/1.33) | ? | ? | ? | ? | ? |
| Phased (FT → +Lovász) | ? | ? | ? | ? | ? |
```

- [ ] Backbone comparison table:

```markdown
| Backbone | mIoU | Damaged IoU | Destroyed IoU | Params | Train Time |
|---|---|---|---|---|---|
| ResNet-34 (baseline) | ? | ? | ? | 21.8M | ? hrs |
| ResNet-50 | ? | ? | ? | 25.6M | ? hrs |
| EfficientNet-B4 | ? | ? | ? | 19.3M | ? hrs |
```

- [ ] A short write-up: which loss function won, which backbone won, and the recommended combination

---

## 👤 Member D (Ridita): Resolution Ablation + Inference Pipeline Lead

### Your Mission
You own two things:
1. **Phase 1 (Days 1–10):** Resolution ablation experiments + building the inference-time pipeline (TTA, sliding window) that boosts mIoU without retraining + integrating the best results from Members A/B/C into a single "champion model"
2. **Phase 2 (Conditional — Days 11–15, ONLY if model performance plateaus):** Finding and merging external satellite damage datasets to increase training data for minority classes

> [!IMPORTANT]
> **Phase 2 is NOT automatic.** On Day 10, we hold a **phase gate review**. If Tier 1–3 improvements have pushed the Champion Model mIoU above ~0.52 and per-class Destroyed IoU above 0.30, we skip dataset extension entirely and move straight to the final combined report. Dataset extension only triggers if we've exhausted architecture/loss/augmentation levers and are still stuck.

---

### PHASE 1: Resolution Ablation + Inference Pipeline (Days 1–10)

#### Task D1: Resolution Ablation

**Hold constant:** Architecture (smp.Unet, resnet34), loss (CE+Dice), sampling (uniform), epochs (40).
**Change only:** The resolution/cropping strategy in Cell 5.

##### Experiment D1a: Baseline Resize (Already Done)

```python
# Current Cell 5
train_transform = A.Compose([
    A.Resize(512, 512),     # Downsamples 1024→512
    ...
])
```
mIoU ≈ 0.4619 (already known)

##### Experiment D1b: RandomCrop at Native Resolution

```python
train_transform = A.Compose([
    A.RandomCrop(512, 512),   # Crops from native 1024×1024
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# IMPORTANT: Validation must also use crops OR full-resolution sliding window
val_transform = A.Compose([
    A.CenterCrop(512, 512),   # Deterministic center crop for validation
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

##### Experiment D1c: Larger Crops (768×768) — if VRAM allows

```python
# Check if T4 can handle 768×768 at batch size 8
IMAGE_SIZE = 768
BATCH_SIZE = 8   # Reduced to fit VRAM

train_transform = A.Compose([
    A.RandomCrop(768, 768),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

> [!WARNING]
> If 768×768 at BS=8 causes OOM, try BS=4 with gradient accumulation of 2 steps (effective BS=8). Note this in your results.

**Deliverable D1:**
- Resolution comparison table:

```markdown
| Resolution Strategy | Crop Size | Batch Size | mIoU | Damaged IoU | Destroyed IoU | Train Time |
|---|---|---|---|---|---|---|
| Resize 1024→512 (baseline) | 512 | 16 | 0.46 | ? | ? | ? |
| RandomCrop 512 | 512 | 16 | ? | ? | ? | ? |
| RandomCrop 768 | 768 | 8 | ? | ? | ? | ? |
```

#### Task D2: Test-Time Augmentation (TTA) Pipeline

**What to do:** Implement TTA in the evaluation cell (Cell 9) so we can boost mIoU for FREE at inference time without any retraining.

```python
@torch.no_grad()
def predict_with_tta(model, image_batch, device='cuda'):
    """
    Apply 8 geometric transforms, predict on each, average softmax outputs.
    image_batch: [B, C, H, W] tensor
    Returns: [B, H, W] predicted class indices
    """
    transforms_and_inverses = [
        (lambda x: x,                                   lambda x: x),                                    # Original
        (lambda x: torch.flip(x, [3]),                   lambda x: torch.flip(x, [3])),                   # H-flip
        (lambda x: torch.flip(x, [2]),                   lambda x: torch.flip(x, [2])),                   # V-flip
        (lambda x: torch.flip(x, [2, 3]),                lambda x: torch.flip(x, [2, 3])),                # Both
        (lambda x: torch.rot90(x, 1, [2, 3]),            lambda x: torch.rot90(x, 3, [2, 3])),            # 90°
        (lambda x: torch.rot90(x, 2, [2, 3]),            lambda x: torch.rot90(x, 2, [2, 3])),            # 180°
        (lambda x: torch.rot90(x, 3, [2, 3]),            lambda x: torch.rot90(x, 1, [2, 3])),            # 270°
        (lambda x: torch.flip(torch.rot90(x, 1, [2,3]), [3]),
         lambda x: torch.rot90(torch.flip(x, [3]), 3, [2,3])),                                           # 90°+flip
    ]

    pred_sum = None
    for fwd, inv in transforms_and_inverses:
        augmented = fwd(image_batch)
        with autocast():
            logits = model(augmented)
        logits = inv(logits)
        probs = torch.softmax(logits, dim=1)

        if pred_sum is None:
            pred_sum = probs
        else:
            pred_sum += probs

    return (pred_sum / len(transforms_and_inverses)).argmax(dim=1)
```

**Deliverable D2:**
- Modified Cell 9 with TTA evaluation
- Comparison: mIoU with vs without TTA on the SAME trained model
- Report the exact mIoU boost

#### Task D3: Sliding Window Inference with Overlap Stitching

**What to do:** For models trained with `A.RandomCrop(512, 512)`, implement evaluation at native 1024×1024 resolution using overlapping 512×512 crops.

```python
@torch.no_grad()
def sliding_window_predict(model, image, crop_size=512, overlap=128, device='cuda'):
    """
    Predict on a full-resolution image using overlapping crops.
    image: [C, H, W] tensor (single image, normalized)
    Returns: [H, W] predicted class indices
    """
    C, H, W = image.shape
    step = crop_size - overlap
    pred_sum = torch.zeros(NUM_CLASSES, H, W, device=device)
    count = torch.zeros(1, H, W, device=device)

    positions = []
    for y in range(0, H - crop_size + 1, step):
        for x in range(0, W - crop_size + 1, step):
            positions.append((y, x))
    # Add edge-anchored positions
    positions.append((H - crop_size, W - crop_size))
    positions.append((0, W - crop_size))
    positions.append((H - crop_size, 0))

    for y, x in positions:
        crop = image[:, y:y+crop_size, x:x+crop_size].unsqueeze(0).to(device)
        with autocast():
            logit = model(crop)
        pred_sum[:, y:y+crop_size, x:x+crop_size] += logit.squeeze(0)
        count[:, y:y+crop_size, x:x+crop_size] += 1

    return (pred_sum / count.clamp(min=1)).argmax(dim=0)
```

**Deliverable D3:**
- Sliding window evaluation function
- Comparison: mIoU with resize-based eval vs sliding window eval on same model
- Report the exact mIoU boost

#### Task D4: Champion Model Integration (Days 9–10)

**What to do:** After Members A, B, and C complete their ablations, combine the best-of-each into a single experiment:

- Best architecture from Member B
- Best loss function from Member C
- Best backbone from Member C
- WeightedRandomSampler from Member A
- Copy-Paste augmentation from Member A
- Best resolution strategy from your D1 experiments
- TTA + sliding window at inference from your D2/D3

Train this combined configuration for 40 epochs and report the final mIoU.

**Deliverable D4:**
- Champion model notebook combining all best settings
- CSV results with per-class IoU
- Comparison: champion mIoU vs baseline mIoU (the final "how much did we improve" number)

---

### PHASE 2: Dataset Extension (CONDITIONAL — Only If Plateau Detected)

> [!CAUTION]
> **Do NOT start Phase 2 unless the Phase Gate Review on Day 10 determines it is necessary.** Phase 2 triggers ONLY if: per-class IoU on Destroyed is still below 0.30 after the champion model from Phase 1, OR overall mIoU has not improved beyond ~0.52 despite all Tier 1–3 improvements.

#### Task D5: External Dataset Survey & Merging (If Triggered)

**Goal:** Find publicly available satellite/aerial damage assessment datasets with building damage labels that can augment our DisasterM3 training set, specifically to add more Damaged/Destroyed examples.

**Datasets to investigate:**

| Dataset | Source | Labels | Expected Benefit |
|---|---|---|---|
| **xBD (xView2)** | [xview2.org](https://xview2.org/) | 4-level damage (no damage, minor, major, destroyed) on building polygons | Direct match to our taxonomy. Largest public disaster dataset (~22K images). |
| **CrowdAI Mapping Challenge** | [crowdai.org](https://www.crowdai.org/challenges/mapping-challenge) | Building footprint segmentation (binary) | Useful for Stage 1 of two-stage pipeline (building vs background). |
| **SpaceNet** | [spacenet.ai](https://spacenet.ai/) | Building footprints from various cities | Binary segmentation, good for pre-training. |
| **ASONAM Disaster** | Academic papers | Various disaster damage labels | May need label harmonization. |

**What to do for each dataset:**
1. Download a sample (100–500 images)
2. Inspect the label format — are the damage categories compatible with ours?
3. Write a conversion script that maps their labels to our 4-class scheme:
   ```
   0 = Background
   1 = Intact (no damage / minor damage)
   2 = Damaged (major damage)
   3 = Destroyed
   ```
4. Resize/crop to 1024×1024 to match our image dimensions
5. Generate combined masks in the same format as Cell 4's output

```python
# ── Template: Dataset Conversion Script ──
def convert_external_dataset(input_dir, output_dir, label_mapping):
    """
    Convert external dataset to DisasterM3 format.
    label_mapping: dict mapping source labels to our {0,1,2,3}
    Example: {"no-damage": 1, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
    """
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)

    converted = 0
    for img_file in Path(input_dir).glob("*.png"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        mask_file = str(img_file).replace("images", "masks")
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        new_mask = np.zeros_like(mask)
        for src_label, dst_label in label_mapping.items():
            new_mask[mask == src_label] = dst_label

        if img.shape[:2] != (1024, 1024):
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            new_mask = cv2.resize(new_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f"{output_dir}/images/{img_file.name}", img)
        cv2.imwrite(f"{output_dir}/masks/{img_file.name}", new_mask)
        converted += 1

    print(f"Converted {converted} image-mask pairs to {output_dir}")
```

**Deliverable D5 (if triggered):**
- External dataset survey report
- Conversion scripts for at least 1 external dataset
- Merged dataset pixel distribution analysis showing improved class balance
- Re-train champion model on merged dataset, report mIoU delta

### Final Deliverables Checklist for Member D

**Phase 1 (Always):**
- [ ] Resolution ablation table (D1a/D1b/D1c)
- [ ] TTA implementation + mIoU comparison (D2)
- [ ] Sliding window implementation + mIoU comparison (D3)
- [ ] Champion model integrating best results from all members (D4)
- [ ] CSV results files for each experiment

**Phase 2 (Only if plateau detected at Phase Gate):**
- [ ] External dataset survey report
- [ ] Conversion script for at least 1 external dataset
- [ ] Merged dataset pixel distribution analysis
- [ ] Re-trained champion model on merged data + mIoU comparison

---

## Timeline & Coordination (15 Days)

> [!IMPORTANT]
> Each Kaggle T4 session gives you ~11.5 hours of GPU time. A 40-epoch training run takes roughly 4–6 hours. That means **each member can run 1–2 experiments per day** if they queue them back-to-back. Plan accordingly.

```
═══════════════════════════════════════════════════════════════════
 DAYS 1–3: FOUNDATION (All members start in parallel)
═══════════════════════════════════════════════════════════════════

Day 1:
├── Member A: Dataset Audit (A1) — pixel counts, class presence histogram
├── Member B: Run baseline B1 (U-Net, verify 0.46 mIoU)
├── Member C: Run baseline C1 (CE+Dice, verify 0.46 mIoU)
└── Member D: Run baseline D1a (Resize 512, verify 0.46 mIoU)

Day 2:
├── Member A: Implement WeightedRandomSampler (A2) + run 10-epoch comparison
├── Member B: Run B2 (U-Net++) + B3 (U-Net++ scse) — queue back-to-back
├── Member C: Run C2 (Focal Tversky 0.3/0.7/2.0)
└── Member D: Run D1b (RandomCrop 512) + implement TTA (D2)

Day 3:
├── Member A: Start Copy-Paste instance extraction (A3 — offline step)
├── Member B: Run B4 (DeepLabV3+) + B5 (ResNet50 U-Net) — queue back-to-back
├── Member C: Run C3 (Focal Tversky 0.25/0.75/1.33)
└── Member D: Run D1c (768×768 crops) + implement sliding window (D3)

═══════════════════════════════════════════════════════════════════
 DAYS 4–7: DEEP ABLATION (All members running experiments)
═══════════════════════════════════════════════════════════════════

Day 4:
├── Member A: Complete Copy-Paste extraction + run 10-epoch comparison
├── Member B: Run B6 (MA-Net)
├── Member C: Run C4 (Phased Compound: Focal Tversky → +Lovász)
└── Member D: Run TTA comparison on baseline model (mIoU with vs without)

Day 5:
├── Member A: Share finalized data pipeline (sampler + copy-paste) with team
├── Member B: Compile architecture comparison table (B1–B6)
├── Member C: Run C5 (ResNet-50 backbone with best loss from C1–C4)
└── Member D: Run sliding window comparison on RandomCrop model

Day 6–7:
├── Member A: Help other members integrate Copy-Paste into their notebooks
├── Member B: Write architecture ablation write-up
├── Member C: Run C6 (EfficientNet-B4 backbone with best loss)
└── Member D: Compile resolution + inference comparison tables

═══════════════════════════════════════════════════════════════════
 DAY 8: SYNC — Share All Results
═══════════════════════════════════════════════════════════════════

Day 8 (Sync Meeting):
├── Member A delivers: Data audit report + sampler/copy-paste code
├── Member B delivers: Architecture comparison table + best arch
├── Member C delivers: Loss + backbone comparison tables + best combo
├── Member D delivers: Resolution + TTA + sliding window tables
└── ALL: Decide the "champion configuration" to combine

═══════════════════════════════════════════════════════════════════
 DAYS 9–10: CHAMPION MODEL
═══════════════════════════════════════════════════════════════════

Day 9:
└── Member D: Train champion model (best arch + best loss + best backbone
    + WeightedRandomSampler + Copy-Paste + best resolution) for 40 epochs

Day 10:
├── Member D: Evaluate champion model with TTA + sliding window
└── ALL: Phase Gate Review

─── PHASE GATE REVIEW (Day 10) ──────────────────────────────────
│ Check: Is champion mIoU > 0.52? Is Destroyed IoU > 0.30?      │
│ YES → Proceed to final report (Days 11–15).                   │
│ NO  → Member D starts Phase 2 dataset extension (Days 11–13). │
─────────────────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════════════
 DAYS 11–15: FINAL REPORT + CONDITIONAL PHASE 2
═══════════════════════════════════════════════════════════════════

Days 11–13:
├── Members A+B+C: Compile final ablation report + write-ups
├── Member D (if Phase 2): External dataset survey + conversion + retrain
└── Member D (if no Phase 2): Help with report + run any extra experiments

Days 14–15:
├── ALL: Finalize ablation report with master comparison table
├── ALL: Upload best model weights to HuggingFace
└── ALL: Final review and submission
```

### Quick Day-Count Summary Per Member

| Member | Days 1–3 | Days 4–7 | Day 8 | Days 9–10 | Days 11–15 |
|---|---|---|---|---|---|
| **A** | Audit + Sampler + Start Copy-Paste | Finish Copy-Paste + share pipeline | Present results | Support | Report writing |
| **B** | Run B1–B6 (6 architectures) | Compile tables + write-up | Present results | Support | Report writing |
| **C** | Run C1–C3 (3 loss functions) | Run C4–C6 (phased loss + backbones) | Present results | Support | Report writing |
| **D** | Run D1a–D1c (3 resolutions) + TTA/SW | Compile tables | Present results | **Champion model** | Report (or Phase 2) |

**Total experiments: ~16 across 4 members in 10 working days, with 5 days for integration and reporting.**

---

## Final Ablation Report Template

When all experiments are done, compile them into this master table:

```markdown
# DisasterM3 Segmentation Ablation Results

## Master Comparison Table

| # | Experiment | Arch | Backbone | Loss | Sampling | Augment | mIoU | Damaged IoU | Destroyed IoU | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Baseline | U-Net | R34 | CE+Dice | Uniform | Basic | 0.46 | ? | ? | ? |
| B2 | Arch: U-Net++ | U-Net++ | R34 | CE+Dice | Uniform | Basic | ? | ? | ? | ? |
| B3 | Arch: U-Net++ scse | U-Net++ scse | R34 | CE+Dice | Uniform | Basic | ? | ? | ? | ? |
| B4 | Arch: DeepLabV3+ | DLV3+ | R34 | CE+Dice | Uniform | Basic | ? | ? | ? | ? |
| B5 | Arch: ResNet50 U-Net | U-Net | R50 | CE+Dice | Uniform | Basic | ? | ? | ? | ? |
| B6 | Arch: MA-Net | MA-Net | R34 | CE+Dice | Uniform | Basic | ? | ? | ? | ? |
| C2 | Loss: FT | U-Net | R34 | FT 0.3/0.7/2.0 | Uniform | Basic | ? | ? | ? | ? |
| C3 | Loss: FT v2 | U-Net | R34 | FT 0.25/0.75/1.33 | Uniform | Basic | ? | ? | ? | ? |
| C4 | Loss: Phased | U-Net | R34 | FT→+Lovász | Uniform | Basic | ? | ? | ? | ? |
| C5 | Backbone: R50 | U-Net | R50 | Best loss | Uniform | Basic | ? | ? | ? | ? |
| C6 | Backbone: EB4 | U-Net | EB4 | Best loss | Uniform | Basic | ? | ? | ? | ? |
| A2 | Sampling: WRS | U-Net | R34 | CE+Dice | WRS 10× | Basic | ? | ? | ? | ? |
| A3 | Aug: CopyPaste | U-Net | R34 | CE+Dice | Uniform | CopyPaste | ? | ? | ? | ? |
| BEST | All combined | Best | Best | Best | WRS | CopyPaste | ? | ? | ? | ? |

## Key Findings
1. Best architecture: ___
2. Best loss function: ___
3. Best backbone: ___
4. Impact of class-balanced sampling: ___
5. Impact of Copy-Paste augmentation: ___
6. Impact of TTA at inference: ___
7. Final best mIoU: ___
```
