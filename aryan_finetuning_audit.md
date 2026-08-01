# Audit: Aryan's U-Net Fine-Tuning Notebook

This document provides a cell-by-cell post-mortem of Aryan's `segmentation-model.ipynb` fine-tuning run. It identifies critical bugs that led to sub-optimal performance (mIoU 0.3461 vs Abrar's 0.4286) and extracts the highly successful optimizations (AMP, mIoU tracking) that should be ported into any future Version 2 (V2) training runs.

---

## 1. Cell-by-Cell Audit

### Cell 2: Configuration
* **What went wrong:** Aryan reverted `NUM_CLASSES` to `3` (collapsing the "Destroyed" class into "Damaged"). This violates the Hybrid Pipeline methodology, which relies on the U-Net accurately identifying 4 separate classes (Background, Intact, Damaged, Destroyed) before passing them to the VLM.
* **What went right:** Increased `BATCH_SIZE = 16`, which is much better for GPU utilization (made possible by his optimizations in Cell 9).

### Cell 4: Mask Generation
* **What went wrong:** Aryan hardcoded the mask synthesis to only map 3 classes, permanently destroying the distinction between heavily damaged and fully destroyed buildings in the ground truth data.
* **What went right:** Aryan properly saved the synthesized masks to the Kaggle working directory (`/kaggle/working/combined_masks`) instead of keeping them in RAM, completely solving the RAM Out-Of-Memory (OOM) crash.

### Cell 6: Model Architecture & Loss (CRITICAL BUG)
* **What went wrong:** Aryan introduced a fatal mismatch. In Cell 2, he set `NUM_CLASSES = 3`. However, in Cell 6, he defined the class pixel counts as a **4-element tensor**: `torch.tensor([6267276585, 395903867, 70675310, 22119406])`. 
* **Impact:** Calculating 4 class weights and passing them into a 3-class `CombinedLoss` (CrossEntropy + Dice) function scrambled the loss gradients. The model was mathematically punished incorrectly, which is why it plateaued at an mIoU of 0.34 and refused to learn further.

### Cell 9: The Training Loop
* **What went wrong:** Hard-capped training at 25 epochs. The model needs closer to 40 epochs to properly converge (Abrar's model converged at Epoch 22, but that was with a smaller batch size of 4. With a batch size of 16, convergence would likely take longer).
* **What went right (MASSIVE WIN):** 
  1. **Automatic Mixed Precision (AMP):** Aryan implemented `torch.cuda.amp.autocast()` and `GradScaler()`. This cuts VRAM usage in half, doubles training speed (finished in 1h 19m), and is the only reason a batch size of 16 was possible on a T4 GPU.
  2. **Dynamic mIoU Validation:** Aryan evaluated the model at the end of every epoch using Mean Intersection over Union (mIoU) and saved the `best_model.pth` based on `best_miou` rather than `best_val_loss`. This is the industry standard for segmentation.

---

## 2. Recommendations for V2 Fine-Tuning

If a second fine-tuning run is required in the future to push the mIoU even higher, the team should **merge Abrar's architectural stability with Aryan's speed optimizations.**

### The "Best of Both Worlds" Setup:
1. **Maintain 4 Classes:** Ensure `NUM_CLASSES = 4` across the config, the mask generation, and the loss weights.
2. **Batch Size:** Use `BATCH_SIZE = 16`.
3. **Epochs:** Train for `NUM_EPOCHS = 40` to guarantee convergence.
4. **Use AMP & mIoU Tracking:** Replace the standard training loop with the optimized AMP loop provided below.

### The Unified V2 Training Loop
For future runs, replace the training loop cell with this unified version:

```python
import time
import os
import datetime
import json
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm
from IPython.display import display, HTML
from torch.cuda.amp import GradScaler, autocast

training_start = time.time()
epoch_times = []

# ── Model ──
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = GradScaler() # AMP Optimization

def compute_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        # Mixed Precision Forward Pass
        with autocast():
            logits = model(images)
            loss, ce_val, dice_val = criterion(logits, masks)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_ious = [[] for _ in range(NUM_CLASSES)]
    
    for images, masks in tqdm(loader, desc="Val", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        with autocast():
            logits = model(images)
            loss, _, _ = criterion(logits, masks)
            
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        
        for pred, mask in zip(preds, masks):
            ious = compute_iou(pred.cpu(), mask.cpu(), NUM_CLASSES)
            for cls, iou in enumerate(ious):
                if not np.isnan(iou):
                    all_ious[cls].append(iou)
                    
    class_ious = [np.mean(cls_ious) if cls_ious else 0.0 for cls_ious in all_ious]
    miou = np.mean([iou for iou in class_ious if iou > 0])
    return total_loss / len(loader), miou

# ── Kaggle Safety Wall & Checkpointing Setup ──
TIME_LIMIT_HOURS = 11.5
deadline = time.time() + TIME_LIMIT_HOURS * 3600
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

if "best_miou" not in locals(): best_miou = 0.0
history = {"train_loss": [], "val_loss": [], "miou": []}

# ── UI Setup ──
progress_html = display(HTML(f"<div><progress value='0' max='{NUM_EPOCHS}' style='width:300px; height:20px; vertical-align: middle;'></progress> [0/{NUM_EPOCHS}]</div>"), display_id=True)
table_html = display(HTML("<table border='1' class='dataframe'><thead><tr style='text-align: left;'><th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>mIoU</th><th>Time</th></tr></thead><tbody></tbody></table>"), display_id=True)
table_rows = ""

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()
    if time.time() > deadline:
        print(f"\n⏰ 11.5 hour time budget reached! Saving and stopping gracefully.")
        break

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
    val_loss, miou = validate_one_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["miou"].append(miou)

    epoch_duration = time.time() - epoch_start
    epoch_times.append(epoch_duration)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    eta_seconds = int(avg_epoch_time * (NUM_EPOCHS - (epoch + 1)))
    eta_str = str(datetime.timedelta(seconds=eta_seconds))
    
    # ── Update UI ──
    progress_html.update(HTML(f"<div><progress value='{epoch+1}' max='{NUM_EPOCHS}' style='width:300px; height:20px; vertical-align: middle;'></progress> [{epoch+1}/{NUM_EPOCHS} &lt; ETA: {eta_str}]</div>"))
    table_rows += f"<tr><td>{epoch+1}</td><td>{train_loss:.4f}</td><td>{val_loss:.4f}</td><td>{miou:.4f}</td><td>{epoch_duration:.1f}s</td></tr>"
    table_html.update(HTML(f"<table border='1' class='dataframe'><thead><tr style='text-align: left;'><th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>mIoU</th><th>Time</th></tr></thead><tbody>{table_rows}</tbody></table>"))

    # Save based on best mIoU
    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
        
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": best_miou,
    }, f"/kaggle/working/checkpoints/unet_epoch_{epoch}.pth")

with open("/kaggle/working/training_history.json", "w") as f:
    json.dump(history, f, indent=2)
```
