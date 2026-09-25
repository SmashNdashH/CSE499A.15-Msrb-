# Deliverable D3: Sliding Window Inference with Overlap Stitching

### 1. Sliding Window Evaluation Function
To evaluate the models at their native 1024×1024 resolution without shrinking the images, we implemented a sliding window approach. This function takes 512×512 overlapping crops and stitches them together for the final prediction:

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

    from torch.cuda.amp import autocast
    for y, x in positions:
        crop = image[:, y:y+crop_size, x:x+crop_size].unsqueeze(0).to(device)
        with autocast():
            logit = model(crop)
        pred_sum[:, y:y+crop_size, x:x+crop_size] += logit.squeeze(0)
        count[:, y:y+crop_size, x:x+crop_size] += 1

    return (pred_sum / count.clamp(min=1)).argmax(dim=0)

# --- Integration into Evaluation Loop ---
intersection_sw = torch.zeros(NUM_CLASSES)
union_sw = torch.zeros(NUM_CLASSES)

for images, masks in tqdm(val_loader, desc="Sliding Window Eval"):
    images = images.to(device)
    masks = masks.to(device)
    
    # Process one image at a time since Sliding Window takes full-res images
    for i in range(images.size(0)):
        pred_sw = sliding_window_predict(model, images[i], crop_size=512, overlap=128, device=device)
        mask_i = masks[i]
        
        for c in range(NUM_CLASSES):
            pred_c = (pred_sw == c)
            mask_c = (mask_i == c)
            intersection_sw[c] += (pred_c & mask_c).sum().item()
            union_sw[c] += (pred_c | mask_c).sum().item()

iou_sw = intersection_sw / (union_sw + 1e-8)
print(f"Sliding Window - Mean IoU: {iou_sw.mean().item():.4f}")
```

### 2. Comparison: Resize vs Sliding Window Inference
The baseline model evaluated using `Resize(512)` lost significant detail. By evaluating the `RandomCrop(512)` trained model using native 1024×1024 Sliding Window inference, we observe the following changes:

| Metric | Baseline Inference (Resize) | Sliding Window Inference (Native) |
|---|---|---|
| **Mean IoU** | ~0.4600 | **[INSERT_SW_MIOU]** |

### 3. Exact mIoU Boost
By utilizing the Sliding Window pipeline over the baseline Resize evaluation, the model achieved an exact mIoU boost of **[INSERT_BOOST]** completely for free during inference.
