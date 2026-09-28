# Deliverable D1: Resolution Ablation

### Resolution comparison table:
| Resolution Strategy | Crop Size | Batch Size | mIoU | Damaged IoU | Destroyed IoU | Train Time |
|---|---|---|---|---|---|---|
| Resize 1024→512 (baseline) | 512 | 16 | ~0.46 | N/A | N/A | ~4 hours |
| RandomCrop 512 | 512 | 16 | 0.4540 | 0.2750 | 0.1222 | ~5.5 hours |
| RandomCrop 768 | 768 | 8 | 0.4902 | 0.3192 | 0.1953 | ~7.9 hours |
