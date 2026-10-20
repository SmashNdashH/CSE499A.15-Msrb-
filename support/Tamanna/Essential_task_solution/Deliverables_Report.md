# Architecture Ablation Results

## 1. Summary of Experiments
This report documents the results of 6 controlled experiments testing different semantic segmentation architectures on the DisasterM3 dataset. All models were trained for 40 epochs on Kaggle T4 GPUs using identical data augmentations and the baseline CE + Dice loss function. 

## 2. Architecture Comparison Table

| Architecture | mIoU | BG IoU | Intact IoU | Damaged IoU | Destroyed IoU | Train Time |
|---|---|---|---|---|---|---|
| U-Net (baseline) | 0.46 | (fill) | (fill) | (fill) | (fill) | (fill) hrs |
| U-Net++ | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) hrs |
| U-Net++ + scse | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) hrs |
| DeepLabV3+ | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) hrs |
| U-Net (ResNet50) | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) hrs |
| MA-Net | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) hrs |

## 3. Conclusion

*(Write 1 paragraph here stating which architecture won and why you think it performed better on this specific dataset — for example, if U-Net++ with scse won, you could mention how the attention mechanism helped it focus better on the small "Destroyed" buildings).*

