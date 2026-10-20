import segmentation_models_pytorch as smp
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

# =====================================================================
# Experiment B1: Baseline U-Net (Already Done)
# Expected mIoU: ~0.46
# =====================================================================
model_B1 = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

# =====================================================================
# Experiment B2: U-Net++
# =====================================================================
model_B2 = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

# =====================================================================
# Experiment B3: U-Net++ with scse Attention
# =====================================================================
model_B3 = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    decoder_attention_type="scse",
).to(DEVICE)

# =====================================================================
# Experiment B4: DeepLabV3+
# Note: May need batch size reduced to 8 if OOM
# =====================================================================
model_B4 = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

# =====================================================================
# Experiment B5: U-Net (Plain) with ResNet-50 Encoder
# Note: Isolates the effect of a deeper backbone.
# =====================================================================
model_B5 = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

# =====================================================================
# Experiment B6: MA-Net (Multi-scale Attention Network)
# Note: May need batch size reduced to 8 if OOM
# =====================================================================
model_B6 = smp.MAnet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)
