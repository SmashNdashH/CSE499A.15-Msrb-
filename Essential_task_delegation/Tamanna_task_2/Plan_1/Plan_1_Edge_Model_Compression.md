# Plan 1: Edge Model Compression & Real-World Latency Benchmark

## 1. What It Is

This task takes the team's best trained PyTorch segmentation checkpoint (`.pth` file, ~130 MB) and converts it into an optimised, compressed model that can run fast on cheap hardware — a laptop CPU, a Jetson Nano, or a field tablet — without needing a GPU at all.

The core technique is **Post-Training INT8 Quantization**: converting model weights from 32-bit floating-point numbers down to 8-bit integers. This shrinks the model by roughly 75% (from ~130 MB to ~33 MB) and makes inference 2–4× faster on CPUs, while keeping accuracy nearly identical (typically less than 1% mIoU drop).

The final deliverable is a **comparison table** showing model size, inference speed (ms per image), memory usage, and accuracy retention — the exact evidence needed to defend our capstone title *"Resource-Efficient Neural Architectures for Post-Disaster Building Damage Assessment."*

---

## 2. Research Paper Origin

| Paper | Venue | Key Idea |
|---|---|---|
| **EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything** | CVPR 2024 | Distilled SAM from 636M → ~10M parameters with minimal accuracy loss; proved that large vision models can be compressed aggressively for edge deployment |
| **QLoRA: Efficient Finetuning of Quantized LLMs** | NeurIPS 2023 | Showed 4-bit quantization of large models preserves task performance; established INT4/INT8 quantization as standard practice |
| **AWQ: Activation-aware Weight Quantization** | MLSys 2024 | Demonstrated that not all weights matter equally — protecting salient weights during quantization preserves accuracy at extreme compression ratios |
| **Grace: Satellite–Ground VLM Collaboration** | Preprint 2025 | Proposed a 2B model on a satellite communicating with a 7B model on the ground — the same two-tier deployment philosophy we adopt |

Our [Research_Directions_Beyond_Finetuning.md](../Research_Directions_Beyond_Finetuning.md) §5 lists edge deployment as **"Highest priority — achievable, clearly measurable, and the most defensible contribution for a resource-efficient project."**

---

## 3. Why This Is Innovative for Our Project

### 3.1 It directly fulfils the capstone title
Our project is called *"Resource-Efficient Neural Architectures…"*. Without measured evidence of resource efficiency, the title is just a label. This plan produces the hard numbers — file size, latency, FPS, RAM — that prove the claim.

### 3.2 Nobody in the team has done this
- **Aryan** worked on data sampling and augmentation.
- **Noman** worked on loss functions and backbones.
- **Ridita** worked on resolution, inference pipelines, and dataset integration.
- **Tamanna (you)** worked on architecture ablation (B2–B5).

Nobody has taken any trained checkpoint and actually deployed or compressed it. This is untouched territory.

### 3.3 It requires zero GPU training hours
The entire workflow runs on your **local laptop CPU**. No Kaggle GPU quota is consumed. No risk of session timeouts or lost training runs.

### 3.4 It produces publication-quality figures
Edge deployment benchmarks with bar charts (accuracy vs. latency trade-off curves) are standard in top-venue papers and make the final report look professional and rigorous.

---

## 4. Complete Roadmap

### Step 1: Gather the Trained Checkpoint

Locate the best available `.pth` file from the Champion Task experiments:

```
Champion_Task/R1-results/d4_R1/final_model.pth     (~130 MB)
```

If this file is on Hugging Face (the D4 notebook pushes checkpoints there), download it locally.

**Verify it loads correctly:**

```python
import segmentation_models_pytorch as smp
import torch

NUM_CLASSES = 4

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,       # We will load our own weights
    in_channels=3,
    classes=NUM_CLASSES,
)
model.load_state_dict(torch.load("final_model.pth", map_location="cpu"))
model.eval()
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"File size: {os.path.getsize('final_model.pth') / 1e6:.1f} MB")
```

---

### Step 2: Export to ONNX

ONNX (Open Neural Network Exchange) is the universal intermediate format. All quantization tools work on ONNX models.

```python
import torch

dummy_input = torch.randn(1, 3, 768, 768)   # Match our training crop size

torch.onnx.export(
    model,
    dummy_input,
    "damage_model_fp32.onnx",
    input_names=["image"],
    output_names=["segmentation_mask"],
    dynamic_axes={
        "image": {0: "batch", 2: "height", 3: "width"},
        "segmentation_mask": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=17,
)
print(f"ONNX FP32 model saved. Size: {os.path.getsize('damage_model_fp32.onnx') / 1e6:.1f} MB")
```

---

### Step 3: Apply INT8 Dynamic Quantization

Dynamic quantization quantizes weights offline and activations on-the-fly during inference. It is the simplest method and requires no calibration data.

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="damage_model_fp32.onnx",
    model_output="damage_model_int8_dynamic.onnx",
    weight_type=QuantType.QInt8,
)

fp32_size = os.path.getsize("damage_model_fp32.onnx") / 1e6
int8_size = os.path.getsize("damage_model_int8_dynamic.onnx") / 1e6
print(f"FP32: {fp32_size:.1f} MB → INT8: {int8_size:.1f} MB ({100*(1 - int8_size/fp32_size):.0f}% reduction)")
```

---

### Step 4: Apply INT8 Static Quantization (Optional, More Accurate)

Static quantization additionally calibrates activation ranges using a small representative dataset, producing tighter quantization and usually better accuracy than dynamic.

```python
from onnxruntime.quantization import (
    quantize_static, QuantType, CalibrationDataReader
)
import numpy as np

class DamageCalibrationReader(CalibrationDataReader):
    """Feed ~50-100 validation images through the model to calibrate activation ranges."""
    def __init__(self, val_images, input_name="image"):
        self.data = iter([
            {input_name: img.unsqueeze(0).numpy()} for img in val_images[:100]
        ])

    def get_next(self):
        return next(self.data, None)

# Prepare calibration data (load 100 validation images with same normalization as training)
calibration_reader = DamageCalibrationReader(val_images_list)

quantize_static(
    model_input="damage_model_fp32.onnx",
    model_output="damage_model_int8_static.onnx",
    calibration_data_reader=calibration_reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
)
```

---

### Step 5: Benchmark — Inference Speed on CPU

Run all three models (FP32, INT8 Dynamic, INT8 Static) on the **same 50 validation images** using ONNX Runtime's CPU provider and measure:

```python
import onnxruntime as ort
import time
import numpy as np

def benchmark_model(onnx_path, test_images, num_runs=50):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: test_images[0]})

    # Timed runs
    latencies = []
    for i in range(num_runs):
        img = test_images[i % len(test_images)]
        start = time.perf_counter()
        session.run(None, {input_name: img})
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "fps": 1000 / np.mean(latencies),
    }

# Run benchmarks
for model_path in ["damage_model_fp32.onnx", "damage_model_int8_dynamic.onnx", "damage_model_int8_static.onnx"]:
    result = benchmark_model(model_path, test_images)
    print(f"{model_path}: {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms  ({result['fps']:.1f} FPS)")
```

---

### Step 6: Benchmark — Accuracy Retention (mIoU)

Run all three models on the **full 645 validation images** and compute mIoU, comparing with the original PyTorch result:

```python
def compute_miou_onnx(onnx_path, val_pairs, num_classes=4):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pair in val_pairs:
        image = load_and_preprocess(pair["image_path"])   # Same normalization as training
        gt_mask = load_mask(pair["mask_path"])

        logits = session.run(None, {input_name: image[np.newaxis]})[0]
        pred = np.argmax(logits[0], axis=0)

        for t in range(num_classes):
            for p in range(num_classes):
                confusion[t, p] += np.sum((gt_mask == t) & (pred == p))

    # Compute per-class IoU
    ious = []
    for c in range(num_classes):
        intersection = confusion[c, c]
        union = confusion[c, :].sum() + confusion[:, c].sum() - intersection
        ious.append(intersection / max(union, 1))

    return {
        "miou": np.mean(ious),
        "per_class": ious,
    }
```

---

### Step 7: Compile the Final Report Table

This is the deliverable for the capstone report and viva slides:

```markdown
| Model Variant           | Size (MB) | Latency (ms) | FPS   | mIoU   | Destroyed IoU | Accuracy Drop |
|-------------------------|-----------|--------------|-------|--------|---------------|---------------|
| PyTorch FP32 (original) | ~130      | baseline     | —     | 0.5374 | 0.2686        | —             |
| ONNX FP32               | ~130      | ???          | ???   | ???    | ???           | ???           |
| ONNX INT8 Dynamic       | ~33       | ???          | ???   | ???    | ???           | < 1%          |
| ONNX INT8 Static        | ~33       | ???          | ???   | ???    | ???           | < 1%          |
```

Also measure and report:
- **Peak RAM usage** during inference (use `tracemalloc` or `psutil`).
- **Hardware specs** of the test machine (CPU model, RAM, OS).

---

### Step 8: Create Visualisation Figures

For the report and slides, generate:

1. **Bar chart:** Model size comparison (FP32 vs INT8).
2. **Bar chart:** Latency comparison (ms per image).
3. **Scatter plot:** mIoU vs. Latency trade-off (each model variant as a point).
4. **Side-by-side masks:** Same validation image predicted by FP32 and INT8, showing they look virtually identical.

```python
import matplotlib.pyplot as plt

models = ["PyTorch FP32", "ONNX FP32", "INT8 Dynamic", "INT8 Static"]
sizes = [130, 130, 33, 33]      # Fill with actual measured values
latencies = [450, 380, 120, 110] # Fill with actual measured values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(models, sizes, color=["#4A90D9", "#5BA5E8", "#2ECC71", "#27AE60"])
axes[0].set_ylabel("Model Size (MB)")
axes[0].set_title("Model Compression")

axes[1].bar(models, latencies, color=["#E74C3C", "#F39C12", "#2ECC71", "#27AE60"])
axes[1].set_ylabel("Latency (ms / image)")
axes[1].set_title("CPU Inference Speed")

plt.tight_layout()
plt.savefig("edge_benchmark_comparison.png", dpi=150)
```

---

## 5. Required Tools (All Free)

| Tool | Purpose | Install |
|---|---|---|
| PyTorch | Load the trained checkpoint | `pip install torch` |
| segmentation-models-pytorch | Rebuild the model architecture | `pip install segmentation-models-pytorch` |
| ONNX | Model export format | `pip install onnx` |
| ONNX Runtime | CPU inference + quantization | `pip install onnxruntime` |
| NumPy, OpenCV, Matplotlib | Data handling and visualisation | `pip install numpy opencv-python matplotlib` |

**No GPU needed. Everything runs on your laptop CPU.**

---

## 6. Expected Outcomes

- **Model size:** ~130 MB → ~33 MB (**~75% reduction**).
- **CPU inference speed:** ~400 ms → ~100–120 ms per image (**~3–4× speedup**).
- **Accuracy retention:** mIoU drops by less than 0.5–1.0% (expected to remain above 0.53).
- **Destroyed IoU:** Negligible change (quantization affects all classes roughly equally).

These numbers directly prove the capstone's "resource-efficient" claim with hard, reproducible evidence.
