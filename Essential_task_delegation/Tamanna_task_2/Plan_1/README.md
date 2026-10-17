# Plan 1: Edge Model Compression & Real-World Latency Benchmark

## Overview

This folder contains the complete implementation for **INT8 quantization** of our trained disaster damage segmentation model. The pipeline converts our ~130 MB PyTorch checkpoint into a ~33 MB ONNX INT8 model that runs **3–4× faster on CPU** with **<1% mIoU drop**.

This directly proves our capstone title: *"Resource-Efficient Neural Architectures for Post-Disaster Building Damage Assessment."*

---

## Folder Structure

```
Plan_1/
├── README.md                        ← This file
├── requirements_plan1.txt           ← Dependencies (pip install -r)
├── config.py                        ← All paths & hyperparameters in one place
├── step1_verify_checkpoint.py       ← Load & verify the .pth checkpoint
├── step2_export_onnx.py             ← Export PyTorch → ONNX FP32
├── step3_quantize_dynamic.py        ← INT8 Dynamic Quantization (no calibration data)
├── step4_quantize_static.py         ← INT8 Static Quantization (calibration-based)
├── step5_benchmark_speed.py         ← Latency benchmark (all 3 models on CPU)
├── step6_benchmark_accuracy.py      ← mIoU evaluation (all models on validation set)
├── step7_generate_report.py         ← Compile final comparison table (CSV + Markdown)
├── step8_visualize.py               ← Generate bar charts & side-by-side mask plots
├── run_all.py                       ← Run the entire pipeline end-to-end
└── outputs/                         ← All generated files land here
    ├── damage_model_fp32.onnx
    ├── damage_model_int8_dynamic.onnx
    ├── damage_model_int8_static.onnx
    ├── benchmark_results.csv
    ├── edge_benchmark_comparison.png
    ├── accuracy_vs_latency.png
    └── side_by_side_masks.png
```

---

## How to Run

### Prerequisites
1. Download the trained checkpoint (`final_model.pth`) from Hugging Face or the Kaggle output.
2. Place it at the path specified in `config.py` (default: `../../Champion_Task/R1-results/d4_R1/final_model.pth`).
3. Make sure validation images and masks are accessible (paths are in `config.py`).

### Install Dependencies
```bash
pip install -r requirements_plan1.txt
```

### Run Everything at Once
```bash
python run_all.py
```

### Or Run Step-by-Step
```bash
python step1_verify_checkpoint.py
python step2_export_onnx.py
python step3_quantize_dynamic.py
python step4_quantize_static.py       # Optional — needs validation images
python step5_benchmark_speed.py
python step6_benchmark_accuracy.py
python step7_generate_report.py
python step8_visualize.py
```

---

## Expected Outcomes

| Metric | Before | After (INT8) |
|--------|--------|-------------|
| Model Size | ~130 MB | ~33 MB (**75% reduction**) |
| CPU Latency | ~400 ms/image | ~100–120 ms/image (**3–4× faster**) |
| mIoU | 0.5374 | >0.53 (**<1% drop**) |
| Destroyed IoU | 0.2686 | ~0.265 (**negligible change**) |

---

## Notes

- **No GPU needed.** Everything runs on your laptop CPU.
- **No retraining.** This is post-training quantization only.
- All outputs are saved in the `outputs/` folder.
- The report table is what goes into the capstone report and viva slides.
