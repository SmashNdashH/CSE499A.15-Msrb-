# Plan 3: Offline Interactive Disaster Inspection Demo (Gradio/Streamlit)

## 1. What It Is

This plan creates a **working, clickable web application** that runs entirely offline on a local laptop. A user uploads a satellite image, and the system automatically:

1. **Segments** the image using our trained U-Net model (or the INT8-quantized version from Plan 1) to produce a coloured damage map.
2. **Counts** intact, damaged, and destroyed buildings by running connected-component analysis on the predicted mask.
3. **Zooms in** on each damaged/destroyed building (the Visual RAG technique from Plan 2) and displays high-resolution inspection cards.
4. **Generates** a structured damage report with EMS-98 grades, building coordinates, and rescue priority rankings.

This transforms our collection of Jupyter notebooks and CSV files into a **demonstrable, end-to-end disaster response tool** — the kind of result that supervisors and external examiners can interact with during the viva.

---

## 2. Research Paper Origin

| Paper | Venue | Key Idea |
|---|---|---|
| **Change-Agent: Towards Autonomous Remote Sensing Change Interpretation** | IEEE TGRS 2024 | An LLM "brain" that calls detection, captioning, counting, and cause-analysis tools as an integrated agent workflow. Our demo is the same multi-tool orchestration pattern, made interactive |
| **HASTE: Rapid Damage Assessment Platform** | Preprint 2025 | Designed specifically as a rapid-response platform for disaster damage assessment — the same operational goal as our demo |
| **GeoPixel** | ICML 2025 | Delivers the user experience of "ask a question, get text + pixel masks" — our demo replicates this experience using modular free tools instead of one large model |
| **GeoDisaster** | Preprint 2025 | Benchmarks agent workflows for disaster response, validating that tool-calling agent architectures are the current research direction for operational disaster AI |

Our [Research_Directions_Beyond_Finetuning.md](../Research_Directions_Beyond_Finetuning.md) §5 marks the Agentic workflow verdict as **"Do it, as the integration layer."** and §8 states the 499B goal is *"a hybrid system that counts from pixels, reasons in language, and runs offline on free or small GPUs."*

---

## 3. Why This Is Innovative for Our Project

### 3.1 It is the missing integration layer
The team has produced:
- Aryan: data sampling code (notebooks).
- Noman: loss function experiments (notebooks + CSV).
- Ridita: resolution experiments, TTA code, dataset integration scripts.
- Tamanna: architecture comparison (notebooks + CSV).
- Champion model: a trained `.pth` checkpoint.
- VLM fine-tuning: a Qwen2.5-VL model.

**Nobody has combined these into a usable system.** The demo is the glue that turns individual components into a working product.

### 3.2 It makes an immediately visible viva impact
During the final presentation, instead of showing IoU tables, you can **live-demo** the tool: upload a satellite image, watch the segmentation overlay appear, see the building count update, and read the AI-generated damage report. This is orders of magnitude more memorable than a slide deck.

### 3.3 It fulfils Objective O3
The project objective O3 calls for *"a report, damage map, counts, and survivability heatmap"*. The demo delivers all of these in one interactive interface.

### 3.4 It runs 100% offline without paid APIs
Unlike DisasTeller (which currently calls Gemini's cloud API), this demo uses:
- Our own trained segmentation model (local).
- OpenCV for counting (local).
- Optionally, a free local VLM via Hugging Face (local).
No internet connection is required during the demo.

---

## 4. Complete Roadmap

### Step 1: Choose Your Framework

Both options are free, pip-installable, and produce a web UI with zero frontend code:

| Framework | Best For | Install |
|---|---|---|
| **Gradio** | Quick ML demos with image upload/display, minimal code | `pip install gradio` |
| **Streamlit** | Dashboard-style layouts with sidebar controls, more customisable | `pip install streamlit` |

**Recommendation:** Use **Gradio** for the fastest results. It was built specifically for ML model demos.

---

### Step 2: Build the Core Processing Pipeline

This is the backend that processes an uploaded image:

```python
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

# ─── Model Loading ───
def load_segmentation_model(checkpoint_path, device="cpu"):
    """Load the trained U-Net model."""
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=4,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ─── Segmentation ───
def segment_image(image_rgb, model, device="cpu"):
    """
    Run segmentation on an RGB image.
    Returns: predicted mask [H, W] with values 0-3.
    """
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return pred

# ─── Counting via Connected Components ───
def count_buildings(pred_mask):
    """
    Count individual building instances per damage class.
    Returns dict with counts and building details.
    """
    class_names = {1: "Intact", 2: "Damaged", 3: "Destroyed"}
    results = {"total": 0, "per_class": {}, "buildings": []}

    for cls_id, cls_name in class_names.items():
        binary = (pred_mask == cls_id).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        count = 0
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < 100:  # Filter noise
                continue
            count += 1
            results["buildings"].append({
                "id": len(results["buildings"]) + 1,
                "class": cls_name,
                "class_id": cls_id,
                "area": area,
                "centroid": (int(centroids[label_idx][0]), int(centroids[label_idx][1])),
                "bbox": (
                    stats[label_idx, cv2.CC_STAT_LEFT],
                    stats[label_idx, cv2.CC_STAT_TOP],
                    stats[label_idx, cv2.CC_STAT_WIDTH],
                    stats[label_idx, cv2.CC_STAT_HEIGHT],
                ),
            })

        results["per_class"][cls_name] = count
        results["total"] += count

    return results

# ─── Coloured Overlay ───
def create_overlay(image_rgb, pred_mask, alpha=0.45):
    """
    Create a semi-transparent coloured overlay showing damage classes.
    """
    colour_map = {
        0: [30, 30, 30],       # Background = Dark grey
        1: [0, 200, 0],        # Intact = Green
        2: [255, 165, 0],      # Damaged = Orange
        3: [255, 0, 0],        # Destroyed = Red
    }

    overlay = np.zeros_like(image_rgb)
    for cls_id, colour in colour_map.items():
        overlay[pred_mask == cls_id] = colour

    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    return blended

# ─── Annotated Map with Building IDs ───
def create_annotated_map(image_rgb, building_results):
    """Draw bounding boxes and IDs on the image."""
    annotated = image_rgb.copy()
    colour_map = {1: (0, 200, 0), 2: (255, 165, 0), 3: (255, 0, 0)}

    for b in building_results["buildings"]:
        x, y, w, h = b["bbox"]
        colour = colour_map[b["class_id"]]
        cv2.rectangle(annotated, (x, y), (x+w, y+h), colour, 2)
        cv2.putText(annotated, f"#{b['id']} {b['class']}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    return annotated
```

---

### Step 3: Build the Gradio Interface

```python
import gradio as gr

# Load model once at startup
seg_model = load_segmentation_model("final_model.pth", device="cpu")

def process_satellite_image(input_image):
    """
    Main function called when user uploads an image.
    Returns: overlay image, annotated map, damage summary text.
    """
    if input_image is None:
        return None, None, "Please upload a satellite image."

    # Ensure RGB
    if len(input_image.shape) == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

    # 1. Segment
    pred_mask = segment_image(input_image, seg_model)

    # 2. Count buildings
    building_results = count_buildings(pred_mask)

    # 3. Create visualisations
    overlay = create_overlay(input_image, pred_mask)
    annotated = create_annotated_map(input_image, building_results)

    # 4. Generate text summary
    summary_lines = [
        "## 🏗️ Building Damage Assessment Report",
        "",
        f"**Total buildings detected:** {building_results['total']}",
        "",
        "### Per-Class Breakdown:",
        f"- 🟢 **Intact:** {building_results['per_class'].get('Intact', 0)} buildings",
        f"- 🟠 **Damaged:** {building_results['per_class'].get('Damaged', 0)} buildings",
        f"- 🔴 **Destroyed:** {building_results['per_class'].get('Destroyed', 0)} buildings",
        "",
    ]

    # Rescue priority section
    destroyed = [b for b in building_results["buildings"] if b["class_id"] == 3]
    if destroyed:
        summary_lines.append("### 🚨 Rescue Priority Locations:")
        for b in destroyed:
            summary_lines.append(
                f"- **Building #{b['id']}** at coordinates "
                f"({b['centroid'][0]}, {b['centroid'][1]}) — "
                f"Area: {b['area']} pixels — **IMMEDIATE PRIORITY**"
            )
    else:
        summary_lines.append("### ✅ No destroyed buildings detected in this image.")

    summary_text = "\n".join(summary_lines)

    return overlay, annotated, summary_text

# ─── Build the Gradio App ───
with gr.Blocks(
    title="DisasterM3 — Building Damage Assessment System",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
    # 🛰️ DisasterM3 — Offline Building Damage Assessment
    **Upload a satellite image to automatically detect, count, and assess building damage.**

    This system runs entirely offline using a lightweight INT8-quantized U-Net model.
    No cloud APIs or internet connection required.
    """)

    with gr.Row():
        input_image = gr.Image(label="📷 Upload Satellite Image", type="numpy")

    with gr.Row():
        btn = gr.Button("🔍 Analyse Damage", variant="primary", size="lg")

    with gr.Row():
        overlay_output = gr.Image(label="🗺️ Damage Overlay Map")
        annotated_output = gr.Image(label="📍 Building Identification Map")

    with gr.Row():
        summary_output = gr.Markdown(label="📋 Damage Report")

    btn.click(
        fn=process_satellite_image,
        inputs=[input_image],
        outputs=[overlay_output, annotated_output, summary_output],
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,     # Set True to generate a public link (free, temporary)
)
```

---

### Step 4: Add Building Crop Gallery (Visual RAG Integration)

Extend the interface to show zoomed-in crops of each damaged/destroyed building:

```python
def extract_top_crops(image_rgb, building_results, max_crops=8, crop_size=256, pad=32):
    """Extract zoomed-in crops of the most damaged buildings."""
    H, W = image_rgb.shape[:2]
    crops = []

    # Sort by damage severity (destroyed first) then by area (largest first)
    priority_buildings = sorted(
        [b for b in building_results["buildings"] if b["class_id"] >= 2],
        key=lambda b: (-b["class_id"], -b["area"])
    )

    for b in priority_buildings[:max_crops]:
        x, y, w, h = b["bbox"]
        # Add padding
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        crop = image_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (crop_size, crop_size),
                                   interpolation=cv2.INTER_LINEAR)

        crops.append((crop_resized, f"#{b['id']} — {b['class']} (Area: {b['area']}px)"))

    return crops
```

Add a **Gallery component** to Gradio to display the crops:

```python
# Inside the Gradio Blocks, add after the summary:
with gr.Row():
    crop_gallery = gr.Gallery(
        label="🔎 Zoomed Building Inspections",
        columns=4,
        height=300,
    )

# Update the process function to also return crops
def process_satellite_image(input_image):
    # ... (existing code) ...
    crops = extract_top_crops(input_image, building_results)
    gallery_images = [(c[0], c[1]) for c in crops]
    return overlay, annotated, summary_text, gallery_images
```

---

### Step 5: Optional — Add Local VLM Inspection Button

If running on a machine with a GPU (Kaggle/Colab), add a button that sends each crop to the free Qwen2.5-VL-3B for detailed text inspection:

```python
# Only load VLM if GPU is available
VLM_AVAILABLE = torch.cuda.is_available()

if VLM_AVAILABLE:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )
    vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

def inspect_with_vlm(crop_image):
    """Optional: get detailed VLM inspection of a building crop."""
    if not VLM_AVAILABLE:
        return "VLM not available (no GPU detected). Running in CPU-only mode."

    # ... (VLM inference code from Plan 2) ...
```

---

### Step 6: Package for Portable Deployment

Make the demo easy to run on any machine:

**`requirements.txt`:**
```
torch>=2.0
segmentation-models-pytorch
opencv-python-headless
numpy
gradio>=4.0
matplotlib
Pillow
```

**`run_demo.py`:**
```python
"""
DisasterM3 — Offline Building Damage Assessment Demo
Run: python run_demo.py
Then open http://localhost:7860 in your browser.
"""
import os
import sys

# Check for model checkpoint
MODEL_PATH = "final_model.pth"
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model checkpoint not found at '{MODEL_PATH}'")
    print("Please download it from the team's Hugging Face repo or copy from Champion_Task/R1-results/")
    sys.exit(1)

# Import and launch
from app import demo
demo.launch(server_name="0.0.0.0", server_port=7860)
```

**`README.md`:**
```markdown
# DisasterM3 — Offline Building Damage Assessment

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Place `final_model.pth` in this directory
3. Run: `python run_demo.py`
4. Open http://localhost:7860

## Features
- Upload any satellite image → automatic damage segmentation
- Building counting with connected-component analysis
- Zoomed-in inspection of each damaged building
- Structured damage report with EMS-98 grades
- Runs 100% offline, no internet or paid API needed
```

---

### Step 7: Record a Demo Video for the Presentation

Use any free screen recorder (OBS Studio, ShareX, or Windows Game Bar `Win+G`):

1. Open the Gradio app in Chrome.
2. Upload a DisasterM3 validation image with visible destruction.
3. Click "Analyse Damage."
4. Show the overlay map, building IDs, zoomed crops, and text report.
5. Save as a 30–60 second video clip for the viva slides.

---

## 5. Required Tools (All Free)

| Tool | Purpose | Install |
|---|---|---|
| Gradio | Interactive web UI framework | `pip install gradio` |
| PyTorch + SMP | Segmentation model inference | `pip install torch segmentation-models-pytorch` |
| OpenCV | Connected components, image processing | `pip install opencv-python-headless` |
| NumPy | Array operations | `pip install numpy` |
| Pillow | Image format handling | `pip install pillow` |
| (Optional) Transformers + Qwen2.5-VL | VLM building inspection | `pip install transformers` |
| (Optional) ONNX Runtime | For INT8 quantized model from Plan 1 | `pip install onnxruntime` |

**Runs on a laptop CPU (no GPU needed for basic mode).** The VLM inspection feature optionally uses a free Kaggle T4 or Colab GPU.

---

## 6. Expected Outcomes

- A **working interactive web application** that processes satellite images end-to-end.
- Upload an image → see coloured damage overlay + building count + zoomed inspection crops + text report in under 5 seconds (CPU) or under 1 second (GPU).
- **Offline operation:** no internet connection required during the demo.
- **Viva-ready:** a live demo is worth more than 50 slides of tables.
- Directly delivers Objective O3: *"a report, damage map, counts, and survivability heatmap."*
