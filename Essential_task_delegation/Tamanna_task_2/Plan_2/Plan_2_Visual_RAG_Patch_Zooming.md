# Plan 2: Visual RAG — Patch-Level Zooming for Building Damage Inspection

## 1. What It Is

**Visual RAG (Retrieval-Augmented Generation)** solves a fundamental problem with free Vision-Language Models (VLMs): when you feed them a full 1024×1024 satellite image, they internally downsample it to fit their limited visual token budget (typically 512–1024 tokens). This turns individual buildings into blurry 2–3 pixel smudges, making the VLM unable to judge damage at the building level.

Visual RAG fixes this by treating the segmentation output as a **retrieval index**. Instead of feeding the whole image, we:

1. Use our trained segmentation model to locate the exact pixel coordinates of damaged and destroyed buildings.
2. Extract high-resolution **256×256 or 128×128 cropped patches** around each identified building from the **original 1024×1024 image** (no downsampling).
3. Feed only those sharp, zoomed-in patches into a free open-source VLM (e.g., Qwen2.5-VL-3B-Instruct).
4. The VLM inspects each building crop individually and produces a structured damage report grounded in the EMS-98 damage scale.

The result is a system where the segmentation model acts as **"eyes that point"** and the VLM acts as **"an expert inspector that explains"** — each doing what it's best at.

---

## 2. Research Paper Origin

| Paper | Venue | Key Idea |
|---|---|---|
| **ImageRAG: Dynamic Image Retrieval for Reference-Guided Generation in Remote Sensing** | IEEE Geoscience and Remote Sensing Magazine (GRSM), 2025 | Proposed retrieving only relevant high-resolution image patches for VLM analysis instead of feeding full images. Showed that focused patch retrieval dramatically improves VLM accuracy on remote sensing tasks |
| **GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks** | ICCV 2025 | Benchmarked VLMs on 31 geospatial sub-tasks. Even GPT-4o achieved only ~40% on spatial reasoning questions, proving that **VLMs alone cannot reliably localise or count** — they need help from specialist vision tools |
| **Change-Agent: Towards Autonomous Remote Sensing Change Interpretation** | IEEE TGRS 2024 | Demonstrated an LLM "brain" that calls a change-detection model as a tool, combining detection with language explanations |
| **GeoPixel** | ICML 2025 | A remote-sensing multimodal model that answers with pixel masks attached to text — the user experience we want to replicate using modular, free tools |

Our [Research_Directions_Beyond_Finetuning.md](../Research_Directions_Beyond_Finetuning.md) §5 identifies RAG as **"Do it. No training, cheap, and it targets the VLM's spatial blindness."**

---

## 3. Why This Is Innovative for Our Project

### 3.1 It solves a proven failure mode
In CSE499A, our VLM (Qwen2.5-VL-7B) fell **10.62 points** on building counting tasks compared to the base model. The core reason: the VLM tried to count small buildings from a heavily downsampled full satellite image. Visual RAG eliminates this problem by feeding it only sharp, zoomed-in building patches.

### 3.2 Nobody in the team has built this bridge
- The segmentation team (A/B/C/D) produced pixel masks but never connected them to language output.
- The VLM fine-tuning work in 499A used raw full images, not retrieval-guided patches.
- Nobody has built the pipeline that goes: **segmentation coordinates → crop extraction → VLM inspection → structured report**.

### 3.3 It requires zero model training
The segmentation model is already trained. The VLM (Qwen2.5-VL-3B-Instruct) is used directly from Hugging Face with zero fine-tuning. The entire contribution is the **retrieval and prompting pipeline** — pure Python engineering.

### 3.4 It produces dramatic demo results
In a viva or presentation, showing a satellite image where the system automatically zooms into each damaged building and writes an expert inspection note is far more impressive than a static IoU table.

---

## 4. Complete Roadmap

### Step 1: Generate Segmentation Masks on Validation Images

Use the team's best trained model to produce predicted damage masks on the 645 validation images:

```python
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np

# Load the best trained model
model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=4)
model.load_state_dict(torch.load("final_model.pth", map_location="cpu"))
model.eval()

def predict_mask(image_path, model):
    """Run segmentation on a single image and return the predicted class map."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize same as training
    img_tensor = img.astype(np.float32) / 255.0
    img_tensor = (img_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        logits = model(img_tensor)
        pred = logits.argmax(dim=1).squeeze(0).numpy()

    return pred  # Shape: [H, W], values: 0=BG, 1=Intact, 2=Damaged, 3=Destroyed
```

---

### Step 2: Extract Per-Building Bounding Box Crops

Use connected-component analysis to find individual building instances from the predicted mask, then crop high-resolution patches from the **original image**:

```python
def extract_building_crops(image_path, pred_mask, target_classes=[2, 3],
                           min_area=100, pad=32, crop_size=256):
    """
    Extract zoomed-in crops of damaged/destroyed buildings.

    Args:
        image_path: Path to the original 1024×1024 satellite image.
        pred_mask: Predicted segmentation mask [H, W].
        target_classes: Which classes to extract (2=Damaged, 3=Destroyed).
        min_area: Minimum pixel area to consider a building (filters noise).
        pad: Padding around the bounding box (pixels).
        crop_size: Final crop size for the VLM.

    Returns:
        List of dicts with keys: crop_image, bbox, class_id, area, building_id.
    """
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    H, W = original_img.shape[:2]

    buildings = []
    building_id = 0

    for cls_id in target_classes:
        binary = (pred_mask == cls_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            # Bounding box with padding
            x = max(0, stats[label_idx, cv2.CC_STAT_LEFT] - pad)
            y = max(0, stats[label_idx, cv2.CC_STAT_TOP] - pad)
            w = min(W - x, stats[label_idx, cv2.CC_STAT_WIDTH] + 2 * pad)
            h = min(H - y, stats[label_idx, cv2.CC_STAT_HEIGHT] + 2 * pad)

            # Crop from the ORIGINAL high-resolution image (not the mask)
            crop = original_img[y:y+h, x:x+w]

            # Resize to uniform size for the VLM
            crop_resized = cv2.resize(crop, (crop_size, crop_size),
                                       interpolation=cv2.INTER_LINEAR)

            class_name = "Damaged" if cls_id == 2 else "Destroyed"
            building_id += 1

            buildings.append({
                "building_id": building_id,
                "crop_image": crop_resized,
                "bbox": (x, y, w, h),
                "centroid": (int(centroids[label_idx][0]), int(centroids[label_idx][1])),
                "class_id": cls_id,
                "class_name": class_name,
                "area_pixels": area,
            })

    return buildings
```

---

### Step 3: Prepare the VLM Inspection Prompt

Design a structured prompt that tells the VLM exactly what to look for in each building crop:

```python
def build_inspection_prompt(building_info):
    """
    Create a structured prompt for the VLM to inspect a single building crop.
    """
    prompt = f"""You are an expert structural damage assessor inspecting satellite imagery
after a natural disaster. You are looking at a zoomed-in {building_info['crop_size']}×{building_info['crop_size']}
pixel crop of a single building that was predicted as "{building_info['class_name']}" by
an automated segmentation model.

Analyse this building crop and provide:

1. **Observed Damage:** What specific structural damage do you see?
   (e.g., roof collapse, wall failure, debris field, intact structure)

2. **EMS-98 Grade:** Assign a damage grade using the European Macroseismic Scale:
   - Grade 1: No damage
   - Grade 2: Slight damage (cracks in walls)
   - Grade 3: Moderate damage (large cracks, roof tile loss)
   - Grade 4: Heavy damage (partial structural failure, roof collapse)
   - Grade 5: Destruction (total or near-total collapse)

3. **Confidence:** How confident are you in this assessment? (High / Medium / Low)

4. **Rescue Priority:** Based on the damage, should search-and-rescue teams
   prioritise this location? (Immediate / Standard / Low)

Respond in a structured JSON format."""

    return prompt
```

---

### Step 4: Connect to a Free Open-Source VLM

Use **Qwen2.5-VL-3B-Instruct** (free on Hugging Face, runs on Kaggle T4 with ~6 GB VRAM):

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Load the free VLM (runs on Kaggle T4 or Google Colab)
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

def inspect_building(crop_image_rgb, prompt, vlm_model, processor):
    """
    Send a building crop + prompt to the VLM and get the inspection report.
    """
    pil_image = Image.fromarray(crop_image_rgb)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input],
        images=[pil_image],
        return_tensors="pt",
    ).to(vlm_model.device)

    with torch.no_grad():
        output_ids = vlm_model.generate(**inputs, max_new_tokens=300)

    response = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True)[0]
    return response
```

---

### Step 5: Build the Full End-to-End Pipeline

Connect all components into one clean function:

```python
def visual_rag_pipeline(image_path, model, vlm_model, processor):
    """
    Full Visual RAG pipeline:
    1. Segment the image to find damaged/destroyed buildings.
    2. Extract zoomed-in crops of each building.
    3. Send each crop to the VLM for detailed inspection.
    4. Compile a structured damage report.
    """
    # Step 1: Segmentation
    pred_mask = predict_mask(image_path, model)

    # Step 2: Extract building crops
    buildings = extract_building_crops(image_path, pred_mask)

    # Step 3: Count summary
    summary = {
        "total_buildings_detected": len(buildings),
        "damaged_count": sum(1 for b in buildings if b["class_id"] == 2),
        "destroyed_count": sum(1 for b in buildings if b["class_id"] == 3),
        "inspections": [],
    }

    # Step 4: VLM inspection for each building
    for building in buildings:
        prompt = build_inspection_prompt({
            "class_name": building["class_name"],
            "crop_size": 256,
        })

        inspection = inspect_building(building["crop_image"], prompt, vlm_model, processor)

        summary["inspections"].append({
            "building_id": building["building_id"],
            "location": building["centroid"],
            "predicted_class": building["class_name"],
            "area_pixels": building["area_pixels"],
            "vlm_assessment": inspection,
        })

    return summary
```

---

### Step 6: Generate the Visual Report with Annotated Map

Create a visual output that overlays building IDs and damage colours on the original satellite image:

```python
def create_annotated_map(image_path, buildings, pred_mask, save_path):
    """
    Draw coloured bounding boxes and building IDs on the original image.
    """
    img = cv2.imread(image_path)

    colour_map = {
        1: (0, 255, 0),    # Intact = Green
        2: (0, 165, 255),  # Damaged = Orange
        3: (0, 0, 255),    # Destroyed = Red
    }

    for b in buildings:
        x, y, w, h = b["bbox"]
        colour = colour_map[b["class_id"]]
        cv2.rectangle(img, (x, y), (x+w, y+h), colour, 2)
        cv2.putText(img, f"#{b['building_id']}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    cv2.imwrite(save_path, img)
    print(f"Annotated map saved to {save_path}")
```

---

### Step 7: Evaluate on DisasterM3 Validation Samples

Run the pipeline on 10–20 representative validation images (mix of intact-only, damaged, and destroyed-heavy images) and evaluate:

1. **Does the VLM agree with the segmentation model's class prediction?** If the segmenter says "Destroyed" but the VLM says "Minor cracks only", that's a disagreement worth analysing.
2. **Does the zoomed crop improve the VLM's reasoning compared to the full image?** Run the same VLM on the full 1024×1024 image (without cropping) and compare answer quality.
3. **Are the EMS-98 grades consistent?** Check a sample of 20 buildings manually.

---

## 5. Required Tools (All Free)

| Tool | Purpose | Install |
|---|---|---|
| PyTorch + SMP | Run the segmentation model | `pip install torch segmentation-models-pytorch` |
| OpenCV + NumPy | Image loading, connected components, crop extraction | `pip install opencv-python numpy` |
| Transformers | Load free Qwen2.5-VL-3B from Hugging Face | `pip install transformers` |
| Qwen2.5-VL-3B-Instruct | Free open-weight VLM (3B parameters, fits in T4) | Available on Hugging Face |
| Pillow | Image format conversion for the VLM | `pip install pillow` |
| Matplotlib | Visualisation and report figures | `pip install matplotlib` |

**VLM inference needs a free Kaggle T4 GPU or Google Colab.** Segmentation and crop extraction can run on a laptop CPU.

---

## 6. Expected Outcomes

- A working pipeline where: **Satellite Image → Segmentation Mask → Building Crops → VLM Inspection → Structured Damage Report with Annotated Map**.
- Proof that **zoomed-in Visual RAG dramatically improves VLM damage assessment** compared to full-image prompting.
- Publication-quality figures showing the annotated satellite map with numbered buildings and per-building inspection cards.
- A direct answer to the DisasterM3 benchmark's counting and referring-segmentation tasks, which our VLM struggled with in 499A.
