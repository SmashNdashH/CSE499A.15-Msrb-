# TEOChat Implementations for Post-Disaster Building Damage Assessment
**CSE499A / CSE499B — Senior Design Capstone Project**  
*Enhancing DisasterM3 Building Damage Counting via Stanford TEOChat (ICLR 2025) Methodologies*

---

## 1. Executive Summary & Research Motivation

### The Problem
During visual instruction tuning of **Qwen2.5-VL-7B** on the DisasterM3 benchmark, our model experienced a **-10.62% accuracy regression** on the **Building Damage Counting (BDC)** task (dropping from 34.2% base accuracy to 23.58%). 
* **Root Cause (Deviation D4):** Due to 16 GB VRAM constraints on Kaggle T4 GPUs, input images were downscaled to 512×512 with visual token patch compression. At this resolution, small satellite building footprints blur into contiguous pixel blobs, preventing the VLM's vision encoder from resolving individual buildings.

### The Stanford ICLR 2025 Empirical Justification
In *TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data* (Stanford University, ICLR 2025), researchers evaluated whether generalist VLMs could perform building localization on the xBD satellite dataset:
* **Pure VLM Localization:** **`38.9% F1`**
* **Specialist U-Net Localization:** **`85.9% F1`** *(Table 1, Page 7)*

> **Core Research Insight:** Pure vision-language models struggle to detect fine building boundaries in satellite imagery. The optimal architecture is a **Two-Stage Hybrid System**: a high-resolution specialist U-Net handles pixel localization and passes its findings to the VLM for multimodal reasoning and rescue guidance.

Below are the **5 specific TEOChat implementations** engineered into our pipeline.

---

## 2. Implementation 1: Normalized Bounding Box Grounding Syntax

### Concept (TEOChat Section 4 & Table 8)
Instead of feeding the VLM an isolated scalar number (e.g., `"Count: 4"`), the pipeline converts detected building footprints into **natural language coordinate tokens** normalized from `0` to `1000`:
$$\text{norm\_coord} = \text{int}\left(\frac{\text{pixel\_coord}}{\text{dimension}} \times 1000\right)$$

### Code Implementation
```python
# Extracted from support/single_image_unet_vlm_pipeline.py
norm_box = [
    int(round((y / h) * 1000)),
    int(round((x / w) * 1000)),
    int(round(((y + bh) / h) * 1000)),
    int(round(((x + bw) / w) * 1000))
]
```

### Injected Prompt Structure
```text
[High-Resolution U-Net Damage Findings]:
- Total Damaged/Destroyed Buildings Detected: 4
- Detected Building Coordinates [ymin, xmin, ymax, xmax]:
  - Building 1: location [700, 650, 790, 760] in Southeast (Severity: damaged)
  - Building 2: location [720, 800, 805, 895] in Southeast (Severity: destroyed)
  - Building 3: location [850, 720, 930, 820] in Southeast (Severity: destroyed)
  - Building 4: location [860, 860, 935, 950] in Southeast (Severity: damaged)
```

### Why It Makes the Project Better
* **Visual Attention Steering:** Qwen2.5-VL natively interprets `[ymin, xmin, ymax, xmax]` coordinates. The text tokens force the cross-attention heads to inspect the exact spatial image patches.
* **Eliminates Hallucination:** The VLM does not guess how many structures exist in blurry terrain; it verifies specific candidate locations proposed by the U-Net.

---

## 3. Implementation 2: 3×3 Geospatial Sector / Quadrant Reasoning

### Concept (TEOChat Section 3.1.2)
TEOChat formulates spatial reasoning by dividing the image into a **3×3 geographic grid** (Northwest, North, Northeast, West, Center, East, Southwest, South, Southeast) and predicting the sector most affected by the disaster.

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  Northwest (NW)      │     North (N)        │  Northeast (NE)      │
├──────────────────────┼──────────────────────┼──────────────────────┤
│     West (W)         │      Center          │     East (E)         │
├──────────────────────┼──────────────────────┼──────────────────────┤
│  Southwest (SW)      │     South (S)        │  Southeast (SE)      │
│                      │                      │  [★ DAMAGE HOTSPOT]  │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

### Code Implementation
```python
# Extracted from support/single_image_unet_vlm_pipeline.py
QUADRANTS = [
    ["Northwest", "North", "Northeast"],
    ["West",      "Center", "East"],
    ["Southwest", "South", "Southeast"]
]

col = min(2, max(0, int(cx / (w / 3.0))))
row = min(2, max(0, int(cy / (h / 3.0))))
quadrant = QUADRANTS[row][col]
most_affected_quadrant = max(quad_counts.items(), key=lambda x: x[1])[0]
```

### Generated Prompt & QA Task
```text
[Spatial Structural Distribution]:
- Damage Cluster Counts: Southeast: 4, Northwest: 0

Question: Which part of the image was most affected by the disaster?
Answer specifying the primary sector and explain where rescue teams should be dispatched first.
```

### Why It Makes the Project Better
* **Tactical Rescue Command:** First responders do not only need an aggregate count; they require **geographic prioritization** to deploy search-and-rescue teams.
* **Contextual Grounding:** Equips the VLM to answer both quantitative count queries and spatial distribution queries.

---

## 4. Implementation 3: Selective 256×256 High-Resolution Zoom Cropping

### Concept (TEOChat Appendix B, Page 19)
To retain fine visual details on 1024×1024 satellite imagery, TEOChat sliced images into 256×256 patches. In our resource-constrained setup, our U-Net acts as an **intelligent spatial filter**, identifying only the active disaster areas to crop.

```
  Full Satellite Scene (1024x1024)
  ┌────────────────────────────────────┐
  │                                    │
  │   [Intact]                         │
  │                                    │
  │              ┌────────┐            │
  │              │256x256 │◄───────────┼── U-Net selects highest-density
  │              │Crop    │            │   damage cluster for high-res zoom
  │              └────────┘            │
  │                [Rubble/Debris]     │
  └────────────────────────────────────┘
```

### Code Implementation
```python
# Extracted from support/single_image_unet_vlm_pipeline.py
pad = 256
crop_x1 = max(0, int(cx - pad // 2))
crop_y1 = max(0, int(cy - pad // 2))
crop_x2 = min(w, crop_x1 + pad)
crop_y2 = min(h, crop_y1 + pad)

high_res_crop = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
cv2.imwrite(f"./temp_crops/crop_building_{idx}.jpg", high_res_crop)
```

### Why It Makes the Project Better
* **Preserves Native Sensor Resolution:** The VLM inspects structural debris, wall cracks, and roof collapses at 100% native resolution.
* **Avoids OOM Crashes:** Feeding 1-2 selective 256px crops consumes a fraction of the VRAM required to feed full 1024px multi-image inputs.

---

## 5. Implementation 4: Structured Prompt Conditioning (TEOChat Table 8)

### Concept (TEOChat Table 8, Page 20)
Instead of unstructured prompting, the input is segmented into three distinct functional blocks: Perception Prior, Grounded Question, and Structured Decision.

### Full Prompt Structure
```text
[High-Resolution U-Net Damage Findings]:
- Total Damaged/Destroyed Buildings Detected: 4
- Primary Hazard Concentration Sector: Southeast
- Detected Building Coordinates [ymin, xmin, ymax, xmax]:
  - Building 1: location [700, 650, 790, 760] in Southeast (Severity: damaged)
  - Building 2: location [720, 800, 805, 895] in Southeast (Severity: destroyed)
  - Building 3: location [850, 720, 930, 820] in Southeast (Severity: destroyed)
  - Building 4: location [860, 860, 935, 950] in Southeast (Severity: damaged)

--- DisasterM3 Benchmark Question ---
How many buildings in the image have been damaged or destroyed?
Options:
A. 1
B. 4
C. 8
D. 15
Please output only the single best choice (e.g., 'The answer is B.') and explain your count.
```

### Why It Makes the Project Better
* **Determinism & Precision:** Guides the language model heads to map the perceptual count directly to multiple-choice options (`B. 4`) without random probabilistic drift.
* **Multimodal Reasoning Kept Intact:** Unlike hardcoded script overrides, the VLM actively verifies the visual evidence and outputs tactical explanations for emergency personnel.

---

## 6. Implementation 5: Academic Defense Script

When defending the project methodology before evaluators or supervisors:

| Question from Evaluator | Defense Answer Grounded in TEOChat (ICLR 2025) |
| :--- | :--- |
| **"Why not just train the VLM end-to-end without U-Net?"** | *"In Stanford's TEOChat study (ICLR 2025, Table 1), an end-to-end VLM achieved only 38.9% F1 on satellite building localization, whereas a specialist U-Net achieved 85.9% F1. Our hybrid architecture combines U-Net's high-resolution pixel localization with Qwen2.5-VL's multimodal reasoning, directly overcoming the 512px token bottleneck."* |
| **"Why not feed 1024×1024 images directly into Qwen2.5-VL?"** | *"Full 1024×1024 images expand visual tokens quadratically ($O(N^2)$), causing immediate Out-of-Memory crashes on T4 GPUs. TEOChat solved this via 256×256 crops. Our pipeline uses U-Net to crop only the active damage zones, preserving native resolution within the 16 GB VRAM budget."* |
| **"What if the U-Net makes a false detection?"** | *"The U-Net does not output the final answer. It provides candidate bounding boxes in text. The VLM acts as a visual verifier, inspecting the image tokens at those coordinates to confirm or reject structural collapse."* |

---

## 7. File Map in Codebase

| Implementation File | Role & Contents |
| :--- | :--- |
| [`kaggle_champion_unet_vlm.ipynb`](file:///g:/NSU/499B/CSE499A.15-Msrb-/kaggle_champion_unet_vlm.ipynb) | Complete 4-cell runnable Kaggle notebook integrating Champion U-Net with Qwen2.5-VL. |
| [`support/single_image_unet_vlm_pipeline.py`](file:///g:/NSU/499B/CSE499A.15-Msrb-/support/single_image_unet_vlm_pipeline.py) | Python module implementing `SingleImageExtractor` (quadrants, boxes, 256px crops) and `TEOChatPromptBuilder`. |
| [`support/run_champion_unet_vlm.py`](file:///g:/NSU/499B/support/run_champion_unet_vlm.py) | Standalone CLI runner loading ResNet-50 4-class weights and executing the pipeline. |
| [`support/unet_findings_to_vlm.py`](file:///g:/NSU/499B/CSE499A.15-Msrb-/support/unet_findings_to_vlm.py) | Core geometric contour extractor supporting 3-class, 4-class, and 5-class xBD masks. |
