# Hybrid Pipeline Delegation Guide

This guide is designed for your teammate (Aryan) to take over the segmentation model fine-tuning and integrate it into the Hybrid Spatial-Semantic Triage Pipeline.

---

## Part 1: Segmentation Model Comparison & Recommendation

Before diving into the code, we must decide *which* segmentation model to fine-tune on the 37,204 DisasterM3 segmentation entries. We evaluated the models from the DisasterM3 paper (Appendix B.4) and the Sentinel-1/2 research paper against our strict Kaggle hardware limits (1× T4 GPU, 16 GB VRAM, 12-hour session limits).

### 1. LISA (7B) and PSALM (Phi-1.5 1.3B) — *Not Recommended*
- **Compute Requirements:** The DisasterM3 paper explicitly states that both LISA and PSALM were fine-tuned across **4× H100 GPUs**. 
- **Feasibility:** Attempting to train a 7B or 1.3B parameter Vision-Language Segmentation model on a single 16 GB T4 GPU is effectively impossible without heavy QLoRA quantization, which would push the training time well past Kaggle's 12-hour session limit. 

### 2. Geospatial Foundation Models (Prithvi, DOFA) — *Not Recommended*
- **Research Findings:** The Sentinel-1/2 paper (*The Potential of Copernicus Satellites for Disaster Response*) specifically evaluated GeoFMs like Prithvi-EO-2.0. They concluded that for damage mapping, **"architectural sophistication does not seem to bring much advantage... geospatial foundation models bring little practical benefit."** They are heavy, prone to overfitting on event-based splits, and slow to train.

### 3. SMP U-Net (ResNet34 Backbone) — **Highly Recommended**
- **Compute Requirements:** ~21M parameters. Fits comfortably in 16 GB VRAM with a batch size of 4 at 512×512 resolution.
- **Feasibility:** Training 40 epochs easily completes within the 12-hour Kaggle window.
- **Research Findings:** The Sentinel paper confirms that a simpler U-Net approach often *outperforms* complex state-space models and GeoFMs in generalizing to unseen disasters. 

**Decision:** We are proceeding with fine-tuning a **U-Net** using the `segmentation_models_pytorch` (smp) library, heavily customized to mirror the data handling of the VLM pipeline.

---

## Part 2: Step-by-Step Execution Guide

This pipeline uses a **Two-Stage Hybrid Approach**. 
- **Layer 1 (The U-Net):** Extracts spatial data (bounding boxes and building counts) from the 1024×1024 image.
- **Layer 2 (The VLM):** Receives cropped bounding boxes from Layer 1 to perform contextual reasoning (DTR, BBR, ORR) without being blinded by the 512-token resolution cap.

### Step 1: Train the Segmentation Model (Layer 1)
You need to run the U-Net training pipeline on the Kaggle T4 environment.

**File:** `[train_unet_disasterm3.ipynb](file:///d:/CSE499AB_project/support/Abrar/Implementation_DisasterM3/train_unet_disasterm3.ipynb)`

**Instructions:**
1. Upload the notebook to a new Kaggle Kernel.
2. Attach the **DisasterM3 mirror dataset** (`datasets/abrarmohammedtanzim/disasterm3-mirror`).
3. Set the accelerator to **GPU T4 x2** (though it will use 1 by default).
4. **Important Config:** Ensure you have your `HF_TOKEN` in Kaggle Secrets to allow the notebook to push checkpoints to Hugging Face automatically (Cell 10).
5. Click **Run All**. The first cell will install dependencies and halt. Restart the kernel, and click **Run All** again.
6. The model will train exclusively on the 37,204 Referring Expression Segmentation entries (the 40% of data that was dropped during VLM training).

### Step 2: Validate the Programmatic BDC Fix
Once the U-Net produces `.pth` weights (or using raw xBD ground truth masks for testing), we need to bypass the VLM entirely for the Building Damage Counting (BDC) task.

**File:** `[hybrid_bdc_counter.py](file:///d:/CSE499AB_project/support/Abrar/Implementation_DisasterM3/hybrid_bdc_counter.py)`

**Instructions:**
1. This script takes the U-Net's segmentation mask output and uses OpenCV contour analysis to mathematically count the destroyed buildings.
2. Run it locally or in a Kaggle cell to verify:
   ```bash
   python hybrid_bdc_counter.py --mask_dir /path/to/unet/output_masks/
   ```
3. This directly addresses the **−10.62% BDC performance degradation**.

### Step 3: Consolidate the Hybrid Inference Pipeline
This is where both models come together. Because both models would exceed the 16 GB VRAM limit if loaded simultaneously, they must be run sequentially.

**Target File (To Be Built in Tier 2):** `hybrid_triage_pipeline.ipynb`

**The Workflow you will implement:**
1. **Load Layer 1 (U-Net):** Load your trained `best_model.pth`.
2. **Process Batch:** Feed the raw high-res images into the U-Net. Save the output masks.
3. **Extract & Unload:** Use `[hybrid_bdc_counter.py](file:///d:/CSE499AB_project/support/Abrar/Implementation_DisasterM3/hybrid_bdc_counter.py)`'s `extract_bounding_boxes()` function to get the coordinates of damaged clusters. **Crucial step: Unload the U-Net from VRAM** (`del model; torch.cuda.empty_cache()`).
4. **Load Layer 2 (Qwen2.5-VL):** Load Abrar's fine-tuned VLM adapter.
5. **Crop & Reason:** Crop the original images using the bounding boxes from Step 3. Feed these small, focused patches to the VLM. Because the patches are small, the VLM's 512-token limit now captures incredibly high detail. 

By executing this sequentially, you eliminate the VLM's spatial blindness (fixing BDC and DRE) while maintaining its superior contextual reasoning (DTR, BBR, ORR).

---

## Part 3: How Your Existing Scripts Fit into the New Workflow

To get your initial foundational code running on Kaggle within the strict time/hardware limits (1× T4 GPU, 16 GB VRAM, 12-hour session limits), much of your logic was integrated and upgraded into a new consolidated notebook. Here is where every script belongs in the new timeline:

### 1. Data Preparation Phase (Run Before Training)
* **`[json_to_mask.py](file:///d:/CSE499AB_project/support/Aryan%20Codes/json_to_mask.py)`**: This is your preprocessing script. It takes the raw xBD polygon JSON files and physically draws the 1024×1024 `.png` segmentation masks that the U-Net needs to learn from. **(You still run this locally before uploading to Kaggle).**
* **`[data_loader.py](file:///d:/CSE499AB_project/support/Aryan%20Codes/data_loader.py)`**: This was your PyTorch logic for loading those images into memory. **Where it is now:** The exact logic (simplifying 5-class damage to 3-class) has been embedded directly into Cell 5 of the new `[train_unet_disasterm3.ipynb](file:///d:/CSE499AB_project/support/Abrar/Implementation_DisasterM3/train_unet_disasterm3.ipynb)` notebook so it works seamlessly on Kaggle.

### 2. The Training Phase (Upgraded for Kaggle)
* **`[model.py](file:///d:/CSE499AB_project/support/Aryan%20Codes/model.py)`**: This was your original architecture (a ResNet34 encoder with a sequential decoder). **Where it is now:** This was upgraded in Cell 6 of the new notebook to use `segmentation_models_pytorch` (smp), adding the proper skip connections that the initial model was missing.
* **`[train.py](file:///d:/CSE499AB_project/support/Aryan%20Codes/train.py)`**: This was your local training loop. **Where it is now:** Completely replaced by the new `[train_unet_disasterm3.ipynb](file:///d:/CSE499AB_project/support/Abrar/Implementation_DisasterM3/train_unet_disasterm3.ipynb)` notebook, which adds Kaggle-specific safeguards (12-hour TimeLimitCallback, Hugging Face checkpoint pushing, and proper mIoU validation metrics).

### 3. The Inference & Triage Phase (The New Hybrid Pipeline)
* **`[hybrid_bdc_counter.py](file:///d:/CSE499AB_project/support/Abrar/Implementation_DisasterM3/hybrid_bdc_counter.py)`**: This is the **bridge** between your U-Net and Abrar's VLM. After your U-Net generates a mask for an image, this script mathematically counts the damaged buildings (bypassing the VLM entirely to fix the BDC regression) and crops the bounding boxes so the VLM can look at them in high resolution.

### 4. The Evaluation Phase (The Final Proof)
* **`[evaluate_damage_alignment.py](file:///d:/CSE499AB_project/support/Aryan%20Codes/evaluate_damage_alignment.py)`**: Once the hybrid pipeline finishes processing the test images, you feed the final textual outputs into this script to check if the generated text accurately reflects the real damage metadata. This proves the Hybrid approach beats the base model.

**Summary for Aryan:** You no longer need to manually run `[model.py](file:///d:/CSE499AB_project/support/Aryan%20Codes/model.py)`, `[train.py](file:///d:/CSE499AB_project/support/Aryan%20Codes/train.py)`, or `[data_loader.py](file:///d:/CSE499AB_project/support/Aryan%20Codes/data_loader.py)`. Just run **`[json_to_mask.py](file:///d:/CSE499AB_project/support/Aryan%20Codes/json_to_mask.py)`** to prep the data, upload it to Kaggle, and hit "Run All" on the new **`[train_unet_disasterm3.ipynb](file:///d:/CSE499AB_project/support/Abrar/Implementation_DisasterM3/train_unet_disasterm3.ipynb)`** notebook.
