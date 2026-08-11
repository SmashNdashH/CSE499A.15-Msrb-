>  **[Main Project Overview](README.md)** |  **[Latest Team Update](UPDATE.md)**

---

# Post-Disaster Rescue Guidance via VLM Fine-Tuning

> **CSE499A – Section 15, Group 5, Update 5**
> 
> **Team:** 
> - Abdullah Al Noman (2022095042)
> - Tamanna Akter Mou (2211951042)
> - Aryan Sami (2231407042)
> - Ridita Afrin Riya (2211622042)
> - Abrar Mohammed Tanzim Alam (2222864042)

## Work Done by Abrar Mohammed Tanzim Alam

1. **Completed QLoRA Fine-Tuning (`train_qwen_disasterm3.ipynb`):** Successfully executed the 11th and final checkpoint-resume session on Kaggle's 12-hour constrained hardware, reaching 100% completion (217/217 steps) of the 1-epoch fine-tuning schedule. The final QLoRA adapter weights were securely backed up to the Hugging Face Hub (`AbrarAlam/disasterm3-qwen25vl7b-qlora`).
2. **Model Merging and FP16 Precision Push (`1_merge_and_push_model.ipynb`):** Engineered an isolated session environment to merge the QLoRA adapter with the base Qwen2.5-VL-7B model, actively circumventing `numpy`/`scipy` ABI mismatch crashes. The final fused 15 GB checkpoint was successfully deployed to Hugging Face (`AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP`) in native FP16 precision.
3. **vLLM Evaluation Pipeline Engineering (`3_vllm_evaluation.ipynb`):** Developed and executed the evaluation notebook to run the paper's benchmark script on the 6 MCQ tasks from Table 2. Overcame severe environmental blockers (T4 FlashAttention incompatibility and `bitsandbytes`/`triton` conflicts) by hot-wiring the vLLM engine to utilize **GPU T4 2x Tensor Parallelism** (`tensor_parallel_size=2`) with eager execution (`enforce_eager=True`).
4. **Automated Evaluation Execution (`3_vllm_evaluation.ipynb`):** Dynamically patched the original authors' evaluation script at runtime to eradicate Windows-style pathing `FileNotFoundError` crashes and initialized the evaluation loop across the 6 target MCQ tasks. Segmented the massive 12-hour evaluation workload into dual 6-hour Kaggle sessions to ensure data persistence against kernel timeouts.
5. **Interactive VLM UI (`gradio_app.py` & `test_qwen_disasterm3_interactive.ipynb`):** Built and deployed a live interactive Gradio dashboard to test the fine-tuned model. You can explore the live demo on Kaggle here: [Interactive Kaggle Demo](https://www.kaggle.com/code/abrarmohammedtanzim/test-qwen-disasterm3-interactive)

**Final Benchmarking Results (Comparison to Base & Fine-Tuned Qwen2.5-VL-7B)**
| Task | Status | Our Acc. | Base Acc. | Δ vs Base | Paper FT | Δ vs FT |
|---|---|---|---|---|---|---|
| **Disaster Type (DTR)** | COMPLETE | **68.33%** | 66.6% | +1.73% | 83.6% | **-15.27%** |
| **Bearing Body (BBR)** | DONE* | **16.39%** | 4.7% | +11.69% | 21.5% | **-5.11%** |
| **Building Damage (BDC)** | DONE* | **23.58%** | 34.2% | -10.62% | 34.3% | **-10.72%** |
| **Road Damage (DRE)** | DONE* | **26.24%** | 29.3% | -3.06% | 29.4% | **-3.16%** |
| **Landuse (DSR)** | DONE* | **31.31%** | 28.3% | +3.01% | 37.7% | **-6.39%** |
| **Relational Reasoning (ORR)** | COMPLETE | **22.50%** | 23.9% | -1.40% | 36.2% | **-13.70%** |

**Analysis:** *vs Base (unfine-tuned Qwen2.5-VL-7B):* Our fine-tuning yielded improvements on classification tasks — Bearing Body Recognition (BBR) (**+11.69%**, trained on 7,766 samples), Landuse / Disaster Scene Recognition (DSR) (**+3.01%**, trained on 7,090 samples), and Disaster Type Recognition (DTR) (**+1.73%**, trained on 1,627 samples). However, it caused regressions on counting and spatial reasoning tasks: Building Damage Counting (BDC) (**-10.62%**), Road Damage Estimation (DRE) (**-3.06%**), and Object Relational Reasoning (ORR) (**-1.40%**). *vs Paper's Fine-Tuned Model (4x H100, full BF16, unbounded resolution):* We underperform on **all 6 tasks**, with gaps ranging from -3.16% (DRE) to -15.27% (DTR). This confirms that our hardware-constrained reproduction (QLoRA on 1x T4) is not competitive with the paper's full-resource fine-tune, which is the expected outcome. These regressions are fully explained by Deviation **D4** (image resolution capped to 512 x 28 x 28 pixels) and Deviation **D2** (4-bit NF4 QLoRA instead of full BF16 LoRA).

**Data Skipping Justification:** A total of 9 dataset items (2 in BBR, 4 in BDC, 2 in RDC, and 1 in Landuse) were intentionally skipped and excluded from the final metrics. This was handled via a `try...except` safety block inside the patched `run_vllm.py`. The underlying raw `.png` files for these specific items in the Kaggle dataset mirror were fundamentally corrupted (throwing `PIL.UnidentifiedImageError`). Because the vision encoder physically could not load the byte data, they were gracefully skipped to prevent fatal engine crashes, resulting in the completed predictions being marginally lower than the theoretical benchmark totals.

## Work Done by Tamanna Akter Mou

1. **Literature and Framework Integration Review:** Analyzed the AnyDisasterMapping repository, which provides a unified benchmark and processing pipeline for multi-hazard disaster-mapping tasks. Documented how integrating its pretrained geospatial backbone, specifically the SegFormer MiT-B2 architecture, can address the localization and reasoning gaps identified in our previous updates by providing a stronger visual feature extractor than standard baseline U-Nets.

## Work Done by Aryan Sami

1. **Paper Review Objective:** Reviewed *DisasterInsight: A Multimodal Benchmark for Function-Aware and Grounded Disaster Assessment* by Tehrani, Xu, Haglund, Berg, and Felsberg (Linköping University). The work is currently available as an arXiv preprint (arXiv:2601.18493v1, cs.CV, January 2026) and has not yet been accepted at a peer-reviewed venue.
2. **Benchmark Structure Audit:** Reviewed how DisasterInsight restructures xBD into 112,507 building-centered instances by dividing scenes into patches and pairing each building with its own bounding box. This lightweight instance-construction approach is worth comparing with DisasterM3's mask-based entries, 40% of which were found unusable during our earlier dataset audit.
3. **Function-Label Pipeline:** Studied the benchmark's method for deriving building-function labels—such as hospital, school, and residential—from OpenStreetMap tags and consolidating them into eight categories. This provides a possible direction for extending our xBD-S12/Palisades pipeline beyond its current three-class, damage-only schema.
4. **Report-Generation Design:** Examined the paper's two-tier report format, consisting of a short situational summary and a longer risk-and-recovery narrative. Its use of BLEU-4, ROUGE-L, and BERTScore rather than exact-match metrics is relevant to our rescue-actionability evaluation approach for command-center advisories.
5. **LoRA Configuration Comparison:** Compared the paper's fine-tuning setup—LoRA rank r=64 and α=128 on TeoChat/Qwen2.5-VL with a frozen vision encoder—with our QLoRA configuration of r=64 and α=16. The substantially higher α/r ratio may partly explain its larger reported performance gains.
6. **Shared Failure Mode:** Identified that building-function classification remained weak after fine-tuning, achieving approximately 18% F1. This aligns with our finding that coarse satellite-image resolution, rather than model architecture alone, is the principal bottleneck for fine-grained classification in our pipeline.
7. **U-Net V2 Fine-Tuning Breakthrough:** Successfully executed a second U-Net fine-tuning run implementing AMP optimizations and a corrected 4-class mapping architecture (Background, Intact, Damaged, Destroyed). This resolved previous gradient scaling bugs and propelled the model to a new state-of-the-art **Mean IoU of 0.4619** and a Mean F1 of 0.5808, significantly outperforming the V1 baseline (0.4286).

## Work Done by Ridita Afrin Riya

1. **Local Environment Setup and Model Optimization:** Installed the Ollama framework locally to enable running large Vision-Language Models (VLMs) offline for image analysis. Transitioned to the LLaVA (Large Language-and-Vision Assistant) model to mitigate significant performance bottlenecks and high inference times caused by local hardware constraints when processing image pairs.
2. **Pipeline Codebase Synchronization and Debugging:** Synchronized the local workspace with the remote repository by pulling the latest changes to ensure the implementation utilized the most up-to-date scripts. Investigated the `VLM_Damage_Assessment.py` script and iteratively modified the codebase to process smaller data subsets (scaling from the hardcoded 227 pairs down to 10, 5, and finally 2 data points) to accommodate hardware limitations and facilitate rapid testing.
3. **Damage Assessment Pipeline Execution:** Executed the pipeline flow to load `pre_disaster.png` and `post_disaster.png` image pairs from folders within the `./data/images/` directory. Prompted the local LLaVA model to act as a damage assessment expert, categorizing structural damage on a 1–4 scale and streaming an analysis of building conditions and potential hazards.
4. **Output Generation and Formatting:** Formatted the model's analysis to save dynamically as detailed `.txt` reports into the `./results/vlm_assessments/llava/` directory. Applied a strict naming convention corresponding to the specific data point (e.g., `00000000_assessment.txt`).

---

**Next Week's Plan:** Evaluate strategies to mitigate high model running times—such as batching, resizing images, or utilizing cloud compute if local hardware remains insufficient—and scale up the analysis to process the full dataset once performance is properly optimized.
