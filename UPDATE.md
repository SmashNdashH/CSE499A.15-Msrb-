> 📖 **[Main Project Overview](README.md)** | 🚀 **[Latest Team Update](UPDATE.md)**

---

# Post-Disaster Rescue Guidance via VLM Fine-Tuning

> **CSE499A -- Section 15, Group 5, Update 3**
> 
> **Team:** 
> - Abdullah Al Noman (2022095042)
> - Tamanna Akter Mou (2211951042)
> - Aryan Sami (2231407042)
> - Ridita Afrin Riya (2211622042)
> - Abrar Mohammed Tanzim Alam (2222864042)

## Work Done by Abrar Mohammed Tanzim Alam

My primary objective this week was to reproduce the fine-tuning and evaluation pipeline of the DisasterM3 (DM3) paper (Wang et al., NeurIPS 2025), whose released repository provides only benchmarking code—no training scripts. This required self-assembling the full training, merging, and evaluation infrastructure from scratch on Kaggle's single-T4 hardware, matching the hyperparameters in Appendix B.3. The implementation spans six sequential notebooks:

1. **Dataset Inspection (`disasterm3_dataset_inspection.ipynb`):** Audited the 87 MB `train_release.json` manifest. Found 37,204 Referring Expression Segmentation entries had empty `training_answer` fields (ground truth is a mask path, not text)—invalid for text generation. Curated effective set: **55,764 valid instruction pairs** across 8 tasks.
2. **Dataset Mirroring (`disasterm3_dataset_mirror.ipynb`):** Transferred the full 44 GB DM3 dataset from Hugging Face to a permanent private Kaggle Dataset, staging through `/tmp` (~70 GB) to bypass the 20 GB `/kaggle/working` cap—eliminating repeated multi-hour downloads.
3. **QLoRA Fine-Tuning (`train_qwen_disasterm3.ipynb`):** Built the training pipeline (`transformers`+`peft`+`trl` SFTTrainer) for a single T4 (16 GB VRAM), replacing the paper's 4xH100 full-precision setup with 4-bit NF4 QLoRA, fp16, SDPA attention, and grad-accum 256. Fixed a LoRA misconfig where the vision tower wrongly received adapters. Ran 5 checkpoint-resume sessions, reaching step 102/217, loss 0.83→0.54; backed up adapter weights to Kaggle Dataset + HF Hub (`AbrarAlam/disasterm3-qwen25vl7b-qlora`).
4. **LoRA Merge (`1_merge_and_push_model.ipynb`):** Merged the QLoRA adapter with base Qwen2.5-VL-7B in FP16, pushed the fused 15 GB checkpoint to HF. Required an isolated session to avoid `numpy`/`scipy` ABI mismatches and `triton` crashes.
5. **Benchmark Mirroring (`2_dataset_creation.ipynb`):** Downloaded all 22,381 benchmark assets from HF's `DisasterM3_Bench` folder (11 GB), packaged into a permanent Kaggle Dataset. Overcame authenticated rate-limiting and `snapshot_download` deadlocks via token-injection and cache-bypass fallback.
6. **vLLM Evaluation (`3_vllm_evaluation.ipynb`):** Configured the paper's official eval script with vLLM serving to benchmark the merged model on the 6 MCQ tasks from Table 2.

## Work Done by Tamanna Akter Mou

1. **Proposed-Solution-Report:** After observing the implementation progress of the team, I organized the project narrative into the proposed-solution report. I connected each member's implementation to the final two-tier framework so the report does not look like separate unrelated tasks.
2. **Orchestration and Evaluation Writing:** I expanded the command-center side of the paper by describing practical outputs such as damage reports, social media alerts, medical staging, HR allocation, public advisories, and reconstruction plans. I also emphasized rescue-actionability evaluation so the generated text is judged by safety and usefulness, not only by text similarity. This helped align the update paper with the proposed-solution report and made the weekly progress easier to defend during presentation.

## Work Done by Abdullah Al Noman

1. **Super-Resolution Implementation:** Implemented the super-resolution model (from last week's presented paper, *Shakya-From Pixels to Semantics*) to enhance visual fidelity and structural detail of post-disaster AIDER dataset.
2. **Data Engineering and Tensor Construction:** Built `prep_vrt_input.py` to satisfy the Video Restoration Transformer's (VRT) spatial-temporal tensor requirement. Replicated the static 1024x1024 frame four times, emulating a zero-motion pseudo-video clip to produce the required 5D tensor.
3. **Weight Configuration and Mapping:** Downloaded the pre-trained mathematical weight configurations matching the `001_VRT_videosr_bi_REDS_6frames`. Mapped the weights to the codebase's strict local pathway to ensure proper structural state-dict binding.
4. **Inference and Memory Optimization:** The massive computational overhead of generating a 4096x4096 output array was mitigated by partitioning the 1024px input space into small, sequential spatial tiles of 256x256 pixels.
5. **Execution and Output Generation:** Overlapping tile margins were seamlessly fused together using a recurrent blending utility to eliminate block artifacts across the upscaled geography. Generated the final 4X super-resolved structural assessment asset directly to the disk with a crisp, verified resolution of exactly 4096x4096 pixels.

## Work Done by Aryan Sami

1. **Resolution and Localization Gaps Identified:** Documented how 10m ground sampling distance limits fine-grained damage detection in dense urban settings, with building localization remaining the weakest pipeline stage.
2. **Multi-Source Data Integration Rationale:** Outlined how combining pre-/post-event optical (Sentinel-2) and radar (Sentinel-1) imagery improves robustness to cloud cover and lighting versus single-source approaches.
3. **Dataset-to-Model Mapping:** Mapped the xBD-S12 dataset against our rescue-oriented instruction-response format, clarifying how satellite comparisons and damage classifications convert into training-ready multimodal examples.
4. **Area of Interest and Acquisition Setup:** Defined the fire-affected extent for the Jan 2025 Palisades wildfire; configured acquisition for Sentinel-1 and Sentinel-2 L2A, pulling matched pre/post scenes.
5. **Multi-Sensor Co-Registration:** Reprojected all four rasters onto a shared 4m grid for pixel-level alignment across sensors and time steps, making optical and radar streams directly comparable.
6. **Model Inference Execution:** Ran the pretrained two-step ensemble from HF Hub on the co-registered Palisades imagery, producing a 3-class damage map matching the xBD-S12 label schema.
7. **Output Validation:** Verified the damage map by overlaying it on post-event optical imagery to confirm correspondence with the visible burn scar.

## Work Done by Ridita Afrin Riya

1. **System Architecture and Pipeline Development:** Designed and implemented "DisasTeller," an autonomous, multi-agent AI framework utilizing CrewAI, LangChain, and Large Vision-Language Models (Google Gemini 3.1 Flash-Lite).
2. **Specialized Multi-Agent Orchestration:** Divided complex disaster response workloads across four specialized agents: the *Expert Team* for visual damage grading, the *Alerts Team* for hazard warnings, the *Emergency Team* for staging triage shelters, and the *Assignment Team* for managing HR logistics and budgets.
3. **API Mitigation and System Stabilization:** Addressed critical system crashes caused by multi-agent parallel querying that rapidly exhausted Gemini's free-tier limits. Migrated to the highly efficient `gemini-3.1-flash-lite` model and implemented a strict `max_rpm=2` constraint within CrewAI to act as a pacing mechanism.
4. **Simulation Testing and Validation:** Executed and validated a full-scale simulation based on the Wajima Earthquake scenario. Demonstrated the system's ability to autonomously analyze field images, correctly identify a severed bridge as a "Grade 5 Total Collapse," route 80 Search and Rescue personnel, and strategically coordinate a nearby medical triage center.

---

**Next Week's Plan:** Complete the remaining training steps (102/217 -> 217/217) and execute the vLLM evaluation pipeline against the 6 MCQ benchmark tasks. Report accuracy metrics (DSR, DTR, BBR, BDC, DRE, ORR) and compare against the paper's published baselines to quantify the impact of our hardware-constrained QLoRA adaptation.