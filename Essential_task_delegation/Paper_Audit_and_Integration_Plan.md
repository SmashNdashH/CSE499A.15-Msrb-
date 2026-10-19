# Paper Audit and Integration Plan

Written 2026-10-17. For each of 11 sources (10 papers plus the BDAChat code repository): what it did, how it did it, and how it can help us reach our CSE499B goals. The audit covers each paper's abstract, method and main results, not every page. The PDFs are in [others/lit review_research papers/](<../others/lit review_research papers/>). A twelfth source, Distilling Step-by-Step (§4.12), was added on 19 October 2026; its PDF is in the same folder.

## 1. Our goals: the yardstick for everything below

**Project:** Resource-Efficient Neural Architectures for Post-Disaster Damage Assessment and Rescue Guidance.

**CSE499B aim (review deck):** a hybrid system that counts from pixels, reasons in language, and runs offline on free or small GPUs.

| Objective | Key result |
|---|---|
| **O1. Strengthen the spatial stage** | KR1: a champion U-Net passes the Phase Gate on full-resolution validation images, with mIoU > 0.52 and Destroyed IoU > 0.30. |
| **O2. Ground the VLM spatially** | KR2: hybrid counting (champion U-Net plus a higher-resolution VLM) beats the base model's 34.2% building damage counting (BDC) accuracy, with all six tasks re-scored. |
| **O3. Build one offline triage system** | KR3: DisasTeller runs on our fine-tuned VLM with no Gemini calls and returns a report, damage map, counts and survivability heatmap. |
| **O4. Produce an edge-sized model** | KR4: a distilled Qwen2.5-VL-3B fits in about 6 GB, with per-task accuracy reported against the 7B teacher. |

Source: the Objectives section of the submitted CSE499B review report ([others/CSE499B_Review_report.pdf](../others/CSE499B_Review_report.pdf)); the same wording is in `CSE499B_Review_report_update.tex`, lines 59–65.

### 1.1 The CSE499A future-work items are part of these goals

The CSE499A final presentation ended with a Future Work list ([final_presentation.html:657-662](../final_presentation.html#L657-L662); in more detail in [CSE499A_Final_report.tex:507-515](../CSE499A_Final_report.tex#L507-L515)). The CSE499B review report turned most of it into the goals above:

| CSE499A future-work item | CSE499B goal | Where the review report says so |
|---|---|---|
| Resolution unlocking | **O2** | KR2: "a higher-resolution VLM" |
| Textual bridge prompting | **O2 and O3**, renamed "counts as text" | KR2 (hybrid counting). The architecture figure: "the VLM reads per-building counts as text and drives DisasTeller's report agents". Future Direction 3 says the same |
| Survivability heatmap | **O3** | KR3 names it: "returns a report, damage map, counts and survivability heatmap" |
| DisasTeller backend swap | **O3** | KR3: "DisasTeller runs on our fine-tuned VLM with no Gemini calls" |
| 7B → 3B distillation | **O4** | KR4 names it: "a distilled Qwen2.5-VL-3B fits in about 6 GB" |
| Full unified deployment | **Partly O3** | "One offline triage system", checked by "an offline end-to-end run on held-out images". The container and drone feed are not in any key result |
| PINN reverse simulation | **Not a goal** | Not in the CSE499B report. Research idea only; Research_Directions §5 rates it out of scope |

## 2. Where we stand (17 October 2026)

| Goal | Deliverable | Status | Where to look |
|---|---|---|---|
| O1 | Phase Gate | **Not met; fine-tuning closed (18 Oct).** mIoU passes in every run. R9 (sampler + copy-paste) is the best Destroyed recipe: 0.3001 with TTA on seed 42, but 0.2941 on its seed-7 repeat (R9s7), so the two-seed mean is 0.2971. Next is the E1 diagnosis, then R11 (before + after input). | `Champion_Task/R9-results/`, `Champion_Task/R9s7-results/`, [Ablation_Study_Reference_and_QA_Prep.md](Ablation_Study_Reference_and_QA_Prep.md) §7 |
| O1 | Extra training data (BRIGHT) | Planned; the plan's files disagree with each other (see 2.2) | [BRIGHT_DisasterM3_Integration/](BRIGHT_DisasterM3_Integration/) |
| O2 | Hybrid counting | Never re-measured since the intact-building counting bug was fixed. The only recorded number is 13.11%. | [hybrid_intermediate_results.md:13](../hybrid_intermediate_results.md#L13) |
| O2, O3 | Textual bridge ("counts as text") | Not built, but the interactive demo is one change away. Its hybrid mode prints the U-Net counts (e.g. "Intact 51, Damaged 3") but sends Qwen only the collage image and the original question (`ask_model(collage_img, question)`). Adding the counts to that question is the textual bridge. | [test_qwen_disasterm3_interactive.ipynb](../support/Abrar/Implementation_DisasterM3/test_qwen_disasterm3_interactive.ipynb); screenshots `Screenshot 2026-08-14 022013.png` and `022552.png` |
| O2 | Higher-resolution VLM | Not started (ImageRAG and GeoPixel's tiling are the references; section 4) | — |
| O3 | DisasTeller on our model | Not started. Every model call is Gemini: `gemini-3.1-flash-lite` for the agents and image tools, and `gemini-embedding-001` for the guideline search. Both must be replaced and the Chroma database rebuilt. Six reports from a Gemini run exist (20 Jul). | [DisasTeller/](../support/Ridita_Integration_of_VLM__/DisasTeller/) |
| O3 | Survivability heatmap | No code. The files named "heatmap" are a dataset-statistics plot and DisasterVQA score tables. The closest output is DisasTeller's `4_anotated.png`, a labelled map, not a heatmap. | See 2.1 for the design |
| O3 | One offline system | The parts exist separately: VRT super-resolution, the Gradio app and interactive notebook (U-Net + Qwen), and DisasTeller (on Gemini). No container or drone-feed code. | `support/Noman_Super Resolution/`, `support/Abrar/Implementation_DisasterM3/gradio_app.py` |
| O4 | 7B → 3B distillation | Prepared, never run. The teacher chain-of-thought notebook has no executed cells, and the 3B notebook's saved outputs are left over from the original 7B run. No 3B model exists yet. | `support/Abrar/Implementation_DisasterM3/*3b_distill.ipynb` |
| O4 | Edge benchmark | Not started (AWQ and Grace are the references; section 4) | — |
| Not a goal | PINN reverse simulation | Idea only | [pathways to still try (Consolidated_Audit_Report).md](<../pathways to still try (Consolidated_Audit_Report).md>) |

### 2.1 Survivability heatmap design (O3)

The CSE499A report proposes two ways to turn the U-Net damage mask into a heatmap. They answer different questions, so the heatmap can use both.

| | Distance transform (`cv2.distanceTransform`) | KDE (kernel density estimation) |
|---|---|---|
| What it computes | For every pixel, the distance to the nearest damaged or destroyed building | A smooth "bump" on every damaged or destroyed building; overlapping bumps add up |
| What the map shows | Near collapses means danger; far away means safer | Where damage is concentrated; dense clusters glow brightest |
| Rescue use | Safe assembly points and dangerous corridors | Where to search first |
| Code | `cv2.distanceTransform` on the inverted damage mask | `cv2.GaussianBlur` on the damage mask (a quick KDE), or `scipy.stats.gaussian_kde` on building centres |

No training is needed; the input is the champion U-Net's mask.

### 2.2 Open issues in the BRIGHT integration folder (O1)

`BRIGHT_DisasterM3_Integration/` holds an integration plan (14 Oct), a roadmap (17 Oct, "100% optical only") and `bright_disasterm3_standardizer.py`. They disagree in three places:
1. **The script would load radar images.** The roadmap says never to load BRIGHT's post-disaster radar files, but the script saves whatever is in each `post-event/` folder as `bright_<tile>_post_disaster.png`. In BRIGHT, those files are the SAR images.
2. **AIDER can't train the segmenter.** The roadmap says to use AIDER for optical rubble, but AIDER has only one label per image, from tilted drone photos, with no masks. Phase2_Plan §2 ruled it out for this reason.
3. **The event lists differ, and neither is clean.** The plan and the script name 8 "clean" events (La Palma, Ukraine, Acapulco, Kyaukpyu, Goma, Les Cayes, Boulder, Noto); the roadmap names 5 different ones. Checked on 19 Oct against DisasterM3's train and benchmark lists, all 14 BRIGHT events are in DisasterM3. Noto is in its benchmark with optical after-images, and the other 7 of those 8 appear only as radar questions (see 4.7).

BRIGHT's pre-disaster footprints also add only intact buildings, not the destroyed examples we're short of. The E1 diagnosis would show whether better building finding helps.

## 3. The common thread

Across these papers, specialist models are good at finding and counting, VLMs are good at grading and explaining, and the strongest systems split those jobs. That is already our hybrid design, so most papers sharpen the plan rather than change it.

## 4. Paper by paper

### 4.1 BDAChat (code repository), our closest competitor
- **Source:** Advanced Engineering Informatics 71, 104320 (2026). The paper is paywalled, so this comes from the MIT-licensed repo ([GitHub](https://github.com/WangYong921/BDAChat); README and figures saved in [BDAChat_repo/](<../others/lit review_research papers/BDAChat_repo/>)).
- **How it works:**
  1. A modified SAM finds buildings and roads. It is SAM ViT-B with an adapter, a LinkNet decoder and Multi-LoRA, fine-tuned on building masks, and it needs no prompts.
  2. A script (`sgts.py`) pairs each building's pre- and post-disaster crops.
  3. A Video-LLaVA-7B model (LoRA rank 128, 8-bit) gives each building a damage class, the disaster type and an explanation.
  4. The outputs are drawn as a fine-grained damage map and a damage-aware heatmap.
- **Training hardware:** DeepSpeed ZeRO-3 with CPU offload, so multi-GPU.
- **Key number:** macro F1 0.763 on xBD, at object level.
- **How we use it:**
  - Position against it: our SAM 3 finds buildings with no training, and we grade with a small classifier instead of a 7B model per building.
  - Reuse `sgts.py` for our R16 crop step and its heatmap script for O3.
  - Score our R16 grader with the same object-level metric so the two compare directly.

### 4.2 TEOChat, a VLM that reads before-and-after images
- **Source:** ICLR 2025, arXiv 2410.06234.
- **How it works:**
  - Video-LLaVA initialisation, with one shared CLIP image encoder for every image.
  - LLaMA 2 7B trained with LoRA in 8-bit on 554k examples.
  - Image labels ("Image 1", "Image 2") are placed between the image tokens.
  - It trained on a single 16 GB GPU.
- **Key numbers:**
  - Finding buildings on xBD: F1 38.9, against 85.9 for a specialist model.
  - Grading the damage of buildings it's given: F1 50.0, against 26.5 for the specialist.
  - "Are there destroyed buildings?" questions on xBD: 89.9% accuracy.
- **How we use it (O2, O3):**
  - It's direct evidence for our split: the VLM is bad at finding buildings and good at grading given ones.
  - Give Qwen2.5-VL both images, labelled "before" and "after", plus building boxes from our masks. DisasterM3 has a pre-disaster image for every entry.
  - Training this way fits on a T4.

### 4.3 GeoPixel, a VLM that outputs masks
- **Source:** ICML 2025, arXiv 2501.13925.
- **How it works:**
  - A 7B model (InternLM-XComposer-2.5) that reads images up to 4K by splitting them into up to 9 tiles of 560 px plus a scaled-down whole view.
  - A frozen SAM-2 encoder and a small decoder turn special `[SEG]` tokens into masks during the conversation.
  - Its dataset (GeoPixelD, 600,817 objects) was built with set-of-marks prompting.
  - It trained for about 3 days on 2× A6000 48 GB GPUs.
- **Limits for us:** it reads a single image only and isn't about disasters.
- **How we use it (O2, O3):** don't adopt the model, but borrow two ideas:
  1. **Tiling** (local tiles plus a whole view) for the "higher-resolution VLM" in KR2.
  2. **Set-of-marks:** draw numbered marks from our masks on the image, so the VLM can say "building 12 is destroyed" and every claim can be checked against a mask.

### 4.4 ImageRAG, picking the right parts of a big image
- **Source:** IEEE GRSM, arXiv 2411.07688.
- **How it works:**
  1. An LLM pulls key phrases out of the question.
  2. A remote-sensing CLIP model scores every tile of the image against those phrases.
  3. If a tile scores at least 0.5, the best tiles are used directly. Otherwise a fallback looks the phrases up in a database of labelled remote-sensing images and searches by image similarity.
  4. The chosen tiles go to the VLM with their coordinates ("Sub-patch 1 at location [x1,y1,x2,y2]").
- **Key numbers:**
  - Giving the VLM the correct region adds 8.7 points with no training.
  - A fine-tuned model with chain-of-thought adds 18 points, reaching 64.7%.
  - Counting accuracy still tops out at 42–50%.
- **How we use it (O2):**
  - Use our segmenter as the "retriever", since it finds buildings better than CLIP.
  - Send full-resolution crops of the damaged areas, with coordinates, alongside the whole image.
  - Try it with no training first.
  - Even ImageRAG can't count reliably, so counts should come from masks.

### 4.5 DamageScope, a database that answers damage questions
- **Source:** 2026 preprint, arXiv 2608.21529 (NEC Labs).
- **How it works:**
  1. Cuts each property out of the image using known building footprints, and sorts each crop into one of 4 damage groups.
  2. Clusters similar buildings within each group (RemoteCLIP features with hierarchical clustering).
  3. Asks GPT-4o to describe only one representative per cluster, as JSON.
  4. Stores the fields in SQLite and the explanations in a vector database (ChromaDB). An LLM decides whether each question needs an SQL query or a similarity search.
- **Key numbers:**
  - Loading the data is 14× faster (2,436 s down to 173 s for 139 buildings).
  - Tokens per question drop from 7,050 to 2,420, and latency from 2.10 s to 0.69 s.
- **Caveats:**
  - It was tested on only 2 xBD tiles, and answer accuracy wasn't measured.
  - It uses cloud GPT-4o.
- **How we use it (O2, O3, O4):** it's the blueprint for the offline DisasTeller back end.
  - Build one table row per building from our masks: ID, class, area, position.
  - Let Qwen write SQL, so "how many destroyed buildings within 100 m" is answered exactly by a query.
  - Explain only the cluster representatives, which means far fewer VLM calls on a T4 or laptop.

### 4.6 GEOBench-VLM, a benchmark
- **Source:** ICCV 2025, arXiv 2411.19325.
- **What it is:** 31 tasks with over 10,000 human-checked multiple-choice questions, including counting damaged buildings on xBD and naming the disaster type from before and after images.
- **Key numbers:**
  - The best model overall scores 41.7%.
  - For counting damaged buildings, GPT-4o scores 56.7% and Qwen2-VL 50.0%.
  - Every model fails when the answer options are within 20% of each other, so their numeric reasoning is weak.
- **How we use it (O2):**
  - It's an outside test for our hybrid counting, to compare with GPT-4o.
  - Check first that its xBD tiles don't overlap our training data.

### 4.7 BRIGHT, a dataset
- **Source:** ESSD 2025 and the 2025 IEEE GRSS Data Fusion Contest; arXiv 2501.06019.
- **What it is:**
  - Pre-disaster optical images with post-disaster radar (SAR) images, at 0.3–1 m resolution.
  - 14 events across 23 regions, giving 4,246 tile pairs of 1024×1024.
  - The same 4 classes as ours: background, intact, damaged, destroyed.
  - Among building pixels: 82.8% intact, 10.7% damaged, 6.5% destroyed.
- **Key findings:**
  - **Two-stage beats single-stage:** finding buildings first and then grading them beats one-step segmentation (ChangeMamba 67.63 mIoU against U-Net 64.94).
  - **The before image helps grading itself:** adding the pre-disaster image improves damage grading, not just building finding (DeepLabV3+ Destroyed IoU rises from 59.54 to 64.94).
  - **Post-processing:**
    - TTA adds 0.87 mIoU.
    - Object-based majority voting adds 0.69 per event but costs 0.41 overall.
    - Ensembling adds 0.82.
    - All three together add 1.23.
  - **Unseen disasters:** scores on new events fall below 40% mIoU.
  - **Post-image sensor:** optical post-disaster images beat radar (69.76 against 65.56 mIoU), and fusing both is best (70.79).
- **How we use it (O1):**
  - It's published evidence for R11 (feeding the before image) and for R16 (two-stage grading).
  - **Object-based majority voting** gives each predicted building blob the class most of its pixels have. It needs no training, and we can try it on R1 and R9 predictions right away.
  - **Data overlap:** all 14 BRIGHT events are in DisasterM3 (checked 19 Oct against its train and benchmark lists, which record each picture's sensor):
    - 6 of DisasterM3's 8 non-xBD training events come from BRIGHT, with optical after-images; Kalehe and Shovi don't come from BRIGHT;
    - Noto is in the benchmark, with optical after-images;
    - the other 7 appear only as radar questions, in both the training data and the benchmark.
  - **Optical images:** BRIGHT's post-disaster images are radar, so only its pre-disaster optical images (building footprints) suit our optical model. Radar is future work.

### 4.8 QLoRA
- **Source:** NeurIPS 2023, arXiv 2305.14314.
- **How it works:**
  - Keeps a frozen copy of the model in 4-bit (NF4) and trains only small LoRA adapters.
  - Double quantization and paged optimizers save further memory.
  - It fine-tunes a 65B model on one 48 GB GPU and matches full 16-bit fine-tuning.
- **Lesson:** data quality matters more than size (9k good examples beat 450k).
- **How we use it (O4):**
  - It's already our CSE499A method, via Unsloth. Use it for the 3B distillation.
  - Keep the teacher-generated data small and clean.

### 4.9 AWQ
- **Source:** MLSys 2024, arXiv 2306.00978.
- **How it works:**
  - 4-bit compression after training.
  - It finds the roughly 1% of weights that matter most from activation statistics and protects them by rescaling.
  - No retraining is needed, and it works on vision-language models.
- **Key numbers:**
  - 3.2–3.3× faster than the default FP16 code.
  - VILA-7B runs 3.1× faster.
  - It runs on Jetson Orin.
- **How we use it (KR4):**
  - QLoRA is for training and AWQ is for running the model. Compress the distilled 3B with AWQ.
  - Report memory (target 6 GB or less), tokens per second and per-task accuracy loss against the 7B.

### 4.10 Grace, a two-tier VLM system
- **Source:** 2025 preprint, arXiv 2510.24242.
- **How it works:**
  - A Qwen2-VL-2B runs on the satellite (a 16 GB Jetson Orin NX) and a Qwen2-VL-7B on the ground.
  - Both use retrieval from a stored archive. The ground updates the satellite's archive during each roughly 5-minute contact window.
  - A dispatcher keeps an answer on board only if the archive has relevant material and the model is confident. Confidence is the average probability the model gave its own output words. Otherwise the job goes to the ground.
- **Key numbers:**
  - 76–95% lower latency with no accuracy loss.
  - The 2B scores 55.9% against the 7B's 68.0% on RSVQA.
- **How we use it (O4):**
  - Apply the same confidence check to our field 3B and base 7B. When the 3B isn't confident, queue the question for the 7B.
  - Report how many questions stay on the 3B, the accuracy against the 7B alone, and the latency. That turns "two tiers" into a measured result.

### 4.11 EfficientSAM3
- **Source:** arXiv 2511.15833; weights on GitHub.
- **How it works:**
  - A recipe for distilling SAM 3 into small backbones (RepViT, TinyViT, EfficientViT), in three stages: image encoder, video memory, then end-to-end.
  - Only the image stage matters to us.
- **Caveat:** the paper reports no results yet ("Coming Soon").
- **How we use it (O4, L4):**
  - It's a candidate for finding buildings in the field.
  - Measure it ourselves against SAM 3 on the 645 validation pre-disaster images, for building IoU and time per image.
  - If it's weak on satellite images, our U-Net is the field fallback.

### 4.12 Distilling Step-by-Step, explanations as an extra training task
- **Source:** Findings of ACL 2023, arXiv 2305.02301 (Hsieh et al., University of Washington and Google). Added 19 October 2026; the abstract and method (§1–3) were checked against the PDF. Code: github.com/google-research/distilling-step-by-step.
- **How it works:**
  1. A large model, prompted with a few worked examples (chain-of-thought), writes an answer and a short explanation (a rationale) for each training example.
  2. The small student trains on two tasks with the same input, told apart by a task prefix: `[label]` gives the answer and `[rationale]` writes the explanation. The loss is the answer loss plus a weighted explanation loss.
  3. At test time the student only gives the answer. In the paper's words, the rationale "is not required in the test time", so explanations cost nothing at run time.
- **Key numbers** (4 language benchmarks):
  - Against plain fine-tuning and distillation: better results with over 50% fewer training examples on average, and up to 85% fewer.
  - A 770M-parameter T5 trained this way beat the 540B PaLM, prompted with a few examples, using only 80% of the available data on a benchmark task. Standard fine-tuning of the same T5 couldn't match it even with 100%.
- **How we use it (O4, O3):** it's the design for roadmap task L2b; the decision is recorded in §5.3.
  - L2's smoke test (19 Oct) showed that our fine-tuned 7B answers in one sentence and doesn't explain. So the explanations come from the base Qwen2.5-VL-7B. Unlike the paper, we give it the correct answer and true facts from DisasterM3's masks, because a VLM's own counts are unreliable, and keep an explanation only when it agrees with those facts.
  - The 3B learns two skills: a short answer to the normal prompt, and an explanation when the user asks "why?". Answers stay fast in the field (Grace, AWQ), and users can still ask for reasons (BDAChat's strength).

### 4.13 DisasterM3, the dataset paper (how it scores models, and its fine-tuning recipe)
- **Source:** NeurIPS 2025 Datasets and Benchmarks; arXiv 2505.21089v2. The local PDF has the main paper and the checklist (15 pages). Appendices B and C were read online on 19 October 2026.
- **How it scores (Table 2, §4, Appendix B.1):**
  - Table 2 is the "optical-optical setting": only questions whose after-image is a photo. Radar questions are reported separately (Figure 7).
  - Models run in vLLM with each model's default input and generation settings. Its script (`support/Abrar/Implementation_DisasterM3/temp_junjue/pyscripts/run_vllm.py`) raises the temperature to at least 0.01; for Qwen2.5-VL-7B that is effectively greedy.
  - Wrong counting options are generated at ±20% and ±40% of the true count (§3.2), so a counter must land within about 10%.
  - Accuracy changes with the prompt's wording (Figure 9), so results are compared on the same prompts.
- **Key numbers (Table 2, optical):**
  - The base Qwen2.5-VL-7B averages 31.2% (counting 34.2%).
  - The authors' fine-tune of it on the Instruct set averages 40.4% (counting 34.3%).
  - Fine-tuning didn't help counting for them either. They report overfitting and density bias in VLM counting (§4.2).
- **Fine-tuning recipe (Appendix B.3):** LoRA on the language model with the vision encoder frozen; rank 64, alpha 16, dropout 0.05; AdamW, lr 2e-4, β 0.9 / 0.95, cosine schedule; global batch 256; 1 epoch; 4× H100. Precision and image size aren't stated. The fine-tuned weights aren't released.
- **How we use it:** like-for-like comparisons (roadmap H1) and the recipe check behind roadmap task L0, a better 7B teacher before distilling.

## 5. Integration plan, cheapest first

### 5.1 Quick wins on our own code (from section 2)

| Order | Action | Goal | Cost |
|---|---|---|---|
| 1 | Rescore hybrid counting with intact buildings counted, using R1 or R9, on DisasterM3's counting questions. This gives the first real KR2 number. | O2 | Inference only |
| 2 | Textual bridge: add the U-Net counts to the question passed to `ask_model` in the interactive notebook | O2, O3 | One-line change |
| 3 | First survivability heatmap: distance transform plus Gaussian blur on the champion mask (about 30 lines) | O3 | CPU, no training |
| 4 | DisasTeller on local models: replace the chat and embedding models, then rebuild the Chroma database | O3 | Light GPU |
| 5 | Run the teacher chain-of-thought notebook, then the 3B training (clear the stale 7B outputs first) | O4 | Heavy GPU |

### 5.2 From the papers

| Order | Action | Comes from | Goal | Cost |
|---|---|---|---|---|
| 1 | Object-based majority voting on R1/R9 predictions | BRIGHT | O1 | CPU, no training |
| 2 | Per-building SQLite table + SQL counting, tested on DisasterM3 counting | DamageScope, TEOChat | O2 | No training |
| 3 | Before+after prompt with numbered marks and full-resolution crops of damaged areas | TEOChat, GeoPixel, ImageRAG | O2, O3 | No training first |
| 4 | R11 (before+after U-Net), then the R16 grader scored with the object-level metric | BRIGHT, BDAChat | O1 | ~8 GPU-h, then a small model |
| 5 | AWQ 4-bit 3B + Grace-style confidence handoff to 7B | AWQ, Grace, QLoRA | O4 | Light GPU |
| 6 | External counting test on GEOBench-VLM | GEOBench-VLM | O2 | Inference only |
| 7 | EfficientSAM3 vs SAM 3 measurement | EfficientSAM3 | O4 (L4) | Inference only |
| 8 | Explanations as a second skill for the distilled 3B: a short answer by default, an explanation when asked | Distilling Step-by-Step, BDAChat | O4, O3 (L2b) | ~3–5 GPU-h, plus the L2 training |

### 5.3 Decided: how the distilled 3B learns (19 October 2026)

**The question.** Users will talk to the VLM directly: they ask it questions, or give it new post-disaster images. Should the distilled 3B (roadmap L2) learn only short checked answers (**A**), or reason before every answer (**B**)? L2's smoke test settled one fact first: our fine-tuned 7B answers in one sentence, in the style of its training labels, even when the prompt asks it to think step by step.

**What the papers say:**

| Paper | Lesson | What it means for A vs B |
|---|---|---|
| BDAChat (§4.1), the closest competitor | Its main strength is explaining *why* a building is damaged, and users value that. Our plan borrows it by having the VLM explain a building after it's graded. | Explanations matter to users. Points to B. |
| ImageRAG (§4.4) | Fine-tuning with step-by-step reasoning added 18 points (to 64.7%). | Reasoning in the training data can raise accuracy. Points to B. |
| TEOChat, GEOBench-VLM (§4.2, §4.6) | VLMs are bad at finding and counting buildings, and their number reasoning is weak. | An explanation the model invents about counts or locations would be wrong, so explanations must rest on facts from the masks. A warning for B. |
| Grace, AWQ (§4.10, §4.9) | The field model must be fast: Grace's whole gain is lower latency, and AWQ is measured in tokens per second. | Reasoning before every answer makes every answer slower. Points to A. |
| QLoRA (§4.8) | Small, clean training data beats large, noisy data. | Whatever the 3B learns from must be checked, which L2's answer check does. |
| Our own design (Research_Directions §7–8) | Specialists find and count; the VLM reasons and reports; the 3B is the fast field model, and it hands hard questions to the 7B. | The 3B should answer fast and explain when asked. |
| Distilling Step-by-Step (§4.12) | Explanations trained as a second task make a small model more accurate with less data, and aren't needed at answer time. | Both: short answers by default, explanations on request. |

**A or B alone:**
- **A alone** (short checked answers): fast and reliable, but the 3B never learns to say why.
- **B alone** (reasoning before every answer): users always get a reason, but every answer is slower on a laptop. Our 7B can't write explanations either, so B needs another model to write them, one that may invent evidence.

**Decision: A now, then explanations as a second skill (roadmap L2b).**
1. **L2 step 1 runs as A:** the 7B's one-sentence answers, kept only when they match the label. Step 2 also trains the 3B on all of DisasterM3's tasks, including the damage descriptions and restoration advice, so it already writes full sentences when users ask for those.
2. **L2b adds explanations** as a separate, small run. A model is given the correct answer plus true facts from the masks (counts, where the damage is) and asked to explain what in the images shows it. Only explanations that agree with the facts are kept. The 3B then learns two skills:
   - with a normal question, it answers briefly and quickly;
   - when the user asks "why?", it explains.
3. **The 3B is not trained to reason before every answer.** That is slower in the field, clashes with the benchmark's letters-only answers, and risks teaching invented evidence.

**Update, 19 October 2026: improve the teacher first.** Like for like with the DisasterM3 paper (§4.13), our fine-tuned 7B beats the base model (36.1% against 31.2% on optical questions) but trails the authors' fine-tune of the same model (40.4%). A student learns from its teacher, so roadmap task L0 improves the 7B before L2 runs. How the 3B learns from it (answers only, or soft labels) is still to choose.

## 6. What not to adopt
- **GeoPixel's single big model:** too heavy, and it has no before image.
- **BRIGHT's radar data:** future work.
- **DamageScope's cloud GPT-4o:** use our local Qwen instead.

## 7. Check before citing
- **Not peer-reviewed yet:** DamageScope (2026), Grace (2025) and EfficientSAM3, which also has no results.
- **BDAChat:** these details come from its repo, because the full paper is paywalled and NSU doesn't subscribe to the journal on ScienceDirect (checked 18 October 2026). Before quoting more than the abstract and README, request the PDF from the corresponding authors (Jiawei Cui and Changhai Zhai; their emails are behind the envelope icons on the article page, DOI 10.1016/j.aei.2026.104320), or through ResearchGate or an NSU interlibrary loan.
- **GEOBench-VLM:** check for overlap with our xBD training tiles before claiming a result.
- **Distilling Step-by-Step:** its abstract and method (§1–3) were checked against the PDF on 19 October 2026. Check the result tables before quoting any other number.
- **DisasterM3's appendix:** read online (arXiv HTML, v2). The text extraction doubled some values ("6464", "1616"); they were read as rank 64 and alpha 16, which our training notebook's header also cites. Check the PDF of the supplementary material before quoting other appendix numbers.

## 8. Sources (local copies)

| Source | File in `others/lit review_research papers/` |
|---|---|
| BDAChat (repo) | `BDAChat_repo/README.md`, `BDAChat_repo/assets/` |
| TEOChat | `TEOChat A Large Vision-Language Assistant for Temporal Earth Observation Data.pdf` |
| GeoPixel | `GeoPixel Pixel Grounding Large Multimodal Model in Remote Sensing.pdf` |
| ImageRAG | `ImageRAG Enhancing Ultra High Resolution Remote Sensing Imagery Analysis with ImageRAG.pdf` |
| DamageScope | `DamageScope Vision–Language Retrieval at Scale for Disaster Damage.pdf` |
| GEOBench-VLM | `GEOBench-VLM Benchmarking Vision-Language Models for Geospatial Tasks.pdf` |
| BRIGHT | `BRIGHT A globally distributed multimodal building.pdf` |
| QLoRA | `QLoRA Efficient Finetuning of Quantized LLMs.pdf` |
| AWQ | `AWQ Activation-aware Weight Quantization for LLM Compression and Acceleration.pdf` |
| Grace | `Grace Enabling Near-realtime Remote Sensing via Satellite-Ground Collaboration of Large Vision-Language Models.pdf` |
| EfficientSAM3 | `EfficientSAM3 Progressive Hierarchical Distillation for Video Concept Segmentation from SAM1, 2, and 3.pdf` |
| Distilling Step-by-Step | `Distilling Step-by-Step Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes.pdf` |
| DisasterM3 | `NeurIPS-2025-disasterm3-a-remote-sensing-vision-language-dataset-for-disaster-damage-assessment-and-response-Paper-Datasets_and_Benchmarks_Track.pdf` (main paper and checklist; appendices online, arXiv 2505.21089v2) |

Related documents: [Research_Directions_Beyond_Finetuning.md](Research_Directions_Beyond_Finetuning.md) (SAM 3 plan, positioning against GeoPixel and BDAChat) · [Phase2_Plan.md](Phase2_Plan.md) (E1, R11, R12) · [Next_Steps_Roadmap.md](Next_Steps_Roadmap.md) (order of work).
