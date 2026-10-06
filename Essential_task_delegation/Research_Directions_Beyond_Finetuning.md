# Beyond Fine-Tuning: Research Directions, Related Work and the SAM 3 Plan

**Context:** A record of the October 2026 discussion on what to do once segmentation fine-tuning stopped producing gains. It covers five structural options, current top-venue work in our domain, an assessment of six candidate techniques (RAG, agents, diffusion, PNNs, LLMs, edge devices), the xView2 winners, an explainer on foundation models and SAM, a SAM 3-based design with a test plan, and how that design compares with GeoPixel and BDAChat. Use it for the 499B report's related-work and future-work sections, and for viva questions.

**Last updated:** 6 October 2026. Venues and figures were checked against the papers, project pages or official repositories (links in [Sources](#11-sources)). Items marked *preprint* are not yet peer-reviewed; check them before citing. Our own numbers come from the D4 champion notebooks in `Champion_Task/`.

**Related docs:** [Segmentation_Model_Explained.md](Segmentation_Model_Explained.md) (how our U-Net and dataset work) · [segmentation_improvement_guide.md](segmentation_improvement_guide.md) (Tiers 1–6) · [team_task_assignments.md](team_task_assignments.md) (Phase Gate, Task D5) · [Ablation_Study_Reference_and_QA_Prep.md](Ablation_Study_Reference_and_QA_Prep.md) (verified sprint numbers and errata).

## Contents
1. [Why look beyond fine-tuning](#1-why-look-beyond-fine-tuning)
2. [Five structural levers](#2-five-structural-levers)
3. [Related work at top venues](#3-related-work-at-top-venues)
4. [The xView2 winners](#4-the-xview2-winners)
5. [Six candidate techniques, assessed](#5-six-candidate-techniques-assessed)
6. [Background: foundation models, SAM and pixel masks](#6-background-foundation-models-sam-and-pixel-masks)
7. [The SAM 3 plan](#7-the-sam-3-plan)
8. [Positioning: GeoPixel, BDAChat and our design](#8-positioning-geopixel-bdachat-and-our-design)
9. [Recommended order for the rest of 499B](#9-recommended-order-for-the-rest-of-499b)
10. [Corrections made during this review](#10-corrections-made-during-this-review)
11. [Sources](#11-sources)

---

## 1. Why look beyond fine-tuning

The D4 champion runs changed one setting at a time from the R1 recipe (U-Net + ResNet-50, Focal Tversky → Lovász, 768 crops, 40 epochs) and scored every run the same way: full validation images, last epoch.

| Run | Change from R1 | mIoU | Destroyed IoU |
|---|---|---|---|
| R1 | reference | 0.5374 | 0.2686 |
| R1b | same config and seed, run again (this was meant to be R4, see §10) | 0.5427 | 0.2796 |
| R6 | seed 7 | 0.5402 | 0.2815 |
| R2 | copy-paste | 0.5369 | 0.2681 |
| R3 | Focal Tversky 0.25 / 0.75 / 1.33 | 0.5402 | 0.2790 |
| R5 | weighted sampler | 0.5354 | **0.2898** |
| R8 | enhanced augmentation | 0.5342 | 0.2703 |

What this shows:
- **Every run lands between 0.534 and 0.543 mIoU.** Two runs with the identical config and seed differ by 0.005, and Destroyed IoU varies by about 0.013 between identical recipes. All the differences are within noise.
- **The sampler (R5) is the only change that clearly helped Destroyed IoU.** It lowered Intact IoU in exchange.
- **No run passes the Phase Gate's Destroyed IoU > 0.30.**

**Conclusion:** more tuning of losses, sampling or augmentation is unlikely to help. Bigger gains need *structural* changes: what the model sees, how the problem is framed, what data it learns from, or what the project contributes.

---

## 2. Five structural levers

Ranked by expected gain for the effort.

| # | Lever | What changes | Why it should help | Cost |
|---|---|---|---|---|
| 1 | **Give the model the pre-disaster image too** | Our model sees only the post-disaster image. DisasterM3 includes a matching pre-disaster image (`*_pre_*.png`) for every training photo. | The model can compare each building before and after, instead of guessing whether rubble used to be a building. All five xView2 winners (§4), ChangeMamba's damage model and TEOChat use pre + post pairs. | **Low.** `smp.Unet(in_channels=6)` on the stacked pair is one model change, plus loading the pre image with the same crop and flips. A Siamese encoder is the stronger version. |
| 2 | **Two stages: find buildings, then grade each one** | Stage 1 segments building vs background. Stage 2 classifies each building's damage from its pre/post crops. | Rare *pixels* become rare *objects*: the training set has 40,704 damaged and 16,682 destroyed building instances, compared with 0.33% of pixels. Counting comes out per building. This is how xView2's 1st and 2nd place solutions worked, and it's Tier 5 of our improvement guide. | **Medium.** A new pipeline, but each stage is simple. |
| 3 | **More data instead of more tuning** | Add xBD (already on this PC under `data/xView2 Challenge Dataset - train and test`) or BRIGHT, for pre-training or joint training. | More rare-class examples is the most reliable fix for imbalance. This is Phase 2, Task D5 in `team_task_assignments.md`. | **Medium.** **Check for overlap first:** DisasterM3 already contains xBD-style events (e.g. `woolsey_fire_…` in its Bench split), so adding xBD could leak validation or test images. |
| 4 | **Start from foundation models, train less** | Use SAM / SAM 2 / SAM 3 for building outlines, or AnyChange (zero-shot change detection), and train only a small damage head. | Fewer settings to tune, and we inherit models trained on far more data. | **Low to medium.** See §6–§7. |
| 5 | **Change the contribution, not the model** | Freeze R1 as the segmentation champion. Spend 499B on the system: hybrid counting, offline DisasTeller, 7B→3B distillation and an edge benchmark. | The title is *Resource-Efficient Neural Architectures…*; a measured, working offline system is the contribution. | **Low to medium**, and already in objectives O2–O4. |

The instance counts in lever 2 come from the copy-paste bank built from training images only, keeping instances of at least 100 pixels (`Champion_Task/combined-attempt (1).ipynb` output).

**Also:** report results with the **xView2 metric** (0.3 × building-localisation F1 + 0.7 × damage F1) alongside our mIoU, so our numbers can be compared with published work.

---

## 3. Related work at top venues

NeurIPS, ICLR, ICML, ICCV, CVPR and AAAI are top-tier (CORE A*) conferences. IEEE TPAMI and TGRS are leading journals, not conferences.

| Work | Venue | What it does | Relevance to us |
|---|---|---|---|
| **DisasterM3** | NeurIPS 2025 (Datasets & Benchmarks) | Our dataset: 26,988 bi-temporal images, 123k instruction pairs, 36 events, 9 tasks, optical + SAR | Our benchmark |
| **TEOChat** | ICLR 2025 | A VLM that reads *time series* of satellite images, trained on tasks including xBD damage assessment | Shows that giving a VLM pre + post images improves damage reasoning (lever 1) |
| **GeoPixel** | ICML 2025 | A remote-sensing multimodal model that answers with **pixel masks** attached to its text, at up to 4K resolution | The "answer with masks" experience we want, built as one large model (§8) |
| **GEOBench-VLM** | ICCV 2025 | Benchmark of VLMs on 31 geospatial sub-tasks. The best model (GPT-4o) gets only about 40% on multiple-choice questions. | Evidence that VLMs alone can't localise or count reliably, which justifies our hybrid design |
| **Segment Any Change (AnyChange)** | NeurIPS 2024 | Zero-shot change detection built on SAM, no training | Lever 4 |
| **DiffusionSat** | ICLR 2024 | Generative foundation model for satellite images: generation, super-resolution, inpainting | Diffusion for remote sensing (§5) |
| **ChangeDiff** | AAAI 2025 | Diffusion model that generates change-detection training pairs *and labels* from text prompts | Synthetic rare-class data |
| **Changen2** | IEEE TPAMI 2025 (journal) | Generative change foundation model; pre-trained versions transfer to disaster assessment; synthetic data released | Synthetic data and pre-training |
| **ChangeMamba** | IEEE TGRS 2024 (journal) | State-space (Mamba) change detection, including a building-damage model on xBD | A strong pre/post baseline |
| **Change-Agent** | IEEE TGRS 2024 (journal) | An LLM "brain" that calls a change-detection model as a tool for detection, captioning, counting and cause analysis | The agent pattern for our hybrid |
| **BDAChat framework** | Advanced Engineering Informatics, Vol. 71 Part B, 2026 (journal) | Modified SAM + per-building pre/post pairing + a fine-tuned temporal VLM (BDAChat) for interpretable object-level damage assessment; builds a new dataset (OLBDA) | **Our closest related work** (§8) |
| **BRIGHT** | ESSD 2025 (journal); IEEE GRSS Data Fusion Contest 2025 | Optical + SAR building-damage dataset, 14 regions, over 380,000 buildings | External data (lever 3) and all-weather use |

**Preprints to watch (not peer-reviewed):**
- **GeoDisaster** and **DisasterBench**: benchmarks for agent workflows in disaster response.
- **HASTE**: a rapid damage-assessment platform.
- **SegEarth-OV3** and **OmniOVCD**: SAM 3 applied to remote sensing (§7).

---

## 4. The xView2 winners

The xView2 challenge was run in 2019 by the US Defense Innovation Unit on the xBD dataset. Its scoring is 0.3 × localisation F1 + 0.7 × damage F1. The top five solutions are public in the [DIUx-xView GitHub organisation](https://github.com/DIUx-xView).

| Place | Winner | Pre/post handling | Stages | Models |
|---|---|---|---|---|
| **1st** | **Victor Durnov** | Building-finding models trained on **pre-disaster images only** (buildings are clearest before damage), then converted into **Siamese** damage classifiers that read pre and post with shared weights | **Two** | Ensemble of ResNet-34, SE-ResNeXt-50, SENet-154 and DPN-92 encoders. Dice + Focal loss (+ cross-entropy for damage), sampling that favours damaged images. |
| **2nd** | **Selim Seferbekov** | U-Net finds buildings; a **Siamese U-Net** with a shared encoder compares pre and post | **Two** | DPN-92 and DenseNet-161 encoders |
| **3rd** | **Eugene Khvedchenya** ("BloodAxe") | **Shared encoder** for pre and post; features joined and passed to one decoder | One | ResNet / DenseNet / EfficientNet encoders with U-Net and FPN decoders; weighted cross-entropy, heavy augmentation, pseudo-labelling. Private test score 0.805. |
| **4th** | **Zhuo Zheng** ("Z-Zheng") | Building localisation and damage end to end, with a "guided ResNeXt Siamese" network | One | ResNeXt Siamese. The same GitHub account later published AnyChange and Changen2. |
| **5th** | **SI Analytics** (team) | **Dual-HRNet**: one HRNet branch for pre, one for post, features merged at every stage | One | HRNet |

**Lessons for us:**
- **All five used both pre and post images**, so lever 1 matters more than anything else.
- **Two stages was the choice of the top two, not all five.**
- **1st place found buildings on the pre-disaster image.** Our model finds buildings on the post image, where destroyed buildings no longer look like buildings. That's part of why Destroyed is so hard for us.

---

## 5. Six candidate techniques, assessed

"PNN" has two common meanings, so both are covered.

| Technique | What it would do in our project | Related work | Verdict |
|---|---|---|---|
| **RAG** | DisasTeller already retrieves from EMS-98. Two upgrades: **(a) knowledge RAG**, retrieving damage scales, building codes and rescue protocols so reports cite sources instead of inventing them; **(b) visual RAG**, selecting only the relevant high-resolution patches for the VLM, which works around its 512-token image cap. | ImageRAG (IEEE GRSM 2025) | **Do it.** No training, cheap, and it targets the VLM's spatial blindness. It won't raise segmentation IoU. |
| **Agentic workflow** | Make the hybrid explicit: a local LLM plans and calls tools (segmenter, counter, collage builder, EMS-98 retriever, heatmap). This is objective O3, an offline DisasTeller. | Change-Agent (TGRS 2024); SAM 3 Agent; 2026 agent benchmarks | **Do it, as the integration layer.** Keep the tools deterministic, because agents add latency and new ways to fail. Evaluate on DisasterM3's tasks. |
| **Stable diffusion** | (1) synthetic damaged buildings for rare classes; (2) super-resolution, following on from our VRT study; (3) inpainting. | ChangeDiff (AAAI 2025), Changen2 (TPAMI 2025), DiffusionSat (ICLR 2024), Neural Disaster Simulation (Remote Sensing of Environment 2025) | **Future work.** Ordinary Stable Diffusion is trained on web photos, not satellite images. Synthetic labels can be wrong, it's compute-heavy, and R2 showed that simply adding instances (copy-paste) didn't help. Cheapest test: pre-train on the released Changen2 synthetic data. |
| **PNN: Progressive Neural Networks** | Add a new network "column" per new disaster type without forgetting old ones (continual learning) | Rusu et al., 2016 | **Skip.** The model grows with every disaster type, which works against the edge goal. Joint training or per-event fine-tuning is simpler. |
| **PNN: Physics-Informed Neural Networks** | Build physics equations (e.g. how flood water spreads) into the network to simulate floods or tsunamis. Could someday feed a survivability heatmap. | Raissi et al., 2019; physics-informed graph networks for flood forecasting (CACAIE 2025) | **Out of scope.** They predict how a hazard spreads; they don't segment damage from images. |
| **LLMs** | We already use Qwen2.5-VL. Best roles: report writing, reasoning and tool orchestration. Pixel-grounded models (GeoPixel; LISA, CVPR 2024) output masks directly but are 7B+ and heavy for a T4. | TEOChat, GeoPixel, GEOBench-VLM | **Keep the LLM for language and planning**; specialist models handle pixels and counts. Feeding it pre + post images (as TEOChat does) is worth a test. |
| **Edge devices** | The core of our title. Convert the U-Net (~33M parameters, 130 MB) to INT8 with ONNX/TensorRT (~33 MB); try lighter backbones (ResNet-34, SegFormer-B0/B2); quantise the 3B student to 4-bit (~2–3 GB). Measure **latency, memory and accuracy drop** on a laptop CPU or a Jetson. | Grace (preprint, 2025): a 2B VLM on a satellite with a 7B model on the ground; QLoRA (NeurIPS 2023); AWQ (MLSys 2024); EfficientSAM3 | **Highest priority.** Achievable, clearly measurable, and the most defensible contribution for a "resource-efficient" project. |

---

## 6. Background: foundation models, SAM and pixel masks

### 6.1 What a foundation model is
A **foundation model** is a very large model trained once, by a big lab, on huge and varied data. Everyone else then reuses it instead of training from scratch. We already use the idea twice:
- **U-Net encoder:** ResNet-50 pretrained on ImageNet (1.2M photos), fine-tuned on DisasterM3.
- **VLM:** Qwen2.5-VL-7B, fine-tuned with QLoRA (small adapter layers only).

"Start from foundation models, **train less**" means keeping the big model **frozen** and training nothing, or only a **small head**. That leaves far fewer settings to tune.

### 6.2 SAM, SAM 2 and SAM 3 (Meta AI)

| | SAM | SAM 2 | SAM 3 |
|---|---|---|---|
| Published | ICCV 2023 | ICLR 2025 | November 2025 (arXiv 2511.16719) |
| How you prompt it | Click, box or rough mask | Same, plus video | Same, **plus text** ("building") or example boxes |
| Output | Outline of the thing you pointed at, or every object (automatic mode) | Same, carried across video frames by a memory | **Every** instance matching the concept, in one pass |
| Size | Largest about 636M parameters | 38.9M (Tiny) / 46M (Small) / 80.8M (Base+) / 224.4M (Large) | About 848M (vision ~450M, text ~300M, detector and tracker ~100M) |
| Notes | Trained on SA-1B: about 11M images, 1.1B masks | More accurate than SAM on images and about 6× faster | Includes "SAM 3 Agent" (an LLM that breaks complex questions into simple prompts) |

How SAM-family models work:

```
Image ──► Image encoder (big, run once per image) ──┐
                                                     ├──► Mask decoder (small, fast) ──► outline(s)
Prompt (click / box / mask / text) ──► Prompt encoder┘
```

**What SAM does not do:** SAM 1 and 2 have no idea what classes are. They cut out "a thing" but can't say whether it's a building, or whether it's intact or destroyed. SAM 3 can find "buildings" by text, but whether it can grade *damage* is unproven (§7.1).

For our 33M-parameter U-Net, SAM 2 Tiny is a comparable size. The original SAM and SAM 3 are about 20–25× larger.

### 6.3 What a "pixel mask" is, and how ours differ from GeoPixel's
A **pixel mask** is an image-sized map marking which pixels belong to something. White means "belongs", black means "doesn't". What differs is who produces it, when, and why.

| | Our U-Net's mask | GeoPixel-style masks |
|---|---|---|
| What it is | **One** map with 4 values (0/1/2/3), every pixel labelled | **Several** black-and-white masks, one per thing mentioned |
| Classes | Fixed: background, intact, damaged, destroyed | Whatever the conversation is about ("the flooded road near the river") |
| Trigger | Every image, automatically | A user's question |
| Output | Mask only | **A text answer with masks attached to its phrases** |
| Separate objects | No: all destroyed pixels share one value | Usually yes: each building mentioned gets its own mask |
| Model | Small U-Net (~33M parameters) | Large multimodal model (7B+) |

**Our dataset already uses GeoPixel's format.** DisasterM3's segmentation task is "Referring Expression Segmentation": text prompt in, black-and-white mask out. For example, *"Classify and segment the undamaged buildings."* → `masks/train_building_intact_mask/bata_explosion_0.png`. Our notebook stacks the three per-class answers into one 4-class map so a plain U-Net can learn all classes at once.

---

## 7. The SAM 3 plan

SAM 3 connects four earlier ideas: the pre-disaster image, the two-stage pipeline, foundation models and the agent layer. It also shrinks the "too many variables" problem, because SAM 3 stays frozen.

### 7.1 Practical facts

| Question | Answer | Implication |
|---|---|---|
| Size | ~848M parameters, ~3.4 GB of weights | About 25× our U-Net. A **server** model, not an edge model. |
| Speed and memory | ~30 ms per image on an H200. One source says it fits on 16 GB GPUs; another measured a 19.5 GB peak on an A10. | **A Kaggle T4 (16 GB) is unconfirmed.** Smoke-test first. The T4 lacks fast bf16, so run it in fp16. |
| Zero-shot buildings on satellite images | SegEarth-OV3 (*preprint*, 2026) ran SAM 3 with **no training** and reports building IoU of **64.3% on xBD**, 86.9% on WHU-Aerial and 72.4% on Inria. | Building-finding without training is realistic; trained xView2 solutions are still better at it. |
| Weak spots | Lower-resolution or blurry imagery (42% mIoU on GF-2 satellite data); trained mostly on sharp natural photos | Rubble on post-disaster images is exactly this kind of texture. |
| Complex questions | **SAM 3 Agent**: a multimodal LLM breaks complex questions into simple noun phrases, calls SAM 3, checks the masks and repeats. Zero-shot, it beats earlier methods on reasoning-segmentation benchmarks. | This is our agent layer. |
| Change detection | OmniOVCD (*preprint*, 2026): training-free change detection with SAM 3 on LEVIR-CD, WHU-CD, S2Looking and SECOND. **No damage-grading results.** | Zero-shot damage *grading* with SAM 3 is unproven. |
| Edge versions | **EfficientSAM3**: distilled students of ~89–95M parameters (~90% smaller), ONNX and CoreML export | A path to field devices |
| Access | Weights **gated on Hugging Face** under the SAM Licence; some academic requests have been rejected | **Request access early.** Fall back to EfficientSAM3 or SAM 2. |

### 7.2 Three-layer design

```
                    ┌──────────────── REASONING LAYER ("brain") ────────────────┐
 User question ───► │ Qwen2.5-VL (7B at base / 3B in field), SAM-3-Agent style  │
 "How many destroyed│ splits the question into simple prompts, calls tools,      │
  buildings within  │ computes counts and distances itself, writes the report   │
  100 m of the      │ + RAG: EMS-98 damage scale, rescue protocols (DisasTeller) │
  flood?"           └───────────┬──────────────────────────────┬────────────────┘
                                │ "building"                   │ "flood water"
                    ┌───────────▼──────── PERCEPTION LAYER ("eyes") ───────────────┐
 PRE image  ──────► │ Stage 1: SAM 3 "building" on the PRE image                     │
                    │          → list of every building (no training)                │
 PRE + POST crops ► │ Stage 2: small Siamese classifier per building                 │
                    │          → intact / damaged / destroyed (the only part trained)│
                    └───────────┬──────────────────────────────────────────────────┘
                                ▼
            per-building masks + labels → counts, maps, heatmaps, report
                    ┌────────────────── EFFICIENCY LAYER ──────────────────────────┐
                    │ Base station: full SAM 3 + 7B        Field: EfficientSAM3 or   │
                    │ (or SAM 3 as a teacher that           our U-Net + 3B 4-bit     │
                    │  pseudo-labels data for small models)                          │
                    └────────────────────────────────────────────────────────────────┘
```

### 7.3 How the earlier ideas fit in

| Earlier idea | Where it shows up |
|---|---|
| Lever 1: pre-disaster image | SAM 3 finds buildings on the **pre** image, where they're intact and sharp (xView2 1st place's idea). Destroyed buildings are found by footprint even though they're rubble in the post image. |
| Lever 2: two stages | Stage 1 is SAM 3, stage 2 a small classifier. Rare pixels become rare objects (40,704 damaged, 16,682 destroyed instances). |
| Lever 4: foundation models | SAM 3 is frozen and adds **zero** settings to tune; only one small classifier is trained. |
| Agentic workflow + RAG | The VLM is the SAM 3 Agent "brain". DisasterM3 contains these questions, e.g. "Segment intact buildings near the flooding (within 100m)", and building counting, where our VLM alone fell 10.62 points in 499A. |
| Edge devices | Two deployment tiers, matching Grace's satellite-and-ground split. SAM 3 can act as a teacher whose pseudo-labels train small field models. |

### 7.4 Test plan, cheapest steps first

| Step | What | Training? | Effort | Pass check |
|---|---|---|---|---|
| **S0** | Request Hugging Face access. Smoke-test SAM 3 on a Kaggle T4: does a 1024×1024 image fit in memory, and how long does it take? | No | ½ day | Runs on a T4 (otherwise 2×T4 or EfficientSAM3) |
| **S1** | Prompt "building" on the 645 validation images, **both pre and post**. Measure building IoU against the combined ground-truth masks and per-image building counts, and compare with R1's building-vs-background IoU. | No | 1 day | SAM 3 on the pre image ≥ R1 → it becomes stage 1 |
| **S2** | Zero-shot damage prompts on post images ("destroyed building", "collapsed building", "rubble"). Compare Destroyed IoU with R1's 0.27. | No | ½ day | Likely to fail; a clear negative result is still worth reporting |
| **S3** | Stage 2: train a Siamese ResNet-18 on pre+post crops of training buildings. On validation, classify the buildings SAM 3 found and score with our full-image protocol **and** the xView2 metric. | Small model only | 2–3 days | Destroyed IoU beats R1 by more than the noise bar (0.013) |
| **S4** | Connect the VLM as the agent. Evaluate on DisasterM3's counting and referring-segmentation questions against our 499A scores (34.2% base, 23.58% fine-tuned). | No | 2–3 days | Counting beats the 34.2% base model (objective O2) |
| **S5** | Edge tier: EfficientSAM3 vs SAM 3 accuracy and latency, or distil SAM 3's pseudo-labels into our U-Net. | Optional | 2 days | A measured trade-off table for the report |

S0 and S1 take about two days and need no training. They decide whether the whole direction is viable.

### 7.5 Risks
1. **Gated access** is denied.
2. **SAM 3 doesn't fit** or runs too slowly on a T4.
3. **Pre and post images aren't perfectly aligned.** xView2 1st place dilated masks to absorb shifts; we can crop with a margin.
4. **The remote-sensing SAM 3 results are 2026 preprints**, not peer-reviewed.

---

## 8. Positioning: GeoPixel, BDAChat and our design

### 8.1 Is a GeoPixel-style answer our goal?
Yes, as the *user experience*. The 499B goal in our review deck is *"a hybrid system that counts from pixels, reasons in language, and runs offline on free or small GPUs"*. Objective O3 asks for *"a report, damage map, counts and survivability heatmap."* What differs is the architecture: GeoPixel uses one large model, while we use cooperating specialist parts.

What our system's answer would look like:

```
User:    "Which buildings were destroyed?"

Pixels:  U-Net (or SAM 3 + damage classifier) → each building as its own mask + label
Counts:  computed from the masks, not guessed           → 3 destroyed, 1 damaged
Words:   VLM writes the answer using those facts + EMS-98 retrieval

Answer:  "3 buildings near the coast were destroyed (#4, #7, #12, red on the map)
          and the warehouse in the centre (#9, orange) is damaged.
          EMS-98 grade 5 for the destroyed ones: prioritise search and rescue there."
          └─ map overlay with each numbered building highlighted
```

What's still missing:
1. **Per-building masks.** The U-Net produces one 4-class map. We need each building separated, either by splitting the map into connected regions or with SAM 3.
2. **Linking words to masks.** Give each building an ID that the VLM refers to.
3. **Offline report generation.** DisasTeller still calls Gemini; objective O3 replaces it with our own model.

### 8.2 Three-way comparison

BDAChat details come from its abstract and search summaries; ScienceDirect blocks automated access to the full text. Cells marked *not stated* need checking in the PDF (try NSU library access).

| | **GeoPixel-style** (ICML 2025) | **BDAChat framework** (Adv. Eng. Informatics 2026) | **Our SAM 3 design** |
|---|---|---|---|
| Overall shape | One big model does everything | Modular: modified SAM → pair each building's pre/post views → temporal VLM | Modular: SAM 3 → per-building classifier → VLM agent |
| Who finds buildings | The model itself | A **modified SAM** (prompting *not stated*) | **SAM 3, text prompt "building", frozen**, on the pre image |
| Uses pre + post | No, single image | **Yes**, paired per building | **Yes**: pre to find, pre + post to grade |
| Who grades damage | Not damage-specific | **The VLM itself**, per building | **A small Siamese CNN**; the VLM doesn't grade |
| Explanations | Text with masks attached | **Its main strength**: causal reasoning on *why* a building is damaged, plus disaster recognition | VLM report + EMS-98 retrieval → grades and rescue advice |
| Counting | Estimated by the model | Possible from per-building output; *not stated* as a focus | **Explicit goal**: counts from masks |
| Training needed | Fine-tune a large pixel-grounding model | Modified SAM + fine-tuned VLM on a new dataset | SAM 3 frozen; one small classifier; reuse our Qwen fine-tune |
| Data | Their own grounding dataset (GeoPixelD) | A new dataset they built (OLBDA): bi-temporal, multi-hazard, object-level | DisasterM3 (NeurIPS 2025): 9 tasks, including counting and referring segmentation |
| Hardware / offline | Heavy | *Not stated* | **Designed for it**: T4, plus a field tier, offline |
| Complex questions | Partly | *Not stated* | **Yes**, via the SAM-3-Agent-style brain |

How it fits our principles (GeoPixel vs our SAM 3 design):

| Our principle | GeoPixel-style | Our SAM 3 design |
|---|---|---|
| Counts from pixels | ❌ estimated by the model | ✅ from per-building masks |
| Reasons in language | ✅ | ✅ our fine-tuned Qwen |
| Offline on free or small GPUs | ❌ 7B+ with 4K inputs | ⚠️ full SAM 3 is server-sized, but a field tier exists |
| Reuses our work | ❌ new large model | ✅ Qwen fine-tune, R1 U-Net, DisasTeller |
| Fewer variables to tune | ❌ more | ✅ SAM 3 frozen, one small classifier |
| Fits objectives O1–O4 | Partly (O2) | All four |

### 8.3 The key design difference from BDAChat: who grades the damage

| | BDAChat: the VLM grades | Ours: a small classifier grades |
|---|---|---|
| Strength | Grade and explanation come from one model, which makes it interpretable | Cheap, fast, easy to measure, works on small hardware |
| Weakness | Needs a large fine-tuned VLM for every building | Explanations come from the VLM afterwards, not from the grader |

We can borrow BDAChat's strength cheaply: after the classifier grades a building, ask the VLM to explain that crop ("roof collapsed, debris field"). This is optional and needs no training.

### 8.4 What we can and cannot claim
**We cannot claim** to be the first to combine SAM with a VLM for building damage. BDAChat already does object-level segmentation plus a temporal VLM. Cite it directly.

**Defensible contributions:**
1. **No training and no manual prompts to find buildings**: SAM 3 works from the text "building". Check that BDAChat's modified SAM doesn't already do this.
2. **Resource efficiency**: free T4, two-tier deployment, distilled 3B, quantisation. This is the title's core claim.
3. **Reliable counting** from masks, measured on DisasterM3's counting task.
4. **Agent-style questions** combining several concepts (e.g. buildings near flooding).
5. **Offline, rescue-oriented output**: EMS-98-grounded reports instead of a cloud API.

**Draft related-work sentence:**
> *BDAChat (2026) showed that segmenting buildings and grading them with a fine-tuned temporal VLM yields interpretable object-level damage assessment. We pursue the same object-level formulation under a resource constraint: buildings are found zero-shot by a text-prompted foundation model (SAM 3), damage is graded by a lightweight bi-temporal classifier, and a distilled VLM is reserved for reasoning and reporting, enabling offline deployment on free or edge hardware.*

**Summary:**
- **GeoPixel** is the *experience* we want, built as one heavy model. Cite it for contrast.
- **BDAChat** is our *closest competitor* in design. Position against it carefully.
- **Our SAM 3 design** is BDAChat's object-level idea, rebuilt to be cheap, countable and offline.

---

## 9. Recommended order for the rest of 499B
1. **SAM 3 steps S0–S1** (about 2 days, no training). They decide whether the SAM 3 direction is viable.
2. **A pre + post input run** for the U-Net (lever 1). It's one change, tested with the same noise-bar method as R1–R8.
3. **An edge benchmark**: INT8 U-Net and 4-bit 3B VLM, with latency, memory and accuracy.
4. **RAG plus the agent layer**, to build the offline DisasTeller (O3).
5. **Two-stage object-level damage** (SAM 3 S3), if S1 passes.
6. **Diffusion-generated data and PNNs** as future work in the report.

---

## 10. Corrections made during this review
- **`segmentation_improvement_guide.md` (and `.html`), corrected 6 Oct 2026:**
  - The two-stage section credited the xView2 winners to "DeepDamageNet, 2024". The challenge ran in 2019 and DeepDamageNet is an unrelated 2024 preprint. It now names the real winners (§4).
  - The class mix inside buildings read "~58% intact", which didn't add up to 100%. It's ~81% intact, 14.5% damaged and 4.6% destroyed (5.86 / 1.05 / 0.33 out of the 7.24% of pixels that are buildings).
  - `Ablation_Study_Reference_and_QA_Prep.md` notes both fixes. Copies inside `.kilo/worktrees/` were not changed.
- **R4 never ran U-Net++.** Its notebook still had `EXPERIMENT = "R1"`, so it trained R1 again (R1b in §1). It also likely pushed to the R1 Hugging Face repo. A real R4 run is still needed.
- **FULLSTACK's Destroyed IoU of 0.3013 isn't comparable with D4.** It was scored on a 768 centre crop using the best epoch, while D4 scores full images using the last epoch. It does not pass the Phase Gate.
- **BDAChat's segmentation:** an earlier note called it "SAM with prompts"; the abstract only says a *modified SAM*.

---

## 11. Sources

**Our dataset and benchmarks**
- [DisasterM3 (arXiv 2505.21089, NeurIPS 2025)](https://arxiv.org/abs/2505.21089)
- [DisasterM3 on Hugging Face](https://huggingface.co/datasets/Kingdrone-Junjue/DisasterM3)
- [GEOBench-VLM (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Danish_GEOBench-VLM_Benchmarking_Vision-Language_Models_for_Geospatial_Tasks_ICCV_2025_paper.html)
- [BRIGHT (ESSD 2025)](https://essd.copernicus.org/articles/17/6217/2025/)

**Vision-language and pixel-grounded models**
- [TEOChat (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html)
- [GeoPixel (ICML 2025)](https://icml.cc/virtual/2025/poster/44111)
- [BDAChat / Integrating segmentation and VLM for building damage assessment (Adv. Eng. Informatics 2026)](https://www.sciencedirect.com/science/article/abs/pii/S1474034626000121)

**Change detection and damage models**
- [Segment Any Change / AnyChange (NeurIPS 2024)](https://neurips.cc/virtual/2024/poster/96101)
- [ChangeMamba (TGRS 2024)](https://github.com/ChenHongruixuan/ChangeMamba)
- [Change-Agent (TGRS 2024)](https://github.com/Chen-Yang-Liu/Change-Agent)
- [DeepDamageNet (2024 preprint, not an xView2 winner)](https://arxiv.org/pdf/2405.04800)
- [YOLO-E + SAM2 for earthquake-damaged buildings](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12299807/)

**xView2 winners**
- [DIUx-xView repositories](https://github.com/orgs/DIUx-xView/repositories)
- [1st place](https://github.com/DIUx-xView/xView2_first_place)
- [Victor Durnov on GitHub](https://github.com/vdurnov)
- [2nd place](https://github.com/DIUx-xView/xView2_second_place)
- [3rd place (BloodAxe)](https://github.com/BloodAxe/xView2-Solution)
- [3rd place write-up](https://computer-vision-talks.com/2020-01-xview2-solution-writeup/)
- [4th place (Z-Zheng)](https://github.com/Z-Zheng/xview2_4th_solution)
- [5th place (Dual-HRNet)](https://github.com/DIUx-xView/xView2_fifth_place)
- [xFBD paper, which summarises the winners' methods](https://arxiv.org/pdf/2212.13876)

**Generative models and synthetic data**
- [DiffusionSat (ICLR 2024)](https://arxiv.org/abs/2312.03606)
- [ChangeDiff (AAAI 2025)](https://mlanthology.org/aaai/2025/zang2025aaai-changediff/)
- [Changen2 (TPAMI)](https://arxiv.org/abs/2406.17998)
- [Changen2 synthetic data](https://huggingface.co/datasets/EVER-Z/Changen2-S1-15k)
- [Neural Disaster Simulation (RSE 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0034425725003839)

**SAM family**
- [SAM 2 (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf)
- [SAM 2 model sizes](https://docs.ultralytics.com/models/sam-2)
- [SAM 3 paper](https://arxiv.org/pdf/2511.16719)
- [SAM 3 code](https://github.com/facebookresearch/sam3)
- [SAM 3 technical overview](https://datature.io/blog/sam-3-a-technical-deep-dive-into-metas-next-generation-segmentation-model)
- [SAM 3 / 3.1 blog](https://ai.meta.com/blog/segment-anything-model-3/)
- [SAM 3 gated-access issues](https://github.com/facebookresearch/sam3/issues/624)
- [SegEarth-OV3 (preprint)](https://arxiv.org/html/2512.08730v2)
- [OmniOVCD (preprint)](https://arxiv.org/abs/2601.13895)
- [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3)
- [SAM 3 point prompts on geospatial data](https://samgeo.gishub.org/examples/sam3_point_prompts_batch/)

**RAG, agents and edge**
- [ImageRAG (IEEE GRSM)](https://arxiv.org/abs/2411.07688v3)
- [Grace: satellite–ground VLM collaboration (preprint)](https://arxiv.org/abs/2510.24242)
- [GeoDisaster (preprint)](https://arxiv.org/pdf/2606.17246)
- [DisasterBench (preprint)](https://arxiv.org/pdf/2606.06217)
- [HASTE (preprint)](https://arxiv.org/pdf/2607.11838)

**Physics-informed models**
- [Physics-informed GNN for flood forecasting (CACAIE 2025)](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13484)
