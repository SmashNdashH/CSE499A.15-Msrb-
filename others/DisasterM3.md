# DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response

**arXiv:2505.21089v2 [cs.CV] 20 Oct 2025**

**Authors:** Junjue Wang$^{1*}$, Weihao Xuan$^{1,2*}$, Heli Qi$^{2,3}$, Zhihao Liu$^{1}$, Kunyi Liu$^{3}$, Yuhan Wu$^{4}$, Hongruixuan Chen$^{1}$, Jian Song$^{2}$, Junshi Xia$^{2}$, Zhuo Zheng$^{5}$, Naoto Yokoya$^{1,2\dagger}$

$^1$ The University of Tokyo, $^2$ RIKEN AIP, $^3$ Waseda University, $^4$ Stony Brook University, $^5$ Stanford University

---

## Abstract

The recent advent of large vision-language models (VLMs) [16, 42, 6] has achieved substantial milestones in computer vision due to their exceptional ability to reason about visual and linguistic clues and summarize high-level human-readable text. Inspired by the success of the generic domain, remote sensing has also explored the applications of VLMs, i.e., image classification [14], image captioning [13], visual question answering [41], etc. These remote sensing-tailored VLMs show great potential as general-purpose task solvers for multi-task scenarios. Unlike existing research that primarily addresses general geospatial tasks, our work explores the reasoning capabilities of VLMs in extreme disaster scenarios, thereby supporting rescue teams and planning personnel in making informed decisions.

## 1 Introduction

Onset natural and man-made disasters represent one of humanity’s greatest challenges, causing devastating impacts across national borders [46, 8]. These catastrophic events (including earthquakes, tsunamis, floods, explosions, storms, etc) claim tens of thousands of lives globally each year while causing massive infrastructure damage and economic losses [33, 25]. Remote sensing (RS), as an ultra-long-distance Earth observation technology, has been widely used in disaster scenarios, i.e., hurricane damage assessment [29], landslide detection [36], mapping of burn area and ecological impacts [26], etc. Considering the urgency and timeliness of disaster relief, developing AI-based algorithms is necessary.

Large vision-language models (VLMs) have made great achievements in Earth vision. However, complex disaster scenes with diverse disaster types, geographic regions, and satellite sensors have posed new challenges for VLM applications. To fill this gap, we curate a remote sensing vision-language dataset (DisasterM3) for global-scale disaster assessment and response. DisasterM3 includes 26,988 bi-temporal satellite images and 123k instruction pairs across 5 continents, with three characteristics:

1.  **Multi-hazard:** DisasterM3 involves 36 historical disaster events with significant impacts, which are categorized into 10 common natural and man-made disasters.
2.  **Multi-sensor:** Extreme weather during disasters often hinders optical sensor imaging, making it necessary to combine Synthetic Aperture Radar (SAR) imagery for post-disaster scenes.
3.  **Multi-task:** Based on real-world scenarios, DisasterM3 includes 9 disaster-related visual perception and reasoning tasks, harnessing the full potential of VLM’s reasoning ability with progressing from disaster-bearing body recognition to structural damage assessment and object relational reasoning, culminating in the generation of long-form disaster reports.

We extensively evaluated 14 generic and remote sensing VLMs on our benchmark, revealing that state-of-the-art models struggle with the disaster tasks, largely due to the lack of a disaster-specific corpus, cross-sensor gap, and damage object counting insensitivity. Focusing on these issues, we fine-tune four VLMs using our dataset and achieve stable improvements (up to 10.4%↑QA, 2.1↑Report, 40.8%↑Referring Seg.) with robust cross-sensor and cross-disaster generalization capabilities. Project: https://github.com/Junjue-Wang/DisasterM3.

## 2 Related Work

### General Vision-language Model

Assisted by the strong reasoning abilities of large language models, VLMs have transformed the visual perception domain by enabling the interpretation and reasoning about images through natural language interfaces. Several leading VLMs, including Flamingo [1], MiniGPT-4 [51], LLaVA [17], LLaVA-OneVision [16], InstructBLIP [6], and Qwen2-VL [42], have achieved remarkable results on vision-language tasks. However, these models are limited to generating only textual outputs that describe the image holistically. This restricts their applicability in damage assessment tasks that require the pixel-level detailed understanding. Several approaches have emerged to extend VLMs with fine-grained visual understanding. Ferret [47], Kosmos-2 [27], and VisionLLM [43] incorporate grounding functionalities through bounding box coordinate regression. Besides, LISA [15], PixelLM [31], GLaMM [30], and PerceptionGPT [28], integrate mask decoders to generate object masks from specialized tokens. For richer representation, PSALM [50] and HyperSeg [44] leverage queries in Mask2Former for unified segmentation. Despite their capabilities, generic VLMs exhibit substantial limitations in disaster scenarios due to insufficient domain-specific knowledge, restricting their operational utility in emergency response applications.

### Remote Sensing Vision-language Dataset

Following the substantial progress of general VLMs, the RS field has likewise undergone accelerated development, accompanied by the emergence of numerous specialized vision-language datasets. Focusing on holistic analysis, EarthVQA [41] and RSIEval [11] datasets provide manual instructions for visual question answering (VQA) and image captioning tasks. Leveraging GPT-4, VRSBench [18] introduced visual grounding tasks to evaluate the object reasoning abilities and XLRSBench [39] focuses on ultra-high-resolution image understanding. GeoChatSet [14] and TeoChatlas [13] collect the existing classification and detection datasets for secondary development, formulating the unified instruction-following datasets. Although TeoChatlas involves some disaster scenes, the instructions focus on common recognition tasks. FloodNet [29] is a VQA disaster dataset that assesses the buildings and roads affected by Hurricane Harvey. Limited by its single disaster and simple tasks, it is difficult to fully unleash the potential of VLMs. Overall, RS visual-language datasets for general geospatial tasks have reached a considerable level of maturity, yet there persists a notable deficiency in datasets addressing specialized geoscience challenges. For this case, we design the DisasterM3 dataset that is tailored for global disaster assessment and response with multi-sensor images, bi-temporal inputs, refined damage masks, and diverse visual understanding tasks in the context of disaster.

**Table 1: Comparison of DisasterM3 with existing remote sensing vision-language datasets.**

| Dataset | Propose | #Optical | #SAR | #MT pairs* | #Text | Recognition | Counting | Localization | Reasoning | Caption |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RSICD [23] | General | 10,921 | - | - | 54,605 | ✓ | ✓ | - | ✗ | ✗ |
| RSICap [11] | General | 2,585 | - | - | 2,585 | ✓ | ✓ | - | ✗ | ✗ |
| DIOR-RSVG [49] | General | 17,402 | - | - | 38,320 | ✗ | ✗ | Box | ✗ | ✗ |
| RRSIS-D [20] | General | 17,402 | - | - | 17,402 | ✗ | ✗ | Pixel | ✗ | ✗ |
| RSVQA-HR [22] | General | 10,659 | - | - | 1,066,316 | ✓ | ✓ | - | ✗ | ✗ |
| EarthVQA [41] | General | 6,000 | - | - | 208,593 | ✓ | ✓ | - | ✗ | ✗ |
| RSIEval [11] | General | 100 | - | - | 933 | ✓ | ✓ | - | ✗ | ✗ |
| VRSBench [18] | General | 29,614 | - | - | 205,317 | ✓ | ✓ | Box | ✗ | ✗ |
| XLRSBench [39] | General | 1,400 | - | - | 45,942 | ✓ | ✓ | Box | ✓ | ✗ |
| GeoChatSet [14] | General | 106,747 | - | - | 308,861 | ✓ | ✓ | Box | ✓ | ✗ |
| TeoChatlas [13] | General | 351,957 | - | 245,210 | 554,071 | ✓ | ✓ | Box | ✗ | ✓ |
| FloodNet [29] | Disaster | 2,348 | - | - | 7,345 | ✓ | ✓ | - | ✗ | ✗ |
| **DisasterM3 (Ours)** | **Disaster** | **22,214** | **4,774** | **15,881** | **123,010** | **✓** | **✓** | **Pixel** | **✓** | **✓** |

*MT pairs (multi-temporal pairs) denote the number of pre/post-disaster image pairs.*

## 3 DisasterM3 Dataset

We collect 36 historical natural and man-made significant disasters to construct the DisasterM3 dataset. There are 26 events from the xBD [9] and BRIGHT [5] dataset, we extend 10 new events using Maxar’s Open Data program [24]. Considering these optical sensors (WorldView series) have similar spatial resolutions, all pre- and post-disaster images were pre-processed into 0.8 m. We collect the post-disaster Synthetic Aperture Radar (SAR) images from Capella Space [4] and Umbra [37]. Considering the amplitude data in the VV or HH bands, SAR images were terrain-corrected, stretched into [0, 255], and finally resampled to match the optical resolution. We performed the georeferencing to ensure that the pre- and post-disaster image pairs are strictly aligned spatially.

Following the United Nations Satellite Centre (UNOSAT) Emergency Mapping Products [38], and the Federal Emergency Management Agency (FEMA) [7], we design 9 essential tasks required for disaster assessment and response, evaluating the VLM performances from different aspects.

### 3.1 Perception and Reasoning Tasks in the Context of Disaster

#### Disaster Recognition
The disaster recognition tasks provide a brief description of disaster scenes, i.e., disaster types, land-use types and key disaster-bearing body. The disaster type follows the official definition and we chose 13 common land-use types from the AID dataset [45] for annotation. The land-use answers include: airport, bridge, river, forest, low vegetation, pond, parking, port, viaduct, residential area, industrial area, commercial area, and sea. Disaster-bearing bodies are the key resources that are damaged by disasters [8], and we focus on 12 types, i.e., building, stadium, open-space ground, bridge, dam, road, port facility, storage tank, farmland, forest, coastline, and mining area.

#### Damage Assessment
The damage assessment provides a quantitative analysis of disaster-bearing body. We chose the road and building, two important man-made structures for damage assessment. We annotate instance-level building damage masks using ‘intact’, ‘damaged’, and ‘destroyed’ types following FEMA guidelines. As a critical transportation hub, road accessibility plays a vital role in emergency response and recovery efforts. We classify the damaged roads into three types, i.e., ‘intact’, ‘flooded (blocked by water)’, and ‘debris covered (blocked by debris)’. Based on these damage masks, the building counting and road area estimation instructions were automatically generated.

#### Disaster Referring Segmentation
Each disaster includes different forming factors and prone environments. In addition to disaster-bearing-body mapping, we identify the key visual objects and perform risk analysis using referring segmentation. As shown in Fig. 3, the first example shows an earthquake scene. In addition to referring segmentation for disaster-bearing body, we also design the task for finding the optimal rescue places. Similarly, for the volcano eruption scene, we set the instruction tasks to individually map damaged buildings and roads, as well as the lava. Considering the situation, the intact buildings near the lava are also required for segmentation. By polygon distance analysis using the ArcGIS toolbox, the intact buildings within a 100-meter proximity to lava are segmented, providing early warning information.

#### Damaged Object Relational Reasoning
To capture the spatial relationships between multiple damaged objects, relational reasoning tasks are designed. In the wildfire scene, the spatial relationships between unaffected buildings and refuge squares, as well as between burnt grassland and unaffected trees, reveal crucial patterns in disaster response and spread prevention. The war conflict scene depicts the damaged industrial area, where the relationships between key facilities, factories, and transportation hubs are clarified.

#### Disaster Comprehensive Report
To go beyond traditional perception tasks, the comprehensive reports are designed for the holistic analysis of disaster situations. The earthquake caption describes the collapsed buildings and blocked roads, causing severe traffic congestion. Immediate response advice prioritizes the deployment of temporary shelters within the stadium for displaced survivors, a recommendation visibly implemented in the post-disaster image. Long-term recovery focuses on earthquake-resistant strategies in rebuilding and disaster protocols to mitigate seismic risks. Comprehensive disaster reports equip rescuers with enhanced situational awareness and evidence-based decision support.

### 3.2 Dataset Construction Pipeline

Following the common vision-language data pipeline [21, 48], we divided the whole dataset into Instruct (17,190 Optical images, 3,798 SAR images, and 92,968 instruction pairs) and Bench sets (5,024 Optical images, 976 SAR images, and 30,042 instruction pairs). We describe the detailed annotation process in Fig. 6. As for recognition tasks, GPT-4o was employed to generate diverse prompt variations with similar semantic intent. Disaster domain experts subsequently annotate correct answers for these prompts. To formulate the multiple-choice Bench set, correct answers were combined with other options. Regarding counting tasks, we counted semantic polygons using annotated road and building damage masks, generating correct answers. The similar options are generated with controlled deviations (±20% and ±40%) to maintain plausibility.

As for relational reasoning, the experts annotate bounding boxes and describe the concrete relationship. We use GPT-4o to analyze the image by listing other significant relationships, generating alternatives. As for disaster reports, by referring to bi-temporal images and all basic task information, multiple experts draft the disaster caption and restoration advice following goals of UNITAR and FEMA projects. GPT-4o then polished the reports and corrected grammar errors. Finally, the multi-round verification was performed for controlling quality (Appendix §A).

## 4 Benchmark Experiments

### Implementation Setting
As the DisasterM3 dataset features multi-sensor and multi-task, we comprehensively benchmark VLMs under four settings: Optical-Optical and Optical-SAR QA tasks, as well as Optical-Optical and Optical-SAR referring segmentation tasks. As for QA tasks, LLaVA-OneVision [16], InternVL3 [53], Kimi-VL [35], and Qwen2.5-VL [3]. In addition, we also tested commercial models such as GPT-4o [12] and Claude-3 [2] for comparison with the open-source models. As for remote sensing VLMs GeoChat [14], TeoChat [13], EarthDial [34] are chosen for evaluation. As for referring segmentation models, generic VLMs models such as LISA [15], PSALM [50], and HyperSeg [44], alongside the remote sensing model GeoPixel [32] were benchmarked. We fine-tuned Qwen2.5-VL-7B, InternVL3-8B, LISA and PSALM on our Instruct set.

### Evaluation Metrics
Following common settings [16, 34], we adopted accuracy (%) for the multiple-choice tasks, i.e., disaster scene recognition (DSR), disaster type recognition (DTR), bearing body recognition (BBR), damaged building counting (BDC), damaged road estimation (DRE), object relational reasoning (ORR). The open-ended tasks are scored using GPT-4.1 at a scale of 5 points. Disaster caption is measured from damage assessment precision (DAP), damage detail recall (DDR), and factual correctness (FC). Restoration advice is measured from recovery necessity (RN), strategic completeness (SC), and action priority precision (APP). The average accuracy (AVG) denotes the overall performance. As for referring segmentation, we chose cIoU and mIoU following previous work [15, 50].

### 4.1 Comparative Results

**Domain gap for disaster scenarios.** Tab. 2 presents performance evaluations on optical-optical settings for QA tasks. As a traditional VLM, LLaVA-1.5 exhibited significant limitations when processing disaster scenes due to the domain gap. By leveraging extensive multi-modal pre-training datasets and implementing the AnyRes architecture, LLaVA-OV demonstrates enhancements in both accuracy and multi-image processing capabilities. As efficient Mixture-of-Experts (MoE) VLMs, Kimi-VL-A3B-Think exceeds Kimi-VL-A3B-Instruct in mathematical counting tasks (BDC, DRE). However, the non-negligible domain gap limits their application on complex tasks, particularly degrading performance to near-random levels on the ORR task.

**Larger VLMs achieve higher performances.** By scaling up LLMs, InternVL3 and Qwen2.5-VL series demonstrate consistent trends that larger LLMs achieve superior performances, confirming established scaling laws observed in general-domain applications. The commercial models, i.e., GPT-4o and GPT-4.1, showcase competitive performances across all tasks due to their massive corpus.

**Remote sensing VLMs still struggle with disaster tasks.** Despite being specifically trained on aerial and satellite imagery, existing remote sensing VLMs exhibit feature representations that inadequately transfer to the unique characteristics of disaster scenarios.

**Fine-tuned models improve comprehensively.** By fine-tuning on DisasterM3 Instruct set, the performances of Qwen2.5-VL and InternVL3 have been significantly improved, narrowing the domain gap. Disaster-specific terminology integration during training significantly enhances report generation quality.

**Table 2: Benchmarking results of VLMs on DisasterM3 Bench set with optical-optical setting.**

| Method | Accuracy (%) AVG | DSR | DTR | BBR | BDC | DRE | ORR | Disaster Caption AVG | DAP | DDR | FC | Restoration Advice AVG | RN | APP | SC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Guess | - | - | 20 | - | 20 | 20 | 20 | - | - | - | - | - | - | - | - |
| **Open-source models** | | | | | | | | | | | | | | | |
| LLaVA-1.5-7B [19] | 12.1 | 4.2 | - | - | - | - | 20.0 | - | - | - | - | - | - | - | - |
| LLaVA-OV-7B [17] | 24.5 | 16.3 | 53.5 | 3.7 | 26.4 | 24.2 | 22.7 | 1.66 | 1.50 | 1.53 | 1.93 | 2.30 | 3.01 | 2.08 | 1.81 |
| Kimi-VL-A3B-Instruct [35] | 25.6 | 28.9 | 66.3 | 4.0 | 20.4 | 15.0 | 18.9 | 1.69 | 1.53 | 1.72 | 1.81 | 2.67 | 3.57 | 2.40 | 2.05 |
| Kimi-VL-A3B-Think [35] | 26.7 | 27.0 | 51.6 | 7.4 | 24.4 | 25.4 | 24.4 | 1.61 | 1.39 | 1.68 | 1.75 | 2.61 | 3.35 | 2.34 | 2.15 |
| InternVL3-8B [53] | 31.3 | 39.6 | 53.5 | 4.0 | 30.3 | 24.1 | 36.2 | 1.96 | 1.88 | 1.92 | 2.09 | 2.75 | 3.52 | 2.53 | 2.21 |
| InternVL3-14B [53] | 35.7 | 42.5 | 62.0 | 4.9 | 27.4 | 23.6 | 54.1 | 2.08 | 2.01 | 2.01 | 2.22 | 2.86 | 3.67 | 2.62 | 2.29 |
| InternVL3-78B [53] | 39.3 | 43.5 | 72.5 | 5.3 | 29.4 | 28.7 | 56.1 | 2.79 | 2.74 | 2.75 | 2.89 | 2.90 | 3.64 | 2.64 | 2.43 |
| Qwen2.5-VL-3B [3] | 26.2 | 30.8 | 56.1 | 5.7 | 29.9 | 21.2 | 13.8 | 1.00 | 0.83 | 1.05 | 1.12 | 2.15 | 2.98 | 1.77 | 1.71 |
| Qwen2.5-VL-7B [3] | 31.2 | 28.3 | 66.6 | 4.7 | 34.2 | 29.3 | 23.9 | 1.75 | 1.69 | 1.71 | 1.85 | 1.95 | 2.53 | 1.83 | 1.49 |
| Qwen2.5-VL-32B [3] | 35.3 | 36.7 | 54.7 | 11.6 | 33.2 | 30.9 | 44.8 | 1.55 | 1.42 | 1.52 | 1.72 | 2.96 | 3.63 | 2.71 | 2.55 |
| Qwen2.5-VL-72B [3] | 40.5 | 47.0 | 74.8 | 6.8 | 34.8 | 28.9 | 50.8 | 2.01 | 1.99 | 2.00 | 2.05 | 2.92 | 3.79 | 2.70 | 2.27 |
| GeoChat-7B [14] | 10.7 | 6.1 | - | - | - | - | 15.3 | - | - | - | - | - | - | - | - |
| TeoChat-7B [13] | 23.0 | 6.9 | 64.9 | 2.0 | 22.5 | 23.3 | 18.2 | 1.77 | 1.61 | 1.74 | 1.96 | 1.95 | 2.59 | 1.77 | 1.49 |
| EarthDial-4B [34] | 22.9 | 10.6 | 58.1 | 3.2 | 30.2 | 20.8 | 14.5 | 1.53 | 1.22 | 1.64 | 1.73 | 2.42 | 3.21 | 2.08 | 1.98 |
| **Commercial models** | | | | | | | | | | | | | | | |
| GPT-4o [12] | 39.3 | 49.4 | 80.5 | 10.6 | 24.2 | 21.4 | 49.8 | 2.27 | 2.25 | 2.28 | 2.28 | 3.19 | 3.92 | 2.95 | 2.69 |
| GPT-4.1 [12] | 42.3 | 52.4 | 79.6 | 7.2 | 25.5 | 25.0 | 64.0 | 2.57 | 2.60 | 2.58 | 2.54 | 3.14 | 3.94 | 2.93 | 2.56 |
| **Fine-tuned models** | | | | | | | | | | | | | | | |
| Qwen2.5-VL-7B [3] | 40.4 | 37.7 | 83.6 | 21.5 | 34.3 | 29.4 | 36.2 | 3.90 | 3.76 | 3.53 | 4.41 | 3.11 | 3.73 | 2.88 | 2.73 |
| ∆ | ↑9.2 | ↑9.4 | ↑17.0 | ↑16.8 | ↑0.1 | ↑0.1 | ↑12.3 | ↑2.15 | ↑2.07 | ↑1.82 | ↑2.56 | ↑1.26 | ↑1.20 | ↑1.83 | ↑1.24 |
| InternVL3-8B [53] | 41.7 | 42.6 | 79.3 | 23.9 | 29.1 | 24.9 | 50.6 | 3.83 | 3.69 | 3.49 | 4.32 | 3.31 | 3.92 | 3.10 | 2.90 |
| ∆ | ↑10.4 | ↑3.0 | ↑25.8 | ↑19.9 | ↓-1.2 | ↑0.8 | ↑14.4 | ↑1.87 | ↑1.81 | ↑1.57 | ↑2.23 | ↑0.56 | ↑0.40 | ↑0.57 | ↑0.69 |

**Underrepresentation for SAR images.** Disasters are often accompanied by extreme weather, with clouds and rain blocking optical sensors. In this case, the active imaging method SAR can penetrate clouds and fog to obtain accurate surface information. Fig. 7 shows the VLMs’ performances evaluated on paired optical-SAR images. Due to the reduced semantics compared to optical imagery and underrepresentation in generic VLM, the performance using post-SAR images yielded substantially diminished performance across all evaluation tasks.

**Mask token matters in disaster referring segmentation.** Tab. 3 shows the compared results of referring segmentation models with multi-sensor settings. After fine-tuning, LISA and PSALM have achieved significant gains in two settings with the injection of disaster reasoning knowledge during the training. It is notable that PSALM exceeds LISA with much smaller parameters. We attribute this to a more robust mask token representation in PSALM.

**Table 3: Benchmarking results of referring segmentation VLMs on DisasterM3 Bench set.**

| Model | Optical-Optical (%) | | | | | Optical-SAR (%) | | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | mIoU | cIoU | Road | Building | Other | mIoU | cIoU | Road | Building |
| **Open-source models** | | | | | | | | | |
| PSALM-1.3B [50] | 9.7 | 6.3 | 2.6 | 10.2 | 16.3 | 8.1 | 8.8 | 5.1 | 11.1 |
| HyperSeg-3B [44] | 16.6 | 14.5 | 7.5 | 17.0 | 25.4 | 8.8 | 10.3 | 4.5 | 13.1 |
| LISA-7B [15] | 27.5 | 22.1 | 11.9 | 25.0 | 45.6 | 10.9 | 12.7 | 6.0 | 15.7 |
| GeoPixel-7B [32] | 14.3 | 14.2 | 8.5 | 18.1 | 16.2 | 4.0 | 5.1 | 1.8 | 6.2 |
| **Fine-tuned models** | | | | | | | | | |
| LISA-7B [15] | 44.8 | 43.7 | 27.6 | 41.2 | 65.5 | 28.2 | 29.6 | 21.5 | 34.9 |
| ∆ | ↑17.3 | ↑21.6 | ↑15.7 | ↑16.2 | ↑19.9 | ↑17.3 | ↑16.9 | ↑15.5 | ↑19.2 |
| PSALM-1.3B [50] | 50.5 | 44.5 | 30.5 | 49.1 | 71.9 | 31.8 | 35.2 | 24.3 | 39.3 |
| ∆ | ↑40.8 | ↑38.2 | ↑27.9 | ↑38.9 | ↑55.6 | ↑23.7 | ↑26.4 | ↑19.2 | ↑28.2 |

### 4.2 Detailed Analysis

**Performance variation across disaster categories.** VLM performance exhibits variation across disaster types due to differing disaster causal factors and prone environments. As shown in Tab. 4, all methods demonstrate higher performance on landslide events while achieving notably lower metrics on earthquake, tornado, and explosion scenarios.

**Table 4: Performance variation across disaster categories.**

| Method | AVG | Landslide | Earthquake | Tornado | Conflict | Fire | Explosion | Tsunami | Hurricane | Volcano | Flooding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LLaVA-OV [17] | 21.2 | 23.6 | 17.1 | 19.2 | 22.8 | 25.4 | 18.1 | 19.5 | 23.2 | 23.5 | 19.9 |
| Kimi-VL-A3B-Instruct [35] | 19.9 | 22.2 | 17.5 | 20.1 | 16.3 | 26.3 | 13.2 | 19.2 | 21.9 | 23.5 | 18.4 |
| Kimi-VL-A3B-Think [35] | 22.0 | 26.4 | 19.6 | 21.0 | 17.4 | 26.9 | 16.9 | 22.6 | 21.8 | 25.0 | 22.3 |
| InternVL3-8B [53] | 27.5 | 41.7 | 22.2 | 24.4 | 21.7 | 33.0 | 20.9 | 28.0 | 27.3 | 28.8 | 27.0 |
| InternVL3-14B [53] | 30.0 | 48.6 | 22.7 | 26.6 | 27.2 | 33.7 | 21.1 | 27.3 | 29.9 | 31.3 | 31.5 |
| InternVL3-78B [53] | 31.8 | 48.6 | 26.3 | 27.9 | 25.0 | 37.1 | 25.6 | 32.0 | 31.7 | 32.9 | 30.9 |
| Qwen2.5-VL-3B [3] | 24.5 | 26.4 | 19.5 | 21.9 | 27.5 | 32.1 | 17.9 | 24.2 | 24.2 | 29.6 | 21.8 |
| Qwen2.5-VL-7B [3] | 25.6 | 34.3 | 21.0 | 24.9 | 17.3 | 33.7 | 16.7 | 28.3 | 27.8 | 25.6 | 25.9 |
| Qwen2.5-VL-32B [3] | 31.0 | 50.0 | 26.4 | 27.1 | 26.6 | 35.7 | 23.9 | 32.8 | 29.4 | 30.8 | 27.7 |
| Qwen2.5-VL-72B [3] | 31.8 | 47.2 | 25.0 | 31.1 | 19.0 | 39.0 | 24.0 | 33.4 | 34.0 | 33.9 | 31.2 |
| GPT-4o [12] | 30.7 | 52.8 | 24.8 | 25.7 | 19.6 | 33.7 | 25.6 | 28.0 | 29.7 | 35.1 | 32.0 |
| GPT-4.1 [12] | 32.4 | 51.4 | 26.9 | 26.7 | 21.7 | 35.2 | 27.7 | 28.5 | 33.4 | 35.8 | 36.5 |
| **Fine-tuned models** | | | | | | | | | | | |
| Qwen2.5-VL-7B [3] | 32.9 | 41.7 | 26.5 | 30.7 | 27.7 | 40.3 | 22.3 | 33.0 | 34.0 | 41.8 | 31.1 |
| ∆ | ↑7.3 | ↑7.4 | ↑5.5 | ↑5.8 | ↑10.4 | ↑6.6 | ↑5.6 | ↑4.7 | ↑6.2 | ↑16.2 | ↑5.2 |
| InternVL3-8B [53] | 34.7 | 56.9 | 26.0 | 31.1 | 26.1 | 40.3 | 27.4 | 33.1 | 34.9 | 39.4 | 32.2 |
| ∆ | ↑7.2 | ↑15.2 | ↑3.8 | ↑6.7 | ↑4.4 | ↑7.3 | ↑6.5 | ↑5.1 | ↑7.6 | ↑10.6 | ↑5.2 |

**Performance biases in VLMs for damage counting.** Remote sensing imagery typically encompasses numerous objects exhibiting diverse scales and morphologies, with counting challenges becoming particularly acute when conducting fine-grained damage assessment. Fig. 8 illustrates building damage assessment accuracy as a function of building density within analyzed scenes.

**Effects of different prompts.** As shown in Fig. 9, we evaluated the robustness of VLMs with five different prompts, where quartiles, ranges of accuracies are plotted. Due to limited LLM capabilities, LLaVA-OV, TeoChat, EarthDial, and Kimi models exhibit higher sensitivity to prompt variations. Besides, InternVL3 and Qwen2.5-VL series models show similar patterns wherein larger LLMs display enhanced stability. Following enrichment with the disaster-specific corpus from the DisasterM3 dataset, the fine-tuned Qwen2.5-VL-7B and InternVL3-8B model demonstrate good stability to prompt variations.

## 5 Limitations and Future Directions

While DisasterM3 represents a significant step forward in disaster-oriented vision-language research, we acknowledge several limitations that open avenues for future work.

1.  **Multi-resolution generalization:** Our standardization to 0.8 m resolution ensures controlled experimentation but limits evaluation of model robustness to diverse spatial resolutions encountered in operational settings. Future work should incorporate multi-resolution imagery from platforms like Sentinel-2 (10m) and Landsat (30m), leveraging our existing annotations through geo-registration.
2.  **Enhanced sensor diversity:** Although we include both optical and SAR imagery, our SAR data is limited to single polarization. Integrating multi-polarization data (e.g., Sentinel-1’s VV+VH) would provide richer scattering information about debris orientation and surface characteristics, enabling more comprehensive damage assessment.
3.  **Cross-sensor performance gap:** The significant performance degradation on SAR imagery highlights the need for advanced multi-modal pretraining and cross-modal alignment strategies to better bridge the optical-SAR domain gap.
4.  **Counting task optimization:** To address overfitting in damage object counting, promising directions include object-sensitive encoders (e.g., DINOv2), numerical difference loss, synthetic data generation via diffusion models for high-density scenarios, and knowledge distillation strategies.
5.  **Living benchmark commitment:** We will maintain DisasterM3 as an evolving resource by regularly incorporating new disaster events from the Maxar Open Data Program, ensuring continued relevance and growth in geographic and temporal coverage for the disaster response community.

## 6 Conclusion

Inspired by the rapid development of generic VLMs, the remote sensing vision-language datasets and methods have also been gradually explored. To promote interactive AI disaster response, we propose DisasterM3, a multi-hazard, multi-sensor, and multi-task remote sensing dataset for vision-language understanding. DisasterM3 includes 26,988 bi-temporal images and 123k instruction pairs, 36 disaster events across 5 continents. The comprehensive benchmarking of 14 advanced VLMs evaluate both their capabilities and inherent limitations in disaster contexts. Additionally, through fine-tuning four VLMs with the disaster-specific corpus from DisasterM3, we demonstrate substantial performance enhancements across all evaluation tasks. We believe the proposed dataset and baselines will help bridge the gap in VLM-based disaster applications within Earth vision.

## Acknowledgments

This work was supported in part by the Council for Science, Technology and Innovation (CSTI) and the Cross-ministerial Strategic Innovation Promotion Program (SIP) "Development of a Resilient Smart Network System against Natural Disasters" (funding agency: NIED), KAKENHI (25K03145) as well as the NVIDIA Academic Grant Program. This work used computational resources Miyabi supercomputer provided by The University of Tokyo through Joint Usage/Research Center for Interdisciplinary Large-scale Information Infrastructures and High Performance Computing Infrastructure in Japan (Project ID: jh250017). Weihao Xuan is supported by RIKEN Junior Research Associate (JRA) Program. We also thank Ritwik Gupta for sharing the valuable xBD dataset and for his expertise in disaster response guidance.
