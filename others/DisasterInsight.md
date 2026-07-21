# DisasterInsight: A Multimodal Benchmark for Function-Aware and Grounded Disaster Assessment

**arXiv:2601.18493v1 [cs.CV] 26 Jan 2026**

**Authors:** Sara Tehrani$^*$, Yonghao Xu$^{\dagger *}$, Leif Haglund$^{\ddagger}$, Amanda Berg$^{\ddagger}$, Michael Felsberg$^*$

$^*$ Computer Vision Laboratory, Linköping University, Sweden.
$^\dagger$ Corresponding author.
$^\ddagger$ Computer Vision Laboratory, Linköping University, Sweden; and Vantor, Linköping, Sweden.

---

## Abstract

Timely interpretation of satellite imagery is critical for disaster response, yet existing vision–language benchmarks for remote sensing largely focus on coarse labels and image-level recognition, overlooking the functional understanding and instruction robustness required in real humanitarian workflows. We introduce **DisasterInsight**, a multimodal benchmark designed to evaluate vision–language models (VLMs) on realistic disaster analysis tasks. DisasterInsight restructures the xBD dataset into approximately 112K building-centered instances and supports instruction-diverse evaluation across multiple tasks, including building-function classification, damage-level and disaster-type classification, counting, and structured report generation aligned with humanitarian assessment guidelines. To establish domain-adapted baselines, we propose **DI-Chat**, obtained by fine-tuning existing VLM backbones on disaster-specific instruction data using parameter-efficient Low-Rank Adaptation (LoRA).

Extensive experiments on state-of-the-art generic and remote-sensing VLMs reveal substantial performance gaps across tasks, particularly in damage understanding and structured report generation. DI-Chat achieves significant improvements on damage-level and disaster-type classification as well as report generation quality, while building-function classification remains challenging for all evaluated models. DisasterInsight provides a unified benchmark for studying grounded multimodal reasoning in disaster imagery.

**Keywords:** Vision-Language Models; Remote Sensing; Disaster Response; Multimodal Benchmark.

## 1 Introduction

Rapid and accurate interpretation of satellite imagery is a cornerstone of effective disaster response. Following large-scale hazards such as earthquakes, floods, or hurricanes, emergency coordination agencies require fine-grained, building-level assessments to support triage, resource allocation, and risk mitigation decisions [1]. Operational disaster-response guidelines from the International Search and Rescue Advisory Group (INSARAG) and the Federal Emergency Management Agency (FEMA) further emphasize that, beyond estimating physical damage, responders must assess the function and use of affected structures, such as whether a damaged building serves as a hospital, school, industrial facility, or residential shelter, as these directly influence priorities including medical surge capacity, shelter availability, search-and-rescue planning, and infrastructure restoration.

Despite recent advances in vision–language models (VLMs), existing remote sensing benchmarks do not adequately capture these operational requirements. Most datasets emphasize coarse damage labels or scene-level analysis, treating buildings as an undifferentiated class and overlooking functional semantics and per-building reasoning [2, 3, 4]. Recent multimodal benchmarks incorporating natural language outputs have explored captioning and question answering over disaster imagery, but typically rely on unstructured, scene-level narratives that are not explicitly grounded to individual buildings or aligned with humanitarian reporting practices [5, 6, 7]. To address these limitations, we introduce DisasterInsight, a multimodal benchmark designed to reflect real-world disaster assessment workflows.

Building upon the xBD dataset [4], DisasterInsight restructures imagery into 112,507 building-centered instances and defines five complementary tasks: (i) building function classification, (ii) damage level classification, (iii) disaster type classification, (iv) building counting, and (v) structured report generation following established humanitarian assessment guidelines. Unlike prior benchmarks focused on scene-level recognition or generic captioning, DisasterInsight explicitly evaluates spatial grounding via bounding boxes, temporal reasoning using pre- and post-event imagery, functional understanding through OpenStreetMap-derived labels [8], and instruction robustness through diverse prompt formulations. To establish a domain-adapted baseline, we propose DI-Chat, a vision–language model obtained by fine-tuning TeoChat [5] using parameter-efficient Low-Rank Adaptation (LoRA), following a Video-LLaVA-style [9] multimodal training framework.

Experiments across generic and remote-sensing VLMs reveal substantial performance gaps on DisasterInsight, particularly for damage understanding and structured report generation, while building-function classification remains challenging for all models. These results underscore the importance of disaster-specific instruction tuning and grounded evaluation. Our work makes three primary contributions toward building-centered, function-aware disaster assessment:

*   **DisasterInsight** is introduced as a large-scale, building-centered vision–language benchmark for disaster analysis, comprising instance-level building function annotations derived from OpenStreetMap and an instruction-diverse evaluation protocol with multiple prompt formulations per task, enabling robust assessment of function-aware and instruction-following capabilities under realistic disaster-response scenarios.
*   A **structured report generation task** is defined to produce building-grounded disaster narratives aligned with humanitarian assessment practices, moving beyond generic image captioning.
*   **DI-Chat** is proposed as a domain-adapted instruction-finetuning framework on DisasterInsight and is instantiated on multiple vision–language backbones, including TeoChat and Qwen2.5-VL, to establish strong disaster-aware baselines and to systematically study the impact of domain-specific instruction tuning across disaster-analysis tasks.

## 2 Related Work

### 2.1 Disaster Assessment Datasets
Remote sensing (RS) datasets support automated disaster assessment through damage mapping and change analysis. xBD [4], BRIGHT [2], and FloodNet [3] provide pre- and post-event imagery with damage annotations across multiple hazards, but largely treat buildings as a single class and emphasize scene-level labels. As a result, they do not support function-aware, instance-level evaluation or structured reporting aligned with humanitarian assessment practices.

### 2.2 Vision–Language Models in Remote Sensing
Recent work extends vision–language models (VLMs) to remote sensing by adapting LLaVA-style architectures to satellite and aerial imagery, including GeoChat [7], TeoChat [5], and EarthDial [10]. Generic VLMs such as LLaVA-OneVision [11] and Qwen-VL [12] have also been evaluated via zero-shot prompting, but typically lack geographic priors and struggle with fine-grained, per-building semantics. Existing RS VLM benchmarks remain predominantly scene-level and do not explicitly require building-grounded reasoning or structured operational outputs.

### 2.3 LLM-Based Disaster Reporting
Large language models (LLMs) have been applied to crisis summarization, event reconstruction, and social-media analysis [13, 14, 15], but these approaches rely primarily on text-only inputs and lack visual grounding. DisasterM3 [6] introduces multimodal summaries, yet its outputs remain scene-level and are not explicitly grounded to individual buildings or constrained by operational reporting guidelines. In contrast, the report-generation task considered in this work requires building-level grounding, integration of function and damage metadata, bi-temporal reasoning, and adherence to professional reporting constraints.

### 2.4 Comparison with Existing Disaster and VLM Datasets
Table 1 positions DisasterInsight relative to representative remote sensing and disaster-focused datasets. RS datasets such as RSICD [16], BRIGHT [2], FloodNet [3], and xBD [4] lack building-function annotations and structured reporting, while disaster–LLM datasets (CrisisMMD [17], CrisisFACTS [18], DisasterQA [19]) provide rich text but no satellite imagery or visual grounding. DisasterM3 [6] is the closest multimodal benchmark, yet its summaries are scene-level and not building-grounded.

**Table 1: Comparison of DisasterInsight (DI) with representative remote sensing and disaster-focused multimodal datasets.**

| Dataset | Domain | Pre/Post | Instance | Func. | Damage | Report | Count | Loc. | Mod. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RSICD [16] | RS | ✗ | ✗ | ✗ | ✗ | Cap. | ✗ | Scene | RGB |
| BRIGHT [2] | RS | ✓ | Poly. | ✗ | ✓ | ✗ | ✗ | Poly. | RGB |
| FloodNet [3] | RS+VL | ✓ | Box | ✗ | ✓ | VQA | ✓ | Box | RGB |
| xBD [4] | RS | ✓ | Poly. | ✗ | ✓ | ✗ | ✗ | Poly. | RGB |
| CrisisMMD [17] | LLM+Social | ✗ | Tweet | ✗ | ✓ | Sum. | ✗ | ✗ | Txt+Img |
| CrisisFACTS [18] | LLM | ✗ | Doc | ✗ | ✗ | Sum. | ✗ | ✗ | Txt |
| DisasterQA [19] | LLM | ✗ | N/A | ✗ | ✗ | QA | ✗ | ✗ | Txt |
| DisasterM3 [6] | RS+VL | ✓ | Pixel | ✗ | ✓ | Scene | ✓ | Pixel | RGB+SAR |
| **DI (Ours)** | **RS+VL+LLM** | **✓** | **Box** | **✓** | **✓** | **Struct.** | **✓** | **Box** | **RGB** |

In contrast, DisasterInsight integrates instance-level building-function labels, multi-hazard bi-temporal imagery, structured report generation aligned with humanitarian practices, and building-level tasks spanning classification, counting, and narrative generation, forming a unified evaluation setting for realistic building-level disaster assessment.

## 3 The DisasterInsight Benchmark

Humanitarian disaster response workflows operate at the level of individual structures, where building function, damage severity, and spatial context jointly inform prioritization and resource allocation. However, evaluating vision–language models under such building-centered and function-aware settings remains challenging due to the lack of benchmarks that reflect operational assessment procedures. In practice, agencies such as FEMA and INSARAG conduct post-disaster assessment by integrating information about building function, damage level, hazard context, and situational risk to support response planning and reporting. Existing benchmarks do not fully support this level of building-centered granularity, nor do they unify classification, counting, and structured report generation within a single multimodal evaluation framework.

As illustrated in Fig. 1, DisasterInsight is designed to reflect the analytical steps performed during post-disaster assessment. The benchmark integrates object-level recognition, scene-level aggregation, and grounded language generation by combining classification, counting, and structured reporting tasks within a unified multimodal framework. This design enables controlled evaluation of how vision–language models integrate spatial, temporal, and semantic cues when reasoning about disaster-affected environments.

### 3.1 Dataset Foundation and Instance Construction
DisasterInsight builds upon the xBD dataset, which provides 1024×1024 pre- and post-disaster RGB satellite images from very high spatial resolution (sub-0.8 m GSD) sensors, covering 19 disaster events across five hazard types. Each image includes polygon-based building annotations and metadata specifying location, disaster type, and damage level.

To construct object-centric inputs for multimodal models, each image is divided into sixteen non-overlapping 256×256 patches. Patches without buildings are discarded. When multiple buildings appear in a single patch, each building is treated as a separate instance by pairing the shared patch with its corresponding bounding box and metadata.

Each building-centered instance comprises paired pre- and post-disaster image patches, bounding-box coordinates, disaster type and event identifier, damage level, and a building-function label (Section 3.2). This process yields 112,507 building-centered instances, converting xBD from a scene-level dataset into one suitable for fine-grained multimodal reasoning.

### 3.2 Building Function Annotation from OpenStreetMap
To enable function-aware disaster assessment, DisasterInsight augments xBD building instances with functional semantics derived from OpenStreetMap (OSM). OSM is a widely used volunteered geographic information resource with extensive global coverage and rich annotations describing building usage and infrastructure roles, and has been broadly adopted in humanitarian mapping and disaster response [20, 21]. Each xBD building footprint is spatially matched to corresponding OSM building polygons using geographic coordinates provided in the xBD metadata. Raw OSM building tags are highly granular and heterogeneous, with over 170 distinct building-related labels observed across the dataset.

Following established approaches for inferring functional building types from OSM metadata [22], we consolidate these tags into eight high-level functional categories relevant to disaster assessment: **Residential, Recreational Leisure, Religious Cultural, Industrial Utilities, Educational Facilities, Medical Facilities, Government Emergency, and Commercial**. This consolidation reduces annotation noise while preserving functionally meaningful distinctions relevant to post-disaster assessment and infrastructure analysis. The resulting class distribution is highly imbalanced, with residential buildings comprising the majority of instances, reflecting real-world urban composition (Fig. 2(c)).

### 3.3 Task Definitions and Granularity
DisasterInsight defines five tasks spanning perception, counting, and narrative reasoning. Building Function Classification and Damage Level Classification operate at the object level, using individual bounding-box instances. In contrast, Disaster Type Classification, Building Counting, and Structured Report Generation operate at the scene level, using summarized building metadata derived from all structures within a sub-image. While the benchmark contains 112,507 unique building-centered instances, each instance participates in multiple tasks with varying levels of granularity. Combined with multi-prompt evaluation, this design yields a total of 270,096 instruction-following instances across the benchmark (202,837 for training and 67,259 for evaluation).

All classification tasks in DisasterInsight are formulated as closed-set, multi-option prediction problems. Models are instructed to select exactly one label from a predefined category set (e.g., building function, damage level, or disaster type), and to output only the label token. This constrained formulation ensures consistent parsing, reduces output variability, and enables fair and automatic evaluation across diverse vision–language models.

### 3.4 Multi-Prompt Evaluation Strategy
Vision–language models are known to be sensitive to prompt phrasing. To reduce reliance on any single instruction style, DisasterInsight evaluates all tasks under prompt diversity. We design six prompt categories that vary in tone, structure, and contextual framing: (i) simple direct classification, (ii) analyst-style queries, (iii) short and concise instructions, (iv) instructions with explicit reasoning cues, (v) conversational prompts, and (vi) prompts framed in a humanitarian assessment context. For each instance and task, one prompt template is sampled uniformly at random. This single-sample randomization avoids overfitting to any specific instruction style while maintaining computational efficiency.

### 3.5 Input Preprocessing and Scene-Level Summarization
In dense urban scenes, a single 256 × 256 image patch may contain over one hundred building instances. Directly passing all bounding boxes and per-building metadata to large language models results in excessively long prompts that are difficult to interpret and prone to degraded generation quality. To address this, DisasterInsight introduces a structured preprocessing and summarization strategy designed specifically for scene-level reasoning and report generation. For each image patch, individual building annotations, including bounding–box coordinates, functional category, and damage level, are aggregated into compact summary entries. These summaries group buildings by functional category, damage state, and coarse spatial location within the patch (e.g., north, central area, south), and record the count of buildings sharing the same attributes. This representation preserves essential semantic and spatial information while producing concise, interpretable inputs suitable for instruction-following language models.

DisasterInsight defines a structured report generation task in which models are required to produce two narratives per instance: (i) a short summary describing immediately observable impacts, and (ii) a longer analysis providing expanded, response-relevant context. Both outputs follow professional reporting conventions used by humanitarian assessment guidelines, focusing on observable damage, structural condition, and situational considerations.

### 3.6 Ground-Truth Report Generation and LLM Selection
Ground-truth reports in DisasterInsight are generated using a large language model under strict grounding and content constraints, reflecting the role of narrative situation reports in operational disaster response. To account for different response phases, each building instance is associated with two report types: a short-form summary for rapid situational awareness and a long-form narrative supporting risk assessment and response planning. Short reports provide a compact, strictly descriptive overview of observable impacts, while long reports expand the narrative to include safety risks, humanitarian implications, and recovery considerations. Both formats are required to remain grounded in pre- and post-disaster imagery, avoid unverified details, and maintain a professional tone consistent with humanitarian reporting practices.

Among several evaluated models (Qwen3-30B-A3B-Instruct, DeepSeek-V3.1, GPT-OSS-20B, and Llama 3.3 70B Instruct), **Llama 3.3 70B Instruct** demonstrated the strongest adherence to grounding constraints and was selected to generate all ground-truth reports.

### 3.7 DI-Chat: Domain-Adapted Instruction-Tuned Baselines
To complement the DisasterInsight benchmark and enable domain-adapted evaluation, DI-Chat is introduced as an instruction-tuned baseline derived from DisasterInsight. DI-Chat is not a standalone architecture, but a general instruction fine-tuning framework that adapts existing vision–language models to disaster-specific tasks using the structured instructions and annotations provided by the benchmark. In this work, DI-Chat is instantiated on multiple vision–language backbones, including TeoChat and Qwen2.5-VL, resulting in two baseline models denoted as **DI-Chat (TeoChat)** and **DI-Chat (Qwen2.5-VL)**. Both variants are fine-tuned using parameter-efficient Low-Rank Adaptation (LoRA) on the training split of DisasterInsight, following a multimodal instruction-tuning paradigm similar to Video-LLaVA-style training.

### 3.8 Evaluation Metrics
*   **Classification tasks:** Macro-averaged F1 (%) to account for severe class imbalance across building functions, damage levels, and disaster types, and accuracy for completeness. For damage-level classification, which follows an ordinal severity structure, ordinal mean absolute error (MAE) is additionally reported.
*   **Building counting:** MAE and RMSE.
*   **Structured report generation:** BLEU-4, ROUGE-L, and BERTScore F1.

## 4 Experiments

We evaluate a diverse set of open-source and remote-sensing–oriented vision–language models (VLMs) on the five tasks: Building Function Classification (BFC), Damage Level Classification (DLC), Disaster Type Classification (DTC), Building Counting (BC), and Structured Report Generation (SRG). The evaluated models include LLaVA-OneVision [11], Qwen2.5-VL-7B [12], Qwen3-VL-8B, Qwen3-VL-30B [27], TeoChat [5], and two domain-adapted baselines obtained via instruction fine-tuning on DisasterInsight, denoted as DI-Chat (TeoChat) and DI-Chat (Qwen2.5-VL).

### 4.1 Experimental Setup
We instantiate the DI-Chat framework by fine-tuning two vision–language backbones, TeoChat [5] and Qwen2.5-VL-7B [12], using parameter-efficient Low-Rank Adaptation (LoRA) within a Video-LLaVA-inspired multimodal training framework. LoRA uses $(r, \alpha) = (64, 128)$ for TeoChat and $(32, 64)$ for Qwen2.5-VL-7B, with LoRA applied to the language model only and frozen vision backbones in both cases.

### 4.2 Task Difficulty and Category Imbalance
Fig. 2(c) illustrates the class distributions across training and validation splits for each task. Building Function Classification (BFC) exhibits severe imbalance, with residential buildings dominating the dataset, while functionally critical categories such as medical, government, and religious structures are sparsely represented. Damage Level Classification shows a similar skew toward "no-damage" instances, whereas Disaster Type Classification is comparatively more balanced across hazards.

### 4.3 Quantitative Results Across Tasks
Table 2 summarizes performance across classification, structured report generation, and counting tasks. Generic vision–language models achieve moderate accuracy on Damage Level and Disaster Type Classification, but perform poorly on Building Function Classification, highlighting the difficulty of inferring fine-grained functional semantics from satellite imagery under severe class imbalance.

Both DI-Chat variants consistently outperform generic and remote-sensing baselines on Damage Level and Disaster Type Classification, structured report generation, and building counting. For Building Function Classification, DI-Chat improves performance relative to its corresponding backbone models, but does not surpass the strongest generic baseline (LLaVA-OneVision), indicating that function inference remains a challenging open problem. Notably, DI-Chat (Qwen2.5-VL-7B) substantially outperforms DI-Chat (TeoChat) on BFC, suggesting that stronger vision–language backbones benefit more from domain-specific instruction tuning.

**Table 2: Quantitative performance comparison of vision–language models on DisasterInsight.**

| Method | F1 (%) ↑ | Accuracy (%) ↑ | DLC Ord. MAE ↓ | Report Gen. ↑ | Counting ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | **AVG** | **BFC** | **DLC** | **DTC** | **AVG** | **BFC** | **DLC** | **DTC** | | **BLEU** | **R-L** | **BERT** | **MAE** | **RMSE** |
| LLaVA-OneVision | 19.14 | 18.22 | 28.02 | 11.19 | 54.92 | 82.93 | 68.02 | 13.82 | 0.549 | 2.25 | 19.28 | 84.69 | 3.76 | 6.19 |
| Qwen2.5-VL-7B | 16.20 | 14.51 | 18.36 | 15.73 | 65.80 | 82.50 | 74.49 | 24.12 | 0.498 | 1.87 | 18.66 | 83.93 | 4.21 | 6.60 |
| Qwen3-VL-8B | 20.67 | 16.89 | 34.51 | 10.60 | 57.47 | 84.01 | 69.93 | 18.48 | 0.561 | 1.22 | 16.59 | 82.75 | 4.17 | 7.38 |
| Qwen3-VL-30B | 21.24 | 15.43 | 28.72 | 19.57 | 60.12 | 84.21 | 68.31 | 27.85 | 0.657 | 1.17 | 16.52 | 82.50 | 3.60 | 5.68 |
| TeoChat | 15.73 | 4.99 | 14.19 | 28.02 | 67.35 | 78.06 | 54.67 | 60.14 | 0.406 | 3.38 | 22.15 | 84.42 | 4.43 | 7.07 |
| **DI-Chat (TeoChat)** | **46.36** | **5.94** | **54.41** | **78.73** | **87.76** | **83.37** | **84.20** | **95.70** | **0.220** | **18.45** | **35.85** | **88.03** | **3.31** | **5.82** |
| **DI-Chat (Qwen2.5-VL-7B)** | **46.13** | **17.74** | **45.19** | **75.46** | **85.23** | **86.80** | **80.44** | **88.45** | **0.352** | **16.43** | **35.19** | **89.45** | **3.82** | **6.36** |

### 4.4 Summary of Experimental Findings
Across all tasks and metrics, three conclusions emerge:
1.  **Building-function classification** is the most challenging task due to high semantic ambiguity and extreme class imbalance.
2.  **Generic vision–language models** are not well suited for structured disaster reporting without domain adaptation.
3.  **Domain-specific instruction tuning** using the DI-Chat framework yields large and consistent improvements across damage-level and disaster-type classification, building counting, and structured narrative generation, while also improving building-function classification relative to backbone models.

## 5 Discussion and Limitations
Our results reveal several important insights into multimodal reasoning for disaster imagery. Among all evaluated tasks, building-function classification remains the most challenging due to severe class imbalance and the ambiguity of inferring functional semantics from overhead imagery. This highlights function recognition as a core open problem for disaster-response benchmarks and motivates building-centered, function-aware evaluation beyond scene-level analysis.

While generic vision–language models achieve moderate accuracy on damage-level classification, their macro-F1 scores remain low under class imbalance and severity confusion. For damage-level classification, macro-F1 captures exact label agreement but ignores the ordinal nature of damage severity. We therefore additionally report ordinal MAE, which penalizes large severity confusions (e.g., no-damage vs. destroyed) more strongly than adjacent ones, providing a more operationally meaningful assessment.

DisasterInsight has limitations, including noisy OpenStreetMap-derived function labels, reliance on optical imagery, and automatically generated reports; future work will address these through multimodal extensions, improved annotations, and expert human evaluation.

## 6 Conclusion
We presented DisasterInsight, a multimodal benchmark for evaluating vision–language models in disaster-response settings. By integrating instance-level building-function classification, damage and hazard reasoning, building counting, and structured report generation aligned with humanitarian guidelines, DisasterInsight captures key components of real-world operational workflows. Our experiments show that general-purpose vision–language models struggle with function-aware reasoning and structured narrative generation, highlighting a gap between current model capabilities and practical disaster-response needs. In contrast, domain-specific instruction tuning with the DI-Chat framework yields substantial improvements in damage assessment, disaster-type recognition, building counting, and structured report generation, while building-function classification remains a challenging open problem. Overall, DisasterInsight provides an operationally grounded benchmark to advance more accurate and trustworthy multimodal AI for disaster assessment.

## References
[1] L. Dong and J. Shan, “A comprehensive review of earthquake-induced building damage detection with remote sensing techniques,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 84, pp. 85–99, 2013.
[2] J. Chen, Z. Liu, B. Wang, P. Wang, and F. Lu, “Bright: A benchmark for remote sensing humanitarian analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19477–19486, 2022.
[3] L. Feng, W. Liu, D. Zhang, et al., “Floodnet: A high resolution satellite dataset for flood detection and semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 904–905, 2020.
[4] R. Gupta, R. Hosfelt, S. Sajeev, N. Patel, B. Goodman, J. Doshi, E. Heim, H. Choset, and M. Gaston, “xbd: A dataset for assessing building damage from satellite imagery,” arXiv preprint arXiv:1911.09296, 2019.
[5] H. Chen, Y. Zhang, Z. Wang, et al., “Teochat: A multimodal geospatial foundation model,” arXiv preprint arXiv:2402.11579, 2024.
[6] J. Liu, Y. Wang, Y. Zhang, et al., “Disasterm3: A multimodal and multitask benchmark for disaster management,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18744–18754, 2023.
[7] X. Zhu, R. Tang, Z. Wang, et al., “Geochat: Grounded large vision-language model for remote sensing,” arXiv preprint arXiv:2311.03659, 2023.
[8] M. Haklay and P. Weber, “Openstreetmap: User-generated street maps,” IEEE Pervasive Computing, vol. 7, no. 4, pp. 12–18, 2008.
[9] B. Lin, Y. Ye, B. Zhu, J. Cui, M. Ning, P. Jin, and L. Yuan, “Video-llava: Learning united visual representation by alignment before projection,” in Proceedings of the 2024 conference on empirical methods in natural language processing, pp. 5971–5984, 2024.
[10] P. Soni, R. Ganti, V. Ramanathan, C. L. Zitnick, and M. Paluri, “Earthdial: Turning multi-sensory earth observations to interactive dialogues,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
[11] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, ..., and C. Li, “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.
[12] J. Bai, Z. Wang, S. Wang, et al., “Qwen-vl: A versatile vision-language model for understanding and generation,” arXiv preprint arXiv:2308.15187, 2023.
[13] M. Imran, F. Ofli, et al., “Twitter as a sensing platform for emergency events and humanitarian crises,” in Proceedings of the ACM on Human-Computer Interaction, 2016.
[14] H. Purohit, G. Dong, et al., “Emergency events detection and classification on twitter,” in Proceedings of the International Conference on Social Informatics, pp. 1–12, 2014.
[15] S. Vieweg, A. L. Hughes, et al., “Microblogging during two natural hazards events: what twitter may contribute to situational awareness,” in Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pp. 1079–1088, 2010.
[16] X. Wu, T. Qiu, L. Zhan, et al., “Rsicd: A remote sensing image captioning dataset,” in Proceedings of the IEEE International Geoscience and Remote Sensing Symposium, pp. 1337–1340, 2017.
[17] F. Alam, F. Ofli, and M. Imran, “Crisismmd: Multimodal twitter dataset for crisis response,” in Proceedings of the International AAAI Conference on Web and Social Media, vol. 13, pp. 750–751, 2018.
[18] A. Ahmad, A. Khandelwal, A. B. Patwa, et al., “Crisisfacts: A benchmark for crisis fact verification,” in Proceedings of the ACM Web Conference, pp. 1076–1086, 2022.
[19] M. Patro, T. Chakraborty, et al., “Disasterqa: A dataset for question answering in disaster scenarios,” in Proceedings of the Web Conference, pp. 1835–1846, 2022.
[20] C. Barrington-Leigh and A. Millard-Ball, “The world’s user-generated road map is >60% complete,” PLOS ONE, vol. 12, no. 8, p. e0180698, 2017.
[21] P. Neis and A. Zipf, “The evolution of openstreetmap: from a user-generated geodata project to a volunteered geographic information provider,” in Proceedings of the 4th International Conference on Cartography and GIS, pp. 267–274, 2012.
[22] J. Yan, J. Zhang, and H. Fan, “Tagging buildings in openstreetmap: A machine learning approach to inferring functional types,” International Journal of Geographical Information Science, vol. 36, no. 5, pp. 987–1008, 2022.
[23] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al., “Deepseek-v3 technical report,” arXiv preprint arXiv:2412.19437, 2024.
[24] S. Agarwal, L. Ahmad, J. Ai, S. Altman, A. Applebaum, E. Arbus, R. K. Arora, Y. Bai, B. Baker, H. Bao, et al., “gpt-oss-120b & gpt-oss-20b model card,” arXiv preprint arXiv:2508.10925, 2025.
[25] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.
[26] J. Edstedt, A. Berg, M. Felsberg, J. Karlsson, F. Benavente, A. Novak, and G. G. Pihlgren, “Vidharm: A clip based dataset for harmful content detection,” in 2022 26th International Conference on Pattern Recognition (ICPR), pp. 1543–1549, 2022.
[27] S. Bai, S. Cai, et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.
