# EdgeV-SE: Self-Reflective Fine-Tuning Framework for Edge-Deployable Vision-Language Models

**Published in Applied Sciences, 2026, 16, 818**
**Received:** 22 December 2025 | **Revised:** 9 January 2026 | **Accepted:** 12 January 2026 | **Published:** 13 January 2026

**Authors:** Yoonmo Jeon, Seunghun Lee, and Woongsup Kim*

**Department of Information and Communication Engineering, Dongguk University, Seoul 04620, Republic of Korea**
\* Correspondence: woongsup@dongguk.edu

### Featured Application
The proposed framework enables the deployment of robust Vision-Language Models on resource-constrained off-the-shelf edge devices, such as the NVIDIA Jetson series. Its primary application is real-time disaster damage assessment using satellite imagery in communication-denied environments, facilitating immediate decision-making for first responders.

---

## Abstract

The deployment of Vision-Language Models (VLMs) in Satellite IoT scenarios is critical for real-time disaster assessment but is often hindered by the substantial memory and compute requirements of state-of-the-art models. While parameter-efficient fine-tuning (PEFT) enables adaptation with minimal computational overhead, standard supervised methods often fail to ensure robustness and reliability on resource-constrained edge devices. To address this, we propose **EdgeV-SE**, a self-reflective fine-tuning framework that significantly enhances the performance of VLM without introducing any inference-time overhead. Our framework incorporates an uncertainty-aware self-reflection mechanism with asymmetric dual pathways: a generative linguistic pathway and an auxiliary discriminative visual pathway.

By estimating uncertainty from the linguistic pathway using a log-likelihood margin between class verbalizers, EdgeV-SE identifies ambiguous samples and refines its decision boundaries via consistency regularization and cross-pathway mutual learning. Experimental results on hurricane damage assessment demonstrate that our approach improves image classification accuracy, enhances image–text semantic alignment, and achieves superior caption quality. Notably, our work achieves these gains while maintaining practical deployment on a commercial off-the-shelf edge device such as NVIDIA Jetson Orin Nano, preserving the inference latency and memory footprint. Overall, our work contributes a unified self-reflective fine-tuning framework that improves robustness, calibration, and deployability of VLMs on edge devices.

**Keywords:** Vision-Language Model (VLM); edge computing; self-reflective learning; consistency regularization; mutual learning; satellite IoT; NVIDIA Jetson; disaster analysis

## 1. Introduction

With the increasing frequency and intensity of natural disasters due to climate change, rapid and accurate analysis of satellite imagery for damage assessment has become critically important. In large-scale hurricane or flood events, communication infrastructure is often damaged or unavailable, and network connectivity to ground stations is frequently intermittent. These disaster-zone conditions pose significant challenges for cloud-based AI, particularly in terms of latency, bandwidth, and availability. As a result, Intelligent satellite IoT, where satellites, high-altitude platforms, or unmanned aerial vehicles are paired with on-board or nearby edge devices that perform local inference, has emerged as a practical alternative for real-time disaster response.

In this context, Vision–Language Models (VLMs) are particularly attractive. A single VLM can both classify damage severity and generate human-readable descriptions of the scene, providing interpretable evidence (e.g., “standing water covering roads”, “intact roofs and dry ground”) that is valuable for analysts and first responders. However, most state-of-the-art VLMs are designed for server-class GPUs with abundant memory, rendering them too resource-intensive for direct deployment on small edge devices in Satellite IoT scenarios.

For example, popular open-source models such as LLaVA-1.5-7B require server-class GPUs with substantial memory footprints, particularly in FP16 settings. In addition, top-tier commercial models like GPT-4V have undisclosed architectures and model scales and are deployed via cloud-based inference, which precludes practical on-device deployment. The primary goal of our research is to develop a high-performance VLM that can efficiently run on commercial off-the-shelf edge devices, such as the NVIDIA Jetson Orin Nano, which has been widely adopted for on-device deep learning applications. Jetson Orin Nano provides only 8 GB of unified LPDDR5 memory and modest compute throughput, even though it is specifically designed as an edge-inference module. This fundamental resource mismatch makes it infeasible to deploy such large VLMs directly on edge devices, motivating the use of more compact yet expressive architectures together with intelligent fine-tuning strategies.

Among existing models, we choose BLIP-Large as a representative encoder–decoder VLM that is close to the upper limit of what can be deployed on commercial off-the-shelf edge devices, such as the NVIDIA Jetson series, in FP16. For complex and specialized tasks such as hurricane damage assessment from satellite imagery, characterized by scarce labeled data and subtle visual differences between damage and no-damage classes, standard supervised fine-tuning (SFT) alone is often data-inefficient and insufficient to deliver the robustness and reliability required in real-world settings.

In this work, we propose **EdgeV-SE (Edge-deployable Vision–Language Models using Self Evaluation and Self Enhancement)**, a self-reflective fine-tuning framework designed to enable models to identify their own uncertainty and reconcile internal inconsistencies between complementary pathways.

### Main Contributions
*   We propose a self-reflective fine-tuning framework, **EdgeV-SE**, that enables the model to recognize its own uncertainty and learn by resolving internal inconsistencies.
*   We introduce an efficient mechanism that enhances prediction reliability with minimal overhead by designing **asymmetric dual pathways** (a generative linguistic path and a discriminative visual path) within the VLM and performing internal cross-validation through mutual learning.
*   We demonstrate the practical effectiveness of our approach by empirically validating the proposed model’s superior classification accuracy and inference efficiency on an actual edge device.

## 2. Related Works

### 2.1. Efficient VLMs for Edge Environments
Research on deploying VLMs on edge devices has primarily focused on model compression techniques such as Quantization, semantic-aware token pruning, and Knowledge Distillation. More recently, efforts have been made to design lightweight architectures specifically for the edge, such as MobileVLM and edge-cloud collaborative approaches for Vision-Language Models. While effective, these approaches often involve a significant trade-off in accuracy or require redesigning and pre-training models from scratch. An alternative is Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA and its variant QLoRA, which freeze the pre-trained weights and train only a small number of adaptive modules to improve memory efficiency.

### 2.2. Self-Supervised and Self-Reflective Learning Under Weak Supervision
Semi-Supervised Learning (SSL) aims to improve model performance by leveraging a small amount of labeled data along with a large amount of unlabeled data. Consistency regularization, a key principle in SSL, posits that a model’s prediction should remain invariant to non-essential perturbations of its input, such as data augmentation. The effectiveness of this principle has been demonstrated in seminal works like FixMatch and Unsupervised Data Augmentation (UDA). Meanwhile, Self-Supervised Learning generates supervisory signals from the data itself to guide the learning process, as seen in contrastive methods like MoCo and SimCLR.

### 2.3. Uncertainty-Aware Learning and Calibration
Reliable deployment in resource-constrained or safety-critical environments depends not only on accuracy but also on how well a model’s confidence reflects its probability of correctness. Prior work has studied predictive uncertainty in deep networks through approximate Bayesian approaches such as Monte Carlo dropout. Ensemble-based methods provide another practical alternative, showing that independently trained model ensembles yield strong uncertainty estimates and often improve calibration.

### 2.4. Vision-Language Interaction and Hallucination Mitigation Strategy in VLM
While large language models (LLMs) exhibit impressive reasoning abilities, they are inherently blind to visual inputs. In contrast, vision models perceive visual information effectively but typically lack comparable reasoning capacity. This complementarity has driven a growing convergence of language and vision models. Recent works have explored deep interaction mechanisms. By embedding trainable visual expert modules within the attention layers of the language model, CogVLM enables fine-grained vision–language interactions at deeper stages of representation.

## 3. Self-Reflective Fine-Tuning (EdgeV-SE)

EdgeV-SE augments standard supervised fine-tuning (SFT) with a training-time self-reflective mechanism derived from the model’s internal disagreement. The training update for each mini-batch proceeds in four phases:

1.  **Discrepancy induction:** Creates complementary predictions from asymmetric linguistic vs. visual pathways to expose internal disagreement.
2.  **Uncertainty diagnosis:** Estimates sample uncertainty via a margin-based self-assessment and computes an uncertainty weight.
3.  **Discrepancy resolution:** Focuses learning on uncertain samples by enforcing augmentation consistency and cross-pathway agreement.
4.  **Supervised consolidation:** Aggregulates supervised losses and reflection losses to update parameters.

**Algorithm 1: EdgeV-SE Training Procedure (per mini-batch)**
*   **Input:** Model parameters $\theta$, visual head $H$, batch $\mathcal{B} = \{(v_{0,i}, y_i, c_i)\}_{i=1}^B$, Hyperparameters $\tau, T, \lambda_{uncert}, \lambda_{consist}, \lambda_{mutual}, \lambda_{LLC}$
*   **1. Discrepancy Induction:** Compute linguistic logits and visual logits from asymmetric pathways.
*   **2. Uncertainty Diagnosis:** Calculate margin $\Delta_i$ and uncertainty weight $w_i^{uncert}$.
*   **3. Discrepancy Resolution:** Apply consistency regularization $\mathcal{L}_{consist,i}^{w}$ and mutual learning $\mathcal{L}_{mutual,i}^{w}$.
*   **4. Parameter Update:** Minimize $\mathcal{L}_{total} = \mathcal{L}_{SFT} + \lambda_{LLC}\mathcal{L}_{LLC} + \lambda_{consist}\mathcal{L}_{consist}^w + \lambda_{mutual}\mathcal{L}_{mutual}^w$.

### 3.1. Discrepancy Induction: Asymmetric Dual Pathways
The linguistic pathway functions as the **Generative Branch** (acting as a ‘Theorist’), interpreting high-level semantics such as debris, flooding, or roof collapse. Given an input image, it evaluates the likelihood of each class by conditioning on prompt-based class descriptions.

The visual pathway functions as the **Discriminative Branch** (acting as an "Empiricist"), predicting class probabilities directly from the vision encoder, focusing on visual details. A lightweight head produces logits and probabilities:
$$z^V(v_0) = MLP(h(v_0))$$
$$p^V(k|v_0) = softmax(z^V(v_0))_k$$

### 3.2. Recognizing Uncertainty: Identifying ‘Uncertain’ Samples via Self-Diagnosis
This phase quantifies the uncertainty of the class prediction for a given image by measuring the margin between the highest and second-highest class probabilities. To quantify its own uncertainty, the model measures a prediction margin from the linguistic pathway:
$$\Delta(v_0) = l_c(v_0) - \max_{k \neq c} l_k(v_0)$$
where $c$ denotes the class with the maximum predicted probability for image $v_0$. A small absolute margin $\Delta(v_0)$ implies indecision or conflict between the two outcomes, corresponding to an ambiguous or high-uncertainty sample.

The model assigns a higher learning weight to such uncertainty based on a threshold $\tau$:
$$w^{uncert}(v_0) = \begin{cases} \lambda_{uncert}, & \text{if } |\Delta(v_0)| < \tau \\ 1, & \text{if } |\Delta(v_0)| \geq \tau \end{cases}$$

### 3.3. Discrepancy Resolution: Internalizing Knowledge Through Consistency and Mutual Consensus

#### (a) Consistency Regularization
EdgeV-SE constructs two stochastically augmented views $v_1 = aug_1(v_0)$ and $v_2 = aug_2(v_0)$ using mild spatial/photometric transformations. Then, EdgeV-SE enforces consistency based on their prediction margins:
$$\mathcal{L}_{consist}^w(x) = w^{uncert}(v_0) \cdot (\Delta(v_1) - \Delta(v_2))^2$$

#### (b) Mutual Learning ($\mathcal{L}_{mutual}^w(x)$)
The Linguistic Pathway ("Theorist") and the Visual Pathway ("Empiricist") act as soft targets for each other. We quantify the cross-pathway agreement using the Jensen–Shannon divergence (JSD) between $p^L$ and $p^V$. We define the mutual loss as a weighted JSD:
$$\mathcal{L}_{mutual}^w(x) = w^{uncert}(v_0) \cdot \frac{1}{2} [KL(p^L \| m) + KL(p^V \| m)], \quad m = \frac{1}{2}(p^L + p^V)$$

### 3.4. Supervised Consolidation
The total objective integrates four components:
$$\mathcal{L}_{total} = \mathcal{L}_{SFT} + \lambda_{LLC}\mathcal{L}_{LLC} + \lambda_{consist}\mathcal{L}_{consist}^w + \lambda_{mutual}\mathcal{L}_{mutual}^w$$

## 4. Experimental Setup and Implementation

### 4.1. Dataset and Preprocessing
We employed a balanced subset of a hurricane damage assessment dataset consisting of satellite images categorized into two classes: damage and no_damage. Since the original dataset provides only image-level labels and lacks human-written captions, we generated domain-specific pseudo-reference captions using LLaVA-1.5-7B. The final curated dataset contains 5000 damage/5000 no-damage images for training, 1000/1000 images for validation, and 1000/1000 images for testing, resulting in 14,000 images in total.

### 4.2. Model Configuration and Training Details
We use the Salesforce/blip-image-captioning-large model as the base Vision-Language Model (VLM). For parameter-efficient fine-tuning, all methods except Standard SFT used the same QLoRA setting ($r = 16, \alpha = 32, dropout = 0.05$) applied to selected text layers and the last two vision transformer blocks. The main hyperparameters for the self-reflective fine-tuning were empirically set as $\tau = 0.7, \lambda_{uncert} = 1.5, \lambda_{consist} = 0.3, \lambda_{mutual} = 0.2, \lambda_{LLC} = 0.3$. The temperature was set to $T = 2.0$.

### 4.3. Evaluation Metrics
For classification, we report Accuracy, class-wise F1, and Macro-F1, along with Expected Calibration Error (ECE). Caption quality is assessed using CIDEr-D, BERTScore, and CLIPScore.

## 5. Experimental Results and Analysis

### 5.1. Component-Wise Analysis and Ablation Study

**Table 2. Evolution and Ablation (Class-wise F1).**

| Phase | Model | Damage F1 | No-Damage F1 | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Variants | Self-Reflection type 1 | 0.918 ± 0.004 | 0.918 ± 0.004 | Generate-then-Correct |
| Variants | Self-Reflection type 2 | 0.964 ± 0.003 | 0.964 ± 0.003 | Direct LL + Multi-task |
| Ablation | Model A | 0.966 ± 0.003 | 0.966 ± 0.003 | SFT + Consistency |
| Ablation | Model B | 0.981 ± 0.002 | 0.980 ± 0.002 | SFT + Mutual Learning |
| Ours | **EdgeV-SE** | **0.985 ± 0.004** | **0.986 ± 0.002** | Combine all |

**Table 3. Captioning Performance for Progressive Model Variants.**

| Phase | Model | CIDEr-D (↑) | BERTScore (↑) | CLIPScore (↑) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Variants | Self-Reflection type 1 | 26.08 ± 0.62 | 88.05 ± 0.15 | 28.46 ± 0.18 | Generate-then-Correct |
| Variants | Self-Reflection type 2 | 37.49 ± 0.55 | 90.77 ± 0.11 | 28.79 ± 0.14 | Direct LL + Multi-task |
| Ablation | Model A | 38.14 ± 0.48 | 90.79 ± 0.09 | 28.51 ± 0.13 | SFT + Consistency |
| Ablation | Model B | 38.00 ± 0.50 | 90.60 ± 0.10 | 28.55 ± 0.12 | SFT + Mutual Learning |
| Ours | **EdgeV-SE** | **38.37 ± 0.42** | **90.82 ± 0.08** | **28.21 ± 0.11** | Combine all |

### 5.2. Comparison with Advanced Fine-Tuning Baselines

**Table 5. Baseline Classification Performance (Standard SFT vs. Controlled SFT).**

| Method | LoRA Scope | LoRA (r, α) | Trainable Params (%) | Macro-F1 |
| :--- | :--- | :--- | :--- | :--- |
| Standard SFT | Text layers only | (128, 256) | 14.860 | 0.911 ± 0.007 |
| Controlled SFT | Text + last 2 Vision Blocks | (16, 32) | 1.330 | 0.930 ± 0.005 |

**Table 6. Comparative Classification Performance.**

| Method | Evaluation Mechanism | Macro-F1 | Remarks |
| :--- | :--- | :--- | :--- |
| Standard SFT | Cross-Entropy | 0.911 | Weak SFT Baseline |
| Controlled SFT | Cross-Entropy | 0.930 | Strong SFT Baseline |
| Self-Rewarding VLM | Internal Reward Optimization | 0.979 | LL-based Reward Selection |
| Iterative Self-Correction | Progressive Self-Correction | 0.972 | Uncertainty-based Refinement (Adapted) |
| **EdgeV-SE** | **Dual Pathways + Consistency + Mutual Learning** | **0.985** | **Self-Reflective Optimization** |

### 5.3. On-Device Performance

We benchmarked our model on an NVIDIA Jetson Orin Nano (8 GB) configured in 15 W power mode (MAXN) using FP16 precision with batch = 1.

**Table 8. On-Device Performance Benchmark (NVIDIA Jetson Orin Nano 8 GB).**

| Method | Latency (ms/Image) | Throughput (FPS) | Peak Memory (GB) | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Controlled SFT | 1836.5 | 0.544 | 0.915 | Strong SFT Baseline |
| **EdgeV-SE** | **1837.2** | **0.544** | **0.915** | **No Additional Overhead** |

### 5.4. Qualitative Analysis
Qualitative results show that EdgeV-SE is more reliable in visual grounding. For instance, it correctly distinguishes turbid flooding from dry ground (high sensitivity) and swimming pools from floodwater (high specificity/resistance to hallucination), whereas the baseline SFT often fails in these hard cases.

### 5.5. Temperature Scaling and Calibration Procedure
We apply temperature scaling as a post-hoc calibration method. The temperature $T$ is selected on the validation set by minimizing the Expected Calibration Error (ECE). $T = 2.0$ consistently achieves the lowest ECE (1.3%).

### 5.6. Robustness Under Common Corruptions and Calibration
EdgeV-SE exhibits graceful degradation under several common perturbations (e.g., mild rotation/brightness/blur). However, severe contrast collapse and stronger Gaussian noise are principal failure modes.

### 5.9. Generalization to Wildfire Damage Assessment
We conducted additional experiments on the Wildfire Prediction Dataset. Despite the domain shift, EdgeV-SE consistently outperforms the standard supervised fine-tuning (SFT) baseline in terms of both Accuracy and Macro-F1.

**Table 12. Performance comparison on the wildfire damage assessment dataset.**

| Method | Accuracy | Macro-F1 | ECE (%) |
| :--- | :--- | :--- | :--- |
| Controlled SFT | 0.913 | 0.911 | 5.8 |
| **EdgeV-SE** | **0.961** | **0.960** | **1.2** |

## 6. Discussion

### 6.1. Synergistic Mechanisms of Self-Reflection
The significant performance gains of EdgeV-SE, improving Macro-F1 from 0.911 to 0.985, stem from the mechanism’s ability to selectively weigh gradient updates based on internal uncertainty. The mutual learning objective serves as a regularization term that prevents the “semantic drift” often observed in VLMs.

### 6.2. Operational Feasibility vs. Hard Real-Time Constraints
Our benchmark on the Jetson Orin Nano shows an inference speed of approximately 0.54 FPS (1.84 s/image). While this does not meet "video-rate" real-time, it satisfies the "operational real-time" constraints of Satellite IoT disaster response where data transmission is the primary bottleneck.

### 6.5. Limitations and Future Directions
*   **Pseudo-Ground Truth:** Despite filtering noisy pseudo-captions, some residual hallucinations may remain.
*   **Domain Specificity:** While validated on hurricane and wildfire datasets, broader validation across additional disaster categories remains important.
*   **Comparison scope:** This study focuses on BLIP-Large. Future work could apply EdgeV-SE to newer edge-specific architectures like MobileVLM.

## 7. Conclusions

We proposed a self-reflective fine-tuning framework, **EdgeV-SE**, for edge-deployable VLMs. EdgeV-SE integrates uncertainty-aware weighting, margin-level multi-view semantic consistency, and dual-pathway mutual learning. Our proposed model yields a substantial improvement in classification performance, increasing Macro-F1 from 0.911 to 0.985 over standard supervised fine-tuning without introducing any additional inference-time overhead. Crucially, these gains are achieved while preserving edge feasibility, maintaining low latency and high throughput on resource-constrained platforms such as the Jetson Orin Nano.

## References
1. Masson-Delmotte, V.; et al. Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change; Cambridge University Press: Cambridge, UK; New York, NY, USA, 2021.
2. Liu, Z.; Jiang, Y.; Rong, J. Resource Allocation Strategy for Satellite Edge Computing Based on Task Dependency. Appl. Sci. 2023, 13, 10027.
3. Liu, H.; Li, C.; Wu, Q.; Lee, Y.J. Visual Instruction Tuning. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), New Orleans, LA, USA, 10–16 December 2023; Volume 36, pp. 34892–34916.
4. OpenAI. GPT-4V(ision) System Card; OpenAI: San Francisco, CA, USA, 2023.
5. NVIDIA Corporation. Jetson Orin Nano Developer Kit User Guide.
6. Shin, D.-J.; Kim, J.-J. A Deep Learning Framework Performance Evaluation to Use YOLO in Nvidia Jetson Platform. Appl. Sci. 2022, 12, 3734.
7. Yang, D.; et al. Retrieve-then-compare mitigates visual hallucination in multi-modal large language models. Intell. Robot. 2025, 5, 248–275.
8. Ba, X.; et al. Multimodal semantic communication system based on graph neural networks. Intell. Robot. 2025, 5, 805–826.
9. Li, J.; Li, D.; Xiong, C.; Hoi, S. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. arXiv 2022, arXiv:2201.12086.
10. Lin, T.-Y.; et al. Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 22–29 October 2017; pp. 2980–2988.
11. Shinn, N.; et al. Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv 2023, arXiv:2303.11366.
12. Madaan, A.; et al. Self-Refine: Iterative Refinement with Self-Feedback. arXiv 2023, arXiv:2303.17651.
13. Zhang, Y.; et al. Deep Mutual Learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Salt Lake City, UT, USA, 18–22 June 2018; pp. 4320–4328.
14. Jacob, B.; et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Salt Lake City, UT, USA, 18–22 June 2018; pp. 2704–2713.
15. Seo, H.; Choi, Y.S. V-PRUNE: Semantic-Aware Patch Pruning Before Tokenization in Vision-Language Model Inference. Appl. Sci. 2025, 15, 9463.
16. Hinton, G.; Vinyals, O.; Dean, J. Distilling the Knowledge in a Neural Network. arXiv 2015, arXiv:1503.02531.
17. Chu, X.; et al. MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices. arXiv 2023, arXiv:2312.16886.
18. Liu, X.; et al. Aligned Vector Quantization for Edge-Cloud Collaborative Vision-Language Models. arXiv 2024, arXiv:2411.05961.
19. Hu, E.J.; et al. LoRA: Low-Rank Adaptation of Large Language Models. In Proceedings of the International Conference on Learning Representations (ICLR), Virtual, 25–29 April 2022.
20. Dettmers, T.; et al. QLoRA: Efficient Finetuning of Quantized LLMs. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), New Orleans, LA, USA, 10–16 December 2023; Volume 36, pp. 10088–10115.
21. Van Engelen, J.E.; Hoos, H.H. A survey on semi-supervised learning. Mach. Learn. 2020, 109, 373–440.
22. Sohn, K.; et al. FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), Virtual, 6–12 December 2020; Volume 33, pp. 596–608.
23. Xie, Q.; et al. Unsupervised Data Augmentation for Consistency Training. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), Virtual, 6–12 December 2020; Volume 33, pp. 6256–6268.
24. Tarvainen, A.; Valpola, H. Mean Teachers Are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), Long Beach, CA, USA, 4–9 December 2017; Volume 30.
25. He, K.; et al. Momentum Contrast for Unsupervised Visual Representation Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 13–19 June 2020; pp. 9729–9738.
26. Chen, T.; et al. A Simple Framework for Contrastive Learning of Visual Representations. In Proceedings of the 37th International Conference on Machine Learning (ICML), Virtual, 13–18 July 2020; pp. 1597–1607.
27. Wei, J.; et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), New Orleans, LA, USA, 28 November–9 December 2022; Volume 35, pp. 24824–24837.
28. Wang, X.; et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In Proceedings of the International Conference on Learning Representations (ICLR), Kigali, Rwanda, 1–5 May 2023.
29. Zhang, L.; et al. Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Republic of Korea, 27 October–2 November 2019; pp. 3712–3721.
30. Gal, Y.; Ghahramani, Z. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. arXiv 2016, arXiv:1506.02142.
31. Lakshminarayanan, B.; et al. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. arXiv 2017, arXiv:1612.01474.
32. Guo, C.; et al. On Calibration of Modern Neural Networks. arXiv 2017, arXiv:1706.04599.
33. Shrivastava, A.; et al. Training Region-based Object Detectors with Online Hard Example Mining. arXiv 2016, arXiv:1604.03540.
34. Leng, S.; et al. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 16–22 June 2024; pp. 13872–13882.
35. Radford, A.; et al. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR, Virtual, 18–24 July 2021; Volume 139, pp. 8748–8763.
36. Wang, W.; et al. CogVLM: Visual Expert for Pretrained Language Models. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), Vancouver, BC, Canada, 10–15 December 2024; Volume 37.
37. Alayrac, J.B.; et al. Flamingo: A Visual Language Model for Few-Shot Learning. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), New Orleans, LA, USA, 28 November–9 December 2022; Volume 35.
38. Karamcheti, S.; et al. Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models. In Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR, Vienna, Austria, 21–27 July 2024; Volume 235, pp. 23123–23144.
39. Huang, Q.; et al. OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 16–22 June 2024; pp. 13418–13427.
40. Yin, S.; et al. Woodpecker: Hallucination Correction for Multimodal Large Language Models. arXiv 2023, arXiv:2310.16045.
41. Zhou, Y.; et al. Analyzing and Mitigating Object Hallucination in Large Vision-Language Models. In Proceedings of the International Conference on Learning Representations (ICLR), Vienna, Austria, 7–11 May 2024.
42. Yu, T.; et al. RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback. In Proceedings of the CVPR 2024, Seattle, WA, USA, 16–22 June 2024; pp. 13807–13816.
43. Zhao, Z.; et al. Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization. arXiv 2023, arXiv:2311.16839.
44. Vo, A.; et al. Vision Language Models are Biased. arXiv 2025, arXiv:2505.23941.
45. Lin, J. Divergence Measures Based on the Shannon Entropy. IEEE Trans. Inf. Theory 1991, 37, 145–151.
46. Mader, K. Satellite Images of Hurricane Damage. Available online: https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage.
47. Vedantam, R.; et al. CIDEr: Consensus-based Image Description Evaluation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, USA, 7–12 June 2015; pp. 4566–4575.
48. Zhang, T.; et al. BERTScore: Evaluating Text Generation with BERT. In Proceedings of the International Conference on Learning Representations (ICLR), Addis Ababa, Ethiopia, 26–30 April 2020.
49. Hessel, J.; et al. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), Online, 7–11 November 2021; pp. 7514–7528.
50. Yuan, W.; et al. Self-Rewarding Language Models. arXiv 2024, arXiv:2401.10020.
51. Rafailov, R.; et al. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), New Orleans, LA, USA, 10–16 December 2023; Volume 36, pp. 53728–53741.
52. Aaba, A. Wildfire Prediction Dataset (Satellite Images). Kaggle. Derived from Forest Fires—Open Government Portal, Government of Canada. Available online: https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset.
