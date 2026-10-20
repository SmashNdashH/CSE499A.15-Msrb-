# Enhancing Ultra High Resolution Remote Sensing Imagery Analysis with ImageRAG

> **Authors:** Zilun Zhang, Haozhan Shen, Tiancheng Zhao, Zian Guan, Bin Chen, Yuhao Wang, Xu Jia, Yuxiang Cai, Yongheng Shang, Jianwei Yin
> **Published:** IEEE Geoscience and Remote Sensing Magazine (GRSM), 2025
> **arXiv:** https://arxiv.org/abs/2411.07688
> **DOI:** 10.1109/mgrs.2025.3574742
> **Code:** https://github.com/om-ai-lab/ImageRAG
> **Model:** https://huggingface.co/omlab/ImageRAG

---

JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 1

# Enhancing Ultra High Resolution Remote Sensing Imagery Analysis with ImageRAG


Zilun Zhang _†_, Haozhan Shen _†_, Tiancheng Zhao, Zian Guan, Bin Chen,

Yuhao Wang, Xu Jia, Yuxiang Cai, Yongheng Shang, Jianwei Yin



**_Abstract_** **—Ultra** **High** **Resolution** **(UHR)** **remote** **sensing** **im-**
**agery** **(RSI)** **(e.g.** **10,000** _×_ **10,000** **pixels)** **poses** **a** **significant**
**challenge** **for** **current** **Remote** **Sensing** **Vision** **Language** **Models**
**(RSVLMs).** **If** **choose** **to** **resize** **the** **UHR** **image** **to** **standard** **input**
**image** **size,** **the** **extensive** **spatial** **and** **contextual** **information** **that**
**UHR** **images** **contain** **will** **be** **neglected.** **Otherwise,** **the** **original**
**size** **of** **these** **images** **often** **exceeds** **the** **token** **limits** **of** **standard**
**RSVLMs,** **making** **it** **difficult** **to** **process** **the** **entire** **image** **and**
**capture** **long-range** **dependencies** **to** **answer** **the** **query** **based**
**on** **the** **abundant** **visual** **context.** **In** **this** **paper,** **we** **introduce**
**ImageRAG** **for** **RS,** **a** **framework** **to** **address** **the** **complexities**
**of** **analyzing** **UHR** **remote** **sensing** **imagery** **with** **a** **little** **train-**
**ing** **requirement.** **By** **transforming** **UHR** **remote** **sensing** **image**
**analysis** **task** **to** **image’s** **long** **context** **selection** **task,** **we** **design**
**an** **innovative** **image** **contextual** **retrieval** **mechanism** **based** **on**
**the** **Retrieval-Augmented** **Generation** **(RAG)** **technique,** **denoted**
**as** **ImageRAG.** **ImageRAG’s** **core** **innovation** **lies** **in** **its** **ability** **to**
**selectively retrieve and focus on the most relevant portions of the**
**UHR** **image** **as** **visual** **contexts** **that** **pertain** **to** **a** **given** **query.** **Fast**
**path** **and** **slow** **path** **are** **proposed** **in** **this** **framework** **to** **handle**
**this** **task** **efficiently** **and** **effectively.** **ImageRAG** **allows** **RSVLMs**
**to** **manage** **extensive** **context** **and** **spatial** **information** **from** **UHR**
**RSI,** **ensuring** **the** **analysis** **is** **both** **accurate** **and** **efficient.** **Code**
**will** **be** **released** **in** **[https://github.com/om-ai-lab/ImageRAG.](https://github.com/om-ai-lab/ImageRAG)**


I. INTRODUCTION


In the field of remote sensing (RS), ultra-high-resolution
(UHR) images often cover vast areas, encompassing diverse
landscapes and a wide range of geospatial features. For deep
learning applications such as visual question answering, semantic segmentation, object detection, and change detection,
processing these large-scale images directly poses significant
challenges. The high spatial resolution results in massive image sizes, making it difficult to directly train neural networks
end-to-end with such images due to the hardware limitation.
Additionally, the variability in scale, class distribution, and
object sizes within these large images can lead to suboptimal
performance if not handled properly. To address these, a
common preprocessing step is to cut the original UHR images
into smaller patches (e.g. 224 _×_ 224 pixels or 512 _×_ 512 pixels)

[1] [2] that can fit in regular deep learning workflows.

Multimodal Large Language Models (MLLM, in this paper
we specifically refer to generative Vision-Language Models


_†_ : Equal Contribution
Zilun Zhang, Haozhan Shen, Yuhao Wang, Yuxiang Cai, Yongheng Shang,
and Jianwei Yin are with the College of Computer Science and Technology,
Zhejiang University, Hangzhou, China; Tiancheng Zhao is with the Binjiang
Research Institute of Zhejiang University; Bin Chen and Xu Jia are with the
School of Software Engineering of Zhejiang University, Zian Guan is with the
Polytechnic Institute of Zhejiang University. Corresponding author: Jianwei
Yin. E-mail: zilun.zhang@zju.edu.cn; tianchez@zju-bj.com.



Fig. 1: An example of challenging VQA task that requires
analyzing small targets in a high-resolution image. Models
such as GeoChat, InternVL2.5, and _V_ <sup>_∗_</sup> failed to answer.
InternVL2.5 with the aid of ImageRAG and InternVL2.5 with
human-provided visual cue can answer the question correctly.


using Large Language Models as base model) such as Geochat

[3], EarthGPT [4], SkysenseGPT [5], VHM [6], etc. have
demonstrated remarkable potential in RS tasks, including
image captioning, visual grounding, relation reasoning, object
detection, and visual question answering (VQA). However, the
input image resolutions for these Remote Sensing Multimodal
Large Language Models (RSMLLMs) are often limited and
relatively small compared with the original satellite image.
For example, models like LLaVA1.5 [7] and VHM [6] utilize
image inputs of 336 _×_ 336 pixels, while Geochat [3] and SkysenseGPT [5] process images at 504 _×_ 504 pixels. Contrastive
Vision-language models (VLMs) specifically trained for RS,
such as GeoRSCLIP [8] and RemoteCLIP [9], work with even
smaller inputs, typically at 224 _×_ 224 pixels.


Recently, Wang et al. proposed XLRS-Bench [10], a benchmark for evaluating MLLMs on UHR RSI. Li et al. introduced
STAR, a large-scale dataset for scene-graph generation in
very-high-resolution satellite images, containing rich texture
information (more than 400k <subject, relationship, object>
triplets). Zhang et al. presented the MME-RealWorld dataset

[11], which includes a remote-sensing VQA subset with
UHR images. These datasets could be potential benchmarks
for VLMs based large remote sensing image understanding. Regarding specialized models, Luo et al. proposed a


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 2



coarse-to-fine, text-guided token-pruning approach for large
remote-sensing images understanding [12]; however, its performance still leaves room for improvement.

In Figure 1, we provide a qualitative example to illustrate
how current RSMLLMs struggle to answer a challenging
question that requires identifying small objects in a highresolution image. The model’s limitations in handling fine
details and distinguishing small features become evident, leading to inaccurate responses when tasked with analyzing such
intricate visual information (the model can answer correctly
when the zoom-in image is provided).

To quantitatively identify this problem, we designed an
experiment. We tested several RSMLLMs on a remote sensing
subset of the MME-RealWorld benchmark (a VQA dataset
with high-resolution images and tiny targets; details are provided in Section II-A) with 3738 questions. This subset is
denoted as **MME-RealWorld-RS** in our paper. The evaluation
metric is the overall accuracy of regular VQA task (see section
III-A for detail). RSMLLMs include InternVL2.5 (input image
resolution: 448 _×_ 448 pixels) [13], SkysenseGPT (input image
resolution: 504 _×_ 504 pixels) [5], Geochat (input image resolution: 504 _×_ 504 pixels) [3], and VHM (input image resolution:
336 _×_ 336 pixels) [6]. The score for LLaVA1.5 (input image
resolution: 336 _×_ 336 pixels) [7] is listed as a baseline, since
the model structures of Geochat, SkysenseGPT, and VHM
are derived from LLaVA1.5. The input image resolutions for
these models are listed, and the input image will be resized
to such fixed resolutions during training. Dynamic Resolution
(DR) technique from InternVL 1.5 [14] is an input image
preprocessing approach that divides the images into 448 _×_ 448
pixel tiles based on the aspect ratio and resolution of the
input images. An increased number of tiles allows for a
higher degree of image magnification, thereby enabling the
observation of finer details within the image. InternVL2.5
using Dynamic Resolution with a max number of 6 and 12
dynamic tiles are compared as well (InternVL + DR6 and
InternVL + DR12 in the figure).

Figure 2 presents the results of this experiment. **In general,**
**model** **performance** **tends** **to** **improve** **with** **increased** **input**
**image** **resolution** (the Dynamic Resolution technique can
be seen as an enhancement of input image resolution, and
the level of enhancement grows as the maximum number of
dynamic tiles increases). That makes sense because increasing
the input image resolution magnifies tiny objects, allowing for
better detection and analysis of small details in UHR RSI.
InternVL2.5 shows better performance than SkysenseGPT and
GeoChat, even with a smaller input image size. This could be
attributed to the fact that SkysenseGPT and GeoChat utilize
a vision encoder that is pre-trained with an input resolution
of 336 _×_ 336 pixels and interpolate the input image resolution
to 504 _×_ 504 pixels during the Supervised Fine-tuning stage.
In contrast, the vision encoder of InternVL2.5 is trained from
scratch with an input image resolution of 448 _×_ 448 pixels.

The trend of this experiment can be interpreted in another
way: If models resize the input image to a much lower
resolution (compared with the original image size), image
details such as tiny objects will become hard to notice and may
be neglected when the model thinks and generates answers.



50


40


30


20


10


0


|Performance of MLLMs on MME-RealWorld-RS|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||
|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|~~48.72~~<br>47.16<br>40.56|
|~~48.72~~<br>47.16<br>40.56|||||||||||||||
|~~48.72~~<br>47.16<br>40.56|||||||||||||||
|||||||~~29.03~~|~~29.03~~|~~29.03~~|~~29.03~~|~~29.03~~|~~29.03~~|~~29.03~~|~~29.03~~|~~29.03~~|
|||||||||27.55<br>23.31<br>22.12|27.55<br>23.31<br>22.12|27.55<br>23.31<br>22.12|27.55<br>23.31<br>22.12|27.55<br>23.31<br>22.12|27.55<br>23.31<br>22.12|27.55<br>23.31<br>22.12|
|||||||||27.55<br>23.31<br>22.12|||||||
|||||||||27.55<br>23.31<br>22.12|||||||
|||||||||27.55<br>23.31<br>22.12|||||||
||||||||||||||||
||||||||||||||||



InternVL+DR12



InternVL+DR12 InternVL+DR6

448 x 448 448 x 448





InternVL+DR6 InternVL SkySenseGPT

448 x 448 448 x 448 504 x 504



SkySenseGPT GeoChat VHM LLaVA1.5

504 x 504 504 x 504 336 x 336 336 x 336



Fig. 2: The performance of MLLMs in the MME-RealWorldRS benchmark’s remote sensing subset. The specified image
resolutions for model input are listed in the end. “DR”
represents the **Dynamic** **Resolution** technique, with 6 or 12
indicating the maximum number of tiles obtained through
Dynamic Resolution. In general, model performance tends to
improve with increased input image resolution (DR can be
seen as an enhancement of input image resolution)


This makes MLLMs difficult to apply in UHR RSI. We
identify four types of approaches for applying MLLMs to
UHR RSI, each with its own set of limitations.

The first approach involves resizing UHR images to a
smaller size in order to be compatible with current MLLMs.
Most RSMLLMs use a fixed input image resolution and number of visual tokens because they initialize the model weights
with a pretrained MLLM. Take Encoder-MLP-LLM structured
RSMLLMs as an example (e.g., LLaVA-based models such as
Geochat, VHM, and SkysenseGPT), they first resize images to
336 _×_ 336, then divide the images into 576 visual patches
of 14 _×_ 14. Next, these visual patches are projected into
576 visual tokens. These tokens share a unified representation
space with language tokens. Finally, they process the tokens
using an LLM. However, these operations significantly reduces
the visibility of small objects in the images, making them
challenging to detect, even for humans. For instance, VHM

[6] claims its difficulty in handling small objects, likely due
to limitations in input image resolution of 336 _×_ 336.

The second approach divides UHR images into smaller subimages that can be sequentially processed by MLLMs. While
this allows for compatibility with existing model architectures,
it results in the loss of global and relative information and
relationships present in the original large-scale image, as only
portions of the image are considered at a time.

The third approach references techniques from general
LLMs for managing long context, such as Positional Interpolation [15] and LongROPE [16]. Or adopting architecture
from video MLLMs like LongVILA [17], which can extend
the context window effectively. These approach could enable
the integration of entire UHR images while maintaining global
information. However, this would require retraining the models
from scratch and could be limited by LLM’s context length.


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 3







(a) Spatial Relationship Task



(b) Color Recognition Task





(c) Object Counting Task (d) Image Size Statistics of MME-RealWorld-RS


Fig. 3: Visualization of Subtasks from MME-RealWorld-RS Dataset and Statistics of the Images. Image examples are taken
from the Appendix of MME-RealWorld [11] Paper.



The fourth approach employs guided visual search methods
that focus on relevant patches, such as _V_ <sup>_∗_</sup> [18], or hybrid architectures like LongLLaVA [19], which enable the processing
of very large input images, and not neglect the small targets.
Similar to the drawbacks of the third approach, this method
also requires retraining the model and demands task-specific
annotations, adding to the complexity and effort needed.


Three crucial aspects for MLLMs to effectively handle UHR
RSI are: (1) managing small targets, ensuring that the model
can accurately aware and analyze fine details within images;
(2) processing the UHR image in a way that integrates with
MLLMs without significantly increasing the number of image
tokens, which would lead to high computational costs; and (3)
achieving these goals while minimizing the need for additional
training or specialized annotation.


To address these problems, we contribute the **ImageRAG**
framework, which offers several key advantages.


_•_ It retrieves and emphasizes relevant visual context from
the UHR image based on the text query, allowing the
MLLM to focus on **important** **details,** **even** **tiny** **ones** .

_•_ It integrates various **external** **knowledge** **sources** to
guide the model, enhancing the understanding of the
query and the UHR RSI.

_•_ ImageRAG only requires a little amount of training,
making it a practical solution for efficiently handling
UHR RSI.



II. BENCHMARK


_A._ _MME-RealWorld-RS_


**The** **Remote** **Sensing** **subset** **of** **the** **MME-RealWorld**
**benchmark** is designed to evaluate the capabilities of MLLMs
in handling high-resolution images with VQAs (multiple
choice questions). These images are characterized by their
extremely high quality and rich details, making them challenging even for human annotators. The data sources include the
FAIR1M dataset [20], the ISPRS Potsdam dataset <sup>1</sup>, VGoogle

[21], VBing [21], and VArcGIS [21]. These datasets are
sourced from Google Earth, Bing World Imagery, and ArcGIS
World Imagery. It includes sub-tasks such as spatial relationship understanding, color recognition, and object counting.

**Spatial Relationship Understanding** task (1,257 QA pairs)
involves understanding the absolute and relative spatial relationships between objects in the images. **Color** **Recognition**
task (1,226 QA pairs) requires the model to identify and
describe the colors of specific objects in the images. **Object**
**Counting** task (1,255 QA pairs) involves counting specific objects within the images, such as aircraft, vehicles, or buildings.
These tasks are challenging due to the large image size and
small target size, which can be easily overlooked.

A demonstration of MME-RealWorld-RS is presented in
Figure 3, which includes examples of 3 subtasks and dataset
statistics. This dataset comprises 1,265 high-resolution images
that were manually selected from over 70,000 public remote
sensing images. The average image size is 5 _,_ 602 _×_ 4 _,_ 445
pixels across all related images (some images may be repeated


1https://paperswithcode.com/dataset/isprs-potsdam


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 4



in different tasks). There are 3,738 QA pairs in the dataset and
the questions and answers are generated by 20 volunteers manually, while an expert examined the quality of the questions
to ensure they conformed to the required standards.


_B._ _MME-RealWorld-Lite-RS_


The MME-RealWorld-RS dataset is an ideal benchmark for
evaluating the ImageRAG framework due to its distinctive
characteristics. It includes UHR RSI and features extremely
tiny objects, challenging the framework’s ability to detect
and classify small-scale targets effectively and efficiently. In
addition, it contains non-ordinary object classes, offering a
diverse set of objects beyond the typical categories found in
standard remote sensing benchmarks. This diversity ensures
a comprehensive evaluation of the framework’s capabilities.
Lastly, target-of-interest in the question is unique (for Color
Recognition task and Spatial Relationship Understanding
task), simplifying the evaluation process and ensuring clear,
unambiguous answer. These features collectively make MMERealWorld-RS a robust and suitable benchmark for assessing
the ImageRAG framework’s performance.

However, evaluating only the final accuracy for VQA questions is too simplistic. Similar to the RAG framework, we aim
to determine whether ImageRAG can retrieve the correct and
useful visual cues (e.g. image patches) to assist the MLLM
in analyzing and determining the final answer. This requires
the annotation of regions of interest (e.g., 2D coordinates), a
feature that MME-RealWorld-RS does not inherently provide.

To address this issue, we annotated MME-RealWorld-liteRS, which is a RS subset of MME-RealWorld-lite <sup>2</sup> . Similar
to MME-RealWorld-RS, it contains 150 QA pairs, with 50
for each of the three subtasks. **Specifically,** **we** **examined**
**the** **VQA** **triplets** **one-by-one** **and** **labeled** **the** **coordinates**
**of** **the** **rectangular** **Region-of-Interest** **(ROI)** **in** **the** **image,**
**based on the provided question and answer.** The coordinates
are in [ _x_ 1 _, y_ 1 _, x_ 2 _, y_ 2], representing the x-y coordinates of the
top-left and the bottom-right points of the ROI box. One
annotator annotated the box, and two annotators examined
the correctness of the region-of-interest for each VQA triplet.
We established the following guidelines to guide our labeling
process:


_•_ One Region-of-Interest box per VQA triplet.

_•_ The Region-of-Interest box must contain all objects mentioned in the question.

_•_ The Region-of-Interest box should be as small as possible
while still satisfying the previous condition.

_•_ Two annotators must have an agreement on the annotated
Region-of-Interest box, and the uniqueness of target-ofinterest must be checked.
We used the jupyter bbox widget <sup>3</sup> to assist the label process.
During the annotation process, we found that some labels from
MME-RealWorld were incorrect or ambiguous. We corrected
them and provided a correction list in the Appendix A.

Figure 4 presents the distribution of ROI area ratios (ROI
Area Image Size) of MME-RealWorld-Lite-RS. The median


2https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite
3https://github.com/gereleth/jupyter-bbox-widget


|Col1|75|Distribution of ROI Area Ratios (ROI Area / Image Size)|Col4|Col5|Col6|
|---|---|---|---|---|---|
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency|75|||||
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency||||||
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency||||||
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency||||||
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency||||||
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency||26|26|26|26|
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency||||||
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency|||14|14|14|
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency||||||
|10<br>20<br>30<br>40<br>50<br>60<br>70<br>Frequency||||4<br>6<br>3<br>3<br>2<br>3<br>2<br>2<br>1|4<br>6<br>3<br>3<br>2<br>3<br>2<br>2<br>1|



ROI Area Ratio


Fig. 4: Distribution of ROI area ratios (ROI Area / Image Size)


of ROI area ratios is 0.00497, which means the ROI area
usually takes up only a tiny portion of the entire UHR RSI. The
extracted key phrases (from questions in MME-RealWorldLite-RS dataset) can be found in Figure 5.


Fig. 5: Key Phrases Word Cloud


III. TASK


When evaluating ImageRAG using a VQA dataset with
multiple-choice questions, **accuracy** is the most commonly
used metric. However, this metric only **indirectly** reflects the
capability of the ImageRAG model, as it does not directly assess the retrieval results. Accuracy can improve if the retrieved
visual cues are closely related to the question and the MLLM
has the capability to understand and utilize these visual cues
to answer the question. Based on this analysis, we break down
the assessment into two key components: (1) evaluating how
closely the retrieved visual cues align with the ground truth
region-of-interest box, and (2) assessing the ability of MLLMs
to make judgments by inferring visual cues. In summary, we
have three subtasks to evaluate the performance of ImageRAG:
Visual Cue Retrieval Task, Inferring VQA Task, and Regular
VQA task. We will expand former two in next subsections.


_A._ _Regular_ _VQA_ _Task_


The regular VQA task requires models to understand image
content and provide accurate answers to given natural language
questions. In the context of MME-RealWorld-RS and MMERealWorld-lite-RS, models must respond with only a letter (A,
B, C, D, or E) corresponding to the input question and image.
Given a MLLM, a question _Ti_ and image _Ii_, the regular VQA
task can be represented as follow ( _Ri_ is the model output such
as _A, B, C, D, E_ ):










JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 5



_Ri_ = MLLM( _Ii, Ti_ ) (1)


The accuracy (Acc) in the context of the regular VQA task
can be calculated using the following formula:


Acc = <sup><u>#</u></sup> <sup><u>of</u></sup> <sup><u>Questions</u></sup> <sup><u>with</u></sup> <sup><u>Correct</u></sup> <sup><u>Response</u></sup> (2)

# of All Questions


_B._ _Inferring_ _VQA_ _Task_


For ordinary RAG framework, the retrieved context can be
directly organized by a text prompt and fed into the LLM
since they are both in text modality. The generalization ability
of the LLM will automatically make the final decision by
inferring the selected text context as evidence. However, this
is non-trivial for MLLMs, which need to infer multiple visual
contexts (as model input) from image modality and facilitate
decision-making. Usually, MLLMs are not trained for such
a target. To assess the capability of MLLMs in using visual
cues to aid the final decision-making process, we propose the
**Inferring** **VQA** **Task** for MLLMs.

The inferring VQA task, as hinted by the name, is a variant
of the VQA task that involves providing visual context and
prompt to infer. The task can be formalized as follow:


_Ri_ = MLLM( _Ii, Vi, Ti_ _|_ Prompt) (3)


In this paper, the _Vi_ is the ROI box, which is an absolutely
corrected visual cue. Same as VQA task, Overall Accuracy
(of multiple choice questions) is the metric to evaluate the
Inferring VQA Task. MLLM that achieves higher accuracy
with given visual cues is a more effective inferring model.
**Inferring** **model** **using** **ROI** **box** **can** **be** **considered** **as** **an**
**upper** **bound** **of** **ImageRAG** **framework** . The evaluation
metric of inferring VQA task is accuracy. This task is designed
to evaluate the generation stage of ImageRAG framework.


_C._ _Visual_ _Cue_ _Retrieval_ _Task_


Similar to the RAG framework, the goal of the **Visual**
**Cue** **Retrieval** **Task** in ImageRAG is to retrieve useful image
patches as visual cues (evidence) to assist MLLMs (inferring
model) in making accurate decisions. Ideally, the retrieved
visual cues should overlap with the ground truth region-ofinterest box. Therefore, recall@k is chosen as the evaluation
metric of this retrieval subtask.

For each question from the MME-RealWorld-Lite-RS
dataset, the ImageRAG framework outputs at most _k_ visual
cues _{Vi}_ <sup>_k_</sup> _i_ =1 <sup>,</sup> <sup>and</sup> <sup>there</sup> <sup>is</sup> <sup>a</sup> <sup>ground</sup> <sup>truth</sup> <sup>ROI</sup> <sup>box</sup> <sup>(</sup> <sup>_G_</sup> <sup>)</sup> <sup>for</sup>

each question, as we mentioned in section II-B. We define the
**Recall@k** **with** **IoU** **threshold** **T** as follow:



are boxes represented by 4-coordinate [ _x_ 1 _, y_ 1 _, x_ 2 _, y_ 2], and I is
the indicator function <sup>4</sup> depending on the IoU score and IoU
threshold T.



I ( **IoU** ( **Vi** _,_ **G** ) _≥_ _T_ ) =




1 _,_ if **IoU** ( _Vi, G_ ) _≥_ _T_
(5)
0 _,_ otherwise



Finally, the mean recall is calculated by taking the average
of all Recall@k values across all questions in the MMERealWorld-Lite-RS dataset. Mathematically, it can be represented as:



MR = <sup><u>1</u></sup>

_N_



_N_



_n_ =1



Recall@k _n_ (6)



where N is the total number of questions in the dataset, and
Recall@ _kn_ is the Recall@k with IoU threshold T calculated
with the retrieved visual cues and the ROI box of n-th
question. This task is designed to evaluate the retrieval stage
of ImageRAG framework.


IV. OVERVIEW OF THE IMAGERAG FRAMEWORK

















































Recall@k = <sup><u>1</u></sup>

_k_



_k_



_i_ =1



I ( **IoU** ( _Vi, G_ ) _≥_ _T_ ) (4)



Fig. 6: The ImageRAG Framework (up) and Ordinary RAG
(down). **Dashed** **lines** represent the slow path of ImageRAG,
and the **solid lines** represents the fast path. Detail introduction
on ImageRAG can be found in section IV.


As shown in Figure 6 (down), ordinary RAG boosts the
capabilities of LLMs by retrieving and referencing relevant
document chunks from external knowledge database through
the semantic similarity. The process involves two main stages:
Retrieval and Generation. In this way, RAG effectively reduces


4https://en.wikipedia.org/wiki/Indicator_function



Where **IoU** ( _Vi, G_ ) is the regular intersection over union
score between a visual cue _Vi_ and the ROI box _G_, which


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 6



the problem such as generating factually incorrect and out-ofdate content [22]. Moreover, a domain-specialized LLM can
be achieved by equipping the LLM with RAG and a domain
knowledge database [23].

A challenge in applying RAG to UHR RSI is how to
extend RAG to visual modality. It requires VLMs to associate
text and visual embeddings, which may face difficulties in
aligning visual concepts in satellite views with corresponding
text descriptions. Besides, consider image patches as contexts
could result in no visual contexts be found to aid generation.
We will expand this and provide a solution in section IV-A4.

Our ImageRAG framework for RSMLLM references the
idea of RAG, but focusing on retrieving **visual** **contexts** as
evidences for the text query. As shown in Figure 6 (up),
ImageRAG also contains two stages, **Retrieval** **Stage** and
**Generation** **Stage** . We denote a given image as _Ii_ and
corresponding text query (instruction) as _Ti_ . The ImageRAG
framework aims to **retrieve** **a** **set** **of** **relevant** **visual** **contexts**
_Vi_ to augment input and **generate response** _Ri_ . There are two
modes for ImageRAG to work, **fast** **path** and **slow** **path** .


_A._ _Retrieval_ _Stage_


Given an Image _I_, a text query _T_, a **Patch** **Division** **Algo-**
**rithm** _F_, an **Question** **Analyzing** **Module** _G_, a **Text-Image**
**Retrieval** **Module** _M_ text-img (including an image encoder _f_ img,
a text encoder _f_ text, a patch select function _H_ fast), a **Text-**
**Image** **Vector** **Database** _D_ with threshold _δ_, and an **Image-**
**Image** **Retrieval** **Module** _M_ img-img (including image encoder
_f_ img and patch select function _H_ slow). A set of visual cues _V_
can be selected using the following strategy:



_3)_ _Text-Image Retrieval Module:_ **Relevant** **visual** **contexts**
_V_ (a subset of _P_ ) with respect to key phrases _Q_ can be
identified with a Text-Image Retrieval Module _M_ text-img. This
module ensures that the image regions most relevant to the
textual contents are selected, enhancing the model’s ability to
focus on meaningful image areas for the given query.

After the input image and text query are processed by
_F_ and _G_, _n_ key phrases ( _Q_ ) and _m_ image patches ( _P_ )
will be encoded to text and image embeddings (d-dimension
vectors) through a text encoder _f_ text and an image encoder _f_ img,
respectively. The similarity matrix between _n_ text embeddings
and _m_ image embeddings is denoted as _S_ fast ( _S_ fast _∈_ R <sup>_n×m_</sup> )
by calculate the cosine similarities between the embeddings
of key phrases _Q_ and the embeddings of image patches _P_ .
Then, the cosine similarities will be normalized to [0 _,_ 1] rowwise with the softmax function and a scale factor _γ_ [27].


_M_ text = _f_ text( _Q_ ) _,_ where _M_ text _∈_ R <sup>_n×d_</sup> (9)


_M_ img = _f_ text( _Q_ ) _,_ where _M_ img _∈_ R <sup>_n×d_</sup> (10)



_S_ fast = Softmax( _γ · M_ text @ _M_ img <sup>T</sup> <sup>)</sup> (11)


where @ represents matrix multiplication and T represent the
matrix transpose.

Each entry of the _S_ fast matrix is in fact a similarity measurement that indicates how close an image patch and a text
keyphrase are in the embedding space. Visual cues _V_ will
be selected based on similarity matrix _S_ fast using a selection
function _H_ fast with threshold _ϵ_ .


_V_ = _H_ fast( _P, S_ fast _, ϵ_ ) = _{vi}_ <sup>_k_</sup> _i_ =1 (12)


where _k_ means there are _k_ patches satisfied the condition
(e.g. patches with confidence greater than threshold _ϵ_ ) and
selected as visual cue. The **confidence** _ci_ **for** **each** **visual** **cue**
_vi_ is determined by looking up the _S_ fast matrix finding the
corresponding similarity.

If _k_ _>_ 0, the selected _k_ visual cues will send to the MLLM
directly for answer generation, which we call " **fast** **path** ".

_4)_ _Text-Text Retrieval Module and Vector Database:_ If _k_ =
0, a more complicate " **slow** **path** " will be proceed. First, we
introduce the Text-Image Vector Database _D_, which stores
million-scale labeled RSI with the key-value pairs as follows:
the key is the **text** **of** **the** **class** **name** **or** **the** **image** **caption**,
and the value is the **image** **embeddings** obtained using the
image encoder _f_ image with the set of images associated with
that class or caption.

Given a set of query key phrases _Q_ = _{ti}_ <sup>_n_</sup> _i_ =1 <sup>,</sup> <sup>_D_</sup> <sup>retrieves</sup>

the corresponding labels or captions _L_ = _{lp}_ <sup>_k_</sup> _p_ =1 <sup>whose</sup>

text embeddings is close to the query text embedding _M_ text
(i.e. distance between two text embeddings is below certain
threshold _δ_ ). Formally, the retrieval process can be expressed
as:



_V_ =




_M_ text-img( _I, T_ _|_ ( _F, G, f_ img _, f_ text _, H_ fast)) fast path
_M_ img-img( _I, T, D_ _|_ ( _F, G, f_ img _, f_ text _, H_ slow)) slow path



_1)_ _Image_ _Patch_ _Division_ _Approach:_ There are several image patchification approaches. For instance, Vision Transformer [24] divides image into fixed-size, non-overlapping
grids with dimensions such as 32 _×_ 32 or 16 _×_ 16 pixels.
The Swin Transformer [25] uses a hierarchical architecture
to partition the image at multiple scales, employing shifted
windows to capture more diverse contextual information. DetailCLIP [26] introduced the "Complete Cover" method, which
aims to partition the image that patches can cover objects of
any scale. Suppose the patch division approach _F_ outputs _m_
image patches for each image, the patchification process can
be formulated as:


_P_ = _F_ ( _I_ ) = _{pi}_ <sup>_m_</sup> _i_ =1 (7)


Where _P_ denotes a **set** **of** **image** **patches** for image _I_ .

_2)_ _Question_ _Analyzing_ _Module:_ The instruction analyzing
module _G_ processes the input question _T_ and extracts a set of
key phrases _Q_ consisting of _n_ key phrases. The whole process
is entirely based on the input text instruction.


_Q_ = _G_ ( _T_ ) = _{ti}_ <sup>_n_</sup> _i_ =1 (8)



_L_ = _{lp}_ <sup>_k_</sup> _p_ =1 <sup>=</sup> <sup>_D_</sup> <sup>(</sup> <sup>_Q, δ_</sup> <sup>)</sup> (13)



where _lp_ represents a label or caption in the database related
to at least one key phrase from _Q_, and _δ_ is a distance threshold.


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 7



For each label or caption _lp_, there is an associated collection
of _s_ images _Ep_ = _{e_ <sup>_p_</sup> _i_ <sup>_}s_</sup> _i_ =1 <sup>from</sup> <sup>the</sup> <sup>vector</sup> <sup>database.</sup> <sup>We</sup> <sup>then</sup>




<sup>_p_</sup> _i_ <sup>_}s_</sup> _i_



of _s_ images _Ep_ = _{e_ <sup>_p_</sup> _i_ <sup>_}s_</sup> _i_ =1 <sup>from</sup> <sup>the</sup> <sup>vector</sup> <sup>database.</sup> <sup>We</sup> <sup>then</sup>

use a proxy selection function _g_ to select the proxy image
embedding among the _s_ image embeddings. The proxy image
embedding can be expressed as following:



_E_ ˜ _p_ = _g_ ( _Ep_ ) = _g_ ( _{e_ <sup>_p_</sup> _i_



_i_ <sup>_s_</sup> =1 <sup>)</sup> (14)




<sup>_p_</sup> _i_ <sup>_}_</sup> _i_ <sup>_s_</sup>



In summary, a set of key phrases _Q_ will form a set of
relevant visual concepts _E_, where

_E_ = _{E_ <sup>˜</sup> _p}_ <sup>_k_</sup> _p_ =1 (15)


**The** **motivation** **behind** **this** **step** **is** **to** **address** **the**
**potential** **limitations** **of** **the** **Text-Image** **Retrieval** **Module**
_M_ text-img, **which** **may** **not** **fully** **understand** **the** **visual** **con-**
**cept** **of** **certain** **phrases** **in** **the** **domain** **of** **remote** **sensing**
since _M_ text-img could utilize a general VLM rather than RSspecialized one. For instance, the appearance of an aircraft
can vary significantly depending on the perspective, such as a
main view versus an satellite view. **Fast path failure indicates**
**that** **no** **visual** **concept** **has** **been** **found** **for** **the** **key** **phrases**
**with high confidence.** However, it is unlikely that a text query
would target a "void" concept since they are proposed by user.
A plausible explanation is that Text-Image Retrieval Module
_M_ text-img utilizes VLM that is trained on general image-text
pairs, making VLM is difficult to associate visual concepts in
the RS domain with the textual descriptions. To address this,
the **slow** **path** **leverages** **the** **text** **embeddings** **of** **the** **phrases**
**and** **labels** **(or** **captions)** **as** **anchors** **and** **retrieves** **image**
**embeddings** **from** **the** **RS** **domain** **within** **the** **database** **for**
**phrase** **concepts** . These retrieved image embeddings serve as
visual evidence and can be used for image-to-image search
later, thereby enhancing the model’s understanding of domainspecific visual concepts.

_5)_ _Image-Image_ _Retrieval_ _Module:_ When visual evidence
_E_ are obtained, we can calculate the similarity matrix of image
patches _P_ and visual evidence _E_ just like equation 11.



V. IMPLEMENTATION DETAIL FOR IMAGERAG


In this section, we will outline the settings such as choices
of models, parameters, training recipes, etc., along with the
reason behind these choices.


_A._ _Patch_ _Division_ _Algorithm_


We select three patch division algorithms to patchify the
images. The first is the Vision Transformer [24] style (" **ViT** "),
which partitions images into fixed-size, non-overlapping
patches of size m _×_ m. The second is a much denser division
approach called " **Complete** **Cover** " patch from DetailCLIP

[26], which aims to partition images so that patches can cover
objects of any scale, resulting in numerous non-overlapping
multiscale image patches. The third is a compromise between
the former two: a cascade approach (" **Cascade** **Grid** ") that
divides UHR RSIs into 1 _×_ 1, _· · ·_, n _×_ n grids. This results
in overlapping patches at different scales compared to the ViT
style, but not as densely as the DetailCLIP style.

For ViT-style patch division, we set m to 448 _×_ 448,
which matches the input size of InternVL2.5. For DetailCLIPstyle division, we set the scale parameter to _<u>ac</u>_ <sup>=</sup> <sup>20.</sup> <sup>This</sup>
algorithm divides each image into overlapping, multiscaled
image patches. The smallest patch size is approximately 200
_×_ 200 pixels, and about 600 patches are generated per image.
For the "Cascade Grid" approach, we set n to 10, resulting in
385 patches per UHR RSI.


_B._ _Question_ _Analyzing_ _Module_


This module is crucial for the ImageRAG framework, as it
extracts the text of the key elements from the given question,
which we plan to use consistently in later modules.

The Question Analyzing Module G is implemented using
**Qwen2.5-32B** [28], with the _temperature_ set to 1.0, _top_ _ _p_
set to 0.99, and _max_ _ _tokens_ set to 512. The input prompt
can be found in Appendix D. We utilize SGLang [29] to host
QWen2.5-32B for API serving. If a question fails to be parsed
by QWen2.5-32B more than 10 times, we switch the Question
Analyzing Module to **KeyBERT** [30], with an n-gram range
from 2 to 4 and a maximum of 3 phrases. We found that
this implementation can already parse the question ideally,
therefore, no extra technique was applied.

We asked the Question Analyzing Module to extract important keywords or phrases that include adjectives, while
avoiding standalone adjectives or phrases related to position
and orientation since we found they will confuse the MLLM
by hacking the content of choices. Additionally, we aimed
to exclude overly vague words such as "image", "picture",
"photo", etc.


_C._ _Text-Image_ _Retrieval_ _Module_


_1)_ _Selection_ _of_ _Image_ _and_ _Text_ _Encoder:_ We select image
and text encoders from four CLIP-based models: CLIP [27],
RemoteCLIP [9], GeoRSCLIP [8], and MCIPCLIP [31]. The
rationale is as follows:


_•_ **CLIP** : A classic, generalized model suitable for baseline
tasks.



_S_ slow = Softmax( _γ · E_ @ _M_ img



T) (16)



Visual context _V_ for **slow** **path** will be selected based on
similarity matrix _S_ slow using the selection function _H_ slow.

_V_ = _H_ slow( _P, S_ slow) = _{vi}_ <sup>_k_</sup> _i_ =1 (17)


_B._ _Generation_ _Stage_


Once visual cues _V_ are selected from _P_, a set of image
patches from image _I_, MLLM can use such visual contexts for
response generation. Unlike Ordinary RAG, which can directly
organize retrieved text content with a prompt and send to a
LLM to generate the response, ImageRAG must handle visual
context. This means ImageRAG needs to select a MLLM that
can utilize the visual contexts as visual cues.

We designed a training set and trained a MLLM for this
propose to make it be able to accept additional visual cues to
focus on. The final response _Ri_ for given image _Ii_ and text
query _Ti_ with visual cues _Vi_ and specifically designed prompt
will be calculated following equation 3.


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 8




_•_ **RemoteCLIP** : A satellite imagery specialized CLIP variant, making it appropriate for Remote Sensing benchmarks.

_•_ **GeoRSCLIP** : It is trained not only on satellite view data
but also on a large amount of aerial view data, providing
greater diversity in remote sensing tasks.

_•_ **MCIPCLIP** : While CLIP effective at text-image retrieval, it struggles to differentiate between visually distinct images with similar captions, leading to suboptimal
performance in image-based similarity searches. To address this, MCIPCLIP uses an ArcMargin loss [32] and an
Multi-Caption-ArcMargin loss [31], enhancing its imageimage retrieval capabilities through metric learning.
_2)_ _The_ _Similarity_ _Matrix:_ As mentioned in section IV-A3,
the similarity matrix _S_ fast between _n_ key phrases and _m_
image patches is calculated using text and image embeddings
encoded by corresponding encoder. Key phrases are prompted
with the CLIP template <sup>5</sup> and the average text embedding is
taken. A sentence that includes all key phrases along with the
entire image as an image patch are added separately.

_3)_ _Patch_ _Selection_ _Function:_ Given _S_ fast ( _S_ fast _∈_ R <sup>_n×m_</sup> ),
_H_ fast first selects the 2 most frequently appearing image
patches across all key phrases. Then, for each key phrase,
the image patch with the highest similarity is selected. After
removing duplicate image patches, the **top-5** **image** **patches**
**with** **high** **similarity** **are** **retained** . **The** **similarity** **score,**
**ranging** **from** **0** **to** **1,** **for** **each** **image** **patch** **serves** **as** **the**
**confidence** **measure** **for** **that** **patch.** If no image patch is
selected with a confidence score above _ϵ_ (set to 0.5), the
slow path will be triggered. Otherwise the image patches with
confidence above _ϵ_ will be returned.


_D._ _Text-Text_ _Retrieval_ _Module_ _and_ _Vector_ _Database_


Two Text-Image Vector Databases are set up: one that
contains class-wise labeled images related to remote sensing,
referred to as the Labeled Remote Sensing Database (LRSD),
and another that includes RS-related images with captions,
named Captioned Remote Sensing Database (CRSD). Unlike
the simplified definitions in Section IV-A4 and equation 13,
we now expand the definitions definition in practice from _D_
to _D_ 1 and _D_ 2 and adjust the distance threshold from _δ_ to the
respective _δ_ 1 and _δ_ 2 for each vector database.

_1)_ _LRSD:_ We collect images of different instances by cropping objects from commonly used remote sensing datasets,
including those for classification, object detection, and segmentation. DOTAv2.0 [33], FAIR1M [20], iSAID [2], SODAA [34], LoveDA [35], MillionAID [36], and FMoW [37] are
selected. For classification datasets, we use the entire image.
For object detection datasets, we crop the bounding box with
a zoom-out ratio of 1.3 (to include some background of the
object). For segmentation datasets, we convert the segmentation mask to a detection box, and cropping follows the
same method as for detection datasets. We only crop objects
from the training set of each dataset, and a deduplication
process with d-hash <sup>6</sup> is applied. 230,958 duplicate images


5https://github.com/openai/CLIP/blob/main/notebooks/
Prompt_Engineering_for_ImageNet.ipynb

6https://github.com/benhoyt/dhash



are removed. Table I shows the details of LRSD. There are
images from 142 classes in the LRSD. List of class names can
be found in Appendix E.


TABLE I: LRSD Per Dataset Statistics. Average Object Size
is in pixels.


**Dataset** **Task** **Type** **Count** **Average** **Object** **Size**


FMoW Detection 363,572 357.25 _×_ 282.84
DOTAv2.0 Detection 261,277 33.56 _×_ 33.54
FAIR1M Detection 213,107 26.70 _×_ 24.30
iSAID Segmentation 321,235 23.79 _×_ 24.34
SODA-A Segmentation 337,336 22.17 _×_ 21.96
LoveDA Segmentation 92,254 161.88 _×_ 148.98
MillionAID Classification 7,128 543.17 _×_ 543.17
Total     - 1,595,909     

_2)_ _CRSD:_ We use PUB11 subset from RS5M dataset as
the CRSD due to its diversity. There are **3,007,809** RS related
image-text pairs collected from 11 public large-scale English image-text paired datasets, including LAION2B-en [38],
LAION400M [39], LAIONCOCO, COYO700M [40], CC3M

[41], CC12M [42], YFCC15M [43], WIT [44], Redcaps [45],
SBU [46], and Visual Genome [47].

_3)_ _Retrieval_ _Process:_ Chroma <sup>7</sup> is chosen as our vector
database. Label and caption information from LRSD and
CRSD are encoded using Sentence-Bert [48] ("all-MiniLML6-v2" model) and stored in Chroma. The _L_ 2 distance is
employed as the metric for measuring the distance between
two text embeddings in the vector database.

When key phrases are routed to the slow path, their text embeddings (generated by Sentence-Bert) are computed. These
embeddings are then used to search the LRSD for labels with
a close match (i.e., an _L_ 2 distance below _δ_ 1, with _δ_ set to 0.3).
If such labels exist, they and their corresponding images are
returned and the proxy image embedding will be calculated.
If no matches are found in LRSD, the text embeddings are
used to search the more diverse but noisier CRSD, returning
candidates with an _L_ 2 distance below _δ_ 2 (set to 0.5).

_4)_ _Proxy_ _Selection_ _Function:_ As we mentioned in section
IV-A4 and equation 14, we need to calculate proxy image embeddings that represent the visual concepts for each class in the
LRSD database for image-image retrieval. Three approaches
are implemented, which are "Prototype", "Clustering", and
"Reranking".

_•_ **Prototype** : Following the approach of ProtoNet [49], all
image embeddings are normalized first and the mean
image embedding is computed from all images that share
the same class label. This mean image embedding is used
as the proxy image embedding of the class.

_•_ **Clustering** : We first cluster the image embeddings within
the same class using Density-Based Spatial Clustering
(DBSCAN) [50] (scikit-learn implementation <sup>8</sup> ). We set
epsilon to 0.3 and min_samples to 5. Then, we select the
largest cluster and calculate the mean image embeddings
of all images belonging to it. This mean image embedding
serves as the proxy for visual concept representation in
its respective class.


7https://github.com/chroma-core/chroma
8https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 9




_•_ **Reranking** : Reranking is a key technique for enhancing
the quality and relevance of search results in RAG. The
process involves taking the initial set of data retrieved
based on a user’s query and reordering it using advanced
models that can better understand context and semantic
meaning. To implement reranking, we first use the class
label in the vector database to locate the original key
phrase. We then take the corresponding text feature from
the fast path. Next, we calculate the similarity matrix
between the key phrase text feature and the image features
within the label class. We select the top 3 image features
and compute their mean to use as the proxy image feature.


_E._ _Image-Image_ _Retrieval_ _Module_


Similar to the Text-Image Retrieval Module in the fast path
from section V-C2, _S_ slow is calculated in the Image-Image
Retrieval Module. However, unlike _S_ fast, which measures the
similarity between text key phrases and image patches, _S_ slow
calculates the similarity between visual evidence (proxy image
features) and image patches. The ranking process and confidence calculation are consistent with those in the fast path.
_H_ slow selects 3 image patches with the highest confidence.


_F._ _Multimodal_ _Large_ _Language_ _Model_ _with_ _Visual_ _Cues_


Despite their promising capabilities, recent advanced
MLLMs often struggle to link the positional relationships
between the global image and sub-images, where sub-images
are an array of sub-patches from the identical global image [51]. This means that when these visual contents are input
collectively, the current MLLMs cannot accurately localize the
position of the sub-images within the global image. In other
words, MLLMs cannot directly infer the answer based on
visual cues. Moreover, general instruction-tuned MLLMs tend
to exhibit suboptimal performance in specialized professional
domains, such as Remote Sensing [52] [5] and Medical Imaging [53] [54]. Consequently, it is unwise to integrate an opensourced MLLM into our framework without modification.

To address these two issues, we curated a global-local
**visual-cue-aware** **dataset** specialized for the remote sensing
domain to fine-tune general MLLMs, which we call **Zoom4K** .
Specifically, we leveraged the bounding-box annotations of
an _instance_ in the global image from existing off-the-shelf
RS datasets. A 512 _×_ 512 pixels sub-patch containing this
_instance_ and _k_ other arbitrary sub-patches were first cropped
from the global image (use as distractor for robustness consideration). Denoting the bounding boxes of these sub-patches
as _{bgt, b_ 1 _, . . ., bk}_, _I_ as the global image, and _Ibi_ as the subimages cropped from _I_ corresponding to _bi_, we then combined
( _I,_ _R_ ( _{_ ( _bi, Ii_ ) _}i∈{gt,_ 1 _,...,k}_ ) _,_ _Qpos_ ) as the MLLM input.
Here, _Qpos_ represents a question querying the position of
the _instance_ (e.g., "Where is the _{instance}_ in the
picture?"), and _R_ ( _·_ ) is a function used to randomly shuffle
the elements within the array. Finally, the position of the
_instance_ was viewed as the target output of the MLLM.

We categorize the position output as follows: _center,_ _top,_
_bottom, left, right, top-left, top-right, bottom-left, bottom-right,_
_center-left, center-right_ . These categories are determined based









Fig. 7: A demonstration of **Zoom4K** and how to fine-tune
Multimodal Large Language Model with visual cues


on the center coordinates of the bounding box annotation of the
_instance_, as illustrated in Figure 7. The position data, totaling
approximately 4,000 entries, is sourced from the training sets
of DOTA2 [33], SODA [34], and FIT [5] (linked to original
images in STAR [55]). All data are UHR RSI. **Zoom1K** is a
subset of Zoom4K that only uses STAR as its data source.

To further enhance the model’s capabilities, we curated an
additional 10,000 VQA dataset ( **VQA10K** ) for model training,
aiming to strengthen the model’s global-local understanding
while maintaining its VQA capability. Specifically, we used
STAR [55] and FIT [5] to construct the VQA data. These
datasets divide an UHR RSI into a series of small images,
each with 512 _×_ 512 pixels, and annotate each one with several
VQA samples based on its individual visual content. Denoting
the original UHR image as _I_, the bounding box of the tiny
image located in _I_ as _b_, and the visual content of the tiny
image as _Ib_, we combine ( _I,_ ( _b, Ib_ ) _,_ _Q_ ) as the MLLM input,
where _Q_ represents the annotated question in STAR, and the
corresponding annotated answer _A_ is treated as the target
output. We then used the next-token-prediction loss [56] to
train the MLLM on these two constructed position and VQA
datasets. The fine-tuned model obtained through this process
is denoted with the suffix **-Infer** in our paper.

Through this training process, we achieve two key objectives: (1) We endow the model with the ability to perceive the
position of sub-images within the global image, creating an
inferring model that is visual-cue-aware; (2) We inject remote
sensing domain knowledge into the model.

We used Low-Rank Adaptation (LoRA) [57] with an alpha
value of 32 to fine-tune the InternVL2.5-8B model on a mixed
dataset of Zoom4K and VQA10K. This process resulted in the
InternVL2.5-8B-Infer, a model for inferring VQA task.


VI. EXPERIMENT


_A._ _Experiment_ _Setup_


In this section, we present the experimental results for the
tasks introduced in Section III. We select InternVL2.5-8B [13]
as our base model, and MME-RealWorld-lite-RS (introduced
in Section II-B) as the primary benchmark. Other settings are
detailed in Section V. InternVL2.5-8B-Infer is the **MLLM**
**with** **visual** **cues** introduced in section V-F. The prompt
template for inferring model can be found in Appendix C.


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 10


TABLE II: Experimental results of **Regular** **VQA** **Task** on the MME-RealWorld-lite-RS. "Position", "Color", and "Count"
each indicate a specific sub-task domain mentioned in section II-B. The model with the "-Infer" suffix is an inferring model.
MGM-7B is fine-tuned with VRSBench training data. "5-epoch" means the model has been fine-tuned for five epochs. "HighRes Model" means the high-resolution vision encoder is applied, and "Native-Res" is the abbreviation for "Native-Resolution".
"MoE" stands for Mixture-of-Experts architecture. "ROI Box (GT)" indicates the use of ground truth Region-of-Interest boxes
as visual cues. Inferring models using ROI boxes can be seen as the upper bound of the ImageRAG framework.


**<mark>Model</mark>** **<mark>LLM</mark>** **<mark>Image</mark>** **<mark>Encoder</mark>** **<mark>Image</mark>** **<mark>Resolution</mark>** **<mark>Note</mark>** **<mark>Accuracy</mark>**
**<mark>Task</mark>** **<mark>Split</mark>** **<mark>Position</mark>** **<mark>Color</mark>** **<mark>Count</mark>** **<mark>Average</mark>**


**<mark>MME-RealWorld-Lite-RS</mark>**
LLaVA1.5 [7] Vicuna1.5-7B CLIP-ViT-L-14-336px 336 _×_ 336   - 24.00 26.00 18.00 22.67
MGM [58] Vicuna1.5-7B CLIP-ViT-L-14-336px 336 _×_ 336 VRSBench [52] + LoRA 20.00 28.00 28.00 25.33
VHM [59] Vicuna1.5-7B CLIP-ViT-L-14-336px 336 _×_ 336   - 28.00 26.00 28.00 27.33
Geochat [3] Vicuna1.5-7B CLIP-ViT-L-14-336px 504 _×_ 504    - 22.00 36.00 26.00 28.00
SkysenseGPT [5] Vicuna1.5-7B CLIP-ViT-L-14-336px 504 _×_ 504  - 30.00 38.00 30.00 32.67


LLaVA-HR [60] Vicuna1.5-7B CLIP-ViT-L-14-336px 1024 _×_ 1024 High-Res Model 32.00 38.00 16.00 28.67
LLaVA-UHDv2 [61] Qwen2.0-7B CLIP-ViT-L-14-336px 672 _×_ 1008 High-Res Model 46.00 32.00 10.00 29.33
Kimi-VL-A3B [62] MoE-16B MoonViT-SO-400M Native-Res High-Res Model 58.00 48.00 16.00 40.67


InternVL2.5-8B [13] Qwen2.5-7B InternViT-300M-V2.5 448 _×_ 448 Vanilla Model Baseline 56.00 54.00 30.00 46.67
InternVL2.5-8B-Infer Qwen2.5-7B InternViT-300M-V2.5 448 _×_ 448 Inferring Model Baseline 54.00 54.00 **32.00** 46.67
InternVL2.5-8B-Infer Qwen2.5-7B InternViT-300M-V2.5 448 _×_ 448 5-epoch 52.00 52.00 30.00 44.67
<mark>InternVL2.5-8B-Infer</mark> <mark>Qwen2.5-7B</mark> <mark>InternViT-300M-V2.5</mark> <mark>448</mark> _<mark>×</mark>_ <mark>448</mark> **<mark>ImageRAG</mark>** **<mark>64.00</mark>** **<mark>62.00</mark>** <mark>30.00</mark> **<mark>52.00</mark>**


InternVL2.5-8B [13] Qwen2.5-7B InternViT-300M-V2.5 448 _×_ 448 ROI Box (GT) 56.00 68.00 42.00 55.33
InternVL2.5-8B-Infer Qwen2.5-7B InternViT-300M-V2.5 448 _×_ 448 ROI Box (GT) 58.00 78.00 44.00 60.00



Unless specified, we set _ϵ_ to 0.5, _δ_ 1 to 0.3, and _δ_ 2 to 0.5.
The Patch Division Algorithm is chosen as "Cascade Grid",
the Image and Text Encoder are selected from "CLIP", and
the Proxy Selection Function adopts "Clustering". The random
seeds across all experiments are set to 2024.


_B._ _Result_ _of_ _Regular_ _VQA_ _task_


We compare the InternVL2.5-8B-Infer model with other
well known RSMLLMs including Mini-Gemeni [58] (finetuned in VRSBench [52] using LoRA), VHM-7B [59],
Geochat [3], SkysenseGPT [5]. Competitive high-resolution
MLLMs are compared as well, such as LLaVA-HR [60],
LLaVA-UHDv2 [61], and Kimi-VL-A3B-Instruct [62] (An
16B Mixture-of-Experts MLLM with 2.8B activate parameters
released in April 2025. Notably, its vision encoder supports
native resolution.) All compared models use their individual
conversation template and the question template from MMERealWorld dataset [11] (can be found in Appendix B. Table
II shows the result of different MLLMs and InternVL2.5-8BInfer model in MME-RealWorld-lite-RS dataset. Base LLM,
image encoder, supported image resolution are listed.

The InternVL2.5-8B model exhibits remarkable performance in the Regular VQA task on the MME-RealWorldlite-RS dataset, significantly surpassing other MLLMs with
an average accuracy increase of 6% to 24%. As detailed
in Table VI-B, it is evident that both fine-tuning MLLMs
with remote sensing SFT data (data-centric approach) and
enhancing the vision encoder’s supported image resolution of
MLLMs (model-centric approach) are equally effective for the
Regular VQA task involving high-resolution remote sensing
images. When compared to LLaVA1.5, an MLLM lacking
support for high-resolution image input and fine-tuning with
remote sensing data, the data-centric approach, such as that
employed by Geochat and SkysenseGPT, can elevate results



from 22.67% to 32.67%. On the other hand, the model-centric
approach, as seen in LLaVA-HR and LLaVA-UHDv2, can
enhance results from 22.67% to 29.33%. Furthermore, a more
powerful LLM backbone is also vital. For instance, KimiVL-A3B, equipped with a 16B MoE language model, and
InternVL2.5-8B, with a Qwen2.5-7B language model, both
demonstrate an average accuracy of over 40%.

The InternVL2.5-8B-Infer model (inferring model baseline),
with no ground truth ROI boxes provided, maintains performance comparable to the vanilla InternVL2.5-8B model
baseline (46.67% v.s. 46.67%). This suggests that our finetuning process to develop an inferring model capable of
reasoning with visual cues does not compromise the integrity
or effectiveness of the original model. However, when the
model is fine-tuned for more epochs (e.g. 5 epochs), there is a
noticeable decline in performance ( **-2%** ). This contrasts with
results presented in Table III, where even with 5 epochs of
fine-tuning, the model shows improved performance ( **+2%** )
when provided with ground truth ROI boxes. This indicates
that additional training steps may cause the model to become
more reliant on the provided visual cues, potentially reducing
its ability to generalize or reason effectively without accurate
ROI information.

The InternVL2.5-8B-Infer model, when provided with
ground truth ROI boxes, establishes a **performance** **upper**
**bound** for this task (60%). As shown in Table II, the application of the ImageRAG method leads to a significant performance boost, raising the accuracy from a baseline (both vanilla
model and inferring model) of 46.67% to 52% ( **+5.33%** ).
This substantial improvement highlights the effectiveness of
the ImageRAG technique in enhancing the model’s ability
to reason with retrieved visual cues. However, despite this
notable enhancement, there remains a gap between the current
performance and the upper bound set by the ground truth ROI
box. This indicates that while ImageRAG is effective, there is


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 11


TABLE III: Experimental results of **Inferring** **VQA** **Task** on the MME-RealWorld-lite-RS. "Position", "Color", and "Count"
each indicate a specific sub-task domain mentioned in section II-B. The model with the "-Infer" suffix is an inferring model.
"5-epoch" means the inferring model has been fine-tuned for five epochs. "ROI Box (GT)" indicates the use of ground truth
Region-of-Interest boxes as visual cues. "Random Box" means a random box from UHR image is given for the model. Zoom1K,
Zoom4K, and VQA10k are the training set mentioned in section V-F, and "CoT" represents the Chain-of-Thought technique.


**<mark>Model</mark>** **<mark>Technique</mark>** **<mark>Fine-tune</mark>** **<mark>Data</mark>** **<mark>Image</mark>** **<mark>Resolution</mark>** **<mark>Accuracy</mark>**
**<mark>Task</mark>** **<mark>Split</mark>** **<mark>Position</mark>** **<mark>Color</mark>** **<mark>Count</mark>** **<mark>Average</mark>**
InternVL2.5-8B [13] Vanilla Model Baseline    - 448 _×_ 448 56.00 54.00 30.00 46.67
InternVL2.5-8B-Infer Inferring Model Baseline Zoom4K + VQA10K 448 _×_ 448 54.00 54.00 32.00 46.67
InternVL2.5-8B [13] ROI Box (GT)    - 448 _×_ 448 56.00 68.00 42.00 55.33
InternVL2.5-8B-Infer ROI Box (GT) Zoom1K 448 _×_ 448 52.00 76.00 40.00 56.00
InternVL2.5-8B-Infer ROI Box (GT) Zoom4K 448 _×_ 448 54.00 72.00 46.00 57.33
InternVL2.5-8B-Infer ROI Box (GT) Zoom4K + VQA10K 448 _×_ 448 58.00 78.00 44.00 60.00
InternVL2.5-8B-Infer Random Box Zoom4K + VQA10K 448 _×_ 448 46.00 42.00 18.00 35.33
InternVL2.5-8B-Infer ROI Box (GT) + 5-epoch Zoom4K + VQA10K 448 _×_ 448 **68.00** 76.00 42.00 62.00
InternVL2.5-8B-Infer ROI Box (GT) + CoT Zoom4K + VQA10K 448 _×_ 448 62.00 **82.00** **50.00** **64.67**



still room for improvement and optimization to reach the full
potential of the model.


_C._ _Result_ _of_ _Inferring_ _VQA_ _Task_


In Table III, we provides a detailed comparison of various
techniques for the inferring VQA task. Compared with vanilla
model baseline, introducing ROI box (can be considered as
an image patch contains question related visual cues) with
text prompt in zero-shot manner already significantly boosts
Color accuracy from 54% to 68%, while maintaining Position
accuracy (56%) and improving Counting accuracy to 42%,
resulting in an overall performance increase of **8.66%** . This
indicates that using correct visual cues in a training-free model
can already enhance its ability to understand color, position
and counting questions, due to better localization of relevant
visual features.

Further fine-tuning with Zoom1K enhances Color accuracy
(from 68% to 76%), but it negatively impacts the Position and
Count tasks. This decline is reversed when the training data is
increased from 1K to 4K. Additionally, incorporating VQAlike data (QA pairs with UHR RSI) significantly boosts all
subtasks, achieving an average accuracy of 60% ( **+13.33%** ).
Adding the zero-shot Chain-of-Thought (CoT) [63] reasoning
technique yields the highest overall performance, with Position
accuracy of 62%, Color accuracy of 82%, Count accuracy of
50%, and an average of 64.67% ( **+18%** ).
These results highlight the effectiveness of visual-cueaware inferring model in improving VQA performance. The
best-performing model demonstrates significant improvements
across all subtasks, indicating that these techniques effectively
address the challenges of the inferring VQA task.


_D._ _Result_ _of_ _Visual_ _Cue_ _Retrieval_ _Task_


In this section, we aim to verify if ImageRAG can retrieve
useful visual cues by assessing the overlap between these cues
and the ground truth ROI box. Unlike our approach in the
main experiment (section VI-B), where we filtered visual cues
based on confidence and _ϵ_, potentially resulting in no visual
cue output if the confidences were too low, we now force the
model to output visual cues even if they do not meet the _ϵ_
threshold. Mean Recall is calculated following equation 6.



TABLE IV: Experimental results of **Visual** **Cue** **Retrieval**
**Task** on the MME-RealWorld-lite-RS. "Mode" indicates ImageRAG use fast path only or mix of fast path and slow
path mentioned in section IV-A. "Vector Database" represents
the usage of LRSD and CRSD mentioned in section V-D,
and "Mean Recall" indicate the average of all Recall@3 from
equation 6. "T" is the IoU threshold from equation 5.


**<mark>Mode</mark>** **<mark>Vector</mark>** **<mark>Database</mark>** **<mark>MR</mark>** **<mark>(T=0.1)</mark>** **<mark>MR</mark>** **<mark>(T=0.3)</mark>**
Fast Path    - 16.89 4.44
Fast + Slow Path LRSD 17.11 5.33
Fast + Slow Path CRSD 17.11 5.11
Fast + Slow Path LRSD + CRSD **17.33** **5.33**


From Table IV, we can see that introducing the slow path
mechanism boosts the mean recall for both T=0.1 and T=0.3,
with even higher mean recall at T=0.1 when both LRSD and
CRSD are used. However, the result is not significant because
we force the model to output visual cues even if they do not
meet the _ϵ_ threshold, which hurt the result under this metric.
However, if we did not use this metric, the comparison of the
mean recall will not be fair. The reason is, in practice, we find
that both the vanilla and inferring models are sensitive to the
accuracy of the retrieved visual cues. Inaccurate visual cues
can sometimes lead to worse ImageRAG performance than
the baseline. In Table III, we demonstrate this phenomenon: a
noticeable ( **-26.67%** ) decline is happened when we replace the
ground truth ROI box with a random box. Therefore, during
deployment, we adopt a conservative strategy: we do not
provide the visual cue if the retrieved result is not believable
( **providing** **incorrect** **information** **is** **worse** **than** **providing**
**none** ). That is, we filter out visual cues with lower confidence,
which enhances the overall framework’s accuracy.


When slow path is disabled, fast path is used less frequently
(10.67% utilization). The model tends to answer questions
without ImageRAG, in a zero-shot manner. As fast path is
uncertain, the framework often outputs void visual cues. However, when slow path is applied, the model uses ImageRAG
more frequently to answer questions (45.33% utilization).


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 12



_E._ _Result_ _Robustness_ _on_ _Larger_ _Dataset_


In Section VI-B, ImageRAG showed remarkable performance on the MME-RealWorld-Lite-RS dataset. However, this
dataset is relatively small, containing only 150 samples in
total, with 50 samples per subtask. To test how ImageRAG
performs with larger datasets, we created a series of datasets
with different volumes.

We add 360 more samples (120 per subtask) from MMERealWorld-RS, following the labeling standards outlined in
section II-B. These new samples do not overlap with those
in MME-RealWorld-Lite-RS and were examined by human
to make sure the question and answer are not ambiguous
nor vague. We refer to the original MME-RealWorld-Lite-RS
dataset as D150. We also present D300, D450, and D510, each
with an equal number of samples for each subtask. Moreover,
each smaller dataset is a subset of the larger ones. This means
D150 _⊆_ D300 _⊆_ D450 _⊆_ D510.


TABLE V: Performance of ImageRAG on larger datasets.


D150 **64.00** 62.00 30.00 **52.00**
D300 54.00 **65.00** **33.00** 50.67
D450 58.67 63.33 32.00 51.33
D510 59.41 64.12 32.35 51.96


As shown in Table V, ImageRAG demonstrates consistent
performance across datasets of varying sizes, indicating that it
is robust on larger datasets.


VII. ABLATION STUDY


In this ablation study, we examine the key factors that influence ImageRAG’s performance, such as the Patch Division Algorithm, the Image and Text Encoder, and the Proxy Selection
Function. We use an ensemble of results choose from different
thresholds to study: _ϵ_ _∈{_ 0 _._ 3 _,_ 0 _._ 5 _,_ 0 _._ 7 _}_, _δ_ 1 _∈{_ 0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 3 _}_,
and _δ_ 2 _∈{_ 0 _._ 3 _,_ 0 _._ 5 _,_ 0 _._ 7 _}_ . The rationale behind this setting is we
aim to limit the return of inaccurate visual cues, particularly
for _δ_ 1, as it controls the output from querying LRSD database.
Additionally, we explore how the performance of the inferring
model changes when the ROI size is enlarged, which may
cause the evidence to become more vague.


_A._ _Encoding_ _Models_


As mentioned in Section V-C1, we chose CLIP, RemoteCLIP, GeoRSCLIP, and MCIPCLIP for comparison. Results
are presented in Figure 8. Notably, the CLIP model achieves
the highest average and maximum accuracy, outperforming
two models fine-tuned on remote sensing data (GeoRSCLIP
and RemoteCLIP). Despite being designed for image-image
retrieval task (crucial in the slow Path), MCIPCLIP does
not show the expected performance improvement in average
performance.


_B._ _Patch_ _Division_ _Algorithm_


Table VI presents the performance of different patch division algorithms. "Complete Cover" achieves the highest



_D._ _ROI_ _size_ _meets_ _MLLM_ _with_ _visual_ _cues_


To study how enlarging the ROI size affects the inferring
model’s performance, we expanded the width and height of the
ROI box using multipliers from 1 to 6, significantly increasing
the ROI area while keeping its center unchanged. As shown
in Figure 9, the overall accuracy of the regular VQA task
declined substantially with larger ROI sizes. This is because






















JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 13



61


60


59


58


57


56


55


54


53


|60.|00|Col3|Col4|Per|Per|ception Score (Avera|ge)|
|---|---|---|---|---|---|---|---|
|60.|00<br><br>|||<br>Per|<br>Per|||
||~~59~~|~~33~~||||||
|||||||||
|||||||||
|||56.|00|||||
|||||||||
||||54.|00<br>|00<br>|||
|||||~~53~~|~~53~~|~~33~~<br>~~53~~|~~33~~|
|||||~~53~~|~~53~~|||



1 2 3 4 5 6
ROI Box Scale Ratio







Fig. 9: A demonstration of **Zoom4K** and how to fine-tune
Multimodal Large Language Model with visual cues


the expanded visual cues become less precise and lose critical
details, making them less effective for the inferring model.
This experiment demonstrates that we shouldn’t provide the
inferring model with overly large visual cues, even though
they contain the ground truth ROI box.


VIII. RESULT VISUALIZATION AND DISCUSSION


In Figures 10 and 11, we visualize correct and incorrect
responses with high-quality retrieval results. For each QA
pair, we list the questions and key phrases from the Question
Analyzing Module. We also include the ground truth ROI box
and the image patch with the maximum IoU in relation to the
ROI box. For each question, the top-3 retrieved visual cues
are displayed alongside their confidence scores.

From both figures, we can see that the Question Analyzing
Module does a good job of extracting key phrases from
questions. The Image-Text retrieval module and Image-Image
retrieval module can obtain useful visual cues, some of which
even overlap with the best available patch (the one with
the largest IoU with the ROI box among all patches). The
ImageRAG framework can identify rare concepts in remote
sensing, such as lighthouses, excavators, coastal structures,
even **Gorillas** . All the retrieved objects are extremely small
in the original UHR RSI.

In Figure 10, the rank 1 retrieved results are sufficient for
directly answering the Color and Counting questions. The inferring model can interpret the positional relationship between
local and global images when coordinates are provided. The
visual connection of the visual cues to the question aligns with
their confidence scores.

Figure 11 is particularly interesting. In the case of the
triangular roundabout, the target can be located precisely,
with the patch fitting tightly with the ROI box. However,
despite this accurate visual cue, the inferring model fails
to answer correctly, suggesting room for improvement in
its positional inferring capabilities. Regarding the Gorillas
example, ImageRAG retrieves the correct patch, though not
perfectly aligned and slightly zoomed out. Yet, the model
cannot overcome the trap in positional descriptions (the correct
answer is "At the top of the Y-shaped intersection," which



actually refers to the bottom part of this Y-shaped intersection).
Moreover, even with a perfect patch, the framework doesn’t
further enhance the inferring model’s counting ability, as it
fails to count "red cars" although the best patch is found.


IX. COMPUTATIONAL RESOURCE


The LoRA fine-tuning process took 2 hours for 2 epochs
using 8 NVIDIA A100-80GB GPUs <sup>9</sup>, and hosting Qwen-32BInstruct required 4 NVIDIA A100-40GB GPUs. As for running ImageRAG, a single NVIDIA A100-40GB or NVIDIA
RTX 4090-24GB would be enough.


X. A COOKBOOK FOR ADAPTING IMAGERAG TO

DIFFERENT IMAGE MODALITIES AND DOMAINS


In this paper, we focus on the optical ultra high resolution
remote sensing image, but ImageRAG is a geneal framework
that is able to be applied in different UHR image modalities
(e.g. SAR and Hyperspectral data), and image domains (e.g.
Medical Imaging). Several modules can be redesigned to better
fit the specific task, and some others can remain invariant.


_A._ _Patchify_


In section V-A, we introduced three selected patch division
algorithm: "ViT", "Complete Cover", "Cascade Grid". These
are model-agnostic approaches, which means it can be adopted
to different image modalities and domains without any modification or retraining.


_B._ _Question_ _Analyzing_ _Module_


We deploy Qwen2.5-32B on SGLang for API serving
to extract the target-of-interest from questions, ensuring the
extracted key phrases are as accurate as possible. However,
since the target-of-interest extraction task is relatively simple,
this model could potentially be replaced by lighter general
LLMs such as MiniCPM3-4B [64] and Qwen3-0.6B in realworld deployment scenarios, or domain-specific LLM like
Med-PaLM 2 [65] when dealing with UHR medical images.


_C._ _Text-Image_ _Retrieval_ _Module_


This module is of critical importance to the ImageRAG
framework, since image patches, target-of-interest text, image
data from vector database are all encoded as feature vectors
using the VLM within this module. More importantly, the
visual cues are retrieved based on the similarities between
these features. In our study, we tried domain-specific contrastive VLMs (RemoteCLIP, GeoRSCLIP) and general VLMs
(CLIP, MCIPCLIP) for UHR optical remote sensing images.
For encoding SAR, Multispectral, infrared or hyperspectral
imaging alongside corresponding text, GeoLangBind [66] can
be considered. As for medical imaging from other image
domain, MedCLIP [67] can be chosen.


9https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 14

































Fig. 10: Good retrieval results with correct responses.



_D._ _Text-Text_ _Retrieval_ _Module_ _and_ _Vector_ _Database_


This module matches the data from the vector database
with the key phrases extracted from the question. SentenceBERT is generally suitable for text-text retrieval. When dealing
with modality-specific or domain-specific data, it is advisable
to collect images accompanied by text labels or captions to
construct the vector database effectively. In scenarios where
access the image-text paired data is limited, the calculation
of proxy image embeddings can be replaced by leveraging all
available data through in-context few-shot learning [68], which
involves constructing task examples alongside the question to
send to the VLM. This approach aligns with the ultimate goal
of the slow path, which is to select useful image patches with
the aid of external data as visual cues to send to the MLLM.



XI. LIMITATION


The ImageRAG framework has two major limitations. First,
the time cost is significant. Second, the performance heavily
depends on the accuracy of retrieved visual cues.

As shown in Table VIII, we compared the relative time cost
of ImageRAG across different strategies. For fair comparison,
we pre-extracted all patches and their corresponding features
and used them as caches (This process is parallelized using
Ray <sup>10</sup> with multiple GPUs). Vector Databases are pre-built.
Compared with the baseline setting, which does not use
ImageRAG, the additional time cost primarily comes from
question analyzing and visual cue retrieval. When using the
fast path for all questions, the time cost doubles (1.14 seconds
compared to 2.70 seconds). If all questions require the worst

10https://github.com/ray-project/ray


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 15































Fig. 11: Good retrieval results with incorrect responses.



TABLE VIII: We compared the time cost of ImageRAG across
different strategies. "Mode" shows whether ImageRAG used
only the fast path or a combination of fast and slow paths as
detailed in Section IV-A. "Vector Database" reflects if LRSD
and CRSD were used as described in Section V-D. "Time
Cost" represents the average processing time per question in
seconds. All patches and their corresponding features are preextracted and used as caches. Vector Databases are pre-built.


**<mark>Mode</mark>** **<mark>Vector</mark>** **<mark>Database</mark>** **<mark>Time</mark>** **<mark>Cost</mark>** **<mark>Note</mark>**
Baseline    - 1.14 sec Direct Inference
Fast Path Only  - 2.70 sec All Fast Path
Fast + Slow Path LRSD 2.89 sec All Slow Path
Fast + Slow Path LRSD + CRSD 3.02 sec All Slow Path
Fast + Slow Path LRSD + CRSD 2.81 sec Fast & Slow Path



case scenario—where the fast path fails and the slow path
must search both vector databases to find visual cues—the
time cost is almost tripled (1.14 seconds compared to 3.02
seconds). In practical scenarios, when mixing fast and slow
paths across the dataset, the time cost is nearly 2.5 times higher
than the baseline (1.14 seconds compared to 2.81 seconds).
However, the absolute time cost should be acceptable in
practical development as the analysis processing time remains
measured in seconds. To further enhance speed, the Qwen2.532B model used for question processing can be replaced with
a lighter model as mentioned in section X-B.

Moreover, as shown in Section VI-C and Section VII-D,
the performance of ImageRAG is significantly affected by
the accuracy and size of the retrieved visual cues. When
the retrieved visual cues are overly large or incorrect, the


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 16



performance of ImageRAG can greatly decline (Figure 9) and
may even drop below the baseline (as seen in the "Random
Box" row of Table III). Techniques referenced in section XII-C
that improved the accuracy of the retrieved results of RAG can
also be applied to mitigate this issue.


XII. RELATED WORK


_A._ _Remote_ _Sensing_ _meets_ _Vision-Language_ _Models_


Remote Sensing Vision-Language Models (RSVLMs) are
developed to analyze geospatial data by incorporating both
visual and linguistic information [69]. These models are pretrained on large-scale RSI and relevant text, enabling them to
adapt to various VLM tasks in RS domains. RSVLMS have
demonstrated promising results in specialized applications,
including scene classification [4], [9], object detection [3], [5],
semantic segmentation [70], [71], image captioning [3], [4],
text-image retrieval [8], [9], visual grounding [3], [5], [52],
image generation [8], [72], [73], etc. Many of them are capable
to complete multiple downstream tasks.

RSVLMs can be categorized into three types based on their
input-output mechanisms [74]. Contrastive RSVLMs process
both text and images as inputs, generating similarity scores
essential for tasks such as image-text retrieval and zero-shot
scene classification. Conversational (generative) RSVLMs also
take text and images as inputs but produce textual responses,
leveraging LLMs for tasks like captioning and visual question
answering. Besides, some RSVLMs are conditioned on either
text or images to generate synthetic remote sensing images,
typically employing conditional diffusion processes for controlled image synthesis.

For contrastive RSVLMs, research primarily focuses on extending CLIP [27] to RS applications, emphasizing the development of RS-specific datasets and benchmarks. Liu et al. proposed RemoteCLIP [9], a vision-language foundation model
for remote sensing that achieves significant improvements
in various downstream tasks through multi-task pre-training
and data expansion. Zhang et al. developed GeoRSCLIP [8]
based on the large-scale remote sensing image-text dataset
RS5M, which demonstrates excellent performance in zeroshot classification, cross-modal text-image retrieval, and semantic localization tasks. Mall et al. introduced GRAFT [75],
a method to train vision-language models for satellite images without textual annotations by using ground images as
an intermediary. Wang et al. constructed SkyScript [76], a
large and semantically diverse vision-language dataset for
remote sensing, and developed SkyCLIP through continual
pre-training, which shows superior performance in zero-shot
scene classification, fine-grained attribute classification, and
cross-modal retrieval tasks.

A typical conversational RSVLM comprises three main
components: a pre-trained visual encoder, a pre-trained LLM,
and a modality interface that connects them. RSGPT [77] constructs a high-quality remote sensing vision-language model
by leveraging large-scale image-text pairs for pre-training,
focusing on image captioning and visual question answering
tasks. GeoChat [3] develops a novel multimodal framework



that unifies various remote sensing tasks through visual perception and language model alignment, achieving state-of-theart performance on multiple benchmarks. SkyEyeGPT [78]
proposes a unified multimodal framework that leverages instruction tuning to handle diverse remote sensing tasks, showing superior performance across different granularity levels.
EarthGPT [4] creates a versatile multimodal large language
model tailored for remote sensing, integrating multi-sensor
data and multi-task instruction tuning to excel in tasks.
LHRS-Bot [79] builds large-scale remote sensing datasets and
employs a multi-level vision-language alignment strategy to
enhance model performance, establishing a comprehensive
benchmark for evaluating multimodal models in the remote
sensing domain. RS-CapRet [80] employs a frozen LLM and
a contrastively trained vision encoder for efficient fine-tuning
in remote sensing image captioning and retrieval. VHM [6]
enhances remote sensing analysis with rich-captioned and
honest-question datasets, improving factual consistency in
vision-language tasks. SkySenseGPT [5] is optimized for finegrained relation understanding, leveraging the large-scale FITRS instruction dataset.

Generative foundation models, particularly diffusion models, constitute a fundamental class in image generation tasks
and offer significant potential for advancing remote sensing
applications. DiffusionSat [72] presents a large-scale generative foundation model for satellite imagery, integrating text
and metadata conditioning to enable high-resolution generation, super-resolution, temporal generation, and inpainting,
outperforming previous models. CRS-Diff [81] introduces a
controllable remote sensing image generation model based on
diffusion models, incorporating multi-modal conditioning to
achieve precise, high-quality, and flexible RS image synthesis.
Changen2 [82] proposes a generative change foundation model
for remote sensing change detection, leveraging a probabilistic
graphical model and diffusion transformers to synthesize realistic multi-temporal change data, enabling zero-shot change
detection and improved transferability.


_B._ _Vision-Language_ _Model_ _with_ _Visual_ _Cues_


In our default setting, we adopt the approach for the VQA
LLM similar to _V_ <sup>_∗_</sup> [83] and Zoom Eye [51], emphasizing the
perception of visual cues by incorporating their sub-patches
after considering the global view of the entire image.

Apart from this, a cascade of alternative methods could
guide the attention of MLLMs to specific regions within an
image. For instance, the VQA LLM could be substituted with
training-free models like ControlMLLM [84], which assigns
attention scores to a particular region of the image, thereby directing the model’s focus to those localized areas. In addition,
methods that integrate coordinates of bounding boxes, points,
or even visual features from local regions as supplementary
visual cues may be employed to enhance the model’s regional
awareness. Notable examples of such approaches include
Kosmos-2 [85], Shikra [86], and Ferretv2 [87].

Furthermore, conventional MLLMs typically rely on patchlevel vision encoders to encode the input image (e.g., Vision Transformer [88]), which may perform suboptimally in


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 17



regional information processing. Inspired by related works
on region-based vision-language pre-training [89], [90] that
achieves superior location understanding in virtue of a pretrained object detector [91], we could incorporate visual features extracted from the recent outstanding object detectors

[92], [93] into our VQA LLM to enhance it with objectenteric visual cues understanding. A similar approach has been
explored in ChatRex [94].


_C._ _Retrieval-Augmented_ _Generation_


Retrieval-Augmented Generation (RAG) addresses the limitations of traditional generative models in handling specialized or long-tail knowledge. Early models like GPT, trained
on vast corpora, excel at general queries but struggle with
domain-specific or rare information, often generating hallucinations [95]. RAG, introduced by Facebook AI Research in
2020 [96], enhances generative models by integrating realtime document retrieval, improving accuracy and contextual
grounding. Gao et al. [97] categorize RAG into Naive, Advanced, and Modular paradigms, detailing key components
like retrievers, generators, and augmentation methods. A
comparative study by Ovadia et al. [98] shows that RAG
outperforms unsupervised fine-tuning, particularly in scenarios
involving new or unseen knowledge, underscoring its superiority in knowledge injection and model adaptation.

The effectiveness of RAG systems heavily depends on the
quality and relevance of the retrieved knowledge, which directly influences the accuracy and factual grounding of generated content. To enhance retrieval efficiency and overcome the
limitations of traditional methods, several advancements have
been proposed, particularly for zero-shot and few-shot retrieval
tasks. Techniques such as HyDE [99] and REINA [100] utilize
LLMs to generate hypothetical documents, improving retrieval
performance without requiring labeled data. The RewriteRetrieve-Read [101] framework introduces a query rewriting
step, allowing the input query to be better aligned with
retrieval modules. By using reinforcement learning to adapt
queries, R3 enhances retrieval quality, improving performance
in open-domain and multiple-choice question answering tasks.
Promptagator [102] demonstrates the effectiveness of few-shot
learning in dense retrieval, utilizing LLMs to generate synthetic training data from minimal examples, surpassing models
trained on large-scale datasets like MS MARCO. This underscores the viability of few-shot learning and LLM-generated
synthetic data in resource-constrained settings. To bridge the
preference gap between retrievers and LLMs, Zixuan Ke et
al. [103] introduce the BGM framework, which employs a
sequence-to-sequence model to align retrieved information
with LLM preferences.

Methods like RECITE [104] and ITER-RETGEN [105]
focus on improving factual accuracy and integrating retrieved
knowledge, ensuring grounded, accurate responses through
knowledge recitation and iterative retrieval-generation. Frameworks such as Selfmem [106] and Self-RAG [107] enhance
factual consistency via self-reflection. Techniques like StepBack Prompting [108] improve reasoning by guiding LLMs
to abstract concepts. The GENREAD approach [109] replaces



traditional retrieval with LLM-generated contextual documents, demonstrating superior performance in knowledgeintensive tasks like open-domain QA and fact-checking.

Iterative and active retrieval-generation strategies aim to
dynamically enhance both retrieval and generation processes.
ITER-RETGEN [105] alternates between retrieval and generation to improve response quality, while FLARE [110] predicts
and adapts to future information needs during generation.
These methods boost relevance and accuracy in knowledgeintensive tasks. The Adaptive-RAG framework [111] selects
retrieval strategies based on query complexity, outperforming
static models in efficiency and accuracy. FunnelRAG [112]
proposes a progressive retrieval paradigm with coarse-to-fine
granularity for RAG, to enable load balancing and improve
retrieval performance.

Reducing hallucinations in response is a critical challenge
in RAG systems. To address this, various strategies focus
on improving the trustworthiness of generated outputs by
leveraging more reliable retrieval mechanisms and robust
post-generation processes. Methods such as RAGTruth [113]
introduce large-scale datasets designed to detect and mitigate
hallucinations, enabling models to generate more trustworthy
responses. SEER [114] proposes a novel evidence extraction
learning paradigm, which utilizes the model to calibrate its
extraction preference via self-alignment. Ayala et al. [115]
use external knowledge sources to reduce errors in structured
output generation, enhancing the reliability of RAG systems
in practical applications.

Task-specific advancements in RAG systems focus on refining retrieval-augmented models for particular applications,
improving their efficiency and effectiveness in complex tasks.
Demonstrate-Search-Predict [116] introduces a modular approach that breaks down complex problems into manageable
tasks, enhancing performance in multi-hop reasoning and
open-domain question answering. Similarly, RA-DIT [117]
uses dual instruction tuning to fine-tune the retriever and generative model, optimizing their collaboration for knowledgeintensive benchmarks. These methods highlight the importance
of tailoring RAG systems to specific tasks, enabling more
effective and accurate solutions across diverse domains.


_D._ _Multimodal_ _RAG_


Multimodal RAG technology is an extension of the traditional RAG model, designed to enhance the performance of
generative tasks by incorporating information from multiple
data modalities [118]. Unlike the traditional RAG, which
processes only textual data, multimodal RAG can handle not
only text but also other modalities such as images, audio, and
video. It is capable of extracting information from various
modalities and integrating this information to generate richer
and more accurate outputs. In multimodal RAG systems,
embeddings for various data types, such as text and images,
are generated through modality-specific encoders [119]. These
encoders share a unified embedding space, which is also
employed for encoding the query.

The latest advancements in RAG in the image domain
have led to significant improvements [120]. RA-CM3 [121]


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 18



enhances both text-to-image and image-to-text generation by
combining the CLIP retriever and the CM3 Transformer generator, achieving a performance boost while reducing computational costs by over 30%. Mortaheb et al. introduced a reranking mechanism based on a relevance score model [122],
improving context selection during retrieval and reducing
hallucinations, thereby enhancing the quality of generated
responses. Yu et al.’s VisRAG [123] framework bypasses the
text parsing stage to directly process multi-modal documents
containing both text and images, achieving substantial improvements in multi-modal tasks. Bonomo and Bianco’s Visual
RAG [124] expands the visual knowledge of large MLLMs
without the need for fine-tuning by dynamically retrieving
relevant examples, offering high computational efficiency.
Riedler and Langer’s work on multimodal inputs for industrial
applications demonstrates that integrating both images and text
in RAG systems significantly improves performance [125].

RAG has also made significant progress in the video domain, driving innovations in long video comprehension. Yongdong Luo et al. propose Video-RAG [126], which enhances
long video understanding by integrating visually-aligned auxiliary texts into large video-language models, surpassing models like Gemini1.5-Pro and GPT-4o. Jeong et al. introduce
VideoRAG [127], a method that dynamically retrieves relevant
videos based on user queries and combines both visual and
textual information to generate more contextually rich responses, showing marked improvements over traditional RAG
approaches. Ma et al. present DrVideo [128], a system that
converts long videos into text-based documents and iteratively
retrieves missing information, achieving high accuracy in key
frame identification. Arefeen et al. introduce iRAG [129], an
incremental method that accelerates video-to-text processing
by using lightweight models for fast indexing and heavyweight
models for detailed extraction, making it well-suited for realtime video analysis. Zhang et al. propose OmAgent [130], a
multi-modal agent framework that minimizes information loss
in video understanding tasks by dynamically invoking retrieval
tools, thereby enhancing reasoning and event localization.


XIII. CONCLUSION AND FUTURE WORK


In this work, we introduced the **ImageRAG** framework. It
retrieves relevant visual context from UHR remote sensing
images based on key phrases in text queries, enabling the
MLLM to focus on **important** **details,** **including** **tiny** **ones**,
and answer questions by inferring them. ImageRAG integrates
various **external** **knowledge** **databases** to guide the model,
enhancing its understanding of the query and UHR RSI.
Notably, ImageRAG requires minimal training (only finetuning the inferring model), making it a practical solution for
efficiently handling UHR RSI. We also introduce the MMERealWorld-lite-RS benchmark, demonstrating that ImageRAG
achieves strong performance across Regular VQA, Inferring
VQA, and Visual Cue Retrieval tasks on it.

In the future, we plan to apply more RAG techniques to
ImageRAG, particularly enhancing the ranking component.
We will also prioritize optimizing ImageRAG’s efficiency and
scalability. For the inferring model, we will investigate how



to minimize the negative influence of inaccurate visual cues.
Moreover, to boost performance on specialized imaging data,
we intend to introduce adapter module with trainable parameters. This will enable effective adaptation to domain-specific
data such as SAR, infrared, and hyperspectral imagery. The
approach may adopt an in-context learning style, minimizing
the need for extensive additional training data while preserving
the model’s general capabilities.


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 19



APPENDIX A
MME-REALWORLD-LITE-RS CORRECTIONS


Two majority corrections are "Multiple Correct Answers"
and "Incorrect Label".









APPENDIX D
PROMPT TEMPLATE FOR QUESTION ANALYZING MODULE





APPENDIX B
MME-REALWORLD PROMPT TEMPLATE





APPENDIX C
INFERRING MODEL PROMPT TEMPLATE


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 20


APPENDIX E
CLASS NAME IN LRSD




JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 21



REFERENCES


[1] G.-S. Xia, X. Bai, J. Ding, Z. Zhu, S. Belongie, J. Luo, M. Datcu,

M. Pelillo, and L. Zhang, “Dota: A large-scale dataset for object
detection in aerial images,” in _The_ _IEEE_ _Conference_ _on_ _Computer_
_Vision_ _and_ _Pattern_ _Recognition_ _(CVPR)_, June 2018.

[2] S. Waqas Zamir, A. Arora, A. Gupta, S. Khan, G. Sun, F. Shah
baz Khan, F. Zhu, L. Shao, G.-S. Xia, and X. Bai, “isaid: A large-scale
dataset for instance segmentation in aerial images,” in _Proceedings_
_of_ _the_ _IEEE_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_
_Workshops_, 2019, pp. 28–37.

[3] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S.

Khan, “Geochat: Grounded large vision-language model for remote
sensing,” _The_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_
_Recognition_, 2024.

[4] W. Zhang, M. Cai, T. Zhang, Y. Zhuang, and X. Mao, “Earthgpt: A

universal multi-modal large language model for multi-sensor image
comprehension in remote sensing domain,” 2024. [Online]. Available:
[https://arxiv.org/abs/2401.16822](https://arxiv.org/abs/2401.16822)

[5] J. Luo, Z. Pang, Y. Zhang, T. Wang, L. Wang, B. Dang, J. Lao, J. Wang,

J. Chen, Y. Tan, and Y. Li, “Skysensegpt: A fine-grained instruction
tuning dataset and model for remote sensing vision-language understanding,” _arXiv_ _preprint_ _arXiv:2406.10100_, 2024.

[6] C. Pang, X. Weng, J. Wu, J. Li, Y. Liu, J. Sun, W. Li, S. Wang,

L. Feng, G.-S. Xia, and C. He, “Vhm: Versatile and honest vision
language model for remote sensing image analysis,” 2024. [Online].
Available: [https://arxiv.org/abs/2403.20213](https://arxiv.org/abs/2403.20213)

[7] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual

instruction tuning,” 2023.

[8] Z. Zhang, T. Zhao, Y. Guo, and J. Yin, “Rs5m and georsclip: A

large scale vision-language dataset and a large vision-language model
for remote sensing,” _IEEE_ _Transactions_ _on_ _Geoscience_ _and_ _Remote_
_Sensing_, pp. 1–1, 2024.

[9] F. Liu, D. Chen, Z. Guan, X. Zhou, J. Zhu, Q. Ye, L. Fu,

and J. Zhou, “Remoteclip: A vision language foundation model
for remote sensing,” _IEEE_ _Transactions_ _on_ _Geoscience_ _and_ _Remote_
_Sensing_, vol. 62, pp. 1–16, 2024. [Online]. Available: [https:](https://doi.org/10.1109/TGRS.2024.3390838)
[//doi.org/10.1109/TGRS.2024.3390838](https://doi.org/10.1109/TGRS.2024.3390838)

[10] F. Wang, H. Wang, M. Chen, D. Wang, Y. Wang, Z. Guo, Q. Ma,

L. Lan, W. Yang, J. Zhang, Z. Liu, and M. Sun, “Xlrs-bench:
Could your multimodal llms understand extremely large ultrahigh-resolution remote sensing imagery?” 2025. [Online]. Available:
[https://arxiv.org/abs/2503.23771](https://arxiv.org/abs/2503.23771)

[11] Y.-F. Zhang, H. Zhang, H. Tian, C. Fu, S. Zhang, J. Wu, F. Li, K. Wang,

Q. Wen, Z. Zhang _et_ _al._, “Mme-realworld: Could your multimodal
llm challenge high-resolution real-world scenarios that are difficult for
humans?” _arXiv_ _preprint_ _arXiv:2408.13257_, 2024.

[12] J. Luo, Y. Zhang, X. Yang, K. Wu, Q. Zhu, L. Liang, J. Chen, and

Y. Li, “When large vision-language model meets large remote sensing
imagery: Coarse-to-fine text-guided token pruning,” 2025. [Online].
Available: [https://arxiv.org/abs/2503.07588](https://arxiv.org/abs/2503.07588)

[13] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye,

H. Tian, Z. Liu _et_ _al._, “Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling,”
_arXiv_ _preprint_ _arXiv:2412.05271_, 2024.

[14] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu,

J. Luo, Z. Ma _et_ _al._, “How far are we to gpt-4v? closing the gap
to commercial multimodal models with open-source suites,” _arXiv_
_preprint_ _arXiv:2404.16821_, 2024.

[15] S. Chen, S. Wong, L. Chen, and Y. Tian, “Extending context window

of large language models via positional interpolation,” 2023. [Online].
Available: [https://arxiv.org/abs/2306.15595](https://arxiv.org/abs/2306.15595)

[16] Y. Ding, L. L. Zhang, C. Zhang, Y. Xu, N. Shang, J. Xu, F. Yang, and

M. Yang, “Longrope: Extending llm context window beyond 2 million
tokens,” 2024. [Online]. Available: [https://arxiv.org/abs/2402.13753](https://arxiv.org/abs/2402.13753)

[17] F. Xue, Y. Chen, D. Li, Q. Hu, L. Zhu, X. Li, Y. Fang,
H. Tang, S. Yang, Z. Liu, E. He, H. Yin, P. Molchanov, J. Kautz,
L. Fan, Y. Zhu, Y. Lu, and S. Han, “Longvila: Scaling long-context
visual language models for long videos,” 2024. [Online]. Available:
[https://arxiv.org/abs/2408.10188](https://arxiv.org/abs/2408.10188)

[18] P. Wu and S. Xie, “V*: Guided visual search as a core
mechanism in multimodal llms,” 2023. [Online]. Available: [https:](https://arxiv.org/abs/2312.14135)
[//arxiv.org/abs/2312.14135](https://arxiv.org/abs/2312.14135)

[19] X. Wang, D. Song, S. Chen, C. Zhang, and B. Wang, “Longllava:

Scaling multi-modal llms to 1000 images efficiently via a hybrid
architecture,” 2024. [Online]. Available: [https://arxiv.org/abs/2409.](https://arxiv.org/abs/2409.02889)
[02889](https://arxiv.org/abs/2409.02889)




[20] X. Sun, P. Wang, Z. Yan, F. Xu, R. Wang, W. Diao, J. Chen, J. Li,

Y. Feng, T. Xu, M. Weinmann, S. Hinz, C. Wang, and K. Fu,
“Fair1m: A benchmark dataset for fine-grained object recognition in
high-resolution remote sensing imagery,” 2021. [Online]. Available:
[https://arxiv.org/abs/2103.05569](https://arxiv.org/abs/2103.05569)

[21] D. Hou, Z. Miao, H. Xing, and H. Wu, “V-rsir: An open access web
based image annotation tool for remote sensing image retrieval,” _IEEE_
_Access_, vol. 7, pp. 83 852–83 862, 2019.

[22] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai,

J. Sun, M. Wang, and H. Wang, “Retrieval-augmented generation
for large language models: A survey,” 2024. [Online]. Available:
[https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)

[23] R. C. Barron, V. Grantcharov, S. Wanna, M. E. Eren, M. Bhattarai,

N. Solovyev, G. Tompkins, C. Nicholas, K. Rasmussen, C. Matuszek,
and B. S. Alexandrov, “Domain-specific retrieval-augmented generation
using vector stores, knowledge graphs, and tensor factorization,” 2024.

[Online]. Available: [https://arxiv.org/abs/2410.02721](https://arxiv.org/abs/2410.02721)

[24] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,

T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly,
J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words:
Transformers for image recognition at scale,” 2021. [Online].
Available: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

[25] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo,

“Swin transformer: Hierarchical vision transformer using shifted
windows,” 2021. [Online]. Available: [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)

[26] Z. Zhang, C. Shen, Y. Shen, H. Xiong, and X. Zhou, “Injecting

image details into clip’s feature space,” 2023. [Online]. Available:
[https://arxiv.org/abs/2208.14649](https://arxiv.org/abs/2208.14649)

[27] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,

G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever,
“Learning transferable visual models from natural language supervision,” 2021.

[28] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu,

F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang,
J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li,
M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Xia, X. Ren,
X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and
Z. Qiu, “Qwen2.5 technical report,” _arXiv_ _preprint_ _arXiv:2412.15115_,
2024.

[29] L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao,

C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng,
“Sglang: Efficient execution of structured language model programs,”
2024. [Online]. Available: [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)

[30] M. Grootendorst, “Keybert: Minimal keyword extraction with bert.”

2020. [Online]. Available: [https://doi.org/10.5281/zenodo.4461265](https://doi.org/10.5281/zenodo.4461265)

[31] K. Schall, K. U. Barthel, N. Hezel, and K. Jung, “Optimizing

clip models for image retrieval with maintained joint-embedding
[alignment,” 2024. [Online]. Available: https://arxiv.org/abs/2409.01936](https://arxiv.org/abs/2409.01936)

[32] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular

margin loss for deep face recognition,” in _2019_ _IEEE/CVF_ _Conference_
_on Computer Vision and Pattern Recognition (CVPR)_, 2019, pp. 4685–
4694.

[33] J. Ding, N. Xue, G.-S. Xia, X. Bai, W. Yang, M. Y. Yang,

S. Belongie, J. Luo, M. Datcu, M. Pelillo, and L. Zhang, “Object
detection in aerial images: A large-scale benchmark and challenges,”
_IEEE_ _Transactions_ _on_ _Pattern_ _Analysis_ _and_ _Machine_ _Intelligence_,
vol. 44, no. 11, p. 7778–7796, Nov. 2022. [Online]. Available:
[http://dx.doi.org/10.1109/TPAMI.2021.3117983](http://dx.doi.org/10.1109/TPAMI.2021.3117983)

[34] G. Cheng, X. Yuan, X. Yao, K. Yan, Q. Zeng, X. Xie,
and J. Han, “Towards large-scale small object detection: Survey
and benchmarks,” _IEEE_ _Transactions_ _on_ _Pattern_ _Analysis_ _and_
_Machine_ _Intelligence_, p. 1–20, 2023. [Online]. Available: [http:](http://dx.doi.org/10.1109/TPAMI.2023.3290594)
[//dx.doi.org/10.1109/TPAMI.2023.3290594](http://dx.doi.org/10.1109/TPAMI.2023.3290594)

[35] J. Wang, Z. Zheng, A. Ma, X. Lu, and Y. Zhong, “Loveda: A remote

sensing land-cover dataset for domain adaptive semantic segmentation,”
in _Proceedings_ _of_ _the_ _Neural_ _Information_ _Processing_ _Systems_
_Track_ _on_ _Datasets_ _and_ _Benchmarks_, J. Vanschoren and S. Yeung,
Eds., vol. 1. Curran Associates, Inc., 2021. [Online]. Available:
[https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf)
[2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf)

[36] Y. Long, G.-S. Xia, S. Li, W. Yang, M. Y. Yang, X. X. Zhu,

L. Zhang, and D. Li, “On creating benchmark dataset for aerial image
interpretation: Reviews, guidances and million-aid,” 2021. [Online].
Available: [https://arxiv.org/abs/2006.12485](https://arxiv.org/abs/2006.12485)

[37] G. Christie, N. Fendley, J. Wilson, and R. Mukherjee, “Functional

map of the world,” 2018. [Online]. Available: [https://arxiv.org/abs/](https://arxiv.org/abs/1711.07846)
[1711.07846](https://arxiv.org/abs/1711.07846)


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 22




[38] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wight
man, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman,
P. Schramowski, S. Kundurthy, K. Crowson, L. Schmidt, R. Kaczmarczyk, and J. Jitsev, “Laion-5b: An open large-scale dataset for training
next generation image-text models,” 2022.

[39] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis,

A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki, “Laion-400m:
Open dataset of clip-filtered 400 million image-text pairs,” 2021.

[40] M. Byeon, B. Park, H. Kim, S. Lee, W. Baek, and S. Kim, “Coyo-700m:

Image-text pair dataset,” [https://github.com/kakaobrain/coyo-dataset,](https://github.com/kakaobrain/coyo-dataset)
2022.

[41] P. Sharma, N. Ding, S. Goodman, and R. Soricut, “Conceptual captions:

A cleaned, hypernymed, image alt-text dataset for automatic image
captioning,” in _Proceedings_ _of_ _ACL_, 2018.

[42] S. Changpinyo, P. Sharma, N. Ding, and R. Soricut, “Conceptual 12M:

Pushing web-scale image-text pre-training to recognize long-tail visual
concepts,” in _CVPR_, 2021.

[43] B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni,

D. Poland, D. Borth, and L.-J. Li, “YFCC100m,” _Communications_ _of_
_the_ _ACM_, vol. 59, no. 2, pp. 64–73, jan 2016. [Online]. Available:
[https://doi.org/10.1145%2F2812802](https://doi.org/10.1145%2F2812802)

[44] K. Srinivasan, K. Raman, J. Chen, M. Bendersky, and M. Najork,

“Wit: Wikipedia-based image text dataset for multimodal multilingual
machine learning,” _arXiv_ _preprint_ _arXiv:2103.01913_, 2021.

[45] K. Desai, G. Kaul, Z. Aysola, and J. Johnson, “RedCaps: Web-curated

image-text data created by the people, for the people,” in _NeurIPS_
_Datasets_ _and_ _Benchmarks_, 2021.

[46] V. Ordonez, G. Kulkarni, and T. L. Berg, “Im2text: Describing im
ages using 1 million captioned photographs,” in _Neural_ _Information_
_Processing_ _Systems_ _(NIPS)_, 2011.

[47] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz,

S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma, M. Bernstein, and
L. Fei-Fei, “Visual genome: Connecting language and vision using
crowdsourced dense image annotations,” 2016. [Online]. Available:
[https://arxiv.org/abs/1602.07332](https://arxiv.org/abs/1602.07332)

[48] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings

using siamese bert-networks,” in _Proceedings_ _of_ _the_ _2019_ _Conference_
_on_ _Empirical_ _Methods_ _in_ _Natural_ _Language_ _Processing_ . Association
for Computational Linguistics, 11 2019. [Online]. Available: [https:](https://arxiv.org/abs/1908.10084)
[//arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

[49] J. Snell, K. Swersky, and R. S. Zemel, “Prototypical networks for

few-shot learning,” 2017. [Online]. Available: [https://arxiv.org/abs/](https://arxiv.org/abs/1703.05175)
[1703.05175](https://arxiv.org/abs/1703.05175)

[50] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A density-based

algorithm for discovering clusters in large spatial databases with noise,”
in _Proceedings_ _of_ _the_ _Second_ _International_ _Conference_ _on_ _Knowledge_
_Discovery_ _and_ _Data_ _Mining_, ser. KDD’96. AAAI Press, 1996, p.
226–231.

[51] H. Shen, K. Zhao, T. Zhao, R. Xu, Z. Zhang, M. Zhu, and J. Yin,

“Zoomeye: Enhancing multimodal llms with human-like zooming
capabilities through tree-based image exploration,” _arXiv_ _preprint_
_arXiv:2411.16044_, 2024.

[52] X. Li, J. Ding, and M. Elhoseiny, “Vrsbench: A versatile vision
language benchmark dataset for remote sensing image understanding,”
2024. [Online]. Available: [https://arxiv.org/abs/2406.12384](https://arxiv.org/abs/2406.12384)

[53] F. Bai, Y. Du, T. Huang, M. Q. H. Meng, and B. Zhao, “M3d:

Advancing 3d medical image analysis with multi-modal large language
models,” 2024. [Online]. Available: [https://arxiv.org/abs/2404.00578](https://arxiv.org/abs/2404.00578)

[54] R. AlSaad, A. Abd-Alrazaq, S. Boughorbel, A. Ahmed, M.-A. Renault,

R. Damseh, and J. Sheikh, “Multimodal large language models in
health care: Applications, challenges, and future outlook,” _J._ _Med._
_Internet_ _Res._, vol. 26, p. e59505, Sep. 2024.

[55] Y. Li, L. Wang, T. Wang, X. Yang, J. Luo, Q. Wang, Y. Deng,

W. Wang, X. Sun, H. Li, B. Dang, Y. Zhang, Y. Yu, and J. Yan, “Star:
A first-ever dataset and a large-scale benchmark for scene graph
generation in large-size satellite imagery,” 2024. [Online]. Available:
[https://arxiv.org/abs/2406.09410](https://arxiv.org/abs/2406.09410)

[56] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.

Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,”
2023. [Online]. Available: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

[57] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang,

and W. Chen, “Lora: Low-rank adaptation of large language models,”
2021. [Online]. Available: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

[58] Y. Li, Y. Zhang, C. Wang, Z. Zhong, Y. Chen, R. Chu, S. Liu, and J. Jia,

“Mini-gemini: Mining the potential of multi-modality vision language
models,” 2024. [Online]. Available: [https://arxiv.org/abs/2403.18814](https://arxiv.org/abs/2403.18814)




[59] C. Pang, J. Wu, J. Li, Y. Liu, J. Sun, W. Li, X. Weng, S. Wang,

L. Feng, G.-S. Xia, and C. He, “H2rsvlm: Towards helpful and
honest remote sensing large vision language model,” 2024. [Online].
Available: [https://arxiv.org/abs/2403.20213](https://arxiv.org/abs/2403.20213)

[60] G. Luo, Y. Zhou, Y. Zhang, X. Zheng, X. Sun, and R. Ji, “Feast your

eyes: Mixture-of-resolution adaptation for multimodal large language
models,” 2024. [Online]. Available: [https://arxiv.org/abs/2403.03003](https://arxiv.org/abs/2403.03003)

[61] Y. Zhang, Y. Liu, Z. Guo, Y. Zhang, X. Yang, C. Chen, J. Song,

B. Zheng, Y. Yao, Z. Liu, T.-S. Chua, and M. Sun, “Llava-uhd v2:
an mllm integrating high-resolution feature pyramid via hierarchical
window transformer,” _arXiv_ _preprint_ _arXiv:2412.13871_, 2024.

[62] K. Team, A. Du, B. Yin, B. Xing, B. Qu, B. Wang, C. Chen,

C. Zhang, C. Du, C. Wei, C. Wang, D. Zhang, D. Du, D. Wang,
E. Yuan, E. Lu, F. Li, F. Sung, G. Wei, G. Lai, H. Zhu, H. Ding,
H. Hu, H. Yang, H. Zhang, H. Wu, H. Yao, H. Lu, H. Wang,
H. Gao, H. Zheng, J. Li, J. Su, J. Wang, J. Deng, J. Qiu, J. Xie,
J. Wang, J. Liu, J. Yan, K. Ouyang, L. Chen, L. Sui, L. Yu, M. Dong,
M. Dong, N. Xu, P. Cheng, Q. Gu, R. Zhou, S. Liu, S. Cao, T. Yu,
T. Song, T. Bai, W. Song, W. He, W. Huang, W. Xu, X. Yuan,
X. Yao, X. Wu, X. Zu, X. Zhou, X. Wang, Y. Charles, Y. Zhong,
Y. Li, Y. Hu, Y. Chen, Y. Wang, Y. Liu, Y. Miao, Y. Qin, Y. Chen,
Y. Bao, Y. Wang, Y. Kang, Y. Liu, Y. Du, Y. Wu, Y. Wang, Y. Yan,
Z. Zhou, Z. Li, Z. Jiang, Z. Zhang, Z. Yang, Z. Huang, Z. Huang,
Z. Zhao, and Z. Chen, “Kimi-VL technical report,” 2025. [Online].
Available: [https://arxiv.org/abs/2504.07491](https://arxiv.org/abs/2504.07491)

[63] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia,

E. Chi, Q. Le, and D. Zhou, “Chain-of-thought prompting elicits
reasoning in large language models,” 2023. [Online]. Available:
[https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

[64] S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang,

Y. Huang, W. Zhao _et_ _al._, “Minicpm: Unveiling the potential of
small language models with scalable training strategies,” _arXiv preprint_
_arXiv:2404.06395_, 2024.

[65] K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, M. Amin,

L. Hou, K. Clark, S. R. Pfohl, H. Cole-Lewis, D. Neal, Q. M. Rashid,
M. Schaekermann, A. Wang, D. Dash, J. H. Chen, N. H. Shah,
S. Lachgar, P. A. Mansfield, S. Prakash, B. Green, E. Dominowska,
B. Agüera Y Arcas, N. Tomašev, Y. Liu, R. Wong, C. Semturs, S. S.
Mahdavi, J. K. Barral, D. R. Webster, G. S. Corrado, Y. Matias,
S. Azizi, A. Karthikesalingam, and V. Natarajan, “Toward expert-level
medical question answering with large language models,” _Nat._ _Med._,
vol. 31, no. 3, pp. 943–950, Mar. 2025.

[66] Z. Xiong, Y. Wang, W. Yu, A. J. Stewart, J. Zhao, N. Lehmann,

T. Dujardin, Z. Yuan, P. Ghamisi, and X. X. Zhu, “Geolangbind:
Unifying earth observation with agglomerative vision-language
foundation models,” 2025. [Online]. Available: [https://arxiv.org/abs/](https://arxiv.org/abs/2503.06312)
[2503.06312](https://arxiv.org/abs/2503.06312)

[67] Z. Wang, Z. Wu, D. Agarwal, and J. Sun, “Medclip: Contrastive

learning from unpaired medical images and text,” 2022. [Online].
Available: [https://arxiv.org/abs/2210.10163](https://arxiv.org/abs/2210.10163)

[68] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson,

K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford,
S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. Menick,
S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. Binkowski,
R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan, “Flamingo:
a visual language model for few-shot learning,” 2022.

[69] X. Li, C. Wen, Y. Hu, Z. Yuan, and X. X. Zhu, “Vision-language

models in remote sensing: Current progress and future trends,” 2024.

[Online]. Available: [https://arxiv.org/abs/2305.05726](https://arxiv.org/abs/2305.05726)

[70] Z. Yuan, L. Mou, Y. Hua, and X. X. Zhu, “Rrsis: Referring

remote sensing image segmentation,” 2024. [Online]. Available:
[https://arxiv.org/abs/2306.08625](https://arxiv.org/abs/2306.08625)

[71] S. Liu, Y. Ma, X. Zhang, H. Wang, J. Ji, X. Sun, and
R. Ji, “Rotated multi-scale interaction network for referring remote
sensing image segmentation,” 2024. [Online]. Available: [https:](https://arxiv.org/abs/2312.12470)
[//arxiv.org/abs/2312.12470](https://arxiv.org/abs/2312.12470)

[72] S. Khanna, P. Liu, L. Zhou, C. Meng, R. Rombach, M. Burke,

D. Lobell, and S. Ermon, “Diffusionsat: A generative foundation
model for satellite imagery,” 2024. [Online]. Available: [https:](https://arxiv.org/abs/2312.03606)
[//arxiv.org/abs/2312.03606](https://arxiv.org/abs/2312.03606)

[73] Z. Yu, C. Liu, L. Liu, Z. Shi, and Z. Zou, “Metaearth: A generative

foundation model for global-scale remote sensing image generation,”
2024. [Online]. Available: [https://arxiv.org/abs/2405.13570](https://arxiv.org/abs/2405.13570)

[74] Y. Zhou, L. Feng, Y. Ke, X. Jiang, J. Yan, X. Yang, and W. Zhang,

“Towards vision-language geo-foundation model: A survey,” 2024.

[Online]. Available: [https://arxiv.org/abs/2406.09385](https://arxiv.org/abs/2406.09385)


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 23




[75] U. Mall, C. P. Phoo, M. K. Liu, C. Vondrick, B. Hariharan, and

K. Bala, “Remote sensing vision-language foundation models without
annotations via ground remote alignment,” 2023. [Online]. Available:
[https://arxiv.org/abs/2312.06960](https://arxiv.org/abs/2312.06960)

[76] Z. Wang, R. Prabha, T. Huang, J. Wu, and R. Rajagopal,
“Skyscript: A large and semantically diverse vision-language dataset
for remote sensing,” _Proceedings_ _of_ _the_ _AAAI_ _Conference_ _on_ _Artificial_
_Intelligence_, vol. 38, no. 6, pp. 5805–5813, Mar. 2024. [Online].
Available: [https://ojs.aaai.org/index.php/AAAI/article/view/28393](https://ojs.aaai.org/index.php/AAAI/article/view/28393)

[77] Y. Hu, J. Yuan, C. Wen, X. Lu, and X. Li, “Rsgpt: A remote sensing

vision language model and benchmark,” 2023. [Online]. Available:
[https://arxiv.org/abs/2307.15266](https://arxiv.org/abs/2307.15266)

[78] Y. Zhan, Z. Xiong, and Y. Yuan, “Skyeyegpt: Unifying remote

sensing vision-language tasks via instruction tuning with large
language model,” _ISPRS_ _Journal_ _of_ _Photogrammetry_ _and_ _Remote_
_Sensing_, vol. 221, pp. 64–77, 2025. [Online]. Available: [https:](https://www.sciencedirect.com/science/article/pii/S0924271625000206)
[//www.sciencedirect.com/science/article/pii/S0924271625000206](https://www.sciencedirect.com/science/article/pii/S0924271625000206)

[79] D. Muhtar, Z. Li, F. Gu, X. Zhang, and P. Xiao, “Lhrs-bot: Empowering

remote sensing with vgi-enhanced large multimodal language model,”
in _Computer_ _Vision_ _–_ _ECCV_ _2024_, A. Leonardis, E. Ricci, S. Roth,
O. Russakovsky, T. Sattler, and G. Varol, Eds. Cham: Springer Nature
Switzerland, 2025, pp. 440–457.

[80] J. D. Silva, J. Magalhães, D. Tuia, and B. Martins, “Large language

models for captioning and retrieving remote sensing images,” 2024.

[Online]. Available: [https://arxiv.org/abs/2402.06475](https://arxiv.org/abs/2402.06475)

[81] D. Tang, X. Cao, X. Hou, Z. Jiang, J. Liu, and D. Meng, “Crs-diff:

Controllable remote sensing image generation with diffusion model,”
2024. [Online]. Available: [https://arxiv.org/abs/2403.11614](https://arxiv.org/abs/2403.11614)

[82] Z. Zheng, S. Ermon, D. Kim, L. Zhang, and Y. Zhong, “Changen2:

Multi-temporal remote sensing generative change foundation model,”
_IEEE_ _Transactions_ _on_ _Pattern_ _Analysis_ _and_ _Machine_ _Intelligence_,
vol. 47, no. 2, pp. 725–741, 2025.

[83] P. Wu and S. Xie, “V?: Guided visual search as a core mechanism

in multimodal llms,” in _Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_
_Computer_ _Vision_ _and_ _Pattern_ _Recognition_, 2024, pp. 13 084–13 094.

[84] M. Wu, X. Cai, J. Ji, J. Li, O. Huang, G. Luo, H. Fei, G. Jiang,

X. Sun, and R. Ji, “Controlmllm: Training-free visual prompt learning
for multimodal large language models,” 2024. [Online]. Available:
[https://arxiv.org/abs/2407.21534](https://arxiv.org/abs/2407.21534)

[85] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei,

“Kosmos-2: Grounding multimodal large language models to the
world,” _arXiv_ _preprint_ _arXiv:2306.14824_, 2023.

[86] K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao,

“Shikra: Unleashing multimodal llm’s referential dialogue magic,”
_arXiv_ _preprint_ _arXiv:2306.15195_, 2023.

[87] H. Zhang, H. You, P. Dufter, B. Zhang, C. Chen, H.-Y. Chen, T.-J.

Fu, W. Y. Wang, S.-F. Chang, Z. Gan, and Y. Yang, “Ferret-v2: An
improved baseline for referring and grounding with large language
models,” 2024. [Online]. Available: [https://arxiv.org/abs/2404.07973](https://arxiv.org/abs/2404.07973)

[88] A. Dosovitskiy, “An image is worth 16x16 words: Transformers for

image recognition at scale,” _arXiv_ _preprint_ _arXiv:2010.11929_, 2020.

[89] Y.-C. Chen, L. Li, L. Yu, A. El Kholy, F. Ahmed, Z. Gan, Y. Cheng,

and J. Liu, “Uniter: Universal image-text representation learning,” in
_Computer_ _Vision–ECCV_ _2020:_ _16th_ _European_ _Conference,_ _Glasgow,_
_UK, August 23–28, 2020, Proceedings, Part XXX_ . Springer, 2020, pp.
104–120.

[90] P. Zhang, X. Li, X. Hu, J. Yang, L. Zhang, L. Wang, Y. Choi, and

J. Gao, “Vinvl: Revisiting visual representations in vision-language
models,” in _Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_ _Computer_
_Vision_ _and_ _Pattern_ _Recognition_, 2021, pp. 5579–5588.

[91] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time

object detection with region proposal networks,” _IEEE_ _transactions_ _on_
_pattern_ _analysis_ _and_ _machine_ _intelligence_, vol. 39, no. 6, pp. 1137–
1149, 2016.

[92] X. Zhou, R. Girdhar, A. Joulin, P. Krähenbühl, and I. Misra, “Detecting

twenty-thousand classes using image-level supervision,” in _Computer_
_Vision–ECCV_ _2022:_ _17th_ _European_ _Conference,_ _Tel_ _Aviv,_ _Israel,_ _Octo-_
_ber_ _23–27,_ _2022,_ _Proceedings,_ _Part_ _IX_ . Springer, 2022, pp. 350–368.

[93] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li,

J. Yang, H. Su, J. Zhu _et_ _al._, “Grounding dino: Marrying dino with
grounded pre-training for open-set object detection,” _arXiv_ _preprint_
_arXiv:2303.05499_, 2023.

[94] Q. Jiang, G. luo, Y. Yang, Y. Xiong, Y. Chen, Z. Zeng,
T. Ren, and L. Zhang, “Chatrex: Taming multimodal llm for
joint perception and understanding,” 2024. [Online]. Available:
[https://arxiv.org/abs/2411.18363](https://arxiv.org/abs/2411.18363)




[95] Y. Zhang, Y. Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang, E. Zhao,

Y. Zhang, Y. Chen _et al._, “Siren’s song in the ai ocean: a survey on hallucination in large language models,” _arXiv preprint arXiv:2309.01219_,
2023.

[96] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,

H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel _et_ _al._, “Retrievalaugmented generation for knowledge-intensive nlp tasks,” _Advances_ _in_
_neural_ _information_ _processing_ _systems_, vol. 33, pp. 9459–9474, 2020.

[97] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun,

H. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,” _arXiv_ _preprint_ _arXiv:2312.10997_, vol. 2,
2023.

[98] O. Ovadia, M. Brief, M. Mishaeli, and O. Elisha, “Fine-tuning or

retrieval? comparing knowledge injection in llms,” _arXiv_ _preprint_
_arXiv:2312.05934_, 2023.

[99] L. Gao, X. Ma, J. Lin, and J. Callan, “Precise zero-shot dense retrieval

without relevance labels,” in _Proceedings_ _of_ _the_ _61st_ _Annual_ _Meeting_
_of_ _the_ _Association_ _for_ _Computational_ _Linguistics_ _(Volume_ _1:_ _Long_
_Papers)_, 2023, pp. 1762–1777.

[100] S. Wang, Y. Xu, Y. Fang, Y. Liu, S. Sun, R. Xu, C. Zhu, and

M. Zeng, “Training data is more valuable than you think: A simple
and effective method by retrieving from training data,” _arXiv_ _preprint_
_arXiv:2203.08773_, 2022.

[101] X. Ma, Y. Gong, P. He, H. Zhao, and N. Duan, “Query rewriting in

retrieval-augmented large language models,” in _Proceedings of the 2023_
_Conference_ _on_ _Empirical_ _Methods_ _in_ _Natural_ _Language_ _Processing_,
2023, pp. 5303–5315.

[102] Z. Dai, V. Y. Zhao, J. Ma, Y. Luan, J. Ni, J. Lu, A. Bakalov, K. Guu,

K. B. Hall, and M.-W. Chang, “Promptagator: Few-shot dense retrieval
from 8 examples,” _arXiv_ _preprint_ _arXiv:2209.11755_, 2022.

[103] Z. Ke, W. Kong, C. Li, M. Zhang, Q. Mei, and M. Bendersky,

“Bridging the preference gap between retrievers and llms,” _arXiv_
_preprint_ _arXiv:2401.06954_, 2024.

[104] Z. Sun, X. Wang, Y. Tay, Y. Yang, and D. Zhou, “Recitation-augmented

language models,” _arXiv_ _preprint_ _arXiv:2210.01296_, 2022.

[105] Z. Shao, Y. Gong, Y. Shen, M. Huang, N. Duan, and W. Chen,

“Enhancing retrieval-augmented large language models with iterative
retrieval-generation synergy,” _arXiv_ _preprint_ _arXiv:2305.15294_, 2023.

[106] X. Cheng, D. Luo, X. Chen, L. Liu, D. Zhao, and R. Yan, “Lift yourself

up: Retrieval-augmented text generation with self-memory,” _Advances_
_in_ _Neural_ _Information_ _Processing_ _Systems_, vol. 36, pp. 43 780–43 799,
2023.

[107] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning

to retrieve, generate, and critique through self-reflection,” in _The_
_Twelfth_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2023.

[108] H. S. Zheng, S. Mishra, X. Chen, H.-T. Cheng, E. H. Chi, Q. V. Le,

and D. Zhou, “Take a step back: Evoking reasoning via abstraction in
large language models,” _arXiv_ _preprint_ _arXiv:2310.06117_, 2023.

[109] W. Yu, D. Iter, S. Wang, Y. Xu, M. Ju, S. Sanyal, C. Zhu, M. Zeng,

and M. Jiang, “Generate rather than retrieve: Large language models
are strong context generators,” _arXiv preprint arXiv:2209.10063_, 2022.

[110] Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang,

J. Callan, and G. Neubig, “Active retrieval augmented generation,” in
_Proceedings_ _of_ _the_ _2023_ _Conference_ _on_ _Empirical_ _Methods_ _in_ _Natural_
_Language_ _Processing_, 2023, pp. 7969–7992.

[111] S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park, “Adaptive-rag:

Learning to adapt retrieval-augmented large language models through
question complexity,” _arXiv_ _preprint_ _arXiv:2403.14403_, 2024.

[112] X. Zhao, Y. Zhong, Z. Sun, X. Hu, Z. Liu, D. Li, B. Hu,

and M. Zhang, “FunnelRAG: A coarse-to-fine progressive retrieval
paradigm for RAG,” in _Findings_ _of_ _the_ _Association_ _for_ _Computational_
_Linguistics:_ _NAACL_ _2025_, L. Chiruzzo, A. Ritter, and L. Wang,
Eds. Albuquerque, New Mexico: Association for Computational
Linguistics, Apr. 2025, pp. 3029–3046. [Online]. Available: [https:](https://aclanthology.org/2025.findings-naacl.165/)
[//aclanthology.org/2025.findings-naacl.165/](https://aclanthology.org/2025.findings-naacl.165/)

[113] C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song,

and T. Zhang, “Ragtruth: A hallucination corpus for developing
trustworthy retrieval-augmented language models,” _arXiv_ _preprint_
_arXiv:2401.00396_, 2023.

[114] X. Zhao, D. Li, Y. Zhong, B. Hu, Y. Chen, B. Hu, and M. Zhang,

“SEER: Self-aligned evidence extraction for retrieval-augmented
generation,” in _Proceedings_ _of_ _the_ _2024_ _Conference_ _on_ _Empirical_
_Methods_ _in_ _Natural_ _Language_ _Processing_, Y. Al-Onaizan, M. Bansal,
and Y.-N. Chen, Eds. Miami, Florida, USA: Association for
Computational Linguistics, Nov. 2024, pp. 3027–3041. [Online].
Available: [https://aclanthology.org/2024.emnlp-main.178/](https://aclanthology.org/2024.emnlp-main.178/)


JOURNAL OF L <sup>A</sup> TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 24


[115] P. Béchard and O. M. Ayala, “Reducing hallucination in struc
tured outputs via retrieval-augmented generation,” _arXiv_ _preprint_
_arXiv:2404.08189_, 2024.

[116] O. Khattab, K. Santhanam, X. L. Li, D. Hall, P. Liang, C. Potts,

and M. Zaharia, “Demonstrate-search-predict: Composing retrieval
and language models for knowledge-intensive nlp,” _arXiv_ _preprint_
_arXiv:2212.14024_, 2022.

[117] X. V. Lin, X. Chen, M. Chen, W. Shi, M. Lomeli, R. James, P. Ro
driguez, J. Kahn, G. Szilvasy, M. Lewis _et_ _al._, “Ra-dit: Retrievalaugmented dual instruction tuning,” in _The_ _Twelfth_ _International_ _Con-_
_ference_ _on_ _Learning_ _Representations_, 2023.

[118] M. M. Abootorabi, A. Zobeiri, M. Dehghani, M. Mohammadkhani,

B. Mohammadi, O. Ghahroodi, M. S. Baghshah, and E. Asgari, “Ask
in any modality: A comprehensive survey on multimodal retrievalaugmented generation,” _arXiv_ _preprint_ _arXiv:2502.08826_, 2025.

[119] M. Mortaheb, M. A. A. Khojastepour, S. T. Chakradhar, and S. Ulukus,

“Re-ranking the context for multimodal retrieval augmented generation,” _arXiv_ _preprint_ _arXiv:2501.04695_, 2025.

[120] X. Zheng, Z. Weng, Y. Lyu, L. Jiang, H. Xue, B. Ren, D. Paudel,

N. Sebe, L. V. Gool, and X. Hu, “Retrieval augmented generation and
understanding in vision: A survey and new outlook,” 2025. [Online].
Available: [https://arxiv.org/abs/2503.18016](https://arxiv.org/abs/2503.18016)

[121] M. Yasunaga, A. Aghajanyan, W. Shi, R. James, J. Leskovec, P. Liang,

M. Lewis, L. Zettlemoyer, and W.-t. Yih, “Retrieval-augmented multimodal language modeling,” _arXiv_ _preprint_ _arXiv:2211.12561_, 2022.

[122] M. Mortaheb, M. A. A. Khojastepour, S. T. Chakradhar, and S. Ulukus,

“Rag-check: Evaluating multimodal retrieval augmented generation
performance,” _arXiv_ _preprint_ _arXiv:2501.03995_, 2025.

[123] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y. Yan, Z. Liu, S. Wang, X. Han,

Z. Liu _et_ _al._, “Visrag: Vision-based retrieval-augmented generation on
multi-modality documents,” _arXiv_ _preprint_ _arXiv:2410.10594_, 2024.

[124] M. Bonomo and S. Bianco, “Visual rag: Expanding mllm visual knowl
edge without fine-tuning,” _arXiv_ _preprint_ _arXiv:2501.10834_, 2025.

[125] M. Riedler and S. Langer, “Beyond text: Optimizing rag with

multimodal inputs for industrial applications,” _arXiv_ _preprint_
_arXiv:2410.21943_, 2024.

[126] Y. Luo, X. Zheng, X. Yang, G. Li, H. Lin, J. Huang, J. Ji, F. Chao,

J. Luo, and R. Ji, “Video-rag: Visually-aligned retrieval-augmented
long video comprehension,” _arXiv_ _preprint_ _arXiv:2411.13093_, 2024.

[127] S. Jeong, K. Kim, J. Baek, and S. J. Hwang, “Videorag:
Retrieval-augmented generation over video corpus,” _arXiv_ _preprint_
_arXiv:2501.05874_, 2025.

[128] Z. Ma, C. Gou, H. Shi, B. Sun, S. Li, H. Rezatofighi, and J. Cai,

“Drvideo: Document retrieval based long video understanding,” _arXiv_
_preprint_ _arXiv:2406.12846_, 2024.

[129] M. A. Arefeen, B. Debnath, M. Y. S. Uddin, and S. Chakradhar,

“irag: Advancing rag for videos with an incremental approach,” in
_Proceedings of the 33rd ACM International Conference on Information_
_and_ _Knowledge_ _Management_, 2024, pp. 4341–4348.

[130] L. Zhang, T. Zhao, H. Ying, Y. Ma, and K. Lee, “Omagent: A multi
modal agent framework for complex video understanding with task
divide-and-conquer,” _arXiv_ _preprint_ _arXiv:2406.16620_, 2024.


