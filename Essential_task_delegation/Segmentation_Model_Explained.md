# How Our Segmentation Model Works: Dataset, U-Net, Fine-Tuning and Inference

**Context:** A plain-English reference for the building-damage segmentation model used in the D4 champion runs (`Champion_Task/D4_champion_model.ipynb`) and the data it learns from. Use it when explaining the model in a meeting, the report or the viva. Settings are from the R1 reference recipe unless stated otherwise.

**Notebook map:** Cell 4 builds the masks · Cell 5 builds the datasets · Cell 6 adds copy-paste and the sampler · Cell 7 defines the loss · Cell 8 builds the model and the evaluation code · Cell 9 is the training loop · Cell 10 runs the final evaluation.

---

## 1. What the model does

A **semantic segmentation** model gives **every pixel** a class label. A classifier gives one label per image ("this image shows damage") and a detector draws boxes. A segmentation model draws the exact outline.

- **Input:** one post-disaster optical satellite image, RGB, usually 1024×1024. The model sees only the post-disaster image, not the pre-disaster one (`in_channels=3`).
- **Output:** a map of the same size where each pixel is one of 4 classes:

| ID | Class |
|---|---|
| 0 | Background |
| 1 | Intact building |
| 2 | Damaged building |
| 3 | Destroyed building |

---

## 2. The dataset, explained from scratch

### 2.1 A picture is a grid of tiny squares (pixels)
Every photo in our training data is **1024 × 1024 = 1,048,576 pixels**, about 1 million tiny squares. The model's job is to put a label on each square: background, intact, damaged or destroyed.

### 2.2 A mask is a colouring sheet
A **mask** is a black-and-white copy of the photo, the same size. **White means "this square is X", black means "not X".**

The dataset gives **up to three separate sheets per photo**:
- an *intact* sheet, where white marks intact buildings;
- a *damaged* sheet, where white marks damaged buildings;
- a *destroyed* sheet, where white marks destroyed buildings.

**A sheet is listed only if the photo actually has that kind of building.** A photo with no destroyed buildings has no destroyed sheet.

### 2.3 Where the sheets live
Our model reads only these three folders:

```
DisasterM3_Instruct/masks/masks/
├── train_building_intact_mask/      → class 1 (Intact)
├── train_building_damaged_mask/     → class 2 (Damaged)
└── train_building_destroyed_mask/   → class 3 (Destroyed)
```

- The other seven folders in `masks/masks/` (roads, flooding, lava) belong to other DisasterM3 tasks and are never read.
- Each building folder holds 9,746 PNGs, but the manifest lists only the ones that actually contain that class. For optical photos that's **10,667 sheets**. The unlisted files are either completely black (all 15,813 were checked and none has a labelled pixel) or belong to SAR images, which we skip.

**Example: one photo on the Kaggle mirror**

```
Post-disaster image (input):
/kaggle/input/datasets/abrarmohammedtanzim/disasterm3-mirror/DisasterM3_Instruct/train_images/train_images/bata_explosion_post_0.png

Its sheets in the dataset (this photo has no destroyed buildings, so no destroyed sheet):
/kaggle/input/datasets/abrarmohammedtanzim/disasterm3-mirror/DisasterM3_Instruct/masks/masks/train_building_intact_mask/bata_explosion_0.png    149,303 white pixels
/kaggle/input/datasets/abrarmohammedtanzim/disasterm3-mirror/DisasterM3_Instruct/masks/masks/train_building_damaged_mask/bata_explosion_0.png      1,457 white pixels

Combined 4-class mask built by Cell 4 (exists only while the Kaggle session runs):
/tmp/combined_masks/bata_explosion_post_0.png
```

- **Names:** images are named `<event>_post_<N>.png` and sheets `<event>_<N>.png`. The manifest links them by putting both paths in the same entry. A `bata_explosion_pre_0.png` (before the disaster) also exists, but our model doesn't use it.
- **Folder nesting:** `masks/masks/` is how the Kaggle mirror is laid out. `train_images/train_images/` is assumed to follow the same pattern. The notebook's `resolve_path` tries both nested and flat paths, so either works.
- **On Hugging Face** the same files are inside `train_images.zip` and `masks.zip` (see 2.8).

### 2.4 From 92,968 entries to 6,443 photos
`train_release.json` is a big list of **tasks**, not images. DisasterM3 was built for chatbot-style vision-language models, so most tasks are questions and captions. The notebook's log lines are steps down this ladder:

```
92,968  every task in the list (questions, captions, counting, colouring...)    "Loaded 92,968 entries from manifest"
   └─ 37,204  colouring (segmentation) tasks: roads, buildings, floods, lava
        └─ 14,531  building colouring tasks, optical + radar (SAR) photos          "Building Damage Assessment entries: 14,531"
             └─ 10,667  building sheets from optical photos only  ← we use these
                  └─ 6,443  photos once each photo's sheets are stacked              "Unique images with at least one mask: 6,443"
                       ├─ 5,798  photos to learn from (training)                     "Train: 5,798 images"
                       └─   645  photos to test on (validation)                      "Val: 645 images"
```

The 37,204 colouring tasks break down as:

| Segmentation task | Entries |
|---|---|
| Road damage | 21,006 |
| **Building damage (ours)** | **14,531** |
| Flooding areas | 801 |
| Intact buildings near flooding | 801 |
| Intact buildings near lava | 33 |
| Lava areas | 32 |
| **Total** | **37,204** |

So "37.2k segmentation masks" is the whole colouring pile. Our model uses only the building sheets from optical photos. An "entry" is one photo plus one sheet plus a text prompt, and some sheets appear twice with different prompts. For example, the 14,531 building entries point to 13,425 unique files.

### 2.5 Why 10,667 sheets become 6,443 photos
**10,667 counts sheets; 6,443 counts photos.** Think of a school with **6,443 students** (photos) and **10,667 certificates** (sheets). Some students have 1 certificate, some 2, some 3:

| Photo has… | Photos | Sheets |
|---|---|---|
| 1 sheet | 3,349 | 3,349 |
| 2 sheets | 1,964 | 3,928 |
| 3 sheets | 1,130 | 3,390 |
| **Total** | **6,443 photos** | **10,667 sheets** |

Which classes appear together:

| Classes in the photo | Photos |
|---|---|
| intact only | 2,669 |
| intact + damaged + destroyed | 1,130 |
| intact + damaged | 966 |
| intact + destroyed | 619 |
| damaged only | 423 |
| damaged + destroyed | 379 |
| destroyed only | 257 |

Cell 4 stacks each photo's sheets into **one coloured map** with values 0 (background), 1 (intact), 2 (damaged) and 3 (destroyed):

```
intact sheet    (white → 1) ┐
damaged sheet   (white → 2) ├─► 1 combined mask (values 0/1/2/3) for 1 photo
destroyed sheet (white → 3) ┘   (only the sheets that exist for that photo)
```

Nothing is thrown away; the sheets are just combined. If two sheets mark the same pixel, the later class wins, so destroyed overrides damaged and damaged overrides intact.

### 2.6 How rare each class is
These statistics are over all 6,443 combined masks.

**How many photos contain each class:**

| Class | Photos containing it | Share |
|---|---|---|
| Background | 6,443 | 100% (every photo has ground, roads or trees) |
| Intact | 5,384 | 83.6% |
| Damaged | 2,898 | 45.0% |
| Destroyed | 2,385 | 37.0% |

These match the sheet counts: 5,384 + 2,898 + 2,385 = **10,667**. There is one sheet per class per photo, so "photos containing intact" equals "intact sheets".

**How many pixels belong to each class.** All photos together hold 6,443 × 1,048,576 = **6,755,975,168 pixels**. The easier way to read it is **out of every 1,000 pixels**:

| Class | Total pixels | Out of 1,000 | In an average photo |
|---|---|---|---|
| Background | 6,267,276,585 (92.77%) | **928** | about 972,700 |
| Intact | 395,903,867 (5.86%) | **59** | about 61,400 |
| Damaged | 70,675,310 (1.05%) | **10** | about 11,000 |
| Destroyed | 22,119,406 (0.33%) | **3** | about 3,400 |

Picture a stadium with 1,000 people: 928 wear grey (background), 59 wear green (intact), 10 wear orange (damaged), and **only 3 wear red (destroyed)**.

**Destroyed pixels per photo (the histogram):**
- **4,058 photos (63%) have zero destroyed pixels**, because 6,443 − 2,385 = 4,058.
- Most photos that do have destroyed buildings have only a little.
- A handful have a lot. The extreme case is one photo at about 465,000 destroyed pixels, about 44% of the image.
- The histogram's vertical axis is logarithmic (1 → 10 → 100 → 1,000), so its tall bars are far taller than they look.

### 2.7 Why this matters for the model
1. **A lazy model looks good on simple accuracy.** If it said "background" for every pixel, it would be right **92.8% of the time** while finding zero buildings. That's why we measure **IoU per class** instead of overall accuracy.
2. **Destroyed is the hardest class.** It's **3 pixels in 1,000**, and **63% of photos** have none at all, so the model rarely sees examples. That's why Destroyed IoU sits at about 0.27–0.29 while background is about 0.95.
3. **Every rare-class trick in the recipe exists for this imbalance:**
   - Focal Tversky makes missing a rare pixel cost more;
   - the sampler shows the 37% of photos with destroyed buildings more often;
   - copy-paste adds extra destroyed buildings to training crops (section 4.5).

**In one sentence:** 6,443 photos of about 1 million pixels each, where 93% of pixels are background and only 0.3% are destroyed, and the whole project is about teaching the model to find that 0.3%.

### 2.8 Other things in the dataset
Hugging Face: `Kingdrone-Junjue/DisasterM3`. Kaggle mirror: `abrarmohammedtanzim/disasterm3-mirror`.

| Folder or file | Size | What it is | Used by our model? |
|---|---|---|---|
| `train_images/` | 29.7 GB, 20,988 PNGs | Pre- and post-disaster photos (9,746 pre, 11,242 post), optical and SAR, for every task | Only the 6,443 optical post-disaster photos |
| `masks/` | 173 MB, 50,279 PNGs | Black-and-white sheets in 10 folders: building intact/damaged/destroyed (9,746 each), road intact/flooded/debris-covered (6,458 each), `flooding_mask` (801), `intact_buildings_near_flooding` (801), `intact_buildings_near_lava` (33), `volcano_lava` (32) | Only the 3 building folders |
| `box_train_images/` | 2.96 GB, 1,882 photos | Post-disaster photos with red and blue boxes drawn on them, for the VLM's "relational reasoning" questions | No |
| `train_release.json` | 87 MB | The manifest: 92,968 task entries | Yes, to find photos and sheets |
| `DisasterM3_Bench/` | — | The official **test split**: `test_images/`, `masks/` (`test_building_*_mask`, ...) and `benchmark_release.json` (30,042 entries). It has 2,067 optical building-damage photos with masks | Not yet. It could serve as a true held-out test set for the final champion; check for overlap with the training photos first |
| `.cache/huggingface/` (Kaggle mirror only) | tiny | Leftover download metadata from when the mirror was created | No; it's not data |

---

## 3. Architecture: U-Net with a ResNet-50 encoder

The notebook builds it in one line (Cell 8):

```python
smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=4)
```

```
Image 3×1024×1024
 ENCODER (ResNet-50, ImageNet-pretrained)          DECODER (starts from random weights)
  1/2  · 64 ch   ───────────── skip ─────────────►  up×2 + merge → 32 ch  (1/2)
  1/4  · 256 ch  ───────────── skip ─────────────►  up×2 + merge → 64 ch  (1/4)
  1/8  · 512 ch  ───────────── skip ─────────────►  up×2 + merge → 128 ch (1/8)
  1/16 · 1024 ch ───────────── skip ─────────────►  up×2 + merge → 256 ch (1/16)
  1/32 · 2048 ch ── bottleneck ──────────────────►  start          up×2 → 16 ch (full size)
                                                                     │
                                         conv head → 4 scores per pixel → highest wins
```

| Part | Job | Detail |
|---|---|---|
| **Encoder** (left side of the U) | Works out **what** is in the image | ResNet-50 pretrained on ImageNet. 5 stages; each halves the resolution and adds channels (64 → 2048). Early layers detect edges and textures; deep layers recognise roofs, rubble and roads. Exact position gets blurry on the way down. |
| **Bottleneck** | Most abstract summary of the scene | 1/32 resolution, 2048 channels (32×32 for a 1024 image). |
| **Decoder** (right side) | Recovers **where** things are | 5 blocks. Each upsamples ×2, joins the matching skip connection, then applies two 3×3 conv + BatchNorm + ReLU layers. Channels shrink 256 → 16 until it is back at full size. |
| **Skip connections** | Restore fine detail | Each decoder block receives the encoder map at the same resolution, which brings back sharp building edges. Without them, buildings come out as blobs. This is the defining U-Net idea. |
| **Head** | Turns features into a prediction | A 3×3 conv gives 4 scores (logits) per pixel. Softmax turns them into probabilities, and the highest one becomes the pixel's class. |

**Why one model handles 768 crops and full 1024 images:** U-Net is *fully convolutional*, so it accepts any input whose sides are divisible by 32. We train on 768×768 crops and evaluate on the whole image, padded to a multiple of 32. That is why the D4 notebook needs no sliding window.

**Other architectures the team tried** keep the encoder idea and change the decoder: U-Net++ (nested, denser skips; B2/B3, and the planned R4), DeepLabV3+ (atrous convolutions; B4) and MAnet (attention blocks; B5).

---

## 4. Fine-tuning: how the mask teaches the model

### 4.1 Starting point
- **Encoder:** starts from ImageNet weights (about 1.2M everyday photos), so it already knows generic edges, textures and shapes.
- **Decoder and head:** start from random weights.
- **Nothing is frozen.** Every weight in the encoder, decoder and head is updated. "Fine-tuning" here means *transfer learning*: adapting general visual features to aerial imagery and building damage.

### 4.2 The data
Section 2 covers this in detail. In short: 6,443 optical post-disaster photos, each with one 4-class mask built by Cell 4 from the dataset's binary sheets. `train_test_split(pairs, test_size=0.1, random_state=42)` gives **5,798 training** and **645 validation** photos, and every experiment uses this same split.

### 4.3 One training step
1. **Batch:** take 8 training images. In R5 the sampler decides which images are picked.
2. **Augment:** random 768×768 crop at native resolution, plus flips, 90° rotations and brightness/contrast changes (and copy-paste in R2). The mask gets exactly the same crop, flip and rotation.
3. **Forward pass:** the image goes through the encoder, decoder and head, giving 4 scores per pixel.
4. **Loss:** compare those scores with the mask's label **at every pixel**. **This is the only place the mask is used.**
5. **Backpropagation:** work out how much each weight contributed to the error.
6. **Update:** AdamW nudges every weight to reduce the error (lr 1e-4, weight decay 1e-5, gradient clipping at 1.0, mixed precision).

724 batches per epoch × 40 epochs ≈ **29,000 weight updates**. The learning rate follows a cosine curve from 1e-4 down to almost zero.

### 4.4 What it ends up learning
Through those updates, the encoder learns which visual features signal each class, and the decoder learns to draw them precisely:

| Class | Typical visual cue |
|---|---|
| Intact | Clean roof with regular, straight edges |
| Damaged | Partial collapse, roof holes, debris on or around the roof |
| Destroyed | Rubble with no recognisable building outline left |
| Background | Roads, vegetation, water, bare ground, everything else |

### 4.5 Why the loss is unusual: rare classes
Damaged and destroyed pixels are only 1.05% and 0.33% of all pixels (section 2.6). With a plain loss, the model can score well by predicting mostly background and intact. The recipe pushes back in two ways:

- **Loss side**
  - **Focal Tversky** (α = 0.3, β = 0.7, γ = 2.0). β > α makes a *missed* pixel (false negative) cost more than a false alarm (false positive). γ makes the model focus on the pixels it still gets wrong.
  - **Lovász-Softmax** for the last 20% of training (epochs 32–40, loss = 0.6 × Focal Tversky + 0.4 × Lovász). It optimises IoU directly, which is the metric we report.
- **Data side**
  - **Weighted sampler** (R5): images containing destroyed buildings are shown 10× as often, and damaged 5×.
  - **Copy-paste** (R2): extra damaged or destroyed building instances, taken from training images only, are pasted onto training crops.

---

## 5. Fine-tuning vs inference

The **forward pass** (encoder, then decoder with skips, then head) is identical in both phases. The **mask and the loss exist only in fine-tuning.**

| | Fine-tuning (training) | Inference |
|---|---|---|
| Encoder and decoder run (forward pass) | ✅ | ✅ |
| Mask used | ✅ (the answer key) | ❌ (there isn't one) |
| Loss computed | ✅ | ❌ |
| Weights change | ✅ | ❌ (frozen) |
| Random crops, flips, copy-paste, sampler | ✅ | ❌ |
| BatchNorm | uses the current batch's statistics (`model.train()`) | uses the averages saved from training (`model.eval()`) |
| TTA (8 flipped/rotated views averaged) | ❌ | ✅ (optional) |

- **Fine-tuning:** the encoder *learns* what each class looks like, and the decoder *learns* how to draw sharp outlines.
- **Inference:** the model *applies* what it learned. There is no mask, no loss and no learning.

**Validation during training is also inference.** Every 2 epochs, the notebook switches to `model.eval()` and `torch.no_grad()`, predicts the 645 validation images and compares them with their masks *only to compute a score*. That comparison never updates the weights. The reported checkpoint is the **last epoch**; `best_by_val.pth` is kept for reference only.

**How we run inference for reported numbers (E-protocol):**
- the full native image, padded to a multiple of 32, batch size 1;
- optionally 8-view TTA (4 rotations × with/without a horizontal flip), averaging the softmax probabilities before choosing the class;
- metrics from one confusion matrix over all 645 validation images.

---

## 6. What this means for our experiments

| Change | Phase it affects | What it changes | Runs |
|---|---|---|---|
| Loss parameters | Training | What the weights learn | R3 |
| Weighted sampler | Training | What the weights learn | R5 |
| Copy-paste | Training | What the weights learn | R2 |
| Enhanced augmentation | Training | What the weights learn | R8 |
| Decoder architecture | Training (and model shape) | What the weights learn | R4 (planned) |
| Seed | Training | Random choices only; measures noise | R6 |
| TTA | Inference only | How the same frozen weights are used. Added +0.003 to +0.007 mIoU in every D4 run. | all D4 runs (TTA row) |
| Evaluation protocol (full image vs centre crop, last vs best epoch) | Scoring only | The **number**, not the model | E-protocol vs older runs |

The last row is why results from different notebooks cannot be compared directly. For example, the FULLSTACK run's Destroyed IoU of 0.3013 was scored on a 768 centre crop using the best epoch, while the D4 runs score the full image using the last epoch. See [Ablation_Study_Reference_and_QA_Prep.md](Ablation_Study_Reference_and_QA_Prep.md) for the errata on older results.

---

## 7. One-paragraph answer for a viva

*"Our model is a U-Net with a ResNet-50 encoder. The encoder, pretrained on ImageNet, compresses the image and recognises what is in it. The decoder upsamples back to full resolution, and skip connections from the encoder restore sharp building edges, so every pixel gets one of four labels: background, intact, damaged or destroyed. We fine-tune the whole network on 6,443 DisasterM3 post-disaster images, where only 0.33% of pixels are destroyed buildings. For each training crop, the loss compares the predicted label with the ground-truth mask at every pixel, and backpropagation updates the weights. Because destroyed pixels are so rare, we use a Focal Tversky loss that penalises missed damage more than false alarms, then add Lovász to optimise IoU directly. At inference the weights are frozen: we run only the forward pass, optionally averaging 8 flipped and rotated views (TTA), and the mask is used only to score the result."*
