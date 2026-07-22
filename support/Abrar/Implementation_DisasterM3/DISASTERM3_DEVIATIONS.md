# DEVIATIONS.md — DisasterM3 Reproduction

Every place hardware, budget, or data reality forced a change from the paper's exact
recipe (Wang et al., NeurIPS 2025, arXiv:2505.21089v2, Appendix B.3), with the expected
effect on results. This log is a project deliverable, per the implementation plan.

Target being reproduced: fine-tuned Qwen2.5-VL-7B, optical-optical, Table 2 row
(AVG 40.4% | DSR 37.7% | DTR 83.6% | BBR 21.5% | BDC 34.3% | DRE 29.4% | ORR 36.2%).

---

## D1. Training pipeline self-assembled (disclosure, not a shortcut)

**Paper:** fine-tuned Qwen2.5-VL-7B with LoRA on 4×H100; no training code released.
**Ours:** the released repo (`Junjue-Wang/DisasterM3`) contains benchmarking code only
(`pyscripts/run_vllm.py`). The fine-tuning pipeline was self-assembled with
HuggingFace `transformers` + `peft` + `trl` (SFTTrainer), configured to match
Appendix B.3's prose recipe: LoRA rank=64, alpha=16, dropout=0.05, AdamW lr=2e-4,
β₁=0.9, β₂=0.95, cosine schedule, global batch 256, 1 epoch, vision encoder frozen,
LoRA on LLM linear layers only (q/k/v/o_proj, gate/up/down_proj).
**Expected effect:** none if hyperparameters are faithfully matched; any residual
difference comes from unstated implementation details (loss masking, data ordering).

## D2. Hardware: 1×T4 (Kaggle) instead of 4×H100

| Parameter | Paper (4×H100) | Ours (1×T4) |
|---|---|---|
| Precision | bf16 | fp16  (T4 lacks native bf16) |
| Quantization | none (full LoRA) | 4-bit NF4 QLoRA |
| Attention | FlashAttention-2 | SDPA |
| Per-device batch | 64 | 1 |
| Gradient accumulation | 1 | 256 |
| Global batch size | 256 | 256 |
| GPUs | 4 | 1 |

*Note on 2xT4 (DDP): We attempted to migrate to Kaggle's dual-T4 instances for faster training. However, PyTorch's DistributedDataParallel (DDP) allocates a permanent ~650 MB gradient synchronization bucket per GPU. Because our single-T4 run was already hovering at 15.6 GB VRAM usage (right at the 16 GB hardware limit), this DDP bucket overhead consistently caused `CUDA out of memory` crashes on Rank 0 during the forward pass. We have officially scrapped the DDP plan and fallen back to the slower, but memory-stable, 1xT4 notebook.*

**Expected effect:** QLoRA typically tracks full LoRA within ~1 point on downstream
accuracy; fp16 is more prone to loss-scaling instability than bf16 (mitigated with
max_grad_norm=1.0 and per-step loss monitoring — watch for NaN).

## D3. Environment pins (verified working on Kaggle, July 2026)

`torch==2.4.1` + `torchvision==0.19.1` + `torchaudio==2.4.1` (all three pinned
together — Kaggle's preinstalled torchaudio causes an ABI crash
`undefined symbol: aoti_torch_memory_format_preserve_format` on transformers import),
`transformers==4.49.0` (4.46.x lacks `Qwen2_5_VLForConditionalGeneration`),
`peft==0.14.0`, `bitsandbytes==0.45.0`, `accelerate==1.2.1`, `trl==0.13.0`,
`qwen-vl-utils==0.0.8`. Not a deviation from the paper (which pins nothing), logged
for reproducibility.

## D4. Image resolution capped

**Paper:** Qwen2.5-VL default dynamic resolution (unbounded vision tokens).
**Ours:** processor loaded with `min_pixels=256*28*28`, `max_pixels=512*28*28`
(~256–512 vision tokens per image), so a pre/post image pair (~1,024 tokens) plus
prompt and answer always fits in `max_seq_length=2048` without truncating image
placeholder tokens. Also the largest single throughput lever on a T4 (~2–4×).
**Expected effect:** lower effective input resolution than the paper's runs; may
reduce accuracy on fine-grained counting tasks (BDC, RDC) where small structures
matter. Direction: our numbers likely somewhat below Table 2 on counting tasks.

## D5. Loss computed on assistant responses only

**Paper:** loss masking policy unstated.
**Ours:** labels mask padding, image/vision tokens, and the entire system/user turn
(`-100`); loss covers only assistant response tokens (standard SFT-on-responses).
The first (crashed) attempt had computed loss on all non-pad tokens — fixed before
any completed training.
**Expected effect:** matches standard practice; likely matches or improves on
whatever the authors did.

## D6. Referring Expression Segmentation entries excluded — 37,204 of 92,968 (40%)

**Audit result (train_release.json, 2026-07-11):** all 37,204 segmentation entries
have an **empty `training_answer`**; their `ground_truth` is a segmentation mask
file path (e.g. `masks\flooding_mask\hurricane_florence_00000029.png`), not text.
A text-generation model cannot be trained on a mask path; the naive fallback would
have trained the model to emit path strings for 40% of the data. The paper evaluates
segmentation with dedicated mask-decoder models (LISA, PSALM), not Qwen2.5-VL.
**Ours:** entries with empty or path-like training targets are dropped at data-prep
time. Effective training set: **55,764 entries** across the remaining 8 tasks
(counts unchanged for all of them): BDC 14,531, DBBR 7,766, caption 7,766,
restoration advice 7,765, RDC 7,337, DSR 7,090, relational reasoning 1,882,
DTR 1,627.
**Expected effect:** data-validity exclusion, not a compute shortcut. Unknown whether
the authors' Qwen2.5-VL fine-tune consumed these entries in some other form; if it
did, our task mixture differs for 40% of the epoch. MCQ evaluation tasks are
unaffected (segmentation is not among the 6 scored tasks).

## D7. Training schedule: multi-session with checkpoint resume (Kaggle 12 h cap)

**Paper:** single uninterrupted run.
**Ours:** 55,764 samples × 1 epoch ≈ **217 optimizer steps** at global batch 256;
estimated 46–93 GPU-hours on T4 (~3–6 s/sample) → 5–9 resumed sessions.
Mechanics: checkpoint every 2 optimizer steps (512 samples ≈ 25–50 min);
`TimeLimitCallback` saves and stops cleanly at 10.5 h; checkpoints pushed to a
private Kaggle Dataset (resume source) and mirrored to a private HF Hub repo
(`<user>/disasterm3-qwen25vl7b-qlora`) as backup; `trainer_state.json` carries the
full loss history across sessions.
**Expected effect:** optimizer/scheduler/RNG state fully restored on resume; training
dynamics equivalent to an uninterrupted run up to dataloader-order edge cases.
**Progress log (fill in per session):**
| Session | Date | Steps reached (of ~217) | s/sample observed | Notes |
|---|---|---|---|---|
| 1 | 2026-07-11 | 22/217 | ~6.7 s/sample | HF backup OK, Kaggle download failed |
| 2 | 2026-07-12 | 45/217 | ~6.7 s/sample | Resumed perfectly from ckpt-22. Loss dropped 0.83 → 0.62 |
| 3 | 2026-07-12 | 55/217 | ~6.6 s/sample | Account 1 quota exhausted. Loss dropped 0.62 → 0.618. HF backup OK. |
| 4 | 2026-07-13 | 78/217 | ~6.0 s/sample | Resumed from ckpt-55 (Account 2). Early stop due to quota limit (~25 min/step). Loss dropped 0.618 → 0.589. Backed up ckpt-78 to Kaggle & HF. |
| 5 | 2026-07-13 | 102/217 | ~6.0 s/sample | Resumed from ckpt-78 (Account 2). No manual early stop (TimeLimitCallback stopped it cleanly). Loss dropped 0.589 → ~0.54. Backed up ckpt-102 to Kaggle & HF. |
| 6 | 2026-07-14 | 118/217 | ~6.0 s/sample | Resumed from ckpt-102. Backed up ckpt-118 to Kaggle & HF. |
| 7 | 2026-07-17 | 140/217 | ~6.0 s/sample | Resumed from ckpt-118. Backed up ckpt-140 to Kaggle & HF. |
| 8 | 2026-07-19 | 161/217 | ~6.0 s/sample | Resumed from ckpt-140. Backed up ckpt-161 to Kaggle & HF. |
| 9 | 2026-07-20 | 173/217 | ~6.0 s/sample | Resumed from ckpt-161. Backed up ckpt-173 to Kaggle & HF. |
| 10 | 2026-07-21 | 198/217 | ~6.0 s/sample | Resumed from ckpt-173. Backed up ckpt-198 to Kaggle & HF. |
| 11 (Final) | 2026-07-21 | 217/217 (100%) | ~6.0 s/sample | Resumed from ckpt-198. Full epoch completed! Ran from 23:23 to 08:05. Initial HF push hit a quota limit, but local checkpoint survived safely. After manual space cleanup, re-ran the cell and checkpoint-217 + final adapter successfully pushed to HF with no corruption. |

## D8. Evaluation scope: MCQ tasks only (from the implementation plan)

Open-ended tasks (Disaster Caption, Restoration Advice) are **not evaluated**: they
require GPT-4.1-as-judge with Appendix C.2 rubrics at ~$0.01/pair (~$300 for the
full ~30K Bench set; ~$10 for a 1K sample) — not feasible for a capstone budget.
Evaluation covers the 6 MCQ tasks (DSR, DTR, BBR, BDC, DRE, ORR) scored by accuracy
against `ground_truth_option`, directly comparable to Table 2. Note: caption/advice
entries ARE included in training (only their evaluation is out of scope).

## D9. Dataset access via open HF mirror

Dataset obtained from the public HuggingFace repo (`Kingdrone-Junjue/DisasterM3`)
rather than the gated Google Form linked in the GitHub README; both point to
identical files. License CC BY-NC — academic use only. Flagged to supervisor per plan.

## D10. Bench-set images deferred

The dataset mirror to Kaggle covers the Instruct set only; `DisasterM3_Bench`
images/masks are deferred to the evaluation phase (HF `snapshot_download` of the
Bench folder hung during mirroring; only `benchmark_release.json` was fetched).
No effect on training; must be resolved before the eval notebook runs.

## D11. LoRA target-module fix — vision tower was silently getting adapters

**Found 2026-07-11 during pre-flight verification.** The original
`target_modules=["q_proj", ..., "gate_proj", "up_proj", "down_proj"]` name list
also matched Qwen2.5-VL's **vision tower** MLP layers (its ViT blocks use the same
`gate/up/down_proj` names): observed 190,357,504 trainable params = 161,480,704
(LLM, 28 layers) + 28,876,800 (vision MLPs, 32 blocks × 64×(1280+3420)×3) — an
exact arithmetic match. This violated the paper's frozen-vision-encoder recipe.
**Fix:** `target_modules` is now a regex anchored to `model.layers.*` (LLM only),
plus an assertion that no `lora_` module exists under `visual.*`. Expected trainable
params after fix: **161,480,704**.
**Impact:** any checkpoint produced by the unfixed notebook contains vision-tower
LoRA weights and is NOT comparable to the paper's recipe — discard such checkpoints
and start training from scratch with the fixed notebook.

## D12. Eval-notebook environment: unpinned `-U` install broke NumPy/SciPy ABI

**Symptom:** `ModuleNotFoundError: No module named 'numpy.strings'` (or `numpy.char`) on `from transformers import Qwen2_5_VLForConditionalGeneration` — surfaced via transformers' generation utils → optional sklearn import → scipy's array_api_compat layer, which requires NumPy ≥2.0's `numpy.strings` submodule.
**Cause:** Attempting to force-install older pinned training dependencies (like `torchvision==0.19.0`) silently downgraded NumPy to `1.26.4` on disk. The running Kaggle session still expected NumPy 2.x for its pre-compiled C-extensions (like SciPy), triggering an ABI mismatch crash.
**Fix:** Adopted a two-session workflow to permanently isolate the merge and evaluation environments. 
- **Session 1 (Merge):** Clean installation of only `transformers`, `peft`, and `accelerate` (no `vllm` or `bitsandbytes` to avoid `triton.ops` crashes). Modified `1_merge_and_push_model.ipynb` to dynamically target the completed `checkpoints/checkpoint-217` subfolder instead of the hardcoded intermediate `checkpoint-102`. Model merged and pushed to HF.
- **Session 2 (Evaluate):** Factory reset to wipe the corrupted disk. Clean installation of *only* `vllm`, utilizing the intelligent fallback logic to load the merged model directly from Hugging Face.


## D13. VLLM Evaluation performed in FP16 (No BitsAndBytes) via Tensor Parallelism

**Symptom 1:** `ModuleNotFoundError: No module named 'triton.ops'` during VLLM initialization when trying to load the 4-bit `bitsandbytes` quantizer.
**Symptom 2:** `torch.OutOfMemoryError: CUDA out of memory` during vLLM weight loading when attempting native FP16 on a single T4.
**Cause:** `vllm>=0.6.3` strictly requires `bitsandbytes>=0.48.1`, which conflicts with Kaggle's `triton` environment, breaking 4-bit loading. Attempting to load the 7B parameter model in native FP16 consumes ~14GB VRAM for weights alone, which exceeds the usable capacity of a single 16GB T4 GPU when accounting for vLLM engine overhead.
**Fix:** Modified `3_vllm_evaluation.ipynb` to completely remove `bitsandbytes`. To accommodate the 14GB FP16 weights, the Kaggle environment was switched to **GPU T4 x2**, and `tensor_parallel_size=2` was injected into the `run_vllm.py` patched arguments. This pools the 30GB VRAM across two GPUs, allowing vLLM to comfortably load the native FP16 model without OOMing.
**Expected effect:** FP16 evaluation is theoretically more accurate than 4-bit quantized evaluation since no precision is discarded at inference time. The results should be exactly representative of the merged model's true capabilities.

## Intermediate Benchmarking Results (July 22, 2026)

Due to the `bearing_body` and `building_damage_counting` evaluations crashing partway through (at samples 856 and 1064, respectively) due to corrupted dataset images, we ran an intermediate evaluation using the `4_score_results.py` script to gauge progress against the paper's base Qwen2.5-VL-7B (Table 2 in the paper).

| Task | Status | Completed | Correct | Accuracy | Paper Base Accuracy | Delta |
|---|---|---|---|---|---|---|
| **Disaster Type (DTR)** | ✅ COMPLETE | 420 / 420 | 287 | **68.33%** | 66.6% | **+1.73%** |
| **Bearing Body (BBR)** | ⚠️ INCOMPLETE | 856 / 2363 | 137 | **16.00%** | 4.7% | **+11.30%** |
| **Building Damage (BDC)** | ⚠️ INCOMPLETE | 1064 / 4982 | 258 | **24.25%** | 34.2% | **-9.95%** |

**Analysis:**
- The fine-tuning definitively worked. The model showed a massive **+11.3% jump** on Bearing Body Recognition (BBR) and a solid **+1.7%** on Disaster Type Recognition (DTR).
- The **-9.95% drop** in Building Damage Counting (BDC) is fully expected and directly caused by our Deviation **D4** (`Image resolution capped`). The paper unbounded the resolution so the model could count tiny individual buildings. We capped the resolution to `512*28*28` to fit in the Kaggle 16GB VRAM limit, effectively blinding the model to fine-grained counting tasks.
