# GLOSSARY.md — Terms, Parameters & Decision Definitions
### DisasterM3 Reproduction — CSE499A/B, Group 5

Purpose: every technical term, hyperparameter, and named decision used in the weekly
report and `DISASTERM3_DEVIATIONS.md` is defined here in plain language, with a
one-line justification and a pointer to the fuller deviation entry where relevant.
Organized so a specific question ("what is X, why did you pick it") can be answered
by looking up X directly.

---

## 1. Fine-tuning method terms

**LoRA (Low-Rank Adaptation).** Instead of updating all 7 billion parameters of the
base model, LoRA freezes the original weights and injects a small pair of trainable
"adapter" matrices next to certain layers. Only those small matrices are trained —
far fewer parameters, far less memory, and the adapter can be merged back into the
base model afterward or kept separate. This is the method the paper itself uses
(Appendix B.3), so using LoRA is exact-reproduction, not a deviation.

**QLoRA.** LoRA applied on top of a 4-bit **quantized** copy of the base model
(see "Quantization / NF4" below) instead of a full-precision copy. The adapter
matrices themselves still train in higher precision; only the frozen base weights
are compressed to 4 bits. This *is* a deviation from the paper (they used full LoRA
on unquantized bf16 weights on H100s) — logged as **D2**, made necessary because a
16GB T4 cannot hold a full-precision 7B model plus training memory (activations,
optimizer states, gradients) at once.

**Rank (LoRA rank = 64).** The size of the small adapter matrices — specifically,
their inner dimension. A higher rank means more trainable parameters per adapted
layer (more capacity to learn), at the cost of more memory and compute. Rank 64 is
the value stated in Appendix B.3; we matched it exactly (not a deviation).

**Alpha (LoRA alpha = 16).** A scaling factor applied to the adapter's output before
it's added back to the frozen layer's output. Alpha and rank together set the
adapter's effective learning "strength" — `alpha/rank` is the actual scale factor
applied. Taken directly from Appendix B.3.

**Dropout (LoRA dropout = 0.05).** During training, 5% of the adapter's internal
activations are randomly zeroed out on each forward pass, as a regularization
technique to reduce overfitting. Standard practice; taken from Appendix B.3.

**Target modules.** Which layers inside the model actually get a LoRA adapter
attached. The paper specifies "LLM linear layers only" with the vision encoder
frozen entirely. See **D11** for a real bug we caught here: the naive name-matching
list (specifically: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, 
and `down_proj`) accidentally also matched the vision tower's internal layers, which 
happen to use the same names — meaning an earlier draft was silently training 
vision-tower adapters in violation of the paper's frozen-encoder recipe. Fixed 
with a path-anchored regex plus an explicit assertion.

**Quantization / NF4 (4-bit NormalFloat).** A technique for storing the frozen base
model's weights using only 4 bits per number instead of the usual 16 or 32, cutting
memory roughly 4× at a small cost to numerical precision. NF4 specifically is a
data type designed to represent neural-network weight distributions more accurately
than a naive 4-bit format. Used only for the *frozen* base weights during
QLoRA training (see D2) — not used at inference time (see Section 3, "why FP16 not
4-bit for evaluation").

**Optimizer / AdamW.** In deep learning, the optimizer is the mathematical engine/algorithm responsible for physically changing the weights and biases of your model so that it actually learns.
- **The Problem:** During training, your model looks at a disaster image and makes a guess. The system calculates exactly how wrong that guess was (the Loss) and calculates the mathematical direction the weights need to shift to fix the error (the Gradients).
- **The Optimizer's Job:** The model cannot change its own weights. The optimizer (in your specific project, an algorithm called AdamW) takes those gradients, multiplies them by your Learning Rate step-size, and actively turns the "dials" (the weights) inside your LoRA adapters. Taken directly from Appendix B.3; not a deviation.

**β₁ / β₂ (Adam betas = 0.9 / 0.95).** Two smoothing coefficients internal to the
AdamW optimizer that control how much weight recent gradients get versus historical
ones. Values taken directly from Appendix B.3.

**Learning rate (2×10⁻⁴).** The step size used when updating weights each
optimizer step. Taken directly from Appendix B.3.

**Cosine (LR) schedule.** The learning rate doesn't stay fixed — it follows a curve
shaped like a cosine wave, starting near the target rate and smoothly decaying
toward zero by the end of training. Critically, the shape of this curve is computed
based on the **total number of planned steps** — if training is stopped early
(partial epoch), the rate never finishes decaying, which is why we deliberately let
the schedule run to its full, planned 217 steps rather than reporting an
intermediate checkpoint as final (see Section 4, "why we didn't stop early").

**Epoch.** One full pass through the entire training dataset. The paper trains for
exactly 1 epoch; so do we (on our reduced 55,764-entry set — see **D6**).

**Global batch size (256) vs. per-device batch size vs. gradient accumulation.**
"Global batch size" is how many training examples' gradients are averaged together
before one optimizer update happens — 256, matching the paper. A T4 cannot fit 256
examples in memory at once, so we process 1 example at a time ("per-device batch
size = 1") and accumulate (sum) the gradients over 256 such micro-steps before
actually updating the weights ("gradient accumulation steps = 256"). Mathematically
equivalent to a true batch of 256 in one shot; only the wall-clock speed differs.
Logged in **D2**.

**Checkpoint.** A saved snapshot of the model's current adapter weights (plus
optimizer/scheduler state) at a given training step, so training can be paused and
resumed later without starting over. Necessary because of Kaggle's 12-hour session
limit — see **D7**.

---

## 2. Hardware & precision terms

**T4.** The specific NVIDIA GPU model provided free on Kaggle/Colab — 16 GB VRAM,
based on the older "Turing" architecture (2018). Contrasted with the paper's
4×H100 (a much newer, far more powerful and expensive GPU generation). See **D2**.

**VRAM.** The GPU's own dedicated memory, separate from the computer's regular RAM.
Everything the GPU needs during training or inference — model weights, activations,
gradients, optimizer state — must fit inside VRAM, or the process crashes with an
"out of memory" (OOM) error.

**bf16 vs. fp16 (precision formats).** Two different ways of representing
non-integer numbers using 16 bits total. bf16 ("brain float 16") allocates more
bits to the exponent (better numerical range, more stable training) at the cost of
fewer bits for precision. fp16 has the opposite trade-off and is more prone to
numerical instability ("loss-scaling" issues, occasional NaN/inf values) during
training. **T4 hardware does not support bf16 natively** — this is a hardware
limitation, not a choice — so we use fp16 for training and mitigate instability
with gradient clipping (`max_grad_norm=1.0`). See **D2**.

**FlashAttention-2 vs. SDPA.** Two different low-level implementations of the
"attention" computation inside a transformer. FlashAttention-2 is faster and more
memory-efficient but requires a GPU of "compute capability" ≥8.0 (Ampere generation
or newer — A100, H100, RTX 30-series+). T4 is compute capability 7.5 and **cannot
run FlashAttention-2 at all** — this is a hardware/driver limitation. SDPA
("Scaled Dot-Product Attention," PyTorch's built-in fallback implementation) is
used instead. See **D2**.

**Compute capability.** A version number NVIDIA assigns to each GPU architecture
generation, used by software to determine which optimized code paths are
available. T4 = 7.5 ("Turing"); A100 = 8.0 ("Ampere"); H100 = 9.0 ("Hopper").
FlashAttention-2 requires ≥8.0, which is why T4 is excluded from using it.

**Tensor parallelism vs. data parallelism (DDP).** Two different ways to use
multiple GPUs together, easily confused:
- **Data parallelism (DDP)**, used during *training*, gives each GPU a full copy
  of the model and a different slice of the training batch, then synchronizes
  gradients across GPUs after each step. This synchronization step needs extra
  permanent memory on every GPU (a "gradient bucket") — this is what caused the
  2×T4 training attempt to OOM, documented in **D2**'s note, and is why training
  stayed on a single T4.
- **Tensor parallelism**, used during *inference/evaluation* (not training), splits
  the model's own weight matrices themselves across GPUs — each GPU holds and
  computes only half the model. No gradient synchronization is involved at all,
  since there's no training happening. This is a completely different mechanism
  from DDP and does **not** hit the same OOM cause — this is why 2×T4 was viable
  for evaluation (via `tensor_parallel_size=2`) even though it had already failed
  for training. See **D13**.

---

## 3. Serving / evaluation terms

**vLLM.** A specialized inference-serving library, separate from the `transformers`
library used for training, optimized for fast and memory-efficient text generation.
Used here because it's what the paper's own released benchmarking script
(`pyscripts/run_vllm.py`) is built on top of — using it (rather than a
hand-written generation loop) is what makes our benchmark numbers directly
comparable to Table 2, since we're reusing the authors' own scoring/parsing logic.

**Merge (`merge_and_unload`).** vLLM cannot efficiently serve a LoRA adapter sitting
on top of a 4-bit quantized base model the way training does. Before evaluation,
the adapter's small trained matrices are mathematically folded directly into the
base model's own weights, producing one ordinary (non-adapter, non-quantized) FP16
model. This "merge" step is a one-time, lossless mathematical operation, not
additional training.

**Why FP16 (not 4-bit) for evaluation, specifically.** Two independent blockers
ruled out running the merged model in 4-bit during evaluation: (1) vLLM's specific
4-bit loading path depends on a `bitsandbytes` version that conflicts with Kaggle's
installed `triton` version, crashing with `ModuleNotFoundError: No module named
'triton.ops'`; and (2) even if that were fixed, evaluating at full FP16 precision
is *more faithful* to the model's true learned capability than re-quantizing it a
second time purely for serving — quantization for evaluation would be an
additional, unnecessary source of error on top of the training-time QLoRA
approximation already logged in D2. See **D13**.

**`tensor_parallel_size`.** The vLLM engine argument that tells it how many GPUs to
split the model's weights across (see "Tensor parallelism" above). Set to 2 to
split the ~14GB of FP16 weights across two T4s (~7GB each), which is what actually
fixed the out-of-memory crash — see **D13**.

**`enforce_eager`.** A vLLM setting that disables an optional performance
optimization (CUDA graph capture + `torch.compile`) in exchange for simpler,
more predictable execution and lower one-time memory/setup overhead. Used here as
a stability choice on constrained hardware, at some cost to raw throughput.

**`gpu_memory_utilization`.** The fraction of total GPU VRAM vLLM is allowed to
claim for itself (weights + KV cache, see below) at startup. Set below 1.0 to leave
a safety margin for CUDA context and other overhead.

**KV cache.** During text generation, the model needs to remember intermediate
computations ("keys" and "values") for every token generated so far, so it doesn't
recompute them from scratch at every new token. This cache also consumes GPU
memory, on top of the model's own weights — part of what `gpu_memory_utilization`
is budgeting for.

**`max_model_len`.** The maximum total number of tokens (prompt + generated answer,
including image tokens — see D4) any single request is allowed to use. Set to
stay safely within memory limits.

**ABI mismatch (`numpy`/`scipy` crashes).** ABI stands for Application Binary
Interface. In Python data science environments, compiling C/C++ backed libraries 
like NumPy and SciPy against different underlying compiler versions can cause them
to become mutually incompatible. An unpinned dependency in an earlier evaluation 
script triggered this, causing immediate `ImportError` crashes. Fixed by 
engineering a strict, isolated session environment (see **D12**).

**Windows-style pathing bug (`FileNotFoundError`).** The original authors'
evaluation script hardcoded file paths using Windows backslashes (`\`). When run
on Kaggle's Linux-based servers, these paths failed to resolve, instantly crashing
the evaluation pipeline. We patched this dynamically at runtime by swapping
the backslashes to forward slashes (`/`).

**Hugging Face Hub.** A central cloud repository for machine learning models. We
pushed both our intermediate QLoRA adapters and our final merged 15GB FP16 model
here to ensure they were permanently backed up and easily downloadable into fresh
Kaggle sessions, preventing data loss from session timeouts.

---

## 4. Evaluation task acronyms & metrics

These are the paper's own task names (Section 2/Table 2), abbreviated for brevity:

| Acronym | Full task name | What it measures |
|---|---|---|
| **DSR** | Disaster Scene Recognition | Identifying land-use/scene types in the image |
| **DTR** | Disaster Type Recognition | Identifying which disaster occurred (flood, hurricane, etc.) |
| **BBR** | Bearing Body Recognition | Identifying key affected structures |
| **BDC** | (Damaged) Building Damage Counting | Counting destroyed/damaged buildings |
| **DRE** | Damaged Road (area) Estimation | Estimating flooded/damaged road area |
| **ORR** | Object Relational Reasoning | Reasoning about relationships between two marked objects |

**MCQ (multiple-choice question) evaluation.** For the six tasks above, the model
is shown a question with several lettered answer options and must output the
correct letter; scoring is a simple exact-match against `ground_truth_option`
(the field in the Bench-set manifest recording the correct answer). This is why
these six tasks — unlike Disaster Caption and Restoration Advice — don't need an
AI judge to score (see **D8**): correctness is unambiguous and mechanical.

**`ground_truth_option`.** The manifest field (in `benchmark_release.json`) storing
which lettered option is correct for a given MCQ item — the value our model's
output is compared against to compute accuracy.

**Accuracy.** Percentage of MCQ items where the model's chosen option exactly
matches `ground_truth_option`. The metric reported in Table 2 for all six MCQ
tasks, and the base metric we report.

**Delta vs Base.** The percentage point difference between our fine-tuned model's
accuracy and the paper's unfine-tuned (base) model's accuracy. Demonstrates the
raw uplift gained from our training process on classification tasks.

**Delta vs Paper FT.** The percentage point difference between our fine-tuned model
and the paper's own fine-tuned model (which used 4× H100s, full BF16, and unbounded
dynamic resolution). We underperformed the paper's FT model across the board, which
is the mathematically expected consequence of our hardware-constrained deviations
(specifically **D4**'s resolution cap and **D2**'s 4-bit precision loss).

**Why Caption / Restoration Advice are excluded from evaluation.** These two tasks
are free-form text (not multiple choice), so the paper scores them with GPT-4.1
acting as an automated judge against a detailed rubric (Appendix C.2). Running
that judge across the full ~30,000-item Bench set costs an estimated $300 (or ~$10
for a 1,000-item sample) — outside capstone budget, so this evaluation axis is
explicitly out of scope, logged in **D8**. Note this only affects *evaluation*;
both tasks were still included during *training*.

---

## 5. Why-this-decision quick index (maps directly to `DISASTERM3_DEVIATIONS.md`)

| If asked... | Short answer | Full justification |
|---|---|---|
| Why self-built training code? | Repo only ships benchmark code, no training script | D1 |
| Why QLoRA/fp16/SDPA instead of the paper's setup? | T4 hardware limitations (VRAM, no bf16, no FlashAttention-2) | D2 |
| Why these exact package versions? | Verified working combination on Kaggle; several version conflicts found and fixed | D3 |
| Why cap image resolution? | Fit sequence length in memory on T4; biggest available speed lever | D4 |
| Why mask loss to assistant tokens only? | Standard SFT practice; paper doesn't specify otherwise | D5 |
| Why drop 40% of the training data (segmentation)? | Those entries have no valid text target (mask file paths, not text) — cannot train a text model on them | D6 |
| Why train across 11 separate sessions? | Kaggle's 12-hour session cap; checkpoint-resume used to preserve an uninterrupted-equivalent training run | D7 |
| Why not evaluate Caption/Restoration Advice? | Requires paid GPT-4.1 judging, outside budget | D8 |
| Why HuggingFace mirror instead of the gated form? | Both point to identical files; open mirror used and disclosed | D9 |
| Why was Bench-set mirroring delayed? | Initial full-folder download hung; resolved before evaluation | D10 |
| What was the LoRA target-module bug? | Vision tower was accidentally receiving adapters, violating frozen-encoder recipe; fixed with path-anchored regex | D11 |
| Why the two-session merge/evaluate split? | An earlier unpinned install broke NumPy/SciPy ABI compatibility; isolating environments fixed it | D12 |
| Why 2×T4 + tensor parallelism for evaluation, not the same setup as training? | FP16 evaluation weights alone (~14GB) don't fit one T4; splitting across two GPUs (inference-only, no DDP conflict) does | D13 |
