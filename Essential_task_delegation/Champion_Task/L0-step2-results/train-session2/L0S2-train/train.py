# ── Cell 3 (written to train.py): L0 step 2's training, one process per T4, launched by Cell 4 ──
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import math
import random
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Trainer,
                          TrainerCallback, TrainingArguments)

WORK = Path("/kaggle/working/L0S2-train")
CFG = json.loads((WORK / "config.json").read_text(encoding="utf-8"))
SMOKE = CFG["smoke_test"]
RANK, WORLD = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
MAIN = RANK == 0
torch.cuda.set_device(RANK)
api = HfApi(token=os.environ.get("HF_TOKEN"))

def log(msg):
    if MAIN:
        print(msg, flush=True)

# ── Data: DTR and ORR, plus a replay sample of the other tasks, as the original notebooks built them ──
DATA_ROOT = Path(CFG["data_root"])
PATH_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

def resolve(rel_path):
    """A manifest path -> its file in the mirror, by the training notebooks' rule (None if missing)."""
    if not rel_path:
        return None
    rel_path = rel_path.replace("\\", "/")
    name = Path(rel_path).name
    for p in (DATA_ROOT / rel_path, DATA_ROOT / "train_images" / rel_path, DATA_ROOT / "box_train_images" / rel_path,
              DATA_ROOT / "train_images" / "train_images" / name,
              DATA_ROOT / "box_train_images" / "box_train_images" / name):
        if p.exists():
            return str(p)
    return None

def first(x):
    return (x[0] if x else "") if isinstance(x, list) else x

def answer_of(e):
    return str(first(e.get("training_answer", ""))).strip()

def example(e):
    """The images (before and after, or the boxed image), the training prompt and the training answer;
    None if an image is missing."""
    keys = ["image_path"] if "image_path" in e else ["pre_image_path", "post_image_path"]
    paths = [resolve(e.get(k, "")) for k in keys]
    if not all(paths):
        return None
    return {"images": paths, "prompt": str(first(e.get("prompts", ""))), "answer": answer_of(e), "task": e["task"]}

raw = json.load(open(DATA_ROOT / "train_release.json", encoding="utf-8"))
usable = [e for e in raw if e.get("task") != "Referring Expression Segmentation"     # masks, no text answer
          and answer_of(e) and not answer_of(e).lower().endswith(PATH_SUFFIXES)]
focus = [e for e in usable if e["task"] in CFG["focus_tasks"]]
pool = [e for e in usable if e["task"] not in CFG["focus_tasks"]]
replay = random.Random(CFG["seed"]).sample(pool, round(CFG["replay_ratio"] * len(focus)))
examples = [x for x in map(example, focus + replay) if x]
by_task = dict(Counter(x["task"] for x in examples))
missing = len(focus) + len(replay) - len(examples)
longest = max(examples, key=lambda x: (len(x["images"]), len(x["answer"])))   # the longest sequence
if SMOKE:       # the longest example (peak memory) and a random sample of the rest (time per example)
    rest = [x for x in examples if x is not longest]
    examples = [longest] + random.Random(CFG["seed"]).sample(rest, CFG["smoke_examples"] - 1)
log(f"✓ Data: {sum(by_task.values()):,} examples {by_task}; missing images: {missing}"
    + (f"; SMOKE TEST: {len(examples)} of them" if SMOKE else ""))

# ── Model: our merged 7B held in 4-bit, with a new LoRA adapter on the language model only ──
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CFG["start_model"], dtype=torch.float16, device_map={"": RANK}, attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
targets = [n for n, _ in model.named_modules() if n.rsplit(".", 1)[-1] in PROJECTIONS and "visual" not in n]
assert len(targets) == 28 * 7, f"expected the 196 language-model projections, found {len(targets)}"
model.enable_input_require_grads()      # gradient checkpointing needs a gradient path from the inputs
model = get_peft_model(model, LoraConfig(r=CFG["lora"]["r"], lora_alpha=CFG["lora"]["alpha"],
                                         lora_dropout=CFG["lora"]["dropout"], target_modules=targets,
                                         bias="none", task_type="CAUSAL_LM"))
trainable = [p for p in model.parameters() if p.requires_grad]
assert sum(p.numel() for p in trainable) == 161_480_704, "not the original's 161,480,704 LoRA parameters"
assert all(p.dtype == torch.float32 for p in trainable), "LoRA weights must be fp32 under fp16 mixed precision"
log(f"✓ Model: 4-bit on {WORLD} GPUs; LoRA on {len(targets)} language-model projections, vision tower untouched; "
    f"{torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU {RANK}")

# ── Batches: full-size images, the chat template, and the loss on the answer only ──
processor = AutoProcessor.from_pretrained(CFG["start_model"])     # the stock Qwen2.5-VL processor: full size
tok = processor.tokenizer
MARKER = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
VISION_IDS = torch.tensor([tok.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")])

def smart_resize(height, width, min_pixels, max_pixels, factor=28):
    """Qwen2.5-VL's rule for the size an image reaches the model at (its image processor's smart_resize)."""
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = math.floor(height / beta / factor) * factor, math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w

def sized(img):
    """The native image, as the evaluation feeds it; only an image above 1036 x 1036 is scaled down."""
    if img.width * img.height <= CFG["max_pixels"]:
        return img
    h, w = smart_resize(img.height, img.width, 56 * 56, CFG["max_pixels"])
    return img.resize((w, h), Image.BICUBIC)

def mask_prompt(row):
    """No loss up to and including the last assistant header: the loss is on the answer only, as in the original."""
    ids = row.tolist()
    for start in range(len(ids) - len(MARKER), -1, -1):
        if ids[start:start + len(MARKER)] == MARKER:
            row[:start + len(MARKER)] = -100
            return row
    raise ValueError("no assistant turn in the sequence")

def collate(batch):
    texts, images = [], []
    for x in batch:
        imgs = [sized(Image.open(p).convert("RGB")) for p in x["images"]]
        messages = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": x["prompt"]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": x["answer"]}]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        images += imgs
    out = processor(text=texts, images=images, padding=True, return_tensors="pt")
    labels = out["input_ids"].clone()
    labels[out["attention_mask"] == 0] = -100
    for row in labels:
        mask_prompt(row)
    assert not torch.isin(labels, VISION_IDS).any(), "image tokens in the loss"
    # The output layer (152k words) runs only from the token before the first answer token to the end. The model's
    # own loss shifts these labels by one, exactly as over the whole sequence; at full size this saves 2-4 GB.
    keep = labels.shape[1] - int((labels != -100).int().argmax(1).min()) + 1
    out["labels"], out["logits_to_keep"] = labels[:, -keep:], keep
    return out

probe = collate([longest])
log(f"✓ Longest example: {probe['input_ids'].shape[1]:,} tokens, {int(probe['image_grid_thw'].prod(1).sum()) // 4:,} "
    f"of them image tokens; the loss covers its last {probe['logits_to_keep'] - 1}")

# ── Training: the original recipe; every checkpoint goes to Hugging Face; a clean stop before Kaggle's limit ──
class Clock(TrainerCallback):
    """Times every optimizer step; past the time limit it saves (and so pushes) and stops."""
    def __init__(self):
        self.start = self.last = time.time()
        self.step_seconds = []
    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        self.step_seconds.append(round(now - self.last, 1))
        self.last = now
        if now - self.start > CFG["time_limit_h"] * 3600 and state.global_step < state.max_steps:
            log(f"⏰ Time limit after step {state.global_step}: saving, pushing and stopping")
            control.should_save, control.should_training_stop = True, True
        return control

class PushToHub(TrainerCallback):
    """Each saved checkpoint goes to the adapter repo; Cell 2 restores the latest one in a new session."""
    def on_save(self, args, state, control, **kwargs):
        if MAIN:
            name = f"checkpoint-{state.global_step}"
            api.upload_folder(folder_path=str(Path(args.output_dir) / name), repo_id=CFG["adapter_repo"],
                              path_in_repo=f"{CFG['hub_dir']}/{name}", commit_message=f"L0 step 2: {name}")
            log(f"✓ Pushed {name} to {CFG['adapter_repo']}/{CFG['hub_dir']}")

GLOBAL = WORLD * CFG["per_device_batch"] * CFG["grad_accum"]
steps = math.ceil(len(examples) / GLOBAL)
args = TrainingArguments(
    output_dir=CFG["output_dir"], num_train_epochs=1, per_device_train_batch_size=CFG["per_device_batch"],
    gradient_accumulation_steps=CFG["grad_accum"], learning_rate=CFG["lr"], adam_beta1=CFG["betas"][0],
    adam_beta2=CFG["betas"][1], weight_decay=CFG["weight_decay"], lr_scheduler_type="cosine",
    warmup_steps=math.ceil(CFG["warmup_ratio"] * steps), max_grad_norm=1.0, fp16=True,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1, save_strategy="steps", save_steps=CFG["save_steps"], save_total_limit=2, report_to="none", optim=CFG["optim"],
    dataloader_num_workers=2, remove_unused_columns=False, ddp_find_unused_parameters=False, seed=CFG["seed"])

class StepOnCpu(Trainer):
    """The trainer collects all 128 micro-batches of a step before training on any, and the data loader has
    already put each one on the GPU: with full-size images that ran out of memory (20 Oct). Keep the waiting
    batches on the CPU (images in fp16, the dtype the model casts them to); training_step moves each back in turn."""
    def get_batch_samples(self, epoch_iterator, *args, **kwargs):
        to_cpu = ({k: (v.to("cpu", torch.float16) if k == "pixel_values" else v.cpu()) if torch.is_tensor(v) else v
                   for k, v in b.items()} for b in epoch_iterator)
        return super().get_batch_samples(to_cpu, *args, **kwargs)

clock = Clock()
trainer = StepOnCpu(model=model, args=args, train_dataset=examples, data_collator=collate,
                    callbacks=[clock] + ([] if SMOKE else [PushToHub()]))
saved = sorted(Path(CFG["output_dir"]).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
resume = str(saved[-1]) if saved and not SMOKE else None
log(f"⏳ Training {len(examples):,} examples in about {steps} optimizer steps of {GLOBAL}"
    + (f", continuing from {resume}" if resume else ""))
trainer.train(resume_from_checkpoint=resume)

# ── What the run did, for Cell 5 ──
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"[GPU {RANK}] peak memory {peak:.1f} GB", flush=True)
if MAIN:
    state = trainer.state
    per_example = clock.step_seconds[-1] / GLOBAL if clock.step_seconds else None
    summary = {
        "smoke_test": SMOKE, "examples": len(examples), "full_examples": sum(by_task.values()), "by_task": by_task,
        "missing_images": missing, "global_step": state.global_step, "max_steps": state.max_steps,
        "complete": state.global_step >= state.max_steps, "resumed_from": resume,
        "longest_tokens": int(probe["input_ids"].shape[1]), "peak_gpu_memory_gb_gpu0": round(peak, 2),
        "step_seconds": clock.step_seconds, "seconds_per_example": per_example and round(per_example, 2),
        "projected_full_hours": per_example and round(per_example * sum(by_task.values()) / 3600, 1),
        "log_history": state.log_history,
        "versions": {p: version(p) for p in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}}
    (WORK / f"train_summary{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(summary, indent=1))
    log(f"✓ Step {state.global_step} of {state.max_steps}{' (complete)' if summary['complete'] else ''}; "
        f"{per_example and round(per_example, 1)} s per example on the last step")
