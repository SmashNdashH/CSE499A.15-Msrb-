# L0 step 2: the merge

`checkpoint-28` of `AbrarAlam/disasterm3-qwen25vl7b-qlora/l0s2/checkpoints` (step 28 of 28, LoRA rank 64, alpha 16) merged into `AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP` in fp16 on the CPU, and pushed as `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP` (public).

## Checks

- Weight tensors changed: 196 of 729, exactly the 196 language-model projections the adapter trained (none outside them, none of them unchanged).
- Changed tensors with inf or NaN: 0.
- Upload: 7 files, 16.6 GB; every file on the Hub has its local size.
- Last three training losses: [0.4905, 0.5365, 0.5335].

Versions: {'torch': '2.11.0+cu128', 'transformers': '5.16.1', 'peft': '0.21.2', 'safetensors': '0.8.0', 'huggingface_hub': '1.29.0'}. Merge 4.3 min, upload 2.4 min.

Next: `l0-step2-full-size-rescore.ipynb` with `MODEL = "new"`, in parts A, B and C.