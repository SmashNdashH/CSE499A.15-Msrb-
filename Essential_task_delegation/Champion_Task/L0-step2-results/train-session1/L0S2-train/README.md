# L0 step 2: top-up training of our 7B at full size

From `AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP` (held in 4-bit), a new LoRA adapter (rank 64, alpha 16) on the language model, trained on disaster type and relational reasoning plus a replay sample of the other tasks, at full size (up to 1036 x 1036). The original recipe otherwise: lr 2e-4, cosine, global batch 256, 1 epoch.

- Steps: 15 of 28 (not complete: run the notebook again to continue)
- Examples: 7,018 (full run: 7,018, {'Disaster Type Recognition': 1627, 'relational reasoning': 1882, 'disaster restoration advice': 533, 'Disaster Bearing Bodies Recognition': 510, 'Disaster Scene Recognition': 477, 'Building Damage Counting': 958, 'disaster caption': 535, 'Road Damage Counting': 496}); missing images: 0
- Longest example: 3,092 tokens; peak GPU memory on GPU 0: 14.2 GB
- Losses: [0.563, 0.538, 0.566, 0.577, 0.563, 0.546, 0.549, 0.555, 0.555, 0.522, 0.573, 0.576, 0.545, 0.532, 0.557]
- Last step: 10.04 s per example on both GPUs; projected for all 7,018 examples: 19.6 h
- Versions: {'torch': '2.11.0+cu128', 'transformers': '5.16.1', 'peft': '0.21.2', 'bitsandbytes': '0.50.2', 'accelerate': '1.14.0'}

Checkpoints: `AbrarAlam/disasterm3-qwen25vl7b-qlora/l0s2/checkpoints`. Merged result (Cell 6): `AbrarAlam/disasterm3-qwen2.5vl7b-l0s2-mergedFP`.