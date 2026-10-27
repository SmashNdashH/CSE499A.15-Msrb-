"""Turns one after-image into input.bin for the browser test, with L1's preprocessing
(RGB, ImageNet mean and std applied to 0-255 values, channels first, float32 little-endian)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if len(sys.argv) != 2:
    sys.exit('usage: python prep.py "<path to a 1024 x 1024 xBD *_post_disaster.png>"')
src = Path(sys.argv[1])
here = Path(__file__).resolve().parent

img = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32)
assert img.shape == (1024, 1024, 3), f"expected a 1024 x 1024 RGB image, got {img.shape}"
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
x = np.ascontiguousarray(((img - MEAN) / STD).transpose(2, 0, 1)[None]).astype("<f4")

x.tofile(here / "input.bin")
(here / "results").mkdir(exist_ok=True)
(here / "results" / "meta.json").write_text(json.dumps({"image": src.name, "shape": list(x.shape)}, indent=1))
print(f"input.bin written from {src.name}: {x.shape}, {x.nbytes / 1e6:.1f} MB")
