# T3 · In-browser segmenter test

Can the edge segmenter run inside a visitor's browser, so the public website can take uploads with no server? This kit runs L1b's INT8 model (and L1's full-precision model) in Chrome and Edge with ONNX Runtime Web, and compares every pixel with the same model run natively.

**Run on 24 Oct 2026 on Abrar's PC: passed.** The plan and the decision to try it are in [System_Design_Notes.md §4.3](../../System_Design_Notes.md).

## Pass rule (fixed before the run)

- It runs in both Chrome and Edge.
- It takes 10 s or less per 1024 × 1024 image after the first run.
- It gives the same class as the native run on at least 99.9% of pixels.

## Results (24 Oct 2026, two runs)

Setup: a PC with 20 threads and a GPU; headless Chrome 153 and Edge 153; ONNX Runtime Web and ONNX Runtime Node 1.20.1; one xBD after-image (`hurricane-harvey_00000099_post_disaster.png`, 202 buildings in its labels). The second run used a fresh copy of this kit, installed with `npm ci`, to check the kit itself.

| Model and engine | Chrome: first run / later runs | Edge: first run / later runs | Same class as the native run |
|---|---|---|---|
| INT8 (33 MB), WebAssembly | 3.3–4.0 s / 2.1–4.8 s | 4.8–6.4 s / 3.3–5.5 s | 100% of pixels, both runs |
| Full precision (130 MB), WebAssembly | 3.5–5.0 s / 3.9–6.5 s | 4.6–5.9 s / 3.7–4.3 s | 100% |
| Full precision, WebGPU | 4.8–5.6 s / 1.2–1.7 s | 3.4–3.8 s / 1.0–1.3 s | 100% |
| INT8, WebGPU | failed | failed | — |

- **Timings vary between runs on the same PC** (the second run was slower), so read them as ranges. Agreement and the per-class pixel counts were identical in both runs, and every later run stayed under the 10 s limit.
- **Native run** (ONNX Runtime Node, CPU): 0.75–1.05 s per image for both models, so the browser takes about 2.5–7 times as long on WebAssembly.
- **INT8 on WebGPU fails** because ONNX Runtime Web 1.20.1 can't run its DequantizeLinear operation on int32 values without a zero point. Possible fixes: a newer ONNX Runtime Web, re-quantizing with zero points, or a half-precision model (about 65 MB) for WebGPU.
- **Recommended setup:** INT8 on WebAssembly by default, with WebGPU as the faster option where the visitor's browser has it.
- The raw numbers are in `results/` (first run) and `results/run2/` (second run).

## How it's scored, and why

- **Agreement:** for each pixel, the class with the highest score among the 4 classes from the browser run is compared with the native run on the same input. Why: the question is whether the browser runs the same model faithfully. The model's accuracy against ground truth is its own property, already measured in L1b.
- **Time:** seconds for one `session.run` on one image, for the first run and three later runs. Why: a visitor waits for every image, and the first run includes one-time setup.
- **Alternatives considered:**
  - IoU against the native mask: gives the same answer when agreement is 100%.
  - Accuracy against ground truth on all 645 validation images: worth doing for R11's version, which is the model the website will ship.
  - The largest difference in raw scores (logits) between the two runs: stricter, but users see pixel classes. Worth adding if agreement ever drops below 100%; ONNX Runtime's own tests compare raw outputs with a tolerance.
  - A visual check: subjective.

## How to run it

Requirements: Node 20 or later (tested with 24), Python with numpy and Pillow, and Chrome and/or Edge.

1. In this folder, run `npm ci`. It installs the pinned packages from `package-lock.json`.
2. Get the two models. They aren't in git: `*.onnx` is ignored, and the full-precision file is 130 MB, over GitHub's limit. They come from the L1 and L1b outputs (`r9_int8_static_keep1_3.onnx` and `r9_fp32.onnx`); ask Abrar for the files or the Kaggle outputs. Save them in this folder as `int8.onnx` and `fp32.onnx`.
3. Run `python prep.py "<path to an xBD *_post_disaster.png>"`. It writes `input.bin`. For the same input as the 24 Oct run, use `hurricane-harvey_00000099_post_disaster.png` from xBD's tier1 train images.
4. Run `node ref.mjs`. It runs both models natively and writes `ref_int8.bin`, `ref_fp32.bin` and `results/native_results.json`.
5. Run `node run.mjs`. It serves this folder at `http://127.0.0.1:8765` with the isolation headers, opens the test page in headless Chrome and then Edge (each with a throwaway profile), and writes `results/result_chrome.json` and `results/result_edge.json`. If a browser is installed somewhere else, set `CHROME_PATH` or `EDGE_PATH`.
6. Compare the results with the pass rule above.

**On another laptop** (for example the booth laptop), run steps 1–6 on that laptop. **On a phone,** threads and WebGPU need HTTPS, so test phones on the Vercel deployment once it sends the two headers below.

## Caveats

- This PC is fast (20 threads), so re-test on the booth laptop and a mid-range phone.
- The test used R9, which sees only the after image. R11 needs its own INT8 export (L1 repeated on R11).
- Vercel must send two security headers, `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, or the browser runs on one thread.
- One image only. Timing depends on the image size, which is fixed at 1024 × 1024, not on its content; agreement could still differ on other images.

## Audit checklist

For a teammate, or an AI helper reviewing this kit:

- [ ] `package-lock.json` resolves `onnxruntime-web` and `onnxruntime-node` to 1.20.1.
- [ ] `prep.py` uses the same preprocessing as L1 (`l1-edge-benchmark.ipynb`): RGB, ImageNet mean and std applied to 0–255 values, channels first.
- [ ] The native run (`ref.mjs`) and the browser run (`index.html`) both read the same `input.bin`.
- [ ] Agreement is computed on every pixel of the full 1024 × 1024 output, by the highest of the 4 class scores.
- [ ] The results show `"isolated": true` and the thread count used. Without the isolation headers the run is single-threaded and slower.
- [ ] Times exclude the model download and page load; `load_s` is session creation.
- [ ] The INT8 WebGPU failure is a gap in ONNX Runtime Web 1.20.1, not a fault in the test.
- [ ] The numbers in `results/` and `results/run2/` match the table above.

## Files

| File | What it does |
|---|---|
| `prep.py` | Turns one after-image into `input.bin`, with L1's preprocessing |
| `ref.mjs` | Runs both models natively and saves their per-pixel classes and times |
| `index.html` | The test page: runs each model and engine in the browser and posts the results |
| `run.mjs` | Serves the folder with the isolation headers, drives headless Chrome and Edge, and saves the results |
| `package.json`, `package-lock.json` | The pinned ONNX Runtime packages |
| `results/` | The first 24 Oct run: `meta.json` (the input), `native_results.json`, `result_chrome.json`, `result_edge.json`; `run2/` holds the second run |
| `.gitignore` | Keeps the models, `input.bin`, the reference masks, `node_modules` and browser profiles out of git |
