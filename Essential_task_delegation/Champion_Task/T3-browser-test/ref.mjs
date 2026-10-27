// Native reference: the same models and input, run with onnxruntime-node on the CPU.
// Writes ref_int8.bin and ref_fp32.bin (the predicted class of every pixel) and results/native_results.json.
import fs from "fs";
import ort from "onnxruntime-node";

const buf = fs.readFileSync("input.bin");
const x = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const argmax = (t) => {
  const [, C, H, W] = t.dims, n = H * W, d = t.data, m = new Uint8Array(n);
  for (let i = 0; i < n; i++) { let b = 0, v = d[i]; for (let c = 1; c < C; c++) { const u = d[c * n + i]; if (u > v) { v = u; b = c; } } m[i] = b; }
  return m;
};
const out = {};
for (const name of ["int8", "fp32"]) {
  const s = await ort.InferenceSession.create(`${name}.onnx`, { graphOptimizationLevel: "all" });
  const feeds = { image: new ort.Tensor("float32", x, [1, 3, 1024, 1024]) };
  const times = [];
  let r;
  for (let k = 0; k < 4; k++) { const t0 = performance.now(); r = await s.run(feeds); times.push((performance.now() - t0) / 1000); }
  const m = argmax(r.logits);
  fs.writeFileSync(`ref_${name}.bin`, m);
  const hist = [0, 0, 0, 0]; for (const v of m) hist[v]++;
  out[name] = { first_s: +times[0].toFixed(2), later_s: times.slice(1).map((t) => +t.toFixed(2)), class_pixels: hist, output_dims: r.logits.dims };
  console.log(name, JSON.stringify(out[name]));
}
fs.mkdirSync("results", { recursive: true });
fs.writeFileSync("results/native_results.json", JSON.stringify(out, null, 1));
