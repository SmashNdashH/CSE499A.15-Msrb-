// Serves this folder with the isolation headers ONNX Runtime Web's threads need, opens the test page
// in headless Chrome and Edge (throwaway profiles), waits for the page to post its results, then closes the browser.
// Browser locations: set CHROME_PATH or EDGE_PATH if they differ from the Windows defaults below.
import http from "http"; import fs from "fs"; import path from "path"; import { spawn } from "child_process";
const ROOT = process.cwd(), PORT = 8765, DEBUG = 9333, OUT = path.join(ROOT, "results");
fs.mkdirSync(OUT, { recursive: true });
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const hdr = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin", "Cache-Control": "no-store" };
const server = http.createServer((req, res) => {
  const u = new URL(req.url, "http://localhost");
  if (req.method === "POST" && u.pathname === "/result") {
    let body = ""; req.on("data", (d) => (body += d));
    req.on("end", () => { fs.writeFileSync(path.join(OUT, `result_${u.searchParams.get("b")}.json`), body); res.writeHead(200, hdr); res.end("ok"); });
    return;
  }
  const p = path.join(ROOT, decodeURIComponent(u.pathname));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404, hdr); res.end(); return; }
  res.writeHead(200, { ...hdr, "Content-Type": types[path.extname(p)] || "application/octet-stream" });
  fs.createReadStream(p).pipe(res);
}).listen(PORT, "127.0.0.1");
const browsers = {
  chrome: process.env.CHROME_PATH || "C:/Program Files/Google/Chrome/Application/chrome.exe",
  edge: process.env.EDGE_PATH || "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
for (const [b, exe] of Object.entries(browsers)) {
  if (!fs.existsSync(exe)) { console.log(`=== ${b}: not found at ${exe}, skipped (set ${b.toUpperCase()}_PATH)`); continue; }
  const rf = path.join(OUT, `result_${b}.json`); if (fs.existsSync(rf)) fs.unlinkSync(rf);
  const proc = spawn(exe, ["--headless=new", `--remote-debugging-port=${DEBUG}`, `--user-data-dir=${path.join(ROOT, "profile_" + b)}`,
    "--no-first-run", "--enable-unsafe-webgpu", `http://127.0.0.1:${PORT}/index.html?b=${b}`], { stdio: "ignore" });
  const t0 = Date.now();
  while (!fs.existsSync(rf) && Date.now() - t0 < 540000) await sleep(1000);
  try {
    const v = await (await fetch(`http://127.0.0.1:${DEBUG}/json/version`)).json();
    const ws = new WebSocket(v.webSocketDebuggerUrl); await new Promise((r) => (ws.onopen = r));
    ws.send(JSON.stringify({ id: 1, method: "Browser.close" })); await sleep(3000);
  } catch { proc.kill(); }
  console.log(`=== ${b} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
  console.log(fs.existsSync(rf) ? JSON.stringify(JSON.parse(fs.readFileSync(rf, "utf8")), null, 1) : "TIMEOUT: no result");
  await sleep(2000);
}
server.close();
