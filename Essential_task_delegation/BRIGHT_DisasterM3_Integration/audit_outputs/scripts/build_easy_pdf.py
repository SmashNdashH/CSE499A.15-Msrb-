"""Build the easy-English, corrected PDF of the DisasterM3 vs BRIGHT optical audit.

Numbers come from data/dm3_bright_optical_eda/summary.json and
data/bright_dm3_dedup_report.json. The HTML is printed to PDF with headless Edge.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\CSE499A.15-Msrb-")
EDA = ROOT / "data" / "dm3_bright_optical_eda"
OUT_DIR = ROOT / "Essential_task_delegation" / "BRIGHT_DisasterM3_Integration"
OUT_PDF = OUT_DIR / "DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf"
SCRATCH = Path(__file__).parent
HTML = SCRATCH / "audit_easy.html"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

S = json.load(open(EDA / "summary.json"))

ACCENT = "#2a78d6"
NEUTRAL = "#bdb9ad"
NEUTRAL_DARK = "#8f8c82"
NEUTRAL_LIGHT = "#e4e2dc"
GRID = "#ecebe7"
INK = "#1d1d1b"
INK2 = "#52514e"
MUTED = "#85837c"
FONT = "Segoe UI, Arial, sans-serif"


def t(x, y, s, size=12, anchor="start", weight=400, fill=INK2, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}" font-family="{FONT}" {extra}>{s}</text>')


def svg(w, h, body, label):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{label}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


# ---------------------------------------------------------------- chart 1: GSD
def chart_gsd():
    gsd = {e["event"]: e for e in S["geometry"]["bright_gsd_by_event"]}
    rows = [  # (label, ours, event key)
        ("Acapulco (Mexico)", False, "mexico-hurricane"),
        ("La Palma", True, "la_palma-volcano"),
        ("Turkey", False, "turkey-earthquake"),
        ("Morocco", False, "morocco-earthquake"),
        ("Goma (Congo)", True, "congo-volcano"),
        ("Derna (Libya)", False, "libya-flood"),
        ("Kyaukpyu (Myanmar)", False, "myanmar-hurricane"),
        ("Ukraine", False, "ukraine-conflict"),
        ("Maui (Hawaii)", False, "hawaii-wildfire"),
        ("Noto", True, "noto-earthquake"),
        ("Les Cayes (Haiti)", True, "haiti-earthquake"),
        ("Beirut", False, "beirut-explosion"),
        ("Bata", False, "bata-explosion"),
        ("Marshall (Boulder)", True, "marshall-wildfire"),
    ]
    W, top, rh = 760, 70, 25
    H = top + rh * len(rows) + 52
    x0, x1, vmax = 190, 590, 1.05

    def X(v):
        return x0 + v / vmax * (x1 - x0)

    b = []
    # legend
    b.append(f'<circle cx="12" cy="14" r="5.5" fill="{ACCENT}"/>' + t(24, 18, "measured: more detail than the paper says"))
    b.append(f'<circle cx="290" cy="14" r="5.5" fill="{NEUTRAL_DARK}"/>' + t(302, 18, "measured: same as the paper"))
    b.append(f'<circle cx="490" cy="14" r="5" fill="#fff" stroke="{NEUTRAL_DARK}" stroke-width="1.5"/>'
             + t(502, 18, "what the BRIGHT paper says (Table 2)"))
    # grid
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        b.append(f'<line x1="{X(v):.1f}" y1="{top-12}" x2="{X(v):.1f}" y2="{H-44}" stroke="{GRID}"/>')
        b.append(t(X(v), H - 28, f"{v:.1f}", 11, "middle", fill=MUTED))
    b.append(t((x0 + x1) / 2, H - 8, "metres per pixel (smaller number = more detail)", 11, "middle", fill=MUTED))
    b.append(f'<line x1="{X(0.8):.1f}" y1="{top-22}" x2="{X(0.8):.1f}" y2="{H-44}" stroke="{MUTED}" stroke-dasharray="4 3"/>')
    b.append(t(X(0.8) - 4, top - 26, "DisasterM3 paper says 0.8 m", 11, "end", fill=MUTED))
    b.append(t(620, top - 26, "measured vs paper", 11, fill=MUTED))
    for i, (label, ours, key) in enumerate(rows):
        e = gsd[key]
        y = top + i * rh + rh / 2
        med, lo, hi = e["gsd_median_m"], e["gsd_min_m"], e["gsd_max_m"]
        pmin, pmax = e["paper_gsd_m"]
        finer = med < 0.9 * pmin
        slight = (not finer) and med < 0.95 * pmin
        b.append(t(x0 - 12, y + 4, label, 12, "end", 700 if ours else 400, INK if ours else INK2))
        if pmax > pmin:
            b.append(f'<line x1="{X(pmin):.1f}" y1="{y}" x2="{X(pmax):.1f}" y2="{y}" stroke="{NEUTRAL_LIGHT}" stroke-width="6" stroke-linecap="round"/>')
        col = ACCENT if finer else NEUTRAL_DARK
        if hi > lo:
            b.append(f'<line x1="{X(lo):.1f}" y1="{y}" x2="{X(hi):.1f}" y2="{y}" stroke="{col}" stroke-width="2"/>')
        for p in sorted({pmin, pmax}):
            b.append(f'<circle cx="{X(p):.1f}" cy="{y}" r="5" fill="#fff" stroke="{NEUTRAL_DARK}" stroke-width="1.5"/>')
        b.append(f'<circle cx="{X(med):.1f}" cy="{y}" r="5.5" fill="{col}" stroke="#fff" stroke-width="1.5"/>')
        paper = f"{pmin:.2f}" if pmin == pmax else f"{pmin:.2f}–{pmax:.2f}"
        if finer:
            b.append(t(620, y + 4, f"{med + 1e-9:.2f} vs {paper} m", 12, fill=INK))
        elif slight:
            b.append(t(620, y + 4, f"{med + 1e-9:.2f} vs {paper} m (a little)", 12, fill=INK2))
    return svg(W, H, "".join(b), "Resolution of each BRIGHT event: measured versus the paper")


# ---------------------------------------------------------- chart 2: luminance
def chart_lum():
    h = S["radiometry"]["lum_hist_by_group"]

    def bins8(v):
        tot = sum(v)
        return [(v[i] + v[i + 1]) / tot * 100 for i in range(0, len(v), 2)]

    series = [
        ("DisasterM3 pool, post photos", bins8(h["DM3 pool post"]), NEUTRAL, ""),
        ("BRIGHT, all 11 events", bins8(h["BRIGHT pre (11 events)"]), NEUTRAL_DARK, 'stroke-dasharray="5 4"'),
        ("BRIGHT, our 5 events", bins8(h["BRIGHT pre (our 5 events)"]), ACCENT, ""),
    ]
    W, H = 760, 320
    x0, x1, y0, y1 = 56, 740, 270, 60
    ymax = 12

    def X(v):
        return x0 + v / 256 * (x1 - x0)

    def Y(p):
        return y0 - p / ymax * (y0 - y1)

    b = []
    lx = 0
    for name, _, col, dash in series:
        b.append(f'<line x1="{lx}" y1="14" x2="{lx+26}" y2="14" stroke="{col}" stroke-width="2.5" {dash}/>')
        b.append(t(lx + 32, 18, name))
        lx += 250
    for p in range(0, ymax + 1, 2):
        b.append(f'<line x1="{x0}" y1="{Y(p):.1f}" x2="{x1}" y2="{Y(p):.1f}" stroke="{GRID}"/>')
        b.append(t(x0 - 8, Y(p) + 4, f"{p}%", 11, "end", fill=MUTED))
    for v in (0, 64, 128, 192, 255):
        b.append(t(X(v), y0 + 18, str(v), 11, "middle", fill=MUTED))
    b.append(t((x0 + x1) / 2, y0 + 40, "pixel brightness (0 = black, 255 = white)", 11, "middle", fill=MUTED))
    for name, vals, col, dash in series:
        pts = " ".join(f"{X(i*8+4):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round" {dash}/>')
    for v, col, lab in ((60, ACCENT, "our 5 events: middle 60"), (84, NEUTRAL_DARK, "pool post photos: middle 84")):
        b.append(f'<line x1="{X(v):.1f}" y1="{y1-8}" x2="{X(v):.1f}" y2="{y0}" stroke="{col}" stroke-width="1.5" stroke-dasharray="2 3"/>')
    b.append(t(X(60) - 6, y1 - 14, "our 5 events: middle 60", 12, "end", 700, INK))
    b.append(t(X(84) + 6, y1 - 14, "pool post photos: middle 84", 12, "start", 700, INK))
    return svg(W, H, "".join(b), "Brightness of pixels: our five BRIGHT events are darker")


# -------------------------------------------------------------- chart 3: shift
def chart_shift():
    hist = S["registration"]["shift_hist_trusted"]
    labels = ["under 0.5", "0.5–1", "1–2", "2–3", "3–5", "5–10", "10–20", "over 20"]
    vals = list(hist.values())
    W, H = 760, 300
    x0, x1, y0, y1 = 20, 740, 250, 70
    vmax = 700
    n = len(vals)
    slot = (x1 - x0) / n
    bw = slot * 0.62
    b = []
    big = sum(vals[3:])
    tot = sum(vals)
    bx0 = x0 + 3 * slot + (slot - bw) / 2
    bx1 = x0 + 7 * slot + (slot + bw) / 2
    b.append(f'<path d="M{bx0:.1f},{y1-14} V{y1-22} H{bx1:.1f} V{y1-14}" fill="none" stroke="{ACCENT}" stroke-width="1.5"/>')
    b.append(t((bx0 + bx1) / 2, y1 - 30, f"{big:,} of {tot:,} pairs ({big/tot:.0%}) moved more than 2 pixels", 12, "middle", 700, INK))
    b.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{NEUTRAL}"/>')
    for i, (lab, v) in enumerate(zip(labels, vals)):
        cx = x0 + i * slot + slot / 2
        hgt = v / vmax * (y0 - y1)
        col = ACCENT if i >= 3 else NEUTRAL
        b.append(f'<path d="M{cx-bw/2:.1f},{y0} V{y0-hgt+4:.1f} q0,-4 4,-4 H{cx+bw/2-4:.1f} q4,0 4,4 V{y0} Z" fill="{col}"/>')
        b.append(t(cx, y0 - hgt - 6, f"{v:,}", 12, "middle", 400, INK))
        b.append(t(cx, y0 + 18, lab, 12, "middle", fill=INK2))
    b.append(t(x1, y0 + 42, "how far the post photo is moved from the pre photo (pixels)", 11, "end", fill=MUTED))
    return svg(W, H, "".join(b), "How far post photos are moved from pre photos")


# ------------------------------------------------------------- chart 4: labels
def chart_labels():
    L = S["labels"]
    rows = [
        ("DisasterM3 pool, learning", L["dm3_pool_train"], False),
        ("DisasterM3 pool, test (frozen val)", L["dm3_pool_val"], False),
        ("BRIGHT, all 14 events", L["bright_all"], False),
        ("BRIGHT, our 5 events", L["bright_our_5_events"], False),
        ("BRIGHT paper (for reference)", L["bright_paper"], True),
    ]
    W, top, rh = 760, 62, 34
    H = top + rh * len(rows) + 10
    x0, x1 = 230, 580
    b = []
    lx = 0
    for name, col in (("intact (not broken)", NEUTRAL_LIGHT), ("damaged", NEUTRAL), ("destroyed", ACCENT)):
        b.append(f'<rect x="{lx}" y="6" width="14" height="14" rx="3" fill="{col}"/>' + t(lx + 20, 18, name))
        lx += 170
    b.append(t(640, top - 12, "destroyed", 11, "end", fill=MUTED))
    b.append(t(756, top - 12, "photos with no", 11, "end", fill=MUTED))
    b.append(t(756, top + 1, "destroyed pixel", 11, "end", fill=MUTED))
    for i, (name, d, ref) in enumerate(rows):
        y = top + i * rh + 8
        op = ' opacity="0.55"' if ref else ""
        b.append(t(x0 - 12, y + 14, name, 12, "end", 700 if "our 5" in name else 400, INK if "our 5" in name else INK2))
        x = x0
        for k, col in (("intact", NEUTRAL_LIGHT), ("damaged", NEUTRAL), ("destroyed", ACCENT)):
            w = d[k] * (x1 - x0)
            if w > 0.5:
                b.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w-2,0.8):.1f}" height="20" rx="2" fill="{col}"{op}/>')
            if k == "intact":
                b.append(t(x + 8, y + 14, f"{d[k]*100:.1f}%", 11, fill=INK))
            if k == "damaged" and d[k] > 0.08:
                b.append(t(x + w / 2, y + 14, f"{d[k]*100:.1f}%", 11, "middle", fill=INK))
            x += w
        b.append(t(640, y + 14, f"{d['destroyed']*100:.1f}%", 12, "end", 700, INK))
        if "images_without_destroyed_px" in d:
            b.append(t(756, y + 14, f"{d['images_without_destroyed_px']*100:.1f}%", 12, "end", fill=INK))
        else:
            b.append(t(756, y + 14, "–", 12, "end", fill=MUTED))
    return svg(W, H, "".join(b), "Damage labels as a share of building pixels")


# ------------------------------------------------------------- chart 5: hazard
def chart_hazard():
    rows = [("fire", 2295, 77, 252), ("hurricane", 1502, 0, 168), ("volcano", 111, 1056, 8),
            ("flooding", 799, 0, 87), ("earthquake", 615, 152, 67), ("tornado", 336, 0, 39),
            ("tsunami", 72, 0, 11), ("explosion", 46, 0, 7), ("landslide", 22, 0, 6)]
    W, top, rh = 760, 58, 28
    H = top + rh * len(rows) + 6
    x0, x1, vmax = 100, 560, 2372
    b = []
    b.append(f'<rect x="0" y="6" width="14" height="14" rx="3" fill="{NEUTRAL}"/>' + t(20, 18, "DisasterM3 pool, learning photos"))
    b.append(f'<rect x="260" y="6" width="14" height="14" rx="3" fill="{ACCENT}"/>' + t(280, 18, "BRIGHT pre tiles, our 5 events"))
    b.append(t(756, top - 12, "test pairs", 11, "end", fill=MUTED))
    for i, (name, dm, br, val) in enumerate(rows):
        y = top + i * rh + 4
        bold = br > 0
        b.append(t(x0 - 10, y + 14, name, 12, "end", 700 if bold else 400, INK if bold else INK2))
        w1 = dm / vmax * (x1 - x0)
        b.append(f'<rect x="{x0}" y="{y}" width="{max(w1-2,1.5):.1f}" height="18" rx="2" fill="{NEUTRAL}"/>')
        w2 = br / vmax * (x1 - x0)
        if br:
            b.append(f'<rect x="{x0+w1:.1f}" y="{y}" width="{w2:.1f}" height="18" rx="2" fill="{ACCENT}"/>')
        lab = f"{dm:,} + {br:,}" if br else f"{dm:,}"
        b.append(t(x0 + w1 + w2 + 8, y + 14, lab, 12, "start", 700 if bold else 400, INK))
        b.append(t(756, y + 14, f"{val}", 12, "end", 700 if bold else 400, INK))
    return svg(W, H, "".join(b), "Learning photos by disaster type after adding our five BRIGHT events")


CHARTS = {
    "{{CHART_GSD}}": chart_gsd(),
    "{{CHART_LUM}}": chart_lum(),
    "{{CHART_SHIFT}}": chart_shift(),
    "{{CHART_LABELS}}": chart_labels(),
    "{{CHART_HAZARD}}": chart_hazard(),
}

html = (SCRATCH / "audit_easy_body.html").read_text(encoding="utf-8")
for k, v in CHARTS.items():
    html = html.replace(k, v)
HTML.write_text(html, encoding="utf-8")

cmd = [EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
       f"--user-data-dir={SCRATCH / 'edge-profile'}",
       f"--print-to-pdf={OUT_PDF}", HTML.as_uri()]
subprocess.run(cmd, check=True, timeout=120)
print("wrote", OUT_PDF, OUT_PDF.stat().st_size)
