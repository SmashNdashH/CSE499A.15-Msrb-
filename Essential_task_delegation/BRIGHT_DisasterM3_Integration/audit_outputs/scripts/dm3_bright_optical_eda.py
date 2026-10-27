"""
dm3_bright_optical_eda.py
=========================
Optical-only audit and EDA of DisasterM3 and BRIGHT: pre- and post-event images, SAR excluded.

Every number comes from the local copies:
  1. Inventory     images by dataset, split, role and modality; JSON vs disk vs published counts.
                   SAR is found twice: by DisasterM3's own tags and by pixels (equal channels = SAR).
  2. Geometry      tile sizes, BRIGHT ground sampling distance (GSD) from GeoTIFF tags,
                   building size in pixels, spectral sharpness.
  3. Radiometry    colour, brightness, contrast, saturation, no-data, haze; pre vs post.
  4. Registration  pre/post shift of every DisasterM3 optical pair (phase correlation).
  5. Labels        damage-class pixels and building components: DisasterM3 optical pool vs BRIGHT.
  6. Overlap       identical images across and within the datasets, and across the frozen split.

Writes summary.json (headline numbers and tables) and per-image / per-pair CSVs to --out.

Author: Ridita Afrin Riya (CSE499B Group 5)
Project: Resource-Efficient Neural Architectures for Post-Disaster Damage Assessment
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Worker processes re-import this module, so only cv2 + numpy load at the top; pandas, PIL, tqdm and
# bright_dm3_dedup (scikit-learn) load inside the functions that run in the main process.
sys.path.insert(0, str(Path(__file__).resolve().parent))
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # GeoTIFF tags are "unknown" to OpenCV

THUMB = 32
HIST_BINS = 64
AREA_EDGES = 2.0 ** np.arange(0, 21)        # component-area bins: 1, 2, 4 ... 1,048,576 px
BUILDING = {"train_building_intact_mask": 1, "train_building_damaged_mask": 2,
            "train_building_destroyed_mask": 3}
ROAD = {"train_road_intact_mask": "intact", "train_road_flooded_mask": "flooded",
        "train_road_debris_covered_mask": "debris"}
XBD_EVENTS = {"guatemala_volcano", "hurricane_florence", "hurricane_harvey", "hurricane_matthew",
              "hurricane_michael", "mexico_earthquake", "midwest_flooding", "palu_tsunami",
              "santa_rosa_wildfire", "socal_fire", "joplin_tornado", "lower_puna_volcano", "moore_tornado",
              "nepal_flooding", "pinery_bushfire", "portugal_wildfire", "sunda_tsunami",
              "tuscaloosa_tornado", "woolsey_fire"}
# BRIGHT paper (ESSD 17, 6217), Table 2: GSD range (m) and building count per event
BRIGHT_PAPER = {
    "beirut-explosion": ((1.0, 1.0), 25496), "bata-explosion": ((0.5, 0.5), 8893),
    "congo-volcano": ((0.33, 0.33), 18741), "haiti-earthquake": ((0.48, 0.48), 18918),
    "la_palma-volcano": ((0.30, 0.35), 30239), "marshall-wildfire": ((0.6, 0.6), 8365),
    "ukraine-conflict": ((0.6, 0.6), 56770), "turkey-earthquake": ((0.30, 0.35), 135033),
    "myanmar-hurricane": ((0.6, 0.6), 8052), "hawaii-wildfire": ((0.6, 0.6), 3995),
    "morocco-earthquake": ((0.35, 0.40), 6269), "libya-flood": ((0.35, 0.35), 10979),
    "mexico-hurricane": ((0.35, 0.8), 18437), "noto-earthquake": ((0.5, 0.5), 8153),
}
OUR_BRIGHT_EVENTS = {"la_palma-volcano", "congo-volcano", "haiti-earthquake", "marshall-wildfire",
                     "noto-earthquake"}
TILE_ID = re.compile(r"^(?P<event>.+?)_(?P<tile>\d+)_(?P<role>pre|post)_disaster$")
ROLE_ID = re.compile(r"^(?P<event>.+?)_(?P<sar>sar_)?(?P<role>pre|post)_(?P<tile>\d+(?:_\d+)?)$")


# ----------------------------------------------------------------------------- names
def name_of(rel):
    return Path(rel.replace("\\", "/")).name


def parse_dm3_name(name):
    """'turkey_earthquake3_sar_post_692.png' -> prefix, event, role, 'sar' in name, naming scheme."""
    stem = Path(name).stem.lower()
    m, scheme = TILE_ID.match(stem), "<event>_<id>_<role>_disaster"
    if not m:
        m, scheme = ROLE_ID.match(stem), "<event>_<role>_<id>"
    if not m:
        return {"prefix": stem, "event": stem, "role": None, "sar_in_name": "sar" in stem, "scheme": "other"}
    return {"prefix": m["event"], "event": re.sub(r"\d+$", "", m["event"]), "role": m["role"],
            "sar_in_name": bool(m.groupdict().get("sar")), "scheme": scheme}


def hazard(event):
    e = event.lower()
    if e.startswith("kalehe"):
        return "landslide"            # files say flooding; DisasterM3 Fig. 2 lists the Kalehe landslide
    for key, kind in (("volcano", "volcano"), ("fire", "fire"), ("hurricane", "hurricane"),
                      ("tornado", "tornado"), ("tsunami", "tsunami"), ("flood", "flooding"),
                      ("landslide", "landslide"), ("earthquake", "earthquake"),
                      ("explosion", "explosion"), ("conflict", "conflict")):
        if key in e:
            return kind
    return "other"


def origin(event):
    from bright_dm3_dedup import dm3_event
    if event in XBD_EVENTS:
        return "xBD"
    return "BRIGHT event" if dm3_event(event) else "new in DisasterM3"


# ----------------------------------------------------------------------------- workers
def read_image(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def hf_ratio(lum):
    """Share of spectral power in the upper half of the band (0.5-1.0 Nyquist) of the central
    512x512 crop. Upsampled or blurred imagery has almost none."""
    h, w = lum.shape
    if min(h, w) < 512:
        return float("nan")
    c = lum[h // 2 - 256:h // 2 + 256, w // 2 - 256:w // 2 + 256]
    c = (c - c.mean()) * np.outer(np.hanning(512), np.hanning(512))
    power = np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - 256, xx - 256) / 256
    band = power[(r > 0.05) & (r <= 1)].sum()
    return float(power[(r > 0.5) & (r <= 1)].sum() / band) if band > 0 else float("nan")


def image_stats(path):
    """Identity fields for every image; radiometry only for colour (optical) images."""
    img = read_image(path)
    if img is None:
        return {"readable": False}, None, None
    h, w = img.shape[:2]
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.int16)
    # Judge colour only on pixels that carry data: black no-data borders have identical channels too.
    # An all-black (blank) image has no colour to show and counts as single-channel.
    data = small.max(axis=2) > 3
    chan_diff = float(np.abs(small[..., 0] - small[..., 1])[data].mean() +
                      np.abs(small[..., 1] - small[..., 2])[data].mean()) if data.any() else 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    thumb = np.zeros_like(t) if t.std() < 2 else (t - t.mean()) / t.std() / np.sqrt(t.size)
    rec = {"readable": True, "h": h, "w": w, "chan_diff": round(chan_diff, 3), "single_channel": chan_diff < 1.0,
           "md5": hashlib.md5(str(img.shape).encode() + img.tobytes()).hexdigest()}
    if rec["single_channel"]:
        return rec, thumb, None

    b, g, r = (img[..., i].astype(np.float32) for i in range(3))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    valid = img.max(axis=2) > 3                       # not black no-data
    if not valid.any():
        rec.update(nodata_frac=1.0)
        return rec, thumb, None
    lum_v = lum[valid]
    inner = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    p1, p99 = np.percentile(lum_v, [1, 99])
    rec.update(
        mean_r=float(r[valid].mean()), mean_g=float(g[valid].mean()), mean_b=float(b[valid].mean()),
        lum_mean=float(lum_v.mean()), lum_std=float(lum_v.std()), lum_p1=float(p1), lum_p99=float(p99),
        sat_mean=float(sat[valid].mean() / 255),
        nodata_frac=float(1 - valid.mean()),
        clipped_frac=float((img.min(axis=2) >= 250).mean()),
        bright_lowsat_frac=float(((lum > 200) & (sat < 30) & valid).mean()),
        dark_channel=float(cv2.erode(small.min(axis=2).astype(np.uint8), np.ones((7, 7), np.uint8)).mean() / 255),
        lap_var=float(lap[inner].var()) if inner.any() else float("nan"),
        hf_ratio=hf_ratio(lum),
    )
    hist = np.histogram(lum_v, bins=HIST_BINS, range=(0, 256))[0].astype(np.int64)
    return rec, thumb, hist


def pair_shift(paths, size=256, min_response=0.1):
    """Pre->post translation in pixels, robust to damage and no-data: phase correlation on a 4x4 grid
    of 256-px tiles that are fully valid in both images; the median of tiles with a clear peak.
    Returns (dx, dy, tiles with a clear peak, valid tiles, spread = median tile distance from dx,dy)."""
    a, b = (read_image(p, cv2.IMREAD_GRAYSCALE) for p in paths)
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        b = cv2.resize(b, a.shape[::-1], interpolation=cv2.INTER_AREA)
    valid = (a > 3) & (b > 3)
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    win = cv2.createHanningWindow((size, size), cv2.CV_32F)
    tiles, n_valid = [], 0
    for y in range(0, a.shape[0] - size + 1, size):
        for x in range(0, a.shape[1] - size + 1, size):
            if valid[y:y + size, x:x + size].mean() < 0.98:
                continue
            n_valid += 1
            (dx, dy), response = cv2.phaseCorrelate(fa[y:y + size, x:x + size], fb[y:y + size, x:x + size], win)
            if response >= min_response:
                tiles.append((dx, dy))
    if not tiles:
        return float("nan"), float("nan"), 0, n_valid, float("nan")
    t = np.array(tiles)
    dx, dy = np.median(t, axis=0)
    return float(dx), float(dy), len(t), n_valid, float(np.median(np.hypot(t[:, 0] - dx, t[:, 1] - dy)))


def components(binary):
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return stats[1:, cv2.CC_STAT_AREA].astype(np.int32)


def pool_masks(job):
    """The 4-class map the champion notebooks build (later class wins), plus what that merge hides."""
    building, road = job
    raw = {}
    for folder, cid in BUILDING.items():
        m = read_image(building[folder], cv2.IMREAD_GRAYSCALE) if folder in building else None
        if m is not None:
            raw[cid] = m
    if not raw:
        return None
    shape = next(iter(raw.values())).shape
    combined, hits, values = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8), set()
    for cid in (1, 2, 3):
        if cid in raw:
            m = raw[cid] if raw[cid].shape == shape else cv2.resize(raw[cid], shape[::-1], interpolation=cv2.INTER_NEAREST)
            values.update(np.unique(m).tolist())
            combined[m > 0] = cid
            hits += m > 0
    rec = {"h": shape[0], "w": shape[1], "px": np.bincount(combined.ravel(), minlength=4)[:4].tolist(),
           "multi_label_px": int((hits > 1).sum()), "mask_values": sorted(values)}
    areas = {cid: components(combined == cid) for cid in (1, 2, 3)}
    areas["building"] = components(combined > 0)
    rec["road_px"] = {}
    for folder, name in ROAD.items():
        m = read_image(road[folder], cv2.IMREAD_GRAYSCALE) if folder in road else None
        rec["road_px"][name] = int((m > 0).sum()) if m is not None else 0
    return rec, areas


def combined_mask(building):
    """The notebooks' 4-class map from the binary building masks of one pool image."""
    out = None
    for folder, cid in BUILDING.items():
        m = read_image(building[folder], cv2.IMREAD_GRAYSCALE) if folder in building else None
        if m is not None:
            out = np.zeros_like(m) if out is None else out
            out[m > 0] = cid
    return out


def bright_target(path):
    m = read_image(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    rec = {"px": np.bincount(m.ravel(), minlength=4)[:4].tolist(), "other_values_px": int((m > 3).sum())}
    areas = {cid: components(m == cid) for cid in (1, 2, 3)}
    areas["building"] = components(m > 0)
    return rec, areas


def geotiff_gsd(path):
    """Metres per pixel from GeoTIFF tags: UTM metres, or WGS84 degrees at the tile's latitude."""
    from PIL import Image
    with Image.open(path) as im:
        scale, tie = im.tag_v2.get(33550), im.tag_v2.get(33922)
    if not scale or not tie:
        return float("nan")
    sx, sy = scale[0], scale[1]
    if sx < 0.01:
        return float((sx * 111320 * math.cos(math.radians(tie[4])) + sy * 110574) / 2)
    return float((sx + sy) / 2)


# ----------------------------------------------------------------------------- helpers
def js_bits(p, q):
    """Jensen-Shannon divergence of two histograms, in bits (0 = identical, 1 = disjoint)."""
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return round(0.5 * kl(p, m) + 0.5 * kl(q, m), 4)


def quartiles(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return None
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "p25": round(float(p25), 4), "median": round(float(p50), 4), "p75": round(float(p75), 4)}


def shares(px):
    """Pixel counts [bg, intact, damaged, destroyed] -> building share and class shares within buildings."""
    px = np.asarray(px, dtype=float)
    building = px[1:].sum()
    return {"building_share_of_all_px": round(building / px.sum(), 4),
            "intact": round(px[1] / building, 4), "damaged": round(px[2] / building, 4),
            "destroyed": round(px[3] / building, 4)}


def median_of(values):
    q = quartiles(values)
    return q["median"] if q else None


def pooled(areas_list, key):
    return np.concatenate([a[key] for a in areas_list if a is not None and len(a[key])] or [np.zeros(0, np.int32)])


def main():
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    from bright_dm3_dedup import build_optical_pool, copy_score, dm3_event
    Image.MAX_IMAGE_PIXELS = None

    ap = argparse.ArgumentParser(description="Optical-only audit and EDA of DisasterM3 and BRIGHT")
    ap.add_argument("--dm3_root", default=r"d:\CSE499A.15-Msrb-\dataset\DisasterM3 Dataset")
    ap.add_argument("--bright_root", default=r"d:\CSE499A.15-Msrb-\dataset\BRIGHT Dataset")
    ap.add_argument("--out", default=r"d:\CSE499A.15-Msrb-\data\dm3_bright_optical_eda")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="Decoding processes (each holds a few 1024x1024 images; 4 fits in 8 GB RAM)")
    ap.add_argument("--cache", default=None,
                    help="Optional pickle of the pixel pass; reused when it exists (delete it after a data change)")
    args = ap.parse_args()
    dm3, bright, out = Path(args.dm3_root), Path(args.bright_root), Path(args.out)
    instruct = dm3 / "DisasterM3_Instruct"
    out.mkdir(parents=True, exist_ok=True)
    S = {}

    print("=" * 70)
    print("  DisasterM3 vs BRIGHT -- optical audit & EDA (SAR excluded)")
    print("=" * 70)

    # ---- 1. Catalogue: what the JSON references, what is on disk ---------------------------
    entries = json.loads((instruct / "train_release.json").read_text(encoding="utf-8"))
    bench = json.loads((dm3 / "DisasterM3_Bench" / "benchmark_release.json").read_text(encoding="utf-8"))

    def references(entry_list):
        post_type, pre_of, pres = defaultdict(set), {}, set()
        for e in entry_list:
            pre, post, kind = e.get("pre_image_path"), e.get("post_image_path"), e.get("image_type") or e.get("post_image_type")
            if post:
                post_type[name_of(post)].add(kind)
                if pre:
                    pre_of[name_of(post)] = name_of(pre)
            if pre:
                pres.add(name_of(pre))
        return post_type, pre_of, pres

    post_type, pre_of, pres = references(entries)
    img_dir = instruct / "train_images" / "train_images"
    dm3_df = pd.DataFrame([{"name": p.name, **parse_dm3_name(p.name)} for p in sorted(img_dir.iterdir())])
    dm3_df["json_role"] = [("post" if n in post_type else "pre" if n in pres else None) for n in dm3_df.name]
    dm3_df["json_type"] = ["/".join(sorted(post_type[n])) if n in post_type else ("Optical (pre)" if n in pres else None)
                           for n in dm3_df.name]
    box_images = len(list((instruct / "box_train_images" / "box_train_images").glob("*.png")))

    bright_pre_files = sorted((bright / "pre-event").rglob("*_pre_disaster.tif"))
    bright_post_files = sorted((bright / "post-event").rglob("*_post_disaster.tif"))
    target_files = sorted((bright / "target").rglob("*_building_damage.tif"))
    bright_pre = pd.DataFrame({"name": [p.name for p in bright_pre_files],
                               "event": [p.name.rsplit("_", 3)[0] for p in bright_pre_files]})
    post_modes = Counter()
    for p in bright_post_files:
        with Image.open(p) as im:
            post_modes[im.mode] += 1
    targets = pd.DataFrame({"name": [p.name for p in target_files],
                            "event": [p.name.rsplit("_", 3)[0] for p in target_files]})
    targets["gsd_m"] = [geotiff_gsd(p) for p in tqdm(target_files, desc="BRIGHT GSD", mininterval=10)]
    bright_pre["gsd_m"] = [geotiff_gsd(p) for p in bright_pre_files]

    # The champion notebooks' optical segmentation pool and its frozen split
    pool_pairs, train, val, _ = build_optical_pool(instruct)
    split_of = {p["post"].name: "train" for p in train} | {p["post"].name: "val" for p in val}
    masks_of = defaultdict(lambda: ({}, {}))
    for e in entries:
        if e.get("task") == "Referring Expression Segmentation" and e.get("image_type") == "Optical":
            rel = e.get("ground_truth", "").replace("\\", "/")
            folder, post = Path(rel).parent.name, name_of(e.get("post_image_path", ""))
            path = instruct / "masks" / rel
            if post in split_of and path.exists():
                if folder in BUILDING:
                    masks_of[post][0][folder] = path
                elif folder in ROAD:
                    masks_of[post][1][folder] = path
    pool_posts = [p["post"].name for p in pool_pairs]

    # ---- 2. Pixels: every DisasterM3 image, every BRIGHT pre tile, pairs, masks, targets -----
    optical_pairs = [(pre_of[n], n) for n in post_type
                     if post_type[n] == {"Optical"} and n in pre_of and (img_dir / n).exists()
                     and (img_dir / pre_of[n]).exists()]
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        dm3_res, br_res, shifts, mask_res, tgt_res = pickle.loads(cache.read_bytes())
        print(f"[*] Pixel results loaded from {cache}")
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            dm3_res = list(tqdm(ex.map(image_stats, [img_dir / n for n in dm3_df.name], chunksize=32),
                                total=len(dm3_df), desc="DisasterM3 images", mininterval=10))
            br_res = list(tqdm(ex.map(image_stats, bright_pre_files, chunksize=16),
                               total=len(bright_pre_files), desc="BRIGHT pre tiles", mininterval=10))
            shifts = list(tqdm(ex.map(pair_shift, [(img_dir / a, img_dir / b) for a, b in optical_pairs], chunksize=16),
                               total=len(optical_pairs), desc="Registration", mininterval=10))
            mask_res = list(tqdm(ex.map(pool_masks, [masks_of[n] for n in pool_posts], chunksize=16),
                                 total=len(pool_posts), desc="Pool masks", mininterval=10))
            tgt_res = list(tqdm(ex.map(bright_target, target_files, chunksize=16),
                                total=len(target_files), desc="BRIGHT targets", mininterval=10))
        if cache:
            cache.write_bytes(pickle.dumps((dm3_res, br_res, shifts, mask_res, tgt_res)))

    def attach(df, results):
        recs = pd.DataFrame([r for r, _, _ in results])
        thumbs = np.stack([t if t is not None else np.zeros(THUMB * THUMB, np.float32) for _, t, _ in results])
        hists = [h for _, _, h in results]
        return pd.concat([df.reset_index(drop=True), recs], axis=1), thumbs, hists

    dm3_df, dm3_thumbs, dm3_hists = attach(dm3_df, dm3_res)
    bright_pre, br_thumbs, br_hists = attach(bright_pre, br_res)
    dm3_df["pixel_modality"] = np.select([~dm3_df.readable.fillna(False).astype(bool),
                                          dm3_df.single_channel.fillna(False).astype(bool)],
                                         ["unreadable", "SAR"], "optical")
    dm3_df["split"] = dm3_df.name.map(split_of)
    pre_split = {pre_of[n]: s for n, s in split_of.items() if n in pre_of}
    dm3_df.loc[dm3_df.split.isna(), "split"] = dm3_df.name.map(pre_split)
    dm3_df["in_pool"] = dm3_df.split.notna()
    dm3_df["hazard"] = dm3_df.event.map(hazard)
    dm3_df["origin"] = dm3_df.event.map(origin)
    row_of = {n: i for i, n in enumerate(dm3_df.name)}

    # ---- 3. Inventory and SAR exclusion ------------------------------------------------------
    posts = dm3_df[dm3_df.role == "post"]
    tagged = posts[posts.json_type.notna()]
    bench_post_type, bench_pre_of, bench_pres = references(bench)
    S["inventory"] = {
        "dm3_instruct": {
            "files_on_disk": int(len(dm3_df)), "box_images_orr": box_images,
            "by_name_role_and_pixels": {f"{r}/{m}": int(v) for (r, m), v in
                                        dm3_df.groupby(["role", "pixel_modality"], dropna=False).size().items()},
            "json_pairs": {"optical": sum(t == {"Optical"} for t in post_type.values()),
                           "sar": sum(t == {"SAR"} for t in post_type.values()),
                           "unique_pre_images": len(pres)},
            "unreferenced_files": {f"{r}/{m}": int(v) for (r, m), v in
                                   dm3_df[dm3_df.json_role.isna()].groupby(["role", "pixel_modality"], dropna=False).size().items()},
            "unreferenced_by_prefix": {k: int(v) for k, v in
                                       dm3_df[dm3_df.json_role.isna()].prefix.value_counts().head(20).items()},
            "optical_pairs_usable": len(optical_pairs),
            "optical_pool": {"images": len(pool_pairs), "train": len(train), "val": len(val)},
            "paper": {"optical_images": 17190, "sar_images": 3798, "instruction_pairs": 92968},
        },
        "dm3_bench_json_only": {
            "entries": len(bench), "unique_pre_images": len(bench_pres),
            "unique_posts": {k: sum(t == {k} for t in bench_post_type.values()) for k in ("Optical", "SAR")},
            "paper": {"optical_images": 5024, "sar_images": 976, "instruction_pairs": 30042},
            "images_on_disk": 0,
        },
        "bright": {
            "pre_optical_tiles_local": int(len(bright_pre)),
            "pre_events_local": sorted(bright_pre.event.unique()),
            "pre_missing_events": {e: int(n) for e, n in targets.event.value_counts().items()
                                   if e not in set(bright_pre.event)},
            "post_tiles": len(bright_post_files), "post_pil_modes": dict(post_modes),
            "targets": int(len(targets)), "paper_tiles": 4246,
        },
        "sar_check": {
            "post_tag_vs_pixels": {f"{t}/{m}": int(v) for (t, m), v in
                                   tagged.groupby(["json_type", "pixel_modality"]).size().items()},
            "pre_images_single_channel": int(((dm3_df.role == "pre") & (dm3_df.pixel_modality == "SAR")).sum()),
            "sar_posts_without_sar_in_name": int(((posts.pixel_modality == "SAR") & ~posts.sar_in_name).sum()),
            "sar_posts_with_sar_in_name": int(((posts.pixel_modality == "SAR") & posts.sar_in_name).sum()),
            "unreadable_files": int((~dm3_df.readable.fillna(False)).sum()),
        },
        "naming_schemes": {k: int(v) for k, v in dm3_df.scheme.value_counts().items()},
    }

    # ---- 4. Events -----------------------------------------------------------------------------
    opt = dm3_df[dm3_df.pixel_modality == "optical"]
    events = []
    for prefix, g in dm3_df.groupby("prefix"):
        events.append({
            "dataset": "DisasterM3", "event": prefix, "hazard": hazard(prefix), "origin": origin(re.sub(r"\d+$", "", prefix)),
            "bright_event": dm3_event(prefix),
            "optical_pre": int(((g.role == "pre") & (g.pixel_modality == "optical")).sum()),
            "optical_post": int(((g.role == "post") & (g.pixel_modality == "optical")).sum()),
            "sar_post": int(((g.role == "post") & (g.pixel_modality == "SAR")).sum()),
            "pool_train": int(((g.role == "post") & (g.split == "train")).sum()),
            "pool_val": int(((g.role == "post") & (g.split == "val")).sum()),
        })
    bench_events = Counter(parse_dm3_name(n)["prefix"] for n, t in bench_post_type.items() if t == {"Optical"})
    for e in events:
        e["bench_optical_post"] = int(bench_events.get(e["event"], 0))
    for prefix in sorted(set(bench_events) - set(dm3_df.prefix)):
        events.append({"dataset": "DisasterM3", "event": prefix, "hazard": hazard(prefix),
                       "origin": origin(re.sub(r"\d+$", "", prefix)), "bright_event": dm3_event(prefix),
                       "optical_pre": 0, "optical_post": 0, "sar_post": 0, "pool_train": 0, "pool_val": 0,
                       "bench_optical_post": int(bench_events[prefix])})
    S["events_dm3"] = events

    # ---- 5. Overlap: identical images across and within datasets ----------------------------
    bright_md5 = defaultdict(list)
    for name, md5 in zip(bright_pre.name, bright_pre.md5):
        if isinstance(md5, str):
            bright_md5[md5].append(name)
    dm3_df["identical_bright_tile"] = dm3_df.md5.map(lambda h: bright_md5.get(h, [None])[0] if isinstance(h, str) else None)
    dm3_groups = dm3_df[dm3_df.md5.notna()].groupby("md5").name.apply(list)
    dup_groups = [names for names in dm3_groups if len(names) > 1]
    by_bright_event = []
    for ev, g in bright_pre.groupby("event"):
        hit = dm3_df[dm3_df.identical_bright_tile.isin(set(g.name))]
        by_bright_event.append({
            "bright_event": ev, "pre_tiles": int(len(g)),
            "tiles_found_in_dm3": int(hit.identical_bright_tile.nunique()),
            "as_pool_pre_train": int(((hit.split == "train") & (hit.role == "pre")).sum()),
            "as_pool_pre_val": int(((hit.split == "val") & (hit.role == "pre")).sum()),
            "dm3_prefixes": sorted(hit.prefix.unique().tolist()),
        })
    # near-identical images shared by the frozen val and the train split (pre or post)
    pool_rows = dm3_df[dm3_df.in_pool]
    val_idx = pool_rows.index[pool_rows.split == "val"].to_numpy()
    train_idx = pool_rows.index[pool_rows.split == "train"].to_numpy()
    sim = dm3_thumbs[val_idx] @ dm3_thumbs[train_idx].T
    cross = []
    for i, j in zip(*np.nonzero(sim >= 0.95)):
        a, b = dm3_df.name.at[val_idx[i]], dm3_df.name.at[train_idx[j]]
        same = dm3_df.md5.at[val_idx[i]] == dm3_df.md5.at[train_idx[j]]
        r = 1.0 if same else copy_score(img_dir / a, img_dir / b)
        if r >= 0.95:
            cross.append({"val_image": a, "train_image": b, "r": round(r, 4), "identical": bool(same)})
    shared_pre = {pre_of[n] for n in split_of if split_of[n] == "val"} & {pre_of[n] for n in split_of if split_of[n] == "train"}
    S["overlap"] = {
        "dm3_images_identical_to_bright_pre": int(dm3_df.identical_bright_tile.notna().sum()),
        "by_bright_event": by_bright_event,
        "dm3_exact_duplicate_groups": len(dup_groups),
        "dm3_exact_duplicate_files": int(sum(len(g) for g in dup_groups)),
        "dm3_duplicate_prefix_pairs": dict(Counter(" = ".join(sorted({parse_dm3_name(n)["prefix"] for n in g}))
                                                   for g in dup_groups).most_common(15)),
        "val_train_near_identical_pairs": len(cross),
        "val_images_with_a_train_copy": len({c["val_image"] for c in cross}),
        "val_train_examples": cross[:10],
        "pre_images_shared_by_val_and_train_pairs": len(shared_pre),
    }

    # ---- 6. Geometry and resolution ------------------------------------------------------------
    gsd_rows = []
    for ev, g in targets.groupby("event"):
        (lo, hi), n_paper = BRIGHT_PAPER.get(ev, ((None, None), None))
        gsd_rows.append({"event": ev, "tiles": int(len(g)), "gsd_median_m": round(float(g.gsd_m.median()), 3),
                         "gsd_min_m": round(float(g.gsd_m.min()), 3), "gsd_max_m": round(float(g.gsd_m.max()), 3),
                         "paper_gsd_m": [lo, hi], "paper_buildings": n_paper})
    copies = dm3_df[dm3_df.identical_bright_tile.notna()].merge(
        bright_pre[["name", "gsd_m"]].rename(columns={"name": "identical_bright_tile"}), on="identical_bright_tile")
    S["geometry"] = {
        "dm3_optical_sizes": {f"{int(h)}x{int(w)}": int(v) for (h, w), v in opt.groupby(["h", "w"]).size().items()},
        "bright_pre_sizes": {f"{int(h)}x{int(w)}": int(v) for (h, w), v in bright_pre.groupby(["h", "w"]).size().items()},
        "bright_gsd_by_event": gsd_rows,
        "dm3_copies_of_bright_gsd_m": quartiles(copies.gsd_m),
        "dm3_paper_claim_gsd_m": 0.8,
        "hf_ratio": {
            "dm3_pool_pre": quartiles(dm3_df.hf_ratio[dm3_df.in_pool & (dm3_df.role == "pre")]),
            "dm3_pool_post": quartiles(dm3_df.hf_ratio[dm3_df.in_pool & (dm3_df.role == "post")]),
            "bright_pre_all": quartiles(bright_pre.hf_ratio),
            "bright_pre_our_5_events": quartiles(bright_pre.hf_ratio[bright_pre.event.isin(OUR_BRIGHT_EVENTS)]),
        },
        "hf_ratio_by_dm3_pool_event_post": {ev: median_of(g.hf_ratio) for ev, g in
                                            dm3_df[dm3_df.in_pool & (dm3_df.role == "post")].groupby("event")},
        "hf_ratio_by_bright_event": {ev: median_of(g.hf_ratio) for ev, g in bright_pre.groupby("event")},
    }

    # ---- 7. Radiometry --------------------------------------------------------------------------
    groups = {
        "DM3 pool pre": dm3_df.in_pool & (dm3_df.role == "pre"),
        "DM3 pool post": dm3_df.in_pool & (dm3_df.role == "post"),
        "DM3 other optical": ~dm3_df.in_pool & (dm3_df.pixel_modality == "optical"),
        "BRIGHT pre (11 events)": None,
        "BRIGHT pre (our 5 events)": None,
    }
    group_hist, radiometry = {}, {}
    cols = ["lum_mean", "lum_std", "sat_mean", "nodata_frac", "bright_lowsat_frac", "dark_channel", "lap_var", "hf_ratio",
            "mean_r", "mean_g", "mean_b"]
    for gname, mask in groups.items():
        if gname.startswith("BRIGHT"):
            sel = bright_pre.event.isin(OUR_BRIGHT_EVENTS) if "5" in gname else bright_pre.event.notna()
            df, hl = bright_pre[sel], [h for h, s in zip(br_hists, sel) if s and h is not None]
        else:
            df, hl = dm3_df[mask], [dm3_hists[i] for i in dm3_df.index[mask] if dm3_hists[i] is not None]
        group_hist[gname] = np.sum(hl, axis=0)
        radiometry[gname] = {c: quartiles(df[c]) for c in cols}
        radiometry[gname]["images"] = int(len(df))
        radiometry[gname]["share_nodata_over_10pct"] = round(float((df.nodata_frac > 0.10).mean()), 4)
        radiometry[gname]["share_bright_lowsat_over_20pct"] = round(float((df.bright_lowsat_frac > 0.20).mean()), 4)
    names = list(group_hist)
    js = {a: {b: js_bits(group_hist[a], group_hist[b]) for b in names} for a in names}
    # pre -> post change inside each DisasterM3 optical pool pair
    pairs_df = pd.DataFrame({"pre": [p["pre"].name if p["pre"] else None for p in pool_pairs], "post": pool_posts})
    pairs_df["split"] = pairs_df.post.map(split_of)
    pairs_df["event"] = pairs_df.post.map(lambda n: parse_dm3_name(n)["prefix"])
    for col in ("lum_mean", "sat_mean", "lum_std"):
        pairs_df[f"pre_{col}"] = pairs_df.pre.map(lambda n: dm3_df[col].at[row_of[n]] if n in row_of else np.nan)
        pairs_df[f"post_{col}"] = pairs_df.post.map(lambda n: dm3_df[col].at[row_of[n]])
    pairs_df["pair_js_bits"] = [js_bits(dm3_hists[row_of[a]], dm3_hists[row_of[b]])
                                if a in row_of and dm3_hists[row_of[a]] is not None and dm3_hists[row_of[b]] is not None
                                else np.nan for a, b in zip(pairs_df.pre, pairs_df.post)]
    S["radiometry"] = {
        "groups": radiometry, "lum_hist_js_bits": js,
        "lum_hist_by_group": {k: (v / v.sum()).round(5).tolist() for k, v in group_hist.items()},
        "pre_to_post": {
            "delta_lum_mean": quartiles(pairs_df.post_lum_mean - pairs_df.pre_lum_mean),
            "delta_sat_mean": quartiles(pairs_df.post_sat_mean - pairs_df.pre_sat_mean),
            "pair_js_bits": quartiles(pairs_df.pair_js_bits),
            "share_pairs_js_over_0_2": round(float((pairs_df.pair_js_bits > 0.2).mean()), 4),
        },
        "dm3_pool_post_by_event": {ev: {"images": int(len(g)), "lum": median_of(g.lum_mean), "sat": median_of(g.sat_mean),
                                        "nodata_share_over_10pct": round(float((g.nodata_frac > 0.1).mean()), 3)}
                                   for ev, g in dm3_df[dm3_df.in_pool & (dm3_df.role == "post")].groupby("event")},
        "bright_pre_by_event": {ev: {"images": int(len(g)), "lum": median_of(g.lum_mean), "sat": median_of(g.sat_mean),
                                    "nodata_share_over_10pct": round(float((g.nodata_frac > 0.1).mean()), 3)}
                                for ev, g in bright_pre.groupby("event")},
    }

    # ---- 8. Registration of DisasterM3 optical pairs -------------------------------------------
    reg = pd.DataFrame([s if s else (np.nan, np.nan, 0, 0, np.nan) for s in shifts],
                       columns=["dx", "dy", "tiles_agreeing", "tiles_valid", "spread_px"])
    reg.insert(0, "post", [b for _, b in optical_pairs])
    reg.insert(0, "pre", [a for a, _ in optical_pairs])
    reg["shift_px"] = np.hypot(reg.dx, reg.dy)
    reg["event"] = reg.post.map(lambda n: parse_dm3_name(n)["prefix"])
    reg["in_pool"] = reg.post.isin(split_of)
    reg["reliable"] = (reg.tiles_agreeing >= 4) & (reg.spread_px <= 1.5)
    trusted = reg[reg.reliable]
    S["registration"] = {
        "pairs": int(len(reg)), "pairs_reliable": int(len(trusted)),
        "rule": "phase correlation on 256-px tiles valid in both images; reliable = >=4 tiles with peak >= 0.1 "
                "agreeing within 1.5 px of their median",
        "pairs_without_valid_tiles": int((reg.tiles_valid == 0).sum()),
        "shift_px_trusted": quartiles(trusted.shift_px),
        "share_trusted_over_1px": round(float((trusted.shift_px > 1).mean()), 4),
        "share_trusted_over_2px": round(float((trusted.shift_px > 2).mean()), 4),
        "share_trusted_over_5px": round(float((trusted.shift_px > 5).mean()), 4),
        "shift_hist_trusted": {f"{lo}-{hi}": int(((trusted.shift_px >= lo) & (trusted.shift_px < hi)).sum())
                               for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 1e9))},
        "by_event_median_px": {ev: {"pairs": int(len(g)), "median_px": round(float(g.shift_px.median()), 2),
                                    "share_over_2px": round(float((g.shift_px > 2).mean()), 3)}
                               for ev, g in trusted.groupby("event")},
        "reliable_share_by_event": {ev: round(float(g.reliable.mean()), 3) for ev, g in reg.groupby("event")},
        "worst": trusted.nlargest(8, "shift_px")[["pre", "post", "dx", "dy", "tiles_agreeing", "spread_px"]]
                 .round(2).to_dict("records"),
    }

    # ---- 9. Labels ---------------------------------------------------------------------------------
    mrec = [m[0] if m else None for m in mask_res]
    mareas = [m[1] if m else None for m in mask_res]
    lab = pd.DataFrame([{"post": n, "split": split_of[n], "event": parse_dm3_name(n)["prefix"],
                         **({"px_bg": r["px"][0], "px_intact": r["px"][1], "px_damaged": r["px"][2],
                             "px_destroyed": r["px"][3], "multi_label_px": r["multi_label_px"],
                             "mask_values": "/".join(map(str, r["mask_values"])),
                             **{f"road_{k}": v for k, v in r["road_px"].items()},
                             "n_intact": len(a[1]), "n_damaged": len(a[2]), "n_destroyed": len(a[3]),
                             "n_buildings": len(a["building"])} if r else {})}
                        for n, r, a in zip(pool_posts, mrec, mareas)])
    trec = [t[0] if t else {"px": [0, 0, 0, 0], "other_values_px": 0} for t in tgt_res]
    tareas = [t[1] if t else None for t in tgt_res]
    targets["px_bg"], targets["px_intact"], targets["px_damaged"], targets["px_destroyed"] = zip(*[t["px"] for t in trec])
    targets["n_buildings"] = [len(a["building"]) if a else 0 for a in tareas]
    targets["n_destroyed"] = [len(a[3]) if a else 0 for a in tareas]

    def label_block(px_cols_df, areas_list, n_destroyed):
        px = px_cols_df[["px_bg", "px_intact", "px_damaged", "px_destroyed"]].sum().to_numpy()
        b_areas = pooled(areas_list, "building")
        return {"images": int(len(px_cols_df)), **shares(px),
                "images_without_destroyed_px": round(float((px_cols_df.px_destroyed == 0).mean()), 4),
                "images_without_any_building": round(float((px_cols_df[["px_intact", "px_damaged", "px_destroyed"]].sum(axis=1) == 0).mean()), 4),
                "building_components": int(len(b_areas)),
                "building_area_px": quartiles(b_areas),
                "destroyed_components_per_image": quartiles(n_destroyed)}

    S["labels"] = {
        "dm3_pool": label_block(lab, mareas, lab.n_destroyed),
        "dm3_pool_train": label_block(lab[lab.split == "train"], [a for a, s in zip(mareas, lab.split) if s == "train"],
                                      lab.n_destroyed[lab.split == "train"]),
        "dm3_pool_val": label_block(lab[lab.split == "val"], [a for a, s in zip(mareas, lab.split) if s == "val"],
                                    lab.n_destroyed[lab.split == "val"]),
        "bright_all": label_block(targets, tareas, targets.n_destroyed),
        "bright_our_5_events": label_block(targets[targets.event.isin(OUR_BRIGHT_EVENTS)],
                                           [a for a, e in zip(tareas, targets.event) if e in OUR_BRIGHT_EVENTS],
                                           targets.n_destroyed[targets.event.isin(OUR_BRIGHT_EVENTS)]),
        "bright_paper": {"building_share_of_all_px": 0.141, "intact": 0.828, "damaged": 0.107, "destroyed": 0.065},
        "dm3_multi_label_px": {"images_with_any": int((lab.multi_label_px > 0).sum()),
                               "share_of_building_px": round(float(lab.multi_label_px.sum() / lab[["px_intact", "px_damaged", "px_destroyed"]].sum().sum()), 6)},
        "dm3_mask_value_sets": {k: int(v) for k, v in lab.mask_values.value_counts().items()},
        "dm3_road_px_share": {k: round(float(lab[f"road_{k}"].sum() / lab[[f"road_{x}" for x in ROAD.values()]].sum().sum()), 4)
                              for k in ROAD.values()},
        "bright_values_outside_0_3_px": int(sum(t["other_values_px"] for t in trec)),
    }
    by_event = []
    for ev, g in lab.groupby("event"):
        b_areas = pooled([mareas[i] for i in g.index], "building")
        by_event.append({"dataset": "DisasterM3 pool", "event": ev, "images": int(len(g)),
                         **shares(g[["px_bg", "px_intact", "px_damaged", "px_destroyed"]].sum().to_numpy()),
                         "images_without_destroyed_px": round(float((g.px_destroyed == 0).mean()), 3),
                         "median_building_area_px": median_of(b_areas), "building_components": int(len(b_areas)),
                         "train": int((g.split == "train").sum()), "val": int((g.split == "val").sum())})
    for ev, g in targets.groupby("event"):
        idx = g.index
        b_areas = pooled([tareas[i] for i in idx], "building")
        by_event.append({"dataset": "BRIGHT", "event": ev, "images": int(len(g)),
                         **shares(g[["px_bg", "px_intact", "px_damaged", "px_destroyed"]].sum().to_numpy()),
                         "images_without_destroyed_px": round(float((g.px_destroyed == 0).mean()), 3),
                         "median_building_area_px": median_of(b_areas),
                         "median_building_area_m2": round(float(np.median(b_areas) * g.gsd_m.median() ** 2), 1),
                         "building_components": int(len(b_areas)), "paper_buildings": BRIGHT_PAPER[ev][1]})
    S["labels"]["by_event"] = by_event
    S["labels"]["area_hist_edges_px"] = AREA_EDGES.astype(int).tolist()
    S["labels"]["building_area_hist"] = {
        "dm3_pool": np.histogram(pooled(mareas, "building"), bins=AREA_EDGES)[0].tolist(),
        "bright_all": np.histogram(pooled(tareas, "building"), bins=AREA_EDGES)[0].tolist(),
        "bright_our_5_events": np.histogram(pooled([a for a, e in zip(tareas, targets.event) if e in OUR_BRIGHT_EVENTS],
                                                   "building"), bins=AREA_EDGES)[0].tolist(),
    }

    # Val pairs whose scene is also in train, and how close each is to its train twin
    pair_post = {p["post"].name: p["post"].name for p in pool_pairs}
    pair_post.update({p["pre"].name: p["post"].name for p in pool_pairs if p["pre"]})
    twins = {}
    for c in cross:
        twins.setdefault(pair_post[c["val_image"]], pair_post[c["train_image"]])
    leak_rows = []
    for v, t in twins.items():
        mv, mt = combined_mask(masks_of[v][0]), combined_mask(masks_of[t][0])
        both = (mv > 0) & (mt > 0)
        leak_rows.append({"val_post": v, "train_post": t, "post_r": round(copy_score(img_dir / v, img_dir / t), 3),
                          "building_iou": round(float(both.sum() / max(((mv > 0) | (mt > 0)).sum(), 1)), 3),
                          "same_damage_class": round(float((mv[both] == mt[both]).mean()), 3) if both.any() else None,
                          "val_destroyed_px": int((mv == 3).sum())})
    leak_df = pd.DataFrame(leak_rows, columns=["val_post", "train_post", "post_r", "building_iou",
                                               "same_damage_class", "val_destroyed_px"])
    S["overlap"]["leaked_val_pairs"] = int(len(leak_df))
    S["overlap"]["leaked_share_of_val_destroyed_px"] = round(
        float(leak_df.val_destroyed_px.sum() / max(lab[lab.split == "val"].px_destroyed.sum(), 1)), 4)
    S["overlap"]["leaked_twins"] = {c: quartiles(leak_df[c]) for c in ("post_r", "building_iou", "same_damage_class")}

    # ---- 10. Write -----------------------------------------------------------------------------------
    leak_df.to_csv(out / "val_train_leaks.csv", index=False)
    keep = [c for c in dm3_df.columns if c != "single_channel"]
    dm3_df[keep].to_csv(out / "dm3_images.csv", index=False)
    bright_pre.to_csv(out / "bright_pre_tiles.csv", index=False)
    reg.to_csv(out / "dm3_optical_pair_registration.csv", index=False)
    pairs_df.to_csv(out / "dm3_pool_pairs_radiometry.csv", index=False)
    lab.to_csv(out / "dm3_pool_labels.csv", index=False)
    targets.to_csv(out / "bright_targets.csv", index=False)
    pd.DataFrame(events).to_csv(out / "dm3_events.csv", index=False)
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")

    inv, lab_s = S["inventory"], S["labels"]
    print(f"[*] DisasterM3 Instruct files : {inv['dm3_instruct']['files_on_disk']:,} "
          f"({inv['dm3_instruct']['by_name_role_and_pixels']})")
    print(f"[*] SAR check (tag/pixels)    : {inv['sar_check']['post_tag_vs_pixels']}")
    print(f"[*] BRIGHT pre / post / target: {inv['bright']['pre_optical_tiles_local']} / "
          f"{inv['bright']['post_tiles']} {inv['bright']['post_pil_modes']} / {inv['bright']['targets']}")
    print(f"[*] Registration (trusted)    : {S['registration']['shift_px_trusted']}")
    print(f"[*] Labels DM3 pool           : {lab_s['dm3_pool']}")
    print(f"[*] Labels BRIGHT             : {lab_s['bright_all']}")
    print(f"[*] Overlap                   : {S['overlap']['dm3_images_identical_to_bright_pre']} DM3 images = BRIGHT tiles; "
          f"{S['overlap']['val_images_with_a_train_copy']} val images with a train copy")
    print(f"[OK] Written to {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
