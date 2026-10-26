"""
dm3_bright_optical_eda_ext.py
=============================
Second pass of the optical-only DisasterM3 / BRIGHT audit: four checks dm3_bright_optical_eda.py does not make.
SAR stays excluded; this script reads that script's CSVs and adds:

  1. Label registration  which image each DisasterM3 building mask is drawn on (pre or post), how far the
                         labels sit from the roofs of the post image a post-only model reads, and the IoU
                         that offset alone costs. BRIGHT label maps are the control: drawn on the pre tile.
  2. Feature gap         ImageNet ResNet-50 embeddings of optical-pool images and BRIGHT pre tiles:
                         how separable they are, how far each BRIGHT event lies from the pool, a 2-D map.
  3. Re-tiled overlap    SIFT matching of DisasterM3 pre images to BRIGHT tiles of the same event where no
                         byte-identical copy exists: shared ground, and the scale (GSD) DisasterM3 really uses.
  4. Spot checks         headline numbers of the first pass, re-measured a second way.

Writes summary_ext.json and label_registration.csv, bright_label_registration.csv, feature_gap_*.csv,
retiled_matches.csv to --eda (the first pass's output folder).

Author: Ridita Afrin Riya (CSE499B Group 5)
Project: Resource-Efficient Neural Architectures for Post-Disaster Damage Assessment
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy import fft as sfft

# Worker processes re-import this module, so only cv2, numpy and scipy.fft load at the top
sys.path.insert(0, str(Path(__file__).resolve().parent))
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

from dm3_bright_optical_eda import BUILDING, OUR_BRIGHT_EVENTS, hazard, quartiles, read_image  # noqa: E402

SEARCH = 64          # largest offset searched, px (the first pass saw up to 56 px)
Q_MIN = 1.5          # correlation peak / second peak needed to trust an offset (BRIGHT control, see summary)
TOL = 1.5            # px: an offset this close to a hypothesis counts as matching it
NOISE_PX = 1.6       # px: 95th percentile of the BRIGHT control's offset; smaller offsets count as none
MIN_PX = 2000        # building pixels a mask needs before its outline can be matched
FEAT_SIDE = 224
RETILED = {"beirut-explosion": "beriut_explosion", "bata-explosion": "bata_explosion",
           "hawaii-wildfire": "hawaii_wildfire",
           "libya-flood": "libya_flood"}        # control: its DisasterM3 pre images are byte copies (scale 1)


# ----------------------------------------------------------------------------- offsets
def edges(img, sigma=1.5):
    """Gradient magnitude minus its local mean: outlines of roofs, roads and mask borders."""
    a = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    g = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    return g - cv2.blur(g, (31, 31))


def spectrum(e):
    """FFT of an edge map, zero-padded by SEARCH so the correlation is linear, not circular."""
    return sfft.rfft2(np.pad(e, SEARCH))                    # float32 in, complex64 out


def offset(fa, fb, shape):
    """Shift d that moves image a onto image b (b(x) ~ a(x - d)), from their padded spectra.
    Same sign convention as cv2.phaseCorrelate in the first pass. Returns (dx, dy, peak quality):
    quality = (peak - median) / (best peak outside +-5 px - median); 1 means no clear winner."""
    c = sfft.irfft2(np.conj(fa) * fb, s=(shape[0] + 2 * SEARCH, shape[1] + 2 * SEARCH))
    r = np.arange(-SEARCH, SEARCH + 1)
    w = c[r[:, None], r[None, :]]
    iy, ix = np.unravel_index(np.argmax(w), w.shape)

    def vertex(n, p):
        den = n[p - 1] - 2 * n[p] + n[p + 1] if 0 < p < len(n) - 1 else 0.0
        return 0.5 * (n[p - 1] - n[p + 1]) / den if den < 0 else 0.0
    dx, dy = ix + vertex(w[iy, :], ix) - SEARCH, iy + vertex(w[:, ix], iy) - SEARCH
    rest = w.copy()
    rest[max(iy - 5, 0):iy + 6, max(ix - 5, 0):ix + 6] = -np.inf
    med = np.median(w)
    return float(dx), float(dy), float((w[iy, ix] - med) / (rest.max() - med + 1e-9))


def class_map(building):
    """The notebooks' 4-class map from the binary building masks of one pool image (later class wins)."""
    out = None
    for folder, cid in BUILDING.items():
        m = read_image(building[folder], cv2.IMREAD_GRAYSCALE) if folder in building else None
        if m is not None:
            out = np.zeros_like(m) if out is None else out
            out[m > 0] = cid
    return out


def shifted_iou(cls, dx, dy):
    """Intersection and union per class between the labels and the same labels moved by (dx, dy):
    what a model that finds every roof of the post image exactly would score against these labels."""
    moved = cv2.warpAffine(cls, np.float32([[1, 0, dx], [0, 1, dy]]), cls.shape[::-1],
                           flags=cv2.INTER_NEAREST, borderValue=0)
    out = {}
    for name, a, b in (("building", cls > 0, moved > 0), ("intact", cls == 1, moved == 1),
                       ("damaged", cls == 2, moved == 2), ("destroyed", cls == 3, moved == 3)):
        out[f"inter_{name}"], out[f"union_{name}"] = int((a & b).sum()), int((a | b).sum())
    return out


def label_registration(job):
    """One pool pair: mask->pre, mask->post (standing buildings only) and pre->post offsets, and the IoU
    the mask->post offset costs when it is measured (offsets under NOISE_PX count as none)."""
    pre_path, post_path, building = job
    cls = class_map(building)
    pre = read_image(pre_path, cv2.IMREAD_GRAYSCALE) if pre_path else None
    post = read_image(post_path, cv2.IMREAD_GRAYSCALE)
    if cls is None or pre is None or post is None or not (cls.shape == pre.shape == post.shape):
        return None
    shape = cls.shape
    standing = (cls == 1) | (cls == 2)                     # destroyed roofs leave no outline in the post image
    rec = {"building_px": int((cls > 0).sum()), "standing_px": int(standing.sum())}
    f_pre, f_post = spectrum(edges(pre)), spectrum(edges(post))
    rec["pp_dx"], rec["pp_dy"], rec["pp_q"] = offset(f_pre, f_post, shape)
    if rec["building_px"] >= MIN_PX:
        rec["mpre_dx"], rec["mpre_dy"], rec["mpre_q"] = offset(spectrum(edges((cls > 0) * np.float32(255))), f_pre, shape)
    if rec["standing_px"] >= MIN_PX:
        rec["mpost_dx"], rec["mpost_dy"], rec["mpost_q"] = offset(spectrum(edges(standing * np.float32(255))), f_post, shape)
        if rec["mpost_q"] >= Q_MIN:
            moved = np.hypot(rec["mpost_dx"], rec["mpost_dy"]) > NOISE_PX
            rec.update(shifted_iou(cls, rec["mpost_dx"] if moved else 0.0, rec["mpost_dy"] if moved else 0.0))
    return rec


def bright_label_registration(job):
    """Control: a BRIGHT label map against its own pre tile (drawn on it, so the offset should be ~0)."""
    pre_path, target_path = job
    pre, t = read_image(pre_path, cv2.IMREAD_GRAYSCALE), read_image(target_path, cv2.IMREAD_UNCHANGED)
    if pre is None or t is None:
        return None
    t = t[..., 0] if t.ndim == 3 else t
    rec = {"building_px": int((t > 0).sum())}
    if rec["building_px"] >= MIN_PX and t.shape == pre.shape:
        rec["mpre_dx"], rec["mpre_dy"], rec["mpre_q"] = offset(spectrum(edges((t > 0) * np.float32(255))),
                                                               spectrum(edges(pre)), pre.shape)
    return rec


# ----------------------------------------------------------------------------- features
def load_rgb(path):
    img = read_image(path)
    if img is None:
        return np.zeros((FEAT_SIDE, FEAT_SIDE, 3), np.uint8)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (FEAT_SIDE, FEAT_SIDE), interpolation=cv2.INTER_AREA)


def embed(paths, ex, block=256):
    """ImageNet ResNet-50 pooled features (2,048-d) of whole tiles resized to 224 px; decoding runs in
    the worker pool one block ahead of the model."""
    import torch
    import torchvision
    from tqdm import tqdm
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    blocks = [paths[i:i + block] for i in range(0, len(paths), block)]
    feats, pending = [], [ex.submit(load_rgb, p) for p in blocks[0]] if blocks else []
    for k in tqdm(range(len(blocks)), desc="ResNet-50 features", mininterval=10):
        arr = np.stack([f.result() for f in pending])
        pending = [ex.submit(load_rgb, p) for p in blocks[k + 1]] if k + 1 < len(blocks) else []
        x = (torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255 - mean) / std
        with torch.inference_mode():
            for i in range(0, len(x), 32):
                feats.append(model(x[i:i + 32]).numpy().astype(np.float32))
    return np.concatenate(feats)


# ----------------------------------------------------------------------------- re-tiled overlap
def sift_of(path, sift):
    g = read_image(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return [], None
    return sift.detectAndCompute(g, (g > 3).astype(np.uint8))


def retiled_overlap(bright_paths, dm3_paths, min_inliers=25):
    """For each DisasterM3 image: the BRIGHT tiles it shares ground with (RANSAC similarity on SIFT
    matches), the scale (BRIGHT px per DisasterM3 px) and the share of the DisasterM3 image each covers."""
    sift = cv2.SIFT_create(nfeatures=4000)
    tile_kp, tile_desc, owner = [], [], []
    for i, p in enumerate(bright_paths):
        kp, d = sift_of(p, sift)
        tile_kp.append(np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32))
        if d is not None:
            tile_desc.append(d)
            owner.append(np.full(len(d), i))
    if not tile_desc:
        return []
    desc, owner = np.concatenate(tile_desc), np.concatenate(owner)
    offsets = np.cumsum([0] + [len(k) for k in tile_kp])
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
    flann.add([desc])
    flann.train()
    rows = []
    square = np.float32([[0, 0], [1024, 0], [1024, 1024], [0, 1024]]).reshape(-1, 1, 2)
    for p in dm3_paths:
        kp, d = sift_of(p, sift)
        if d is None or len(kp) < 20:
            continue
        pts = np.float32([k.pt for k in kp])
        good = [mn[0] for mn in flann.knnMatch(d, k=2) if len(mn) == 2 and mn[0].distance < 0.8 * mn[1].distance]
        by_tile = defaultdict(list)
        for m in good:
            by_tile[int(owner[m.trainIdx])].append((m.queryIdx, m.trainIdx - offsets[owner[m.trainIdx]]))
        for tile, pairs in sorted(by_tile.items(), key=lambda kv: -len(kv[1]))[:6]:
            if len(pairs) < min_inliers:
                break
            src = pts[[q for q, _ in pairs]]
            dst = tile_kp[tile][[t for _, t in pairs]]
            A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                                 maxIters=5000, confidence=0.999)
            if A is None or inl.sum() < min_inliers:
                continue
            scale = float(np.hypot(A[0, 0], A[1, 0]))
            footprint = cv2.transform(square, A).reshape(-1, 2)
            inter, _ = cv2.intersectConvexConvex(footprint, square.reshape(-1, 2))
            rows.append({"dm3_image": Path(p).name, "bright_tile": Path(bright_paths[tile]).name,
                         "inliers": int(inl.sum()), "raw_matches": len(pairs), "scale": round(scale, 4),
                         "rotation_deg": round(float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), 2),
                         "dm3_share_in_tile": round(float(inter / (1024 * 1024 * scale ** 2)), 4)})
    return rows


# ----------------------------------------------------------------------------- spot checks
def channel_gap(path):
    img = read_image(path)
    data = img.max(axis=2) > 3
    small = img.astype(np.int16)
    return float(np.abs(small[..., 0] - small[..., 1])[data].mean() + np.abs(small[..., 1] - small[..., 2])[data].mean())


def same_pixels(paths):
    a, b = (read_image(p) for p in paths)
    return a is not None and b is not None and a.shape == b.shape and bool(np.array_equal(a, b))


def main():
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from bright_dm3_dedup import build_optical_pool

    ap = argparse.ArgumentParser(description="Second pass of the optical-only DisasterM3 / BRIGHT audit")
    ap.add_argument("--dm3_root", default=r"d:\CSE499A.15-Msrb-\dataset\DisasterM3 Dataset")
    ap.add_argument("--bright_root", default=r"d:\CSE499A.15-Msrb-\dataset\BRIGHT Dataset")
    ap.add_argument("--eda", default=r"d:\CSE499A.15-Msrb-\data\dm3_bright_optical_eda",
                    help="Output folder of dm3_bright_optical_eda.py; read and written")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--cache", default=None,
                    help="Optional folder for the ResNet-50 features (about 130 MB); reused when present")
    ap.add_argument("--out", default=None, help="Where to write (default: --eda)")
    ap.add_argument("--limit", type=int, default=0, help="Smoke test: only this many pool pairs and BRIGHT tiles per event")
    args = ap.parse_args()
    dm3, bright, eda = Path(args.dm3_root), Path(args.bright_root), Path(args.eda)
    out = Path(args.out) if args.out else eda
    out.mkdir(parents=True, exist_ok=True)
    instruct = dm3 / "DisasterM3_Instruct"
    img_dir = instruct / "train_images" / "train_images"
    S = {}

    print("=" * 70)
    print("  DisasterM3 vs BRIGHT -- optical audit, second pass (SAR excluded)")
    print("=" * 70)

    images = pd.read_csv(eda / "dm3_images.csv")
    pairs = pd.read_csv(eda / "dm3_pool_pairs_radiometry.csv")[["pre", "post", "split", "event"]]
    reg = pd.read_csv(eda / "dm3_optical_pair_registration.csv")
    bright_pre = pd.read_csv(eda / "bright_pre_tiles.csv")
    leaks = pd.read_csv(eda / "val_train_leaks.csv")
    if args.limit:
        pairs = pairs.groupby("split", group_keys=False).sample(args.limit // 2, random_state=0).reset_index(drop=True)
        bright_pre = bright_pre.groupby("event", group_keys=False).head(max(args.limit // 10, 3)).reset_index(drop=True)
    reliable = reg[reg.reliable].set_index("post")
    pairs["hazard"] = pairs.event.map(hazard)
    origin_of = images.drop_duplicates("prefix").set_index("prefix").origin
    pairs["origin"] = pairs.event.map(origin_of)

    entries = json.loads((instruct / "train_release.json").read_text(encoding="utf-8"))
    masks_of = defaultdict(dict)
    for e in entries:
        if e.get("task") == "Referring Expression Segmentation" and e.get("image_type") == "Optical":
            rel = e.get("ground_truth", "").replace("\\", "/")
            if Path(rel).parent.name in BUILDING:
                masks_of[Path(e.get("post_image_path", "").replace("\\", "/")).name][Path(rel).parent.name] = instruct / "masks" / rel
    bright_files = {p.name: p for p in (bright / "pre-event").rglob("*_pre_disaster.tif")}
    target_files = {p.name.replace("_building_damage", "_pre_disaster"): p
                    for p in (bright / "target").rglob("*_building_damage.tif")}

    with ProcessPoolExecutor(args.workers) as ex:
        # ---- 1. Label registration ---------------------------------------------------------------
        jobs = [(img_dir / pre if isinstance(pre, str) else None, img_dir / post, masks_of[post])
                for pre, post in zip(pairs.pre, pairs.post)]
        lab = list(tqdm(ex.map(label_registration, jobs, chunksize=8), total=len(jobs),
                        desc="Pool label registration", mininterval=10))
        bjobs = [(bright_files[n], target_files[n]) for n in bright_pre.name if n in target_files]
        blab = list(tqdm(ex.map(bright_label_registration, bjobs, chunksize=8), total=len(bjobs),
                         desc="BRIGHT label registration (control)", mininterval=10))

        # ---- 2. Feature gap ----------------------------------------------------------------------
        cache = Path(args.cache) / "resnet50_features.npz" if args.cache else None
        pool_pre = [n for n in pairs.pre if isinstance(n, str)]
        feat_paths = ([img_dir / n for n in pairs.post] + [img_dir / n for n in pool_pre]
                      + [bright_files[n] for n in bright_pre.name])
        if cache and cache.exists():
            feats = np.load(cache)["feats"].astype(np.float32)
            print(f"[*] Features loaded from {cache}")
        else:
            feats = embed(feat_paths, ex)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, feats=feats.astype(np.float16))

    # ---- 1b. Label registration: tables ---------------------------------------------------------
    lr = pd.concat([pairs.reset_index(drop=True), pd.DataFrame([r or {} for r in lab])], axis=1)
    lr = lr.merge(reliable[["dx", "dy"]].rename(columns={"dx": "audit_dx", "dy": "audit_dy"}),
                  left_on="post", right_index=True, how="left")
    lr["audit_shift_px"] = np.hypot(lr.audit_dx, lr.audit_dy)
    lr["mpre_px"] = np.hypot(lr.mpre_dx, lr.mpre_dy)
    lr["mpre_vs_post_hyp_px"] = np.hypot(lr.mpre_dx + lr.audit_dx, lr.mpre_dy + lr.audit_dy)
    lr["mpost_px"] = np.hypot(lr.mpost_dx, lr.mpost_dy)
    trusted_pre = lr.mpre_q >= Q_MIN
    test = lr[trusted_pre & (lr.audit_shift_px >= 3)]
    on_pre = (test.mpre_px <= TOL) & (test.mpre_vs_post_hyp_px > TOL)
    on_post = (test.mpre_vs_post_hyp_px <= TOL) & (test.mpre_px > TOL)
    on_post_img = lr[lr.mpost_q >= Q_MIN]                  # labels measured against the post image's roofs
    lr["roof_shift_px"] = np.hypot(lr.mpost_dx - lr.mpre_dx, lr.mpost_dy - lr.mpre_dy)
    same = lr[trusted_pre & (lr.mpost_q >= Q_MIN) & lr.audit_shift_px.notna()]

    def shares(v):
        return {"over_noise": round(float((v > NOISE_PX).mean()), 4), "over_3px": round(float((v > 3).mean()), 4),
                "over_5px": round(float((v > 5).mean()), 4)}

    br = pd.concat([pd.DataFrame({"name": [n for n in bright_pre.name if n in target_files]}),
                    pd.DataFrame([r or {} for r in blab])], axis=1)
    br["event"] = br.name.str.rsplit("_", n=3).str[0]
    br["mpre_px"] = np.hypot(br.mpre_dx, br.mpre_dy)
    br_trusted = br[br.mpre_q >= Q_MIN]
    both = lr[lr.audit_shift_px.notna() & (lr.pp_q >= Q_MIN)]

    def ceiling(df):
        out = {}
        for c in ("building", "intact", "damaged", "destroyed"):
            u = df[f"union_{c}"].sum()
            out[c] = round(float(df[f"inter_{c}"].sum() / u), 4) if u else None
        return out
    with_iou = lr[lr.inter_building.notna()] if "inter_building" in lr else lr.iloc[0:0]
    S["label_registration"] = {
        "method": "edge maps (gradient magnitude) correlated over +-64 px; mask->pre uses all buildings, "
                  "mask->post only intact+damaged (destroyed roofs leave no outline); peak quality >= "
                  f"{Q_MIN} trusted; hypotheses decided within {TOL} px; offsets under {NOISE_PX} px count as none",
        "bright_control": {"tiles": int(len(br)), "tiles_trusted": int(len(br_trusted)),
                           "offset_px": quartiles(br_trusted.mpre_px),
                           "p95_offset_px": round(float(br_trusted.mpre_px.quantile(0.95)), 3),
                           "share_within_tol": round(float((br_trusted.mpre_px <= TOL).mean()), 4)},
        "pool_mask_to_pre": {"pairs_trusted": int(trusted_pre.sum()), "offset_px": quartiles(lr.mpre_px[trusted_pre]),
                             "share_within_tol": round(float((lr.mpre_px[trusted_pre] <= TOL).mean()), 4)},
        "hypothesis_test": {
            "pairs": int(len(test)), "rule": "pairs with a reliable first-pass shift >= 3 px and a trusted mask->pre offset",
            "labels_on_pre": int(on_pre.sum()), "labels_on_post": int(on_post.sum()),
            "ambiguous": int(len(test) - on_pre.sum() - on_post.sum()),
            "median_px_to_pre_hypothesis": round(float(test.mpre_px.median()), 3),
            "median_px_to_post_hypothesis": round(float(test.mpre_vs_post_hyp_px.median()), 3),
            "by_origin": {o: {"pairs": int(len(g)), "on_pre": int(((g.mpre_px <= TOL) & (g.mpre_vs_post_hyp_px > TOL)).sum()),
                              "on_post": int(((g.mpre_vs_post_hyp_px <= TOL) & (g.mpre_px > TOL)).sum())}
                          for o, g in test.groupby("origin")},
        },
        "labels_to_post_roofs": {
            "pairs_trusted": int(len(on_post_img)), "offset_px": quartiles(on_post_img.mpost_px),
            **shares(on_post_img.mpost_px),
            "val": {"pairs_trusted": int((on_post_img.split == "val").sum()),
                    "offset_px": quartiles(on_post_img.mpost_px[on_post_img.split == "val"]),
                    **shares(on_post_img.mpost_px[on_post_img.split == "val"])},
            "by_event": {ev: {"pairs": int(len(g)), "median_px": round(float(g.mpost_px.median()), 2),
                              "share_over_3px": round(float((g.mpost_px > 3).mean()), 3)}
                         for ev, g in on_post_img.groupby("event") if len(g) >= 10},
            "by_origin": {o: {"pairs": int(len(g)), "median_px": round(float(g.mpost_px.median()), 2),
                              "share_over_3px": round(float((g.mpost_px > 3).mean()), 3)}
                          for o, g in on_post_img.groupby("origin")},
        },
        "roof_vs_image_shift": {
            "rule": "roof shift = mask->post minus mask->pre (both trusted); image shift = the first pass's "
                    "phase correlation, on the same pairs",
            "pairs": int(len(same)), "roof_shift_px": quartiles(same.roof_shift_px),
            "image_shift_px": quartiles(same.audit_shift_px),
            "share_roof_over_2px": round(float((same.roof_shift_px > 2).mean()), 4),
            "share_image_over_2px": round(float((same.audit_shift_px > 2).mean()), 4),
        },
        "pre_post_shift_cross_check": {"pairs": int(len(both)),
                                       "median_disagreement_px": round(float(np.hypot(both.pp_dx - both.audit_dx,
                                                                                      both.pp_dy - both.audit_dy).median()), 3),
                                       "share_within_1px": round(float((np.hypot(both.pp_dx - both.audit_dx,
                                                                                 both.pp_dy - both.audit_dy) <= 1).mean()), 4)},
        "post_only_iou_ceiling": {
            "rule": f"labels vs the same labels moved by the measured label->post offset (none under {NOISE_PX} px); "
                    "pooled intersection / union over the pairs where that offset is measured",
            "pairs": int(len(with_iou)), "all": ceiling(with_iou),
            "val": ceiling(with_iou[with_iou.split == "val"]), "val_pairs": int((with_iou.split == "val").sum()),
            "by_origin": {o: {"pairs": int(len(g)), **ceiling(g)} for o, g in with_iou.groupby("origin")},
            "by_event": {ev: {"pairs": int(len(g)), **{f"iou_{k}": v for k, v in ceiling(g).items()
                                                       if k in ("building", "destroyed")}}
                         for ev, g in with_iou.groupby("event") if len(g) >= 10},
        },
    }
    keep = ["pre", "post", "split", "event", "hazard", "origin", "building_px", "standing_px", "audit_dx", "audit_dy",
            "audit_shift_px", "pp_dx", "pp_dy", "pp_q", "mpre_dx", "mpre_dy", "mpre_q", "mpost_dx", "mpost_dy",
            "mpost_q", "mpre_px", "mpre_vs_post_hyp_px", "mpost_px", "roof_shift_px"] + \
           [c for c in lr.columns if c.startswith(("inter_", "union_"))]
    lr[[c for c in keep if c in lr.columns]].to_csv(out / "label_registration.csv", index=False)
    br.to_csv(out / "bright_label_registration.csv", index=False)

    # ---- 2b. Feature gap: tables ------------------------------------------------------------------
    n_post, n_pre = len(pairs), len(pool_pre)
    X = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    X_post, X_pre, X_br = X[:n_post], X[n_post:n_post + n_pre], X[n_post + n_pre:]
    copies = set(images.loc[images.identical_bright_tile.notna(), "name"])
    pre_not_copy = np.array([n not in copies for n in pool_pre])
    br_events = bright_pre.event.to_numpy()
    ours = np.isin(br_events, list(OUR_BRIGHT_EVENTS))
    Z = PCA(128, random_state=0).fit(X).transform(X)
    Z_post, Z_pre, Z_br = Z[:n_post], Z[n_post:n_post + n_pre], Z[n_post + n_pre:]

    def auc(a, b):
        y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        s = cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"), np.r_[a, b], y, cv=cv, scoring="roc_auc")
        return round(float(s.mean()), 4)
    split = pairs.split.to_numpy()
    sep = {"pool post vs BRIGHT pre (11 events)": auc(Z_post, Z_br),
           "pool post vs BRIGHT pre (our 5 events)": auc(Z_post, Z_br[ours]),
           "pool pre (no BRIGHT copies) vs BRIGHT pre (11 events)": auc(Z_pre[pre_not_copy], Z_br),
           "control: pool pre vs pool post": auc(Z_pre, Z_post),
           "control: pool train post vs frozen-val post": auc(Z_post[split == "train"], Z_post[split == "val"])}

    def nearest(a, b, chunk=1024, exclude_self=False):
        best = np.empty(len(a), np.float32)
        for i in range(0, len(a), chunk):
            s = a[i:i + chunk] @ b.T
            if exclude_self:
                s[np.arange(len(s)), np.arange(i, i + len(s))] = -1
            best[i:i + chunk] = s.max(axis=1)
        return 1 - best                                    # cosine distance to the nearest image
    pool_nn = nearest(X_post, X_post, exclude_self=True)
    edge_d = float(np.quantile(pool_nn, 0.95))
    br_nn = nearest(X_br, X_post)
    post_events = pairs.event.to_numpy()
    cent_dm3 = {ev: X_post[post_events == ev].mean(axis=0) for ev in np.unique(post_events) if (post_events == ev).sum() >= 10}
    cent_dm3 = {k: v / np.linalg.norm(v) for k, v in cent_dm3.items()}
    by_event = []
    for ev in sorted(np.unique(br_events)):
        sel = br_events == ev
        c = X_br[sel].mean(axis=0)
        c /= np.linalg.norm(c)
        sims = sorted(((float(c @ v), k) for k, v in cent_dm3.items()), reverse=True)
        by_event.append({"bright_event": ev, "tiles": int(sel.sum()), "our_subset": ev in OUR_BRIGHT_EVENTS,
                         "median_nn_distance": round(float(np.median(br_nn[sel])), 4),
                         "share_beyond_pool_p95": round(float((br_nn[sel] > edge_d).mean()), 4),
                         "nearest_pool_event": sims[0][1], "nearest_pool_hazard": hazard(sims[0][1]),
                         "nearest_similarity": round(sims[0][0], 4),
                         "second_pool_event": sims[1][1] if len(sims) > 1 else None})
    S["feature_gap"] = {
        "model": "torchvision ResNet-50 IMAGENET1K_V2, global-average-pooled 2,048-d, whole 1024-px tile resized to "
                 "224 px, L2-normalised; separability = 5-fold logistic-regression ROC AUC on 128 PCA components",
        "images": {"pool_post": n_post, "pool_pre": n_pre, "bright_pre": int(len(X_br))},
        "auc": sep,
        "pool_nn_distance": quartiles(pool_nn), "pool_nn_p95": round(edge_d, 4),
        "bright_nn_distance": quartiles(br_nn),
        "bright_our5_nn_distance": quartiles(br_nn[ours]),
        "share_bright_beyond_pool_p95": round(float((br_nn > edge_d).mean()), 4),
        "share_our5_beyond_pool_p95": round(float((br_nn[ours] > edge_d).mean()), 4),
        "by_bright_event": by_event,
    }
    pd.DataFrame(by_event).to_csv(out / "feature_gap_by_bright_event.csv", index=False)
    rng = np.random.default_rng(0)
    pick_post = rng.choice(n_post, size=min(2500, n_post), replace=False)
    pick_br = np.concatenate([rng.choice(np.flatnonzero(br_events == ev), size=min(300, (br_events == ev).sum()), replace=False)
                              for ev in np.unique(br_events)])
    emb = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(
        np.r_[Z_post[pick_post][:, :50], Z_br[pick_br][:, :50]])
    pd.DataFrame({"x": emb[:, 0].round(3), "y": emb[:, 1].round(3),
                  "dataset": ["DisasterM3 pool post"] * len(pick_post) + ["BRIGHT pre"] * len(pick_br),
                  "event": list(post_events[pick_post]) + list(br_events[pick_br]),
                  "hazard": [hazard(e) for e in post_events[pick_post]] + [hazard(e.replace("-", "_")) for e in br_events[pick_br]],
                  "split": list(split[pick_post]) + ["-"] * len(pick_br)}).to_csv(out / "feature_gap_tsne.csv", index=False)

    # ---- 3. Re-tiled overlap ----------------------------------------------------------------------
    pool_split = dict(zip(pairs.pre, pairs.split))
    rows, retiled = [], {}
    for b_ev, prefix in RETILED.items():
        b_paths = [bright_files[n] for n in bright_pre.name[bright_pre.event == b_ev]]
        d_names = images.name[(images.prefix == prefix) & (images.role == "pre") & (images.pixel_modality == "optical")]
        if args.limit:
            b_paths, d_names = b_paths[:args.limit // 4], d_names[:args.limit // 10]
        print(f"[*] SIFT overlap {b_ev}: {len(b_paths)} BRIGHT tiles x {len(d_names)} DisasterM3 pre images")
        hits = retiled_overlap(b_paths, [img_dir / n for n in d_names])
        for h in hits:
            h.update(bright_event=b_ev, split=pool_split.get(h["dm3_image"], "not in pool"))
        rows += hits
        h = pd.DataFrame(hits, columns=["dm3_image", "bright_tile", "inliers", "scale", "dm3_share_in_tile", "split"])
        gsd = float(bright_pre.gsd_m[bright_pre.event == b_ev].median())
        matched = h.groupby("dm3_image").dm3_share_in_tile.sum() if len(h) else pd.Series(dtype=float)
        best = h.sort_values("inliers").groupby("dm3_image").tail(1) if len(h) else h
        retiled[b_ev] = {
            "bright_tiles": len(b_paths), "dm3_pre_images": int(len(d_names)),
            "dm3_images_matched": int(len(matched)), "bright_tiles_matched": int(h.bright_tile.nunique()) if len(h) else 0,
            "scale_bright_px_per_dm3_px": quartiles(best.scale) if len(best) else None,
            "bright_gsd_m": round(gsd, 3),
            "implied_dm3_gsd_m": round(float(best.scale.median() * gsd), 3) if len(best) else None,
            "median_dm3_share_on_bright_ground": round(float(matched.clip(upper=1).median()), 3) if len(matched) else None,
            "pool_images_matched": {s: int((best.split == s).sum()) for s in ("train", "val")} if len(best) else {},
        }
    S["retiled_overlap"] = retiled
    pd.DataFrame(rows).to_csv(out / "retiled_matches.csv", index=False)

    # ---- 4. Spot checks ---------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    modes = {}
    for p in (bright / "post-event").rglob("*_post_disaster.tif"):
        with Image.open(p) as im:
            modes[im.mode] = modes.get(im.mode, 0) + 1
    cp = images[images.identical_bright_tile.notna()].sample(40, random_state=1)
    copy_ok = sum(same_pixels((img_dir / a, bright_files[b])) for a, b in zip(cp.name, cp.identical_bright_tile))
    sar_hidden = images[(images.role == "post") & (images.pixel_modality == "SAR") & ~images.sar_in_name].sample(60, random_state=1)
    opt_post = images[(images.role == "post") & (images.pixel_modality == "optical")].sample(60, random_state=1)
    leak_ok = sum(same_pixels((img_dir / v.replace("_post_", "_pre_"), img_dir / t.replace("_post_", "_pre_")))
                  for v, t in zip(leaks.val_post, leaks.train_post))
    _, train, val, _ = build_optical_pool(instruct)
    S["spot_checks"] = {
        "bright_post_pil_modes": modes,
        "byte_copies_rechecked": {"sampled": 40, "identical_by_array_compare": int(copy_ok)},
        "sar_posts_without_sar_in_name": {"sampled": 60, "single_channel": int(sum(channel_gap(img_dir / n) < 1 for n in sar_hidden.name))},
        "optical_posts": {"sampled": 60, "colour": int(sum(channel_gap(img_dir / n) >= 1 for n in opt_post.name))},
        "val_leak_pre_images_identical": {"pairs": int(len(leaks)), "identical": int(leak_ok)},
        "pool_split_sizes": [len(train), len(val)],
        "val_matches_first_pass": sorted(p["post"].name for p in val) == sorted(pairs.post[pairs.split == "val"]),
    }

    (out / "summary_ext.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
    L = S["label_registration"]
    print(f"[*] BRIGHT control offset      : {L['bright_control']['offset_px']}")
    print(f"[*] Labels on pre / post / ?   : {L['hypothesis_test']['labels_on_pre']} / "
          f"{L['hypothesis_test']['labels_on_post']} / {L['hypothesis_test']['ambiguous']}")
    print(f"[*] Labels vs post roofs       : {L['labels_to_post_roofs']['offset_px']}")
    print(f"[*] Post-only IoU ceiling      : {L['post_only_iou_ceiling']['all']}")
    print(f"[*] Feature AUC                : {S['feature_gap']['auc']}")
    print(f"[*] Re-tiled overlap           : { {k: (v['dm3_images_matched'], v['implied_dm3_gsd_m']) for k, v in retiled.items()} }")
    print(f"[*] Spot checks                : {S['spot_checks']}")
    print(f"[OK] Written to {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
