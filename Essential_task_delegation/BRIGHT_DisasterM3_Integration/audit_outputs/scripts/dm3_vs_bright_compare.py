#!/usr/bin/env python3
"""
DisasterM3 vs BRIGHT -- EO-only audit/EDA comparison (SAR excluded).

  DisasterM3 : pre + post optical pairs
  BRIGHT     : pre-event optical baseline only (BRIGHT's post-event tiles are SAR -> excluded;
               use --bright-post-optical if you have the post-event optical subset)

  python dm3_vs_bright_compare.py --dm3 /data/DisasterM3 --bright /data/BRIGHT --out cmp_out

Covers: spatial resolution (pixel size, GeoTIFF GSD, sharpness / resampling proxies),
class imbalance (damage masks + event imbalance), registration (pre/post phase correlation),
domain shift (histogram JS divergence across dataset/role/event), and cross-dataset overlap
(DisasterM3 reuses BRIGHT/xBD imagery, so near-duplicates across datasets are expected).
Requires dm3_bright_audit.py in the same folder.
"""
import argparse, math, random, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dm3_bright_audit as A  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
BRIGHT_EVENT_RE = r"([A-Za-z\-]+)_\d{4,}"
PUB = pd.DataFrame([
    ["Purpose", "VLM benchmark/instruction data (9 tasks, 123,010 QA pairs)", "Building damage segmentation benchmark"],
    ["Events", "36 (26 from xBD+BRIGHT, 10 new from Maxar Open Data)", "14 events"],
    ["Pre/post optical", "Yes, co-registered pairs (15,881 pairs)", "Pre-event optical only (post-event optical for a 5-event subset)"],
    ["Pixel size", "All resampled to 0.8 m", "0.3-1 m native, event-dependent"],
    ["Damage scheme", "Background/Intact/Damaged/Destroyed (FEMA)", "Intact/Damaged/Destroyed (1/2/3), 'possibly damaged' -> intact"],
    ["Imbalance (published)", "Skewed damaged buildings/roads (Fig. 2, no numbers in text)", "Bldg:bg ~1:7; 82.8/10.7/6.5% intact/damaged/destroyed"],
    ["Registration (published)", "'strictly aligned' (no error reported)", "Optical-SAR RMSE ~1.024 px (1.006 px in buildings), expert control points"],
    ["License", "Academic use only", "CC BY-NC 4.0"],
], columns=["aspect", "DisasterM3 (arXiv 2505.21089)", "BRIGHT (ESSD 17, 6217)"])


# ------------------------------------------------------------------ pixel helpers
def load_rgb(path):
    with Image.open(path) as im:
        if im.mode == "P":
            im = im.convert("RGB")
        a = np.asarray(im)
    if a.dtype != np.uint8:
        a = (a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else max(float(a.max()), 1.0)) * 255).clip(0, 255)
    a = a.astype(np.float32)
    if a.ndim == 2:
        a = a[..., None]
    return a[..., :3] if a.shape[2] >= 3 else np.repeat(a[..., :1], 3, 2), a.shape[2]


def geo_gsd(path):
    """Approx. pixel size in metres from GeoTIFF tags (ModelPixelScale 33550, TiePoint 33922). NaN if absent."""
    try:
        with Image.open(path) as im:
            tg = getattr(im, "tag_v2", None)
            if not tg or 33550 not in tg:
                return np.nan, np.nan
            sx, sy = float(tg[33550][0]), float(tg[33550][1])
            if sx < 0.01:  # degrees
                lat = float(tg[33922][4]) if 33922 in tg and len(tg[33922]) >= 5 else 0.0
                return sx * 111320 * math.cos(math.radians(lat)), sy * 110540
            return sx, sy
    except Exception:
        return np.nan, np.nan


def hf_ratio(g):
    n = min(256, *g.shape)
    c = g[g.shape[0] // 2 - n // 2: g.shape[0] // 2 + n // 2, g.shape[1] // 2 - n // 2: g.shape[1] // 2 + n // 2]
    c = (c - c.mean()) * np.outer(np.hanning(n), np.hanning(n))
    P = np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(n)), np.fft.fftshift(np.fft.fftfreq(n)), indexing="ij")
    r = np.hypot(fx, fy)
    return float(P[r > 0.25].sum() / max(P[r > 0.02].sum(), 1e-9))


def stats(path, source):
    rgb, bands = load_rgb(path)
    g = rgb.mean(2)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    gx, gy = geo_gsd(path)
    th = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((8, 8)), dtype=np.float32)
    ah = int("".join("1" if b else "0" for b in (th > th.mean()).flatten()), 2)
    d = dict(h=g.shape[0], w=g.shape[1], bands=bands, gsd_x=gx, gsd_y=gy, lap_var=float(lap.var()),
             hf_ratio=hf_ratio(g), ahash=ah, **{f"m{i}": float(rgb[..., i].mean()) for i in range(3)},
             **{f"s{i}": float(rgb[..., i].std()) for i in range(3)})
    d["hist"] = np.concatenate([np.histogram(rgb[..., i], bins=32, range=(0, 255))[0] for i in range(3)]).astype(np.float64)
    return d


def js(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    k = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.maximum(b, 1e-12)), 0)))
    return 0.5 * k(p, m) + 0.5 * k(q, m)


def js3(h1, h2):
    return float(np.mean([js(h1[i * 32:(i + 1) * 32], h2[i * 32:(i + 1) * 32]) for i in range(3)]))


def phase_shift(a, b):
    a, b = a - a.mean(), b - b.mean()
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    R = np.fft.fft2(a * w) * np.conj(np.fft.fft2(b * w))
    r = np.fft.ifft2(R / (np.abs(R) + 1e-9)).real
    iy, ix = np.unravel_index(np.argmax(r), r.shape)

    def sub(c, n, get):
        l, m, rr = get(c - 1), get(c), get(c + 1)
        den = l - 2 * m + rr
        return c + (0.5 * (l - rr) / den if abs(den) > 1e-12 else 0.0)

    fy = sub(iy, r.shape[0], lambda k: r[k % r.shape[0], ix])
    fx = sub(ix, r.shape[1], lambda k: r[iy, k % r.shape[1]])
    wrap = lambda v, n: v - n if v > n / 2 else v
    return wrap(fx, r.shape[1]), wrap(fy, r.shape[0]), float(r[iy, ix])


def pair_registration(pre_path, post_path, crop=512):
    a, _ = load_rgb(pre_path)
    b, _ = load_rgb(post_path)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    n = min(crop, h, w)
    y0, x0 = (h - n) // 2, (w - n) // 2
    ga, gb = a[y0:y0 + n, x0:x0 + n].mean(2), b[y0:y0 + n, x0:x0 + n].mean(2)
    dx, dy, pk = phase_shift(ga, gb)
    return dx, dy, pk, (a.shape[:2] != b.shape[:2])


# ------------------------------------------------------------------ pipeline
def collect(name, root, depth, regex, bright_post_optical):
    inv = A.build_inventory([root], depth, regex)
    inv["dataset"] = name
    if name == "BRIGHT" and not bright_post_optical:
        inv.loc[inv.role == "post", "sar"] = True      # BRIGHT post-event tiles are SAR
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm3", required=True)
    ap.add_argument("--bright", required=True)
    ap.add_argument("--out", default="cmp_out")
    ap.add_argument("--dm3-event-depth", type=int, default=1)
    ap.add_argument("--dm3-event-regex", default=None)
    ap.add_argument("--bright-event-regex", default=BRIGHT_EVENT_RE)
    ap.add_argument("--bright-post-optical", action="store_true", help="BRIGHT post-event folder contains optical")
    ap.add_argument("--max-images", type=int, default=2000, help="per dataset/role")
    ap.add_argument("--max-pairs", type=int, default=600)
    ap.add_argument("--min-peak", type=float, default=0.05, help="phase-corr peak needed to trust a shift")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    out = Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    inv = pd.concat([collect("DisasterM3", a.dm3, a.dm3_event_depth, a.dm3_event_regex, True),
                     collect("BRIGHT", a.bright, 1, a.bright_event_regex, a.bright_post_optical)], ignore_index=True)
    inv.drop(columns=["path"]).to_csv(out / "inventory.csv", index=False)
    flags, R = [], ["# DisasterM3 vs BRIGHT -- EO-only comparison\n"]
    imgs = inv[inv.kind == "image"]
    opt = imgs[~imgs.sar & imgs.role.isin(["pre", "post"])]
    opt = opt[~((opt.dataset == "BRIGHT") & (opt.role == "post") & (not a.bright_post_optical))]
    R.append("Scope: SAR excluded. DisasterM3 = pre+post optical; BRIGHT = pre-event optical (post-event is SAR"
             + (", overridden by --bright-post-optical" if a.bright_post_optical else "") + ").\n")
    R.append("## 0. Published facts side by side\n" + A.md(PUB))
    R.append("\n## 1. Inventory\n")
    R.append(A.md(imgs.groupby(["dataset", "role", "sar"]).size().rename("images").reset_index()))
    R.append(f"- SAR files excluded: {int(imgs.sar.sum())}; optical kept: {len(opt)}\n")
    if not len(opt):
        (out / "compare_report.md").write_text("\n".join(R + ["\n**No optical images found -- check roots/tokens.**"]))
        print("no optical images"); return
    # ---- per-image stats
    recs = []
    for (ds, role), g in opt.groupby(["dataset", "role"]):
        for r in g.sample(min(a.max_images, len(g)), random_state=a.seed).itertuples():
            try:
                s = stats(r.path, ds)
                recs.append(dict(dataset=ds, role=role, event=r.event, rel=r.rel, pair_key=r.pair_key, **s))
            except Exception as e:
                flags.append(f"unreadable: {r.rel} ({repr(e)[:60]})")
    st = pd.DataFrame(recs)
    hist = np.stack(st.pop("hist").to_numpy()) if len(st) else np.zeros((0, 96))
    st.drop(columns=["ahash"]).to_csv(out / "image_stats.csv", index=False)
    st["grp"] = st.dataset + "/" + st.role
    if (st.bands == 1).any():
        flags.append(f"{int((st.bands == 1).sum())} single-band 'optical' images (possible SAR).")

    # ---- 2. spatial resolution
    R.append("\n## 2. Spatial resolution\n")
    res = st.groupby("grp").agg(n=("w", "size"), w_med=("w", "median"), h_med=("h", "median"),
                                n_sizes=("w", lambda s: s.nunique()), gsd_x_med=("gsd_x", "median"),
                                gsd_x_min=("gsd_x", "min"), gsd_x_max=("gsd_x", "max"),
                                lap_var=("lap_var", "median"), hf_ratio=("hf_ratio", "median")).round(3).reset_index()
    R.append(A.md(res))
    R.append("- `gsd_*` comes from GeoTIFF tags (metres, approximate); NaN means no georeference (PNG/JPG). "
             "Published: DisasterM3 = 0.8 m for every image; BRIGHT = 0.3-1 m native.\n"
             "- `lap_var`/`hf_ratio` are sharpness proxies: low `hf_ratio` suggests upsampling/blur, high suggests native or downsampled detail.\n")
    d3 = st[st.dataset == "DisasterM3"].gsd_x.dropna()
    if len(d3) and (d3.max() - d3.min() > 0.05):
        flags.append(f"DisasterM3 GSD spread {d3.min():.2f}-{d3.max():.2f} m, but paper states uniform 0.8 m.")
    ev_res = st.groupby(["dataset", "event"]).agg(n=("w", "size"), w_med=("w", "median"), gsd_med=("gsd_x", "median"),
                                                  hf=("hf_ratio", "median")).round(3).reset_index()
    ev_res.to_csv(out / "resolution_per_event.csv", index=False)

    # ---- 3. class imbalance
    R.append("\n## 3. Class imbalance\n")
    for ds in ("DisasterM3", "BRIGHT"):
        m = inv[(inv.dataset == ds) & (inv.kind == "mask") & (~inv.sar)]
        if not len(m):
            R.append(f"- **{ds}**: no mask files detected by path tokens (mask/label/gt/target/seg). "
                     "If labels are inside tables, use dm3_bright_audit.py's table profile.\n")
            continue
        cnt, per = Counter(), {}
        for r in m.sample(min(1500, len(m)), random_state=a.seed).itertuples():
            try:
                with Image.open(r.path) as im:
                    arr = np.asarray(im)
                v, c = np.unique(arr if arr.ndim == 2 else arr[..., 0], return_counts=True)
                for vi, ci in zip(v, c):
                    cnt[int(vi)] += int(ci); per.setdefault(r.event, Counter())[int(vi)] += int(ci)
            except Exception:
                pass
        tot = sum(cnt.values()) or 1
        bld = {k: v for k, v in cnt.items() if k > 0}
        bt = sum(bld.values()) or 1
        R.append(f"- **{ds}** ({len(m)} masks, sampled): bg {cnt.get(0, 0) / tot:.3f} | within-building shares "
                 f"{ {k: round(v / bt, 3) for k, v in sorted(bld.items())} } | building:bg = 1:{(cnt.get(0, 0) / bt):.1f}\n")
        pe = pd.DataFrame(per).fillna(0).T
        pe = pe.div(pe.sum(axis=1), axis=0).round(4); pe.index.name = "event"
        pe.to_csv(out / f"mask_values_per_event_{ds}.csv")
        if bld and max(bld.values()) / max(min(bld.values()), 1) > 5:
            flags.append(f"{ds}: building-class imbalance max/min > 5x ({ {k: round(v / bt, 3) for k, v in bld.items()} }).")
    ec = opt.groupby(["dataset", "event"]).size().rename("images").reset_index()
    for ds, g in ec.groupby("dataset"):
        p = g.images / g.images.sum()
        H = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 1.0
        R.append(f"- **{ds} event balance**: {len(g)} events, top event {p.max():.1%} of optical images, normalised entropy {H:.2f} (1 = uniform).\n")
        if p.max() > 0.25:
            flags.append(f"{ds}: one event holds {p.max():.0%} of optical images.")
    ec.to_csv(out / "events_per_dataset.csv", index=False)

    # ---- 4. registration
    R.append("\n## 4. Registration alignment (optical pre vs post, centre-crop phase correlation)\n")
    reg = []
    po = opt[(opt.dataset == "DisasterM3")]
    if (po.role == "post").any():
        k = po.groupby("pair_key").role.agg(lambda s: set(s))
        keys = [x for x, v in k.items() if v == {"pre", "post"}]
        random.shuffle(keys)
        pth = po.set_index(["pair_key", "role"]).path
        for key in keys[:a.max_pairs]:
            try:
                dx, dy, pk, mism = pair_registration(pth[(key, "pre")], pth[(key, "post")])
                reg.append(dict(pair_key=key, event=po[po.pair_key == key].event.iloc[0], dx=dx, dy=dy,
                                shift_px=math.hypot(dx, dy), peak=pk, size_mismatch=mism))
            except Exception:
                pass
    rg = pd.DataFrame(reg)
    if len(rg):
        rg.to_csv(out / "registration_pairs.csv", index=False)
        ok = rg[rg.peak >= a.min_peak]
        R.append(f"- DisasterM3 pairs measured: {len(rg)} | reliable (peak>={a.min_peak}): {len(ok)} | size mismatch: {int(rg.size_mismatch.sum())}\n")
        if len(ok):
            q = ok.shift_px.quantile([.5, .9, .99]).round(2).tolist()
            R.append(f"- Shift magnitude (px) p50/p90/p99: {q} | RMSE: {math.sqrt((ok.shift_px ** 2).mean()):.2f} px | share >1 px: {(ok.shift_px > 1).mean():.1%} | >2 px: {(ok.shift_px > 2).mean():.1%}\n")
            R.append(A.md(ok.groupby("event").shift_px.agg(n="size", p50="median", p90=lambda s: s.quantile(.9)).round(2).reset_index().sort_values("p90", ascending=False), 15))
            if (ok.shift_px > 2).mean() > 0.05:
                flags.append(f"DisasterM3: {(ok.shift_px > 2).mean():.0%} of reliable pairs shift >2 px despite 'strictly aligned' claim (at 0.8 m, 2 px = 1.6 m).")
        if rg.size_mismatch.any():
            flags.append(f"{int(rg.size_mismatch.sum())} DisasterM3 pre/post pairs differ in size.")
    else:
        R.append("- No complete DisasterM3 pre/post optical pairs measured (check pre/post tokens or pairing).\n")
    R.append("- BRIGHT: optical-SAR registration is out of scope here; paper reports ~1.024 px RMSE (1.006 px in buildings) via RIFT/SRIF/LNIFT proxies. "
             "Phase correlation on pre/post optical includes true damage change, so treat shifts as an upper-bound indicator, not ground truth.\n")

    # ---- 5. domain shift
    R.append("\n## 5. Domain shift (colour/radiometry; mean JS divergence over R,G,B 32-bin histograms, 0 = identical, 1 = disjoint)\n")
    grp_h = {g: hist[(st.grp == g).to_numpy()].sum(0) for g in st.grp.unique()}
    names = sorted(grp_h)
    M = pd.DataFrame([[js3(grp_h[x], grp_h[y]) for y in names] for x in names], index=names, columns=names).round(4)
    M.to_csv(out / "js_dataset_role.csv")
    R.append(A.md(M.reset_index().rename(columns={"index": "group"})))
    ev_h = {}
    for (ds, ev), g in st.groupby(["dataset", "event"]):
        if len(g) >= 5:
            ev_h[(ds, ev)] = hist[g.index.to_numpy()].sum(0)
    def spread(ds):
        ks = [k for k in ev_h if k[0] == ds]
        v = [js3(ev_h[x], ev_h[y]) for i, x in enumerate(ks) for y in ks[i + 1:]]
        return (float(np.mean(v)), len(ks)) if v else (np.nan, len(ks))
    sp = {ds: spread(ds) for ds in ("DisasterM3", "BRIGHT")}
    R.append(f"- Within-dataset event-to-event JS (events with >=5 sampled images): DisasterM3 {sp['DisasterM3'][0]:.3f} ({sp['DisasterM3'][1]} events), "
             f"BRIGHT {sp['BRIGHT'][0]:.3f} ({sp['BRIGHT'][1]} events).\n")
    if "DisasterM3/pre" in grp_h and "BRIGHT/pre" in grp_h:
        cross = js3(grp_h["DisasterM3/pre"], grp_h["BRIGHT/pre"])
        R.append(f"- Cross-dataset pre-event JS (DisasterM3 vs BRIGHT): **{cross:.3f}** -- compare to event spread above: a cross value at or below the within-dataset event spread means the dataset gap is no larger than normal event-to-event variation.\n")
    if "DisasterM3/pre" in grp_h and "DisasterM3/post" in grp_h:
        dm = st[st.dataset == "DisasterM3"].pivot_table(index="pair_key", columns="role", values="m0", aggfunc="first")
        if {"pre", "post"} <= set(dm.columns):
            sh = (dm["post"] - dm["pre"]).dropna()
            if len(sh):
                R.append(f"- DisasterM3 pre->post mean red-channel shift (paired, sampled): median {sh.median():.1f}, |p95| {sh.abs().quantile(.95):.1f} grey levels (acquisition/season/haze change).\n")
    cs = st.groupby("grp").agg(mean_R=("m0", "mean"), mean_G=("m1", "mean"), mean_B=("m2", "mean"),
                               std_R=("s0", "mean"), std_G=("s1", "mean"), std_B=("s2", "mean")).round(2).reset_index()
    R.append(A.md(cs))

    # ---- 6. cross-dataset overlap
    R.append("\n## 6. Cross-dataset overlap (DisasterM3 reuses BRIGHT/xBD imagery)\n")
    x = st[st.role == "pre"]
    d, b = x[x.dataset == "DisasterM3"], x[x.dataset == "BRIGHT"]
    if len(d) and len(b) and len(d) * len(b) < 4e7:
        hd, hb = d.ahash.astype("uint64").to_numpy(), b.ahash.astype("uint64").to_numpy()
        hits = []
        for i, h in enumerate(hd):
            dist = np.unpackbits(np.bitwise_xor(h, hb).view(np.uint8).reshape(-1, 8), axis=1).sum(1)
            for j in np.where(dist <= 4)[0]:
                hits.append((d.rel.iloc[i], b.rel.iloc[j], int(dist[j])))
        ov = pd.DataFrame(hits, columns=["dm3", "bright", "hamming"])
        ov.to_csv(out / "cross_dataset_overlap.csv", index=False)
        R.append(f"- Sampled pre-event images: DM3 {len(d)} x BRIGHT {len(b)} -> **{len(ov)}** aHash matches (Hamming<=4), {ov.dm3.nunique() if len(ov) else 0} distinct DM3 images.\n")
        R.append("- Matches are lower-bound evidence (sampling + tiling differences hide some). If you train on one and test on the other, split by event, not by file.\n")
        if len(ov):
            flags.append(f"{ov.dm3.nunique()} sampled DisasterM3 pre images near-match BRIGHT pre images -> leakage risk when combining/cross-testing.")
    else:
        R.append("- Skipped (missing pre images in one dataset or sample too large).\n")

    # ---- figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        for g, d_ in st.groupby("grp"):
            ax[0].hist(d_.m0 * .299 + d_.m1 * .587 + d_.m2 * .114, bins=40, alpha=.5, label=g, density=True)
            ax[1].hist(np.log10(d_.lap_var + 1), bins=40, alpha=.5, label=g, density=True)
            ax[2].hist(d_.hf_ratio, bins=40, alpha=.5, label=g, density=True)
        ax[0].set_title("Mean luminance"); ax[1].set_title("log10 Laplacian var (sharpness)"); ax[2].set_title("High-freq energy ratio"); ax[0].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(out / "figs/domain_stats.png", dpi=130); plt.close()
        if len(rg):
            plt.figure(figsize=(4.5, 4.5)); plt.scatter(rg.dx, rg.dy, s=6, c=(rg.peak >= a.min_peak), alpha=.5)
            plt.axhline(0, c="k", lw=.5); plt.axvline(0, c="k", lw=.5); plt.title("DM3 pre->post shift (px)"); plt.tight_layout()
            plt.savefig(out / "figs/registration_shift.png", dpi=130); plt.close()
    except Exception:
        pass
    R.insert(1, "## Red flags (auto-detected)\n" + ("\n".join(f"- {f}" for f in flags) if flags else "- none detected") + "\n")
    (out / "compare_report.md").write_text("\n".join(R), encoding="utf-8")
    print(f"Wrote {out / 'compare_report.md'} | flags: {len(flags)}")


if __name__ == "__main__":
    main()
