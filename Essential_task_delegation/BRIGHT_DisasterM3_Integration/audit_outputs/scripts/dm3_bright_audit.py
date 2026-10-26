#!/usr/bin/env python3
"""
Audit + EDA for DisasterM3 (+ BRIGHT) -- OPTICAL pre- and post-event images only (SAR excluded).

Usage:
  python dm3_bright_audit.py --root /data/DisasterM3 --root /data/BRIGHT --out audit_out
  Options: --event-depth 1 (path component holding the event name) | --event-regex "(\\w+-\\w+-\\d{4})"
           --max-images 3000 (sample size for pixel stats) | --max-pairs 1000

Outputs (in --out): audit_report.md, *.csv, figs/*.png
Everything is inferred from file paths + table contents, because the DisasterM3 file layout is
distributed via request form; check the "Path layout" section of the report and tune --event-depth.
"""
import argparse, hashlib, json, random, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
TAB_EXT = {".json", ".jsonl", ".csv", ".parquet"}
SAR_TOK = {"sar", "capella", "umbra", "s1"}
PRE_TOK = {"pre", "before", "t1"}
POST_TOK = {"post", "after", "t2"}
MASK_TOK = {"mask", "masks", "label", "labels", "gt", "target", "targets", "seg"}

# Published reference numbers (DisasterM3 arXiv 2505.21089; BRIGHT ESSD 17, 6217)
REF = {
    "DisasterM3 images total": 26988, "DisasterM3 optical": 22214, "DisasterM3 SAR": 4774,
    "DisasterM3 pre/post pairs": 15881, "DisasterM3 instruction pairs": 123010,
    "Instruct: optical/SAR/pairs": "17190 / 3798 / 92968", "Bench: optical/SAR/pairs": "5024 / 976 / 30042",
    "DisasterM3 events": 36, "BRIGHT events": 14, "BRIGHT tiles (1024x1024)": 4246,
    "BRIGHT building px: intact/damaged/destroyed": "82.8% / 10.7% / 6.5%",
    "BRIGHT building:background px ratio": "~1:7",
}


def toks(s):
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def pair_key(rel):
    parts = re.split(r"([^A-Za-z0-9]+)", str(rel.with_suffix("")))
    kept = "".join(p for p in parts if p.lower() not in PRE_TOK | POST_TOK)
    return re.sub(r"[^A-Za-z0-9]+", "/", kept).strip("/")


def event_of(rel, depth, regex):
    if regex:
        m = re.search(regex, str(rel))
        return m.group(1) if m else "unknown"
    return rel.parts[depth - 1] if len(rel.parts) > depth else rel.stem.split("_")[0]


def md(df, n=40):
    if df is None or len(df) == 0:
        return "_none_\n"
    d = df.head(n)
    cols = list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in d.itertuples(index=False)]
    if len(df) > n:
        out.append(f"\n_({len(df) - n} more rows in CSV)_")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- inventory
def build_inventory(roots, depth, regex):
    rows = []
    for root in roots:
        root = Path(root)
        label = set(toks(root.name))
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel, ext = p.relative_to(root), p.suffix.lower()
            tk = set(toks(str(rel.with_suffix(""))))
            kind = ("table" if ext in TAB_EXT else
                    "mask" if ext in IMG_EXT and tk & MASK_TOK else
                    "image" if ext in IMG_EXT else "other")
            role = ("ambiguous" if tk & PRE_TOK and tk & POST_TOK else
                    "pre" if tk & PRE_TOK else "post" if tk & POST_TOK else "unknown")
            src = ("bright" if "bright" in label | tk else "xbd" if "xbd" in label | tk else root.name)
            rows.append(dict(path=str(p), rel=str(rel), root=root.name, source=src, ext=ext, kind=kind,
                             role=role, sar=bool(tk & SAR_TOK), event=event_of(rel, depth, regex),
                             bytes=p.stat().st_size,
                             pair_key=pair_key(rel) if kind in ("image", "mask") else None))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- images
def analyze_image(path):
    with Image.open(path) as im:
        w, h, mode, bands = im.width, im.height, im.mode, len(im.getbands())
        if mode == "P":
            im = im.convert("RGB")
        a = np.asarray(im)
    if a.dtype != np.uint8:
        mx = 65535.0 if a.dtype == np.uint16 else max(float(a.max()), 1.0)
        a = (a.astype(np.float32) / mx * 255).clip(0, 255)
    a = a.astype(np.float32)
    if a.ndim == 2:
        a = a[..., None]
    rgb = a[..., :3] if a.shape[2] >= 3 else np.repeat(a[..., :1], 3, 2)
    lum = rgb.mean(2)
    gray_like = bool(bands >= 3 and np.abs(rgb[..., 0] - rgb[..., 1]).mean() < 1
                     and np.abs(rgb[..., 1] - rgb[..., 2]).mean() < 1)
    small = np.asarray(Image.fromarray(lum.astype(np.uint8)).resize((32, 32)), dtype=np.float32)
    ah = np.asarray(Image.fromarray(lum.astype(np.uint8)).resize((8, 8)), dtype=np.float32)
    bits = (ah > ah.mean()).flatten()
    ahash = int("".join("1" if b else "0" for b in bits), 2)
    return dict(w=w, h=h, mode=mode, bands=bands, dtype=str(np.asarray(Image.open(path)).dtype),
                mean=round(float(lum.mean()), 2), std=round(float(lum.std()), 2),
                zero_frac=round(float((rgb.sum(2) == 0).mean()), 4),
                sat_frac=round(float((lum >= 250).mean()), 4), gray_like=gray_like,
                sha1=hashlib.sha1(Path(path).read_bytes()).hexdigest(), ahash=ahash, _small=small)


def near_dups(df, max_dist=3, cap=6000):
    if len(df) > cap or len(df) < 2:
        return pd.DataFrame()
    h = df["ahash"].astype("uint64").to_numpy()
    paths = df["rel"].to_numpy()
    out = []
    for i in range(len(h) - 1):
        x = np.bitwise_xor(h[i], h[i + 1:])
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.where(d <= max_dist)[0]:
            out.append((paths[i], paths[i + 1 + j], int(d[j])))
    return pd.DataFrame(out, columns=["a", "b", "hamming"])


# ----------------------------------------------------------------------------- tables
def load_table(p):
    p = Path(p)
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext == ".jsonl":
        return pd.read_json(p, lines=True)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        lists = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        obj = max(lists, key=len) if lists else [obj]
    return pd.DataFrame(obj)


def pick(df, names):
    for c in df.columns:
        if any(n in c.lower() for n in names):
            return c


def conv_turn(conv, who):
    if not isinstance(conv, list):
        return None
    for t in conv:
        if isinstance(t, dict) and (t.get("from") or t.get("role")) in who:
            return t.get("value") or t.get("content")


def image_refs(row):
    refs = []
    for v in row:
        for s in (v if isinstance(v, list) else [v]):
            if isinstance(s, str) and Path(s).suffix.lower() in IMG_EXT:
                refs.append(s)
    return refs


def profile_table(path, sar_names, disk_names):
    df = load_table(path)
    res = dict(file=str(path), n=len(df), columns=list(map(str, df.columns)))
    if "conversations" in df.columns:
        df["_q"] = df["conversations"].map(lambda c: conv_turn(c, {"human", "user"}))
        df["_a"] = df["conversations"].map(lambda c: conv_turn(c, {"gpt", "assistant"}))
        qc, ac = "_q", "_a"
    else:
        qc, ac = pick(df, ["question", "prompt", "instruction", "query"]), pick(df, ["answer", "response", "output", "gt"])
    tc = pick(df, ["task", "subset", "category", "qtype"])
    refs = [image_refs(r) for r in df.itertuples(index=False, name=None)]
    df["_refs"] = refs
    df["_sar_row"] = [any((set(toks(r)) & SAR_TOK) or Path(r).name in sar_names for r in rs) for rs in refs]
    res.update(q_col=qc, a_col=ac, task_col=tc, rows_with_image_refs=int(sum(bool(r) for r in refs)),
               rows_touching_sar=int(df["_sar_row"].sum()))
    res["null_rate"] = df.drop(columns=["_refs", "_sar_row"]).isna().mean().round(4).to_dict()
    key = [c for c in (qc, ac) if c] + ["_refs_s"]
    df["_refs_s"] = df["_refs"].map(lambda r: "|".join(r))
    res["exact_duplicate_rows"] = int(df.duplicated(subset=key).sum())
    if qc and ac:
        g = df.groupby([qc, "_refs_s"])[ac].agg(lambda s: s.astype(str).nunique())
        res["conflicting_answers"] = int((g > 1).sum())
        res["q_words_p50_p95"] = [int(x) for x in df[qc].astype(str).str.split().str.len().quantile([.5, .95])]
        res["a_words_p50_p95"] = [int(x) for x in df[ac].astype(str).str.split().str.len().quantile([.5, .95])]
        res["empty_q_or_a"] = int((df[qc].astype(str).str.strip().eq("") | df[ac].astype(str).str.strip().eq("")).sum())
    base = df if not tc else df
    per_task = []
    for t, d in (base.groupby(tc) if tc else [("all", base)]):
        row = dict(task=t, rows=len(d), rows_optical_only=int((~d["_sar_row"]).sum()))
        if ac:
            a = d[ac].astype(str)
            if a.nunique() <= 30:
                row["majority_answer_share"] = round(a.value_counts(normalize=True).iloc[0], 3)
            let = a[a.str.fullmatch(r"\(?[A-E]\)?\.?")].str.strip("().").str.upper()
            if len(let) > 0.5 * len(a):
                row["letter_max_share"] = round(let.value_counts(normalize=True).iloc[0], 3)
        per_task.append(row)
    res["per_task"] = pd.DataFrame(per_task)
    allrefs = {Path(r).name for rs in refs for r in rs}
    res["referenced_but_missing"] = sorted(n for n in allrefs if n not in disk_names)[:50]
    res["n_referenced_missing"] = int(sum(n not in disk_names for n in allrefs))
    res["_ref_names"] = allrefs
    return res


# ----------------------------------------------------------------------------- figures
def figs(out, inv_opt, istats, mask_bal):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    d = Path(out) / "figs"
    d.mkdir(exist_ok=True)
    made = []

    def save(name):
        plt.tight_layout()
        plt.savefig(d / name, dpi=130)
        plt.close()
        made.append(name)

    if len(inv_opt):
        c = inv_opt.groupby(["event", "role"]).size().unstack(fill_value=0)
        c.plot(kind="barh", stacked=True, figsize=(8, max(3, 0.28 * len(c))))
        plt.title("Optical images per event (pre/post)")
        save("events_pre_post.png")
    if len(istats):
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].scatter(istats.w, istats.h, s=6, alpha=.4)
        ax[0].set_title("Image size (w x h)")
        for r, g in istats.groupby("role"):
            ax[1].hist(g["mean"], bins=40, alpha=.5, label=r)
        ax[1].set_title("Mean luminance")
        ax[1].legend()
        ax[2].hist(istats.zero_frac, bins=40)
        ax[2].set_title("Fraction of all-zero pixels (nodata)")
        save("image_stats.png")
    if mask_bal is not None and len(mask_bal):
        (mask_bal.set_index("value")["share"] * 100).plot(kind="bar", figsize=(5, 3.4))
        plt.ylabel("% pixels")
        plt.title("Mask value distribution (sampled)")
        save("mask_values.png")
    return made


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", default="audit_out")
    ap.add_argument("--event-depth", type=int, default=1)
    ap.add_argument("--event-regex", default=None)
    ap.add_argument("--max-images", type=int, default=3000)
    ap.add_argument("--max-pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flags, R = [], []

    inv = build_inventory(a.root, a.event_depth, a.event_regex)
    inv.drop(columns=["path"]).to_csv(out / "inventory.csv", index=False)
    imgs = inv[inv.kind == "image"]
    sar = imgs[imgs.sar]
    opt = imgs[~imgs.sar]
    R.append("# DisasterM3 + BRIGHT -- optical pre/post audit & EDA\n")
    R.append("Scope: optical images only. Files whose path contains a SAR token "
             f"({', '.join(sorted(SAR_TOK))}) are excluded and counted below.\n")
    R.append("## 1. Inventory & SAR exclusion\n")
    R.append(md(inv.groupby(["source", "kind"]).size().reset_index(name="files")))
    R.append(f"- Images: {len(imgs)} total | **SAR excluded: {len(sar)}** | optical kept: {len(opt)}\n")
    R.append(f"- Optical role split: {opt.role.value_counts().to_dict()}\n")
    R.append("### Path layout (use this to tune --event-depth / --event-regex)\n")
    R.append(md(inv.rel.map(lambda s: s.split("/")[0]).value_counts().head(15).rename_axis("top-level").reset_index(name="files")))
    if (opt.role.isin(["unknown", "ambiguous"])).any():
        n = int(opt.role.isin(["unknown", "ambiguous"]).sum())
        flags.append(f"{n} optical images have no/ambiguous pre|post token in path (role cannot be inferred).")
    if inv.event.nunique() > 80:
        flags.append(f"{inv.event.nunique()} distinct 'event' values -- --event-depth is probably wrong.")

    # pairing
    R.append("\n## 2. Pre/post pairing (optical)\n")
    po = opt[opt.role.isin(["pre", "post"])]
    g = po.groupby(["source", "pair_key"]).role.agg(lambda s: tuple(sorted(set(s))))
    paired = g[g == ("post", "pre")]
    pre_only, post_only = g[g == ("pre",)], g[g == ("post",)]
    R.append(f"- Complete pairs: **{len(paired)}** | pre without post: {len(pre_only)} | post without pre: {len(post_only)}\n")
    dupe_roles = po.groupby(["source", "pair_key", "role"]).size()
    if (dupe_roles > 1).any():
        flags.append(f"{int((dupe_roles > 1).sum())} pair keys map to >1 file per role (ambiguous pairing).")
    if len(pre_only) + len(post_only):
        flags.append(f"{len(pre_only)} pre-only and {len(post_only)} post-only optical images (unpaired).")
    ev = po.groupby(["source", "event", "role"]).size().unstack(fill_value=0).reset_index()
    ev["pairs_est"] = ev[[c for c in ("pre", "post") if c in ev]].min(axis=1)
    ev.to_csv(out / "events.csv", index=False)
    R.append(md(ev.sort_values("pairs_est", ascending=False)))
    small_ev = ev[ev.pairs_est < 20]
    if len(small_ev):
        flags.append(f"{len(small_ev)} events have <20 optical pairs: {', '.join(small_ev.event.astype(str).head(8))}.")

    # pixel stats
    R.append("\n## 3. Pixel-level stats (sampled)\n")
    samp = opt.sample(min(a.max_images, len(opt)), random_state=a.seed) if len(opt) else opt
    recs, thumbs, errs = [], {}, []
    for r in samp.itertuples():
        try:
            s = analyze_image(r.path)
            thumbs[r.rel] = s.pop("_small")
            recs.append(dict(rel=r.rel, source=r.source, event=r.event, role=r.role, pair_key=r.pair_key, **s))
        except Exception as e:
            errs.append((r.rel, repr(e)[:120]))
    ist = pd.DataFrame(recs)
    ist.drop(columns=["ahash"], errors="ignore").to_csv(out / "image_stats.csv", index=False)
    R.append(f"- Analyzed {len(ist)} images; unreadable/corrupt: **{len(errs)}**\n")
    if errs:
        flags.append(f"{len(errs)} optical images failed to open (see report).")
        R.append(md(pd.DataFrame(errs, columns=["file", "error"]), 15))
    if len(ist):
        R.append(md(ist.groupby(["source", "role"]).agg(n=("w", "size"), w_med=("w", "median"), h_med=("h", "median"),
                                                          mean_lum=("mean", "mean"), std_lum=("std", "mean"),
                                                          nodata=("zero_frac", "mean")).round(3).reset_index()))
        R.append(f"- Distinct (w,h): {ist.groupby(['w', 'h']).size().sort_values(ascending=False).head(8).to_dict()}\n")
        R.append(f"- Modes: {ist['mode'].value_counts().to_dict()} | dtypes: {ist['dtype'].value_counts().to_dict()}\n")
        gl = int(ist.gray_like.sum())
        if gl:
            flags.append(f"{gl} 'optical' images are channel-identical (grayscale) -- possible SAR leaked past the path filter.")
            ist[ist.gray_like][["rel", "event", "role"]].to_csv(out / "possible_sar_leak.csv", index=False)
        blank = int((ist["std"] < 2).sum())
        if blank:
            flags.append(f"{blank} near-constant images (std<2).")
        if (ist.zero_frac > 0.3).any():
            flags.append(f"{int((ist.zero_frac > 0.3).sum())} images have >30% all-zero pixels (nodata padding).")
        # duplicates
        R.append("\n## 4. Duplicates & leakage\n")
        ex = ist[ist.duplicated("sha1", keep=False)].sort_values("sha1")
        R.append(f"- Exact byte-duplicates (within sample): **{len(ex)}** files in {ex.sha1.nunique()} groups\n")
        if len(ex):
            ex[["rel", "sha1"]].to_csv(out / "exact_duplicates.csv", index=False)
            flags.append(f"{len(ex)} exact-duplicate optical files in sample.")
        nd = near_dups(ist)
        R.append(f"- Near-duplicates (aHash Hamming<=3, within sample): **{len(nd)}** pairs\n")
        if len(nd):
            nd.to_csv(out / "near_duplicates.csv", index=False)
        # pair checks
        pr = []
        bykey = ist.groupby(["source", "pair_key", "role"]).rel.first().unstack()
        if {"pre", "post"} <= set(bykey.columns):
            for (src, k), row in bykey.dropna(subset=["pre", "post"]).head(a.max_pairs).iterrows():
                x, y = ist[ist.rel == row["pre"]].iloc[0], ist[ist.rel == row["post"]].iloc[0]
                pr.append(dict(source=src, pair_key=k, same_size=(x.w, x.h) == (y.w, y.h),
                               identical_bytes=x.sha1 == y.sha1,
                               mean_abs_diff=round(float(np.abs(thumbs[x.rel] - thumbs[y.rel]).mean()), 2),
                               lum_shift=round(float(y["mean"] - x["mean"]), 2)))
        pr = pd.DataFrame(pr)
        if len(pr):
            pr.to_csv(out / "pair_checks.csv", index=False)
            R.append(f"- Pairs fully inside the sample: {len(pr)} | size mismatch: **{int((~pr.same_size).sum())}** | "
                     f"pre==post bytes: **{int(pr.identical_bytes.sum())}** | near-zero change (diff<1.0): **{int((pr.mean_abs_diff < 1).sum())}**\n")
            R.append(f"- Pre->post mean-abs-diff percentiles (5/50/95): {pr.mean_abs_diff.quantile([.05, .5, .95]).round(2).tolist()}\n")
            if (~pr.same_size).any():
                flags.append(f"{int((~pr.same_size).sum())} pairs have different pre/post dimensions (misregistration risk).")
            if pr.identical_bytes.any():
                flags.append(f"{int(pr.identical_bytes.sum())} pairs have identical pre and post files.")
        else:
            R.append("- No complete pre/post pairs fell inside the sample; raise --max-images.\n")
    # masks
    R.append("\n## 5. Masks / damage labels (non-SAR)\n")
    msk = inv[(inv.kind == "mask") & (~inv.sar)]
    mbal = None
    if len(msk):
        cnt, per_ev = Counter(), {}
        for r in msk.sample(min(1500, len(msk)), random_state=a.seed).itertuples():
            try:
                with Image.open(r.path) as im:
                    arr = np.asarray(im)
                v, c = np.unique(arr if arr.ndim == 2 else arr[..., 0], return_counts=True)
                for vi, ci in zip(v, c):
                    cnt[int(vi)] += int(ci)
                    per_ev.setdefault(r.event, Counter())[int(vi)] += int(ci)
            except Exception:
                pass
        tot = sum(cnt.values()) or 1
        mbal = pd.DataFrame(sorted(cnt.items()), columns=["value", "pixels"])
        mbal["share"] = (mbal.pixels / tot).round(4)
        R.append(f"- {len(msk)} mask files; sampled value distribution (BRIGHT convention: 0 bg, 1 intact, 2 damaged, 3 destroyed -- verify for xBD-derived masks):\n")
        R.append(md(mbal))
        pe = pd.DataFrame(per_ev).fillna(0).T
        pe = pe.div(pe.sum(axis=1), axis=0).round(4)
        pe.index.name = "event"
        pe.to_csv(out / "mask_values_per_event.csv")
        bld = mbal[mbal.value > 0]
        if len(bld) and bld.pixels.sum():
            sh = (bld.set_index("value").pixels / bld.pixels.sum()).round(3).to_dict()
            R.append(f"- Share within building pixels: {sh} (BRIGHT paper reports 0.828 / 0.107 / 0.065 for intact/damaged/destroyed)\n")
    else:
        R.append("_No mask files detected by path tokens (mask/label/gt/target/seg); DisasterM3 may store masks inside tables._\n")

    # tables
    R.append("\n## 6. Annotation / instruction tables\n")
    disk_names = set(imgs.rel.map(lambda s: Path(s).name))
    sar_names = set(sar.rel.map(lambda s: Path(s).name)) - set(opt.rel.map(lambda s: Path(s).name))  # SAR-only basenames
    ref_union = set()
    for t in inv[inv.kind == "table"].itertuples():
        try:
            pr_ = profile_table(t.path, sar_names, disk_names)
        except Exception as e:
            R.append(f"- `{t.rel}`: could not parse ({repr(e)[:100]})\n")
            continue
        ref_union |= pr_.pop("_ref_names")
        pt = pr_.pop("per_task")
        pt.to_csv(out / f"tasks_{Path(t.rel).stem}.csv", index=False)
        R.append(f"### `{t.rel}`\n")
        R.append("```json\n" + json.dumps({k: v for k, v in pr_.items() if k != "null_rate"}, indent=1, default=str)[:2500] + "\n```\n")
        R.append(md(pt, 30))
        if pr_["rows_touching_sar"]:
            flags.append(f"{t.rel}: {pr_['rows_touching_sar']}/{pr_['n']} rows reference SAR images -> drop for optical-only runs.")
        if pr_.get("exact_duplicate_rows"):
            flags.append(f"{t.rel}: {pr_['exact_duplicate_rows']} exact duplicate rows.")
        if pr_.get("conflicting_answers"):
            flags.append(f"{t.rel}: {pr_['conflicting_answers']} (question,image) keys with conflicting answers.")
        if pr_["n_referenced_missing"]:
            flags.append(f"{t.rel}: {pr_['n_referenced_missing']} referenced images not found on disk.")
        if "majority_answer_share" in pt and (pt.majority_answer_share > 0.4).any():
            flags.append(f"{t.rel}: tasks with majority-answer baseline >40% -> {pt[pt.majority_answer_share > .4].task.tolist()[:6]}")
        if "letter_max_share" in pt and (pt.letter_max_share > 0.3).any():
            flags.append(f"{t.rel}: MCQ answer-letter skew >30% in {pt[pt.letter_max_share > .3].task.tolist()[:6]} (5 options => 20% uniform).")
    if ref_union:
        orphans = len({n for n in disk_names - sar_names} - ref_union)
        R.append(f"\n- Optical images on disk never referenced by any table: **{orphans}**\n")
        if orphans:
            flags.append(f"{orphans} optical images on disk are not referenced by any annotation table.")

    # reference + flags
    R.append("\n## 7. Published reference numbers (for reconciliation)\n")
    R.append(md(pd.DataFrame(list(REF.items()), columns=["quantity", "published"])))
    R.append(f"- Observed: optical files {len(opt)}, SAR files {len(sar)}, complete optical pairs {len(paired)}, events {inv.event.nunique()}\n")
    R.append("- Counts are NOT expected to match exactly: DisasterM3 reports image counts after tiling/pairing and includes SAR; "
             "BRIGHT's own release is pre-event optical + post-event SAR (post-event optical exists only for a few events).\n")
    made = figs(out, opt, ist if len(ist) else pd.DataFrame(), mbal)
    if made:
        R.append("\nFigures: " + ", ".join(f"figs/{m}" for m in made) + "\n")
    R.insert(2, "## Red flags (auto-detected)\n" + ("\n".join(f"- {f}" for f in flags) if flags else "- none detected") + "\n")
    (out / "audit_report.md").write_text("\n".join(R), encoding="utf-8")
    print(f"Wrote {out/'audit_report.md'}  | flags: {len(flags)}")


if __name__ == "__main__":
    sys.exit(main())
