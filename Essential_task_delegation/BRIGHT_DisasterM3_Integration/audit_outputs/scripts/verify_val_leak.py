"""
verify_val_leak.py
==================
Second, independent check of the train/validation leak in the DisasterM3 vs BRIGHT optical audit
(DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf §8, val_train_leaks.csv), and a comparison with
Abrar's E5 check (Champion_Task/E5-results/E5/leaked_pairs_local_check.csv, 21 pairs).
Written on 26 Oct 2026 for audit_outputs/. It only reads its inputs and writes to --out.

  1. Split    rebuilds the optical pool twice: with Cells 4-5 of Champion_Task/d4-champion-model-R11.ipynb
              (copied below, Kaggle paths swapped for --dm3_root) and with build_optical_pool() of
              bright_dm3_dedup.py. Same pairs, same order, same 645 validation pairs? Also checked against
              the audit's dm3_pool_labels.csv and dedup report, and R16c's instances_val.csv.
  2. Hashes   file-byte MD5 and decoded-pixel MD5 (the audit's fingerprint: cv2.imdecode to a BGR array,
              md5 of str(shape) + bytes) of every pool image, every socal_fire / woolsey_fire image and,
              when given, the xBD socal-fire originals (xView2 train) and the xView2 test images.
  3. Twins    before-images shared by socal_fire and woolsey_fire, the audit's near-copy rule re-run on the
              whole pool, and the twin statistics of val_train_leaks.csv recomputed.
  4. E5       Abrar's 21 pairs against the audit's 40, and why a hash of xBD tier 1 finds only 21.

Writes verify_val_leak.json, val_pairs_645.csv, leaks_40_vs_abrar_21.csv, socal_woolsey_pre_twins.csv.

Author: Ridita Afrin Riya (CSE499B Group 5)
Project: Resource-Efficient Neural Architectures for Post-Disaster Damage Assessment
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Worker processes re-import this module, so only cv2 + numpy load at the top
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]                 # scripts -> audit_outputs -> BRIGHT_DisasterM3_Integration -> Essential_task_delegation
sys.path.insert(0, str(HERE))          # bright_dm3_dedup.py sits next to this script
THUMB = 32
SCREEN_R = 0.95                        # the audit's thumbnail screen ...
MATCH_R = 0.95                         # ... and its 256x256 confirmation (bright_dm3_dedup.copy_score)
FIRE = ("socal_fire", "woolsey_fire")
FOLDER_TO_CLASS = {"train_building_intact_mask": 1, "train_building_damaged_mask": 2,
                   "train_building_destroyed_mask": 3}


# ----------------------------------------------------------------------------- workers
def fingerprint(path):
    """(file-byte md5, decoded-pixel md5, 32x32 thumbnail, shape), the last three exactly as
    dm3_bright_optical_eda.image_stats() computes them."""
    data = np.fromfile(str(path), np.uint8)
    file_md5 = hashlib.md5(data.tobytes()).hexdigest()
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return file_md5, None, np.zeros(THUMB * THUMB, np.float32), None
    t = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (THUMB, THUMB),
                   interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    thumb = np.zeros_like(t) if t.std() < 2 else (t - t.mean()) / t.std() / np.sqrt(t.size)
    return file_md5, hashlib.md5(str(img.shape).encode() + img.tobytes()).hexdigest(), thumb, img.shape


def merged_mask(mask_paths):
    """Cell 4's merge of the three binary building masks: 1 intact, 2 damaged, 3 destroyed,
    later classes overwrite earlier ones. None when no mask could be read."""
    combined_mask = None
    for mask_path, class_id in zip(mask_paths, FOLDER_TO_CLASS.values()):
        if mask_path is None:
            continue
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if combined_mask is None:
            combined_mask = np.zeros_like(m, dtype=np.uint8)
        combined_mask[m > 0] = class_id
    return combined_mask


def mask_counts(mask_paths):
    m = merged_mask(mask_paths)
    return None if m is None else (np.bincount(m.ravel(), minlength=4)[:4].tolist(), m.shape)


# ----------------------------------------------------------------------------- the notebook's split
def notebook_pool(data_root, workers):
    """Cells 4 and 5 of d4-champion-model-R11.ipynb with DATA_ROOT = data_root. The notebook writes each
    merged mask to /tmp; here only its pixel counts are kept. Returns the pairs in Cell 4's order, the
    train and validation pairs in Cell 5's order, and Cell 4's printed counts."""
    from sklearn.model_selection import train_test_split
    DATA_ROOT = data_root
    with open(DATA_ROOT / "train_release.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # ── Filter: ONLY Building Damage Assessment segmentation entries ──
    bda_entries = [
        e for e in raw_data
        if e.get("task") == "Referring Expression Segmentation"
        and e.get("cls_description") == "Building Damage Assessment"
    ]

    def resolve_path(rel_path):
        if not rel_path:
            return None
        rel_path = rel_path.replace("\\", "/")
        filename = Path(rel_path).name
        candidates = [
            DATA_ROOT / rel_path,
            DATA_ROOT / "train_images" / rel_path,
            DATA_ROOT / "train_images" / "train_images" / filename,
            DATA_ROOT / "masks" / rel_path,
            DATA_ROOT / "masks" / "masks" / filename,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    # ── Group entries by underlying post-disaster image ──
    by_image = defaultdict(dict)
    for e in bda_entries:
        post_img = e.get("post_image_path", "")
        if e.get("image_type") != "Optical":   # skip SAR duplicates, keep optical only
            continue
        mask_rel = e.get("ground_truth", "")
        folder = Path(mask_rel.replace("\\", "/")).parent.name
        resolved = resolve_path(mask_rel)
        if resolved:
            by_image[post_img][folder] = resolved
            by_image[post_img]["post_image_path"] = post_img
            by_image[post_img]["pre_image_path"] = e.get("pre_image_path")
    counts = {"entries": len(raw_data), "building_damage_entries": len(bda_entries),
              "images_with_a_mask": len(by_image)}
    del raw_data, bda_entries

    # ── Build combined (image, mask) pairs ── (masks merged in worker processes, order kept)
    candidates, skipped_img = [], 0
    for post_img, mask_dict in by_image.items():
        img_path = resolve_path(post_img)
        if not img_path:
            skipped_img += 1
            continue
        candidates.append({"image_path": img_path, "masks": [mask_dict.get(f) for f in FOLDER_TO_CLASS],
                           "pre_image_path": resolve_path(mask_dict.get("pre_image_path"))})
    with ProcessPoolExecutor(workers) as ex:
        merged = list(ex.map(mask_counts, [c["masks"] for c in candidates], chunksize=64))
    pairs = []
    for c, m in zip(candidates, merged):
        if m is None:
            continue  # no valid masks found for this image at all
        c["px"], c["mask_shape"] = m
        pairs.append(c)
    counts.update(pairs=len(pairs), skipped_image_not_found=skipped_img,
                  before_image_not_found=sum(p["pre_image_path"] is None for p in pairs))

    # ── Cell 5: same split as every previous experiment ──
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.1, random_state=42)
    return pairs, train_pairs, val_pairs, counts


# ----------------------------------------------------------------------------- helpers
def name_of(p):
    return Path(str(p).replace("\\", "/")).name


def prefix_of(name):
    return next((f for f in FIRE if name.startswith(f + "_")), None)


def tile_id(name):
    """'socal_fire_00000081_pre_disaster.png' -> '00000081'"""
    return name.split("_")[-3]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    from tqdm import tqdm
    from bright_dm3_dedup import build_optical_pool, copy_score

    ap = argparse.ArgumentParser(description="Re-check the audit's train/validation leak and compare it with E5")
    ap.add_argument("--dm3_root", default=str(REPO / "dataset" / "DisasterM3 Dataset"),
                    help="DisasterM3 root holding DisasterM3_Instruct (train_release.json, train_images, masks)")
    ap.add_argument("--audit_outputs", default=str(HERE.parent / "outputs"),
                    help="Folder with val_train_leaks.csv, dm3_pool_labels.csv, dm3_images.csv, bright_dm3_dedup_report.json")
    ap.add_argument("--e5_csv", default=str(REPO / "Essential_task_delegation" / "Champion_Task" / "E5-results" / "E5"
                                            / "leaked_pairs_local_check.csv"))
    ap.add_argument("--instances_val", default=str(REPO / "Essential_task_delegation" / "Champion_Task" / "R16c-results"
                                                   / "step2" / "instances_val.csv"))
    ap.add_argument("--xbd_train_images", default=None,
                    help="Optional: the xView2 train/images folder (xBD tier 1 originals, holds socal-fire)")
    ap.add_argument("--xbd_test_images", default=None, help="Optional: the xView2 test/images folder (test_pre_*.png)")
    ap.add_argument("--out", default=str(HERE.parent / "outputs" / "verification"))
    ap.add_argument("--workers", type=int, default=2, help="Decoding processes (2 fit in 8 GB RAM with room to spare)")
    args = ap.parse_args()
    instruct = Path(args.dm3_root) / "DisasterM3_Instruct"
    img_dir = instruct / "train_images" / "train_images"
    audit, out = Path(args.audit_outputs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R, clock = {}, time.time()

    print("=" * 70)
    print("  Train/validation leak: second check")
    print("=" * 70)

    # ---- 1. The split: notebook Cells 4-5 vs build_optical_pool() vs the files that used it ---------
    pairs, train, val, nb_counts = notebook_pool(instruct, args.workers)
    d_pairs, d_train, d_val, _ = build_optical_pool(instruct)
    post_name = [name_of(p["image_path"]) for p in pairs]
    val_posts = [name_of(p["image_path"]) for p in val]
    train_posts = [name_of(p["image_path"]) for p in train]
    split_of = {n: "train" for n in train_posts} | {n: "val" for n in val_posts}
    pair_of = {name_of(p["image_path"]): p for p in pairs}
    pre_name_of = {name_of(p["image_path"]): name_of(p["pre_image_path"]) for p in pairs}
    post_of_pre = {v: k for k, v in pre_name_of.items()}

    labels = {r["post"]: r for r in read_csv(audit / "dm3_pool_labels.csv")}
    report = json.loads((audit / "bright_dm3_dedup_report.json").read_text(encoding="utf-8"))
    sorted_val = sorted(val_posts)
    r16c_images = {r["image"] for r in read_csv(args.instances_val)}
    val_destroyed = {n: pair_of[n]["px"][3] for n in val_posts}
    R["split"] = {
        "notebook_cell4_counts": nb_counts,
        "notebook_cell4_printed_on_kaggle": {"entries": 92968, "building_damage_entries": 14531,
                                             "images_with_a_mask": 6443, "pairs": 6443, "skipped": 0,
                                             "before_image_not_found": 0},
        "train_val": [len(train), len(val)],
        "same_pairs_same_order_as_build_optical_pool": post_name == [p["post"].name for p in d_pairs],
        "same_before_images_as_build_optical_pool": [name_of(p["pre_image_path"]) for p in pairs]
                                                    == [p["pre"].name if p["pre"] else "None" for p in d_pairs],
        "same_val_in_same_order_as_build_optical_pool": val_posts == [p["post"].name for p in d_val],
        "same_train_in_same_order_as_build_optical_pool": train_posts == [p["post"].name for p in d_train],
        "val_equals_dedup_report_list": sorted_val == report["dm3_optical_pool"]["val_images"],
        "val_sha1": hashlib.sha1("\n".join(sorted_val).encode()).hexdigest(),
        "dedup_report_val_sha1": report["dm3_optical_pool"]["val_sha1"],
        "split_equals_dm3_pool_labels_csv": all(labels[n]["split"] == s for n, s in split_of.items())
                                            and len(labels) == len(split_of),
        "destroyed_px_equal_dm3_pool_labels_csv": all(float(labels[n]["px_destroyed"] or "nan") == pair_of[n]["px"][3]
                                                      for n in post_name),
        "r16c_instances_val_images": len(r16c_images),
        "r16c_images_all_in_val": r16c_images <= set(val_posts),
        "val_images_not_in_r16c": sorted(set(val_posts) - r16c_images),
        "mask_shapes": dict(Counter(f"{h}x{w}" for h, w in (p["mask_shape"] for p in val))),
        "val_destroyed_px": int(sum(val_destroyed.values())),
        "val_pairs_by_fire_event": {f: sum(n.startswith(f + "_") for n in val_posts) for f in FIRE},
        "val_destroyed_px_by_fire_event": {f: int(sum(v for n, v in val_destroyed.items() if n.startswith(f + "_")))
                                           for f in FIRE},
    }
    del d_pairs, d_train, d_val
    s = R["split"]
    print(f"[*] Pool {nb_counts['pairs']:,} -> {len(train):,} train / {len(val):,} val; same as build_optical_pool: "
          f"{s['same_pairs_same_order_as_build_optical_pool'] and s['same_val_in_same_order_as_build_optical_pool']}; "
          f"val destroyed px {s['val_destroyed_px']:,}; fire val {s['val_pairs_by_fire_event']}")

    # ---- 2. Hashes -------------------------------------------------------------------------------------
    paths = {img_dir / name_of(p[k]) for p in pairs for k in ("image_path", "pre_image_path")}
    paths |= {p for f in FIRE for p in img_dir.glob(f"{f}_*.png")}
    xbd_train = sorted(Path(args.xbd_train_images).glob("socal-fire_*_disaster.png")) if args.xbd_train_images else []
    xbd_test = sorted(Path(args.xbd_test_images).glob("test_*.png")) if args.xbd_test_images else []
    paths = sorted(paths) + xbd_train + xbd_test
    with ProcessPoolExecutor(args.workers) as ex:
        fps = list(tqdm(ex.map(fingerprint, paths, chunksize=32), total=len(paths), desc="Hash", mininterval=30))
    FP = {(p.parent, p.name): f for p, f in zip(paths, fps)}

    def fp(name, folder=img_dir):
        return FP[(folder, name)]

    dm3_names = sorted(n for d, n in FP if d == img_dir)
    dm3_set = set(dm3_names)
    csv_md5 = {r["name"]: r["md5"] for r in read_csv(audit / "dm3_images.csv")}
    R["hashes"] = {
        "images_hashed": {"dm3_instruct": len(dm3_names), "xbd_train_socal_fire": len(xbd_train), "xview2_test": len(xbd_test)},
        "what_the_audit_hashed": "decoded pixels: cv2.imdecode(file, IMREAD_COLOR) -> uint8 BGR array; "
                                 "md5(str(shape).encode() + array.tobytes()); files in DisasterM3_Instruct/train_images/train_images/",
        "pixel_md5_equal_to_dm3_images_csv": sum(fp(n)[1] == csv_md5.get(n) for n in dm3_names),
        "pixel_md5_differs_from_dm3_images_csv": sorted(n for n in dm3_names if fp(n)[1] != csv_md5.get(n)),
        "unreadable": sorted(n for n in dm3_names if fp(n)[1] is None),
    }
    print(f"[*] Hashed {len(paths):,} images; pixel md5 equal to dm3_images.csv for "
          f"{R['hashes']['pixel_md5_equal_to_dm3_images_csv']:,} of {len(dm3_names):,}")

    # ---- 3a. Before-images shared by socal_fire and woolsey_fire (exact, decoded pixels) ---------------
    def where(pre):
        post = post_of_pre.get(pre, pre.replace("_pre_disaster", "_post_disaster"))
        return split_of.get(post, "not in pool")

    fire_pre = [n for n in dm3_names if prefix_of(n) and n.endswith("_pre_disaster.png")]
    fire_all = [n for n in dm3_names if prefix_of(n)]

    def groups(names, key):
        g = defaultdict(list)
        for n in names:
            if fp(n)[key]:
                g[fp(n)[key]].append(n)
        return [sorted(v) for v in g.values() if {prefix_of(n) for n in v} == set(FIRE)]

    pre_groups = groups(fire_pre, 1)
    tier1 = {p.name for p in xbd_train}
    test_pre_md5 = {fp(p.name, p.parent)[1]: p.name for p in xbd_test if p.name.startswith("test_pre_")}
    test_post_md5 = {fp(p.name, p.parent)[1]: p.name for p in xbd_test if p.name.startswith("test_post_")}

    def xbd_source(dm3_name):
        """Where xBD keeps the original of a DisasterM3 socal_fire image (tier 1 = xView2 train)."""
        xname = dm3_name.replace("socal_fire_", "socal-fire_")
        if xname in tier1:
            same = fp(xname, Path(args.xbd_train_images))[1] == fp(dm3_name)[1]
            return "xView2 train (tier 1), pixel-identical" if same else "xView2 train (tier 1), pixels differ"
        lookup = test_pre_md5 if "_pre_" in dm3_name else test_post_md5
        if fp(dm3_name)[1] in lookup:
            return f"xView2 test ({lookup[fp(dm3_name)[1]]})"
        return "not in xView2 train or test" if (xbd_train or xbd_test) else "not checked"

    twin_rows = []
    for i, g in enumerate(sorted(pre_groups)):
        socal = [n for n in g if n.startswith("socal_fire_")]
        wool = [n for n in g if n.startswith("woolsey_fire_")]
        twin_rows.append({"group": i + 1, "pixel_md5": fp(g[0])[1], "socal_fire_pre": " ".join(socal),
                          "woolsey_fire_pre": " ".join(wool), "socal_side": " ".join(where(n) for n in socal),
                          "woolsey_side": " ".join(where(n) for n in wool),
                          "file_bytes_identical": len({fp(n)[0] for n in g}) == 1,
                          "socal_xbd_source": " | ".join(xbd_source(n) for n in socal)})
    in_pool = [r for r in twin_rows if "not in pool" not in r["socal_side"] + " " + r["woolsey_side"]]
    socal_pre = [n for n in fire_pre if n.startswith("socal_fire_")]
    R["twins"] = {
        "fire_images_on_disk": {"pre": len(fire_pre), "all": len(fire_all)},
        "fire_pairs_in_pool": {f: sum(n.startswith(f + "_") for n in post_name) for f in FIRE},
        "shared_pre_groups_all_instruct_files": len(pre_groups),
        "shared_groups_over_pre_and_post": len(groups(fire_all, 1)),
        "shared_pre_groups_by_file_bytes": len(groups(fire_pre, 0)),
        "group_sizes": dict(Counter(len(g) for g in pre_groups)),
        "groups_with_both_sides_in_pool": len(in_pool),
        "groups_with_a_side_outside_pool": len(twin_rows) - len(in_pool),
        "groups_by_split": dict(Counter(f"{r['socal_side']} / {r['woolsey_side']}" for r in twin_rows)),
        "groups_file_bytes_identical": sum(r["file_bytes_identical"] for r in twin_rows),
        "groups_by_socal_xbd_source": dict(Counter(r["socal_xbd_source"].split(" (")[0] if "test" in r["socal_xbd_source"]
                                                   else r["socal_xbd_source"] for r in twin_rows)),
        "dm3_socal_fire_pre_by_xbd_source": dict(Counter(xbd_source(n).split(" (")[0] if "test (" in xbd_source(n)
                                                         else xbd_source(n) for n in socal_pre)),
        "xbd_tier1_socal_fire_pre_missing_from_dm3": sorted(n for n in tier1 if n.endswith("_pre_disaster.png")
                                                            and n.replace("socal-fire_", "socal_fire_") not in dm3_set),
    }

    # every validation pair: is its before-image (or after-image) a pixel copy of one used elsewhere?
    by_md5 = defaultdict(list)
    for n in dm3_names:
        if fp(n)[1]:
            by_md5[fp(n)[1]].append(n)
    val_pair_rows, status_of = [], {}
    for k, v in enumerate(val_posts):
        pre = pre_name_of[v]
        others = [n for n in by_md5[fp(pre)[1]] if n != pre] if fp(pre)[1] else []
        where_twins = sorted({where(n) for n in others})
        status = ("twin in train" if "train" in where_twins else "twin in val" if "val" in where_twins
                  else "twin outside pool" if where_twins else "no twin")
        status_of[v] = status
        val_pair_rows.append({"val_order": k, "val_post": v, "val_pre": pre, "event": v.rsplit("_", 3)[0]
                              if v.endswith("_disaster.png") else v.rsplit("_post_", 1)[0],
                              "destroyed_px": val_destroyed[v], "building_px": sum(pair_of[v]["px"][1:]),
                              "pre_image_twin": " ".join(others), "twin_status": status,
                              "in_r16c_instances_val": v in r16c_images})
    exact_cross = {v for v in val_posts for img in (v, pre_name_of[v]) if fp(img)[1] and
                   any(split_of.get(post_of_pre.get(n, n)) == "train" for n in by_md5[fp(img)[1]] if n != img)}
    fire_val = [v for v in val_posts if prefix_of(v)]
    R["twins"]["val_fire_pairs_by_status"] = dict(Counter(status_of[v] for v in fire_val))
    # only pool images of the other events are hashed, so their twins are looked for inside the pool
    R["twins"]["val_pairs_other_events_with_a_twin_in_pool"] = sorted(v for v in val_posts if not prefix_of(v)
                                                                      and status_of[v] != "no twin")
    R["twins"]["val_pairs_with_an_exact_copy_in_train_any_event"] = len(exact_cross)

    # ---- 3b. The audit's own near-copy rule, re-run on every pool image ---------------------------------
    pool_imgs = sorted({n for v in post_name for n in (v, pre_name_of[v])})
    side = {n: split_of[post_of_pre.get(n, n)] for n in pool_imgs}
    val_imgs = [n for n in pool_imgs if side[n] == "val"]
    train_imgs = [n for n in pool_imgs if side[n] == "train"]
    sim = np.stack([fp(n)[2] for n in val_imgs]) @ np.stack([fp(n)[2] for n in train_imgs]).T
    cross, rejected = [], 0
    for i, j in zip(*np.nonzero(sim >= SCREEN_R)):
        a, b = val_imgs[i], train_imgs[j]
        same = fp(a)[1] is not None and fp(a)[1] == fp(b)[1]
        r = 1.0 if same else copy_score(img_dir / a, img_dir / b)
        if r >= MATCH_R:
            cross.append((a, b, r, same))
        else:
            rejected += 1
    del sim
    twins = {}
    for a, b, _, _ in cross:
        twins.setdefault(post_of_pre.get(a, a), post_of_pre.get(b, b))
    leak_csv = {r["val_post"]: r for r in read_csv(audit / "val_train_leaks.csv")}

    def twin_stats(v, t):
        mv, mt = merged_mask(pair_of[v]["masks"]), merged_mask(pair_of[t]["masks"])
        both = (mv > 0) & (mt > 0)
        return {"post_r": round(copy_score(img_dir / v, img_dir / t), 3),
                "building_iou": round(float(both.sum() / max(((mv > 0) | (mt > 0)).sum(), 1)), 3),
                "same_damage_class": round(float((mv[both] == mt[both]).mean()), 3) if both.any() else None,
                "val_destroyed_px": int((mv == 3).sum())}

    recomputed = {v: twin_stats(v, t) for v, t in twins.items()}

    def same_stats(v):
        c, r = leak_csv[v], recomputed[v]
        return (c["train_post"] == twins[v] and int(c["val_destroyed_px"]) == r["val_destroyed_px"]
                and all((c[k] == "" and r[k] is None) or (r[k] is not None and abs(float(c[k]) - r[k]) < 0.0015)
                        for k in ("post_r", "building_iou", "same_damage_class")))

    leak40 = set(leak_csv)
    R["audit_rule_rerun"] = {
        "val_images": len(val_imgs), "train_images": len(train_imgs),
        "candidate_pairs_confirmed": len(cross), "candidate_pairs_rejected_by_copy_score": rejected,
        "confirmed_by_identical_pixels": sum(c[3] for c in cross),
        "confirmed_matches_by_role": dict(Counter(f"{'pre' if '_pre_' in a else 'post'}-{'pre' if '_pre_' in b else 'post'}"
                                                  for a, b, _, _ in cross)),
        "leaked_val_pairs": len(twins),
        "same_set_as_val_train_leaks_csv": set(twins) == leak40,
        "same_twin_and_stats_as_csv": sum(same_stats(v) for v in twins if v in leak_csv),
        "same_set_as_exact_pre_twins_in_train": set(twins) == {v for v in val_posts if status_of[v] == "twin in train"},
        "same_set_as_exact_copies_any_image": set(twins) == exact_cross,
        "leaked_destroyed_px": int(sum(r["val_destroyed_px"] for r in recomputed.values())),
        "leaked_destroyed_px_csv": int(sum(int(r["val_destroyed_px"]) for r in leak_csv.values())),
        "clean_val_pairs": len(val_posts) - len(twins),
        "clean_val_destroyed_px": int(sum(val_destroyed.values()) - sum(r["val_destroyed_px"] for r in recomputed.values())),
        "twin_medians": {k: float(np.median([r[k] for r in recomputed.values() if r[k] is not None]))
                         for k in ("post_r", "building_iou", "same_damage_class")},
    }
    a = R["audit_rule_rerun"]
    print(f"[*] Audit rule re-run: {a['leaked_val_pairs']} leaked val pairs, same as val_train_leaks.csv: "
          f"{a['same_set_as_val_train_leaks_csv']}; destroyed px {a['leaked_destroyed_px']:,} of {s['val_destroyed_px']:,}")

    # ---- 4. Abrar's E5 check: 21 of the 40, and why ---------------------------------------------------
    e5 = {r["validation_pair"]: r for r in read_csv(args.e5_csv)}
    leak_rows = []
    for v in sorted(leak40 | set(e5)):
        t = leak_csv[v]["train_post"] if v in leak_csv else e5[v]["twin_in_training"]
        socal = v if v.startswith("socal_fire_") else t
        socal_pre, other_pre = pre_name_of[socal], pre_name_of[t if socal == v else v]
        src = xbd_source(socal_pre)
        if v in e5 and v in leak_csv:
            reason = "found by both checks"
        elif v in leak_csv and src == "not checked":
            reason = "xBD originals not checked (pass --xbd_train_images / --xbd_test_images)"
        elif v in leak_csv and "tier 1" not in src:
            reason = (f"the socal_fire side ({socal_pre}) is not an xBD tier-1 image ({src}); a hash of tier-1 "
                      "socal-fire and tier-3 woolsey-fire before-images cannot contain it")
        elif v in leak_csv:
            reason = "socal_fire side is in tier 1: xBD's originals must differ where DisasterM3's copies are identical"
        else:
            reason = "found only by E5"
        leak_rows.append({
            "val_post": v, "train_twin": t, "val_destroyed_px": int(leak_csv[v]["val_destroyed_px"]) if v in leak_csv
            else int(e5[v]["destroyed_px_in_validation_pair"]),
            "in_audit_40": v in leak_csv, "in_e5_21": v in e5,
            "e5_twin_and_px_agree": (e5[v]["twin_in_training"] == t and int(e5[v]["destroyed_px_in_validation_pair"])
                                     == val_destroyed[v]) if v in e5 else "",
            "socal_side_pre": socal_pre, "socal_side_xbd_source": src,
            "twin_pre_pixels_identical": fp(socal_pre)[1] == fp(other_pre)[1],
            "twin_pre_file_bytes_identical": fp(socal_pre)[0] == fp(other_pre)[0],
            "val_in_r16c_instances_val": v in r16c_images, "reason": reason})

    # Abrar's check re-enacted on DisasterM3's files: a twin counts only when its socal side is in tier 1
    def e5_status(v):
        pre = pre_name_of[v]
        twins_t1 = [n for n in by_md5[fp(pre)[1]] if n != pre and
                    (n if n.startswith("socal_fire_") else pre).replace("socal_fire_", "socal-fire_") in tier1]
        w = {where(n) for n in twins_t1}
        return "twin in train" if "train" in w else "twin in val" if "val" in w else "twin outside pool" if w else "no twin"

    e5_list = [v for v in fire_val if v in r16c_images]
    R["e5"] = {
        "e5_pairs": len(e5), "e5_pairs_in_audit_40": len(set(e5) & leak40),
        "e5_twins_and_px_agree": sum(r["e5_twin_and_px_agree"] is True for r in leak_rows),
        "e5_destroyed_px": int(sum(int(r["destroyed_px_in_validation_pair"]) for r in e5.values())),
        "audit_only_pairs": len(leak40 - set(e5)),
        "audit_only_destroyed_px": int(sum(int(leak_csv[v]["val_destroyed_px"]) for v in leak40 - set(e5))),
        "reasons": dict(Counter(r["reason"].split(" (")[0] if r["reason"].startswith("the socal") else r["reason"]
                                for r in leak_rows)),
        "fire_val_pairs_in_r16c_list": len(e5_list),
        "fire_val_pair_missing_from_r16c_list": sorted(set(fire_val) - r16c_images),
        "e5_reenacted_on_dm3_files": dict(Counter(e5_status(v) for v in e5_list)),
        "e5_reported": {"twin in train": 21, "twin in val": 4, "twin not in DisasterM3": 1, "no twin": 59},
        "audit_twin_status_same_85": dict(Counter(status_of[v] for v in e5_list)),
    }
    for row in val_pair_rows:
        row["audit_leak"] = row["val_post"] in leak40
        row["train_twin"] = leak_csv[row["val_post"]]["train_post"] if row["audit_leak"] else ""
        row["e5_leak"] = row["val_post"] in e5
        row["clean"] = not row["audit_leak"]
    R["seconds"] = round(time.time() - clock)

    # ---- 5. Write ------------------------------------------------------------------------------------------
    write_csv(out / "val_pairs_645.csv", val_pair_rows,
              ["val_order", "val_post", "val_pre", "event", "destroyed_px", "building_px", "audit_leak", "train_twin",
               "e5_leak", "clean", "twin_status", "pre_image_twin", "in_r16c_instances_val"])
    write_csv(out / "leaks_40_vs_abrar_21.csv", leak_rows, list(leak_rows[0]))
    write_csv(out / "socal_woolsey_pre_twins.csv", twin_rows, list(twin_rows[0]))
    (out / "verify_val_leak.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print(f"[*] E5: {R['e5']['e5_pairs_in_audit_40']} of its {len(e5)} pairs are in the audit's 40; reasons {R['e5']['reasons']}")
    print(f"[OK] Written to {out}  ({R['seconds']} s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
