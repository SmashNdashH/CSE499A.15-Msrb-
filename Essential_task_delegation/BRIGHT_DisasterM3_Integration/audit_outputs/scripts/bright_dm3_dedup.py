"""
bright_dm3_dedup.py
===================
Leak check for the standardized BRIGHT optical tiles, run before they join DisasterM3 training.

1. Rebuilds the DisasterM3 optical segmentation pool and its 90/10 split exactly as the
   champion notebooks do (train_test_split, random_state=42): 6,443 -> 5,798 train / 645 val.
2. Compares every BRIGHT tile with every DisasterM3 file, by tile name and by pixel content
   (identical pixels, or the same footprint after resampling):
     - a copy in the optical pool (train or frozen val)  -> the tile is excluded
     - a copy anywhere else in DisasterM3 (SAR pairs, ORR box images, Instruct/Bench masks)
                                                           -> recorded; excluded only with --strict
3. Writes the dedup report and rewrites bright_optical_manifest.json: "tiles" keeps the tiles
   that passed, the rest move to "excluded_tiles". Safe to re-run with either policy.

Re-tiled or re-acquired imagery of the same place is not caught by the tile check. The event
table in the report shows which BRIGHT events exist in DisasterM3 at all, and where.

Author: Ridita Afrin Riya (CSE499B Group 5)
Project: Resource-Efficient Neural Architectures for Post-Disaster Damage Assessment
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
BUILDING_MASK_FOLDERS = {"train_building_intact_mask", "train_building_damaged_mask",
                         "train_building_destroyed_mask"}
EXPECTED_SPLIT = (5798, 645)   # train / val sizes of every champion run (E-protocol)
THUMB = 32                     # side of the screening thumbnail
SCREEN_R = 0.95                # thumbnail correlation that triggers a full comparison
MATCH_R = 0.95                 # 256x256 correlation (non-black pixels) that confirms a copy

# BRIGHT event -> DisasterM3 file-name prefixes. DisasterM3 renamed two events
# ("Spain volcano" and "Rwanda volcano" in its Fig. 2) and misspelled two more.
DM3_EVENT_NAMES = {
    "bata-explosion": ["bata_explosion"],
    "beirut-explosion": ["beriut_explosion", "beirut_explosion"],
    "congo-volcano": ["rwanda_volcano"],
    "haiti-earthquake": ["haiti_earthquake"],
    "hawaii-wildfire": ["hawaii_wildfire"],
    "la_palma-volcano": ["la_palma_volcano", "spain_volcano"],
    "libya-flood": ["libya_flood"],
    "marshall-wildfire": ["marshall_wildfire"],
    "mexico-hurricane": ["mexico_hurricane"],
    "morocco-earthquake": ["morocco_earthquake", "morroco_earthquake"],
    "myanmar-hurricane": ["myanmar_hurricane"],
    "noto-earthquake": ["noto_earthquake"],
    "turkey-earthquake": ["turkey_earthquake"],
    "ukraine-conflict": ["ukraine_conflict"],
}


def tile_key(name):
    """'la_palma-volcano_00000001_pre_disaster.tif' and 'la_palma_volcano_00000001.png'
    both become 'la_palma_volcano_00000001'."""
    stem = re.sub(r"\.(png|jpe?g|tiff?)$", "", Path(name).name.lower())
    stem = re.sub(r"_(pre_disaster|post_disaster|building_damage|optical_pre|building_mask)$", "", stem)
    return stem.replace("-", "_")


def dm3_event(name):
    name = Path(name).name.lower()
    for event, prefixes in DM3_EVENT_NAMES.items():
        if name.startswith(tuple(prefixes)):
            return event
    return None


def build_optical_pool(instruct_root):
    """The segmentation pool of Champion_Task/d4-champion-model-*.ipynb (Cells 4-5) and
    train_champion_bright_disasterm3.ipynb (Cell 4): optical Building Damage Assessment entries,
    grouped by post-disaster image in manifest order, split 90/10 with random_state=42.
    Also returns the file names of every image used by a SAR-typed entry."""
    # Imported here, not at the top: worker processes re-import this module and need only cv2/numpy
    from sklearn.model_selection import train_test_split
    entries = json.loads((instruct_root / "train_release.json").read_text(encoding="utf-8"))

    def resolve(rel):
        if not rel:
            return None
        rel = rel.replace("\\", "/")
        name = Path(rel).name
        for cand in (instruct_root / rel, instruct_root / "train_images" / rel,
                     instruct_root / "train_images" / "train_images" / name,
                     instruct_root / "masks" / rel, instruct_root / "masks" / "masks" / name):
            if cand.exists():
                return cand
        return None

    sar_names = set()
    by_image, pre_of = defaultdict(set), {}
    for e in entries:
        if "SAR" in (e.get("image_type"), e.get("post_image_type")):
            sar_names.update(Path(p.replace("\\", "/")).name.lower()
                             for p in (e.get("pre_image_path"), e.get("post_image_path")) if p)
        if (e.get("task") == "Referring Expression Segmentation"
                and e.get("cls_description") == "Building Damage Assessment"
                and e.get("image_type") == "Optical"):
            mask_rel = e.get("ground_truth", "")
            if resolve(mask_rel):
                post = e.get("post_image_path", "")
                by_image[post].add(Path(mask_rel.replace("\\", "/")).parent.name)
                pre_of[post] = e.get("pre_image_path", "")

    pairs = []
    for post, folders in by_image.items():
        post_path = resolve(post)
        if post_path and folders & BUILDING_MASK_FOLDERS:
            pairs.append({"post": post_path, "pre": resolve(pre_of[post])})
    train, val = train_test_split(pairs, test_size=0.1, random_state=42)
    return pairs, train, val, sar_names


def read_image(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def fingerprint(path):
    """(md5 of the decoded pixels, unit-norm z-scored gray thumbnail). Blank images get
    (None, zeros): their content cannot tell one place from another."""
    img = read_image(path)
    blank = (None, np.zeros(THUMB * THUMB, np.float32))
    if img is None:
        return blank
    thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (THUMB, THUMB),
                       interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    if thumb.std() < 2.0:
        return blank
    z = (thumb - thumb.mean()) / thumb.std()
    return hashlib.md5(str(img.shape).encode() + img.tobytes()).hexdigest(), z / np.sqrt(z.size)


def copy_score(path_a, path_b):
    """Correlation of two images at 256x256, over pixels that are not black (no-data) in either."""
    a, b = (cv2.resize(read_image(p, cv2.IMREAD_GRAYSCALE), (256, 256),
                       interpolation=cv2.INTER_AREA).astype(np.float32) for p in (path_a, path_b))
    valid = (a > 0) & (b > 0)
    if valid.mean() < 0.25:
        return 0.0
    a, b = a[valid] - a[valid].mean(), b[valid] - b[valid].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def pixel_copies(b_fp, d_fp, b_paths, d_paths):
    """{bright index: {dm3 index: evidence}} for identical images or the same footprint resampled."""
    by_md5 = defaultdict(list)
    for j, (md5, _) in enumerate(d_fp):
        if md5:
            by_md5[md5].append(j)
    hits = defaultdict(dict)
    for i, (md5, _) in enumerate(b_fp):
        for j in by_md5.get(md5, []) if md5 else []:
            hits[i][j] = "identical pixels"
    B = np.stack([v for _, v in b_fp])
    D = np.stack([v for _, v in d_fp])
    for start in range(0, len(D), 4096):
        for i, j in zip(*np.nonzero(B @ D[start:start + 4096].T >= SCREEN_R)):
            i, j = int(i), int(j) + start
            if j not in hits[i]:
                r = copy_score(b_paths[i], d_paths[j])
                if r >= MATCH_R:
                    hits[i][j] = f"resampled copy (r={r:.3f})"
    return hits


def mask_inside(mask_path, footprint):
    """Share of a DisasterM3 building mask that lies inside the BRIGHT footprint (None if empty)."""
    m = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None or not m.any():
        return None
    if m.shape != footprint.shape:
        m = cv2.resize(m, footprint.shape[::-1], interpolation=cv2.INTER_NEAREST)
    return float((footprint[m > 0] > 0).mean())


def main():
    parser = argparse.ArgumentParser(description="Check BRIGHT optical tiles against DisasterM3 before training")
    parser.add_argument("--bright_manifest", type=str,
                        default=r"d:\CSE499A.15-Msrb-\data\bright_standardized_optical\bright_optical_manifest.json",
                        help="Manifest from bright_disasterm3_standardizer.py (rewritten in place)")
    parser.add_argument("--dm3_root", type=str,
                        default=r"d:\CSE499A.15-Msrb-\dataset\DisasterM3 Dataset",
                        help="DisasterM3 root holding DisasterM3_Instruct and DisasterM3_Bench")
    parser.add_argument("--report", type=str,
                        default=r"d:\CSE499A.15-Msrb-\data\bright_dm3_dedup_report.json",
                        help="Where to write the per-tile dedup report")
    parser.add_argument("--strict", action="store_true",
                        help="Also exclude tiles that exist anywhere in DisasterM3, not only in the optical pool")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Processes used to decode images")
    args = parser.parse_args()

    manifest_path = Path(args.bright_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    # A re-run sees every tile, in the standardizer's order (sorted file names)
    tiles = sorted(manifest.get("tiles", []) + manifest.get("excluded_tiles", []), key=lambda t: t["tile_id"])
    dm3_root = Path(args.dm3_root)
    policy = "strict" if args.strict else "pool"

    print("=" * 70)
    print("  BRIGHT -> DisasterM3 Dedup Check")
    print(f"  BRIGHT tiles : {len(tiles)}  ({manifest_path})")
    print(f"  DisasterM3   : {dm3_root}")
    print(f"  Policy       : {policy}")
    print("=" * 70)

    # 1. The optical pool the champion notebooks train and validate on
    pairs, train, val, sar_names = build_optical_pool(dm3_root / "DisasterM3_Instruct")
    pool_group = {p[k].relative_to(dm3_root).as_posix().lower(): f"pool_{split}"
                  for split, part in (("train", train), ("val", val)) for p in part for k in ("post", "pre") if p[k]}
    val_names = sorted(p["post"].name for p in val)
    val_sha1 = hashlib.sha1("\n".join(val_names).encode()).hexdigest()
    print(f"[*] Optical pool: {len(pairs):,} images -> {len(train):,} train / {len(val):,} frozen val")
    if (len(train), len(val)) != EXPECTED_SPLIT:
        print(f"[!] WARNING: champion runs used {EXPECTED_SPLIT[0]:,} / {EXPECTED_SPLIT[1]:,}. This DisasterM3 "
              "copy gives a different split, so its val set is not the frozen one.")

    # 2. Every DisasterM3 file, labelled by where it sits
    dm3_images, dm3_masks = [], []
    for path in sorted(dm3_root.rglob("*")):
        rel = path.relative_to(dm3_root).as_posix()
        if rel.startswith(".cache/") or path.suffix.lower() not in IMG_EXT or not path.is_file():
            continue
        parts = rel.split("/")
        top = "bench" if parts[0] == "DisasterM3_Bench" else "instruct"
        if "masks" in parts:
            dm3_masks.append((rel, f"{top}_mask"))
        elif top == "bench":
            dm3_images.append((rel, "bench_image"))
        elif "box_train_images" in parts:
            dm3_images.append((rel, "instruct_box_image"))
        else:
            dm3_images.append((rel, pool_group.get(rel.lower()) or
                               ("instruct_sar_pair" if path.name.lower() in sar_names else "instruct_other_image")))
    by_name = defaultdict(list)
    for rel, group in dm3_images + dm3_masks:
        by_name[tile_key(rel)].append((rel, group))
    print(f"[*] DisasterM3 files: {len(dm3_images):,} images, {len(dm3_masks):,} masks")

    # 3. Pixel fingerprints of every BRIGHT tile and every DisasterM3 image
    b_paths = [base / t["image"] for t in tiles]
    d_paths = [dm3_root / rel for rel, _ in dm3_images]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        b_fp = list(tqdm(executor.map(fingerprint, b_paths, chunksize=32),
                         total=len(b_paths), desc="Fingerprint BRIGHT"))
        d_fp = list(tqdm(executor.map(fingerprint, d_paths, chunksize=64),
                         total=len(d_paths), desc="Fingerprint DisasterM3"))
    hits = pixel_copies(b_fp, d_fp, b_paths, d_paths)

    # 4. Per-tile decision
    records, kept, excluded = [], [], []
    summary = defaultdict(Counter)
    for i, t in enumerate(tqdm(tiles, desc="Decide")):
        matches, footprint = {}, None
        for rel, group in by_name.get(tile_key(t["tile_id"]), []):
            evidence = "same tile name"
            if group.endswith("_mask") and "building" in rel:
                if footprint is None:
                    footprint = read_image(base / t["mask"], cv2.IMREAD_GRAYSCALE)
                share = mask_inside(dm3_root / rel, footprint)
                if share is not None:
                    evidence += f", {share:.0%} of its buildings inside the BRIGHT footprint"
            matches[rel] = {"file": rel, "group": group, "evidence": evidence}
        for j, evidence in hits.get(i, {}).items():
            rel, group = dm3_images[j]
            if rel in matches:
                matches[rel]["evidence"] += ", " + evidence
            else:
                matches[rel] = {"file": rel, "group": group, "evidence": evidence}

        groups = sorted({m["group"] for m in matches.values()})
        in_pool = [g for g in groups if g.startswith("pool_")]
        if in_pool:
            reason = "copy of a DisasterM3 optical-pool image (" + ", ".join(in_pool) + ")"
        elif args.strict and groups:
            reason = "copy exists elsewhere in DisasterM3 (" + ", ".join(groups) + ")"
        else:
            reason = None

        s = summary[t["tile_id"].rsplit("_", 1)[0]]
        s["tiles"] += 1
        s["copies_in_pool"] += bool(in_pool)
        s["copies_in_dm3"] += bool(groups)
        s["kept"] += reason is None
        s["kept_if_strict"] += not groups

        entry = {k: v for k, v in t.items() if k not in ("dm3_copies", "excluded_reason")}
        entry["dm3_copies"] = groups
        if reason:
            entry["excluded_reason"] = reason
            excluded.append(entry)
        else:
            kept.append(entry)
        records.append({"tile_id": t["tile_id"], "decision": "exclude" if reason else "keep",
                        "reason": reason, "matches": sorted(matches.values(), key=lambda m: m["file"])})

    # 5. Which BRIGHT events exist in DisasterM3 at all (catches renamed and re-tiled events)
    our_events = {t["tile_id"].rsplit("_", 1)[0] for t in tiles}
    presence = {ev: Counter() for ev in DM3_EVENT_NAMES}
    for split, part in (("pool_train", train), ("pool_val", val)):
        for p in part:
            ev = dm3_event(p["post"].name)
            if ev:
                presence[ev][split] += 1
    for rel, group in dm3_images:
        ev = dm3_event(rel)
        if ev and not group.startswith("pool_"):
            presence[ev]["other_images"] += 1
    for rel, group in dm3_masks:
        ev = dm3_event(rel)
        if ev and group == "bench_mask":
            presence[ev]["bench_masks"] += 1

    rule = ("exclude tiles with any copy in DisasterM3 (optical pool, SAR pairs, ORR images, masks)"
            if args.strict else "exclude tiles that copy a DisasterM3 optical-pool image (train or frozen val)")
    report_path = Path(args.report)
    report = {
        "generated": date.today().isoformat(),
        "policy": policy,
        "rule": rule,
        "bright_manifest": str(manifest_path),
        "dm3_root": str(dm3_root),
        "method": ("Names: BRIGHT tile id vs DisasterM3 file name ('-' -> '_'). Pixels: md5 of the decoded "
                   f"image, plus {THUMB}x{THUMB} thumbnail screen (r >= {SCREEN_R}) confirmed at 256x256 on "
                   f"non-black pixels (r >= {MATCH_R}). Name-matched building masks are checked against the "
                   "BRIGHT footprint. Re-tiled imagery of the same place is only covered by bright_events_in_dm3."),
        "dm3_optical_pool": {"images": len(pairs), "train": len(train), "val": len(val),
                             "val_sha1": val_sha1, "val_images": val_names},
        "bright_events_in_dm3": {ev: {**{k: c[k] for k in ("pool_train", "pool_val", "other_images", "bench_masks")},
                                      "in_our_subset": ev in our_events}
                                 for ev, c in presence.items()},
        "summary_by_event": {ev: dict(c) for ev, c in sorted(summary.items())},
        "tiles": records,
    }
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    pixel_counts = Counter()
    for t in kept:
        for k, v in t["pixel_stats"].items():
            pixel_counts[str(k)] += v
    new_manifest = {
        "total_optical_tiles": len(kept),
        "standardized_tiles": len(tiles),
        "modality": manifest.get("modality"),
        "class_definition": manifest.get("class_definition"),
        "class_pixel_counts": dict(sorted(pixel_counts.items())),
        "dedup": {
            "checked_on": report["generated"],
            "policy": policy,
            "rule": rule,
            "report": str(report_path),
            "dm3_pool": {"images": len(pairs), "train": len(train), "val": len(val), "val_sha1": val_sha1},
            "tiles_kept": len(kept),
            "tiles_excluded": len(excluded),
        },
        "tiles": kept,
        "excluded_tiles": excluded,
    }
    manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("  BRIGHT EVENTS INSIDE DISASTERM3  (pool = optical segmentation train/val)")
    print("=" * 70)
    print(f"  {'BRIGHT event':<20}{'pool train':>11}{'pool val':>10}{'other imgs':>12}{'bench masks':>13}")
    for ev, c in presence.items():
        flag = "   <- our subset" if ev in our_events else ""
        print(f"  {ev:<20}{c['pool_train']:>11}{c['pool_val']:>10}{c['other_images']:>12}{c['bench_masks']:>13}{flag}")

    print("\n" + "=" * 70)
    print("  PER-TILE RESULT FOR OUR BRIGHT TILES")
    print("=" * 70)
    print(f"  {'Event':<20}{'tiles':>7}{'copy in pool':>14}{'copy in DM3':>13}{'kept':>7}{'kept if --strict':>18}")
    total = Counter()
    for ev, c in sorted(summary.items()):
        total.update(c)
        print(f"  {ev:<20}{c['tiles']:>7}{c['copies_in_pool']:>14}{c['copies_in_dm3']:>13}"
              f"{c['kept']:>7}{c['kept_if_strict']:>18}")
    print(f"  {'TOTAL':<20}{total['tiles']:>7}{total['copies_in_pool']:>14}{total['copies_in_dm3']:>13}"
          f"{total['kept']:>7}{total['kept_if_strict']:>18}")
    print("\n" + "=" * 70)
    print(f"  Policy {policy}: kept {len(kept)}, excluded {len(excluded)}")
    print(f"  Frozen val sha1 : {val_sha1}")
    print(f"  Report          : {report_path}")
    print(f"  Manifest        : {manifest_path}")
    print("  Next step       : python create_combined_training_manifest.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
