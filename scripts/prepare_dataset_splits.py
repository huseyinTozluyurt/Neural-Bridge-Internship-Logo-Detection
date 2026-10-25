#!/usr/bin/env python3
"""Create leakage-aware YOLO train/val/test splits in a new output dataset.

Images with identical SHA-256 content are kept together. Source files are never
modified; the output contains copies, a split manifest, and a ready data.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="YOLO dataset data.yaml")
    ap.add_argument("--output", required=True, type=Path, help="New dataset output directory")
    ap.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="Hard-link images instead of copying (same filesystem only)")
    ap.add_argument("--preserve-test", action="store_true", help="Keep current test images fixed; repartition only existing train/val images")
    args = ap.parse_args()
    data = args.data.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not data.is_file(): ap.error(f"data.yaml not found: {data}")
    if out.exists() and any(out.iterdir()): ap.error(f"Output must be absent or empty: {out}")
    ratios = list(args.ratios)
    if any(v < 0 for v in ratios) or sum(ratios) <= 0: ap.error("ratios must be non-negative and have positive sum")
    ratios = [v / sum(ratios) for v in ratios]
    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict): ap.error("Dataset YAML must contain a mapping")
    root = Path(cfg.get("path", data.parent))
    if not root.is_absolute(): root = (data.parent / root).resolve()
    names = cfg.get("names", [])
    names = [str(names[k]) for k in sorted(names, key=lambda x: int(x))] if isinstance(names, dict) else list(names)
    images = {}
    for split in ("train", "val", "test"):
        if split not in cfg: continue
        roots = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
        for value in roots:
            d = Path(value); d = d if d.is_absolute() else root / d
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES: images[p.resolve()] = split
    if not images: ap.error("No images found in configured train/val/test splits")
    records = []
    for image in sorted(images):
        image = Path(image)
        images_root = next((parent for parent in image.parents if parent.name.lower() == "images"), None)
        if images_root:
            label = images_root.parent / "labels" / image.relative_to(images_root).with_suffix(".txt")
        else:
            label = image.parent / "labels" / image.with_suffix(".txt").name
        classes = set()
        if label.is_file():
            for row in label.read_text(encoding="utf-8", errors="replace").splitlines():
                try: classes.add(int(row.split()[0]))
                except (ValueError, IndexError): pass
        records.append({"path": image, "hash": digest(image), "classes": classes, "label": label, "source_split": images[image]})
    groups = {}
    for rec in records: groups.setdefault(rec["hash"], []).append(rec)
    if any(len(g) > 1 for g in groups.values()):
        print(f"Grouped {sum(len(g)-1 for g in groups.values() if len(g)>1)} byte-identical duplicate(s) to prevent split leakage.")
    rng = random.Random(args.seed)

    def stratified_groups(group_list, split_ratios):
        group_list = list(group_list)
        rng.shuffle(group_list)
        required = [i for i, ratio in enumerate(split_ratios) if ratio > 0]
        if len(group_list) < len(required):
            ap.error(f"Need at least {len(required)} unique groups for the requested non-zero splits; found {len(group_list)}")
        totals = Counter(c for group in group_list for rec in group for c in rec["classes"])
        targets = [{c: n * ratio for c, n in totals.items()} for ratio in split_ratios]
        assigned = [Counter() for _ in split_ratios]
        sizes = [0] * len(split_ratios)
        desired = [sum(len(g) for g in group_list) * ratio for ratio in split_ratios]
        group_list.sort(key=lambda g: (-len(g), -len(set().union(*(r["classes"] for r in g)))))
        result = [[] for _ in split_ratios]
        for idx in required:
            result[idx].append(group_list.pop())
            sizes[idx] += len(result[idx][-1])
            assigned[idx].update(c for rec in result[idx][-1] for c in rec["classes"])
        for group in group_list:
            counts = Counter(c for rec in group for c in rec["classes"])
            def score(i):
                class_gap = sum(max(0, targets[i][c] - assigned[i][c]) / max(1, totals[c]) for c in counts)
                size_gap = max(0, desired[i] - sizes[i]) / max(1, desired[i])
                return class_gap + size_gap
            idx = max(range(len(split_ratios)), key=score)
            result[idx].append(group); sizes[idx] += len(group); assigned[idx].update(counts)
        return result

    if args.preserve_test:
        trainval_records = [rec for rec in records if rec["source_split"] in ("train", "val")]
        test_records = [rec for rec in records if rec["source_split"] == "test"]
        test_hashes = {rec["hash"] for rec in test_records}
        leakage = [rec["path"] for rec in trainval_records if rec["hash"] in test_hashes]
        if leakage:
            ap.error(f"Found {len(leakage)} train/val image(s) byte-identical to held-out test images; resolve leakage before splitting")
        trainval_ratio_sum = ratios[0] + ratios[1]
        if trainval_ratio_sum <= 0: ap.error("--preserve-test requires non-zero train and val ratios")
        trainval_groups = {}
        for rec in trainval_records: trainval_groups.setdefault(rec["hash"], []).append(rec)
        fixed_test_groups = {}
        for rec in test_records: fixed_test_groups.setdefault(rec["hash"], []).append(rec)
        train_groups, val_groups = stratified_groups(list(trainval_groups.values()), [ratios[0]/trainval_ratio_sum, ratios[1]/trainval_ratio_sum])
        split_groups = [train_groups, val_groups, list(fixed_test_groups.values())]
    else:
        split_groups = stratified_groups(list(groups.values()), ratios)
    out.mkdir(parents=True, exist_ok=True)
    split_names = ("train", "val", "test")
    manifest = []
    for split, groups_for_split in zip(split_names, split_groups):
        for group in groups_for_split:
            for rec in group:
                image = rec["path"]
                rel = Path(image.name)
                # Prefix identical basenames from separate folders with a stable path hash.
                rel = Path(hashlib.sha1(str(image).encode()).hexdigest()[:10] + "_" + image.name)
                dst_img = out / split / "images" / rel
                dst_lbl = out / split / "labels" / rel.with_suffix(".txt")
                dst_img.parent.mkdir(parents=True, exist_ok=True); dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                if args.link:
                    try: dst_img.hardlink_to(image)
                    except OSError as exc: ap.error(f"Could not hard-link {image}: {exc}; rerun without --link")
                else: shutil.copy2(image, dst_img)
                if rec["label"].is_file(): shutil.copy2(rec["label"], dst_lbl)
                else: dst_lbl.write_text("", encoding="utf-8")
                manifest.append({"source_image": str(image), "source_label": str(rec["label"]), "source_split": rec["source_split"], "split": split, "sha256": rec["hash"], "classes": sorted(rec["classes"])})
    result_yaml = {"path": ".", "train": "train/images", "val": "val/images", "test": "test/images", "nc": len(names), "names": names}
    (out / "data.yaml").write_text(yaml.safe_dump(result_yaml, sort_keys=False), encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps({"seed": args.seed, "ratios": dict(zip(split_names, ratios)), "grouping": "sha256", "preserve_test": args.preserve_test, "records": manifest}, indent=2), encoding="utf-8")
    print("Split complete:", ", ".join(f"{s}={sum(len(g) for g in gs)}" for s, gs in zip(split_names, split_groups)))
    print(f"Dataset YAML: {out / 'data.yaml'}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
