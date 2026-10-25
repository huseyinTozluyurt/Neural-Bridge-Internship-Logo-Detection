#!/usr/bin/env python3
"""Audit a YOLO dataset and create reports plus annotated image previews."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_validation import IMAGE_SUFFIXES, validate_yolo_dataset  # noqa: E402


def _class_names(config: dict[str, Any]) -> list[str]:
    names = config.get("names", [])
    if isinstance(names, dict):
        if any(not isinstance(value, str) or not value.strip() for value in names.values()):
            return []
        try:
            keyed_names = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError):
            return []
        if len(keyed_names) != len(names):
            return []
        if sorted(keyed_names) != list(range(len(keyed_names))):
            return []
        return [keyed_names[index] for index in range(len(keyed_names))]
    if isinstance(names, list) and all(isinstance(name, str) and name.strip() for name in names):
        return names
    return []


def _read_label_file(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read valid-looking YOLO rows while retaining row errors in the audit."""
    boxes: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return boxes, errors
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return boxes, [f"Cannot read {path}: {exc}"]
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_no}: expected 5 fields")
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path}:{line_no}: invalid numeric value")
            continue
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            errors.append(f"{path}:{line_no}: non-finite coordinates")
            continue
        if width <= 0 or height <= 0:
            errors.append(f"{path}:{line_no}: non-positive box width or height")
            continue
        boxes.append({"class_id": class_id, "x": x, "y": y, "width": width, "height": height})
    return boxes, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preview_indices(count: int, limit: int) -> set[int]:
    if limit <= 0 or count <= 0:
        return set()
    if count <= limit:
        return set(range(count))
    if limit <= 1:
        return {0}
    return {round(i * (count - 1) / (limit - 1)) for i in range(limit)}


def _draw_preview(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    class_names: list[str],
) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    boxes, _ = _read_label_file(label_path)
    for box in boxes:
        class_id = box["class_id"]
        x, y, bw, bh = box["x"], box["y"], box["width"], box["height"]
        x1 = max(0, min(width - 1, round((x - bw / 2) * width)))
        y1 = max(0, min(height - 1, round((y - bh / 2) * height)))
        x2 = max(0, min(width - 1, round((x + bw / 2) * width)))
        y2 = max(0, min(height - 1, round((y + bh / 2) * height)))
        color = (40 + (class_id * 67) % 200, 220 - (class_id * 43) % 180, 70 + (class_id * 97) % 170)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        cv2.putText(image, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if not boxes:
        cv2.putText(image, "NO LABEL / NEGATIVE", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def audit_dataset(data_yaml: Path, output_dir: Path, preview_count: int = 20, hash_images: bool = True) -> dict[str, Any]:
    data_yaml = data_yaml.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    validation = validate_yolo_dataset(data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    with data_yaml.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    class_names = _class_names(config if isinstance(config, dict) else {})

    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    class_totals: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"images": 0, "boxes": 0})
    box_values: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    hash_occurrences: dict[str, list[dict[str, str]]] = defaultdict(list)
    image_count = 0
    preview_failures: list[str] = []
    label_issues: list[str] = []
    label_issue_count = 0

    for split, roots in validation.get("resolved_paths", {}).items():
        split_images: list[Path] = []
        for root_string in roots:
            root = Path(root_string)
            if root.is_dir():
                split_images.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        # Keep lexical paths for label lookup. YOLO datasets often symlink image
        # files to shared storage; resolving them here would pair them with the
        # source dataset's labels instead of this split's labels.
        split_images = sorted(set(split_images))
        preview_set = _preview_indices(len(split_images), max(0, preview_count))
        summary = {"images": len(split_images), "positive_images": 0, "negative_images": 0, "objects": 0, "unreadable_images": 0}

        for index, image_path in enumerate(split_images):
            image_count += 1
            image_root = next((Path(root) for root in validation["resolved_paths"][split] if image_path.is_relative_to(Path(root).resolve())), image_path.parent)
            labels_root = image_root.parent / "labels" if image_root.name.lower() == "images" else image_root / "labels"
            relative_image = image_path.relative_to(image_root)
            label_path = labels_root / relative_image.with_suffix(".txt")
            boxes, row_errors = _read_label_file(label_path)
            label_issue_count += len(row_errors)
            if len(label_issues) < 1000:
                label_issues.extend(row_errors[: 1000 - len(label_issues)])
            if boxes:
                summary["positive_images"] += 1
            else:
                summary["negative_images"] += 1
            summary["objects"] += len(boxes)
            for box in boxes:
                class_id = box["class_id"]
                class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                key = (split, class_name)
                class_totals[key]["boxes"] += 1
                box_values[key].append({
                    "area_fraction": box["width"] * box["height"],
                    "width_fraction": box["width"],
                    "height_fraction": box["height"],
                })
            for class_id in {box["class_id"] for box in boxes}:
                class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                class_totals[(split, class_name)]["images"] += 1

            if hash_images:
                try:
                    hash_occurrences[_sha256(image_path)].append({"split": split, "path": str(image_path)})
                except OSError as exc:
                    preview_failures.append(f"Could not hash {image_path}: {exc}")

            if index in preview_set:
                preview_path = output_dir / "previews" / split / relative_image.with_suffix(".jpg")
                if not _draw_preview(image_path, label_path, preview_path, class_names):
                    summary["unreadable_images"] += 1
                    preview_failures.append(f"Could not read/write preview for {image_path}")
        split_reports[split] = summary

    duplicates = []
    for digest, entries in hash_occurrences.items():
        if len(entries) > 1:
            duplicates.append({"sha256": digest, "cross_split": len({entry["split"] for entry in entries}) > 1, "files": entries})

    for split in split_reports:
        for class_name in class_names:
            class_totals[(split, class_name)]

    class_rows = []
    box_rows = []
    for (split, class_name), values in sorted(class_totals.items()):
        class_rows.append({"split": split, "class": class_name, **values})
        measurements = box_values[(split, class_name)]
        areas = [item["area_fraction"] for item in measurements]
        widths = [item["width_fraction"] for item in measurements]
        heights = [item["height_fraction"] for item in measurements]
        box_rows.append({
            "split": split,
            "class": class_name,
            "boxes": len(areas),
            "mean_area_fraction": mean(areas) if areas else 0.0,
            "median_area_fraction": median(areas) if areas else 0.0,
            "boxes_under_1pct_area": sum(area < 0.01 for area in areas),
            "boxes_under_5pct_area": sum(area < 0.05 for area in areas),
            "mean_width_fraction": mean(widths) if widths else 0.0,
            "mean_height_fraction": mean(heights) if heights else 0.0,
        })

    report = {
        "dataset_yaml": str(data_yaml),
        "total_images": image_count,
        "splits": split_reports,
        "classes": class_rows,
        "box_statistics": box_rows,
        "exact_duplicate_groups": duplicates,
        "cross_split_duplicate_groups": sum(item["cross_split"] for item in duplicates),
        "validation": validation,
        "label_issue_count": label_issue_count,
        "label_issues_sample": label_issues,
        "preview_or_hash_issues": preview_failures,
        "hashing_enabled": hash_images,
    }

    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(output_dir / "class_counts.csv", class_rows)
    _write_csv(output_dir / "box_statistics.csv", box_rows)
    duplicate_rows = [
        {"sha256": group["sha256"], "cross_split": group["cross_split"], "split": file["split"], "path": file["path"]}
        for group in duplicates for file in group["files"]
    ]
    _write_csv(output_dir / "duplicate_images.csv", duplicate_rows)
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a YOLO dataset and create reports and annotated previews.")
    parser.add_argument("--data", required=True, type=Path, help="YOLO data.yaml path")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "dataset_audit", help="Report output directory")
    parser.add_argument("--preview-count", type=int, default=20, help="Maximum evenly sampled annotated previews per split (default: 20)")
    parser.add_argument("--skip-hashes", action="store_true", help="Skip exact SHA-256 duplicate-image detection")
    args = parser.parse_args()
    if args.preview_count < 0:
        parser.error("--preview-count must be zero or greater")

    try:
        report = audit_dataset(args.data, args.output, args.preview_count, hash_images=not args.skip_hashes)
    except Exception as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 1

    print(f"Audited {report['total_images']} images from {report['dataset_yaml']}")
    for split, counts in report["splits"].items():
        print(f"  {split}: {counts['images']} images, {counts['positive_images']} with boxes, {counts['negative_images']} without boxes")
    print(f"Exact duplicate groups: {len(report['exact_duplicate_groups'])}; cross-split groups: {report['cross_split_duplicate_groups']}")
    print(f"Validator errors: {len(report['validation'].get('errors', []))}; warnings: {len(report['validation'].get('warnings', []))}")
    print(f"Reports and previews: {args.output.resolve()}")
    if report["validation"].get("errors"):
        print("Review audit_report.json; the existing dataset validator found errors.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
