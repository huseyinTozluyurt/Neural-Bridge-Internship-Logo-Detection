"""Preflight validation for YOLO image/label datasets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_split(value: Any, yaml_path: Path, configured_root: Path | None) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            path = (configured_root / path) if configured_root else (yaml_path.parent / path)
        result.append(path.resolve())
    return result


def validate_yolo_dataset(data_yaml: str | Path) -> dict[str, Any]:
    """Return errors, warnings, and counts after checking a YOLO data YAML."""
    yaml_path = Path(data_yaml).expanduser().resolve()
    report: dict[str, Any] = {"errors": [], "warnings": [], "splits": {}, "resolved_paths": {}}
    if not yaml_path.is_file():
        report["errors"].append(f"Dataset YAML not found: {yaml_path}")
        return report

    try:
        with yaml_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except Exception as exc:
        report["errors"].append(f"Could not read dataset YAML: {exc}")
        return report
    if not isinstance(config, dict):
        report["errors"].append("Dataset YAML must contain a mapping")
        return report

    try:
        nc = int(config["nc"])
        if nc <= 0:
            raise ValueError("nc must be positive")
    except (KeyError, TypeError, ValueError) as exc:
        report["errors"].append(f"Invalid or missing nc: {exc}")
        return report
    names = config.get("names")
    if isinstance(names, dict):
        try:
            keyed_names = {int(key): value for key, value in names.items()}
        except (TypeError, ValueError):
            report["errors"].append("names mapping keys must be numeric class IDs")
            return report
        if len(keyed_names) != len(names):
            report["errors"].append("names mapping contains duplicate numeric class IDs")
            return report
        if sorted(keyed_names) != list(range(len(keyed_names))):
            report["errors"].append("names mapping IDs must be contiguous and start at 0")
            return report
        names = [keyed_names[index] for index in range(len(keyed_names))]
    if not isinstance(names, list) or len(names) != nc:
        report["errors"].append(f"names must contain exactly nc={nc} entries")
    elif any(not isinstance(name, str) or not name.strip() for name in names):
        report["errors"].append("every names entry must be a non-empty string")

    configured_root = None
    if config.get("path"):
        configured_root = Path(str(config["path"])).expanduser()
        if not configured_root.is_absolute():
            configured_root = (yaml_path.parent / configured_root).resolve()

    images_by_split: dict[str, set[Path]] = {}
    class_counts_by_split: dict[str, dict[int, int]] = {}
    for split in ("train", "val", "test"):
        if split not in config:
            if split in ("train", "val"):
                report["errors"].append(f"Dataset YAML is missing required '{split}' split")
            continue
        roots = _resolve_split(config[split], yaml_path, configured_root)
        report["resolved_paths"][split] = [str(path) for path in roots]
        split_report = {"images": 0, "positive_images": 0, "objects": 0, "missing_label_files": 0}
        split_image_paths: set[Path] = set()
        split_class_counts: dict[int, int] = {}
        seen_stems: set[tuple[Path, str]] = set()
        for image_root in roots:
            if not image_root.is_dir():
                report["errors"].append(f"{split} image directory not found: {image_root}")
                continue
            image_files = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not image_files:
                report["errors"].append(f"{split} image directory contains no supported images: {image_root}")
                continue
            labels_root = image_root.parent / "labels" if image_root.name.lower() == "images" else image_root / "labels"
            image_relatives = set()
            for image_path in image_files:
                split_report["images"] += 1
                split_image_paths.add(image_path.resolve())
                stem_key = (image_path.parent, image_path.stem)
                if stem_key in seen_stems:
                    report["errors"].append(f"Duplicate image stem in {split}: {image_path.stem}")
                seen_stems.add(stem_key)
                image_relatives.add(image_path.relative_to(image_root).with_suffix(""))
                label_path = labels_root / image_path.relative_to(image_root).with_suffix(".txt")
                if not label_path.is_file():
                    split_report["missing_label_files"] += 1
                    continue  # YOLO treats missing label files as negative images.
                try:
                    lines = label_path.read_text(encoding="utf-8").splitlines()
                except OSError as exc:
                    report["errors"].append(f"Cannot read {label_path}: {exc}")
                    continue
                image_has_object = False
                for line_no, line in enumerate(lines, start=1):
                    if not line.strip():
                        continue
                    parts = line.split()
                    location = f"{label_path}:{line_no}"
                    if len(parts) != 5:
                        report["errors"].append(f"{location}: expected 5 values (class x_center y_center width height)")
                        continue
                    try:
                        cls = int(parts[0])
                        x, y, width, height = map(float, parts[1:])
                    except ValueError:
                        report["errors"].append(f"{location}: contains a non-numeric value")
                        continue
                    if cls < 0 or cls >= nc:
                        report["errors"].append(f"{location}: class id {cls} is outside 0..{nc - 1}")
                    if not all(math.isfinite(v) for v in (x, y, width, height)):
                        report["errors"].append(f"{location}: coordinates must be finite")
                        continue
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        report["errors"].append(f"{location}: coordinates must be normalized and box width/height positive")
                        continue
                    if x - width / 2 < -1e-6 or y - height / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y + height / 2 > 1 + 1e-6:
                        report["errors"].append(f"{location}: bounding box extends outside the image")
                        continue
                    split_report["objects"] += 1
                    split_class_counts[cls] = split_class_counts.get(cls, 0) + 1
                    image_has_object = True
                if image_has_object:
                    split_report["positive_images"] += 1

            if labels_root.is_dir():
                for label_path in labels_root.rglob("*.txt"):
                    if label_path.name.lower() == "classes.txt":
                        continue  # Roboflow class-name metadata, not an image label.
                    relative = label_path.relative_to(labels_root).with_suffix("")
                    if relative not in image_relatives:
                        report["errors"].append(f"Label has no matching image in {split}: {label_path}")

        images_by_split[split] = split_image_paths
        class_counts_by_split[split] = split_class_counts
        report["splits"][split] = split_report
        if split_report["images"] and split_report["missing_label_files"]:
            report["warnings"].append(
                f"{split}: {split_report['missing_label_files']} images have no label file and will be treated as background negatives"
            )
        if split in ("train", "val") and split_report["images"] and not split_report["objects"]:
            report["errors"].append(f"{split} split has no valid logo annotations")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left in images_by_split and right in images_by_split:
            overlap = images_by_split[left] & images_by_split[right]
            if overlap:
                report["errors"].append(f"{left} and {right} splits share {len(overlap)} image file(s), e.g. {next(iter(overlap))}")
    report["class_object_counts"] = {
        split: {str(class_id): count for class_id, count in sorted(counts.items())}
        for split, counts in class_counts_by_split.items()
    }
    if isinstance(names, list):
        train_counts = class_counts_by_split.get("train", {})
        val_counts = class_counts_by_split.get("val", {})
        test_counts = class_counts_by_split.get("test", {})
        missing_train = [str(names[i]) for i in range(len(names)) if train_counts.get(i, 0) == 0]
        missing_val = [str(names[i]) for i in range(len(names)) if val_counts.get(i, 0) == 0]
        missing_test = [str(names[i]) for i in range(len(names)) if "test" in class_counts_by_split and test_counts.get(i, 0) == 0]
        sparse_val = [
            f"{names[i]} ({val_counts[i]})"
            for i in range(len(names))
            if 0 < val_counts.get(i, 0) < 5
        ]
        if missing_train:
            report["errors"].append("No training annotations for class(es): " + ", ".join(missing_train))
        if missing_val:
            report["warnings"].append("No validation annotations for class(es); their per-class validation performance cannot be measured: " + ", ".join(missing_val))
        if sparse_val:
            report["warnings"].append("Fewer than 5 validation objects for class(es); per-class metrics will be noisy: " + ", ".join(sparse_val))
        if missing_test:
            report["warnings"].append("No test annotations for class(es); their per-class test performance cannot be measured: " + ", ".join(missing_test))
    return report


def print_validation_report(report: dict[str, Any]) -> bool:
    for split, counts in report.get("splits", {}).items():
        print(
            f"{split}: {counts['images']} images, {counts['positive_images']} with objects, "
            f"{counts['objects']} objects, {counts['missing_label_files']} missing label files"
        )
    for warning in report.get("warnings", []):
        print(f"⚠ {warning}")
    for error in report.get("errors", []):
        print(f"❌ {error}")
    if report.get("errors"):
        print("Dataset validation failed; training was not started.")
        return False
    print("✓ Dataset validation passed")
    return True
