#!/usr/bin/env python3
"""Convert a Pascal VOC logo dataset to multi-class YOLO format.

Every VOC ``object/name`` is retained as a class. Source files are read-only;
the generated dataset is staged and moved into place only after conversion
finishes successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_voc_annotation(xml_file: Path) -> tuple[str, int, int, list[tuple[str, float, float, float, float]]]:
    """Return the referenced image, dimensions, and named VOC boxes."""
    try:
        root = ET.parse(xml_file).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Cannot parse VOC annotation {xml_file}: {exc}") from exc

    filename_node = root.find("filename")
    size = root.find("size")
    if filename_node is None or not filename_node.text or size is None:
        raise ValueError(f"{xml_file} must provide filename and size metadata")
    try:
        width = int(size.findtext("width", ""))
        height = int(size.findtext("height", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{xml_file} has invalid image dimensions") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_file} has non-positive image dimensions")

    boxes = []
    for object_index, obj in enumerate(root.findall("object"), start=1):
        name_node = obj.find("name")
        box_node = obj.find("bndbox")
        class_name = name_node.text.strip() if name_node is not None and name_node.text else ""
        if not class_name or box_node is None:
            raise ValueError(f"{xml_file}: object {object_index} is missing a class name or bndbox")
        try:
            xmin = float(box_node.findtext("xmin", ""))
            ymin = float(box_node.findtext("ymin", ""))
            xmax = float(box_node.findtext("xmax", ""))
            ymax = float(box_node.findtext("ymax", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{xml_file}: object {object_index} has a non-numeric box") from exc
        if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
            raise ValueError(f"{xml_file}: object {object_index} has a non-finite box")
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmax <= xmin or ymax <= ymin:
            raise ValueError(f"{xml_file}: object {object_index} has an out-of-bounds box")
        boxes.append((class_name, xmin, ymin, xmax, ymax))

    return filename_node.text.strip().replace("\\", "/"), width, height, boxes


def _split_counts(image_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate integer split sizes, guaranteeing all three splits are non-empty."""
    exact = [image_count * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    for index in sorted(range(3), key=lambda i: exact[i] - counts[i], reverse=True)[: image_count - sum(counts)]:
        counts[index] += 1
    for empty_index in [index for index, count in enumerate(counts) if count == 0]:
        donor = max((index for index, count in enumerate(counts) if count > 1), key=lambda index: counts[index])
        counts[donor] -= 1
        counts[empty_index] += 1
    return tuple(counts)


def _stratified_split_records(
    records: list[dict[str, Any]],
    class_ids: dict[str, int],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Greedily balance per-image class presence and split sizes."""
    split_names = ("train", "valid", "test")
    rng = random.Random(seed)
    record_classes = [
        {class_ids[box[0]] for box in record["boxes"]}
        for record in records
    ]
    total_class_images: Counter[int] = Counter()
    for classes_in_record in record_classes:
        total_class_images.update(classes_in_record)
    indexed_records = list(enumerate(records))
    rng.shuffle(indexed_records)
    indexed_records.sort(key=lambda item: (
        min((total_class_images[class_id] for class_id in record_classes[item[0]]), default=10**12),
        -len(record_classes[item[0]]),
    ))

    target_classes = [Counter({class_id: count * ratio for class_id, count in total_class_images.items()}) for ratio in ratios]
    target_sizes = list(_split_counts(len(records), ratios))
    assigned_classes = [Counter() for _ in ratios]
    assigned_sizes = [0] * len(ratios)
    result: list[list[dict[str, Any]]] = [[] for _ in ratios]

    # Reserve training examples for every class before balancing the other
    # splits. This matters for brands with only a few annotated images.
    uncovered = set(total_class_images)
    reserved_indexes: set[int] = set()
    while uncovered:
        candidates = [item for item in indexed_records if item[0] not in reserved_indexes]
        record_index, record = max(
            candidates,
            key=lambda item: (
                len(record_classes[item[0]] & uncovered),
                sum(1 / total_class_images[class_id] for class_id in record_classes[item[0]] & uncovered),
                rng.random(),
            ),
        )
        covered = record_classes[record_index] & uncovered
        if not covered:
            raise ValueError("Could not reserve at least one training example for every class")
        reserved_indexes.add(record_index)
        uncovered.difference_update(covered)
    if len(reserved_indexes) > target_sizes[0]:
        raise ValueError(
            f"The requested train ratio can hold {target_sizes[0]} images, but at least {len(reserved_indexes)} are needed "
            "to include every class in training. Increase the train ratio or combine classes deliberately."
        )
    for record_index, record in indexed_records:
        if record_index in reserved_indexes:
            result[0].append(record)
            assigned_classes[0].update(record_classes[record_index])
            assigned_sizes[0] += 1

    remaining_records = [item for item in indexed_records if item[0] not in reserved_indexes]
    # Seed validation and test with rare-first images so neither is empty.
    for split_index in (1, 2):
        record_index, record = remaining_records.pop(0)
        result[split_index].append(record)
        assigned_classes[split_index].update(record_classes[record_index])
        assigned_sizes[split_index] += 1

    for record_index, record in remaining_records:
        classes_in_record = record_classes[record_index]

        def score(split_index: int) -> float:
            gains = [
                max(0.0, target_classes[split_index][class_id] - assigned_classes[split_index][class_id])
                / max(1.0, target_classes[split_index][class_id])
                for class_id in classes_in_record
            ]
            class_gain = sum(gains) / len(gains) if gains else 0.0
            size_gap = max(0.0, target_sizes[split_index] - assigned_sizes[split_index]) / max(1.0, target_sizes[split_index])
            return class_gain + size_gap + rng.random() * 1e-9

        available_splits = [index for index in range(3) if assigned_sizes[index] < target_sizes[index]]
        selected = max(available_splits, key=score)
        result[selected].append(record)
        assigned_classes[selected].update(classes_in_record)
        assigned_sizes[selected] += 1
    return dict(zip(split_names, result))


def convert_voc_to_yolo(
    source_dir: str | Path,
    output_dir: str | Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Path:
    source_root = Path(source_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    annotations_dir = source_root / "Annotations"
    images_dir = source_root / "JPEGImages"
    if not annotations_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(f"Expected VOC folders at {annotations_dir} and {images_dir}")
    if destination == source_root or source_root in destination.parents:
        raise ValueError("Output directory cannot be inside the source VOC dataset")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Output must be absent or empty: {destination}")
    if len(ratios) != 3 or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("Train, validation, and test ratios must all be finite and positive")
    ratio_sum = sum(ratios)
    if not math.isfinite(ratio_sum) or ratio_sum <= 0:
        raise ValueError("Split ratios must have a finite positive total")
    ratios = tuple(value / ratio_sum for value in ratios)

    xml_files = sorted(annotations_dir.rglob("*.xml"))
    if not xml_files:
        raise ValueError(f"No VOC XML annotations found in {annotations_dir}")

    records: list[dict[str, Any]] = []
    seen_images: set[Path] = set()
    class_names: set[str] = set()
    image_roots = images_dir.resolve()
    for xml_file in xml_files:
        filename, width, height, boxes = parse_voc_annotation(xml_file)
        relative_image = Path(filename)
        if relative_image.is_absolute():
            raise ValueError(f"VOC filename must be relative in {xml_file}: {filename!r}")
        image_path = (images_dir / relative_image).resolve()
        if image_path != image_roots and image_roots not in image_path.parents:
            raise ValueError(f"VOC filename escapes JPEGImages in {xml_file}: {filename!r}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed by {xml_file} was not found: {image_path}")
        if image_path in seen_images:
            raise ValueError(f"Multiple VOC XML files reference the same image: {image_path}")
        seen_images.add(image_path)
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension in {xml_file}: {image_path.suffix}")
        for class_name, *_ in boxes:
            class_names.add(class_name)
        records.append({
            "xml": xml_file,
            "image": image_path,
            "filename": filename,
            "width": width,
            "height": height,
            "boxes": boxes,
        })
    if not class_names:
        raise ValueError("No named logo objects were found in the VOC annotations")
    if len(records) < 3:
        raise ValueError("At least three annotated images are required to create train, validation, and test splits")

    classes = sorted(class_names, key=lambda name: (name.casefold(), name))
    class_ids = {name: index for index, name in enumerate(classes)}
    split_records = _stratified_split_records(records, class_ids, ratios, seed)
    if not any(record["boxes"] for record in split_records["valid"]):
        raise ValueError(
            "The validation split has no annotated logos. More positive images are needed, or adjust --ratios/--seed."
        )

    summary: dict[str, Any] = {
        "source": str(source_root),
        "output": str(destination),
        "classes": classes,
        "seed": seed,
        "requested_ratios": dict(zip(("train", "val", "test"), ratios)),
        "actual_ratios": {
            "train": 0.0,
            "val": 0.0,
            "test": 0.0,
        },
        "splits": {},
    }
    for split, split_items in zip(("train", "val", "test"), split_records.values()):
        summary["actual_ratios"][split] = len(split_items) / len(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.tmp-", dir=destination.parent) as temp_dir:
        staging = Path(temp_dir) / "dataset"
        staging.mkdir()
        for split, split_items in split_records.items():
            object_counts: Counter[str] = Counter()
            for record in split_items:
                relative_xml = record["xml"].relative_to(annotations_dir).as_posix()
                token = hashlib.sha1(relative_xml.encode("utf-8")).hexdigest()[:12]
                output_name = f"{token}_{record['image'].name}"
                image_target = staging / split / "images" / output_name
                label_target = staging / split / "labels" / f"{Path(output_name).stem}.txt"
                image_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record["image"], image_target)
                label_rows = []
                for class_name, xmin, ymin, xmax, ymax in record["boxes"]:
                    x_center = ((xmin + xmax) / 2) / record["width"]
                    y_center = ((ymin + ymax) / 2) / record["height"]
                    box_width = (xmax - xmin) / record["width"]
                    box_height = (ymax - ymin) / record["height"]
                    label_rows.append(
                        f"{class_ids[class_name]} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
                    )
                    object_counts[class_name] += 1
                label_target.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
            summary["splits"][split] = {
                "images": len(split_items),
                "objects": sum(object_counts.values()),
                "class_object_counts": {name: object_counts[name] for name in classes},
            }

        data_yaml = {
            "path": ".",
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(classes),
            "names": classes,
        }
        (staging / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (staging / "conversion_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)

    print(f"Converted VOC dataset: {source_root}")
    print(f"YOLO output: {destination}")
    print(f"Classes: {len(classes)}")
    for split, counts in summary["splits"].items():
        print(f"  {split}: {counts['images']} images, {counts['objects']} boxes")
    print(f"Dataset YAML: {destination / 'data.yaml'}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=PROJECT_ROOT / "openlogo",
        help="VOC dataset root containing Annotations/ and JPEGImages/ (default: <project>/openlogo)",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "datasets" / "openlogo_yolo",
        help="New YOLO dataset directory (default: <project>/datasets/openlogo_yolo)",
    )
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source.expanduser()
    output = args.output.expanduser()
    source = source if source.is_absolute() else PROJECT_ROOT / source
    output = output if output.is_absolute() else PROJECT_ROOT / output
    try:
        convert_voc_to_yolo(source, output, tuple(args.ratios), args.seed)
        return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
