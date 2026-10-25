#!/usr/bin/env python3
"""Convert a split COCO detection dataset to a clean YOLO dataset.

Category IDs are remapped by category name across all splits, so each class has
one consistent YOLO ID even when a COCO export uses different category IDs or
orders in train/validation/test. Existing source data is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import DATASET_DIR  # noqa: E402

SPLIT_ALIASES = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _category_key(name: str) -> str:
    return " ".join(name.casefold().strip().split())


def _safe_file_path(root: Path, file_name: str) -> Path:
    candidate = (root / file_name).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise ValueError(f"COCO image path escapes its split directory: {file_name!r}")
    return candidate


def _load_split(source_root: Path, split: str) -> tuple[Path, dict[str, Any]] | None:
    matches = []
    for folder_name in SPLIT_ALIASES[split]:
        split_root = source_root / folder_name
        annotation_file = split_root / "_annotations.coco.json"
        if annotation_file.is_file():
            try:
                data = json.loads(annotation_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read COCO annotations {annotation_file}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"COCO annotation root must be an object: {annotation_file}")
            for key in ("images", "annotations", "categories"):
                if not isinstance(data.get(key), list):
                    raise ValueError(f"{annotation_file} must contain a '{key}' list")
            matches.append((split_root, data))
    if len(matches) > 1:
        raise ValueError(f"COCO source has multiple folders for the {split} split; keep either '{SPLIT_ALIASES[split][0]}' or '{SPLIT_ALIASES[split][-1]}'.")
    return matches[0] if matches else None


def _read_categories(splits: dict[str, tuple[Path, dict[str, Any]]]) -> tuple[list[str], dict[str, dict[str, int]]]:
    canonical_name_by_key: dict[str, str] = {}
    local_names: dict[str, dict[str, str]] = {}
    for split, (_, data) in splits.items():
        local_names[split] = {}
        for category in data["categories"]:
            if not isinstance(category, dict) or "id" not in category or not isinstance(category.get("name"), str):
                raise ValueError(f"Malformed COCO category in {split}: {category!r}")
            category_id = str(category["id"])
            name = category["name"].strip()
            key = _category_key(name)
            if not category_id or not key:
                raise ValueError(f"COCO categories need a non-empty ID and name in {split}")
            if category_id in local_names[split]:
                raise ValueError(f"Duplicate COCO category ID {category_id!r} in {split}")
            previous = canonical_name_by_key.get(key)
            if previous is None or (name.casefold(), name) < (previous.casefold(), previous):
                canonical_name_by_key[key] = name
            local_names[split][category_id] = key
    if not canonical_name_by_key:
        raise ValueError("No COCO categories were found.")
    class_keys = sorted(canonical_name_by_key)
    classes = [canonical_name_by_key[key] for key in class_keys]
    class_ids = {key: index for index, key in enumerate(class_keys)}
    mappings = {
        split: {category_id: class_ids[name_key] for category_id, name_key in names.items()}
        for split, names in local_names.items()
    }
    return classes, mappings


def _index_images(data: dict[str, Any], annotation_file: Path) -> dict[str, dict[str, Any]]:
    images = {}
    for item in data["images"]:
        if not isinstance(item, dict) or "id" not in item or not isinstance(item.get("file_name"), str):
            raise ValueError(f"Malformed COCO image entry in {annotation_file}: {item!r}")
        image_id = str(item["id"])
        if image_id in images:
            raise ValueError(f"Duplicate COCO image ID {image_id!r} in {annotation_file}")
        try:
            width, height = int(item["width"]), int(item["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Image {image_id} has invalid dimensions in {annotation_file}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"Image {image_id} has non-positive dimensions in {annotation_file}")
        images[image_id] = item
    return images


def _convert_annotation(ann: dict[str, Any], image: dict[str, Any], category_mapping: dict[str, int], location: str) -> tuple[int, float, float, float, float]:
    if not isinstance(ann, dict):
        raise ValueError(f"Malformed COCO annotation in {location}: {ann!r}")
    image_id = str(ann.get("image_id"))
    category_id = str(ann.get("category_id"))
    if image_id != str(image["id"]):
        raise ValueError(f"Annotation image ID mismatch in {location}")
    if category_id not in category_mapping:
        raise ValueError(f"Annotation uses unknown category ID {category_id!r} in {location}")
    bbox = ann.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Annotation has malformed COCO bbox in {location}: {bbox!r}")
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Annotation has non-numeric bbox in {location}: {bbox!r}") from exc
    image_width, image_height = int(image["width"]), int(image["height"])
    if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
        raise ValueError(f"Annotation has invalid bbox in {location}: {bbox!r}")
    if x < 0 or y < 0 or x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
        raise ValueError(f"Annotation bbox is outside image bounds in {location}: {bbox!r}")
    center_x = (x + width / 2) / image_width
    center_y = (y + height / 2) / image_height
    return category_mapping[category_id], center_x, center_y, width / image_width, height / image_height


def convert_coco_to_yolo(dataset_dir: str | Path | None = None, output_dir: str | Path | None = None) -> Path:
    source_root = Path(dataset_dir).expanduser().resolve() if dataset_dir else Path(DATASET_DIR).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else source_root.parent / "logos_yolo"
    if not source_root.is_dir():
        raise ValueError(f"COCO source dataset directory does not exist: {source_root}")
    if destination == source_root or source_root in destination.parents:
        raise ValueError("Output directory cannot be inside the COCO source dataset.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Output must be absent or empty: {destination}")

    split_data: dict[str, tuple[Path, dict[str, Any]]] = {}
    for split in SPLIT_ALIASES:
        loaded = _load_split(source_root, split)
        if loaded:
            split_data[split] = loaded
    if "train" not in split_data or "val" not in split_data:
        raise ValueError("COCO source must contain both train and validation annotation files (in train/ and valid/ or val/).")

    classes, category_mappings = _read_categories(split_data)
    prepared: dict[str, list[tuple[Path, str, list[tuple[int, float, float, float, float]]]]] = {}
    for split, (split_root, data) in split_data.items():
        annotation_file = split_root / "_annotations.coco.json"
        images = _index_images(data, annotation_file)
        annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ann in data["annotations"]:
            if not isinstance(ann, dict) or "image_id" not in ann:
                raise ValueError(f"Malformed COCO annotation in {annotation_file}: {ann!r}")
            image_key = str(ann["image_id"])
            if image_key not in images:
                raise ValueError(f"Annotation references missing image ID {image_key!r} in {annotation_file}")
            annotations_by_image[image_key].append(ann)

        image_root = split_root / "images"
        records = []
        for image_key, image_info in images.items():
            file_name = image_info["file_name"]
            if Path(file_name).is_absolute():
                raise ValueError(f"COCO file_name must be relative in {annotation_file}: {file_name!r}")
            src_image = _safe_file_path(image_root, file_name)
            if not src_image.is_file():
                # Some COCO exports store files directly under the split folder.
                src_image = _safe_file_path(split_root, file_name)
            if not src_image.is_file():
                raise ValueError(f"Image {file_name!r} listed in {annotation_file} was not found.")
            if src_image.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image extension in COCO source: {src_image}")
            boxes = [
                _convert_annotation(ann, image_info, category_mappings[split], f"{annotation_file}, image={image_key}")
                for ann in annotations_by_image[image_key]
            ]
            records.append((src_image, image_key, boxes))
        prepared[split] = records

    summary = {
        "source": str(source_root),
        "output": str(destination),
        "classes": classes,
        "splits": {split: {"images": len(records), "objects": sum(len(record[2]) for record in records)} for split, records in prepared.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.tmp-", dir=destination.parent) as temp_dir:
        staging = Path(temp_dir) / "dataset"
        staging.mkdir()
        for split, records in prepared.items():
            output_split = "valid" if split == "val" else split
            for src_image, image_key, boxes in records:
                token = hashlib.sha1(f"{split}\0{image_key}\0{src_image.name}".encode()).hexdigest()[:12]
                output_name = f"{token}_{src_image.name}"
                image_target = staging / output_split / "images" / output_name
                label_target = staging / output_split / "labels" / f"{Path(output_name).stem}.txt"
                image_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_image, image_target)
                lines = [f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}" for class_id, x, y, width, height in boxes]
                label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        data_yaml: dict[str, Any] = {"path": ".", "nc": len(classes), "names": classes}
        for split in prepared:
            output_split = "valid" if split == "val" else split
            data_yaml["train" if split == "train" else "val" if split == "val" else "test"] = f"{output_split}/images"
        with (staging / "data.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data_yaml, stream, sort_keys=False, allow_unicode=True)
        (staging / "conversion_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if destination.exists():
            destination.rmdir()  # only an empty directory was allowed above
        staging.rename(destination)
    print(f"Converted COCO dataset: {source_root}")
    print(f"YOLO output: {destination}")
    print(f"Classes: {len(classes)}")
    for split, counts in summary["splits"].items():
        print(f"  {split}: {counts['images']} images, {counts['objects']} boxes")
    print(f"Dataset YAML: {destination / 'data.yaml'}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET_DIR, help="COCO dataset root containing train/valid/test split folders")
    parser.add_argument("--output", type=Path, help="New YOLO output directory (default: sibling folder named logos_yolo)")
    args = parser.parse_args()
    try:
        convert_coco_to_yolo(args.input, args.output)
        return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
