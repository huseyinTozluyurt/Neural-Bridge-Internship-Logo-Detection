#!/usr/bin/env python3
"""Unify multiple YOLO object-detection datasets into one training dataset.

A class-map file controls explicit name aliases; by default, source class names
are unioned into one taxonomy. The script remaps source IDs, groups visual
near-duplicates to keep them within one split, and writes YOLOv7 folders, a
data.yaml, a manifest, and a class-support report. Source datasets are read-only.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import cv2
except ImportError:  # pragma: no cover - dependency message is handled by CLI
    cv2 = None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    image: Path
    label: Path
    source_yaml: Path
    source_key: str
    source_split: str
    boxes: list[tuple[int, float, float, float, float]]
    sha256: str
    dhash: int
    canonical_classes: set[int]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


class BKTree:
    """Small BK-tree for finding image hashes within a Hamming distance."""
    def __init__(self) -> None:
        self.root: tuple[int, list[int], dict[int, Any]] | None = None

    @staticmethod
    def distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def add(self, value: int, index: int) -> list[int]:
        if self.root is None:
            self.root = (value, [index], {})
            return []
        node = self.root
        while True:
            distance = self.distance(value, node[0])
            if distance == 0:
                node[1].append(index)
                return list(node[1][:-1])
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, [index], {})
                return []
            node = child

    def query(self, value: int, radius: int) -> list[int]:
        if self.root is None:
            return []
        found: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node[0])
            if distance <= radius:
                found.extend(node[1])
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return found


def normalize_name(value: str) -> str:
    """Normalize case/spacing/punctuation without guessing brand synonyms."""
    return "".join(char.casefold() for char in value if char.isalnum())


def load_class_map(path: Path) -> tuple[list[str] | None, dict[str, str]]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read class map {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Class map must contain a YAML mapping.")
    configured_classes = config.get("classes", "auto")
    classes = None if configured_classes == "auto" else configured_classes
    if classes is not None and not isinstance(classes, list):
        raise ValueError("'classes' must be a list or 'auto'.")
    classes = [str(name).strip() for name in classes] if classes is not None else None
    if classes is not None and (not classes or any(not name or not normalize_name(name) for name in classes)):
        raise ValueError("'classes' must contain non-empty canonical class names or be set to 'auto'.")
    if classes is None:
        classes = []
    if len({normalize_name(name) for name in classes}) != len(classes):
        raise ValueError("Canonical class names must be unique after case/punctuation normalization.")
    canonical_by_key = {normalize_name(name): name for name in classes}
    aliases: dict[str, str] = {key: name for key, name in canonical_by_key.items()}
    configured_aliases = config.get("aliases", {})
    if not isinstance(configured_aliases, dict):
        raise ValueError("'aliases' must map canonical class names to lists of source-name aliases.")
    for canonical, values in configured_aliases.items():
        canonical = str(canonical).strip()
        if not canonical or not normalize_name(canonical):
            raise ValueError("Alias target names must contain at least one letter or digit.")
        canonical_key = normalize_name(canonical)
        if classes and canonical_key not in canonical_by_key:
            raise ValueError(f"Alias target {canonical!r} is not in the canonical 'classes' list.")
        if not isinstance(values, list):
            raise ValueError(f"Aliases for {canonical!r} must be a list.")
        for alias in [canonical, *values]:
            key = normalize_name(str(alias))
            if not key:
                raise ValueError(f"Empty alias configured for {canonical!r}.")
            previous = aliases.get(key)
            target_name = canonical_by_key.get(canonical_key, canonical)
            if previous is not None and previous != target_name:
                raise ValueError(f"Alias {alias!r} maps to both {previous!r} and {canonical!r}.")
            aliases[key] = target_name
    return (classes if classes else None), aliases


def class_names(config: dict[str, Any]) -> list[str]:
    names = config.get("names")
    if isinstance(names, list):
        if any(not isinstance(value, str) for value in names):
            raise ValueError("Dataset YAML class names must be strings.")
        result = [value.strip() for value in names]
    elif isinstance(names, dict):
        if any(not isinstance(value, str) for value in names.values()):
            raise ValueError("Dataset YAML class names must be strings.")
        try:
            keyed_names = {int(key): value.strip() for key, value in names.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError("Dataset YAML 'names' mapping keys must be numeric class IDs.") from exc
        if len(keyed_names) != len(names):
            raise ValueError("Dataset YAML 'names' contains duplicate numeric class IDs.")
        if sorted(keyed_names) != list(range(len(keyed_names))):
            raise ValueError("Dataset YAML 'names' IDs must be contiguous and start at 0.")
        result = [keyed_names[index] for index in range(len(keyed_names))]
    else:
        raise ValueError("Dataset YAML must define class 'names' as a list or numeric-key mapping.")
    if not result:
        raise ValueError("Dataset YAML contains no class names.")
    if any(not name.strip() for name in result):
        raise ValueError("Source dataset class names cannot be empty.")
    if "nc" in config:
        try:
            declared_count = int(config["nc"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Source dataset 'nc' must be an integer.") from exc
        if declared_count != len(result):
            raise ValueError(f"Source dataset declares nc={declared_count} but has {len(result)} class names.")
    return result


def resolve_split_paths(config: dict[str, Any], yaml_path: Path) -> dict[str, list[Path]]:
    root_value = config.get("path") or yaml_path.parent
    if not isinstance(root_value, (str, Path)):
        raise ValueError(f"Dataset YAML 'path' must be a directory path: {yaml_path}")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    result: dict[str, list[Path]] = {}
    for split in SPLITS:
        source_key = "valid" if split == "val" and "val" not in config and "valid" in config else split
        if source_key not in config:
            continue
        values = config[source_key] if isinstance(config[source_key], list) else [config[source_key]]
        paths = []
        for value in values:
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"Dataset YAML split '{source_key}' must contain directory paths: {yaml_path}")
            candidate = Path(value).expanduser()
            paths.append(candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve())
        result[split] = paths
    return result


def label_path_for(image: Path, images_root: Path) -> Path:
    relative = image.relative_to(images_root)
    if images_root.name.casefold() == "images":
        return images_root.parent / "labels" / relative.with_suffix(".txt")
    sibling_labels = images_root.parent / "labels"
    if sibling_labels.is_dir():
        return sibling_labels / relative.with_suffix(".txt")
    return images_root / "labels" / relative.with_suffix(".txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> int:
    if cv2 is None:
        raise RuntimeError("Perceptual duplicate grouping needs OpenCV (cv2). Install the project's OpenCV requirements.")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def read_boxes(
    path: Path,
    source_names: list[str],
    aliases: dict[str, str],
    canonical_ids: dict[str, int],
    exclude_unmapped_image: bool = False,
) -> list[tuple[int, float, float, float, float]] | None:
    if not path.is_file():
        return []
    boxes = []
    has_unmapped_class = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields; found {len(fields)}")
        try:
            source_id = int(fields[0])
            coords = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid class ID or coordinate") from exc
        if source_id < 0 or source_id >= len(source_names):
            raise ValueError(f"{path}:{line_number}: class ID {source_id} is outside the source names list")
        if any(not (0.0 <= value <= 1.0) for value in coords) or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{path}:{line_number}: normalized xywh coordinates must be in [0, 1] with positive size")
        x, y, width, height = coords
        if x - width / 2 < -1e-6 or y - height / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y + height / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_number}: bounding box extends outside the image")
        source_name = source_names[source_id]
        key = normalize_name(source_name)
        canonical = aliases.get(key)
        if canonical is None:
            if exclude_unmapped_image:
                has_unmapped_class = True
                continue
            raise ValueError(f"Unmapped source class {source_name!r} in {path}. Add it to the class map or remove that source class.")
        boxes.append((canonical_ids[canonical], *coords))
    if has_unmapped_class:
        return None
    return boxes


def collect_samples(
    data_yamls: list[Path],
    aliases: dict[str, str],
    canonical_ids: dict[str, int],
    compute_dhash: bool,
    allow_missing_labels: bool,
    exclude_unmapped_images: bool,
) -> tuple[list[Sample], list[dict[str, Any]], list[dict[str, str]]]:
    samples: list[Sample] = []
    source_summary = []
    excluded_images: list[dict[str, str]] = []
    for source_index, yaml_path in enumerate(data_yamls):
        yaml_path = yaml_path.expanduser().resolve()
        if not yaml_path.is_file():
            raise ValueError(f"Dataset YAML not found: {yaml_path}")
        try:
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {yaml_path}: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"Dataset YAML must be a mapping: {yaml_path}")
        source_names = class_names(config)
        found_for_source = 0
        missing_labels_for_source = 0
        excluded_for_source = 0
        for split, roots in resolve_split_paths(config, yaml_path).items():
            for root_index, images_root in enumerate(roots):
                if not images_root.is_dir():
                    raise ValueError(f"Configured {split} image directory does not exist: {images_root}")
                for image in sorted(path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
                    label = label_path_for(image, images_root)
                    if not label.is_file():
                        missing_labels_for_source += 1
                        if not allow_missing_labels:
                            raise ValueError(f"Missing YOLO label file for {image}: {label}. Add the correct empty label file for a confirmed negative image or pass --allow-missing-labels-as-negative only when missing labels are intentional negatives.")
                    boxes = read_boxes(label, source_names, aliases, canonical_ids, exclude_unmapped_images)
                    if boxes is None:
                        excluded_for_source += 1
                        excluded_images.append({"image": str(image), "reason": "contains at least one unmapped class annotation"})
                        continue
                    found_for_source += 1
                    samples.append(Sample(
                        image=image.resolve(), label=label.resolve(), source_yaml=yaml_path,
                        source_key=f"{source_index}:{split}:{root_index}:{image.relative_to(images_root).as_posix()}",
                        source_split=split, boxes=boxes,
                        sha256=sha256_file(image), dhash=difference_hash(image) if compute_dhash else 0,
                        canonical_classes={box[0] for box in boxes},
                    ))
        if found_for_source == 0:
            raise ValueError(f"No images found in configured splits for {yaml_path}")
        roboflow_metadata = config.get("roboflow", {})
        if not isinstance(roboflow_metadata, dict):
            roboflow_metadata = {}
        source_summary.append({
            "data_yaml": str(yaml_path),
            "dataset_name": str(
                config.get("dataset_name")
                or roboflow_metadata.get("project")
                or yaml_path.parent.name
            ),
            "license": str(config.get("license") or roboflow_metadata.get("license") or "") or None,
            "source_url": str(config.get("source_url") or roboflow_metadata.get("url") or "") or None,
            "source_workspace": str(roboflow_metadata.get("workspace") or "") or None,
            "source_version": roboflow_metadata.get("version"),
            "images_included": found_for_source,
            "classes": len(source_names), "missing_labels_treated_as_negative": missing_labels_for_source,
            "images_excluded_for_unmapped_classes": excluded_for_source,
            "class_mapping": [
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "canonical_name": aliases.get(normalize_name(source_name)),
                    "canonical_id": canonical_ids.get(aliases.get(normalize_name(source_name), "")),
                }
                for source_id, source_name in enumerate(source_names)
            ],
        })
    return samples, source_summary, excluded_images


def duplicate_groups(samples: list[Sample], max_distance: int) -> tuple[list[list[int]], list[dict[str, Any]]]:
    if max_distance == 0:
        exact: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            exact[sample.sha256].append(index)
        groups = list(exact.values())
        duplicates = [
            {"sha256": digest, "files": [str(samples[index].image) for index in indexes]}
            for digest, indexes in exact.items() if len(indexes) > 1
        ]
        return groups, duplicates
    uf = UnionFind(len(samples))
    tree = BKTree()
    exact_buckets: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        for previous in tree.query(sample.dhash, max_distance):
            uf.union(index, previous)
        tree.add(sample.dhash, index)
        exact_buckets[sample.sha256].append(index)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        groups[uf.find(index)].append(index)
    exact_duplicates = [
        {"sha256": digest, "files": [str(samples[index].image) for index in indexes]}
        for digest, indexes in exact_buckets.items() if len(indexes) > 1
    ]
    return list(groups.values()), exact_duplicates


def stratified_group_split(samples: list[Sample], groups: list[list[int]], ratios: list[float], seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    group_records = []
    total_class_images = Counter()
    for indexes in groups:
        counts = Counter(class_id for index in indexes for class_id in samples[index].canonical_classes)
        total_class_images.update(counts)
        group_records.append((indexes, counts))
    rng.shuffle(group_records)
    # Give rare-class groups priority, while retaining deterministic tie-breaking.
    group_records.sort(key=lambda item: (
        min((total_class_images[c] for c in item[1]), default=10**12),
        -len(item[0]),
    ))
    target_classes = [Counter({c: count * ratio for c, count in total_class_images.items()}) for ratio in ratios]
    participating_images = sum(len(indexes) for indexes, _ in group_records)
    target_sizes = [participating_images * ratio for ratio in ratios]
    assigned_classes = [Counter() for _ in ratios]
    assigned_sizes = [0] * len(ratios)
    result: list[list[int]] = [[] for _ in ratios]
    required_splits = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if len(group_records) < len(required_splits):
        raise ValueError(f"Need at least {len(required_splits)} independent image groups for the requested non-empty splits; found {len(group_records)}")

    # Guarantee every requested split has at least one independent group.
    for split_index in required_splits:
        indexes, counts = group_records.pop(0)
        result[split_index].extend(indexes)
        assigned_classes[split_index].update(counts)
        assigned_sizes[split_index] += len(indexes)

    for indexes, counts in group_records:
        def score(split_index: int) -> float:
            class_gain = sum(
                min(count, max(0.0, target_classes[split_index][class_id] - assigned_classes[split_index][class_id]))
                / max(1.0, target_classes[split_index][class_id])
                for class_id, count in counts.items()
            )
            size_gap = max(0.0, target_sizes[split_index] - assigned_sizes[split_index]) / max(1.0, target_sizes[split_index])
            return class_gain + size_gap + rng.random() * 1e-9
        chosen = max((index for index, ratio in enumerate(ratios) if ratio > 0), key=score)
        result[chosen].extend(indexes)
        assigned_classes[chosen].update(counts)
        assigned_sizes[chosen] += len(indexes)
    return result


def _group_has_class_boxes(group: list[int], samples: list[Sample], class_id: int) -> bool:
    return any(box[0] == class_id for index in group for box in samples[index].boxes)


def ensure_train_class_coverage(
    samples: list[Sample],
    groups: list[list[int]],
    split_indexes: list[list[int]],
    class_count: int,
) -> list[list[int]]:
    """Ensure every class with any labeled box has at least one training box.

    The stratified group splitter can leave ultra-rare brands entirely in val/test
    when their images share duplicate groups with other splits. Moving whole groups
    preserves duplicate integrity and avoids unusable YOLO heads for those classes.
    """
    index_to_group: dict[int, int] = {}
    for group_id, group in enumerate(groups):
        for index in group:
            index_to_group[index] = group_id

    group_to_split: dict[int, int] = {}
    for split_idx, indexes in enumerate(split_indexes):
        for index in indexes:
            group_to_split[index_to_group[index]] = split_idx

    def box_counts(split_idx: int) -> Counter[int]:
        counts: Counter[int] = Counter()
        for index in split_indexes[split_idx]:
            counts.update(box[0] for box in samples[index].boxes)
        return counts

    train_idx = 0
    train_boxes = box_counts(train_idx)
    all_boxes: Counter[int] = Counter()
    for split_idx in range(len(split_indexes)):
        all_boxes.update(box_counts(split_idx))

    for class_id in range(class_count):
        if all_boxes[class_id] == 0:
            continue
        if train_boxes[class_id] > 0:
            continue
        moved = False
        for donor_split in (2, 1):  # Prefer moving from test, then validation.
            for group_id, group in enumerate(groups):
                if group_to_split.get(group_id) != donor_split:
                    continue
                if not _group_has_class_boxes(group, samples, class_id):
                    continue
                group_set = set(group)
                split_indexes[donor_split] = [index for index in split_indexes[donor_split] if index not in group_set]
                split_indexes[train_idx].extend(group)
                group_to_split[group_id] = train_idx
                train_boxes = box_counts(train_idx)
                moved = True
                break
            if moved:
                break
    return split_indexes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, nargs="+", type=Path, help="One or more source YOLO data.yaml files")
    parser.add_argument("--class-map", required=True, type=Path, help="YAML mapping of source label aliases to canonical classes")
    parser.add_argument("--output", required=True, type=Path, help="New output directory; it must be absent or empty")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-hamming", type=int, default=4, help="Group images whose 64-bit difference hashes are within this Hamming distance; set 0 for exact-image grouping only")
    parser.add_argument("--min-images-per-class", type=int, default=0, help="Require this many independent duplicate groups per class in every requested split; 0 disables the guard")
    parser.add_argument("--allow-missing-labels-as-negative", action="store_true", help="Treat images without a label file as confirmed background negatives; use only when the source dataset intentionally omits empty labels")
    parser.add_argument("--unmapped-labels", choices=("error", "exclude-image"), default="error", help="For explicit class subsets, either fail on unmapped classes or exclude the entire affected image (never drop only its box)")
    split_policy = parser.add_mutually_exclusive_group()
    split_policy.add_argument("--preserve-source-test", dest="preserve_source_test", action="store_true", help="Keep source test images in output test; this is the default")
    split_policy.add_argument("--repartition-source-test", dest="preserve_source_test", action="store_false", help="Allow source test images to be reassigned across splits")
    parser.set_defaults(preserve_source_test=True)
    args = parser.parse_args()
    try:
        if not 0 <= args.duplicate_hamming <= 16:
            raise ValueError("--duplicate-hamming must be between 0 and 16")
        if args.min_images_per_class < 0:
            raise ValueError("--min-images-per-class cannot be negative")
        ratios = [float(value) for value in args.ratios]
        if any(not math.isfinite(value) or value <= 0 for value in ratios):
            raise ValueError("All three split ratios must be finite and positive so train, validation, and test are created.")
        ratio_sum = sum(ratios)
        if not math.isfinite(ratio_sum) or ratio_sum <= 0:
            raise ValueError("Split ratios must have a finite positive total.")
        ratios = [value / ratio_sum for value in ratios]
        configured_classes, aliases = load_class_map(args.class_map.expanduser().resolve())
        out = args.output.expanduser().resolve()
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ValueError(f"Output must be absent or empty: {out}")
        resolved_yamls = [path.expanduser().resolve() for path in args.data]
        observed_names: list[str] = []
        for yaml_path in resolved_yamls:
            source_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if not isinstance(source_config, dict):
                raise ValueError(f"Dataset YAML must be a mapping: {yaml_path}")
            observed_names.extend(class_names(source_config))
            source_root_value = source_config.get("path") or yaml_path.parent
            if not isinstance(source_root_value, (str, Path)):
                raise ValueError(f"Dataset YAML 'path' must be a directory path: {yaml_path}")
            source_root = Path(source_root_value)
            source_root = source_root.resolve() if source_root.is_absolute() else (yaml_path.parent / source_root).resolve()
            if out == source_root or source_root in out.parents:
                raise ValueError(f"Output directory cannot be inside source dataset root: {source_root}")
            for split, image_roots in resolve_split_paths(source_config, yaml_path).items():
                for image_root in image_roots:
                    if out == image_root or image_root in out.parents:
                        raise ValueError(f"Output directory cannot be inside configured {split} image directory: {image_root}")
        if configured_classes is None:
            canonical_by_key: dict[str, str] = {}
            for source_name in observed_names:
                canonical = aliases.get(normalize_name(source_name), source_name.strip())
                key = normalize_name(canonical)
                previous = canonical_by_key.get(key)
                if previous is None or (canonical.casefold(), canonical) < (previous.casefold(), previous):
                    canonical_by_key[key] = canonical
            classes = [canonical_by_key[key] for key in sorted(canonical_by_key)]
            # In automatic mode, labels without explicit aliases keep their
            # source class name. Populate the lookup used by read_boxes() with
            # those identity mappings as well as the configured aliases.
            for source_name in observed_names:
                source_key = normalize_name(source_name)
                canonical = aliases.get(source_key, source_name.strip())
                aliases.setdefault(source_key, canonical_by_key[normalize_name(canonical)])
        else:
            classes = configured_classes
        canonical_ids = {name: index for index, name in enumerate(classes)}
        if args.duplicate_hamming > 0 and cv2 is None:
            raise ValueError("Near-duplicate grouping requires OpenCV (cv2); install project requirements or set --duplicate-hamming 0.")
        samples, source_summary, excluded_images = collect_samples(
            resolved_yamls,
            aliases,
            canonical_ids,
            args.duplicate_hamming > 0,
            args.allow_missing_labels_as_negative,
            args.unmapped_labels == "exclude-image",
        )
        if not samples:
            raise ValueError("No images remain after applying the source and class-map rules.")
        groups, exact_duplicates = duplicate_groups(samples, args.duplicate_hamming)
        preserved_test_group_count = 0
        if args.preserve_source_test:
            fixed_test_groups = [group for group in groups if any(samples[index].source_split == "test" for index in group)]
            fixed_test_indexes = {index for group in fixed_test_groups for index in group}
            preserved_test_group_count = len(fixed_test_groups)
            if fixed_test_groups:
                trainval_groups = [group for group in groups if group[0] not in fixed_test_indexes]
                trainval_ratio = ratios[0] + ratios[1]
                trainval_split = stratified_group_split(
                    samples, trainval_groups, [ratios[0] / trainval_ratio, ratios[1] / trainval_ratio, 0.0], args.seed
                )
                split_indexes = [trainval_split[0], trainval_split[1], [index for group in fixed_test_groups for index in group]]
            else:
                split_indexes = stratified_group_split(samples, groups, ratios, args.seed)
        else:
            split_indexes = stratified_group_split(samples, groups, ratios, args.seed)
        split_indexes = ensure_train_class_coverage(samples, groups, split_indexes, len(classes))
        sample_group_ids = {}
        review_duplicate_groups = []
        for group_number, indexes in enumerate(groups, start=1):
            group_id = f"dup_{group_number:06d}"
            for index in indexes:
                sample_group_ids[index] = group_id
            if len(indexes) > 1:
                review_duplicate_groups.append({
                    "group_id": group_id,
                    "files": [str(samples[index].image) for index in indexes],
                    "sha256": [samples[index].sha256 for index in indexes],
                    "classes": sorted({classes[class_id] for index in indexes for class_id in samples[index].canonical_classes}),
                })

        split_class_images: dict[str, Counter[int]] = {}
        split_class_boxes: dict[str, Counter[int]] = {}
        split_class_groups: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
        index_to_split = {index: split for split, indexes in zip(SPLITS, split_indexes) for index in indexes}
        for group in groups:
            if not group:
                continue
            group_classes = {class_id for index in group for class_id in samples[index].canonical_classes}
            split_class_groups[index_to_split[group[0]]].update(group_classes)
        for split, indexes in zip(SPLITS, split_indexes):
            image_counts: Counter[int] = Counter()
            box_counts: Counter[int] = Counter()
            for index in indexes:
                image_counts.update(samples[index].canonical_classes)
                box_counts.update(box[0] for box in samples[index].boxes)
            split_class_images[split] = image_counts
            split_class_boxes[split] = box_counts
        missing_train_classes = [
            classes[class_id] for class_id in range(len(classes))
            if split_class_boxes["train"][class_id] == 0
        ]
        if missing_train_classes:
            raise ValueError(
                "These classes have no training boxes after splitting: "
                + ", ".join(missing_train_classes[:60])
                + ". Check their source labels or use --repartition-source-test if rare examples were held out in source test."
            )
        if sum(split_class_boxes["val"].values()) == 0:
            raise ValueError("The validation split has no logo boxes; provide positive validation examples or revise the split policy.")
        class_coverage_warnings = {
            split: [
                classes[class_id] for class_id in range(len(classes))
                if split_class_boxes[split][class_id] == 0
            ]
            for split in ("val", "test")
        }
        if args.min_images_per_class:
            low_support = []
            for split, ratio in zip(SPLITS, ratios):
                if ratio == 0:
                    continue
                for class_id, class_name in enumerate(classes):
                    count = split_class_groups[split][class_id]
                    if count < args.min_images_per_class:
                        low_support.append(f"{split}/{class_name}={count} independent groups")
            if low_support:
                raise ValueError("Per-class image minimum not met: " + ", ".join(low_support[:60]))

        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{out.name}.tmp-", dir=out.parent) as temp_dir:
            staging = Path(temp_dir) / "dataset"
            staging.mkdir()
            manifest_records = []
            for split, indexes in zip(SPLITS, split_indexes):
                for index in indexes:
                    sample = samples[index]
                    # Deterministic unique names, even when different sources reuse a filename.
                    token = hashlib.sha1(sample.source_key.encode()).hexdigest()[:12]
                    output_name = f"{token}_{sample.image.name}"
                    destination_image = staging / split / "images" / output_name
                    destination_label = staging / split / "labels" / f"{Path(output_name).stem}.txt"
                    destination_image.parent.mkdir(parents=True, exist_ok=True)
                    destination_label.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sample.image, destination_image)
                    rows = [f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}" for class_id, x, y, width, height in sample.boxes]
                    destination_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                    manifest_records.append({
                        "output_image": str(destination_image.relative_to(staging)),
                        "source_image": str(sample.image),
                        "source_label": str(sample.label),
                        "source_data_yaml": str(sample.source_yaml),
                        "source_key": sample.source_key,
                        "source_split": sample.source_split,
                        "output_split": split,
                        "sha256": sample.sha256,
                        "difference_hash": f"{sample.dhash:016x}",
                        "duplicate_group": sample_group_ids[index],
                        "canonical_classes": [classes[class_id] for class_id in sorted(sample.canonical_classes)],
                    })
            output_yaml = {"path": ".", "train": "train/images", "val": "val/images", "test": "test/images", "nc": len(classes), "names": classes}
            (staging / "data.yaml").write_text(yaml.safe_dump(output_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
            report = {
                "source_datasets": source_summary,
                "class_names": classes,
                "seed": args.seed,
                "split_ratios_requested": dict(zip(SPLITS, ratios)),
                "split_ratios_actual": {
                    split: len(indexes) / len(samples)
                    for split, indexes in zip(SPLITS, split_indexes)
                },
                "preserve_source_test": args.preserve_source_test,
                "source_test_groups_preserved": preserved_test_group_count,
                "class_coverage_warnings": class_coverage_warnings,
                "missing_labels_allowed_as_negative": args.allow_missing_labels_as_negative,
                "unmapped_label_policy": args.unmapped_labels,
                "duplicate_grouping": {
                    "method": "SHA-256 exact bytes" if args.duplicate_hamming == 0 else "64-bit difference hash connected components",
                    "maximum_hamming_distance": args.duplicate_hamming,
                },
                "images": len(samples),
                "near_duplicate_groups": sum(len(group) > 1 for group in groups),
                "duplicate_groups_for_review": review_duplicate_groups,
                "exact_duplicate_groups": exact_duplicates,
                "excluded_images": excluded_images,
                "splits": {
                    split: {
                        "images": len(indexes),
                        "objects": sum(len(samples[index].boxes) for index in indexes),
                        "class_counts": {
                            classes[class_id]: {
                                "images": split_class_images[split][class_id],
                                "independent_groups": split_class_groups[split][class_id],
                                "objects": split_class_boxes[split][class_id],
                            }
                            for class_id in range(len(classes))
                        },
                    }
                    for split, indexes in zip(SPLITS, split_indexes)
                },
                "manifest_file": "unification_manifest.json",
            }
            (staging / "unification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (staging / "unification_manifest.json").write_text(json.dumps(manifest_records, indent=2), encoding="utf-8")
            if out.exists():
                out.rmdir()  # only an empty directory was allowed above
            staging.rename(out)
        print(f"Unified {len(samples)} images into {out}")
        print("Splits: " + ", ".join(f"{split}={len(indexes)}" for split, indexes in zip(SPLITS, split_indexes)))
        print(f"Classes: {len(classes)} | near-duplicate groups: {report['near_duplicate_groups']} | exact duplicate groups: {len(exact_duplicates)}")
        print(f"YOLO dataset config: {out / 'data.yaml'}")
        print(f"Review report: {out / 'unification_report.json'}")
        return 0
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
