#!/usr/bin/env python3
"""Evaluate one or more YOLOv7 checkpoints on the configured test split."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLOV7_ROOT = PROJECT_ROOT / "yolov7"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(YOLOV7_ROOT))

from scripts.dataset_validation import IMAGE_SUFFIXES, validate_yolo_dataset  # noqa: E402
from utils.metrics import ap_per_class  # noqa: E402


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _load_names(data_yaml: Path) -> tuple[dict[int, str], dict[str, Any]]:
    with data_yaml.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping")
    raw_names = config.get("names", [])
    if isinstance(raw_names, dict):
        if any(not isinstance(value, str) or not value.strip() for value in raw_names.values()):
            raise ValueError("Dataset YAML class names must be non-empty strings")
        try:
            names = {int(key): value for key, value in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset YAML 'names' mapping keys must be numeric class IDs") from exc
        if len(names) != len(raw_names):
            raise ValueError("Dataset YAML 'names' contains duplicate numeric class IDs")
        if sorted(names) != list(range(len(names))):
            raise ValueError("Dataset YAML 'names' IDs must be contiguous and start at 0")
    elif isinstance(raw_names, list):
        if any(not isinstance(value, str) or not value.strip() for value in raw_names):
            raise ValueError("Dataset YAML class names must be non-empty strings")
        names = {index: value for index, value in enumerate(raw_names)}
    else:
        raise ValueError("Dataset YAML 'names' must be a list or mapping")
    if any(not isinstance(value, str) or not value.strip() for value in names.values()):
        raise ValueError("Dataset YAML class names must be non-empty strings")
    if "nc" in config and int(config["nc"]) != len(names):
        raise ValueError(f"Dataset YAML declares nc={config['nc']} but provides {len(names)} class names")
    return names, config


def _checkpoint_class_metadata(checkpoint: Path) -> tuple[list[str] | None, int | None]:
    """Read class names/count embedded in a YOLOv7 checkpoint when available."""
    import torch

    try:
        try:
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        except TypeError:  # older PyTorch has no weights_only parameter
            payload = torch.load(str(checkpoint), map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Cannot inspect checkpoint class metadata in {checkpoint}: {exc}") from exc

    models = []
    if isinstance(payload, dict):
        models.extend(payload.get(key) for key in ("ema", "model") if payload.get(key) is not None)
    else:
        models.append(payload)
    for model in models:
        raw_names = getattr(model, "names", None)
        if isinstance(raw_names, dict):
            try:
                names = [str(raw_names[key]) for key in sorted(raw_names, key=lambda value: int(value))]
            except (TypeError, ValueError):
                names = None
        elif isinstance(raw_names, (list, tuple)):
            names = [str(value) for value in raw_names]
        else:
            names = None
        model_yaml = getattr(model, "yaml", None)
        model_nc = model_yaml.get("nc") if isinstance(model_yaml, dict) else getattr(model, "nc", None)
        if names:
            return names, len(names)
        if model_nc is not None:
            try:
                return None, int(model_nc)
            except (TypeError, ValueError):
                pass
    if isinstance(payload, dict) and payload.get("nc") is not None:
        try:
            return None, int(payload["nc"])
        except (TypeError, ValueError):
            pass
    return None, None


def _image_files(roots: list[str]) -> list[tuple[Path, Path]]:
    """Return (image_root, image_path) pairs, preserving symlink paths for labels."""
    result = []
    for root_string in roots:
        root = Path(root_string)
        if root.is_dir():
            result.extend((root, path) for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(result, key=lambda pair: str(pair[1]))


def _label_root(image_root: Path) -> Path:
    return image_root.parent / "labels" if image_root.name.lower() == "images" else image_root / "labels"


def _read_boxes(path: Path, prediction: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) not in ((6,) if prediction else (5,)):
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, width, height = map(float, parts[1:5])
            confidence = float(parts[5]) if prediction else 1.0
        except ValueError:
            continue
        if not np.isfinite([x, y, width, height, confidence]).all() or width <= 0 or height <= 0:
            continue
        boxes.append({"class_id": class_id, "box": (x, y, width, height), "confidence": confidence})
    return boxes


def _iou_xywh(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left, top = max(ax - aw / 2, bx - bw / 2), max(ay - ah / 2, by - bh / 2)
    right, bottom = min(ax + aw / 2, bx + bw / 2), min(ay + ah / 2, by + bh / 2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _match_predictions(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    threshold: float = 0.5,
) -> tuple[set[int], set[int]]:
    """Greedily match confidence-sorted predictions to same-class targets."""
    matched_targets: set[int] = set()
    true_predictions: set[int] = set()
    ordered = sorted(range(len(predictions)), key=lambda index: predictions[index]["confidence"], reverse=True)
    for prediction_index in ordered:
        prediction = predictions[prediction_index]
        candidates = [
            (target_index, _iou_xywh(prediction["box"], target["box"]))
            for target_index, target in enumerate(targets)
            if target["class_id"] == prediction["class_id"] and target_index not in matched_targets
        ]
        if not candidates:
            continue
        target_index, overlap = max(candidates, key=lambda item: item[1])
        if overlap >= threshold:
            matched_targets.add(target_index)
            true_predictions.add(prediction_index)
    return true_predictions, matched_targets


def _ap_statistics(
    samples: list[dict[str, Any]],
    class_names: dict[int, str],
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    all_correct, all_confidence, all_prediction_classes, all_target_classes = [], [], [], []
    for sample in samples:
        predictions, targets = sample["predictions"], sample["targets"]
        correct = np.zeros((len(predictions), len(IOU_THRESHOLDS)), dtype=bool)
        for class_id in {item["class_id"] for item in predictions}:
            prediction_indices = [i for i, item in enumerate(predictions) if item["class_id"] == class_id]
            target_indices = [i for i, item in enumerate(targets) if item["class_id"] == class_id]
            used_targets: set[int] = set()
            prediction_indices.sort(key=lambda i: predictions[i]["confidence"], reverse=True)
            for prediction_index in prediction_indices:
                candidates = [
                    (target_index, _iou_xywh(predictions[prediction_index]["box"], targets[target_index]["box"]))
                    for target_index in target_indices if target_index not in used_targets
                ]
                if not candidates:
                    continue
                target_index, overlap = max(candidates, key=lambda item: item[1])
                if overlap > IOU_THRESHOLDS[0]:
                    used_targets.add(target_index)
                    correct[prediction_index] = overlap > IOU_THRESHOLDS
        all_correct.append(correct)
        all_confidence.extend(item["confidence"] for item in predictions)
        all_prediction_classes.extend(item["class_id"] for item in predictions)
        all_target_classes.extend(item["class_id"] for item in targets)

    tp = np.concatenate(all_correct, axis=0) if all_correct else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
    confidence = np.asarray(all_confidence, dtype=np.float32)
    prediction_classes = np.asarray(all_prediction_classes, dtype=np.float32)
    target_classes = np.asarray(all_target_classes, dtype=np.float32)
    if len(target_classes):
        precision, recall, ap, _, ap_classes = ap_per_class(tp, confidence, prediction_classes, target_classes)
    else:
        precision, recall, ap, ap_classes = np.asarray([]), np.asarray([]), np.zeros((0, len(IOU_THRESHOLDS))), np.asarray([], dtype=int)

    by_class: dict[int, dict[str, float]] = {}
    for index, class_id in enumerate(ap_classes):
        by_class[int(class_id)] = {
            "precision_at_global_best_f1": float(precision[index]),
            "recall_at_global_best_f1": float(recall[index]),
            "ap50": float(ap[index, 0]),
            "ap50_95": float(ap[index].mean()),
        }
    aggregate = {
        "mAP50": float(ap[:, 0].mean()) if ap.size else 0.0,
        "mAP50_95": float(ap.mean()) if ap.size else 0.0,
        "classes_with_test_targets": int(len(ap_classes)),
    }
    return by_class, aggregate


def _save_box_example(image_path: Path, box: tuple[float, ...], text: str, output_path: Path, color: tuple[int, int, int]) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    x, y, box_width, box_height = box
    x1 = max(0, min(width - 1, round((x - box_width / 2) * width)))
    y1 = max(0, min(height - 1, round((y - box_height / 2) * height)))
    x2 = max(0, min(width - 1, round((x + box_width / 2) * width)))
    y2 = max(0, min(height - 1, round((y + box_height / 2) * height)))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_name(checkpoint: Path, output_dir: Path) -> str:
    run_component = checkpoint.parent.parent.name if checkpoint.parent.name == "weights" else checkpoint.parent.name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{run_component}_{checkpoint.stem}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{safe}_{timestamp}"


def evaluate_checkpoint(
    checkpoint: Path,
    data_yaml: Path,
    output_dir: Path,
    device: str = "0",
    img_size: int = 640,
    batch_size: int = 16,
    inference_conf: float = 0.001,
    report_conf: float = 0.25,
    nms_iou: float = 0.6,
    match_iou: float = 0.5,
    max_examples: int = 100,
    allow_unverified_class_mapping: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    data_yaml = data_yaml.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    class_names, data_config = _load_names(data_yaml)
    checkpoint_names, checkpoint_class_count = _checkpoint_class_metadata(checkpoint)
    dataset_name_list = [class_names[index] for index in sorted(class_names)]
    if checkpoint_class_count is not None and checkpoint_class_count != len(dataset_name_list):
        raise ValueError(
            f"Checkpoint has {checkpoint_class_count} classes but data.yaml has {len(dataset_name_list)}. "
            "Use the matching dataset YAML/checkpoint pair."
        )
    if checkpoint_names is not None and checkpoint_names != dataset_name_list:
        mismatch = next(
            (index for index, (checkpoint_name, data_name) in enumerate(zip(checkpoint_names, dataset_name_list)) if checkpoint_name != data_name),
            min(len(checkpoint_names), len(dataset_name_list)),
        )
        checkpoint_name = checkpoint_names[mismatch] if mismatch < len(checkpoint_names) else "<missing>"
        dataset_name = dataset_name_list[mismatch] if mismatch < len(dataset_name_list) else "<missing>"
        raise ValueError(
            f"Checkpoint/data class order mismatch at class ID {mismatch}: checkpoint={checkpoint_name!r}, "
            f"data.yaml={dataset_name!r}. Evaluate with the exact class mapping used for training."
        )
    class_mapping_verified = checkpoint_names is not None
    if not class_mapping_verified and not allow_unverified_class_mapping:
        raise ValueError(
            "Checkpoint does not expose embedded class names, so class order cannot be verified. "
            "Use --allow-unverified-class-mapping only after confirming the checkpoint and data.yaml came from the same training taxonomy."
        )
    if not class_mapping_verified:
        print("WARNING: checkpoint class count may match, but its class-name order could not be verified.")
    validation = validate_yolo_dataset(data_yaml)
    test_roots = validation.get("resolved_paths", {}).get("test", [])
    if not test_roots:
        raise ValueError("Dataset YAML must define a test split")
    images = _image_files(test_roots)
    if not images:
        raise ValueError("Test split contains no readable image paths")
    if validation.get("splits", {}).get("test", {}).get("missing_label_files", 0):
        print("WARNING: Some test images have no label file; recall and AP may be understated or misleading.")
    if validation.get("errors"):
        print(f"WARNING: Dataset preflight found {len(validation['errors'])} issue(s). Evaluation will use the images and labels listed in the test split.")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = _run_name(checkpoint, output_dir)
    run_dir = output_dir / run_name
    while run_dir.exists():
        run_name = f"{run_name}_run"
        run_dir = output_dir / run_name

    normalized_config = dict(data_config)
    for split, paths in validation.get("resolved_paths", {}).items():
        if split in normalized_config:
            normalized_config[split] = paths if isinstance(normalized_config[split], list) else paths[0]
    temp_yaml_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False) as stream:
            yaml.safe_dump(normalized_config, stream, sort_keys=False)
            temp_yaml_path = Path(stream.name)
        command = [
            sys.executable,
            str(YOLOV7_ROOT / "test.py"),
            "--weights", str(checkpoint),
            "--data", str(temp_yaml_path),
            "--task", "test",
            "--batch-size", str(batch_size),
            "--img-size", str(img_size),
            "--conf-thres", str(inference_conf),
            "--iou-thres", str(nms_iou),
            "--device", device,
            "--verbose",
            "--save-txt",
            "--save-conf",
            "--no-trace",
            "--no-half",
            "--project", str(output_dir),
            "--name", run_name,
        ]
        print(f"\nEvaluating {checkpoint.name} on {len(images)} test images")
        result = subprocess.run(command, cwd=str(YOLOV7_ROOT))
        if result.returncode != 0:
            raise RuntimeError(f"YOLOv7 test.py exited with status {result.returncode}")
        if not run_dir.is_dir():
            raise RuntimeError(f"YOLOv7 did not create expected output directory: {run_dir}")
    finally:
        if temp_yaml_path is not None:
            temp_yaml_path.unlink(missing_ok=True)

    samples = []
    false_positive_rows = []
    missed_rows = []
    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "ground_truth": 0, "predictions_at_report_conf": 0})
    example_number = {"fp": 0, "miss": 0}

    for image_root, image_path in images:
        labels_root = _label_root(image_root)
        relative_image = image_path.relative_to(image_root)
        truth_path = labels_root / relative_image.with_suffix(".txt")
        prediction_path = run_dir / "labels" / f"{image_path.stem}.txt"
        targets = _read_boxes(truth_path)
        predictions = _read_boxes(prediction_path, prediction=True)
        samples.append({"image": image_path, "targets": targets, "predictions": predictions})
        for target in targets:
            counts[target["class_id"]]["ground_truth"] += 1

        report_predictions = [item for item in predictions if item["confidence"] >= report_conf]
        for prediction in report_predictions:
            counts[prediction["class_id"]]["predictions_at_report_conf"] += 1

        true_indices: set[int] = set()
        matched_targets: set[int] = set()
        for class_id in {item["class_id"] for item in report_predictions} | {item["class_id"] for item in targets}:
            pred_indices = [i for i, item in enumerate(report_predictions) if item["class_id"] == class_id]
            target_indices = [i for i, item in enumerate(targets) if item["class_id"] == class_id]
            sub_predictions = [report_predictions[i] for i in pred_indices]
            sub_targets = [targets[i] for i in target_indices]
            local_tp, local_target_matches = _match_predictions(sub_predictions, sub_targets, match_iou)
            true_indices.update(pred_indices[i] for i in local_tp)
            matched_targets.update(target_indices[i] for i in local_target_matches)

        for prediction_index, prediction in enumerate(report_predictions):
            class_id = prediction["class_id"]
            class_name = class_names.get(class_id, f"class_{class_id}")
            if prediction_index in true_indices:
                counts[class_id]["tp"] += 1
                continue
            counts[class_id]["fp"] += 1
            row = {
                "image": str(image_path), "class_id": class_id, "class_name": class_name,
                "confidence": prediction["confidence"], "bbox_xywh_normalized": " ".join(map(str, prediction["box"])),
                "example_image": "",
            }
            if example_number["fp"] < max_examples:
                example_number["fp"] += 1
                example_path = run_dir / "examples" / "false_positives" / f"{image_path.stem}_fp_{example_number['fp']:04d}.jpg"
                if _save_box_example(image_path, prediction["box"], f"FP {class_name} {prediction['confidence']:.2f}", example_path, (0, 0, 255)):
                    row["example_image"] = str(example_path)
            false_positive_rows.append(row)

        for target_index, target in enumerate(targets):
            if target_index in matched_targets:
                continue
            class_id = target["class_id"]
            class_name = class_names.get(class_id, f"class_{class_id}")
            counts[class_id]["fn"] += 1
            row = {
                "image": str(image_path), "class_id": class_id, "class_name": class_name,
                "bbox_xywh_normalized": " ".join(map(str, target["box"])), "example_image": "",
            }
            if example_number["miss"] < max_examples:
                example_number["miss"] += 1
                example_path = run_dir / "examples" / "missed_logos" / f"{image_path.stem}_miss_{example_number['miss']:04d}.jpg"
                if _save_box_example(image_path, target["box"], f"MISSED {class_name}", example_path, (0, 165, 255)):
                    row["example_image"] = str(example_path)
            missed_rows.append(row)

    per_class_ap, aggregate_ap = _ap_statistics(samples, class_names)
    per_class_rows = []
    for class_id in sorted(set(class_names) | set(counts) | set(per_class_ap)):
        values = counts[class_id]
        precision = values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else 0.0
        recall = values["tp"] / values["ground_truth"] if values["ground_truth"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_rows.append({
            "class_id": class_id,
            "class_name": class_names.get(class_id, f"class_{class_id}"),
            **values,
            "precision_at_report_conf": precision,
            "recall_at_report_conf": recall,
            "f1_at_report_conf": f1,
            **per_class_ap.get(class_id, {"precision_at_global_best_f1": 0.0, "recall_at_global_best_f1": 0.0, "ap50": 0.0, "ap50_95": 0.0}),
        })

    total_tp = sum(item["tp"] for item in counts.values())
    total_fp = sum(item["fp"] for item in counts.values())
    total_fn = sum(item["fn"] for item in counts.values())
    total_gt = sum(item["ground_truth"] for item in counts.values())
    total_predictions = sum(item["predictions_at_report_conf"] for item in counts.values())
    overall_precision = total_tp / total_predictions if total_predictions else 0.0
    overall_recall = total_tp / total_gt if total_gt else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall else 0.0

    report = {
        "checkpoint": str(checkpoint),
        "dataset_yaml": str(data_yaml),
        "checkpoint_class_mapping_verified": class_mapping_verified,
        "test_image_count": len(images),
        "confidence_threshold_for_error_lists": report_conf,
        "inference_min_confidence": inference_conf,
        "matching_iou_threshold": match_iou,
        "nms_iou_threshold": nms_iou,
        "aggregate": {
            **aggregate_ap,
            "precision_at_report_conf": overall_precision,
            "recall_at_report_conf": overall_recall,
            "f1_at_report_conf": overall_f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "missed_logos": total_fn,
            "ground_truth_logos": total_gt,
            "predictions_at_report_conf": total_predictions,
        },
        "per_class": per_class_rows,
        "false_positive_count": len(false_positive_rows),
        "missed_logo_count": len(missed_rows),
        "examples_saved": {"false_positives": example_number["fp"], "missed_logos": example_number["miss"]},
        "dataset_preflight": {
            "error_count": len(validation.get("errors", [])),
            "warning_count": len(validation.get("warnings", [])),
            "errors_sample": validation.get("errors", [])[:100],
            "warnings": validation.get("warnings", []),
        },
        "output_dir": str(run_dir),
    }

    (run_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(run_dir / "per_class_metrics.csv", per_class_rows)
    _write_csv(run_dir / "false_positives.csv", false_positive_rows)
    _write_csv(run_dir / "missed_logos.csv", missed_rows)
    print(f"\n{checkpoint.name}: mAP50={aggregate_ap['mAP50']:.4f}, mAP50-95={aggregate_ap['mAP50_95']:.4f}")
    print(f"  At confidence >= {report_conf:g}, IoU >= {match_iou:g}: precision={overall_precision:.4f}, recall={overall_recall:.4f}, F1={overall_f1:.4f}")
    print(f"  False positives: {len(false_positive_rows)}; missed logos: {len(missed_rows)}")
    print(f"  Report: {run_dir}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved YOLOv7 checkpoints on a dataset's test split.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML with a test: split")
    parser.add_argument("--weights", type=Path, nargs="+", required=True, help="One or more saved YOLOv7 .pt checkpoints")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "checkpoint_evaluation", help="Output directory for test runs and error examples")
    parser.add_argument("--device", default="0", help="CUDA device (0, 1, ...) or cpu")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-conf", type=float, default=0.001, help="Low cutoff for retaining detections used by AP computation")
    parser.add_argument("--report-conf", type=float, default=0.25, help="Confidence cutoff for FP/FN and fixed-threshold precision/recall")
    parser.add_argument("--nms-iou", type=float, default=0.6, help="NMS IoU threshold used by YOLOv7")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for TP/FP/FN matching")
    parser.add_argument("--max-examples", type=int, default=100, help="Maximum annotated examples of each error type per checkpoint")
    parser.add_argument("--allow-unverified-class-mapping", action="store_true", help="Proceed when a checkpoint has no embedded class names; use only after manually confirming its class order matches data.yaml")
    args = parser.parse_args()
    if not 0 <= args.inference_conf <= args.report_conf <= 1:
        parser.error("Require 0 <= --inference-conf <= --report-conf <= 1")
    if not 0 <= args.match_iou <= 1 or not 0 <= args.nms_iou <= 1:
        parser.error("IoU thresholds must be between 0 and 1")
    if args.img_size <= 0 or args.batch_size <= 0 or args.max_examples < 0:
        parser.error("--img-size and --batch-size must be positive; --max-examples cannot be negative")

    failures = []
    for checkpoint in args.weights:
        try:
            evaluate_checkpoint(
                checkpoint, args.data, args.output, args.device, args.img_size,
                args.batch_size, args.inference_conf, args.report_conf,
                args.nms_iou, args.match_iou, args.max_examples,
                args.allow_unverified_class_mapping,
            )
        except Exception as exc:
            failures.append((checkpoint, str(exc)))
            print(f"ERROR evaluating {checkpoint}: {exc}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} checkpoint(s) failed; successful checkpoint reports remain saved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
