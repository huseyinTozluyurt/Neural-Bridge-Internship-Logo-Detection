#!/usr/bin/env python3
"""Prepare the OpenLogo + FlickrLogos32 dataset and optionally train YOLOv7.

Run from any working directory with the project's Python environment:

    python scripts/prepare_train_multiclass.py --stage prepare
    python scripts/prepare_train_multiclass.py --stage train --device 0

The prepare stage uses the checked-in portable YAMLs and brand alias map,
creates a duplicate-aware 80/10/10 dataset, and audits the result. The train
stage validates that dataset again and launches the existing YOLOv7 wrapper.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = PROJECT_ROOT / "scripts"
OPENLOGO_YAML = PROJECT_ROOT / "datasets" / "unification_inputs" / "openlogo.v1i.yolov7pytorch_data.yaml"
FLICKR_YAML = PROJECT_ROOT / "datasets" / "unification_inputs" / "flickrlogos32_data.yaml"
CLASS_MAP = SCRIPT_DIR / "logo_class_map_openlogo_flickr.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "logos_yolo_openlogo_flickr_multiclass"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "dataset_audits" / "openlogo_flickr_multiclass"


def run_step(description: str, command: Sequence[str]) -> bool:
    print(f"\n=== {description} ===", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode:
        print(f"Step failed with exit code {result.returncode}: {description}", file=sys.stderr)
        return False
    return True


def read_class_count(data_yaml: Path) -> int:
    try:
        config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read dataset YAML {data_yaml}: {exc}") from exc
    names = config.get("names") if isinstance(config, dict) else None
    if isinstance(names, list):
        count = len(names)
    elif isinstance(names, dict):
        count = len(names)
    else:
        raise ValueError(f"Dataset YAML must define class names as a list or mapping: {data_yaml}")
    if count < 2:
        raise ValueError(f"Expected a multi-class dataset, but found {count} class(es): {data_yaml}")
    declared = config.get("nc")
    if declared is not None and int(declared) != count:
        raise ValueError(f"Dataset YAML declares nc={declared}, but lists {count} class names: {data_yaml}")
    return count


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "prepare", "train"), default="all",
                        help="Prepare and train, or run one stage only (default: all)")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET,
                        help="Unified dataset directory (default: datasets/logos_yolo_openlogo_flickr_multiclass)")
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT,
                        help="Location for the dataset audit reports and previews")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument("--duplicate-hamming", type=int, default=4,
                        help="Group near-duplicate images up to this 64-bit dHash distance; 0 uses exact hashes")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Starting batch size; reduce it if CUDA reports out-of-memory")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0", help="YOLOv7 device id (for example 0) or cpu")
    parser.add_argument("--name", default="openlogo_flickr_multiclass_v1",
                        help="YOLOv7 run name under runs/train")
    parser.add_argument("--preview-count", type=int, default=24,
                        help="Annotated examples per split to save during the audit; 0 disables previews")
    parser.add_argument("--repartition-source-test", action="store_true",
                        help="Re-split all source data; by default source test images remain held out")
    args = parser.parse_args()

    output = args.output.expanduser()
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    else:
        output = output.resolve()
    audit_output = args.audit_output.expanduser()
    if not audit_output.is_absolute():
        audit_output = (PROJECT_ROOT / audit_output).resolve()
    else:
        audit_output = audit_output.resolve()
    data_yaml = output / "data.yaml"
    python = sys.executable

    if args.stage in ("all", "prepare"):
        for required in (OPENLOGO_YAML, FLICKR_YAML, CLASS_MAP):
            if not required.is_file():
                print(f"Required input is missing: {required}", file=sys.stderr)
                return 2

        unify_command = [
            python, str(SCRIPT_DIR / "unify_logo_datasets.py"),
            "--data", str(OPENLOGO_YAML), str(FLICKR_YAML),
            "--class-map", str(CLASS_MAP),
            "--output", str(output),
            "--ratios", "0.8", "0.1", "0.1",
            "--seed", str(args.seed),
            "--duplicate-hamming", str(args.duplicate_hamming),
        ]
        if not args.repartition_source_test:
            unify_command.append("--preserve-source-test")
        else:
            unify_command.append("--repartition-source-test")
        if not run_step("Unify OpenLogo + FlickrLogos32", unify_command):
            return 2

    if args.stage in ("all", "prepare", "train"):
        if not data_yaml.is_file():
            print(f"Unified dataset is missing: {data_yaml}. Run with --stage prepare first.", file=sys.stderr)
            return 2
        audit_command = [
            python, str(SCRIPT_DIR / "audit_dataset.py"),
            "--data", str(data_yaml),
            "--output", str(audit_output),
            "--preview-count", str(args.preview_count),
        ]
        if not run_step("Audit unified dataset", audit_command):
            return 2

    if args.stage == "prepare":
        print(f"\nPreparation complete. Dataset YAML: {data_yaml}")
        print(f"Audit reports: {audit_output}")
        return 0

    if args.stage in ("all", "train"):
        try:
            class_count = read_class_count(data_yaml)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        if args.device.casefold() != "cpu" and not cuda_available():
            print(
                "CUDA was requested but this Python environment cannot see a GPU. "
                "The dataset is prepared and audited; install/fix the NVIDIA driver and "
                "a CUDA-enabled PyTorch environment, then rerun with --stage train. "
                "Use --device cpu only for a deliberately small smoke run.",
                file=sys.stderr,
            )
            return 3

        train_command = [
            python, str(SCRIPT_DIR / "train_yolo.py"),
            "--data", str(data_yaml),
            "--weights", "yolov7/yolov7.pt",
            "--device", args.device,
            "--img-size", str(args.img_size),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--epochs", str(args.epochs),
            "--expect-classes", str(class_count),
            "--project", "runs/train",
            "--name", args.name,
        ]
        if not run_step(f"Train YOLOv7 with {class_count} brand classes", train_command):
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
