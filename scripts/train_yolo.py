#!/usr/bin/env python3
"""
YOLOv7 Logo Detection Training Script
Unified wrapper for training YOLOv7 on logo datasets
"""

import os
import sys
import argparse
from pathlib import Path
import tempfile
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "yolov7"))
from scripts.dataset_validation import validate_yolo_dataset, print_validation_report


def train_yolov7(
    data_yaml,
    epochs=100,
    batch_size=16,
    img_size=640,
    weights="yolov7/yolov7.pt",
    cfg="yolov7/cfg/training/yolov7.yaml",
    hyp="yolov7/data/hyp.scratch.p5.yaml",
    project="runs/train",
    name="logo_training",
    device="0",
    workers=4,
    single_cls=False,
    expect_classes=None,
    min_val_objects_per_class=0,
    exist_ok=False,
    resume=False
):
    """
    Train YOLOv7 model on logo dataset

    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        weights: Path to pre-trained weights
        cfg: Path to model configuration
        hyp: Path to hyperparameters file
        project: Project directory
        name: Run name
        device: CUDA device (0, 1, 2, etc or 'cpu')
        workers: Number of data loading workers
        single_cls: Train as single-class dataset
        exist_ok: Allow existing project/name
        resume: Resume from last checkpoint
    """

    print("=" * 75)
    print("YOLOV7 LOGO DETECTION TRAINING")
    print("=" * 75)
    print()
    print(f"Configuration:")
    print(f"  Data: {data_yaml}")
    print(f"  Weights: {weights}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Single class: {single_cls}")
    print(f"  Device: {device}")
    print(f"  Output: {project}/{name}")
    print()

    def project_path(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else project_root / path).resolve()

    data_path = project_path(data_yaml)
    weights_path = project_path(weights)
    cfg_path = project_path(cfg)
    hyp_path = project_path(hyp)
    project_path_out = project_path(project)

    # Verify data.yaml exists
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        return False

    with data_path.open(encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream)
    class_names = data_config.get("names", []) if isinstance(data_config, dict) else []
    if isinstance(class_names, dict):
        class_count = len(class_names)
    elif isinstance(class_names, list):
        class_count = len(class_names)
    else:
        print("❌ Dataset YAML must provide class names as a list or mapping.")
        return False
    if expect_classes is not None and class_count != expect_classes:
        print(f"❌ Expected {expect_classes} classes but this dataset defines {class_count}: {class_names}")
        return False
    if single_cls and expect_classes and expect_classes > 1:
        print("❌ --single-cls collapses all brand IDs into one class and conflicts with --expect-classes > 1.")
        return False
    if class_count == 1 and not single_cls:
        print("⚠ This dataset defines one class only. The model will learn generic logo detection, not brand identities.")
    if single_cls:
        print("⚠ --single-cls is enabled: all logo brands are intentionally collapsed to class 0.")

    print("Checking dataset structure and annotations...")
    dataset_report = validate_yolo_dataset(data_path)
    if not print_validation_report(dataset_report):
        return False
    if min_val_objects_per_class > 0:
        raw_names = data_config.get("names", [])
        names = [str(raw_names[k]) for k in sorted(raw_names, key=lambda value: int(value))] if isinstance(raw_names, dict) else [str(name) for name in raw_names]
        validation_counts = dataset_report.get("class_object_counts", {}).get("val", {})
        insufficient = [
            f"{name} ({validation_counts.get(str(class_id), 0)})"
            for class_id, name in enumerate(names)
            if validation_counts.get(str(class_id), 0) < min_val_objects_per_class
        ]
        if insufficient:
            print(
                f"❌ Validation coverage is below --min-val-objects-per-class={min_val_objects_per_class}: "
                + ", ".join(insufficient)
            )
            print("Rebalance train/validation annotations before training; keep the held-out test split untouched.")
            return False
    # Verify weights exist
    if not weights_path.exists():
        print(f"❌ Error: Weights file not found: {weights_path}")
        print(f"   Download from: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
        return False
    if not cfg_path.is_file():
        print(f"❌ Error: Model config not found: {cfg_path}")
        return False
    if not hyp_path.is_file():
        print(f"❌ Error: Hyperparameter config not found: {hyp_path}")
        return False

    # This YOLOv7 checkout reads split paths relative to its working directory
    # and ignores YAML's `path:` key. Pass it an ephemeral YAML with absolute
    # split paths so datasets work regardless of the caller's current folder.
    with data_path.open(encoding="utf-8") as stream:
        normalized_data = yaml.safe_load(stream)
    for split, paths in dataset_report["resolved_paths"].items():
        normalized_data[split] = paths if isinstance(normalized_data[split], list) else paths[0]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False) as stream:
        yaml.safe_dump(normalized_data, stream, sort_keys=False)
        normalized_data_path = Path(stream.name)

    print("✓ All files validated")
    print()

    # Build command
    train_script = project_root / "yolov7" / "train.py"

    cmd_args = [
        sys.executable,
        str(train_script),
        f"--data={normalized_data_path}",
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        f"--img-size={img_size}",
        f"--weights={weights_path}",
        f"--cfg={cfg_path}",
        f"--hyp={hyp_path}",
        f"--project={project_path_out}",
        f"--name={name}",
        f"--device={device}",
        f"--workers={workers}",
    ]

    if single_cls:
        cmd_args.append("--single-cls")

    if exist_ok:
        cmd_args.append("--exist-ok")

    if resume:
        if isinstance(resume, str):
            cmd_args.append(f"--resume={resume}")
        else:
            cmd_args.append("--resume")

    print("=" * 75)
    print("STARTING TRAINING")
    print("=" * 75)
    print()
    print(f"Command: {' '.join(cmd_args)}")
    print()

    # Run training
    import subprocess
    try:
        result = subprocess.run(cmd_args, cwd=str(project_root))

        if result.returncode == 0:
            print()
            print("=" * 75)
            print("✅ TRAINING COMPLETE!")
            print("=" * 75)
            print()
            print(f"Results saved to: {project}/{name}")
            print()
            print("View results:")
            print(f"  Best weights: {project}/{name}/weights/best.pt")
            print(f"  Last weights: {project}/{name}/weights/last.pt")
            print(f"  Results: {project}/{name}/results.txt")
            print()
            print("Next steps:")
            print("  1. Run inference: python scripts/detect_yolo.py --weights {project}/{name}/weights/best.pt --source <image_path>")
            print("  2. Test on validation: python yolov7/test.py --data {data_yaml} --weights {project}/{name}/weights/best.pt")
            return True
        else:
            print()
            print("❌ Training failed")
            return False

    except KeyboardInterrupt:
        print()
        print("⚠ Training interrupted by user")
        return False
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        normalized_data_path.unlink(missing_ok=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train YOLOv7 on logo detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on OpenLogo dataset
  python scripts/train_yolo.py --data datasets/openlogo_yolo/data.yaml --epochs 100

  # Quick test
  python scripts/train_yolo.py --data datasets/openlogo_yolo/data.yaml --epochs 5 --batch-size 8

  # Single class detection
  python scripts/train_yolo.py --data datasets/openlogo_yolo/data.yaml --single-cls

  # Resume training
  python scripts/train_yolo.py --data datasets/openlogo_yolo/data.yaml --resume

  # Custom settings
  python scripts/train_yolo.py --data datasets/openlogo_yolo/data.yaml --epochs 50 --batch-size 32 --img-size 800
        """
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml file"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="yolov7/yolov7.pt",
        help="Path to pre-trained weights (default: yolov7/yolov7.pt)"
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="yolov7/cfg/training/yolov7.yaml",
        help="Path to model configuration"
    )

    parser.add_argument(
        "--hyp",
        type=str,
        default="yolov7/data/hyp.scratch.p5.yaml",
        help="Path to hyperparameters file"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory (default: runs/train)"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="logo_training",
        help="Run name (default: logo_training)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device (0, 1, 2, etc or 'cpu') (default: 0)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )

    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="Train as single-class dataset"
    )

    parser.add_argument(
        "--expect-classes",
        type=int,
        default=None,
        help="Abort unless data.yaml defines exactly this many classes (for FlickrLogos-32 use 32)",
    )

    parser.add_argument(
        "--min-val-objects-per-class",
        type=int,
        default=0,
        help="Abort if any class has fewer validation objects than this value",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing project/name"
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume from the last checkpoint, or from a specific checkpoint path if given"
    )

    args = parser.parse_args()

    if args.expect_classes is not None and args.expect_classes <= 0:
        parser.error("--expect-classes must be positive")
    if args.min_val_objects_per_class < 0:
        parser.error("--min-val-objects-per-class cannot be negative")

    # Train the model
    success = train_yolov7(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        weights=args.weights,
        cfg=args.cfg,
        hyp=args.hyp,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        single_cls=args.single_cls,
        expect_classes=args.expect_classes,
        min_val_objects_per_class=args.min_val_objects_per_class,
        exist_ok=args.exist_ok,
        resume=args.resume
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
