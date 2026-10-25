#!/usr/bin/env python3
"""
YOLOv7 Logo Detection Inference Script
Unified wrapper for running detection with trained YOLOv7 models
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "yolov7"))


def detect_yolov7(
    weights,
    source,
    conf_thres=0.4,
    iou_thres=0.45,
    img_size=640,
    project="runs/detect",
    name="detection",
    device="0",
    save_txt=True,
    save_conf=True,
    nosave=False,
    classes=None,
    view_img=False,
    exist_ok=False
):
    """
    Run detection with YOLOv7 model

    Args:
        weights: Path to model weights
        source: Input source (image, folder, video, webcam)
        conf_thres: Confidence threshold (default: 0.4)
        iou_thres: IOU threshold for NMS (default: 0.45)
        img_size: Inference size (default: 640)
        project: Project directory (default: runs/detect)
        name: Run name (default: detection)
        device: CUDA device (0, 1, 2, etc or 'cpu')
        save_txt: Save results to *.txt
        save_conf: Save confidences in --save-txt labels
        nosave: Do not save images/videos
        classes: Filter by class (e.g., 0 or 0 2 3)
        view_img: Display results
        exist_ok: Allow existing project/name
    """

    print("=" * 75)
    print("YOLOV7 LOGO DETECTION")
    print("=" * 75)
    print()
    print(f"Configuration:")
    print(f"  Weights: {weights}")
    print(f"  Source: {source}")
    print(f"  Confidence threshold: {conf_thres}")
    print(f"  IOU threshold: {iou_thres}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device}")
    print(f"  Output: {project}/{name}")
    print()

    def project_path(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else project_root / path).resolve()

    # Resolve relative paths from the repository so the wrapper works when
    # launched from a different current directory.
    weights_path = project_path(weights)
    source_is_webcam = str(source).isdigit()
    source_path = source if source_is_webcam else project_path(source)
    project_path_out = project_path(project)

    # Verify weights exist
    if not weights_path.exists():
        print(f"❌ Error: Weights file not found: {weights}")
        return False

    # Verify source exists
    if not source_is_webcam and not source_path.exists():
        print(f"❌ Error: Source not found: {source_path}")
        return False

    print("✓ All files validated")
    print()

    # Build command
    detect_script = project_root / "yolov7" / "detect.py"

    cmd_args = [
        sys.executable,
        str(detect_script),
        f"--weights={weights_path}",
        f"--source={source_path}",
        f"--conf-thres={conf_thres}",
        f"--iou-thres={iou_thres}",
        f"--img-size={img_size}",
        f"--project={project_path_out}",
        f"--name={name}",
        f"--device={device}",
    ]

    if save_txt:
        cmd_args.append("--save-txt")

    if save_conf:
        cmd_args.append("--save-conf")

    if nosave:
        cmd_args.append("--nosave")

    if view_img:
        cmd_args.append("--view-img")

    if exist_ok:
        cmd_args.append("--exist-ok")

    if classes:
        cmd_args.append(f"--classes={' '.join(map(str, classes))}")

    print("=" * 75)
    print("RUNNING DETECTION")
    print("=" * 75)
    print()
    print(f"Command: {' '.join(cmd_args)}")
    print()

    # Run detection
    import subprocess
    try:
        result = subprocess.run(cmd_args, cwd=str(project_root))

        if result.returncode == 0:
            print()
            print("=" * 75)
            print("✅ DETECTION COMPLETE!")
            print("=" * 75)
            print()
            print(f"Results saved to: {project}/{name}")
            print()
            if save_txt:
                print("  Detection labels: labels/*.txt")
            if not nosave:
                print("  Annotated images: *.jpg")
            print()
            return True
        else:
            print()
            print("❌ Detection failed")
            return False

    except KeyboardInterrupt:
        print()
        print("⚠ Detection interrupted by user")
        return False
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run YOLOv7 detection on images/videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect on single image
  python scripts/detect_yolo.py --weights runs/train/openlogo_training3/weights/best.pt --source logo_test/IMG_9480.JPG

  # Detect on folder
  python scripts/detect_yolo.py --weights runs/train/openlogo_training3/weights/best.pt --source logo_test/

  # Lower confidence threshold
  python scripts/detect_yolo.py --weights runs/train/openlogo_training3/weights/best.pt --source logo_test/ --conf 0.25

  # Save to specific folder
  python scripts/detect_yolo.py --weights runs/train/openlogo_training3/weights/best.pt --source logo_test/ --name my_results

  # Don't save images (labels only)
  python scripts/detect_yolo.py --weights runs/train/openlogo_training3/weights/best.pt --source logo_test/ --nosave
        """
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source (image, folder, video, or 0 for webcam)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold (default: 0.4)"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for NMS (default: 0.45)"
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference size (default: 640)"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory (default: runs/detect)"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="detection",
        help="Run name (default: detection)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device (0, 1, 2, etc or 'cpu') (default: 0)"
    )

    parser.add_argument(
        "--no-save-txt",
        action="store_true",
        help="Do not save results to *.txt"
    )

    parser.add_argument(
        "--no-save-conf",
        action="store_true",
        help="Do not save confidences in labels"
    )

    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Do not save images/videos"
    )

    parser.add_argument(
        "--classes",
        nargs='+',
        type=int,
        help="Filter by class (e.g., --classes 0 2 3)"
    )

    parser.add_argument(
        "--view-img",
        action="store_true",
        help="Display results"
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing project/name"
    )

    args = parser.parse_args()

    # Run detection
    success = detect_yolov7(
        weights=args.weights,
        source=args.source,
        conf_thres=args.conf,
        iou_thres=args.iou,
        img_size=args.img_size,
        project=args.project,
        name=args.name,
        device=args.device,
        save_txt=not args.no_save_txt,
        save_conf=not args.no_save_conf,
        nosave=args.nosave,
        classes=args.classes,
        view_img=args.view_img,
        exist_ok=args.exist_ok
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
