#!/usr/bin/env python3
"""Interactively select videos under a folder and run YOLOv7 detection."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_DETECT = PROJECT_ROOT / "yolov7" / "detect.py"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpeg", ".mpg", ".wmv", ".asf", ".webm"}


def videos_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def select_videos(root: Path) -> list[Path]:
    """Browse folders; choose a file or all videos recursively in a folder."""
    current = root
    while True:
        directories = sorted((path for path in current.iterdir() if path.is_dir()), key=lambda p: p.name.lower())
        video_files = sorted((path for path in current.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES), key=lambda p: p.name.lower())

        print(f"\nVideo folder: {current.relative_to(root) if current != root else '.'}")
        choices: list[tuple[str, Path]] = []
        for folder in directories:
            choices.append(("process_folder", folder))
            print(f"  {len(choices)}. Process all videos in folder: {folder.name}/")
            choices.append(("browse_folder", folder))
            print(f"  {len(choices)}. Browse folder: {folder.name}/")
        for video in video_files:
            choices.append(("video", video))
            print(f"  {len(choices)}. Video: {video.name}")

        back_choice = None
        if current != root:
            back_choice = len(choices) + 1
            print(f"  {back_choice}. .. (go to parent folder)")
        quit_choice = len(choices) + (2 if back_choice is not None else 1)
        print(f"  {quit_choice}. Quit")

        try:
            raw_choice = input("Select an option: ").strip()
        except EOFError:
            print("\nNo selection received; exiting.")
            return []
        try:
            choice_number = int(raw_choice)
        except ValueError:
            print("Enter the number shown next to an option.")
            continue

        if choice_number == quit_choice:
            return []
        if back_choice is not None and choice_number == back_choice:
            current = current.parent
            continue
        if choice_number < 1 or choice_number > len(choices):
            print("That option is not in the list.")
            continue

        action, path = choices[choice_number - 1]
        if action == "process_folder":
            selected = videos_in(path)
            if selected:
                return selected
            print(f"No supported video files found under {path}.")
        elif action == "browse_folder":
            current = path
        else:
            return [path]


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "video"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Choose a video or folder under test_videos and run YOLOv7 detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="YOLOv7 checkpoint (.pt)")
    parser.add_argument("--root", default="test_videos", help="Starting folder for the interactive picker")
    parser.add_argument("--project", default="runs/batch_video_detect", help="Parent folder for this batch's results")
    parser.add_argument("--device", default="0", help="CUDA device (for example 0) or cpu")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--no-save-labels", action="store_true", help="Do not save YOLO text labels")
    args = parser.parse_args()

    weights = resolve_project_path(args.weights)
    root = resolve_project_path(args.root)
    project = resolve_project_path(args.project)
    if not weights.is_file():
        parser.error(f"Weights file does not exist: {weights}")
    if not root.is_dir():
        parser.error(f"Video selection folder does not exist: {root}")
    if not YOLO_DETECT.is_file():
        parser.error(f"YOLOv7 detector script not found: {YOLO_DETECT}")

    print(f"Choose videos under: {root}")
    selected = select_videos(root)
    if not selected:
        print("No videos selected; exiting without running detection.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = project / timestamp
    batch_dir.mkdir(parents=True, exist_ok=False)
    print(f"\nSelected {len(selected)} video(s). Results will be saved under: {batch_dir}\n")

    summary: list[dict[str, Any]] = []
    overall_success = True
    for index, video in enumerate(selected, start=1):
        relative = video.relative_to(root)
        parent_parts = tuple(safe_name(part) for part in relative.parent.parts if part != ".")
        run_name = Path(*parent_parts, safe_name(video.stem))
        output_dir = batch_dir / run_name
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(YOLO_DETECT),
            f"--weights={weights}",
            f"--source={video}",
            f"--conf-thres={args.conf}",
            f"--iou-thres={args.iou}",
            f"--img-size={args.img_size}",
            f"--project={batch_dir}",
            f"--name={run_name.as_posix()}",
            f"--device={args.device}",
            "--save-conf",
            "--exist-ok",
        ]
        if not args.no_save_labels:
            command.append("--save-txt")

        print(f"[{index}/{len(selected)}] Detecting {relative}")
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            success = result.returncode == 0
            return_code = result.returncode
        except KeyboardInterrupt:
            success = False
            return_code = 130
            overall_success = False
            summary.append({
                "video": str(relative),
                "status": "interrupted",
                "return_code": return_code,
                "output_dir": str(output_dir),
            })
            print("\nBatch interrupted by user.")
            break
        except OSError as exc:
            success = False
            return_code = -1
            print(f"Could not start detector for {relative}: {exc}")

        status = "success" if success else "failed"
        summary.append({
            "video": str(relative),
            "status": status,
            "return_code": return_code,
            "output_dir": str(output_dir),
        })
        if not success:
            overall_success = False
            print(f"Detection failed for {relative}; continuing with the remaining videos.\n")

    (batch_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (batch_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video", "status", "return_code", "output_dir"])
        writer.writeheader()
        writer.writerows(summary)

    succeeded = sum(item["status"] == "success" for item in summary)
    failed = sum(item["status"] != "success" for item in summary)
    print(f"\nBatch complete: {succeeded} succeeded, {failed} failed or interrupted.")
    print(f"Summary: {batch_dir / 'summary.csv'}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
