#!/usr/bin/env python3
"""Find logo candidates and classify them against a CLIP reference gallery.

The detector proposes regions; CLIP assigns each crop to the most similar
brand prototype. Candidates below the similarity threshold are labeled
``unknown``. For useful open-set behavior, train/use a detector whose single
class means "logo", rather than a detector limited to known brand classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLOV7_ROOT = PROJECT_ROOT / "yolov7"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(YOLOV7_ROOT))

from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_coords  # noqa: E402
from utils.plots import plot_one_box  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

from scripts.clip_embedder import CLIPConfig, CLIPLogoEmbedder  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Input must be an image or a directory of images: {source}")
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"Input path does not exist: {source}")


def load_gallery(gallery_dir: Path, embedder: CLIPLogoEmbedder) -> dict[str, np.ndarray]:
    """Build one normalized CLIP prototype per brand subdirectory."""
    prototypes: dict[str, np.ndarray] = {}
    brand_dirs = sorted(path for path in gallery_dir.iterdir() if path.is_dir())
    if not brand_dirs:
        raise ValueError(f"No brand subdirectories found in reference gallery: {gallery_dir}")

    for brand_dir in brand_dirs:
        features = []
        for path in sorted(brand_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is None or image.size == 0:
                logging.warning("Skipping unreadable reference image: %s", path)
                continue
            height, width = image.shape[:2]
            features.append(embedder.extract(image, [(0, 0, width, height)])[0])
        if features:
            prototype = np.mean(np.stack(features), axis=0)
            norm = np.linalg.norm(prototype)
            if norm > 0:
                prototypes[brand_dir.name] = (prototype / norm).astype(np.float32)
        else:
            logging.warning("No readable reference images for brand: %s", brand_dir.name)

    if not prototypes:
        raise ValueError(f"No usable reference images found under {gallery_dir}")
    return prototypes


def classify_crop(feature: np.ndarray, prototypes: dict[str, np.ndarray], threshold: float) -> tuple[str, float]:
    scores = {brand: float(np.dot(feature, prototype)) for brand, prototype in prototypes.items()}
    brand = max(scores, key=scores.get)
    score = scores[brand]
    return (brand if score >= threshold else "unknown"), score


def infer_image(
    image: np.ndarray,
    model: Any,
    device: torch.device,
    image_size: int,
    stride: int,
    half: bool,
    confidence: float,
    iou_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    padded = letterbox(image, image_size, stride=stride, auto=True)[0]
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor)
    tensor = torch.from_numpy(tensor).to(device)
    tensor = tensor.half() if half else tensor.float()
    tensor /= 255.0
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)

    with torch.inference_mode():
        prediction = model(tensor, augment=False)[0]
    detections = non_max_suppression(
        prediction,
        confidence,
        iou_threshold,
        agnostic=True,
    )[0]
    if detections is None or not len(detections):
        return []
    detections = detections[:max_detections]

    detections[:, :4] = scale_coords(tensor.shape[2:], detections[:, :4], image.shape).round()
    results = []
    height, width = image.shape[:2]
    for row in detections.detach().cpu().numpy():
        x1, y1, x2, y2 = map(int, row[:4])
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        results.append({
            "box": (x1, y1, x2, y2),
            "confidence": float(row[4]),
            "detector_class_id": int(row[5]),
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect logo candidates and label them known or unknown using CLIP reference examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="YOLOv7 logo-detector checkpoint (.pt)")
    parser.add_argument("--source", required=True, help="Input image or directory of images")
    parser.add_argument("--known-logos", required=True, help="Gallery directory with one subdirectory per known brand")
    parser.add_argument("--output-dir", default="runs/unknown_logo_detection", help="Directory for overlays, crops, and reports")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--img-size", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum YOLO candidate confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--similarity-threshold", type=float, default=0.78, help="Minimum CLIP cosine similarity to call a candidate known")
    parser.add_argument("--max-detections", type=int, default=300, help="Maximum candidates per image")
    parser.add_argument("--clip-model", default="ViT-B/32", help="OpenAI CLIP model name")
    parser.add_argument("--clip-device", default=None, help="CLIP device; defaults to the detector device")
    args = parser.parse_args()

    weights = Path(args.weights).expanduser()
    weights = (weights if weights.is_absolute() else PROJECT_ROOT / weights).resolve()
    source = Path(args.source).expanduser()
    source = (source if source.is_absolute() else PROJECT_ROOT / source).resolve()
    gallery_dir = Path(args.known_logos).expanduser()
    gallery_dir = (gallery_dir if gallery_dir.is_absolute() else PROJECT_ROOT / gallery_dir).resolve()
    output_dir = Path(args.output_dir).expanduser()
    output_dir = (output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir).resolve()

    if not weights.is_file():
        parser.error(f"YOLO weights not found: {weights}")
    if not gallery_dir.is_dir():
        parser.error(f"Known-logo gallery not found: {gallery_dir}")
    if not -1.0 <= args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be between -1 and 1")
    try:
        inputs = image_paths(source)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if not inputs:
        parser.error(f"No supported image files found in {source}")

    clip_device = args.clip_device or ("cpu" if args.device.lower() == "cpu" else "cuda")
    try:
        embedder = CLIPLogoEmbedder(CLIPConfig(model_name=args.clip_model, device=clip_device))
    except ImportError as exc:
        parser.error(f"CLIP is required for known/unknown matching. Install OpenAI CLIP in this environment. Details: {exc}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prototypes = load_gallery(gallery_dir, embedder)
    device = select_device(args.device)
    model = attempt_load(str(weights), map_location=device)
    stride = int(model.stride.max())
    image_size = check_img_size(args.img_size, s=stride)
    half = device.type != "cpu"
    if half:
        model.half()
    model.eval()
    detector_names = model.module.names if hasattr(model, "module") else model.names

    annotated_dir = output_dir / "images"
    crops_dir = output_dir / "crops"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for image_path in inputs:
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning("Skipping unreadable image: %s", image_path)
            continue
        detections = infer_image(image, model, device, image_size, stride, half, args.conf, args.iou, args.max_detections)
        bboxes = [(d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) for d in detections]
        features = embedder.extract(image, bboxes)
        relative_path = image_path.name if source.is_file() else image_path.relative_to(source)
        output_image = annotated_dir / relative_path
        output_image.parent.mkdir(parents=True, exist_ok=True)
        display = image.copy()

        for index, (detection, feature) in enumerate(zip(detections, features)):
            label, similarity = classify_crop(feature, prototypes, args.similarity_threshold)
            x1, y1, x2, y2 = detection["box"]
            detector_class_id = detection["detector_class_id"]
            class_name = str(detector_names[detector_class_id]) if detector_class_id < len(detector_names) else str(detector_class_id)
            color = (0, 180, 0) if label != "unknown" else (0, 0, 255)
            plot_one_box((x1, y1, x2, y2), display, label=f"{label} {similarity:.2f} | det {detection['confidence']:.2f}", color=color, line_thickness=2)

            crop_relative = Path(label) / relative_path.parent / f"{relative_path.stem}_det{index:03d}.jpg"
            crop_path = crops_dir / crop_relative
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
            records.append({
                "image": str(relative_path),
                "detection_index": index,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "detector_class": class_name,
                "detector_confidence": detection["confidence"],
                "matched_brand": label,
                "clip_similarity": similarity,
                "crop": str(crop_relative),
            })

        cv2.imwrite(str(output_image), display)
        logging.info("%s: %d logo candidate(s)", image_path, len(detections))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "detections.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image", "detection_index", "x1", "y1", "x2", "y2", "detector_class", "detector_confidence", "matched_brand", "clip_similarity", "crop"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "source": str(source),
        "weights": str(weights),
        "gallery": str(gallery_dir),
        "similarity_threshold": args.similarity_threshold,
        "images_processed": len(inputs),
        "detections": len(records),
        "known": sum(record["matched_brand"] != "unknown" for record in records),
        "unknown": sum(record["matched_brand"] == "unknown" for record in records),
        "brands": sorted(prototypes),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Saved overlays, candidate crops, and reports to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
