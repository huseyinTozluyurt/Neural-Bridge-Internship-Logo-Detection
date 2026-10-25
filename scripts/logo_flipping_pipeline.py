#!/usr/bin/env python3
"""Logo reflection pipeline: detection, masking, geometry capture, inpainting, and reflection.

Process:
1. Detect logos (YOLOv7 + EasyOCR)
2. Refine masks (GrabCut/SAM)
3. Extract geometry (symmetry lines, polygon contours)
4. Inpaint background
5. Reflect logos (line-based or perspective-aware)
6. Blend back into scene
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

import torch

import easyocr

from scripts.sam_masker import SAMConfig, create_sam_masker

# Try to import LaMa for ML-based inpainting
try:
    from simple_lama_inpainting import SimpleLama
    from PIL import Image as PILImage
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False
    SimpleLama = None
    PILImage = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_ROOT = PROJECT_ROOT / "yolov7"
sys.path.insert(0, str(PROJECT_ROOT))
if YOLO_ROOT.exists():
    sys.path.insert(0, str(YOLO_ROOT))

try:
    from yolov7.models.experimental import attempt_load
    from yolov7.utils.datasets import letterbox
    from yolov7.utils.general import (
        check_img_size,
        non_max_suppression,
        scale_coords,
        set_logging,
    )
    from yolov7.utils.torch_utils import select_device
except Exception:  # pragma: no cover - optional dependency
    attempt_load = None
    letterbox = None
    check_img_size = None
    non_max_suppression = None
    scale_coords = None
    set_logging = None
    select_device = None


BBox = Tuple[int, int, int, int]  # x, y, width, height


@dataclass
class DetectionSettings:
    """Configurable parameters for the YOLO + EasyOCR discovery stage."""

    yolo_weights: Optional[Path] = None
    yolo_conf: float = 0.50
    yolo_iou: float = 0.45
    yolo_img_size: int = 640
    yolo_device: str = "0"
    use_easyocr: bool = False
    easyocr_langs: Tuple[str, ...] = ("en",)
    easyocr_conf: float = 0.1  # EasyOCR returns [0, 1], default 0.3 (30%)
    easyocr_min_size: int = 1280
    easyocr_max_size: int = 1920
    merge_gap: int = 12


@dataclass
class DetectionCandidate:
    """Intermediate structure that stores pixel-space detection metadata."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: Optional[int] = None
    sources: Set[str] = field(default_factory=set)

    def copy(self) -> "DetectionCandidate":
        return DetectionCandidate(
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            confidence=self.confidence,
            class_id=self.class_id,
            sources=set(self.sources),
        )

    def clamp(self, width: int, height: int) -> None:
        width = max(1, width)
        height = max(1, height)
        self.x1 = max(0, min(self.x1, width - 1))
        self.y1 = max(0, min(self.y1, height - 1))
        self.x2 = max(self.x1 + 1, min(self.x2, width))
        self.y2 = max(self.y1 + 1, min(self.y2, height))

    def to_bbox(self) -> BBox:
        return (
            self.x1,
            self.y1,
            max(1, self.x2 - self.x1),
            max(1, self.y2 - self.y1),
        )


def _valid_text_candidate(text: str, width: int, height: int, area: int) -> bool:
    content = (text or "").strip()
    if not content:
        return False
    if not any(ch.isalnum() for ch in content):
        return False
    if len(content) == 1 and area < 250:
        return False
    if area < 80 or width < 6 or height < 6:
        return False
    return True


def _resize_for_ocr(image: np.ndarray, min_size: int, max_size: int) -> Tuple[np.ndarray, float]:
    if image is None or image.size == 0:
        return image, 1.0
    h, w = image.shape[:2]
    longest = max(h, w)
    target = float(longest)
    if min_size and longest < min_size:
        target = float(min_size)
    if max_size and target > max_size:
        target = float(max_size)
    if longest <= 0 or abs(target - longest) < 1e-3:
        return image, 1.0
    scale = target / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, scale


def _boxes_close(a: DetectionCandidate, b: DetectionCandidate, gap: int) -> bool:
    gap = max(0, gap)
    width_a = max(1, a.x2 - a.x1)
    height_a = max(1, a.y2 - a.y1)
    width_b = max(1, b.x2 - b.x1)
    height_b = max(1, b.y2 - b.y1)
    adaptive = max(gap, int(0.15 * max(width_a, height_a, width_b, height_b)))

    ax1 = a.x1 - adaptive
    ay1 = a.y1 - adaptive
    ax2 = a.x2 + adaptive
    ay2 = a.y2 + adaptive

    bx1 = b.x1 - adaptive
    by1 = b.y1 - adaptive
    bx2 = b.x2 + adaptive
    by2 = b.y2 + adaptive

    if ax2 >= bx1 and bx2 >= ax1 and ay2 >= by1 and by2 >= ay1:
        return True

    # Horizontal alignment with strong vertical overlap (e.g., icon + wordmark)
    vertical_overlap = min(a.y2, b.y2) - max(a.y1, b.y1)
    if vertical_overlap > 0:
        min_height = max(1, min(height_a, height_b))
        if vertical_overlap / min_height >= 0.35:
            horizontal_gap = max(a.x1 - b.x2, b.x1 - a.x2, 0)
            if horizontal_gap <= adaptive * 2:
                return True

    # Vertical alignment with strong horizontal overlap (stacked text)
    horizontal_overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
    if horizontal_overlap > 0:
        min_width = max(1, min(width_a, width_b))
        if horizontal_overlap / min_width >= 0.35:
            vertical_gap = max(a.y1 - b.y2, b.y1 - a.y2, 0)
            if vertical_gap <= adaptive * 2:
                return True

    return False


def _box_contains(outer: DetectionCandidate, inner: DetectionCandidate, margin: int) -> bool:
    return (
        inner.x1 >= outer.x1 - margin
        and inner.y1 >= outer.y1 - margin
        and inner.x2 <= outer.x2 + margin
        and inner.y2 <= outer.y2 + margin
    )


def _drop_contained_candidates(
    candidates: List[DetectionCandidate], gap: int
) -> List[DetectionCandidate]:
    if len(candidates) < 2:
        return candidates
    margin = max(gap, 8)
    keep = [True] * len(candidates)

    for i in range(len(candidates)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(candidates)):
            if not keep[j]:
                continue
            outer = candidates[i]
            inner = candidates[j]
            if _box_contains(outer, inner, margin):
                outer.x1 = min(outer.x1, inner.x1)
                outer.y1 = min(outer.y1, inner.y1)
                outer.x2 = max(outer.x2, inner.x2)
                outer.y2 = max(outer.y2, inner.y2)
                outer.confidence = max(outer.confidence, inner.confidence)
                outer.sources.update(inner.sources)
                keep[j] = False
            elif _box_contains(inner, outer, margin):
                inner.x1 = min(inner.x1, outer.x1)
                inner.y1 = min(inner.y1, outer.y1)
                inner.x2 = max(inner.x2, outer.x2)
                inner.y2 = max(inner.y2, outer.y2)
                inner.confidence = max(inner.confidence, outer.confidence)
                inner.sources.update(outer.sources)
                keep[i] = False
                break
    return [cand for idx, cand in enumerate(candidates) if keep[idx]]


def _merge_candidates(
    candidates: Sequence[DetectionCandidate],
    gap: int,
    width: int,
    height: int,
) -> List[DetectionCandidate]:
    if not candidates:
        return []

    merged: List[DetectionCandidate] = []
    used = [False] * len(candidates)

    for idx, candidate in enumerate(candidates):
        if used[idx]:
            continue
        current = candidate.copy()
        used[idx] = True

        for j in range(idx + 1, len(candidates)):
            if used[j]:
                continue
            other = candidates[j]
            if _boxes_close(current, other, gap):
                current.x1 = min(current.x1, other.x1)
                current.y1 = min(current.y1, other.y1)
                current.x2 = max(current.x2, other.x2)
                current.y2 = max(current.y2, other.y2)
                current.confidence = max(current.confidence, other.confidence)
                current.sources.update(other.sources)
                used[j] = True

        current.clamp(width, height)
        merged.append(current)

    return _drop_contained_candidates(merged, gap)


class YoloLogoDetector:
    def __init__(
        self,
        weights: Path,
        conf: float,
        iou: float,
        img_size: int,
        device: str = "0",
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for YOLO-based detection.")
        if any(
            dep is None
            for dep in (attempt_load, letterbox, check_img_size, non_max_suppression, scale_coords, select_device)
        ):
            raise ImportError(
                "yolov7 modules were not found. Ensure the yolov7 directory is available on PYTHONPATH."
            )
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")

        set_logging()
        self.device = select_device(device)
        self.conf = conf
        self.iou = iou
        self.model = attempt_load(str(weights_path), map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        self.half = self.device.type != "cpu"
        if self.half:
            self.model.half()
        self.model.eval()

        if self.device.type != "cpu":
            dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
            dummy = dummy.half() if self.half else dummy.float()
            with torch.no_grad():
                self.model(dummy)

        logging.info("Loaded YOLO weights %s (img_size=%d)", weights_path, self.img_size)

    def detect(self, image: np.ndarray) -> List[DetectionCandidate]:
        if image is None or image.size == 0:
            return []

        img_letterboxed, _, _ = letterbox(
            image, new_shape=self.img_size, stride=self.stride, auto=False
        )
        blob = img_letterboxed[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, CHW
        blob = np.ascontiguousarray(blob)
        tensor = torch.from_numpy(blob).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        candidates: List[DetectionCandidate] = []
        with torch.no_grad():
            prediction = self.model(tensor, augment=False)[0]
        prediction = non_max_suppression(prediction, self.conf, self.iou)

        for det in prediction:
            if not len(det):
                continue
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(coord.item()) for coord in xyxy]
                candidate = DetectionCandidate(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(conf.item()),
                    class_id=int(cls.item()),
                    sources={"yolo"},
                )
                candidates.append(candidate)
        return candidates


class EasyOCRTextDetector:
    def __init__(
        self,
        languages: Tuple[str, ...],
        min_confidence: float,
        min_size: int,
        max_size: int,
    ) -> None:
        if easyocr is None:
            raise ImportError("easyocr is not installed; install it or disable --use-easyocr.")
        self.languages = languages or ("en",)
        self.min_confidence = min_confidence
        self.min_size = max(0, min_size)
        self.max_size = max(0, max_size)
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None:
            use_gpu = bool(torch and torch.cuda.is_available())
            self._reader = easyocr.Reader(list(self.languages), gpu=use_gpu)
        return self._reader

    def detect(self, image: np.ndarray) -> List[DetectionCandidate]:
        reader = self._ensure_reader()
        if reader is None or image is None or image.size == 0:
            return []

        resized, scale = _resize_for_ocr(image, self.min_size, self.max_size)
        # EasyOCR expects RGB format, convert from OpenCV's BGR
        proc_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        try:
            results = reader.readtext(proc_rgb, detail=1, paragraph=False)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("EasyOCR failed: %s", exc)
            return []

        threshold = max(0.0, min(1.0, self.min_confidence))
        candidates: List[DetectionCandidate] = []
        for entry in results:
            if len(entry) != 3:
                continue
            bbox, text, conf = entry
            if conf < threshold:
                continue
            xs = [pt[0] / scale for pt in bbox]
            ys = [pt[1] / scale for pt in bbox]
            x1 = int(max(0, min(xs)))
            y1 = int(max(0, min(ys)))
            x2 = int(max(xs))
            y2 = int(max(ys))
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            area = width * height
            if not _valid_text_candidate(text, width, height, area):
                continue
            candidate = DetectionCandidate(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(conf),
                class_id=None,
                sources={"ocr"},
            )
            candidates.append(candidate)
        return candidates



class LogoDetector:
    def __init__(self, settings: DetectionSettings) -> None:
        self.settings = settings
        self.merge_gap = max(0, settings.merge_gap)

        self.yolo = None
        if settings.yolo_weights:
            self.yolo = YoloLogoDetector(
                weights=settings.yolo_weights,
                conf=settings.yolo_conf,
                iou=settings.yolo_iou,
                img_size=settings.yolo_img_size,
                device=settings.yolo_device,
            )

        self.ocr = None
        if settings.use_easyocr:
            try:
                self.ocr = EasyOCRTextDetector(
                    languages=settings.easyocr_langs,
                    min_confidence=settings.easyocr_conf,
                    min_size=settings.easyocr_min_size,
                    max_size=settings.easyocr_max_size,
                )
            except ImportError as exc:
                logging.warning("EasyOCR unavailable: %s", exc)

        if self.yolo is None and self.ocr is None:
            raise RuntimeError(
                "LogoDetector requires at least one source (YOLO weights or EasyOCR)."
            )

    def detect(self, image: np.ndarray) -> List[DetectionCandidate]:
        h, w = image.shape[:2]
        yolo_candidates: List[DetectionCandidate] = []
        ocr_candidates: List[DetectionCandidate] = []

        if self.yolo:
            yolo_candidates = self.yolo.detect(image)
        if self.ocr:
            ocr_candidates = self.ocr.detect(image)

        merged = _merge_candidates(
            [*yolo_candidates, *ocr_candidates],
            gap=self.merge_gap,
            width=w,
            height=h,
        )
        logging.info(
            "Detector produced %d merged regions (YOLO=%d, OCR=%d).",
            len(merged),
            len(yolo_candidates),
            len(ocr_candidates),
        )
        return merged


def detect_skin_pixels(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect skin-colored pixels in an image using multiple color space thresholds.
    
    Uses a combination of HSV and YCrCb color spaces for robust skin detection
    that works across different skin tones.
    
    Args:
        image_bgr: BGR image (OpenCV format)
    
    Returns:
        Binary mask where 255 = skin pixel, 0 = non-skin pixel
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    
    # HSV-based skin detection (works well for lighter skin tones)
    # H: 0-50 (red-yellow range), S: 20-255 (some saturation), V: 40-255 (not too dark)
    hsv_lower = np.array([0, 20, 40], dtype=np.uint8)
    hsv_upper = np.array([50, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    
    # YCrCb-based skin detection (more robust across skin tones)
    # Y: 0-255 (any luminance), Cr: 135-180, Cb: 85-135
    ycrcb_lower = np.array([0, 135, 85], dtype=np.uint8)
    ycrcb_upper = np.array([255, 180, 135], dtype=np.uint8)
    ycrcb_mask = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)
    
    # Combine masks - pixel is skin if detected in BOTH color spaces (more conservative)
    # This reduces false positives from orange/red colored logos
    skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    
    # Morphological cleanup to remove noise and fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
    
    return skin_mask


def compute_skin_ratio_in_mask(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the ratio of skin-colored pixels within a masked region.
    
    Args:
        image_bgr: BGR image
        mask: Binary mask defining the region of interest
    
    Returns:
        Tuple of (skin_ratio, skin_mask_in_roi)
        - skin_ratio: Fraction of pixels in mask that appear to be skin (0-1)
        - skin_mask_in_roi: Binary mask of detected skin pixels within the ROI
    """
    if image_bgr is None or mask is None or mask.size == 0:
        return 0.0, np.zeros_like(mask) if mask is not None else np.zeros((1, 1), dtype=np.uint8)
    
    # Ensure mask is binary
    mask_binary = (mask > 127).astype(np.uint8) * 255
    
    # Detect skin pixels in the entire image
    skin_mask = detect_skin_pixels(image_bgr)
    
    # Find skin pixels within the logo mask region
    skin_in_mask = cv2.bitwise_and(skin_mask, mask_binary)
    
    # Calculate ratio
    mask_pixels = np.count_nonzero(mask_binary)
    skin_pixels = np.count_nonzero(skin_in_mask)
    
    if mask_pixels == 0:
        return 0.0, skin_in_mask
    
    skin_ratio = skin_pixels / mask_pixels
    return skin_ratio, skin_in_mask


def refine_mask_excluding_skin(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    skin_threshold: float = 0.1
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Refine a logo mask by detecting and optionally excluding skin-colored regions.
    
    This helps prevent inpainting over hands that are partially covering logos.
    
    Args:
        image_bgr: BGR image (full frame or ROI)
        mask: Original binary mask
        skin_threshold: If skin ratio exceeds this, return the refined mask
    
    Returns:
        Tuple of (refined_mask, skin_ratio, skin_mask)
        - refined_mask: Mask with skin regions optionally removed
        - skin_ratio: Detected skin ratio in original mask
        - skin_mask: Binary mask of detected skin pixels
    """
    skin_ratio, skin_mask = compute_skin_ratio_in_mask(image_bgr, mask)
    
    if skin_ratio < skin_threshold:
        # Not enough skin detected, return original mask
        return mask.copy(), skin_ratio, skin_mask
    
    # Significant skin detected - refine the mask
    # Remove skin regions from the mask to avoid inpainting over hands
    mask_binary = (mask > 127).astype(np.uint8) * 255
    refined_mask = cv2.bitwise_and(mask_binary, cv2.bitwise_not(skin_mask))
    
    # Ensure we don't remove too much - if refined mask is too small, keep original
    original_pixels = np.count_nonzero(mask_binary)
    refined_pixels = np.count_nonzero(refined_mask)
    
    if original_pixels > 0 and refined_pixels / original_pixels < 0.3:
        # Would remove more than 70% of the mask - likely false positive skin detection
        # This can happen with skin-colored logos (e.g., some sports team logos)
        logging.debug(
            "Skin removal would reduce mask by %.1f%% - keeping original to avoid false positive",
            100 * (1 - refined_pixels / original_pixels)
        )
        return mask.copy(), skin_ratio, skin_mask
    
    logging.debug(
        "Refined mask: removed %.1f%% skin pixels (ratio=%.3f)",
        100 * (original_pixels - refined_pixels) / original_pixels if original_pixels > 0 else 0,
        skin_ratio
    )
    
    return refined_mask, skin_ratio, skin_mask


@dataclass
class SymmetryLine:
    """Represents an infinite symmetry line inside a cropped logo patch."""

    origin: Tuple[float, float]
    direction: Tuple[float, float]

    def as_unit(self) -> "SymmetryLine":
        vx, vy = self.direction
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            raise ValueError("Cannot normalise a near-zero direction vector.")
        return SymmetryLine(self.origin, (vx / norm, vy / norm))

    @property
    def angle_degrees(self) -> float:
        vx, vy = self.direction
        return math.degrees(math.atan2(vy, vx))


@dataclass
class LogoRegion:
    """Holds every intermediate artifact for a single logo."""

    logo_id: int
    bbox: BBox
    padded_origin: Tuple[int, int]
    patch: np.ndarray
    mask: np.ndarray
    masked_patch: np.ndarray
    orientation_line: Optional[SymmetryLine] = None
    polygon: Optional[np.ndarray] = None
    reflected_patch: Optional[np.ndarray] = None
    reflected_mask: Optional[np.ndarray] = None
    artifacts: Dict[str, Path] = field(default_factory=dict)
    detection_sources: Set[str] = field(default_factory=set)
    detection_confidence: float = 0.0
    # Occlusion-aware fields
    occlusion_mask: Optional[np.ndarray] = None  # Detected skin/hand regions in the mask area
    skin_ratio: float = 0.0  # Ratio of skin-colored pixels in the masked region (0-1)
    is_partially_occluded: bool = False  # Flag if logo appears to be partially covered


@dataclass
class FrameProcessingResult:
    """Container with both the flipped image and detection metadata."""

    original_image: np.ndarray
    flipped_image: np.ndarray
    regions: List[LogoRegion]
    detections: List[DetectionCandidate]
    inpainted_background: Optional[np.ndarray] = None  # Background with logos removed but not flipped


@dataclass
class PipelineConfig:
    mask_padding: int = 12
    grabcut_iters: int = 5
    min_mask_quality: float = 0.05  # Minimum mask coverage ratio (5%)
    inpaint_radius: int = 5
    inpaint_expansion: int = 10
    inpaint_mask_margin: int = 0
    naturalise_mode: str = "auto"  # auto | line | polygon
    blend_mode: str = "seamless"  # seamless | alpha
    use_perspective_flip: bool = True  # Enable perspective-aware flipping for distorted logos
    input_is_flipped: bool = False  # For video: if True, input is already flipped (testing mode)
    artifact_dir: Optional[Path] = None
    save_intermediates: bool = False
    bbox_json: Optional[Path] = None
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    use_sam_masker: bool = False
    sam_model_path: Optional[Path] = None
    sam_device: Optional[str] = None
    sam_image_size: int = 512
    # Occlusion-aware inpainting settings
    detect_skin_occlusion: bool = True  # Enable skin/hand detection to avoid inpainting over hands
    skin_detection_threshold: float = 0.15  # Threshold for skin pixel ratio to trigger occlusion handling
    reduce_expansion_on_occlusion: bool = True  # Reduce inpaint_expansion when occlusion is detected
    # Hand-aware inpainting settings (for video processing with MediaPipe)
    hand_exclusion_buffer: int = 15  # Pixels to expand hand mask for exclusion zone (prevents sampling hand pixels)
    use_two_pass_inpainting: bool = True  # Two-pass inpainting: neutralize hands first, then inpaint logo
    inpaint_full_logo_under_hand: bool = True  # Inpaint full logo even under hand (don't subtract occlusion)
    # ML-based inpainting settings (LaMa)
    use_lama_inpainting: bool = False  # Use LaMa ML model instead of cv2.inpaint (better for textures like fabric)
    lama_device: Optional[str] = None  # Device for LaMa (None = auto, 'cuda', 'cpu')


class LogoFlippingPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        if config.artifact_dir:
            config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._current_artifact_dir: Optional[Path] = None
        self._logo_detector: Optional[LogoDetector] = None
        self._detector_ready = False
        self._sam_masker = None
        self._lama_inpainter = None
        
        # Initialize LaMa inpainter if requested
        if config.use_lama_inpainting:
            if not LAMA_AVAILABLE:
                logging.warning(
                    "LaMa inpainting requested but simple-lama-inpainting not installed. "
                    "Install with: pip install simple-lama-inpainting\n"
                    "Falling back to cv2.inpaint."
                )
            else:
                try:
                    self._lama_inpainter = SimpleLama()
                    logging.info("LaMa ML inpainting initialized successfully")
                except Exception as e:
                    logging.warning(f"Failed to initialize LaMa inpainter: {e}. Falling back to cv2.inpaint.")
                    self._lama_inpainter = None
        if config.use_sam_masker:
            model_path = str(config.sam_model_path or "mobile_sam.pt")
            sam_cfg = SAMConfig(
                model_path=model_path,
                device=config.sam_device,
                image_size=config.sam_image_size,
            )
            self._sam_masker = create_sam_masker(sam_cfg)

    def run(self, image_path: Path, output_path: Path) -> None:
        image_path = Path(image_path)
        output_path = Path(output_path)
        self._prepare_artifact_dir(image_path)
        image = self._read_image(image_path)
        processing_result = self._process_image(
            image=image,
            image_path=image_path,
            save_artifacts=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), processing_result.flipped_image)
        logging.info("Saved final corrected image to %s", output_path)

    def process_array(
        self, image: np.ndarray, *, image_name: str = "frame",
        hand_mask: Optional[np.ndarray] = None
    ) -> FrameProcessingResult:
        """
        Process an in-memory frame and return the flipped result and metadata.

        Unlike `run`, this skips any artifact / overlay exports and is therefore
        safe to call repeatedly on individual video frames.
        
        Args:
            image: Input BGR image
            image_name: Name for logging/artifact purposes
            hand_mask: Optional full-frame binary mask of hand regions (255 = hand).
                      Used for two-pass inpainting to prevent hand pixels bleeding
                      into inpainted logo regions.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")
        pseudo_path = Path(image_name)
        return self._process_image(
            image=image,
            image_path=pseudo_path,
            save_artifacts=False,
            hand_mask=hand_mask,
        )

    def _process_image(
        self,
        image: np.ndarray,
        image_path: Path,
        *,
        save_artifacts: bool,
        hand_mask: Optional[np.ndarray] = None,
    ) -> FrameProcessingResult:
        detections = self._detect_logos(image, image_path)

        if self._sam_masker is not None:
            try:
                self._sam_masker.set_image(image)
            except Exception as exc:  # pragma: no cover - runtime guard
                logging.warning("Failed to set SAM image: %s. Disabling SAM for this run.", exc)
                self._sam_masker = None

        if not detections:
            logging.warning("No logo detections found; returning original frame.")
            return FrameProcessingResult(
                original_image=image,
                flipped_image=image.copy(),
                regions=[],
                detections=[],
            )

        regions: List[LogoRegion] = []
        region_idx = 0
        for candidate in detections:
            # Discard purely OCR detections
            sources = set(candidate.sources)
            if sources == {'ocr'}:
                continue
            bbox = candidate.to_bbox()
            region = self._prepare_region(region_idx, image, bbox)
            region.detection_sources = sources
            region.detection_confidence = candidate.confidence
            regions.append(region)
            if save_artifacts:
                self._save_masked_logo(region)
            region_idx += 1

        self._annotate_geometry(regions)

        # Pass hand_mask to inpainting for two-pass processing
        prepared_canvas, inpaint_mask = self._inpaint_background(image, regions, hand_mask=hand_mask)

        # Store inpainted background for video processing (logos removed but not flipped)
        inpainted_background_for_result = prepared_canvas.copy()

        if save_artifacts and self.config.save_intermediates:
            self._save_overlay(image, regions, "detections_masks")
            self._save_mask_image(inpaint_mask, "inpaint_mask.png")
            self._save_image(prepared_canvas, "inpainted.png")

        result_image = prepared_canvas.copy()
        for region in regions:
            reflected_patch, reflected_mask = self._reflect_region(region)
            region.reflected_patch = reflected_patch
            region.reflected_mask = reflected_mask
            result_image = self._paste_region(result_image, region)

        return FrameProcessingResult(
            original_image=image,
            flipped_image=result_image,
            regions=regions,
            detections=detections,
            inpainted_background=inpainted_background_for_result,
        )

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        logging.info("Loaded %s (shape=%s)", path, image.shape)
        return image

    def _detect_logos(
        self, image: np.ndarray, image_path: Path
    ) -> List[DetectionCandidate]:
        if self.config.bbox_json:
            boxes = self._load_bboxes_from_json(self.config.bbox_json)
            return [
                DetectionCandidate(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    confidence=1.0,
                    class_id=None,
                    sources={"json"},
                )
                for (x, y, w, h) in boxes
            ]

        detector = self._get_detector()
        if detector is None:
            raise RuntimeError(
                "No detector is configured. Provide --bbox-json or --yolo-weights "
                "and optionally enable --use-easyocr."
            )
        logging.debug("Running detector on %s", image_path.name)
        return detector.detect(image)

    def _load_bboxes_from_json(self, path: Path) -> List[BBox]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        detections: List[BBox] = []
        for idx, entry in enumerate(data):
            if "bbox" not in entry or len(entry["bbox"]) != 4:
                raise ValueError(f"Entry {idx} in {path} lacks a valid bbox.")
            x, y, w, h = map(int, entry["bbox"])
            detections.append((x, y, w, h))
        logging.info("Loaded %d bounding boxes from %s", len(detections), path)
        return detections

    def _get_detector(self) -> Optional[LogoDetector]:
        if not self._detector_ready:
            detection_cfg = self.config.detection
            if detection_cfg.yolo_weights or detection_cfg.use_easyocr:
                try:
                    self._logo_detector = LogoDetector(detection_cfg)
                except Exception as exc:  # pragma: no cover - init failures
                    raise RuntimeError(f"Failed to initialise detector: {exc}") from exc
            else:
                self._logo_detector = None
            self._detector_ready = True
        return self._logo_detector

    def _prepare_region(self, logo_id: int, image: np.ndarray, bbox: BBox) -> LogoRegion:
        pad = self.config.mask_padding
        max_attempts = 3
        attempt = 0
        region = self._build_region_with_pad(logo_id, image, bbox, pad)
        while (
            attempt < max_attempts
            and self._mask_touches_border(region.mask)
            and not self._roi_hits_image_edge(region, image.shape)
        ):
            attempt += 1
            extra = max(8, max(region.mask.shape[:2]) // 6)
            new_pad = min(pad + extra, max(image.shape[0], image.shape[1]))
            if new_pad == pad:
                break
            logging.debug(
                "Region %d mask touches ROI border (attempt %d). Expanding pad from %d to %d",
                logo_id,
                attempt,
                pad,
                new_pad,
            )
            pad = new_pad
            region = self._build_region_with_pad(logo_id, image, bbox, pad)
        return region

    def _build_region_with_pad(
        self, logo_id: int, image: np.ndarray, bbox: BBox, pad: int
    ) -> LogoRegion:
        x, y, w, h = bbox
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image.shape[1], x + w + pad)
        y1 = min(image.shape[0], y + h + pad)

        roi = image[y0:y1, x0:x1].copy()
        
        if roi.shape[0] < 1 or roi.shape[1] < 1:
            logging.warning("Region %d has invalid ROI size: %s", logo_id, roi.shape)
            return LogoRegion(
                logo_id=logo_id,
                bbox=bbox,
                padded_origin=(x0, y0),
                patch=roi,
                mask=np.zeros(roi.shape[:2], dtype=np.uint8),
                masked_patch=roi,
            )
        
        rect = (x - x0, y - y0, w, h)
        
        rect_x, rect_y, rect_w, rect_h = rect
        if rect_x < 0 or rect_y < 0 or rect_w <= 0 or rect_h <= 0:
            logging.warning("Region %d has invalid rect: %s", logo_id, rect)
            return LogoRegion(
                logo_id=logo_id,
                bbox=bbox,
                padded_origin=(x0, y0),
                patch=roi,
                mask=np.full(roi.shape[:2], 255, dtype=np.uint8),
                masked_patch=roi,
            )
        
        refined_mask = None
        used_sam = False
        if self._sam_masker is not None:
            refined_mask = self._generate_sam_mask(bbox, (x0, y0, x1, y1))
            if refined_mask is not None:
                used_sam = True

        if refined_mask is None:
            refined_mask = self._apply_grabcut_with_fallbacks(
                roi, rect, logo_id, bbox
            )
            used_sam = False

        is_valid, reason = self._validate_mask_quality(refined_mask, bbox)
        if not is_valid and used_sam:
            logging.debug(
                "Region %d SAM mask rejected (%s). Falling back to GrabCut.",
                logo_id,
                reason,
            )
            refined_mask = self._apply_grabcut_with_fallbacks(
                roi, rect, logo_id, bbox
            )
            used_sam = False
            is_valid, reason = self._validate_mask_quality(refined_mask, bbox)

        if not is_valid:
            logging.warning(
                "Region %d mask quality check failed: %s. Mask coverage: %.2f%%. Using fallback.",
                logo_id, reason,
                100.0 * np.count_nonzero(refined_mask) / refined_mask.size if refined_mask.size > 0 else 0
            )
            refined_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            refined_mask[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = 255
        
        # Detect skin/hand occlusion in the masked region
        skin_ratio = 0.0
        occlusion_mask = None
        is_partially_occluded = False
        
        if self.config.detect_skin_occlusion:
            # Apply skin detection to the ROI
            refined_mask_for_inpaint, skin_ratio, skin_mask = refine_mask_excluding_skin(
                roi, refined_mask, skin_threshold=self.config.skin_detection_threshold
            )
            
            if skin_ratio >= self.config.skin_detection_threshold:
                is_partially_occluded = True
                occlusion_mask = skin_mask
                logging.info(
                    "Region %d: Detected potential hand occlusion (skin_ratio=%.2f%%), "
                    "will reduce inpaint expansion",
                    logo_id, skin_ratio * 100
                )
                # Use the refined mask (with skin removed) for display, but keep original for reflection
                # The inpainting will use the refined mask via the occlusion_mask field
        
        masked_logo = cv2.bitwise_and(roi, roi, mask=refined_mask)

        mask_coverage = 100.0 * np.count_nonzero(refined_mask) / refined_mask.size if refined_mask.size > 0 else 0
        logging.debug(
            "Prepared region %d bbox=%s padded_origin=(%d,%d) shape=%s mask_coverage=%.2f%% valid=%s skin_ratio=%.2f%%",
            logo_id,
            bbox,
            x0,
            y0,
            roi.shape,
            mask_coverage,
            is_valid,
            skin_ratio * 100,
        )

        return LogoRegion(
            logo_id=logo_id,
            bbox=bbox,
            padded_origin=(x0, y0),
            patch=roi,
            mask=refined_mask,
            masked_patch=masked_logo,
            occlusion_mask=occlusion_mask,
            skin_ratio=skin_ratio,
            is_partially_occluded=is_partially_occluded,
        )

    def _generate_sam_mask(self, bbox: BBox, roi_bounds: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if self._sam_masker is None:
            return None
        x, y, w, h = bbox
        full_mask = self._sam_masker.mask_from_box((x, y, x + w, y + h))
        if full_mask is None or full_mask.size == 0:
            return None
        x0, y0, x1, y1 = roi_bounds
        cropped = full_mask[y0:y1, x0:x1]
        if cropped.size == 0:
            return None
        if np.count_nonzero(cropped) == 0:
            return None
        cropped = cropped.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)
        return cropped

    @staticmethod
    def _mask_touches_border(mask: np.ndarray) -> bool:
        if mask.size == 0:
            return False
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            return True
        return (
            np.any(mask[0, :] > 0)
            or np.any(mask[-1, :] > 0)
            or np.any(mask[:, 0] > 0)
            or np.any(mask[:, -1] > 0)
        )

    def _validate_mask_quality(self, mask: np.ndarray, bbox: BBox) -> Tuple[bool, str]:
        """
        Validate if a mask has sufficient quality for logo flipping.
        
        Returns:
            (is_valid, reason) - True if mask is good enough, False with reason otherwise
        """
        min_coverage = self.config.min_mask_quality
        
        if mask.size == 0:
            return False, "empty mask"
        
        mask_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        coverage = mask_pixels / total_pixels if total_pixels > 0 else 0
        
        # Check minimum coverage
        if coverage < min_coverage:
            return False, f"coverage too low ({coverage*100:.1f}% < {min_coverage*100:.1f}%)"
        
        # Check if mask is almost entirely filled (likely failed segmentation)
        if coverage > 0.95:
            return False, f"coverage too high ({coverage*100:.1f}% > 95%) - likely failed GrabCut"
        
        # Check bbox coverage - mask should cover reasonable portion of bbox
        x, y, w, h = bbox
        bbox_area = w * h
        if bbox_area > 0:
            bbox_coverage = mask_pixels / bbox_area
            if bbox_coverage < 0.1:
                return False, f"bbox coverage too low ({bbox_coverage*100:.1f}%)"
        
        return True, "OK"

    @staticmethod
    def _roi_hits_image_edge(region: LogoRegion, image_shape: Tuple[int, int, int]) -> bool:
        x0, y0 = region.padded_origin
        h, w = region.patch.shape[:2]
        height, width = image_shape[:2]
        return x0 == 0 or y0 == 0 or (x0 + w) >= width or (y0 + h) >= height

    def _apply_grabcut_with_fallbacks(
        self, roi: np.ndarray, rect: Tuple[int, int, int, int], logo_id: int, bbox: BBox
    ) -> np.ndarray:
        rect_x, rect_y, rect_w, rect_h = rect
        
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(
                roi,
                mask,
                rect,
                bg_model,
                fg_model,
                iterCount=max(1, self.config.grabcut_iters),
                mode=cv2.GC_INIT_WITH_RECT,
            )
            refined_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
            
            logging.debug(
                "Region %d GrabCut succeeded with %d iterations, coverage=%.2f%%",
                logo_id,
                self.config.grabcut_iters,
                100.0 * np.count_nonzero(refined_mask) / refined_mask.size if refined_mask.size > 0 else 0
            )
            
            return refined_mask
            
        except cv2.error as exc:
            logging.warning(
                "GrabCut failed for region %d: %s. Using bbox mask fallback.",
                logo_id, exc
            )
            fallback_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            fallback_mask[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = 255
            return fallback_mask

    def _save_masked_logo(self, region: LogoRegion) -> None:
        artifact_dir = self._ensure_artifact_dir()
        masked_dir = artifact_dir / "masked_logos"
        masked_dir.mkdir(parents=True, exist_ok=True)
        path = masked_dir / f"logo_{region.logo_id:02d}_masked.png"
        # Write with proper alpha so the background stays transparent in debuggers
        masked_patch = region.masked_patch
        if masked_patch.ndim == 2:
            masked_patch = cv2.cvtColor(masked_patch, cv2.COLOR_GRAY2BGR)
        h, w = masked_patch.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = masked_patch
        rgba[..., 3] = region.mask
        cv2.imwrite(str(path), rgba)
        region.artifacts["masked_logo"] = path
        logging.info("Saved masked logo %d to %s", region.logo_id, path)

    def _prepare_artifact_dir(self, image_path: Path) -> None:
        base_dir = self.config.artifact_dir
        if base_dir is None:
            base_dir = PROJECT_ROOT / "outputs" / "logo_pipeline_artifacts"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.config.artifact_dir = base_dir
            logging.warning("artifact_dir was not provided. Falling back to %s", base_dir)
        image_dir_name = image_path.stem or image_path.name or "image"
        run_dir = base_dir / image_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._current_artifact_dir = run_dir
        logging.debug("Artifacts for %s will be stored in %s", image_path.name, run_dir)

    def _ensure_artifact_dir(self) -> Path:
        if self._current_artifact_dir is None:
            raise RuntimeError("Artifact directory requested before it was initialised.")
        return self._current_artifact_dir

    def _annotate_geometry(self, regions: Sequence[LogoRegion]) -> None:
        for region in regions:
            line = estimate_symmetry_line(region.mask)
            poly = estimate_logo_polygon(region.mask)
            region.orientation_line = line
            region.polygon = poly
            
            # Log detailed polygon info
            if poly is not None and len(poly) == 4:
                # Check if polygon is actually skewed/perspective or just a rectangle
                ordered_poly = order_box_points(poly)
                edge_lengths = [
                    np.linalg.norm(ordered_poly[1] - ordered_poly[0]),
                    np.linalg.norm(ordered_poly[2] - ordered_poly[1]),
                    np.linalg.norm(ordered_poly[3] - ordered_poly[2]),
                    np.linalg.norm(ordered_poly[0] - ordered_poly[3]),
                ]
                # Check if opposite edges are very different (indicates perspective)
                top_bottom_diff = abs(edge_lengths[0] - edge_lengths[2])
                left_right_diff = abs(edge_lengths[1] - edge_lengths[3])
                has_perspective = (top_bottom_diff > 5) or (left_right_diff > 5)
                
                logging.debug(
                    "Region %d geometry -> line=%s, polygon=%s, edges=[%.1f,%.1f,%.1f,%.1f], perspective=%s",
                    region.logo_id,
                    f"angle={line.angle_degrees:.1f}deg" if line else "None",
                    ordered_poly.tolist(),
                    edge_lengths[0], edge_lengths[1], edge_lengths[2], edge_lengths[3],
                    "YES" if has_perspective else "NO (rectangular)",
                )
            else:
                logging.debug(
                    "Region %d geometry -> line=%s polygon_pts=%d",
                    region.logo_id,
                    f"angle={line.angle_degrees:.1f}deg" if line else "None",
                    len(poly) if poly is not None else 0,
                )

    def _inpaint_background(
        self, image: np.ndarray, regions: Sequence[LogoRegion],
        hand_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inpaint logo regions from the image.
        
        Args:
            image: Input image (BGR)
            regions: List of LogoRegion objects to inpaint
            hand_mask: Optional full-frame binary mask of hand regions (255 = hand)
                      Used for two-pass inpainting to prevent hand pixels bleeding into result
        
        Returns:
            Tuple of (inpainted_image, inpaint_mask)
        """
        aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        aggregated_occlusion = np.zeros(image.shape[:2], dtype=np.uint8)
        has_occlusion = False
        
        # Determine if we have a valid hand mask from MediaPipe
        has_hand_mask = hand_mask is not None and np.count_nonzero(hand_mask) > 0
        
        # EXPLICIT DEBUG: Log hand mask status at the very start
        if hand_mask is not None:
            hand_pixel_count = np.count_nonzero(hand_mask)
            logging.info(
                "[INPAINT DEBUG] hand_mask received: shape=%s, pixels=%d, has_hand_mask=%s",
                hand_mask.shape, hand_pixel_count, has_hand_mask
            )
        else:
            logging.info("[INPAINT DEBUG] hand_mask is None - no hand-aware inpainting possible")
        
        for region in regions:
            x0, y0 = region.padded_origin
            h, w = region.mask.shape
            
            # KEY FIX for Issue 2: Always use FULL logo mask for inpainting
            # The logo should be completely removed even under hand-occluded areas
            # We'll restore hand pixels AFTER inpainting in the video pipeline
            if self.config.inpaint_full_logo_under_hand:
                # Use full mask - don't subtract occlusion
                region_mask = region.mask
                
                # Still track occlusion for logging/visualization but don't subtract
                if region.is_partially_occluded and region.occlusion_mask is not None:
                    has_occlusion = True
                    occlusion_mask_resized = region.occlusion_mask
                    if occlusion_mask_resized.shape != (h, w):
                        occlusion_mask_resized = cv2.resize(occlusion_mask_resized, (w, h))
                    aggregated_occlusion[y0 : y0 + h, x0 : x0 + w] = np.maximum(
                        aggregated_occlusion[y0 : y0 + h, x0 : x0 + w], occlusion_mask_resized
                    )
                    logging.debug(
                        "Region %d: Detected %.1f%% occlusion but inpainting full logo (inpaint_full_logo_under_hand=True)",
                        region.logo_id,
                        100 * np.count_nonzero(occlusion_mask_resized) / max(1, occlusion_mask_resized.size)
                    )
            else:
                # Legacy behavior: subtract occlusion from inpainting mask
                region_mask = region.mask
                if region.is_partially_occluded and region.occlusion_mask is not None:
                    has_occlusion = True
                    occlusion_mask_resized = region.occlusion_mask
                    if occlusion_mask_resized.shape != (h, w):
                        occlusion_mask_resized = cv2.resize(occlusion_mask_resized, (w, h))
                    region_mask = cv2.bitwise_and(
                        region.mask, 
                        cv2.bitwise_not(occlusion_mask_resized)
                    )
                    aggregated_occlusion[y0 : y0 + h, x0 : x0 + w] = np.maximum(
                        aggregated_occlusion[y0 : y0 + h, x0 : x0 + w], occlusion_mask_resized
                    )
                    logging.debug(
                        "Region %d: Excluding %.1f%% of mask from inpainting (skin/hand area)",
                        region.logo_id,
                        100 * np.count_nonzero(occlusion_mask_resized) / max(1, occlusion_mask_resized.size)
                    )
            
            aggregated_mask[y0 : y0 + h, x0 : x0 + w] = np.maximum(
                aggregated_mask[y0 : y0 + h, x0 : x0 + w], region_mask
            )

        margin = max(0, self.config.inpaint_mask_margin)
        replace_mask = aggregated_mask.copy()
        if margin > 0:
            kernel_size = margin * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            replace_mask = cv2.dilate(replace_mask, kernel, iterations=1)

        # Reduce expansion when occlusion is detected to avoid bleeding into hands
        expansion = max(0, self.config.inpaint_expansion)
        if has_occlusion and self.config.reduce_expansion_on_occlusion and not has_hand_mask:
            # Only reduce expansion if we don't have precise hand mask
            # With hand mask, we use two-pass inpainting which handles this better
            original_expansion = expansion
            expansion = max(3, expansion // 3)  # At least 3px expansion
            logging.info(
                "Occlusion detected (no hand mask): reducing inpaint_expansion from %d to %d",
                original_expansion, expansion
            )
        
        working_mask = replace_mask
        if expansion > 0:
            kernel_size = expansion * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            working_mask = cv2.dilate(replace_mask, kernel, iterations=1)
        
        # ============================================================
        # TWO-PASS INPAINTING (KEY FIX for Issue 1)
        # 
        # Problem: cv2.inpaint() samples pixels within inpaint_radius
        # from the neighborhood. If a hand is nearby, it will sample
        # hand-colored pixels, making the result look dirty.
        #
        # Solution: Two-pass approach
        # Pass 1: Create a "clean" source image where hand pixels are
        #         temporarily replaced with inpainted estimates
        # Pass 2: Run the actual logo inpainting on this clean image
        # ============================================================
        
        source_image = image.copy()
        
        # ============================================================
        # HAND-AWARE INPAINTING - APPROACH 2: Expand inpaint mask to include hand
        # 
        # Problem: cv2.inpaint samples from ALL non-masked pixels within radius.
        # Even neutralizing hand pixels first doesn't help because those
        # neutralized pixels still get sampled during logo inpainting.
        #
        # Solution: Include hand pixels IN the inpainting mask, so they are
        # ALSO inpainted (not sampled from). Then restore original hand pixels
        # after inpainting. This ensures hand colors NEVER influence the result.
        # ============================================================
        
        hand_pixels_added_to_mask = 0
        original_hand_region = None
        
        if has_hand_mask and self.config.use_two_pass_inpainting:
            # Create expanded hand exclusion zone
            buffer = max(0, self.config.hand_exclusion_buffer)
            if buffer > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (buffer * 2 + 1, buffer * 2 + 1)
                )
                hand_exclusion_zone = cv2.dilate(hand_mask, kernel, iterations=1)
            else:
                hand_exclusion_zone = hand_mask.copy()
            
            # Find hand pixels that are NEAR logo regions (within inpaint_radius + buffer)
            # These are the pixels that could "bleed" into the inpainted result
            logo_expanded = cv2.dilate(
                working_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                    (self.config.inpaint_radius * 2 + buffer * 2 + 1, 
                     self.config.inpaint_radius * 2 + buffer * 2 + 1)),
                iterations=1
            )
            
            # Hand pixels that could affect logo inpainting
            hand_near_logo = cv2.bitwise_and(hand_exclusion_zone, logo_expanded)
            hand_pixels_near_logo = np.count_nonzero(hand_near_logo)
            
            if hand_pixels_near_logo > 0:
                logging.info(
                    "Hand-aware inpainting: %d hand pixels within %dpx of logo, adding to inpaint mask",
                    hand_pixels_near_logo, self.config.inpaint_radius + buffer
                )
                
                # Save original hand pixels so we can restore them after inpainting
                original_hand_region = image.copy()
                
                # ADD hand pixels to the inpainting mask - they will be inpainted over
                # This ensures they are NOT used as source pixels for inpainting
                working_mask = cv2.bitwise_or(working_mask, hand_near_logo)
                hand_pixels_added_to_mask = hand_pixels_near_logo
                
                logging.debug(
                    "Expanded inpaint mask by %d hand pixels (total mask: %d pixels)",
                    hand_pixels_added_to_mask,
                    np.count_nonzero(working_mask)
                )
        
        # Inpaint logo regions (and hand pixels if they were added to mask)
        # Use LaMa ML inpainting if available, otherwise fall back to cv2.inpaint
        radius = max(1, self.config.inpaint_radius)
        
        if self._lama_inpainter is not None:
            # LaMa ML inpainting - better for textures like fabric
            # LaMa works with RGB, so convert BGR->RGB and back
            try:
                source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
                source_pil = PILImage.fromarray(source_rgb)
                mask_pil = PILImage.fromarray(working_mask)
                
                result_pil = self._lama_inpainter(source_pil, mask_pil)
                result_rgb = np.array(result_pil)
                inpainted_full = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                
                logging.debug("LaMa ML inpainting completed successfully")
            except Exception as e:
                logging.warning(f"LaMa inpainting failed: {e}. Falling back to cv2.inpaint.")
                inpainted_full = cv2.inpaint(source_image, working_mask, radius, cv2.INPAINT_TELEA)
        else:
            # Traditional cv2.inpaint (Telea algorithm)
            inpainted_full = cv2.inpaint(source_image, working_mask, radius, cv2.INPAINT_TELEA)
        
        if margin > 0 or expansion > 0:
            inpainted = image.copy()
            inpainted[replace_mask > 0] = inpainted_full[replace_mask > 0]
        else:
            inpainted = inpainted_full
        
        # CRITICAL: Restore original hand pixels after inpainting
        # This ensures hands look natural while logos are cleanly removed
        if original_hand_region is not None and hand_pixels_added_to_mask > 0:
            # Restore ONLY the original hand mask (not the dilated exclusion zone)
            # This gives us clean logo removal + natural hands
            hand_bool = hand_mask > 127
            inpainted[hand_bool] = original_hand_region[hand_bool]
            logging.info(
                "Restored %d original hand pixels after inpainting",
                np.count_nonzero(hand_bool)
            )
        
        logging.info(
            "Inpainted background with %s radius=%d expansion=%d margin=%d over %d regions "
            "(occlusion=%s, hand_mask=%s, two_pass=%s)",
            "LaMa" if self._lama_inpainter is not None else "cv2.TELEA",
            radius,
            expansion,
            margin,
            len(regions),
            "detected" if has_occlusion else "none",
            "yes" if has_hand_mask else "no",
            "yes" if (has_hand_mask and self.config.use_two_pass_inpainting) else "no",
        )
        return inpainted, working_mask

    def _reflect_region(self, region: LogoRegion) -> Tuple[np.ndarray, np.ndarray]:
        mode = self._resolve_naturalise_mode(region)
        use_perspective = self.config.use_perspective_flip
        
        if mode == "polygon" and region.polygon is not None:
            return naturalise_with_polygon(region.patch, region.mask, region.polygon, use_perspective)
        if region.orientation_line is None:
            logging.warning(
                "Region %d lacks a symmetry line; falling back to polygon mode.",
                region.logo_id,
            )
            if region.polygon is not None:
                return naturalise_with_polygon(region.patch, region.mask, region.polygon, use_perspective)
            return region.patch.copy(), region.mask.copy()
        return reflect_along_line(region.patch, region.mask, region.orientation_line)

    def _resolve_naturalise_mode(self, region: LogoRegion) -> str:
        mode = self.config.naturalise_mode.lower()
        if mode == "auto":
            if region.polygon is not None and len(region.polygon) >= 4:
                return "polygon"
            return "line"
        return mode

    def _paste_region(self, canvas: np.ndarray, region: LogoRegion, occlusion_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if region.reflected_patch is None or region.reflected_mask is None:
            raise RuntimeError(f"Region {region.logo_id} is missing reflected data.")
        
        patch = region.reflected_patch
        mask = region.reflected_mask
        blend_mode = self.config.blend_mode.lower()
        
        # Check if mask has any content
        mask_pixels = np.count_nonzero(mask)
        if mask_pixels == 0:
            logging.warning(
                "Region %d has empty mask after reflection - skipping paste.",
                region.logo_id,
            )
            return canvas
        
        # Calculate centroid of original mask in full image coordinates
        x0, y0 = region.padded_origin
        orig_moments = cv2.moments(region.mask)
        if orig_moments['m00'] == 0:
            # Fallback to geometric center if moments fail
            orig_cy = region.mask.shape[0] // 2
            orig_cx = region.mask.shape[1] // 2
        else:
            orig_cx = orig_moments['m10'] / orig_moments['m00']
            orig_cy = orig_moments['m01'] / orig_moments['m00']
        orig_centroid_x = int(round(x0 + orig_cx))
        orig_centroid_y = int(round(y0 + orig_cy))
        
        # Calculate centroid of reflected mask
        refl_moments = cv2.moments(mask)
        if refl_moments['m00'] == 0:
            # Fallback to geometric center
            refl_cy = mask.shape[0] // 2
            refl_cx = mask.shape[1] // 2
        else:
            refl_cx = refl_moments['m10'] / refl_moments['m00']
            refl_cy = refl_moments['m01'] / refl_moments['m00']
        
        # Calculate offset to align centroids
        offset_x = orig_centroid_x - int(round(refl_cx))
        offset_y = orig_centroid_y - int(round(refl_cy))
        
        h, w = mask.shape
        canvas_h, canvas_w = canvas.shape[:2]
        
        # Calculate target region, clipping to canvas bounds
        dst_x1 = max(0, offset_x)
        dst_y1 = max(0, offset_y)
        dst_x2 = min(canvas_w, offset_x + w)
        dst_y2 = min(canvas_h, offset_y + h)
        
        # Calculate corresponding source region
        src_x1 = dst_x1 - offset_x
        src_y1 = dst_y1 - offset_y
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            logging.warning(
                "Region %d reflected patch is completely outside canvas bounds - skipping.",
                region.logo_id,
            )
            return canvas
        
        patch_roi = patch[src_y1:src_y2, src_x1:src_x2]
        mask_roi = mask[src_y1:src_y2, src_x1:src_x2]
        
        logging.debug(
            "Pasting region %d: orig_centroid=(%d,%d), refl_centroid=(%.1f,%.1f), offset=(%d,%d), dst=(%d,%d,%d,%d)",
            region.logo_id,
            orig_centroid_x, orig_centroid_y,
            refl_cx, refl_cy,
            offset_x, offset_y,
            dst_x1, dst_y1, dst_x2, dst_y2,
        )

        if blend_mode == "seamless":
            # For seamless clone, we need to work with full patch and provide center in canvas coords
            center = (orig_centroid_x, orig_centroid_y)
            try:
                blended = cv2.seamlessClone(
                    patch,
                    canvas,
                    mask,
                    center,
                    cv2.NORMAL_CLONE,
                )
                canvas[:] = blended
                logging.debug("Region %d pasted using seamless clone.", region.logo_id)
                return canvas
            except cv2.error as exc:
                logging.warning(
                    "seamlessClone failed for region %d (%s); using alpha blend.",
                    region.logo_id,
                    exc,
                )
                blend_mode = "alpha"

        # Alpha blend the ROI
        canvas_roi = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
        alpha = (mask_roi.astype(np.float32) / 255.0)[..., None]
        
        # Apply occlusion mask if provided (don't paste over occluded regions)
        if occlusion_mask is not None and occlusion_mask.shape[:2] == mask_roi.shape[:2]:
            occlusion_float = (255 - occlusion_mask).astype(np.float32) / 255.0  # Invert: 0 where occluded
            alpha = alpha * occlusion_float[..., None]
            occluded_ratio = np.count_nonzero(occlusion_mask) / occlusion_mask.size
            logging.debug(
                f"Logo {region.logo_id}: Applied occlusion mask, "
                f"masking {occluded_ratio*100:.1f}% of logo area"
            )
        
        blended_roi = alpha * patch_roi.astype(np.float32) + (1 - alpha) * canvas_roi.astype(np.float32)
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)
        logging.debug("Region %d pasted using alpha blend.", region.logo_id)
        return canvas

    def _save_overlay(
        self, image: np.ndarray, regions: Sequence[LogoRegion], name: str
    ) -> None:
        overlay = image.copy()
        mask_layer = np.zeros_like(image)
        colors = self._palette()
        for region in regions:
            color = colors[region.logo_id % len(colors)]
            x, y, w, h = region.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            label_parts: List[str] = []
            if region.detection_sources:
                label_parts.append("+".join(sorted(region.detection_sources)))
            if region.detection_confidence:
                label_parts.append(f"{region.detection_confidence:.2f}")
            if label_parts:
                label = " ".join(label_parts)
                cv2.putText(
                    overlay,
                    label,
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            if region.mask is not None:
                x0, y0 = region.padded_origin
                h_mask, w_mask = region.mask.shape
                mask = region.mask
                colored = np.zeros((h_mask, w_mask, 3), dtype=np.uint8)
                colored[:] = color
                mask_layer[y0 : y0 + h_mask, x0 : x0 + w_mask] = np.where(
                    mask[..., None] > 0, colored, mask_layer[y0 : y0 + h_mask, x0 : x0 + w_mask]
                )
            if region.orientation_line is not None:
                draw_symmetry_line(
                    overlay,
                    region.orientation_line,
                    origin_offset=region.padded_origin,
                    color=color,
                )
            if region.polygon is not None:
                draw_polygon(
                    overlay,
                    region.polygon,
                    origin_offset=region.padded_origin,
                    color=color,
                )

        blended = cv2.addWeighted(overlay, 0.7, mask_layer, 0.3, 0)
        self._save_image(blended, f"{name}.png")

    def _save_mask_image(self, mask: np.ndarray, name: str) -> None:
        colored = cv2.applyColorMap(mask, cv2.COLORMAP_VIRIDIS)
        self._save_image(colored, name)

    def _save_image(self, image: np.ndarray, name: str) -> None:
        artifact_dir = self._ensure_artifact_dir()
        path = artifact_dir / name
        cv2.imwrite(str(path), image)
        logging.info("Saved artifact %s", path)

    @staticmethod
    def _palette() -> List[Tuple[int, int, int]]:
        return [
            (255, 99, 71),  # tomato
            (135, 206, 235),  # sky blue
            (60, 179, 113),  # medium sea green
            (238, 130, 238),  # violet
            (255, 215, 0),  # gold
            (70, 130, 180),  # steel blue
        ]


def estimate_symmetry_line(mask: np.ndarray, max_angle_from_vertical: float = 45.0) -> Optional[SymmetryLine]:
    """
    Estimate the symmetry/reflection line for a logo mask using PCA.
    
    Args:
        mask: Binary mask of the logo
        max_angle_from_vertical: Maximum allowed angle deviation from vertical (degrees).
            The reflection axis should be mostly vertical (for horizontal flip).
            If PCA produces an axis more than this many degrees from vertical,
            it will be clamped to prevent accidental vertical flips.
            Default 45° means axis must be within ±45° of vertical.
    
    Returns:
        SymmetryLine or None if estimation fails
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    if eigenvectors is None or len(eigenvectors) == 0:
        return None
    # The principal component aligns with the longest axis of the logo patch,
    # whereas the mirror axis is orthogonal to it. Use the minor component (or
    # a constructed perpendicular if PCA returned a single vector).
    if len(eigenvectors) >= 2:
        direction = eigenvectors[1]
    else:
        direction = eigenvectors[0]
        direction = np.array([-direction[1], direction[0]], dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    
    # Normalize direction
    dx, dy = float(direction[0] / norm), float(direction[1] / norm)
    
    # Calculate angle from vertical (vertical = 90°, so we want angle close to ±90°)
    # atan2(dy, dx) gives angle from horizontal axis
    angle_from_horizontal = math.degrees(math.atan2(dy, dx))
    # Convert to angle from vertical: vertical is at ±90°
    angle_from_vertical = abs(abs(angle_from_horizontal) - 90.0)
    
    # If the angle is too far from vertical (i.e., too horizontal),
    # the flip would be more vertical than horizontal - clamp it
    if angle_from_vertical > max_angle_from_vertical:
        logging.warning(
            f"Symmetry line angle {angle_from_horizontal:.1f}° is {angle_from_vertical:.1f}° from vertical "
            f"(max allowed: {max_angle_from_vertical}°). Clamping to prevent vertical flip."
        )
        # Clamp to max_angle_from_vertical degrees from vertical
        # Vertical is at 90° or -90°, so clamp to (90 - max) or -(90 - max)
        clamped_angle = 90.0 - max_angle_from_vertical
        if angle_from_horizontal < 0:
            clamped_angle = -clamped_angle
        # If original was closer to 0° (horizontal), flip to the closer vertical side
        if abs(angle_from_horizontal) < 45:
            # Was nearly horizontal, pick the closest vertical direction
            clamped_angle = 90.0 - max_angle_from_vertical if angle_from_horizontal >= 0 else -(90.0 - max_angle_from_vertical)
        
        clamped_rad = math.radians(clamped_angle)
        dx = math.cos(clamped_rad)
        dy = math.sin(clamped_rad)
        logging.info(f"Clamped symmetry line to {clamped_angle:.1f}° (dx={dx:.3f}, dy={dy:.3f})")
    
    direction = (dx, dy)
    origin = (float(mean[0, 0]), float(mean[0, 1]))
    return SymmetryLine(origin, direction)


def estimate_logo_polygon(mask: np.ndarray) -> Optional[np.ndarray]:
    """Extract quadrilateral boundary from mask, preserving perspective."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20.0:
        return None
    
    # First try to fit a quadrilateral directly to the contour WITHOUT convex hull
    # This preserves the actual shape better, including perspective distortion
    peri = cv2.arcLength(contour, True)
    
    for epsilon_factor in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]:
        epsilon = epsilon_factor * peri
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            unique_points = np.unique(quad, axis=0)
            if len(unique_points) == 4:
                area = cv2.contourArea(quad)
                if area > 0:
                    logging.debug(
                        "Found 4-point polygon approximation (epsilon=%.4f, area=%.1f)",
                        epsilon_factor, area
                    )
                    return quad
    
    hull = cv2.convexHull(contour)
    peri_hull = cv2.arcLength(hull, True)
    
    for epsilon_factor in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08]:
        epsilon = epsilon_factor * peri_hull
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            unique_points = np.unique(quad, axis=0)
            if len(unique_points) == 4:
                logging.debug(
                    "Found 4-point hull approximation (epsilon=%.4f)",
                    epsilon_factor
                )
                return quad
    
    # Last resort: use minAreaRect (this will NOT capture perspective)
    logging.warning("Using minAreaRect fallback - perspective will not be preserved")
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    return box


def _major_axis_direction(points: np.ndarray) -> Optional[np.ndarray]:
    pts = points.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 2:
        return None
    mean, eigenvectors = cv2.PCACompute(pts, mean=None, maxComponents=2)
    if eigenvectors is None or len(eigenvectors) == 0:
        return None
    direction = eigenvectors[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    return direction / norm


def reflect_along_line(
    patch: np.ndarray, mask: np.ndarray, line: SymmetryLine
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        line = line.as_unit()
    except ValueError:
        return patch.copy(), mask.copy()

    if patch.size == 0 or mask.size == 0:
        return patch.copy(), mask.copy()

    h, w = mask.shape[:2]
    origin = np.array(line.origin, dtype=np.float32)
    direction = np.array(line.direction, dtype=np.float32)

    # Clamp the origin to the patch bounds to avoid runaway translations
    origin[0] = np.clip(origin[0], 0.0, float(w - 1))
    origin[1] = np.clip(origin[1], 0.0, float(h - 1))

    ux, uy = direction
    # Reflection matrix across an arbitrary axis that passes through the origin
    reflection = np.array(
        [
            [2.0 * ux * ux - 1.0, 2.0 * ux * uy],
            [2.0 * ux * uy, 2.0 * uy * uy - 1.0],
        ],
        dtype=np.float32,
    )
    translation = origin - reflection @ origin
    transform = np.hstack([reflection, translation.reshape(2, 1)])
    
    # Calculate bounding box of reflected mask to avoid clipping
    # Get all mask points and transform them to find required output size
    mask_points = np.column_stack(np.where(mask > 0)[::-1]).astype(np.float32)  # (x, y) format
    if len(mask_points) > 0:
        # Transform all mask points
        ones = np.ones((len(mask_points), 1), dtype=np.float32)
        mask_points_homogeneous = np.hstack([mask_points, ones])
        transformed_points = (transform @ mask_points_homogeneous.T).T
        
        # Find bounding box of transformed points
        min_x = int(np.floor(np.min(transformed_points[:, 0])))
        max_x = int(np.ceil(np.max(transformed_points[:, 0])))
        min_y = int(np.floor(np.min(transformed_points[:, 1])))
        max_y = int(np.ceil(np.max(transformed_points[:, 1])))
        
        # Calculate required output size and offset
        offset_x = max(0, -min_x)
        offset_y = max(0, -min_y)
        out_w = max(w, max_x + offset_x + 1)
        out_h = max(h, max_y + offset_y + 1)
        
        # Adjust transform to account for offset
        if offset_x > 0 or offset_y > 0:
            offset_matrix = np.array(
                [
                    [1, 0, offset_x],
                    [0, 1, offset_y],
                ],
                dtype=np.float32,
            )
            transform = offset_matrix @ np.vstack([transform, [0, 0, 1]])
        
        size = (out_w, out_h)
        logging.debug(
            "Reflection output size: %dx%d (original: %dx%d, offset: %d,%d)",
            out_w, out_h, w, h, offset_x, offset_y
        )
    else:
        # Fallback if mask is empty
        size = (w, h)

    reflected_patch = cv2.warpAffine(
        patch,
        transform,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    reflected_mask = cv2.warpAffine(
        mask,
        transform,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    reflected_mask = np.where(reflected_mask > 127, 255, 0).astype(np.uint8)

    return reflected_patch, reflected_mask


def naturalise_with_polygon(
    patch: np.ndarray, mask: np.ndarray, polygon: np.ndarray, use_perspective: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip logo with optional perspective correction."""
    if polygon.shape[0] < 4:
        return patch.copy(), mask.copy()
    
    # Log initial mask coverage
    initial_mask_pixels = np.count_nonzero(mask)
    logging.debug(
        "Polygon transform input: mask_coverage=%.2f%%, polygon_points=%d",
        100.0 * initial_mask_pixels / mask.size if mask.size > 0 else 0,
        len(polygon),
    )
    
    # Check if polygon has perspective distortion
    has_perspective = has_perspective_distortion(polygon, threshold=0.15)
    
    if has_perspective and use_perspective:
        logging.debug("Detected perspective distortion - using unwarp-flip-rewarp")
        return _perspective_aware_flip(patch, mask, polygon)
    else:
        # For mirror correction, we ALWAYS want horizontal flip (left-right).
        # Vertical flip would flip the logo upside down, which is wrong.
        # The major axis direction was previously used to choose flip direction,
        # but this caused bugs where tall logos got vertically flipped.
        flip_code = 1  # Always horizontal flip for mirror correction
        
        logging.debug(
            "Using simple horizontal flip (code=%d, perspective=%s)",
            flip_code,
            "detected but disabled" if has_perspective else "not detected"
        )
        return cv2.flip(patch, flip_code), cv2.flip(mask, flip_code)


def _perspective_aware_flip(
    patch: np.ndarray, mask: np.ndarray, polygon: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Unwarp logo to canonical view, flip, and rewarp to preserve perspective."""
    src_points = order_box_points(polygon)
    
    width_top = np.linalg.norm(src_points[1] - src_points[0])
    width_bottom = np.linalg.norm(src_points[2] - src_points[3])
    width = int(max(width_top, width_bottom))
    
    height_left = np.linalg.norm(src_points[3] - src_points[0])
    height_right = np.linalg.norm(src_points[2] - src_points[1])
    height = int(max(height_left, height_right))
    
    width = max(width, 10)
    height = max(height, 10)
    
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    M_unwarp = cv2.getPerspectiveTransform(src_points, dst_points)
    
    unwarped_patch = cv2.warpPerspective(
        patch, M_unwarp, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    unwarped_mask = cv2.warpPerspective(
        mask, M_unwarp, (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    flipped_patch = cv2.flip(unwarped_patch, 1)
    flipped_mask = cv2.flip(unwarped_mask, 1)
    
    dst_points_flipped = np.array([
        [width - 1, 0],
        [0, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ], dtype=np.float32)
    
    M_rewarp = cv2.getPerspectiveTransform(dst_points_flipped, src_points)
    
    h, w = patch.shape[:2]
    rewarped_patch = cv2.warpPerspective(
        flipped_patch, M_rewarp, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    rewarped_mask = cv2.warpPerspective(
        flipped_mask, M_rewarp, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    rewarped_mask = np.where(rewarped_mask > 127, 255, 0).astype(np.uint8)
    
    logging.debug(
        "Perspective flip: src_poly=%s, canonical=%dx%d, mask_coverage=%.2f%%",
        src_points.tolist(),
        width, height,
        100.0 * np.count_nonzero(rewarped_mask) / rewarped_mask.size if rewarped_mask.size > 0 else 0
    )
    
    return rewarped_patch, rewarped_mask



def order_box_points(points: np.ndarray) -> np.ndarray:
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(s)]  # top-left
    ordered[2] = points[np.argmax(s)]  # bottom-right
    ordered[1] = points[np.argmin(diff)]  # top-right
    ordered[3] = points[np.argmax(diff)]  # bottom-left
    return ordered


def has_perspective_distortion(polygon: np.ndarray, threshold: float = 0.15) -> bool:
    """
    Detect if a quadrilateral has significant perspective distortion.
    
    Args:
        polygon: 4-point polygon (quadrilateral)
        threshold: Relative difference threshold (e.g., 0.15 = 15% difference)
    
    Returns:
        True if polygon has significant perspective distortion
    """
    if polygon.shape[0] != 4:
        return False
    
    ordered = order_box_points(polygon)
    
    # Calculate edge lengths
    edge_lengths = [
        np.linalg.norm(ordered[1] - ordered[0]),  # top
        np.linalg.norm(ordered[2] - ordered[1]),  # right
        np.linalg.norm(ordered[3] - ordered[2]),  # bottom
        np.linalg.norm(ordered[0] - ordered[3]),  # left
    ]
    
    # Check if opposite edges differ significantly (indicates perspective)
    top_bottom_ratio = edge_lengths[0] / max(edge_lengths[2], 1e-6)
    left_right_ratio = edge_lengths[1] / max(edge_lengths[3], 1e-6)
    
    # If ratio deviates from 1.0 by more than threshold, it's perspective
    top_bottom_distorted = abs(top_bottom_ratio - 1.0) > threshold
    left_right_distorted = abs(left_right_ratio - 1.0) > threshold
    
    return top_bottom_distorted or left_right_distorted


def draw_symmetry_line(
    image: np.ndarray,
    line: Optional[SymmetryLine],
    origin_offset: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    if line is None:
        return
    try:
        normalized = line.as_unit()
    except ValueError:
        return
    length = max(image.shape[:2]) * 0.1
    origin = np.array(normalized.origin) + np.array([origin_offset[0], origin_offset[1]])
    direction = np.array(normalized.direction) * length
    p1 = (origin - direction).astype(int)
    p2 = (origin + direction).astype(int)
    cv2.line(image, tuple(p1), tuple(p2), color, 2, cv2.LINE_AA)


def draw_polygon(
    image: np.ndarray,
    polygon: Optional[np.ndarray],
    origin_offset: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    if polygon is None or len(polygon) == 0:
        return
    shifted = polygon + np.array([[origin_offset[0], origin_offset[1]]], dtype=np.float32)
    cv2.polylines(
        image,
        [shifted.astype(np.int32)],
        isClosed=True,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline to flip mirrored logos with optional naturalisation."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input image.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output path.")
    parser.add_argument(
        "--bbox-json",
        type=Path,
        help="Optional JSON file with pre-computed bounding boxes (bypasses detectors).",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        help="YOLOv7 weights to run automatic logo detection.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.50,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold for NMS.",
    )
    parser.add_argument(
        "--yolo-img-size",
        type=int,
        default=1280,
        help="Inference size for YOLO.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Torch device for YOLO inference, e.g., '0' or 'cpu'.",
    )
    parser.add_argument(
        "--use-easyocr",
        action="store_true",
        help="Augment YOLO detections with EasyOCR text regions.",
    )
    parser.add_argument(
        "--easyocr-lang",
        action="append",
        dest="easyocr_langs",
        default=None,
        help="Language code for EasyOCR (repeatable). Defaults to 'en'.",
    )
    parser.add_argument(
        "--easyocr-conf",
        type=float,
        default=0.1,
        help="Minimum EasyOCR confidence (0-1).",
    )
    parser.add_argument(
        "--easyocr-min-size",
        type=int,
        default=1280,
        help="Upscale images below this long-edge size before OCR.",
    )
    parser.add_argument(
        "--easyocr-max-size",
        type=int,
        default=1920,
        help="Downscale images above this long-edge size before OCR.",
    )
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=12,
        help="Gap (pixels) for merging overlapping detections.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory to store masked logos and intermediate artifacts.",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Persist overlays/masks/inpaint previews for inspection.",
    )
    parser.add_argument(
        "--naturalise-mode",
        choices=["auto", "line", "polygon"],
        default="auto",
        help="Choose how to flip and reshape the logo patches.",
    )
    parser.add_argument(
        "--blend-mode",
        choices=["seamless", "alpha"],
        default="seamless",
        help="Blending strategy when pasting the corrected logo.",
    )
    parser.add_argument(
        "--use-perspective-flip",
        action="store_true",
        default=True,
        help="Enable perspective-aware flipping (unwarp-flip-rewarp) for distorted logos.",
    )
    parser.add_argument(
        "--no-perspective-flip",
        action="store_false",
        dest="use_perspective_flip",
        help="Disable perspective-aware flipping (use simple flip instead).",
    )
    parser.add_argument("--mask-padding", type=int, default=12, help="Padding (px).")
    parser.add_argument(
        "--grabcut-iters", type=int, default=5, help="Iteration count for GrabCut."
    )
    parser.add_argument(
        "--inpaint-radius", type=int, default=5, help="Radius for Telea inpainting."
    )
    parser.add_argument(
        "--inpaint-expansion",
        type=int,
        default=10,
        help="Additional dilation (px) applied to the inpaint mask to sample more context.",
    )
    parser.add_argument(
        "--inpaint-mask-margin",
        type=int,
        default=4,
        help="Extra dilation (px) applied when removing the logo to cover imperfect masks.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity level.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(levelname)s] %(message)s",
    )

    artifact_dir = args.artifact_dir or (args.output.parent / f"{args.output.stem}_artifacts")
    easyocr_langs = tuple(args.easyocr_langs) if args.easyocr_langs else ("en",)
    detection_settings = DetectionSettings(
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_img_size=args.yolo_img_size,
        yolo_device=args.device,
        use_easyocr=args.use_easyocr,
        easyocr_langs=easyocr_langs,
        easyocr_conf=args.easyocr_conf,
        easyocr_min_size=args.easyocr_min_size,
        easyocr_max_size=args.easyocr_max_size,
        merge_gap=args.merge_gap,
    )
    config = PipelineConfig(
        mask_padding=max(0, args.mask_padding),
        grabcut_iters=max(1, args.grabcut_iters),
        inpaint_radius=max(1, args.inpaint_radius),
        inpaint_expansion=max(0, args.inpaint_expansion),
        inpaint_mask_margin=max(0, args.inpaint_mask_margin),
        naturalise_mode=args.naturalise_mode,
        blend_mode=args.blend_mode,
        use_perspective_flip=args.use_perspective_flip,
        artifact_dir=artifact_dir,
        save_intermediates=args.save_intermediates,
        bbox_json=args.bbox_json,
        detection=detection_settings,
    )

    pipeline = LogoFlippingPipeline(config)
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()
