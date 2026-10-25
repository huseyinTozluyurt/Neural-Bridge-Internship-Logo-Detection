"""
Wrapper around Ultralytics MobileSAM for bbox-prompted logo masks.

The class intentionally mirrors the tiny API we need inside the flipping
pipeline: set the current frame once, then ask for a mask per bounding box.

Enhanced with:
- Multi-point prompting for better segmentation
- Optional GrabCut refinement for cleaner edges
- Morphological cleanup
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import logging

import cv2
import numpy as np

try:  # pragma: no cover - requires optional dependency
    from ultralytics.models.sam import Predictor as SAMPredictor
except Exception:  # pragma: no cover - environment specific
    SAMPredictor = None


@dataclass
class SAMConfig:
    model_path: str = "mobile_sam.pt"
    device: Optional[str] = None
    image_size: int = 512
    
    # Multi-point prompting strategy
    prompt_strategy: Literal["box_only", "center_only", "center_corners", "grid_3x3"] = "center_corners"
    
    # Refinement options
    use_grabcut_refinement: bool = True
    grabcut_iterations: int = 3
    
    # Morphological cleanup
    morph_close_size: int = 5
    morph_open_size: int = 3


class MobileSAMMasker:
    """Thin wrapper for Ultralytics SAM predictor with enhanced prompting."""

    def __init__(self, config: Optional[SAMConfig] = None) -> None:
        if SAMPredictor is None:
            raise ImportError(
                "Ultralytics is required for MobileSAM (pip install ultralytics)."
            )

        self.config = config or SAMConfig()
        overrides = dict(
            task="segment",
            mode="predict",
            model=self.config.model_path,
            imgsz=self.config.image_size,
            save=False,  # Disable saving images to runs directory
            project=None,  # Disable project directory creation
            exist_ok=True,  # Avoid errors if directories exist
            verbose=False,  # Reduce console output
        )
        if self.config.device is not None:
            overrides["device"] = self.config.device
        self._predictor = SAMPredictor(overrides=overrides)
        self._current_shape: Optional[Tuple[int, int]] = None
        self._current_image: Optional[np.ndarray] = None

    def set_image(self, frame_bgr: np.ndarray) -> None:
        self._predictor.set_image(frame_bgr)
        self._current_shape = frame_bgr.shape[:2]
        self._current_image = frame_bgr.copy()

    def _generate_point_prompts(self, box_xyxy) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate point prompts based on configured strategy."""
        x1, y1, x2, y2 = map(float, box_xyxy)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        strategy = self.config.prompt_strategy
        
        if strategy == "box_only":
            return None, None
        elif strategy == "center_only":
            return np.array([[cx, cy]]), np.array([1])
        elif strategy == "center_corners":
            # Center + 4 corners inset by 20%
            inset = 0.2
            points = np.array([
                [cx, cy],  # center
                [x1 + w * inset, y1 + h * inset],  # top-left
                [x2 - w * inset, y1 + h * inset],  # top-right
                [x1 + w * inset, y2 - h * inset],  # bottom-left
                [x2 - w * inset, y2 - h * inset],  # bottom-right
            ])
            return points, np.ones(5, dtype=np.int32)
        elif strategy == "grid_3x3":
            # 3x3 grid inside bbox
            points = []
            for i in range(3):
                for j in range(3):
                    px = x1 + (i + 0.5) * w / 3
                    py = y1 + (j + 0.5) * h / 3
                    points.append([px, py])
            return np.array(points), np.ones(9, dtype=np.int32)
        
        return None, None

    def _refine_with_grabcut(self, mask: np.ndarray, box_xyxy) -> np.ndarray:
        """Refine SAM mask using GrabCut with mask initialization."""
        if self._current_image is None:
            return mask
        
        h, w = self._current_shape
        x1, y1, x2, y2 = map(int, box_xyxy)
        
        # Initialize GrabCut mask from SAM output
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        gc_mask[mask > 0] = cv2.GC_PR_FGD  # Probable foreground
        gc_mask[mask == 0] = cv2.GC_PR_BGD  # Probable background
        
        # Set definite background outside bbox with margin
        margin = 5
        definite_bg = np.ones((h, w), dtype=bool)
        y1_safe, y2_safe = max(0, y1 - margin), min(h, y2 + margin)
        x1_safe, x2_safe = max(0, x1 - margin), min(w, x2 + margin)
        definite_bg[y1_safe:y2_safe, x1_safe:x2_safe] = False
        gc_mask[definite_bg] = cv2.GC_BGD
        
        # Run GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(
                self._current_image, gc_mask, None, 
                bgd_model, fgd_model,
                self.config.grabcut_iterations, 
                cv2.GC_INIT_WITH_MASK
            )
            
            # Extract final mask
            refined_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)
            
            return refined_mask
        except Exception as e:
            logging.debug(f"GrabCut refinement failed: {e}")
            return mask

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological cleanup to mask."""
        if self.config.morph_close_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.morph_close_size, self.config.morph_close_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        if self.config.morph_open_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.morph_open_size, self.config.morph_open_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def mask_from_box(self, box_xyxy) -> np.ndarray:
        if self._current_shape is None:
            raise RuntimeError("set_image must be called before mask_from_box.")

        x1, y1, x2, y2 = map(float, box_xyxy)
        h, w = self._current_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Generate point prompts
        points, labels = self._generate_point_prompts(box_xyxy)
        
        # Call SAM with bbox and optional points
        if points is not None:
            # Note: Ultralytics SAM API may vary - try with points
            try:
                results = self._predictor(
                    bboxes=[x1, y1, x2, y2],
                    points=points.tolist(),
                    labels=labels.tolist()
                )
            except Exception:
                # Fallback to bbox only
                results = self._predictor(bboxes=[x1, y1, x2, y2])
        else:
            results = self._predictor(bboxes=[x1, y1, x2, y2])

        if not results or results[0].masks is None or results[0].masks.data is None:
            return mask

        # Get the best mask (highest score if multiple)
        mask_data = results[0].masks.data.cpu().numpy()
        if len(mask_data) > 1 and hasattr(results[0], 'scores'):
            scores = results[0].scores.cpu().numpy() if results[0].scores is not None else None
            if scores is not None:
                best_idx = np.argmax(scores)
                first = mask_data[best_idx]
            else:
                first = mask_data[0]
        else:
            first = mask_data[0]
        
        mask = (first > 0.5).astype(np.uint8) * 255
        
        # Apply GrabCut refinement if enabled
        if self.config.use_grabcut_refinement:
            mask = self._refine_with_grabcut(mask, box_xyxy)
        
        # Apply morphological cleanup
        mask = self._apply_morphology(mask)
        
        return mask

    def segment_everything(self, max_masks: Optional[int] = None) -> List[np.ndarray]:
        """
        Run SAM in "segment everything" mode and return the binary masks.

        Args:
            max_masks: Optional cap on how many masks to return. This prevents
                pathological cases where SAM produces hundreds of tiny segments.
        """
        if self._current_shape is None:
            raise RuntimeError("set_image must be called before segment_everything.")

        results = self._predictor()
        if not results or results[0].masks is None or results[0].masks.data is None:
            return []

        data = results[0].masks.data.cpu().numpy()
        masks: List[np.ndarray] = []
        limit = data.shape[0] if max_masks is None else min(max_masks, data.shape[0])
        for idx in range(limit):
            mask_np = (data[idx] > 0.5).astype(np.uint8) * 255
            masks.append(mask_np)
        return masks


def create_sam_masker(config: Optional[SAMConfig] = None) -> Optional[MobileSAMMasker]:
    try:
        return MobileSAMMasker(config)
    except ImportError as exc:  # pragma: no cover - environment specific
        logging.warning("SAM masking disabled: %s", exc)
        return None
