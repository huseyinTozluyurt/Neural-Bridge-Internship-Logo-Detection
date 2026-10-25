#!/usr/bin/env python3
"""
Video logo tracking and flipping pipeline.

This script processes a video to:
1. Detect and track logos across frames using optical flow
2. Merge nested bounding boxes
3. Generate two output videos:
   - Visualization video: Shows bboxes with IDs, confidence, and centroid traces
   - Clean video: Flipped logos without any overlays
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import mediapipe as mp
MEDIAPIPE_AVAILABLE = True

# Import the logo flipping pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.logo_flipping_pipeline import (
    DetectionSettings,
    LogoFlippingPipeline,
    PipelineConfig,
    estimate_symmetry_line,
    reflect_along_line,
)
from scripts.clip_embedder import CLIPConfig, create_clip_embedder


class KalmanFilter:
    """Simple Kalman filter for 2D position and velocity tracking."""
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int], fps: float = 30.0):
        """Initialize Kalman filter for bbox center tracking."""
        x, y, w, h = initial_bbox
        cx, cy = x + w / 2, y + h / 2
        
        self.state = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000
        self.P[2:, 2:] = 500
        
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        
        q_pos, q_vel = 5.0, 10.0
        self.Q = np.array([
            [q_pos, 0, 0, 0],
            [0, q_pos, 0, 0],
            [0, 0, q_vel, 0],
            [0, 0, 0, q_vel],
        ], dtype=np.float32)
        
        r = 10.0
        self.R = np.array([[r, 0], [0, r]], dtype=np.float32)
        
        self.bbox_size = (w, h)
        self.frames_since_update = 0
        self.uncertainty_threshold = 5000
        self.velocity_decay = 0.85  # Velocity decays by 15% per frame without measurement
        self.max_frames_without_update = 10  # After this, stop propagating velocity entirely
        
    def predict_without_update(self) -> Tuple[int, int, int, int]:
        """Predict next state WITHOUT modifying internal state. Use for querying only."""
        # Temporary state prediction
        predicted_state = self.F @ self.state
        cx, cy = predicted_state[0], predicted_state[1]
        w, h = self.bbox_size
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
        
    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next state and return predicted bbox.
        
        IMPORTANT: This modifies internal state. When no measurement is available,
        velocity is decayed to prevent bbox from jumping/drifting indefinitely.
        """
        # Apply velocity decay when no measurement - prevents bbox jumping
        # The longer without measurement, the less we trust velocity
        if self.frames_since_update > 0:
            decay = self.velocity_decay ** self.frames_since_update
            # Don't propagate velocity if we've gone too long without measurement
            if self.frames_since_update >= self.max_frames_without_update:
                decay = 0.0
            self.state[2] *= decay  # Decay vx
            self.state[3] *= decay  # Decay vy
        
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.frames_since_update += 1
        
        cx, cy = self.state[0], self.state[1]
        w, h = self.bbox_size
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
    
    def update(self, measurement_bbox: Tuple[int, int, int, int]):
        """Update filter with new measurement."""
        x, y, w, h = measurement_bbox
        cx, cy = x + w / 2, y + h / 2
        z = np.array([cx, cy], dtype=np.float32)
        
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        alpha = 0.9
        self.bbox_size = (
            alpha * self.bbox_size[0] + (1 - alpha) * w,
            alpha * self.bbox_size[1] + (1 - alpha) * h
        )
        self.frames_since_update = 0
    
    def get_velocity(self) -> Tuple[float, float]:
        return (self.state[2], self.state[3])
    
    def get_position_uncertainty(self) -> float:
        return self.P[0, 0] + self.P[1, 1]
    
    def is_reliable(self) -> bool:
        return self.get_position_uncertainty() < self.uncertainty_threshold
    
    def clamp_velocity(self, max_velocity: float) -> None:
        """Clamp velocity to maximum magnitude."""
        vx, vy = self.state[2], self.state[3]
        velocity_mag = np.sqrt(vx**2 + vy**2)
        
        if velocity_mag > max_velocity:
            scale = max_velocity / velocity_mag
            self.state[2] *= scale
            self.state[3] *= scale
    
    def compute_mahalanobis_distance(self, measurement_bbox: Tuple[int, int, int, int]) -> float:
        """Compute Mahalanobis distance for innovation gating."""
        x, y, w, h = measurement_bbox
        cx, cy = x + w / 2, y + h / 2
        z = np.array([cx, cy], dtype=np.float32)
        
        innovation = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        
        try:
            S_inv = np.linalg.inv(S)
            mahal_sq = innovation @ S_inv @ innovation
            return np.sqrt(max(0, mahal_sq))
        except np.linalg.LinAlgError:
            return float('inf')
    
    def gated_update(self, measurement_bbox: Tuple[int, int, int, int], 
                     gate_threshold: float = 3.0, max_velocity: float = 25.0) -> bool:
        """Update with innovation gating to reject outlier measurements."""
        mahal_dist = self.compute_mahalanobis_distance(measurement_bbox)
        
        if mahal_dist > gate_threshold:
            logging.warning(
                f"Kalman update REJECTED - Mahalanobis distance {mahal_dist:.2f} > {gate_threshold}"
            )
            return False
        
        self.update(measurement_bbox)
        self.clamp_velocity(max_velocity)
        return True


class BboxStabilizer:
    """
    Stabilizes bounding boxes across frames to reduce jitter in pasted logos.
    
    Uses multiple techniques:
    - EMA (Exponential Moving Average) smoothing for position and size
    - Aspect ratio locking to maintain consistent proportions
    - Size quantization to avoid sub-pixel wobble
    - Motion threshold to filter micro-movements
    """
    
    def __init__(self, 
                 ema_alpha: float = 0.3,
                 lock_aspect_ratio: bool = True,
                 size_quantization: int = 4,
                 motion_threshold: float = 1.5,
                 enable_subpixel: bool = True):
        """
        Initialize the BboxStabilizer.
        
        Args:
            ema_alpha: Smoothing factor (0-1). Lower = smoother but more lag.
            lock_aspect_ratio: Maintain reference aspect ratio.
            size_quantization: Snap size to multiples of this value (0 to disable).
            motion_threshold: Minimum movement to update (filters micro-jitter).
            enable_subpixel: Track sub-pixel positions for smoother motion.
        """
        self.ema_alpha = ema_alpha
        self.lock_aspect_ratio = lock_aspect_ratio
        self.size_quantization = size_quantization
        self.motion_threshold = motion_threshold
        self.enable_subpixel = enable_subpixel
        
        # State tracking
        self._smoothed_center: Optional[Tuple[float, float]] = None
        self._smoothed_size: Optional[Tuple[float, float]] = None
        self._reference_aspect_ratio: Optional[float] = None
        self._frame_count: int = 0
        
    def reset(self):
        """Reset stabilizer state."""
        self._smoothed_center = None
        self._smoothed_size = None
        self._reference_aspect_ratio = None
        self._frame_count = 0
    
    def _quantize(self, value: float) -> int:
        """Quantize a value to reduce jitter."""
        if self.size_quantization <= 0:
            return int(round(value))
        return int(round(value / self.size_quantization) * self.size_quantization)
    
    def stabilize(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Stabilize a bounding box.
        
        Args:
            bbox: Input bounding box (x, y, w, h)
            
        Returns:
            Stabilized bounding box (x, y, w, h)
        """
        x, y, w, h = bbox
        
        # Calculate center
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        # First frame - initialize
        if self._smoothed_center is None:
            self._smoothed_center = (cx, cy)
            self._smoothed_size = (float(w), float(h))
            if self.lock_aspect_ratio and h > 0:
                self._reference_aspect_ratio = w / h
            self._frame_count = 1
            
            # Return quantized version
            out_w = self._quantize(w)
            out_h = self._quantize(h)
            out_x = int(round(cx - out_w / 2.0))
            out_y = int(round(cy - out_h / 2.0))
            return (out_x, out_y, out_w, out_h)
        
        self._frame_count += 1
        
        # Check motion threshold
        prev_cx, prev_cy = self._smoothed_center
        motion = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        
        if motion < self.motion_threshold:
            # Motion too small, keep previous position
            cx, cy = prev_cx, prev_cy
        
        # EMA smooth center
        new_cx = self.ema_alpha * cx + (1 - self.ema_alpha) * prev_cx
        new_cy = self.ema_alpha * cy + (1 - self.ema_alpha) * prev_cy
        self._smoothed_center = (new_cx, new_cy)
        
        # EMA smooth size
        prev_w, prev_h = self._smoothed_size
        new_w = self.ema_alpha * w + (1 - self.ema_alpha) * prev_w
        new_h = self.ema_alpha * h + (1 - self.ema_alpha) * prev_h
        
        # Lock aspect ratio if enabled
        if self.lock_aspect_ratio and self._reference_aspect_ratio is not None:
            # Use the average of what w and h suggest
            area = new_w * new_h
            new_h = np.sqrt(area / self._reference_aspect_ratio)
            new_w = new_h * self._reference_aspect_ratio
        
        self._smoothed_size = (new_w, new_h)
        
        # Quantize output
        out_w = self._quantize(new_w)
        out_h = self._quantize(new_h)
        
        # Ensure minimum size
        out_w = max(out_w, self.size_quantization if self.size_quantization > 0 else 4)
        out_h = max(out_h, self.size_quantization if self.size_quantization > 0 else 4)
        
        # Calculate output position
        if self.enable_subpixel:
            # Store sub-pixel center for next frame but output integer coords
            out_x = int(round(new_cx - out_w / 2.0))
            out_y = int(round(new_cy - out_h / 2.0))
        else:
            out_x = int(round(new_cx - out_w / 2.0))
            out_y = int(round(new_cy - out_h / 2.0))
        
        return (out_x, out_y, out_w, out_h)
    
    def get_smoothed_center_float(self) -> Optional[Tuple[float, float]]:
        """Get the current smoothed center as floats (for sub-pixel operations)."""
        return self._smoothed_center
    
    def update_reference_aspect_ratio(self, aspect_ratio: float):
        """Update the reference aspect ratio (useful when logo changes)."""
        self._reference_aspect_ratio = aspect_ratio


class MediaPipeHandDetector:
    """Hand detector using MediaPipe."""
    
    def __init__(self, 
                 max_num_hands: int = 4,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required for hand detection. Install with: pip install mediapipe"
            )
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self._last_hands: List[Dict] = []
        self._last_frame_shape: Optional[Tuple[int, int]] = None
    
    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """Detect hands in frame. Returns list of dicts with 'landmarks', 'bbox', 'handedness'."""
        if frame is None or frame.size == 0:
            return []
        
        h, w = frame.shape[:2]
        self._last_frame_shape = (h, w)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract landmark coordinates
                landmarks = []
                x_coords = []
                y_coords = []
                
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    landmarks.append((x, y))
                    x_coords.append(x)
                    y_coords.append(y)
                
                # Calculate bounding box with padding
                padding = 20
                x_min = max(0, min(x_coords) - padding)
                y_min = max(0, min(y_coords) - padding)
                x_max = min(w, max(x_coords) + padding)
                y_max = min(h, max(y_coords) + padding)
                
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                # Get handedness
                handedness = "Unknown"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                
                hands_data.append({
                    'landmarks': landmarks,
                    'bbox': bbox,
                    'handedness': handedness,
                    'raw_landmarks': hand_landmarks
                })
        
        self._last_hands = hands_data
        return hands_data
    
    def get_hand_mask(self, frame: np.ndarray, hand_data: Dict, 
                      sam_masker: Optional['MobileSAMMasker'] = None) -> np.ndarray:
        """Generate a binary mask for a detected hand."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if not hand_data or 'landmarks' not in hand_data:
            return mask
        
        landmarks = hand_data['landmarks']
        bbox = hand_data['bbox']
        
        # Try SAM first for precise segmentation
        if sam_masker is not None:
            try:
                x, y, bw, bh = bbox
                sam_masker.set_image(frame)
                sam_mask = sam_masker.mask_from_box((x, y, x + bw, y + bh))
                if sam_mask is not None and np.count_nonzero(sam_mask) > 100:
                    return sam_mask
            except Exception:
                pass
        
        # Fallback: Create mask from landmark convex hull
        points = np.array(landmarks, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Try GrabCut refinement
        try:
            x, y, bw, bh = bbox
            expand = 30
            x1 = max(0, x - expand)
            y1 = max(0, y - expand)
            x2 = min(w, x + bw + expand)
            y2 = min(h, y + bh + expand)
            
            if (x2 - x1) > 50 and (y2 - y1) > 50:
                roi = frame[y1:y2, x1:x2]
                mask_roi = mask[y1:y2, x1:x2]
                
                # Create GrabCut mask
                gc_mask = np.where(mask_roi > 127, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
                
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                cv2.grabCut(roi, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
                
                refined_mask = np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
                ).astype(np.uint8)
                
                if np.count_nonzero(refined_mask) > 100:
                    mask[y1:y2, x1:x2] = refined_mask
        except Exception:
            pass
        
        return mask
    
    def get_all_hands_mask(self, frame: np.ndarray, 
                            sam_masker: Optional['MobileSAMMasker'] = None) -> np.ndarray:
        """Detect all hands and return combined binary mask."""
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        hands_data = self.detect_hands(frame)
        
        for hand_data in hands_data:
            hand_mask = self.get_hand_mask(frame, hand_data, sam_masker)
            combined_mask = cv2.bitwise_or(combined_mask, hand_mask)
        
        return combined_mask
    
    def check_hand_logo_intersection(self, hand_bbox: Tuple[int, int, int, int],
                                      logo_bbox: Tuple[int, int, int, int]) -> float:
        """Check intersection between hand and logo bboxes. Returns IoU ratio."""
        hx, hy, hw, hh = hand_bbox
        lx, ly, lw, lh = logo_bbox
        
        # Calculate intersection
        ix1 = max(hx, lx)
        iy1 = max(hy, ly)
        ix2 = min(hx + hw, lx + lw)
        iy2 = min(hy + hh, ly + lh)
        
        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        intersection = inter_w * inter_h
        
        if intersection == 0:
            return 0.0
        
        # Calculate union
        hand_area = hw * hh
        logo_area = lw * lh
        union = hand_area + logo_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def close(self):
        """Release MediaPipe resources."""
        if self.hands:
            self.hands.close()


@dataclass
class PoseData:
    """Body pose data for a single frame."""
    boxes: List[Tuple[float, float, float, float]]  # Person bounding boxes
    keypoints: List[Tuple[float, float]]  # 2D keypoints
    keypoints3D: List[Tuple[float, float, float]]  # 3D keypoints
    keypoints_confidence: List[float]  # Confidence scores
    joint_angles: Optional[List[Dict]] = None  # Joint rotation quaternions
    
    @staticmethod
    def from_json(json_path: Path) -> Optional['PoseData']:
        """Load pose data from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            boxes = [(b['x'], b['y'], b['width'], b['height']) for b in data.get('boxes', [])]
            keypoints = [(kp['x'], kp['y']) for kp in data.get('keypoints', [])]
            keypoints3D = [(kp['x'], kp['y'], kp['z']) for kp in data.get('keypoints3D', [])]
            confidence = data.get('keypoints_confidence', [])
            
            return PoseData(
                boxes=boxes,
                keypoints=keypoints,
                keypoints3D=keypoints3D,
                keypoints_confidence=confidence,
                joint_angles=data.get('jointAngles')
            )
        except Exception as e:
            logging.debug(f"Failed to load pose from {json_path}: {e}")
            return None


@dataclass
class TrackedLogo:
    """Represents a logo being tracked across frames."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    detection_sources: set = field(default_factory=set)
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=20))
    frames_since_detection: int = 0
    color: Tuple[int, int, int] = (255, 0, 0)
    person_id: int = -1
    is_occluded: bool = False
    nearest_keypoint_idx: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    pending_embedding: Optional[np.ndarray] = None
    state: str = "tentative"
    consecutive_hits: int = 0
    total_age: int = 0
    
    # Grid tracking
    mask_grid: Optional[np.ndarray] = None
    grid_cell_size: int = 3
    grid_tracking_points: Optional[Dict] = None
    reference_patch: Optional[np.ndarray] = None
    reference_mask: Optional[np.ndarray] = None
    hand_mask: Optional[np.ndarray] = None
    occlusion_grid: Optional[np.ndarray] = None
    current_mask: Optional[np.ndarray] = None
    
    # Multi-frame tracking
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    displacement_history: deque = field(default_factory=lambda: deque(maxlen=5))
    flow_prediction: Optional[Tuple[int, int, int, int]] = None
    frames_tracked_without_detection: int = 0
    flow_quality: float = 0.0
    velocity: Optional[Tuple[float, float]] = None
    flow_history: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Kalman filter
    kalman_filter: Optional['KalmanFilter'] = None
    use_kalman: bool = True
    
    # Mask management
    high_quality_mask: Optional[np.ndarray] = None
    mask_quality_score: float = 0.0
    mask_source: str = "none"
    logo_color_histogram: Optional[np.ndarray] = None
    logo_features: Optional[Dict] = None
    frames_since_mask_update: int = 0
    occluder_mask: Optional[np.ndarray] = None
    
    # Occlusion recovery cache
    last_unoccluded_patch: Optional[np.ndarray] = None
    last_unoccluded_flipped_patch: Optional[np.ndarray] = None
    last_unoccluded_mask: Optional[np.ndarray] = None
    last_unoccluded_bbox: Optional[Tuple[int, int, int, int]] = None
    last_unoccluded_inpainted_bg: Optional[np.ndarray] = None
    frames_since_unoccluded: int = 0
    is_hand_occluded: bool = False
    current_hand_mask: Optional[np.ndarray] = None
    cache_quality_score: float = 0.0
    cache_frame_idx: int = -1
    cache_is_locked: bool = False
    cache_original_size: Optional[Tuple[int, int]] = None
    cache_mask_source: str = "none"
    was_pasted_from_cache: bool = False
    
    # Bbox stabilizer for reducing jitter when pasting
    bbox_stabilizer: Optional['BboxStabilizer'] = None

    def update_bbox(
        self,
        new_bbox: Tuple[int, int, int, int],
        confidence: Optional[float] = None,
        detection_sources: Optional[set] = None,
        mask: Optional[np.ndarray] = None,
    ):
        self.bbox = new_bbox
        if confidence is not None:
            self.confidence = confidence
        if detection_sources is not None:
            self.detection_sources = set(detection_sources)
        if mask is not None:
            self.current_mask = mask.copy()
        x, y, w, h = self.bbox
        self.centroid_history.append((x + w // 2, y + h // 2))
        self.frames_since_detection = 0
        self.consecutive_hits += 1
        self.total_age += 1
        
        if self.use_kalman:
            if self.kalman_filter is None:
                self.kalman_filter = KalmanFilter(new_bbox)
            else:
                accepted = self.kalman_filter.gated_update(new_bbox, gate_threshold=3.0, max_velocity=25.0)
                if not accepted:
                    logging.warning(f"Track {self.track_id}: Kalman rejected bbox update")
        
        # Feed the stabilizer to keep it tracking (even when we don't use stabilized output)
        # This ensures the stabilizer has a smooth history when we do need it
        if self.bbox_stabilizer is not None:
            self.bbox_stabilizer.stabilize(new_bbox)
        
        self.frames_since_mask_update = 0

    def mark_missed(self) -> None:
        self.frames_since_detection += 1
        self.consecutive_hits = 0
    
    def update_mask_position(self, old_bbox: Tuple[int, int, int, int], new_bbox: Tuple[int, int, int, int]) -> None:
        """Update mask position when bbox is moved by tracking."""
        if self.current_mask is None:
            return
        
        old_x, old_y, old_w, old_h = old_bbox
        new_x, new_y, new_w, new_h = new_bbox
        
        dx = new_x - old_x
        dy = new_y - old_y
        scale_x = new_w / max(1, old_w)
        scale_y = new_h / max(1, old_h)
        
        import cv2
        M = np.float32([[scale_x, 0, dx], [0, scale_y, dy]])
        
        h, w = self.current_mask.shape[:2]
        self.current_mask = cv2.warpAffine(
            self.current_mask, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    def update_embedding(self, new_embedding: np.ndarray, momentum: float = 0.85) -> None:
        if new_embedding is None or new_embedding.size == 0:
            return
        if self.embedding is None:
            self.embedding = new_embedding.astype(np.float32)
            return
        weighted = (
            momentum * self.embedding + (1.0 - momentum) * new_embedding
        ).astype(np.float32)
        norm = np.linalg.norm(weighted) + 1e-6
        self.embedding = weighted / norm

    @property
    def centroid(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


class VideoLogoTracker:
    """Tracks logos across video frames using detection + optical flow."""
    
    def __init__(
        self,
        detection_settings: DetectionSettings,
        iou_threshold: float = 0.3,
        max_frames_to_track: int = 60,
        pose_data_dir: Optional[Path] = None,
        use_pose_prediction: bool = True,
        use_clip_embeddings: bool = True,
        clip_model: str = "ViT-B/32",
        clip_device: Optional[str] = None,
        use_optical_flow: bool = True,
        use_fine_grid_tracking: bool = False,
        grid_cell_size: int = 10,
        use_hand_detection: bool = True,
        hand_detection_confidence: float = 0.5,
    ):
        self.detection_settings = detection_settings
        self.iou_threshold = iou_threshold
        self.max_frames_to_track = max_frames_to_track
        self.use_pose_prediction = use_pose_prediction
        self.use_optical_flow = use_optical_flow
        self.use_fine_grid_tracking = use_fine_grid_tracking
        self.use_hand_detection = use_hand_detection

        self.tracked_logos: List[TrackedLogo] = []
        self.next_track_id = 0
        self.frame_count = 0
        self.video_fps = 30.0

        # Pose data
        self.pose_data_dir = pose_data_dir
        self.pose_cache: Dict[int, Optional[PoseData]] = {}
        
        # Hand detection
        self.hand_detector: Optional[MediaPipeHandDetector] = None
        if use_hand_detection and MEDIAPIPE_AVAILABLE:
            try:
                self.hand_detector = MediaPipeHandDetector(
                    max_num_hands=4,
                    min_detection_confidence=hand_detection_confidence,
                    min_tracking_confidence=hand_detection_confidence
                )
            except Exception:
                self.hand_detector = None
        
        self.hand_logo_iou_threshold = 0.05
        self.max_frames_hand_occluded = 60
        
        # Grid tracking
        self.prev_gray: Optional[np.ndarray] = None
        self.gray_frame_history: deque = deque(maxlen=5)
        self.grid_cell_size = grid_cell_size
        self.max_frames_without_detection = 300
        self.min_points_per_cell = 1
        self.min_flow_quality_for_tracking = 0.4
        
        # Multi-frame flow validation
        self.flow_validation_window = 4
        self.flow_consistency_threshold = 2.5
        self.max_flow_direction_change = 60.0
        
        # Motion constraints
        self.max_displacement_per_frame = 20.0
        self.max_displacement_during_occlusion = 8.0
        self.normal_smoothing_alpha = 0.7
        self.occlusion_smoothing_alpha = 0.3
        self.damping_transition_frames = 5
        self.kalman_innovation_gate = 3.0
        self.kalman_max_velocity = 25.0
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=2,
            blockSize=3
        )
        self.occlusion_threshold = 0.25
        self.spatial_cluster_threshold = 2.0

        # Color palette for different tracks
        self.colors = [
            (255, 99, 71),
            (135, 206, 235),
            (60, 179, 113),
            (238, 130, 238),
            (255, 215, 0),
            (70, 130, 180),
            (255, 140, 0),
            (147, 112, 219),
            (0, 255, 127),
            (255, 20, 147),
        ]

        clip_cfg = CLIPConfig(model_name=clip_model, device=clip_device)
        self.embedder = create_clip_embedder(clip_cfg) if use_clip_embeddings else None
        self.appearance_threshold = 0.45
        self.max_association_cost = 0.75
        self.reid_threshold = 0.35
        self.invalid_cost = 1e6
        self.confirmation_hits = 3
        self.tentative_patience = 5

        self.track_archive: List[Dict[str, Any]] = []
        self.archive_maxlen = 64
        
        # SAM mask refinement
        self.enable_sam_per_frame = False
        self.enable_sam_mask_refinement = True
        self.sam_bbox_padding = 0.2
        self.sam_mask_min_quality = 0.3
        self.sam_refinement_cooldown = 2
        self.sam_refinement_on_detection = True
        self.sam_trigger_flow_quality = 0.6
        self.sam_trigger_occlusion_ratio = 0.15
        self.sam_trigger_mask_quality = 0.5
        self.sam_trigger_color_divergence = 0.4
        self.sam_cooldown_frames = 3
    
    def _extract_mask_grid(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                           mask: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """Extract fine-grained grid covering logo mask with tracking points."""
        if not self.use_optical_flow:
            return None
            
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]
        
        # Clamp bbox to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # If no mask provided, use entire bbox
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Calculate grid dimensions (how many cells fit in the mask)
        mask_h, mask_w = mask.shape[:2]
        grid_rows = (mask_h + self.grid_cell_size - 1) // self.grid_cell_size
        grid_cols = (mask_w + self.grid_cell_size - 1) // self.grid_cell_size
        
        if grid_rows < 2 or grid_cols < 2:
            logging.debug(f"Mask too small for grid ({mask_h}×{mask_w} px)")
            return None
        
        # Limit grid size for performance (e.g., max 100x100 = 10000 cells)
        max_grid_cells = 10000
        if grid_rows * grid_cols > max_grid_cells:
            logging.warning(
                f"Grid too large ({grid_rows}×{grid_cols}), would create {grid_rows*grid_cols} cells. "
                f"Skipping fine-grained tracking for this logo."
            )
            return None
        
        # Create boolean grid showing which cells overlap with mask
        mask_grid = np.zeros((grid_rows, grid_cols), dtype=bool)
        
        # Mark cells that overlap with mask
        for row in range(grid_rows):
            for col in range(grid_cols):
                cy1 = row * self.grid_cell_size
                cx1 = col * self.grid_cell_size
                cy2 = min((row + 1) * self.grid_cell_size, mask_h)
                cx2 = min((col + 1) * self.grid_cell_size, mask_w)
                
                cell_mask_region = mask[cy1:cy2, cx1:cx2]
                mask_coverage = np.count_nonzero(cell_mask_region) / (cell_mask_region.size + 1e-6)
                
                if mask_coverage >= 0.3:
                    mask_grid[row, col] = True
        
        # Extract all tracking points at once (much faster than per-cell)
        roi_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Ensure mask matches the bbox dimensions exactly
        bbox_h, bbox_w = y2 - y1, x2 - x1
        if mask.shape != (bbox_h, bbox_w):
            # Resize mask to match bbox dimensions
            mask = cv2.resize(mask, (bbox_w, bbox_h), interpolation=cv2.INTER_NEAREST)
            mask_h, mask_w = mask.shape
            # Recalculate grid dimensions with correct mask size
            grid_rows = (mask_h + self.grid_cell_size - 1) // self.grid_cell_size
            grid_cols = (mask_w + self.grid_cell_size - 1) // self.grid_cell_size
            
            # Recreate mask_grid with correct dimensions
            mask_grid = np.zeros((grid_rows, grid_cols), dtype=bool)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    cy1 = row * self.grid_cell_size
                    cx1 = col * self.grid_cell_size
                    cy2 = min((row + 1) * self.grid_cell_size, mask_h)
                    cx2 = min((col + 1) * self.grid_cell_size, mask_w)
                    
                    cell_mask_region = mask[cy1:cy2, cx1:cx2]
                    mask_coverage = np.count_nonzero(cell_mask_region) / (cell_mask_region.size + 1e-6)
                    
                    if mask_coverage >= 0.3:
                        mask_grid[row, col] = True
        
        roi_mask[y1:y2, x1:x2] = mask
        
        all_points = cv2.goodFeaturesToTrack(
            gray,
            mask=roi_mask,
            maxCorners=min(300, grid_rows * grid_cols * 3),  # Cap at 300 points
            qualityLevel=self.feature_params['qualityLevel'],
            minDistance=2,
            blockSize=3
        )
        
        # Assign points to grid cells
        cell_points = {}
        if all_points is not None and len(all_points) > 0:
            for pt in all_points:
                px, py = pt.ravel()
                # Convert to mask coordinates
                mx = int(px - x1)
                my = int(py - y1)
                
                if 0 <= mx < mask_w and 0 <= my < mask_h:
                    # Find which cell this point belongs to
                    col = mx // self.grid_cell_size
                    row = my // self.grid_cell_size
                    
                    if row < grid_rows and col < grid_cols and mask_grid[row, col]:
                        if (row, col) not in cell_points:
                            cell_points[(row, col)] = []
                        cell_points[(row, col)].append(pt)
        
        # Convert lists to numpy arrays and add fallback points for empty cells
        for row in range(grid_rows):
            for col in range(grid_cols):
                if not mask_grid[row, col]:
                    continue
                
                if (row, col) in cell_points and len(cell_points[(row, col)]) > 0:
                    # Convert list to numpy array
                    cell_points[(row, col)] = np.array(cell_points[(row, col)], dtype=np.float32)
                else:
                    # Fallback: place one point at cell center
                    cy1 = row * self.grid_cell_size
                    cx1 = col * self.grid_cell_size
                    cy2 = min((row + 1) * self.grid_cell_size, mask_h)
                    cx2 = min((col + 1) * self.grid_cell_size, mask_w)
                    center_y = y1 + (cy1 + cy2) / 2
                    center_x = x1 + (cx1 + cx2) / 2
                    cell_points[(row, col)] = np.array([[[center_x, center_y]]], dtype=np.float32)
        
        active_cells = np.count_nonzero(mask_grid)
        if active_cells < 4:
            logging.debug(f"Too few active cells ({active_cells})")
            return None
        
        logging.debug(
            f"Created {grid_rows}×{grid_cols} grid ({active_cells} active cells) "
            f"with {self.grid_cell_size}×{self.grid_cell_size}px cells"
        )
        
        return {
            'grid_shape': (grid_rows, grid_cols),
            'cell_points': cell_points,
            'mask_grid': mask_grid,
            'bbox': bbox,
        }
    
    def _validate_flow_consistency(self, logo: TrackedLogo, new_displacement: Tuple[float, float]) -> bool:
        """Validate flow consistency across temporal window. Returns True if consistent."""
        if len(logo.flow_history) < self.flow_validation_window - 1:
            return True
        
        # Get recent displacements including new one
        recent_flows = list(logo.flow_history) + [new_displacement]
        
        # Check 1: Magnitude consistency
        magnitudes = [np.linalg.norm(flow) for flow in recent_flows]
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        
        # If displacement magnitude suddenly changes significantly, reject
        # (e.g., hand starts moving while logo should be stationary)
        if mean_mag > 0.1:  # Only check if there's meaningful motion
            coefficient_of_variation = std_mag / mean_mag
            if coefficient_of_variation > self.flow_consistency_threshold:
                logging.debug(
                    f"Track {logo.track_id}: Flow REJECTED - inconsistent magnitude "
                    f"(CV={coefficient_of_variation:.2f} > {self.flow_consistency_threshold}, "
                    f"magnitudes={[f'{m:.1f}' for m in magnitudes]})"
                )
                return False
        
        # Check 2: Direction consistency
        # Calculate angles between consecutive flow vectors
        for i in range(len(recent_flows) - 1):
            flow1 = np.array(recent_flows[i])
            flow2 = np.array(recent_flows[i + 1])
            
            mag1 = np.linalg.norm(flow1)
            mag2 = np.linalg.norm(flow2)
            
            # Only check direction if both flows are significant
            if mag1 > 1.0 and mag2 > 1.0:
                # Calculate angle between vectors
                cos_angle = np.dot(flow1, flow2) / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                angle_deg = np.degrees(np.arccos(cos_angle))
                
                if angle_deg > self.max_flow_direction_change:
                    logging.debug(
                        f"Track {logo.track_id}: Flow REJECTED - direction change too large "
                        f"(angle={angle_deg:.1f}° > {self.max_flow_direction_change}°, "
                        f"flow1={flow1}, flow2={flow2})"
                    )
                    return False
        
        # All checks passed
        return True
    
    def _track_mask_grid(self, curr_gray: np.ndarray, logo: TrackedLogo) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray, float]]:
        """Track logo using fine-grained mask grid. Returns (new_bbox, occlusion_grid, flow_quality) or None."""
        if not self.use_optical_flow or self.prev_gray is None:
            return None
        
        if logo.mask_grid is None or logo.grid_tracking_points is None:
            return None
        
        grid_shape = logo.grid_tracking_points['grid_shape']
        cell_points = logo.grid_tracking_points['cell_points']
        mask_grid = logo.grid_tracking_points['mask_grid']
        grid_rows, grid_cols = grid_shape
        
        cell_displacements = []
        cell_weights = []
        occlusion_grid = np.zeros(grid_shape, dtype=bool)
        
        active_cells = 0
        lost_cells = 0
        
        for (row, col), points in cell_points.items():
            if not mask_grid[row, col]:
                continue
            
            if points is None or len(points) == 0:
                occlusion_grid[row, col] = True
                lost_cells += 1
                continue
            
            # Track points in this cell
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, points, None, **self.lk_params
            )
            
            if next_points is None or status is None:
                occlusion_grid[row, col] = True
                lost_cells += 1
                continue
            
            # Filter good points
            good_new = next_points[status == 1]
            good_old = points[status == 1]
            
            if len(good_new) == 0:
                occlusion_grid[row, col] = True
                lost_cells += 1
            else:
                # Calculate displacement for this cell
                displacement = np.mean(good_new - good_old, axis=0)
                cell_displacements.append(displacement)
                cell_weights.append(len(good_new))  # Weight by number of good points
                
                # Update cell points
                cell_points[(row, col)] = good_new.reshape(-1, 1, 2)
                active_cells += 1
        
        # Check if we have enough active cells
        total_mask_cells = np.count_nonzero(mask_grid)
        active_ratio = active_cells / total_mask_cells if total_mask_cells > 0 else 0
        
        # Calculate flow quality metric (0-1)
        # Quality based on: active cell ratio and displacement consistency
        # Lower quality if we had to reject many outliers
        flow_quality = active_ratio
        
        if active_ratio < 0.3:  # Less than 30% cells active - tracking likely failed
            logging.debug(
                f"Track {logo.track_id}: Too few active cells ({active_cells}/{total_mask_cells}, "
                f"{lost_cells} lost)"
            )
            return None
        
        # Calculate overall displacement with outlier rejection
        if len(cell_displacements) == 0:
            return None
        
        cell_displacements = np.array(cell_displacements)
        cell_weights = np.array(cell_weights)
        
        # Robust displacement estimation: reject outliers (e.g., hand motion)
        if len(cell_displacements) >= 3:
            # Calculate median displacement
            median_displacement = np.median(cell_displacements, axis=0)
            
            # Calculate distance from median for each cell
            distances = np.linalg.norm(cell_displacements - median_displacement, axis=1)
            
            # Use MAD (Median Absolute Deviation) for robust threshold
            mad = np.median(distances)
            threshold = max(3.0 * mad, 5.0)  # At least 5 pixels tolerance
            
            # Keep only inliers (consistent motion)
            inliers = distances < threshold
            
            if np.sum(inliers) >= max(3, len(cell_displacements) * 0.5):  # At least 50% inliers
                cell_displacements = cell_displacements[inliers]
                cell_weights = cell_weights[inliers]
                logging.debug(
                    f"Track {logo.track_id}: Filtered {np.sum(~inliers)}/{len(inliers)} outlier cells "
                    f"(threshold={threshold:.1f}px)"
                )
            else:
                # Too many outliers - tracking is unreliable
                logging.warning(
                    f"Track {logo.track_id}: Too many outliers ({np.sum(~inliers)}/{len(inliers)}), "
                    f"tracking may be compromised"
                )
                return None
        
        # Weighted average displacement from inliers
        total_weight = np.sum(cell_weights)
        if total_weight > 0:
            displacement = np.sum(cell_displacements * cell_weights[:, None], axis=0) / total_weight
        else:
            displacement = np.median(cell_displacements, axis=0)
        
        # Additional sanity check: displacement should be reasonable
        displacement_magnitude = np.linalg.norm(displacement)
        if displacement_magnitude > 50:  # More than 50 pixels per frame is suspicious
            logging.warning(
                f"Track {logo.track_id}: Excessive displacement {displacement_magnitude:.1f}px, "
                f"likely tracking failure"
            )
            return None
        
        # MOTION CONSTRAINT: Apply displacement clamping (Strategy A + B)
        # This prevents unnatural bbox jumps during fast hand occlusion
        displacement_clamped = self._clamp_displacement(
            displacement[0], displacement[1], logo, 
            is_hand_nearby=logo.is_hand_occluded
        )
        displacement = np.array(displacement_clamped)
        
        # MULTI-FRAME FLOW VALIDATION: Check consistency across temporal window
        # This prevents bbox from following hand during gradual occlusion
        displacement_tuple = (displacement[0], displacement[1])
        if not self._validate_flow_consistency(logo, displacement_tuple):
            logging.warning(
                f"Track {logo.track_id}: Flow update REJECTED due to inconsistency across temporal window. "
                f"Logo likely occluded by moving object (hand). Freezing bbox position."
            )
            # Don't update bbox, but return current position with low quality to signal uncertainty
            return (logo.bbox, occlusion_grid, 0.3)  # Low quality = consider re-detection soon
        
        # Flow is consistent - accept update and store in history
        logo.flow_history.append(displacement_tuple)
        
        # Update bbox
        x, y, w, h = logo.bbox
        new_x = int(x + displacement[0])
        new_y = int(y + displacement[1])
        
        logging.debug(
            f"Track {logo.track_id}: Grid tracking - {active_cells}/{total_mask_cells} active cells "
            f"({active_ratio*100:.1f}%), displacement=({displacement[0]:.1f}, {displacement[1]:.1f}), "
            f"quality={flow_quality:.2f}"
        )
        
        return ((new_x, new_y, w, h), occlusion_grid, flow_quality)
    
    def _should_trigger_sam_refinement(self, logo: TrackedLogo, frame: np.ndarray) -> bool:
        """Determine if SAM should be triggered for mask refinement."""
        if self.enable_sam_per_frame:
            return True
        
        if logo.frames_since_mask_update < self.sam_cooldown_frames:
            return False
        
        # Trigger on: low flow quality, high occlusion, low mask quality, or color divergence
        if logo.flow_quality > 0 and logo.flow_quality < self.sam_trigger_flow_quality:
            return True
        
        if logo.occlusion_grid is not None:
            total_cells = np.count_nonzero(logo.mask_grid) if logo.mask_grid is not None else 0
            occluded_cells = np.count_nonzero(logo.occlusion_grid)
            if total_cells > 0:
                occlusion_ratio = occluded_cells / total_cells
                if occlusion_ratio > self.sam_trigger_occlusion_ratio:
                    return True
        
        if logo.mask_quality_score > 0 and logo.mask_quality_score < self.sam_trigger_mask_quality:
            return True
        
        if logo.logo_color_histogram is not None:
            current_histogram = self._extract_logo_color_signature(frame, logo.current_mask, logo.bbox)
            if current_histogram is not None and current_histogram.size > 0:
                correlation = cv2.compareHist(
                    logo.logo_color_histogram, current_histogram, cv2.HISTCMP_CORREL
                )
                if correlation < self.sam_trigger_color_divergence:
                    return True
        
        return False
    
    def _detect_occlusion_with_sam(self, frame: np.ndarray, logo: TrackedLogo, 
                                     pipeline: 'LogoFlippingPipeline') -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Use SAM to detect precise logo/occluder boundary. Returns (logo_mask, occluder_mask) or None."""
        if pipeline._sam_masker is None:
            return None
        
        try:
            # Set SAM image
            pipeline._sam_masker.set_image(frame)
            
            # Step 1: Get fresh logo mask with bbox prompt
            x, y, w, h = logo.bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            
            logo_mask_full = pipeline._sam_masker.mask_from_box((x, y, x + w, y + h))
            
            if logo_mask_full is None or np.count_nonzero(logo_mask_full) < 10:
                logging.debug(f"Track {logo.track_id}: SAM failed to segment logo")
                return None
            
            # Crop to bbox region
            logo_mask = logo_mask_full[y1:y2, x1:x2]
            
            # Step 2: Check if we have occluded grid cells to guide occluder detection
            if logo.occlusion_grid is None or not np.any(logo.occlusion_grid):
                # No occlusion detected by grid, return logo mask as-is
                logging.debug(f"Track {logo.track_id}: No occlusion grid, using logo mask as-is")
                return (logo_mask, None)
            
            # Step 3: Generate point prompts from occluded grid cells
            # These points are likely on the occluder (hand/object covering logo)
            occluded_points = []
            
            if logo.grid_tracking_points is not None:
                grid_shape = logo.grid_tracking_points['grid_shape']
                mask_grid = logo.grid_tracking_points['mask_grid']
                grid_rows, grid_cols = grid_shape
                
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        # If this cell is marked as occluded
                        if logo.occlusion_grid[row, col] and mask_grid[row, col]:
                            # Calculate cell center in frame coordinates
                            cell_center_y = y1 + int((row + 0.5) * self.grid_cell_size)
                            cell_center_x = x1 + int((col + 0.5) * self.grid_cell_size)
                            
                            # Make sure it's within frame bounds
                            if 0 <= cell_center_x < frame.shape[1] and 0 <= cell_center_y < frame.shape[0]:
                                occluded_points.append((cell_center_x, cell_center_y))
            
            if not occluded_points:
                logging.debug(f"Track {logo.track_id}: No valid occluded points for SAM prompting")
                return (logo_mask, None)
            
            # Limit to top 10 occluded points (for performance)
            if len(occluded_points) > 10:
                # Sample uniformly
                step = len(occluded_points) // 10
                occluded_points = occluded_points[::step][:10]
            
            logging.debug(
                f"Track {logo.track_id}: Using {len(occluded_points)} occluded points to segment occluder"
            )
            
            # Step 4: Segment occluder using point prompts
            # Convert points to format SAM expects: [[x1, y1], [x2, y2], ...]
            point_coords = np.array(occluded_points, dtype=np.float32)
            point_labels = np.ones(len(occluded_points), dtype=np.int32)  # All foreground points
            
            # Use SAM predictor with point prompts
            results = pipeline._sam_masker._predictor(
                points=[point_coords.tolist()],
                labels=[point_labels.tolist()]
            )
            
            if not results or results[0].masks is None or results[0].masks.data is None:
                logging.debug(f"Track {logo.track_id}: SAM point prompting failed")
                return (logo_mask, None)
            
            # Get the first (best) mask from SAM
            occluder_mask_full = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            
            # Crop to bbox region
            occluder_mask = occluder_mask_full[y1:y2, x1:x2]
            
            occluder_pixels = np.count_nonzero(occluder_mask)
            
            if occluder_pixels < 50:  # Too small to be real occluder
                logging.debug(f"Track {logo.track_id}: Occluder mask too small ({occluder_pixels} pixels)")
                return (logo_mask, None)
            
            # Step 5: Subtract occluder from logo mask → visible logo portion
            visible_logo_mask = cv2.bitwise_and(
                logo_mask,
                cv2.bitwise_not(occluder_mask)
            )
            
            visible_pixels = np.count_nonzero(visible_logo_mask)
            total_logo_pixels = np.count_nonzero(logo_mask)
            occlusion_percent = (1 - visible_pixels / max(1, total_logo_pixels)) * 100
            
            logging.info(
                f"Track {logo.track_id}: SAM occlusion detection successful - "
                f"{occluder_pixels} occluder pixels, {occlusion_percent:.1f}% logo occluded"
            )
            
            return (visible_logo_mask, occluder_mask)
        
        except Exception as e:
            logging.warning(f"Track {logo.track_id}: SAM occlusion detection failed: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            return None
    
    def _calculate_mask_quality(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate mask quality score (0-1) based on coverage, shape, and coherence."""
        if mask is None or mask.size == 0:
            return 0.0
        
        h, w = mask.shape[:2]
        bbox_area = bbox[2] * bbox[3]
        mask_pixels = np.count_nonzero(mask)
        
        if mask_pixels == 0:
            return 0.0
        
        # 1. Coverage ratio (how much of bbox is filled)
        coverage = mask_pixels / max(1, h * w)
        coverage_score = min(1.0, coverage / 0.3)  # Ideal: 30%+ coverage
        
        # 2. Compactness (perimeter^2 / area - lower is more compact/circular)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if area > 0:
                compactness = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
                compactness_score = min(1.0, compactness)  # Circle = 1.0
            else:
                compactness_score = 0.0
        else:
            compactness_score = 0.0
        
        # 3. Bbox fit (does mask fit well within bbox)
        y_coords, x_coords = np.nonzero(mask)
        if len(x_coords) > 0:
            mask_bbox_w = x_coords.max() - x_coords.min() + 1
            mask_bbox_h = y_coords.max() - y_coords.min() + 1
            bbox_fit = min(mask_bbox_w / bbox[2], mask_bbox_h / bbox[3])
            bbox_fit_score = min(1.0, bbox_fit)
        else:
            bbox_fit_score = 0.0
        
        # Weighted combination
        quality = 0.5 * coverage_score + 0.3 * compactness_score + 0.2 * bbox_fit_score
        
        return float(quality)
    
    def _extract_logo_color_signature(self, frame: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract HSV color histogram from logo region."""
        x, y, w, h = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(48, dtype=np.float32)
        
        roi = frame[y1:y2, x1:x2]
        mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1)) if mask.shape[:2] != (y2 - y1, x2 - x1) else mask
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel (masked)
        h_hist = cv2.calcHist([hsv], [0], mask_resized, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], mask_resized, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], mask_resized, [16], [0, 256])
        
        # Normalize
        h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-6)
        s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-6)
        v_hist = v_hist.flatten() / (np.sum(v_hist) + 1e-6)
        
        # Concatenate
        histogram = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        
        return histogram
    
    def _refine_mask_by_color(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                               reference_histogram: np.ndarray, threshold: float = 0.6) -> Optional[np.ndarray]:
        """Refine mask by detecting logo colors in current frame."""
        x, y, w, h = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for current ROI
        h_hist = cv2.calcHist([hsv_roi], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv_roi], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv_roi], [2], None, [16], [0, 256]).flatten()
        
        h_hist = h_hist / (np.sum(h_hist) + 1e-6)
        s_hist = s_hist / (np.sum(s_hist) + 1e-6)
        v_hist = v_hist / (np.sum(v_hist) + 1e-6)
        
        current_histogram = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        
        # Compare histograms
        correlation = cv2.compareHist(reference_histogram, current_histogram, cv2.HISTCMP_CORREL)
        
        if correlation < threshold:
            return None
        
        # Create color-based mask using back-projection
        h_ref = reference_histogram[:16].reshape(-1, 1)
        s_ref = reference_histogram[16:32].reshape(-1, 1)
        v_ref = reference_histogram[32:].reshape(-1, 1)
        
        # Back-project each channel
        h_backproj = cv2.calcBackProject([hsv_roi], [0], h_ref, [0, 180], 1)
        s_backproj = cv2.calcBackProject([hsv_roi], [1], s_ref, [0, 256], 1)
        v_backproj = cv2.calcBackProject([hsv_roi], [2], v_ref, [0, 256], 1)
        
        # Combine (weighted average)
        back_proj = (0.5 * h_backproj + 0.3 * s_backproj + 0.2 * v_backproj).astype(np.uint8)
        
        # Threshold
        _, mask = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep only largest component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask)
            cv2.drawContours(mask_refined, [largest], 0, 255, -1)
            return mask_refined
        
        return None
    
    def _update_mask_with_sam(self, frame: np.ndarray, logo: TrackedLogo, 
                               pipeline: 'LogoFlippingPipeline') -> bool:
        """Re-segment logo with SAM to get high-quality mask."""
        if pipeline._sam_masker is None:
            return False
        
        try:
            # Set SAM image
            pipeline._sam_masker.set_image(frame)
            
            # Get mask for logo bbox
            x, y, w, h = logo.bbox
            mask = pipeline._sam_masker.mask_from_box((x, y, x + w, y + h))
            
            if mask is None or mask.size == 0:
                return False
            
            # Crop to bbox region
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            mask_cropped = mask[y1:y2, x1:x2]
            
            if np.count_nonzero(mask_cropped) < 10:  # Too few pixels
                return False
            
            # Calculate quality
            quality = self._calculate_mask_quality(mask_cropped, logo.bbox)
            
            if quality > logo.mask_quality_score:
                logo.high_quality_mask = mask_cropped.copy()
                logo.current_mask = mask_cropped.copy()
                logo.mask_quality_score = quality
                logo.mask_source = "sam"
                logo.frames_since_mask_update = 0
                logo.logo_color_histogram = self._extract_logo_color_signature(
                    frame, mask_cropped, logo.bbox
                )
                return True
        
        except Exception:
            pass
        
        return False
    
    def _refine_mask_with_sam_on_bbox(self, frame: np.ndarray, logo: TrackedLogo,
                                       pipeline: 'LogoFlippingPipeline') -> Optional[np.ndarray]:
        """Refine logo mask by running SAM on the tracked bbox with padding."""
        if pipeline is None or pipeline._sam_masker is None:
            return None
        
        if not self.enable_sam_mask_refinement:
            return None
        
        # Check cooldown
        if logo.frames_since_mask_update < self.sam_refinement_cooldown:
            return None
        
        try:
            x, y, w, h = logo.bbox
            h_img, w_img = frame.shape[:2]
            
            # Add padding around bbox (20% by default)
            pad_x = int(w * self.sam_bbox_padding)
            pad_y = int(h * self.sam_bbox_padding)
            
            # Padded bbox bounds (clamped to image)
            px1 = max(0, x - pad_x)
            py1 = max(0, y - pad_y)
            px2 = min(w_img, x + w + pad_x)
            py2 = min(h_img, y + h + pad_y)
            
            if px2 <= px1 or py2 <= py1:
                return None
            
            # Original bbox in padded coordinates (for SAM prompt)
            bbox_in_padded = (
                x - px1,  # x offset
                y - py1,  # y offset
                x - px1 + w,  # x2
                y - py1 + h   # y2
            )
            
            # Extract padded region
            padded_region = frame[py1:py2, px1:px2]
            
            # Run SAM on padded region with bbox prompt
            pipeline._sam_masker.set_image(padded_region)
            sam_mask_padded = pipeline._sam_masker.mask_from_box(bbox_in_padded)
            
            if sam_mask_padded is None or np.count_nonzero(sam_mask_padded) < 10:
                logging.debug(f"Track {logo.track_id}: SAM returned empty/small mask on bbox")
                return None
            
            # Extract mask for original bbox region (remove padding)
            mask_x1 = x - px1
            mask_y1 = y - py1
            mask_x2 = mask_x1 + w
            mask_y2 = mask_y1 + h
            
            # Clamp to mask bounds
            mask_x1 = max(0, mask_x1)
            mask_y1 = max(0, mask_y1)
            mask_x2 = min(sam_mask_padded.shape[1], mask_x2)
            mask_y2 = min(sam_mask_padded.shape[0], mask_y2)
            
            sam_mask_cropped = sam_mask_padded[mask_y1:mask_y2, mask_x1:mask_x2]
            
            if sam_mask_cropped.size == 0 or np.count_nonzero(sam_mask_cropped) < 10:
                return None
            
            # Resize to exact bbox dimensions if needed
            target_h, target_w = h, w
            if sam_mask_cropped.shape[:2] != (target_h, target_w):
                sam_mask_cropped = cv2.resize(
                    sam_mask_cropped, (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Validate mask quality
            quality = self._calculate_mask_quality(sam_mask_cropped, logo.bbox)
            
            if quality < self.sam_mask_min_quality:
                logging.debug(
                    f"Track {logo.track_id}: SAM mask quality too low ({quality:.3f} < {self.sam_mask_min_quality})"
                )
                return None
            
            # Update logo mask state
            logo.high_quality_mask = sam_mask_cropped.copy()
            logo.current_mask = sam_mask_cropped.copy()
            logo.mask_quality_score = quality
            logo.mask_source = "sam_bbox"
            logo.frames_since_mask_update = 0
            
            # Update color signature from new mask
            logo.logo_color_histogram = self._extract_logo_color_signature(
                frame, sam_mask_cropped, logo.bbox
            )
            
            mask_pixels = np.count_nonzero(sam_mask_cropped)
            logging.info(
                f"Track {logo.track_id}: SAM bbox refinement - "
                f"quality={quality:.3f}, pixels={mask_pixels}, size={target_w}x{target_h}"
            )
            
            return sam_mask_cropped
        
        except Exception as e:
            logging.warning(f"Track {logo.track_id}: SAM bbox refinement failed: {e}")
            return None
    
    def _clamp_displacement(self, dx: float, dy: float, logo: TrackedLogo, 
                            is_hand_nearby: bool = False) -> Tuple[float, float]:
        """Clamp displacement to prevent unnatural bbox jumps."""
        displacement_mag = np.sqrt(dx**2 + dy**2)
        
        if is_hand_nearby or logo.is_hand_occluded:
            max_disp = self.max_displacement_during_occlusion
            smoothing_alpha = self.occlusion_smoothing_alpha
        else:
            max_disp = self.max_displacement_per_frame
            smoothing_alpha = self.normal_smoothing_alpha
        
        if displacement_mag > max_disp:
            scale = max_disp / displacement_mag
            dx, dy = dx * scale, dy * scale
        
        # Adaptive smoothing with velocity history
        if logo.velocity is not None and len(logo.displacement_history) >= 2:
            prev_dx, prev_dy = logo.velocity
            dx = smoothing_alpha * dx + (1 - smoothing_alpha) * prev_dx
            dy = smoothing_alpha * dy + (1 - smoothing_alpha) * prev_dy
        
        return (dx, dy)

    def _predict_bbox_from_velocity(self, logo: TrackedLogo) -> Optional[Tuple[int, int, int, int]]:
        """Predict next bbox position using Kalman filter or weighted EMA fallback.
        
        When logo is not detected by YOLO, this predicts where it should be.
        The Kalman filter now applies velocity decay to prevent bbox from jumping.
        """
        # Try Kalman filter first
        if logo.use_kalman and logo.kalman_filter is not None:
            # Check if Kalman is still reliable before using its prediction
            if not logo.kalman_filter.is_reliable():
                logging.debug(
                    f"Track {logo.track_id}: Kalman filter unreliable "
                    f"(uncertainty={logo.kalman_filter.get_position_uncertainty():.1f}), "
                    f"falling back to EMA"
                )
            else:
                predicted_bbox = logo.kalman_filter.predict()
                vx, vy = logo.kalman_filter.get_velocity()
                logo.velocity = (vx, vy)
                
                # Log when velocity is being decayed significantly
                if logo.kalman_filter.frames_since_update > 3:
                    logging.debug(
                        f"Track {logo.track_id}: Kalman prediction with decayed velocity "
                        f"(frames_since_update={logo.kalman_filter.frames_since_update}, "
                        f"velocity=({vx:.2f}, {vy:.2f}))"
                    )
                
                return predicted_bbox
        
        # Fallback: Weighted EMA
        if len(logo.displacement_history) < 2:
            return None
        
        # Apply velocity decay for EMA fallback too - the longer we go without detection,
        # the less we should trust historical velocity
        frames_without_detection = logo.frames_since_detection
        ema_decay = 0.85 ** frames_without_detection  # Same decay factor as Kalman
        
        # After 10+ frames without detection, don't use velocity-based prediction at all
        if frames_without_detection >= 10:
            logging.debug(
                f"Track {logo.track_id}: EMA prediction skipped - too many frames without detection "
                f"({frames_without_detection})"
            )
            return None
        
        # Calculate weighted average with exponential decay (more recent = higher weight)
        recent_displacements = list(logo.displacement_history)
        n = len(recent_displacements)
        
        # Exponential weights: alpha^0, alpha^1, alpha^2, ... (most recent gets highest weight)
        alpha = 0.7  # Decay factor (0.7 means 70% weight retention per frame back)
        weights = np.array([alpha ** i for i in range(n)][::-1])  # Reverse so most recent is first
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted average displacement
        displacements = np.array(recent_displacements)
        avg_dx = np.sum(displacements[:, 0] * weights)
        avg_dy = np.sum(displacements[:, 1] * weights)
        
        # Apply decay based on frames without detection
        avg_dx *= ema_decay
        avg_dy *= ema_decay
        
        # Apply velocity to current bbox
        x, y, w, h = logo.bbox
        predicted_x = int(x + avg_dx)
        predicted_y = int(y + avg_dy)
        
        # Update velocity estimate
        logo.velocity = (avg_dx, avg_dy)
        
        logging.debug(
            f"Track {logo.track_id}: Weighted EMA prediction - dx={avg_dx:.1f}, dy={avg_dy:.1f}, "
            f"decay={ema_decay:.3f}"
        )
        
        return (predicted_x, predicted_y, w, h)
    
    def _detect_hand_occlusion(self, frame: np.ndarray, logo: TrackedLogo) -> Optional[np.ndarray]:
        """Detect hand occlusion using fine-grained grid cell analysis."""
        if logo.reference_patch is None or logo.reference_mask is None:
            return None
        
        if logo.mask_grid is None or logo.grid_tracking_points is None:
            return None
        
        x, y, w, h = logo.bbox
        h_img, w_img = frame.shape[:2]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        current_patch = frame[y1:y2, x1:x2]
        
        if current_patch.shape[:2] != logo.reference_patch.shape[:2]:
            ref_resized = cv2.resize(logo.reference_patch, (current_patch.shape[1], current_patch.shape[0]))
            mask_resized = cv2.resize(logo.reference_mask, (current_patch.shape[1], current_patch.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            ref_resized = logo.reference_patch
            mask_resized = logo.reference_mask
        
        if len(mask_resized.shape) == 3:
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
        
        if mask_resized.dtype != np.uint8:
            mask_resized = mask_resized.astype(np.uint8)
        _, mask_resized = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
        
        grid_shape = logo.grid_tracking_points['grid_shape']
        mask_grid = logo.grid_tracking_points['mask_grid']
        grid_rows, grid_cols = grid_shape
        
        occluded_cells = []
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                if not mask_grid[row, col]:
                    continue
                
                # Calculate cell bounds in patch coordinates
                cy1 = row * self.grid_cell_size
                cx1 = col * self.grid_cell_size
                cy2 = min((row + 1) * self.grid_cell_size, current_patch.shape[0])
                cx2 = min((col + 1) * self.grid_cell_size, current_patch.shape[1])
                
                if cy2 <= cy1 or cx2 <= cx1:
                    continue
                
                # Extract cell regions
                current_cell = current_patch[cy1:cy2, cx1:cx2]
                ref_cell = ref_resized[cy1:cy2, cx1:cx2]
                
                # Calculate pixel difference for this cell
                if len(current_cell.shape) == 3:
                    diff = cv2.absdiff(current_cell, ref_cell)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                else:
                    diff_gray = cv2.absdiff(current_cell, ref_cell)
                
                mean_diff = np.mean(diff_gray)
                
                # Cell is occluded if significant pixel difference or marked as lost in occlusion_grid
                if logo.occlusion_grid is not None and logo.occlusion_grid[row, col]:
                    occluded_cells.append((row, col, cy1, cx1, cy2, cx2))
                elif mean_diff > 30:  # Pixel difference threshold
                    occluded_cells.append((row, col, cy1, cx1, cy2, cx2))
        
        if len(occluded_cells) == 0:
            return None
        
        total_mask_cells = np.count_nonzero(mask_grid)
        occluded_ratio = len(occluded_cells) / total_mask_cells if total_mask_cells > 0 else 0
        
        if occluded_ratio < self.occlusion_threshold:
            return None
        
        if len(occluded_cells) >= 2:
            positions = np.array([[r, c] for r, c, _, _, _, _ in occluded_cells])
            from scipy.spatial.distance import pdist
            distances = pdist(positions, metric='euclidean')
            mean_distance = np.mean(distances) if len(distances) > 0 else float('inf')
            
            # Cells should be clustered (within a few cells of each other)
            if mean_distance > self.spatial_cluster_threshold * grid_rows:
                logging.debug(
                    f"Track {logo.track_id}: Occluded cells not clustered (mean_dist={mean_distance:.1f})"
                )
                return None
        
        mask_h, mask_w = current_patch.shape[:2]
        occlusion_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        
        for row, col, cy1, cx1, cy2, cx2 in occluded_cells:
            occlusion_mask[cy1:cy2, cx1:cx2] = 255
        
        if occlusion_mask.shape != mask_resized.shape:
            logging.warning(
                f"Track {logo.track_id}: Shape mismatch - occlusion_mask {occlusion_mask.shape} "
                f"vs mask_resized {mask_resized.shape}"
            )
            occlusion_mask = cv2.resize(occlusion_mask, (mask_resized.shape[1], mask_resized.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
        
        occlusion_mask = cv2.bitwise_and(occlusion_mask, mask_resized)
        
        logging.debug(
            f"Track {logo.track_id}: Fine-grid occlusion - {len(occluded_cells)}/{total_mask_cells} cells "
            f"({occluded_ratio*100:.1f}%), cell_size={self.grid_cell_size}×{self.grid_cell_size}px"
        )
        
        return occlusion_mask
    
    def _detect_hand_occlusion_mediapipe(self, frame: np.ndarray, logo: TrackedLogo,
                                          pipeline: Optional['LogoFlippingPipeline'] = None,
                                          full_frame_hand_mask: Optional[np.ndarray] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect hand occlusion using pre-computed mask or MediaPipe. Returns (is_occluded, hand_mask_in_logo_coords)."""
        if self.hand_detector is None:
            return (False, None)
        
        x, y, w, h = logo.bbox
        h_img, w_img = frame.shape[:2]
        
        # Clamp bbox to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return (False, None)
        
        # Use pre-computed mask if available (much more accurate than bbox-based detection)
        if full_frame_hand_mask is not None:
            # Directly check mask overlap with logo bbox
            hand_mask_in_logo = full_frame_hand_mask[y1:y2, x1:x2]
            overlap_pixels = np.count_nonzero(hand_mask_in_logo)
            logo_area = (x2 - x1) * (y2 - y1)
            overlap_ratio = overlap_pixels / logo_area if logo_area > 0 else 0
            
            # Debug: Always log overlap info for confirmed logos
            logging.debug(
                f"Track {logo.track_id}: Hand-logo check - "
                f"bbox=({x1},{y1})-({x2},{y2}), "
                f"overlap={overlap_pixels}px ({overlap_ratio*100:.2f}%), "
                f"logo_area={logo_area}px"
            )
            
            # Threshold: at least 50 pixels AND 1% of logo area
            if overlap_pixels >= 50 and overlap_ratio >= 0.01:
                logging.info(
                    f"Track {logo.track_id}: Hand occlusion detected (mask-based) - "
                    f"{overlap_pixels} pixels ({overlap_ratio*100:.1f}% of logo area)"
                )
                return (True, hand_mask_in_logo.copy())
            else:
                return (False, None)
        else:
            logging.debug(f"Track {logo.track_id}: No pre-computed hand mask available")
        
        # Fallback: Detect hands fresh if no pre-computed mask
        hands_data = self.hand_detector.detect_hands(frame)
        
        if not hands_data:
            return (False, None)
        
        # Check each hand for intersection with logo (bbox-based for initial filtering)
        intersecting_hands = []
        for hand_data in hands_data:
            hand_bbox = hand_data['bbox']
            iou = self.hand_detector.check_hand_logo_intersection(hand_bbox, logo.bbox)
            
            if iou >= self.hand_logo_iou_threshold:
                intersecting_hands.append(hand_data)
                logging.debug(
                    f"Track {logo.track_id}: Hand ({hand_data['handedness']}) intersects logo "
                    f"with IoU={iou:.3f}"
                )
        
        if not intersecting_hands:
            return (False, None)
        
        # Generate combined hand mask for all intersecting hands
        sam_masker = pipeline._sam_masker if pipeline is not None else None
        combined_hand_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        
        for hand_data in intersecting_hands:
            hand_mask = self.hand_detector.get_hand_mask(frame, hand_data, sam_masker)
            combined_hand_mask = cv2.bitwise_or(combined_hand_mask, hand_mask)
        
        if np.count_nonzero(combined_hand_mask) == 0:
            return (False, None)
        
        # Crop hand mask to logo bbox coordinates
        hand_mask_in_logo = combined_hand_mask[y1:y2, x1:x2]
        
        # Check if there's actual overlap in the logo region
        overlap_pixels = np.count_nonzero(hand_mask_in_logo)
        logo_area = (x2 - x1) * (y2 - y1)
        overlap_ratio = overlap_pixels / logo_area if logo_area > 0 else 0
        
        if overlap_pixels < 50 or overlap_ratio < 0.01:  # Less than 1% overlap
            return (False, None)
        
        logging.info(
            f"Track {logo.track_id}: Hand occlusion detected (fallback) - "
            f"{overlap_pixels} pixels ({overlap_ratio*100:.1f}% of logo area)"
        )
        
        return (True, hand_mask_in_logo)
    
    def _update_unoccluded_reference(self, frame: np.ndarray, logo: TrackedLogo,
                                      flipped_frame: Optional[np.ndarray] = None) -> None:
        """Update the cached unoccluded logo reference if logo is fully visible."""
        x, y, w, h = logo.bbox
        h_img, w_img = frame.shape[:2]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        # Store original patch
        logo.last_unoccluded_patch = frame[y1:y2, x1:x2].copy()
        logo.last_unoccluded_bbox = (x1, y1, x2 - x1, y2 - y1)
        
        # Store flipped patch if provided
        if flipped_frame is not None:
            logo.last_unoccluded_flipped_patch = flipped_frame[y1:y2, x1:x2].copy()
        
        # Store mask
        if logo.current_mask is not None:
            # Resize mask to match patch size if needed
            if logo.current_mask.shape[:2] != (y2 - y1, x2 - x1):
                logo.last_unoccluded_mask = cv2.resize(
                    logo.current_mask, (x2 - x1, y2 - y1), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                logo.last_unoccluded_mask = logo.current_mask.copy()
        elif logo.high_quality_mask is not None:
            if logo.high_quality_mask.shape[:2] != (y2 - y1, x2 - x1):
                logo.last_unoccluded_mask = cv2.resize(
                    logo.high_quality_mask, (x2 - x1, y2 - y1), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                logo.last_unoccluded_mask = logo.high_quality_mask.copy()
        
        logo.frames_since_unoccluded = 0
        logo.is_hand_occluded = False
        logo.current_hand_mask = None
        
        logging.debug(
            f"Track {logo.track_id}: Updated unoccluded reference "
            f"(patch size: {logo.last_unoccluded_patch.shape})"
        )
    
    def _paste_unoccluded_logo_with_hand_mask(self, canvas: np.ndarray, logo: TrackedLogo,
                                               use_flipped: bool = True) -> np.ndarray:
        """Paste the last unoccluded logo onto canvas, excluding hand-occluded regions."""
        if logo.last_unoccluded_patch is None:
            logging.warning(f"Track {logo.track_id}: No cached unoccluded patch available")
            return canvas
        
        if logo.current_hand_mask is None:
            logging.warning(f"Track {logo.track_id}: No hand mask available for occlusion-aware paste")
            return canvas
        
        # Determine which patch to use
        if use_flipped and logo.last_unoccluded_flipped_patch is not None:
            patch_to_paste = logo.last_unoccluded_flipped_patch
        else:
            patch_to_paste = logo.last_unoccluded_patch
        
        # Get current logo position
        x, y, w, h = logo.bbox
        h_img, w_img = canvas.shape[:2]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return canvas
        
        target_h, target_w = y2 - y1, x2 - x1
        
        # Resize cached patch to current bbox size if needed
        if patch_to_paste.shape[:2] != (target_h, target_w):
            patch_resized = cv2.resize(patch_to_paste, (target_w, target_h))
        else:
            patch_resized = patch_to_paste
        
        # Resize logo mask to current size
        logo_mask = None
        if logo.last_unoccluded_mask is not None:
            if logo.last_unoccluded_mask.shape[:2] != (target_h, target_w):
                logo_mask = cv2.resize(logo.last_unoccluded_mask, (target_w, target_h), 
                                       interpolation=cv2.INTER_NEAREST)
            else:
                logo_mask = logo.last_unoccluded_mask
        else:
            # Fallback: use full patch area as mask
            logo_mask = np.ones((target_h, target_w), dtype=np.uint8) * 255
        
        # Resize hand mask to current size
        hand_mask = logo.current_hand_mask
        if hand_mask.shape[:2] != (target_h, target_w):
            hand_mask = cv2.resize(hand_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Create final paste mask: logo mask minus hand mask
        # This gives us only the visible (non-hand-occluded) parts of the logo
        visible_logo_mask = cv2.bitwise_and(logo_mask, cv2.bitwise_not(hand_mask))
        
        visible_pixels = np.count_nonzero(visible_logo_mask)
        if visible_pixels < 10:
            logging.debug(f"Track {logo.track_id}: Logo fully occluded by hand, skipping paste")
            return canvas
        
        # Create 3-channel mask for blending
        if len(visible_logo_mask.shape) == 2:
            mask_3ch = np.stack([visible_logo_mask] * 3, axis=-1)
        else:
            mask_3ch = visible_logo_mask
        
        # Normalize mask to [0, 1] for blending
        mask_float = mask_3ch.astype(np.float32) / 255.0
        
        # Blend: result = patch * mask + canvas * (1 - mask)
        roi = canvas[y1:y2, x1:x2]
        blended = (patch_resized.astype(np.float32) * mask_float + 
                   roi.astype(np.float32) * (1.0 - mask_float))
        canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        logging.info(
            f"Track {logo.track_id}: Pasted unoccluded logo with hand mask - "
            f"{visible_pixels} visible pixels"
        )
        
        return canvas

    def _load_pose(self, frame_idx: int) -> Optional[PoseData]:
        if frame_idx in self.pose_cache:
            return self.pose_cache[frame_idx]
        
        if self.pose_data_dir is None:
            return None
        
        pose_file = self.pose_data_dir / f"bodypose_{frame_idx}.json"
        pose = PoseData.from_json(pose_file) if pose_file.exists() else None
        self.pose_cache[frame_idx] = pose
        return pose
    
    def _filter_by_body_proximity(self, detections: List, pose: Optional[PoseData], margin: int = 150) -> List:
        if pose is None or not pose.boxes:
            return detections
        
        filtered = []
        for det_bbox, conf, sources in detections:
            x, y, w, h = det_bbox
            
            # Check if ANY part of detection bbox overlaps with expanded person box
            for px, py, pw, ph in pose.boxes:
                # Expand person box by margin
                expanded_x1 = px - margin
                expanded_y1 = py - margin
                expanded_x2 = px + pw + margin
                expanded_y2 = py + ph + margin
                
                # Check bbox overlap (not just center)
                det_x1, det_y1 = x, y
                det_x2, det_y2 = x + w, y + h
                
                # Check if bboxes overlap
                if not (det_x2 < expanded_x1 or det_x1 > expanded_x2 or 
                       det_y2 < expanded_y1 or det_y1 > expanded_y2):
                    filtered.append((det_bbox, conf, sources))
                    break
        
        filtered_count = len(filtered)
        if filtered_count < len(detections):
            logging.debug(f"Pose filtering: {len(detections)} -> {filtered_count} detections (removed {len(detections) - filtered_count})")
        
        return filtered
    
    def _find_nearest_keypoint(self, bbox: Tuple[int, int, int, int], pose: Optional[PoseData]) -> Optional[int]:
        if pose is None or not pose.keypoints:
            return None
        
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        
        min_dist = float('inf')
        nearest_idx = None
        
        # Focus on torso/upper body keypoints (indices 5-14 typically)
        # This avoids using head/face or far extremities
        keypoint_range = min(14, len(pose.keypoints))
        
        for i in range(5, keypoint_range):
            kx, ky = pose.keypoints[i]
            dist = (cx - kx) ** 2 + (cy - ky) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _check_occlusion(self, logo: TrackedLogo, pose: Optional[PoseData]) -> bool:
        if pose is None or not pose.keypoints3D or logo.nearest_keypoint_idx is None:
            return False
        
        x, y, w, h = logo.bbox
        logo_center_x = x + w // 2
        logo_center_y = y + h // 2
        
        for i in range(20, min(34, len(pose.keypoints))):
            if i < len(pose.keypoints) and i < len(pose.keypoints3D):
                kx, ky = pose.keypoints[i]
                # Check 2D overlap
                if abs(kx - logo_center_x) < w and abs(ky - logo_center_y) < h:
                    # Check if in front (3D Z > logo keypoint Z)
                    if logo.nearest_keypoint_idx < len(pose.keypoints3D):
                        hand_z = pose.keypoints3D[i][2]
                        logo_z = pose.keypoints3D[logo.nearest_keypoint_idx][2]
                        if hand_z > logo_z + 50:  # Hand is significantly in front
                            return True
        
        return False
    
    def _get_adaptive_timeout(self, logo: TrackedLogo) -> int:
        base_seconds = 2.0 if logo.state == "tentative" else 4.0
        
        if logo.frames_tracked_without_detection > 0:
            if logo.flow_quality > 0.7:  # High quality flow tracking
                base_seconds = 20.0  # Allow 20 seconds of flow tracking
            elif logo.flow_quality > 0.5:  # Medium quality
                base_seconds = 10.0  # Allow 10 seconds
            elif logo.flow_quality > 0.3:  # Low quality
                base_seconds = 5.0  # Allow 5 seconds
            else:
                base_seconds = 2.0  # Fallback to short timeout
        else:
            # Normal detection-based tracking timeouts
            if logo.embedding is not None:
                base_seconds += 2.0
            area = max(1, logo.bbox[2] * logo.bbox[3])
            if area > 20000:
                base_seconds += 1.5
            elif area > 8000:
                base_seconds += 0.5
        
        timeout_frames = int(base_seconds * self.video_fps)
        logging.debug(
            f"Track {logo.track_id}: Adaptive timeout={timeout_frames} frames ({base_seconds:.1f}s, "
            f"state={logo.state}, flow_quality={logo.flow_quality:.2f}, "
            f"frames_without_detection={logo.frames_tracked_without_detection})"
        )
        return timeout_frames
    

    
    def _compute_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _bbox_contains(self, outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
        x1, y1, w1, h1 = outer
        x2, y2, w2, h2 = inner
        
        return (x2 >= x1 and y2 >= y1 and 
                x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1)
    
    def _merge_nested_bboxes(self, detections: List[Tuple[Tuple[int, int, int, int], float]]) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if len(detections) <= 1:
            return detections

        # Sort by area (largest first) to prioritize keeping larger boxes
        sorted_dets = sorted(detections, key=lambda d: d[0][2] * d[0][3], reverse=True)
        
        merged = []
        
        for bbox_i, conf_i in sorted_dets:
            # Check if this bbox is contained by any already merged bbox
            is_contained = False
            
            for idx, (merged_bbox, merged_conf) in enumerate(merged):
                if self._bbox_contains(merged_bbox, bbox_i):
                    # Current bbox is inside an already merged one, skip it
                    is_contained = True
                    # Update confidence to max
                    merged[idx] = (merged_bbox, max(merged_conf, conf_i))
                    break
                elif self._bbox_contains(bbox_i, merged_bbox):
                    # Current bbox contains a previously merged one, replace it
                    merged[idx] = (bbox_i, max(merged_conf, conf_i))
                    is_contained = True
                    break
            
            if not is_contained:
                merged.append((bbox_i, conf_i))
        
        logging.debug(f"Merged {len(detections)} detections into {len(merged)} (removed {len(detections) - len(merged)} nested boxes)")
        return merged

    def _compute_embeddings_for_detections(
        self, frame: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], float, set]]
    ) -> Optional[np.ndarray]:
        if self.embedder is None or not detections:
            return None
        boxes_xywh = []
        for bbox, _, _ in detections:
            boxes_xywh.append(tuple(map(int, bbox)))
        return self.embedder.extract(frame, boxes_xywh)

    def _build_detection_data(
        self, frame: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], float, set]]
    ) -> List[Dict[str, Any]]:
        embeddings = self._compute_embeddings_for_detections(frame, detections)
        detection_data: List[Dict[str, Any]] = []
        for idx, (bbox, conf, sources) in enumerate(detections):
            emb = None
            if embeddings is not None and idx < len(embeddings):
                candidate = embeddings[idx]
                if float(np.linalg.norm(candidate)) > 1e-6:
                    emb = candidate
            detection_data.append(
                {
                    "bbox": bbox,
                    "confidence": conf,
                    "sources": set(sources),
                    "embedding": emb,
                }
            )
        return detection_data

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        return 1.0 - float(np.dot(a, b) / denom)

    @staticmethod
    def _bbox_iou_xywh(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = max(0, aw) * max(0, ah) + max(0, bw) * max(0, bh) - intersection
        return intersection / union if union > 0 else 0.0

    def _motion_cost(self, logo: TrackedLogo, detection_bbox: Tuple[int, int, int, int]) -> float:
        """Combine overlap and center displacement into a bounded geometric cost."""
        predicted_bbox = logo.flow_prediction
        if predicted_bbox is None and logo.kalman_filter is not None:
            predicted_bbox = logo.kalman_filter.predict_without_update()
        if predicted_bbox is None:
            predicted_bbox = logo.bbox
        iou = self._bbox_iou_xywh(predicted_bbox, detection_bbox)
        px, py, pw, ph = predicted_bbox
        dx, dy, dw, dh = detection_bbox
        p_center = (px + pw / 2, py + ph / 2)
        d_center = (dx + dw / 2, dy + dh / 2)
        distance = float(np.hypot(p_center[0] - d_center[0], p_center[1] - d_center[1]))
        scale = max(1.0, float(np.hypot(max(pw, dw), max(ph, dh))))
        normalized_distance = min(distance / scale, 1.0)
        # Reject clearly unrelated boxes while allowing reasonable frame-to-frame motion.
        if iou == 0.0 and distance > 2.5 * scale:
            return self.invalid_cost
        return 0.65 * (1.0 - iou) + 0.35 * normalized_distance

    def _build_cost_matrix(
        self, tracks: List[TrackedLogo], detection_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        if not tracks or not detection_data:
            return np.zeros((len(tracks), len(detection_data)), dtype=np.float32)
        cost_matrix = np.full(
            (len(tracks), len(detection_data)),
            self.invalid_cost,
            dtype=np.float32,
        )
        for i, logo in enumerate(tracks):
            for j, detection in enumerate(detection_data):
                detection_embedding = detection.get("embedding")
                motion_cost = self._motion_cost(logo, detection["bbox"])
                if motion_cost >= self.invalid_cost:
                    continue
                if logo.embedding is not None and detection_embedding is not None:
                    dist = self._cosine_distance(logo.embedding, detection_embedding)
                    if dist > self.appearance_threshold:
                        continue
                    appearance_cost = dist / max(self.appearance_threshold, 1e-6)
                    cost_matrix[i, j] = 0.65 * appearance_cost + 0.35 * motion_cost
                else:
                    # CLIP is optional; use predicted box overlap and motion when
                    # either side has no usable appearance embedding.
                    cost_matrix[i, j] = motion_cost
        return cost_matrix

    def _spawn_track(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        sources: set,
        embedding: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
        confirmed: bool = False,
    ) -> TrackedLogo:
        if track_id is None:
            track_id = self.next_track_id
            self.next_track_id += 1
        color = self.colors[track_id % len(self.colors)]
        new_logo = TrackedLogo(
            track_id=track_id,
            bbox=bbox,
            confidence=confidence,
            detection_sources=set(sources),
            color=color,
        )
        new_logo.update_bbox(bbox, confidence, sources)
        if embedding is not None:
            new_logo.embedding = embedding.astype(np.float32)
        if confirmed:
            new_logo.state = "confirmed"
            new_logo.consecutive_hits = max(self.confirmation_hits, 1)
        
        # Initialize Kalman filter for new track
        new_logo.kalman_filter = KalmanFilter(bbox, self.video_fps)
        new_logo.use_kalman = True
        
        # Initialize bbox stabilizer for smooth pasting
        new_logo.bbox_stabilizer = BboxStabilizer(
            ema_alpha=0.35,  # Slightly more responsive than 0.3
            lock_aspect_ratio=True,
            size_quantization=4,
            motion_threshold=1.5,
            enable_subpixel=True
        )
        
        return new_logo

    def _archive_track(self, logo: TrackedLogo) -> None:
        if logo.embedding is None:
            return
        entry = {
            "track_id": logo.track_id,
            "embedding": logo.embedding.copy(),
            "area": max(1, logo.bbox[2] * logo.bbox[3]),
        }
        self.track_archive.append(entry)
        if len(self.track_archive) > self.archive_maxlen:
            self.track_archive.pop(0)

    def _revive_from_archive(self, detection: Dict[str, Any]) -> Optional[TrackedLogo]:
        emb = detection.get("embedding")
        if emb is None or not self.track_archive:
            return None
        det_area = max(1, detection["bbox"][2] * detection["bbox"][3])
        best_idx = None
        best_dist = 1.0
        for idx, entry in enumerate(self.track_archive):
            stored_emb = entry["embedding"]
            dist = self._cosine_distance(stored_emb, emb)
            if dist >= self.reid_threshold or dist >= best_dist:
                continue
            area_ratio = det_area / float(entry["area"])
            if 0.5 <= area_ratio <= 2.0:
                best_idx = idx
                best_dist = dist
        if best_idx is None:
            return None
        entry = self.track_archive.pop(best_idx)
        revived = self._spawn_track(
            detection["bbox"],
            detection["confidence"],
            detection["sources"],
            embedding=emb,
            track_id=entry["track_id"],
            confirmed=True,
        )
        return revived
    
    def update(self, frame: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], float, set]], 
               region_masks: Optional[Dict] = None, frame_idx: int = 0, 
               pipeline: Optional['LogoFlippingPipeline'] = None,
               full_frame_hand_mask: Optional[np.ndarray] = None) -> List[TrackedLogo]:
        """Update tracking with new frame and detections."""
        if region_masks is None:
            region_masks = {}
        # Prepare grayscale for optical flow
        curr_gray = None
        if self.use_optical_flow:
            if len(frame.shape) == 3:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                curr_gray = frame.copy()
        
        # Try to track existing logos with fine-grained grid-based optical flow
        if self.use_optical_flow and self.use_fine_grid_tracking and curr_gray is not None:
            for logo in self.tracked_logos:
                if logo.grid_tracking_points is not None and logo.state == "confirmed":
                    # Try grid tracking
                    result = self._track_mask_grid(curr_gray, logo)
                    if result is not None:
                        flow_bbox, occlusion_grid, flow_quality = result
                        
                        # Check if too much occlusion (hand covering logo)
                        if occlusion_grid is not None:
                            total_cells = np.prod(occlusion_grid.shape)
                            occluded_cells = np.count_nonzero(occlusion_grid)
                            occlusion_ratio = occluded_cells / total_cells if total_cells > 0 else 0
                            
                            if occlusion_ratio > 0.5:  # More than 50% occluded
                                logging.warning(
                                    f"Track {logo.track_id}: High occlusion ({occlusion_ratio*100:.1f}%), "
                                    f"skipping grid tracking to avoid tracking hand"
                                )
                                result = None  # Invalidate result
                        
                        if result is not None:
                            # Store as flow prediction - will be used if no detection matches
                            logo.flow_prediction = flow_bbox
                            logo.occlusion_grid = occlusion_grid
                            logo.flow_quality = flow_quality
                            
                            # Track displacement for velocity estimation
                            old_cx, old_cy = logo.bbox[0] + logo.bbox[2]//2, logo.bbox[1] + logo.bbox[3]//2
                            new_cx, new_cy = flow_bbox[0] + flow_bbox[2]//2, flow_bbox[1] + flow_bbox[3]//2
                            displacement = (new_cx - old_cx, new_cy - old_cy)
                            logo.displacement_history.append(displacement)
                    
                    # If grid tracking failed or was rejected due to occlusion
                    if result is None:
                        # Grid tracking failed, try velocity prediction
                        predicted_bbox = self._predict_bbox_from_velocity(logo)
                        if predicted_bbox is not None:
                            logo.flow_prediction = predicted_bbox
                            logo.flow_quality = 0.3  # Lower quality for velocity prediction
                        else:
                            logo.flow_prediction = None
                            logo.flow_quality = 0.0
                else:
                    # No grid tracking available, try velocity prediction
                    if self.use_optical_flow and len(logo.displacement_history) >= 2:
                        predicted_bbox = self._predict_bbox_from_velocity(logo)
                        if predicted_bbox is not None:
                            logo.flow_prediction = predicted_bbox
                            logo.flow_quality = 0.3  # Lower quality for velocity prediction
                        else:
                            logo.flow_prediction = None
                            logo.flow_quality = 0.0
                    else:
                        logo.flow_prediction = None
                        logo.flow_quality = 0.0
        
        curr_pose = self._load_pose(frame_idx) if self.use_pose_prediction else None

        if curr_pose is not None:
            detections = self._filter_by_body_proximity(detections, curr_pose)

        dets_for_merge = [(bbox, conf) for bbox, conf, _ in detections]
        merged = self._merge_nested_bboxes(dets_for_merge)
        merged_detections = []
        for bbox, conf in merged:
            for orig_bbox, orig_conf, orig_sources in detections:
                if bbox == orig_bbox and conf == orig_conf:
                    merged_detections.append((bbox, conf, orig_sources))
                    break
        detections = merged_detections

        detection_data = self._build_detection_data(frame, detections)

        track_indices = list(range(len(self.tracked_logos)))
        detection_indices = list(range(len(detection_data)))
        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(track_indices)
        unmatched_detections = set(detection_indices)

        if track_indices and detection_indices:
            cost_matrix = self._build_cost_matrix(
                [self.tracked_logos[i] for i in track_indices],
                detection_data,
            )
            if cost_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for row, col in zip(row_ind, col_ind):
                    if cost_matrix[row, col] <= self.max_association_cost:
                        track_idx = track_indices[row]
                        matches.append((track_idx, col))

        for track_idx, det_idx in matches:
            unmatched_tracks.discard(track_idx)
            unmatched_detections.discard(det_idx)
            logo = self.tracked_logos[track_idx]
            detection = detection_data[det_idx]
            # Get mask for this detection
            det_mask = region_masks.get(detection["bbox"])
            logo.update_bbox(detection["bbox"], detection["confidence"], detection["sources"], mask=det_mask)
            logo.pending_embedding = detection["embedding"]
            
            # Update high-quality mask if available
            if det_mask is not None:
                quality = self._calculate_mask_quality(det_mask, detection["bbox"])
                if quality > logo.mask_quality_score:
                    logo.high_quality_mask = det_mask.copy()
                    logo.mask_quality_score = quality
                    logo.mask_source = "detection"
                    
                    # Extract color signature for this logo
                    logo.logo_color_histogram = self._extract_logo_color_signature(
                        frame, det_mask, detection["bbox"]
                    )
                    
                    logging.debug(
                        f"Track {logo.track_id}: High-quality mask updated from detection, "
                        f"quality={quality:.3f}, source={logo.mask_source}"
                    )
            
            # Reset counter - we have a fresh detection
            logo.frames_tracked_without_detection = 0
            logo.bbox_history.append(detection["bbox"])
            
            # Clear flow history to reset consistency baseline with fresh detection
            logo.flow_history.clear()
            
            # Update fine-grained mask grid and reference (only if fine-grid tracking enabled)
            if self.use_optical_flow and self.use_fine_grid_tracking:
                # Get mask for this detection if available
                det_mask = region_masks.get(detection["bbox"])
                logo.grid_tracking_points = self._extract_mask_grid(frame, detection["bbox"], det_mask)
                logo.mask_grid = logo.grid_tracking_points['mask_grid'] if logo.grid_tracking_points else None
                
                # Store reference patch and mask
                x, y, w, h = detection["bbox"]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                if x2 > x1 and y2 > y1:
                    logo.reference_patch = frame[y1:y2, x1:x2].copy()
                    if det_mask is not None:
                        logo.reference_mask = det_mask.copy()
                    else:
                        logo.reference_mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255
                logo.hand_mask = None  # Clear hand mask on good detection
                logo.occlusion_grid = None  # Clear occlusion grid
            
            if curr_pose is not None and logo.nearest_keypoint_idx is None:
                logo.nearest_keypoint_idx = self._find_nearest_keypoint(detection["bbox"], curr_pose)
            if logo.state == "tentative" and logo.consecutive_hits >= self.confirmation_hits:
                logo.state = "confirmed"

        # Secondary embedding-based re-identification for ambiguous matches
        if self.embedder is not None and unmatched_tracks and unmatched_detections:
            for det_idx in list(unmatched_detections):
                detection = detection_data[det_idx]
                emb = detection["embedding"]
                if emb is None:
                    continue
                best_track = None
                best_dist = 1.0
                for track_idx in list(unmatched_tracks):
                    logo = self.tracked_logos[track_idx]
                    if logo.embedding is None or logo.frames_since_detection < 2:
                        continue
                    dist = self._cosine_distance(logo.embedding, emb)
                    if dist >= self.reid_threshold or dist >= best_dist:
                        continue
                    best_track = track_idx
                    best_dist = dist
                if best_track is not None:
                    logo = self.tracked_logos[best_track]
                    logo.update_bbox(detection["bbox"], detection["confidence"], detection["sources"])
                    logo.pending_embedding = detection["embedding"]
                    unmatched_tracks.discard(best_track)
                    unmatched_detections.discard(det_idx)
                    if logo.state == "tentative" and logo.consecutive_hits >= self.confirmation_hits:
                        logo.state = "confirmed"

        # Tracks that remained unmatched: use optical flow prediction or velocity prediction
        for track_idx in unmatched_tracks:
            if 0 <= track_idx < len(self.tracked_logos):
                logo = self.tracked_logos[track_idx]
                
                # Increment mask age regardless of whether we use flow prediction
                logo.frames_since_mask_update += 1
                
                # If we have a flow prediction with good quality, use it
                if logo.flow_prediction is not None and logo.flow_quality >= self.min_flow_quality_for_tracking:
                    old_bbox = logo.bbox
                    new_bbox = logo.flow_prediction
                    
                    # Check if movement is reasonable (not jumping too far)
                    old_cx, old_cy = old_bbox[0] + old_bbox[2]//2, old_bbox[1] + old_bbox[3]//2
                    new_cx, new_cy = new_bbox[0] + new_bbox[2]//2, new_bbox[1] + new_bbox[3]//2
                    movement = np.sqrt((new_cx - old_cx)**2 + (new_cy - old_cy)**2)
                    
                    # Reject if movement is too large (likely tracking hand instead of logo)
                    if movement > 50:  # More than 50 pixels per frame
                        logging.warning(
                            f"Track {logo.track_id}: Excessive movement {movement:.1f}px, "
                            f"rejecting flow prediction (possible occlusion)"
                        )
                        logo.mark_missed()
                        continue
                    
                    logo.bbox = new_bbox
                    logo.frames_tracked_without_detection += 1
                    logo.frames_since_detection += 1
                    logo.consecutive_hits = 0
                    
                    logo.update_mask_position(old_bbox, logo.bbox)
                    
                    # Add to centroid history
                    cx = logo.bbox[0] + logo.bbox[2] // 2
                    cy = logo.bbox[1] + logo.bbox[3] // 2
                    logo.centroid_history.append((cx, cy))
                    
                    # Add to bbox history for smoothing
                    logo.bbox_history.append(logo.bbox)
                    
                    logging.debug(
                        f"Track {logo.track_id}: Using flow prediction (quality={logo.flow_quality:.2f}, "
                        f"movement={movement:.1f}px, frames_without_detection={logo.frames_tracked_without_detection})"
                    )
                    continue
                
                # Flow prediction is low quality or unavailable
                # Try to refine mask using color-based detection (for tiny visible parts)
                if logo.logo_color_histogram is not None and logo.frames_since_mask_update > 5:
                    refined_mask = self._refine_mask_by_color(
                        frame, logo.bbox, logo.logo_color_histogram, threshold=0.5
                    )
                    if refined_mask is not None:
                        quality = self._calculate_mask_quality(refined_mask, logo.bbox)
                        if quality > 0.3:  # Minimum quality threshold
                            logo.current_mask = refined_mask
                            logo.mask_quality_score = quality
                            logo.mask_source = "color"
                            logo.frames_since_mask_update = 0
                            
                            logging.debug(
                                f"Track {logo.track_id}: Mask refined by color matching, "
                                f"quality={quality:.3f}"
                            )
                
                # No good flow prediction - mark as missed and KEEP bbox at current position
                # This prevents bbox from jumping when we lose detection
                logo.mark_missed()
                
                # NOTE: Removed the redundant fine-grid flow usage here that was ignoring
                # flow quality checks and causing bbox jumps. The bbox stays frozen when
                # we have no reliable tracking signal, which is the correct behavior for
                # a stationary logo that YOLO temporarily lost.
                
                logging.debug(
                    f"Track {logo.track_id}: No reliable prediction - keeping bbox frozen "
                    f"(flow_quality={logo.flow_quality:.2f}, frames_since_detection={logo.frames_since_detection})"
                )

        # Try to revive long-term archived tracks
        revived_tracks: List[TrackedLogo] = []
        for det_idx in list(unmatched_detections):
            detection = detection_data[det_idx]
            revived = self._revive_from_archive(detection)
            if revived:
                unmatched_detections.discard(det_idx)
                revived_tracks.append(revived)
                logging.info(
                    f"Frame {frame_idx}: Revived Track {revived.track_id} from archive."
                )
        if revived_tracks:
            self.tracked_logos.extend(revived_tracks)

        # Create new tentative tracks for the remaining detections
        for det_idx in unmatched_detections:
            detection = detection_data[det_idx]
            det_mask = region_masks.get(detection["bbox"])
            new_logo = self._spawn_track(
                detection["bbox"],
                detection["confidence"],
                detection["sources"],
                embedding=detection["embedding"],
            )
            if det_mask is not None:
                new_logo.current_mask = det_mask.copy()
            
            # Initialize fine-grained mask grid and reference (only if fine-grid tracking enabled)
            if self.use_optical_flow and self.use_fine_grid_tracking:
                # Initialize bbox history
                new_logo.bbox_history.append(detection["bbox"])
                new_logo.frames_tracked_without_detection = 0
                
                det_mask = region_masks.get(detection["bbox"])
                new_logo.grid_tracking_points = self._extract_mask_grid(frame, detection["bbox"], det_mask)
                new_logo.mask_grid = new_logo.grid_tracking_points['mask_grid'] if new_logo.grid_tracking_points else None
                
                x, y, w, h = detection["bbox"]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                if x2 > x1 and y2 > y1:
                    new_logo.reference_patch = frame[y1:y2, x1:x2].copy()
                    if det_mask is not None:
                        new_logo.reference_mask = det_mask.copy()
                    else:
                        new_logo.reference_mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255
            
            if curr_pose is not None:
                new_logo.nearest_keypoint_idx = self._find_nearest_keypoint(detection["bbox"], curr_pose)
            self.tracked_logos.append(new_logo)
            logging.info(f"Frame {frame_idx}: Created new Track {new_logo.track_id}")

        # Update occlusion state and apply deferred embedding updates
        for logo in self.tracked_logos:
            if curr_pose is not None:
                logo.is_occluded = self._check_occlusion(logo, curr_pose)
            else:
                logo.is_occluded = False
            if logo.pending_embedding is not None and not logo.is_occluded:
                logo.update_embedding(logo.pending_embedding)
            logo.pending_embedding = None
            
            # NOTE: Unoccluded reference caching is now done ONLY in process_video()
            # where we have access to the proper pipeline result (flipped_image, region.bbox).
            # The _update_unoccluded_reference call here was causing quality issues because:
            # 1. It doesn't have access to the flipped frame
            # 2. It uses tracked bbox which may have drifted from original YOLO detection
            # 3. It creates dual-path caching confusion with the process_video() caching
            # if logo.state == "confirmed" and logo.frames_since_detection == 0:
            #     self._update_unoccluded_reference(frame, logo)
            
            # Hand occlusion detection - uses pre-computed mask if available, or detects fresh
            if self.hand_detector is not None and logo.state == "confirmed":
                is_hand_occluded, hand_mask_in_logo = self._detect_hand_occlusion_mediapipe(
                    frame, logo, pipeline, full_frame_hand_mask
                )
                
                if is_hand_occluded and hand_mask_in_logo is not None:
                    # Logo is occluded by hand - mark it and store hand mask
                    logo.is_hand_occluded = True
                    logo.current_hand_mask = hand_mask_in_logo
                    logo.frames_since_unoccluded += 1
                    
                    logging.debug(
                        f"Track {logo.track_id}: Hand occluded for {logo.frames_since_unoccluded} frames"
                    )
                else:
                    # Logo is not occluded by hand
                    if logo.is_hand_occluded:
                        logging.info(
                            f"Track {logo.track_id}: Hand occlusion cleared after "
                            f"{logo.frames_since_unoccluded} frames"
                        )
                    
                    logo.is_hand_occluded = False
                    logo.current_hand_mask = None
                    logo.frames_since_unoccluded = 0
            
            # Smart SAM triggering for mask refinement
            if logo.state == "confirmed" and pipeline is not None:
                # ============================================================
                # HIGH-QUALITY MASK GENERATION (Issue 2 - Solution A)
                # Run SAM on tracked bbox for pixel-perfect segmentation
                # ============================================================
                
                # Always run SAM refinement on fresh detections (frames_since_detection == 0)
                if self.sam_refinement_on_detection and logo.frames_since_detection == 0:
                    sam_mask = self._refine_mask_with_sam_on_bbox(frame, logo, pipeline)
                    if sam_mask is not None:
                        logging.info(
                            f"Track {logo.track_id}: SAM bbox mask on detection - "
                            f"quality={logo.mask_quality_score:.3f}"
                        )
                
                # Also trigger SAM refinement based on quality degradation
                should_refine = self._should_trigger_sam_refinement(logo, frame)
                
                if should_refine:
                    # Try SAM-based occlusion detection (pixel-perfect precision)
                    result = self._detect_occlusion_with_sam(frame, logo, pipeline)
                    
                    if result is not None:
                        refined_mask, occluder_mask = result
                        quality = self._calculate_mask_quality(refined_mask, logo.bbox)
                        
                        # Always update with SAM mask (it's the best)
                        logo.high_quality_mask = refined_mask.copy()
                        logo.current_mask = refined_mask.copy()
                        logo.mask_quality_score = quality
                        logo.mask_source = "sam_refined"
                        logo.frames_since_mask_update = 0
                        logo.occluder_mask = occluder_mask.copy() if occluder_mask is not None else None
                        
                        # Update color signature
                        logo.logo_color_histogram = self._extract_logo_color_signature(
                            frame, refined_mask, logo.bbox
                        )
                        
                        if occluder_mask is not None:
                            occluder_pixels = np.count_nonzero(occluder_mask)
                            visible_pixels = np.count_nonzero(refined_mask)
                            logging.info(
                                f"Track {logo.track_id}: SAM refined - quality={quality:.3f}, "
                                f"visible={visible_pixels}px, occluder={occluder_pixels}px"
                            )
                        else:
                            logging.info(
                                f"Track {logo.track_id}: SAM mask updated (no occlusion), quality={quality:.3f}"
                            )
                    else:
                        # SAM occlusion detection failed, try simpler bbox-based SAM refinement
                        sam_mask = self._refine_mask_with_sam_on_bbox(frame, logo, pipeline)
                        if sam_mask is not None:
                            logging.debug(
                                f"Track {logo.track_id}: SAM bbox fallback refinement, "
                                f"quality={logo.mask_quality_score:.3f}"
                            )
                        else:
                            # All SAM methods failed
                            logging.debug(
                                f"Track {logo.track_id}: SAM refinement failed, keeping current mask"
                            )
            
            # Always prefer high-quality mask for current_mask (used in visualization)
            if (logo.high_quality_mask is not None and 
                logo.mask_quality_score > 0.5 and
                logo.mask_source in ['sam', 'sam_refined', 'sam_bbox', 'detection']):
                # Use high-quality mask instead of degraded warped mask
                logo.current_mask = logo.high_quality_mask.copy()
                logging.debug(
                    f"Track {logo.track_id}: Using high-quality {logo.mask_source} mask for display "
                    f"(quality={logo.mask_quality_score:.3f})"
                )

        # Remove stale tracks
        active_tracks: List[TrackedLogo] = []
        for logo in self.tracked_logos:
            timeout = self._get_adaptive_timeout(logo)
            if logo.state == "tentative" and logo.frames_since_detection > self.tentative_patience:
                logging.info(
                    f"Frame {frame_idx}: Removed tentative Track {logo.track_id} "
                    f"(frames_since_detection={logo.frames_since_detection} > {self.tentative_patience})"
                )
                self._archive_track(logo)
                continue
            if logo.frames_since_detection >= timeout:
                logging.info(
                    f"Frame {frame_idx}: Removed Track {logo.track_id} "
                    f"(frames_since_detection={logo.frames_since_detection} >= timeout={timeout}, "
                    f"flow_quality={logo.flow_quality:.2f}, "
                    f"frames_tracked_without_detection={logo.frames_tracked_without_detection})"
                )
                self._archive_track(logo)
                continue
            active_tracks.append(logo)
        self.tracked_logos = active_tracks
        
        # Store current frame for next optical flow computation
        if self.use_optical_flow and curr_gray is not None:
            self.prev_gray = curr_gray.copy()
            self.gray_frame_history.append(curr_gray.copy())
        
        self.frame_count += 1

        return self.tracked_logos
    
    def draw_hands(self, frame: np.ndarray, hand_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw detected hand regions on frame."""
        if self.hand_detector is None:
            return frame
        
        vis_frame = frame.copy()
        
        # Draw hand mask overlay (cyan color for hands)
        if hand_mask is not None and np.count_nonzero(hand_mask) > 0:
            hand_bool = hand_mask > 127
            # Create cyan overlay for hand regions
            cyan_overlay = np.zeros_like(vis_frame)
            cyan_overlay[:, :, 1] = 255  # Green channel
            cyan_overlay[:, :, 0] = 255  # Blue channel (cyan = green + blue)
            
            vis_frame[hand_bool] = cv2.addWeighted(
                vis_frame[hand_bool],
                0.6,
                cyan_overlay[hand_bool],
                0.4,
                0
            )
        
        # Draw hand landmarks and bboxes
        if hasattr(self.hand_detector, '_last_hands') and self.hand_detector._last_hands:
            for hand_data in self.hand_detector._last_hands:
                # Draw bbox
                x, y, w, h = hand_data['bbox']
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Cyan bbox
                
                # Draw handedness label
                label = f"{hand_data['handedness']}"
                cv2.putText(vis_frame, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw landmarks
                for lx, ly in hand_data['landmarks']:
                    cv2.circle(vis_frame, (lx, ly), 3, (0, 255, 255), -1)  # Yellow points
        
        return vis_frame
    
    def draw_tracks(self, frame: np.ndarray, hand_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw tracking visualization on frame."""
        vis_frame = frame.copy()
        
        # Draw hand visualization first (underneath logo overlays)
        if hand_mask is not None:
            vis_frame = self.draw_hands(vis_frame, hand_mask)
        
        # First pass: Draw mask overlays with transparency
        for logo in self.tracked_logos:
            if logo.state != "confirmed":
                continue
            
            # Use best available mask: high_quality_mask if good, else current_mask
            if (logo.high_quality_mask is not None and 
                logo.mask_quality_score > 0.5 and
                logo.mask_source in ['sam', 'sam_refined', 'detection']):
                mask = logo.high_quality_mask
            elif logo.current_mask is not None:
                mask = logo.current_mask
            else:
                continue
            
            x, y, w, h = logo.bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            
            if logo.hand_mask is not None:
                hand_mask_resized = cv2.resize(logo.hand_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                mask_resized = cv2.bitwise_and(mask_resized, cv2.bitwise_not(hand_mask_resized))
            
            color = logo.color
            overlay = np.zeros_like(vis_frame[y1:y2, x1:x2])
            overlay[:, :, 0] = color[0]
            overlay[:, :, 1] = color[1]
            overlay[:, :, 2] = color[2]
            
            mask_bool = mask_resized > 127
            if np.any(mask_bool):
                vis_frame[y1:y2, x1:x2][mask_bool] = cv2.addWeighted(
                    vis_frame[y1:y2, x1:x2][mask_bool],
                    0.6,
                    overlay[mask_bool],
                    0.4,
                    0
                )
            
            # Draw occluder mask overlay if detected by SAM (red overlay)
            if logo.occluder_mask is not None:
                occluder_resized = cv2.resize(logo.occluder_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                occluder_bool = occluder_resized > 127
                if np.any(occluder_bool):
                    # Red overlay for occluder
                    red_overlay = np.zeros_like(vis_frame[y1:y2, x1:x2])
                    red_overlay[:, :, 2] = 255  # Red channel
                    vis_frame[y1:y2, x1:x2][occluder_bool] = cv2.addWeighted(
                        vis_frame[y1:y2, x1:x2][occluder_bool],
                        0.5,
                        red_overlay[occluder_bool],
                        0.5,
                        0
                    )
        
        for logo in self.tracked_logos:
            if logo.state != "confirmed":
                continue
            
            x, y, w, h = logo.bbox
            color = logo.color
            
            if logo.frames_since_detection > 0:
                self._draw_dashed_rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            else:
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            if self.use_optical_flow and logo.grid_tracking_points is not None:
                grid_shape = logo.grid_tracking_points['grid_shape']
                cell_points = logo.grid_tracking_points['cell_points']
                mask_grid = logo.grid_tracking_points['mask_grid']
                grid_rows, grid_cols = grid_shape
                
                x, y, w, h = logo.bbox
                x1, y1 = max(0, x), max(0, y)
                
                if grid_rows * grid_cols < 400:
                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            if not mask_grid[row, col]:
                                continue
                            
                            # Calculate cell bounds
                            cy1 = y1 + row * self.grid_cell_size
                            cx1 = x1 + col * self.grid_cell_size
                            cy2 = y1 + min((row + 1) * self.grid_cell_size, w)
                            cx2 = x1 + min((col + 1) * self.grid_cell_size, h)
                            
                            # Color-code by occlusion status
                            if logo.occlusion_grid is not None and logo.occlusion_grid[row, col]:
                                cell_color = (0, 0, 255)  # Red for occluded
                            else:
                                cell_color = (0, 255, 0)  # Green for active
                            
                            # Draw cell boundary (very thin for fine grid)
                            cv2.rectangle(vis_frame, (cx1, cy1), (cx2, cy2), cell_color, 1)
                else:
                    # For large grids, just show summary info
                    active_cells = np.count_nonzero(mask_grid)
                    if logo.occlusion_grid is not None:
                        occluded_cells = np.count_nonzero(logo.occlusion_grid)
                    else:
                        occluded_cells = 0
                    
                    # Draw summary text
                    summary = f"{active_cells} cells ({occluded_cells} occ)"
                    cv2.putText(
                        vis_frame,
                        summary,
                        (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )
            
            # Draw hand occlusion mask overlay if detected (legacy or MediaPipe)
            hand_mask_to_draw = None
            if logo.is_hand_occluded and logo.current_hand_mask is not None:
                # MediaPipe-detected hand mask (new system)
                hand_mask_to_draw = logo.current_hand_mask
            elif logo.hand_mask is not None:
                # Legacy hand mask
                hand_mask_to_draw = logo.hand_mask
            
            if hand_mask_to_draw is not None:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                if x2 > x1 and y2 > y1:
                    # Create red/orange overlay for hand-occluded regions
                    mask_resized = cv2.resize(hand_mask_to_draw, (x2 - x1, y2 - y1))
                    # Orange color for hand occlusion (differentiates from SAM occluder which is red)
                    orange_overlay = np.zeros_like(vis_frame[y1:y2, x1:x2])
                    orange_overlay[:, :, 2] = 255  # Red channel
                    orange_overlay[:, :, 1] = 165  # Green channel (orange)
                    
                    hand_bool = mask_resized > 127
                    if np.any(hand_bool):
                        vis_frame[y1:y2, x1:x2][hand_bool] = cv2.addWeighted(
                            vis_frame[y1:y2, x1:x2][hand_bool],
                            0.5,
                            orange_overlay[hand_bool],
                            0.5,
                            0
                        )
            
            sources = logo.detection_sources
            if sources:
                if sources == {'yolo'}:
                    src_label = 'YOLO'
                elif sources == {'ocr'}:
                    src_label = 'OCR'
                else:
                    src_label = 'YOLO+OCR'
            else:
                src_label = ''
            
            if logo.frames_tracked_without_detection > 0:
                flow_marker = f"[FLOW:{logo.frames_tracked_without_detection} Q:{logo.flow_quality:.2f}]"
            else:
                flow_marker = ""
            # Updated hand marker to show MediaPipe detection
            if logo.is_hand_occluded:
                hand_marker = f"[HAND:{logo.frames_since_unoccluded}]"
            elif logo.hand_mask is not None:
                hand_marker = "[HAND]"
            else:
                hand_marker = ""
            occ_marker = "[OCC]" if logo.is_occluded else ""
            mask_marker = f"[{logo.mask_source.upper()}:Q{logo.mask_quality_score:.2f}]" if logo.mask_quality_score > 0 else ""
            label = f"ID:{logo.track_id} {logo.confidence:.2f} {src_label}{flow_marker}{hand_marker}{occ_marker}{mask_marker}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Background for text
            cv2.rectangle(
                vis_frame,
                (x, y - label_size[1] - baseline - 5),
                (x + label_size[0], y),
                color,
                -1
            )
            # Text
            cv2.putText(
                vis_frame,
                label,
                (x, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            # Draw centroid trace
            if len(logo.centroid_history) > 1:
                points = list(logo.centroid_history)
                for i in range(len(points) - 1):
                    # Fade older points
                    alpha = (i + 1) / len(points)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(vis_frame, points[i], points[i + 1], color, thickness, cv2.LINE_AA)
                # Draw current centroid
                cx, cy = points[-1]
                cv2.circle(vis_frame, (cx, cy), 4, color, -1)
        return vis_frame
    
    def _draw_dashed_rectangle(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                                color: Tuple[int, int, int], thickness: int, dash_length: int = 10):
        """Draw a dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top and bottom
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Left and right
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def draw_pose(self, frame: np.ndarray, pose: Optional[PoseData], is_flipped: bool = False) -> np.ndarray:
        """Draw body pose keypoints and skeleton on frame."""
        if pose is None or not pose.keypoints:
            return frame
        
        vis_frame = frame.copy()
        frame_width = vis_frame.shape[1]
        
        # Helper to flip x coordinate if needed
        def flip_x(x):
            return frame_width - x if is_flipped else x
        
        # Draw person bounding boxes
        for px, py, pw, ph in pose.boxes:
            if is_flipped:
                # Flip box: new_x = width - (x + w)
                px_flipped = frame_width - (px + pw)
                cv2.rectangle(vis_frame, (int(px_flipped), int(py)), 
                            (int(px_flipped + pw), int(py + ph)), 
                            (0, 255, 255), 2)  # Yellow boxes
            else:
                cv2.rectangle(vis_frame, (int(px), int(py)), 
                            (int(px + pw), int(py + ph)), 
                            (0, 255, 255), 2)  # Yellow boxes
        
        # Define skeleton connections (standard COCO/MediaPipe format)
        skeleton = [
            # Torso
            (5, 6), (5, 11), (6, 12), (11, 12),
            # Arms
            (5, 7), (7, 9), (6, 8), (8, 10),
            # Legs  
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]
        
        # Draw skeleton connections
        for idx1, idx2 in skeleton:
            if idx1 < len(pose.keypoints) and idx2 < len(pose.keypoints):
                if idx1 < len(pose.keypoints_confidence) and idx2 < len(pose.keypoints_confidence):
                    conf1 = pose.keypoints_confidence[idx1]
                    conf2 = pose.keypoints_confidence[idx2]
                    # Only draw if both keypoints are confident
                    if conf1 > 0.3 and conf2 > 0.3:
                        kx1, ky1 = pose.keypoints[idx1]
                        kx2, ky2 = pose.keypoints[idx2]
                        pt1 = (int(flip_x(kx1)), int(ky1))
                        pt2 = (int(flip_x(kx2)), int(ky2))
                        cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw keypoints with labels
        for i, (kx, ky) in enumerate(pose.keypoints):
            if i < len(pose.keypoints_confidence):
                conf = pose.keypoints_confidence[i]
                if conf > 0.3:  # Only draw confident keypoints
                    x_pos = int(flip_x(kx))
                    y_pos = int(ky)
                    
                    # Color based on confidence (lime green)
                    color_intensity = int(255 * conf)
                    color = (0, color_intensity, 0)  # Green in BGR
                    
                    # Draw outer circle (lime green)
                    cv2.circle(vis_frame, (x_pos, y_pos), 6, (0, 255, 0), 3)
                    
                    # Draw keypoint index label (yellow text)
                    label = str(i)
                    cv2.putText(vis_frame, label, 
                               (x_pos + 8, y_pos - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        
        return vis_frame


def process_video(
    input_path: Path,
    output_viz_path: Path,
    output_clean_path: Path,
    output_inpainted_path: Optional[Path] = None,
    detection_settings: DetectionSettings = None,
    pipeline_config: PipelineConfig = None,
    pose_data_dir: Optional[Path] = None,
    use_pose_prediction: bool = True,
    use_clip_embeddings: bool = True,
    clip_model: str = "ViT-B/32",
    clip_device: Optional[str] = None,
    use_optical_flow: bool = True,
    use_fine_grid_tracking: bool = False,
    grid_cell_size: int = 10,
    sam_per_frame: bool = False,
    use_hand_detection: bool = True,
    hand_detection_confidence: float = 0.5,
):
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Input video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    
    # Auto-generate inpainted output path if not provided
    if output_inpainted_path is None:
        # Default: same directory as viz, named "inp.mp4"
        output_inpainted_path = output_viz_path.parent / "inp.mp4"
    
    # Create output directories if they don't exist
    output_viz_path.parent.mkdir(parents=True, exist_ok=True)
    output_clean_path.parent.mkdir(parents=True, exist_ok=True)
    output_inpainted_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc_options = [
        ('mp4v', 'mp4v'),
        ('avc1', 'H264'),
        ('XVID', 'XVID'),
        ('MJPG', 'MJPEG'),
    ]
    
    out_viz = None
    out_clean = None
    out_inpainted = None
    
    for codec_code, codec_name in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_code)
            out_viz = cv2.VideoWriter(str(output_viz_path), fourcc, fps, (width, height))
            out_clean = cv2.VideoWriter(str(output_clean_path), fourcc, fps, (width, height))
            out_inpainted = cv2.VideoWriter(str(output_inpainted_path), fourcc, fps, (width, height))
            
            if out_viz.isOpened() and out_clean.isOpened() and out_inpainted.isOpened():
                logging.info(f"Using codec: {codec_name} ({codec_code})")
                break
            else:
                if out_viz is not None:
                    out_viz.release()
                if out_clean is not None:
                    out_clean.release()
                if out_inpainted is not None:
                    out_inpainted.release()
                out_viz = None
                out_clean = None
                out_inpainted = None
        except Exception as e:
            logging.debug(f"Codec {codec_name} failed: {e}")
            continue
    
    if out_viz is None or out_clean is None or out_inpainted is None or not out_viz.isOpened() or not out_clean.isOpened() or not out_inpainted.isOpened():
        raise RuntimeError(
            f"Failed to create output video writers. Tried codecs: {[c[1] for c in fourcc_options]}. "
            f"Make sure the output paths are valid: {output_viz_path}, {output_clean_path}, {output_inpainted_path}"
        )
    
    # Initialize tracker and pipeline
    tracker = VideoLogoTracker(
        detection_settings,
        pose_data_dir=pose_data_dir,
        use_pose_prediction=use_pose_prediction,
        use_clip_embeddings=use_clip_embeddings,
        clip_model=clip_model,
        clip_device=clip_device,
        use_optical_flow=use_optical_flow,
        use_fine_grid_tracking=use_fine_grid_tracking,
        grid_cell_size=grid_cell_size,
        use_hand_detection=use_hand_detection,
        hand_detection_confidence=hand_detection_confidence,
    )
    tracker.video_fps = fps  # Set actual video FPS for adaptive timeout
    tracker.enable_sam_per_frame = sam_per_frame  # Enable per-frame SAM if requested
    pipeline = LogoFlippingPipeline(pipeline_config)
    
    import time
    frame_idx = 0
    frame_times = []  # Collect timing for statistics
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            logging.info(f"Processing frame {frame_idx}/{total_frames}")
            start_time = time.time()
            timing_breakdown = {}
            
            frame_original = frame
            
            t_flip_start = time.time()
            if pipeline_config.input_is_flipped:
                frame_test = cv2.flip(frame_original, 1)  # Flip for testing purposes
                logging.debug(f"Frame {frame_idx}: Flipped input for testing")
            else:
                frame_test = frame_original
            timing_breakdown['flip'] = time.time() - t_flip_start
            
            if pipeline_config.input_is_flipped:
                frame_for_detection = frame_original
            else:
                frame_for_detection = frame_test
            
            # ============================================================
            # HAND DETECTION - Must happen BEFORE pipeline.process_array()
            # so the hand mask can be used for two-pass inpainting
            # IMPORTANT: Hand detection on frame_test, then convert to detection space
            # ============================================================
            t_hand_start = time.time()
            full_frame_hand_mask = None
            hand_mask_for_pipeline = None
            if tracker.hand_detector is not None:
                sam_masker = pipeline._sam_masker if pipeline is not None else None
                # Detect hands on frame_test (same as tracking/visualization)
                full_frame_hand_mask = tracker.hand_detector.get_all_hands_mask(
                    frame_test, sam_masker
                )
                hand_pixels = np.count_nonzero(full_frame_hand_mask) if full_frame_hand_mask is not None else 0
                if hand_pixels > 0:
                    logging.info(f"Frame {frame_idx}: [HAND] Detected {hand_pixels} hand pixels in frame_test")
                    # Convert hand mask to detection space for pipeline
                    if pipeline_config.input_is_flipped:
                        # frame_for_detection = frame_original (not flipped)
                        # frame_test = flipped version
                        # So we need to flip the hand mask back
                        hand_mask_for_pipeline = cv2.flip(full_frame_hand_mask, 1)
                        logging.info(f"Frame {frame_idx}: [HAND] Flipped mask for pipeline (input_is_flipped=True)")
                    else:
                        hand_mask_for_pipeline = full_frame_hand_mask
                    logging.info(f"Frame {frame_idx}: [HAND] Passing {np.count_nonzero(hand_mask_for_pipeline)} hand pixels to pipeline")
                else:
                    logging.debug(f"Frame {frame_idx}: [HAND] No hand pixels detected")
            else:
                logging.debug(f"Frame {frame_idx}: [HAND] Hand detector is None (disabled?)")
            timing_breakdown['hand_detection'] = time.time() - t_hand_start
            
            t_detect_start = time.time()
            # Pass hand mask to pipeline for two-pass inpainting
            result = pipeline.process_array(
                frame_for_detection, 
                image_name=f"frame_{frame_idx:06d}",
                hand_mask=hand_mask_for_pipeline
            )
            timing_breakdown['detection_and_flip'] = time.time() - t_detect_start
            num_detections = len(result.regions)
            
            t_mapping_start = time.time()
            width = frame.shape[1]
            detections = []
            region_masks = {}
            for region in result.regions:
                if region.detection_confidence >= detection_settings.yolo_conf:
                    x, y, w, h = region.bbox
                    if pipeline_config.input_is_flipped:
                        x_mapped = width - (x + w)
                        bbox_mapped = (x_mapped, y, w, h)
                        mask_flipped = cv2.flip(region.mask, 1) if region.mask is not None else None
                        if mask_flipped is not None:
                            region_masks[bbox_mapped] = mask_flipped
                    else:
                        bbox_mapped = (x, y, w, h)
                        if region.mask is not None:
                            region_masks[bbox_mapped] = region.mask
                    detections.append((bbox_mapped, region.detection_confidence, set(region.detection_sources)))
            timing_breakdown['bbox_mapping'] = time.time() - t_mapping_start
            
            t_track_start = time.time()
            tracked_logos = tracker.update(
                frame_test, detections, region_masks, 
                frame_idx=frame_idx - 1, pipeline=pipeline,
                full_frame_hand_mask=full_frame_hand_mask
            )
            timing_breakdown['tracking'] = time.time() - t_track_start
            
            # Cache high-quality references from YOLO detections
            for logo in tracked_logos:
                # Skip if not a fresh YOLO detection
                if logo.state != "confirmed" or logo.frames_since_detection != 0:
                    continue
                    
                # Skip if hand is currently occluding (based on is_hand_occluded flag)
                if logo.is_hand_occluded:
                    continue
                
                x, y, w, h = logo.bbox  # In frame_test space
                
                # Find the matching region from pipeline result (any overlap is fine)
                best_region = None
                best_iou = 0.0
                
                for region in result.regions:
                    rx, ry, rw, rh = region.bbox
                    # Convert region bbox to frame_test space if needed
                    if pipeline_config.input_is_flipped:
                        rx_test = frame_test.shape[1] - (rx + rw)
                    else:
                        rx_test = rx
                    
                    # Calculate IoU just to find the best match
                    ix1 = max(x, rx_test)
                    iy1 = max(y, ry)
                    ix2 = min(x + w, rx_test + rw)
                    iy2 = min(y + h, ry + rh)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        intersection = (ix2 - ix1) * (iy2 - iy1)
                        union = w * h + rw * rh - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_region = region
                
                # Just need some overlap to match
                if best_region is None or best_iou < 0.1:
                    continue
                
                # Check occlusion: skip if >20% of bbox/mask is occluded by hand
                occlusion_ratio = 0.0
                if full_frame_hand_mask is not None:
                    h_img, w_img = frame_test.shape[:2]
                    x1_check, y1_check = max(0, x), max(0, y)
                    x2_check, y2_check = min(w_img, x + w), min(h_img, y + h)
                    bbox_area = (x2_check - x1_check) * (y2_check - y1_check)
                    if bbox_area > 0:
                        hand_pixels_in_bbox = np.count_nonzero(
                            full_frame_hand_mask[y1_check:y2_check, x1_check:x2_check]
                        )
                        occlusion_ratio = hand_pixels_in_bbox / bbox_area
                
                if occlusion_ratio > 0.20:
                    logging.debug(
                        f"Track {logo.track_id}: Skipping cache - occlusion {occlusion_ratio*100:.1f}% > 20%"
                    )
                    continue
                
                # Extract using the REGION's bbox (this is the clean YOLO detection)
                # NOT the tracked logo bbox (which might have drift)
                rx, ry, rw, rh = best_region.bbox
                h_det, w_det = result.flipped_image.shape[:2]
                rx1, ry1 = max(0, rx), max(0, ry)
                rx2, ry2 = min(w_det, rx + rw), min(h_det, ry + rh)
                
                if rx2 <= rx1 or ry2 <= ry1:
                    continue
                
                # Validate patch size is reasonable
                patch_h, patch_w = ry2 - ry1, rx2 - rx1
                if patch_h < 10 or patch_w < 10:
                    logging.debug(
                        f"Track {logo.track_id}: Skipping cache - patch too small ({patch_w}x{patch_h})"
                    )
                    continue
                
                # Extract from pipeline result - these are CLEAN from YOLO
                flipped_patch_det = result.flipped_image[ry1:ry2, rx1:rx2].copy()
                
                # ALSO extract the ORIGINAL unflipped logo from the detection frame
                # This is crucial for high-quality caching - the original has NO distortion
                # We can apply a simple flip at paste time which is much cleaner
                original_patch_det = frame_for_detection[ry1:ry2, rx1:rx2].copy()
                
                # Get mask - try SAM first, fall back to pipeline mask
                region_mask_det = None
                mask_source = "unknown"
                
                if pipeline._sam_masker is not None:
                    try:
                        # Use the ORIGINAL (unflipped) frame for SAM since SAM was 
                        # trained on natural images. We'll flip the mask after.
                        # The bbox is in detection space (frame_for_detection)
                        pipeline._sam_masker.set_image(frame_for_detection)
                        
                        # mask_from_box expects (x1, y1, x2, y2) format
                        sam_mask_full = pipeline._sam_masker.mask_from_box((rx, ry, rx + rw, ry + rh))
                        
                        if sam_mask_full is not None and np.count_nonzero(sam_mask_full) > 0:
                            # Crop mask to region bbox
                            region_mask_det = sam_mask_full[ry1:ry2, rx1:rx2].copy()
                            mask_source = "sam"
                            mask_pixels = np.count_nonzero(region_mask_det)
                            logging.info(
                                f"Track {logo.track_id}: SAM mask generated - pixels={mask_pixels}"
                            )
                        else:
                            logging.debug(
                                f"Track {logo.track_id}: SAM returned empty mask, using fallback"
                            )
                    except Exception as e:
                        logging.warning(
                            f"Track {logo.track_id}: SAM mask failed ({e}), using fallback"
                        )
                        region_mask_det = None
                
                # Fallback to region mask from pipeline if SAM failed
                if region_mask_det is None:
                    if best_region.mask is not None:
                        region_mask_det = best_region.mask.copy()
                        mask_source = "pipeline"
                        # Ensure mask matches patch size
                        if region_mask_det.shape[:2] != (patch_h, patch_w):
                            region_mask_det = cv2.resize(
                                region_mask_det, (patch_w, patch_h), 
                                interpolation=cv2.INTER_NEAREST
                            )
                    else:
                        # Last resort: full rectangle mask
                        region_mask_det = np.ones((patch_h, patch_w), dtype=np.uint8) * 255
                        mask_source = "bbox"
                        logging.debug(
                            f"Track {logo.track_id}: Using full bbox as mask (no SAM, no region mask)"
                        )
                
                # Ensure mask is binary and properly formatted
                if region_mask_det.max() > 1:
                    _, region_mask_det = cv2.threshold(region_mask_det, 127, 255, cv2.THRESH_BINARY)
                
                # Get inpainted background if available
                if hasattr(result, 'inpainted_background') and result.inpainted_background is not None:
                    bg_patch_det = result.inpainted_background[ry1:ry2, rx1:rx2].copy()
                else:
                    bg_patch_det = None
                
                # FLIP THE ENTIRE PATCH (logo + background) using the proper symmetry line
                # This ensures consistent flip direction based on the logo's orientation
                symmetry_line = estimate_symmetry_line(region_mask_det)
                
                if symmetry_line is not None:
                    # Flip the original patch using the symmetry line
                    flipped_original, flipped_mask = reflect_along_line(
                        original_patch_det, region_mask_det, symmetry_line
                    )
                    
                    # Also flip the background patch if available
                    if bg_patch_det is not None:
                        flipped_bg, _ = reflect_along_line(
                            bg_patch_det, region_mask_det, symmetry_line
                        )
                    else:
                        flipped_bg = None
                    
                    logging.debug(
                        f"Track {logo.track_id}: Flipped patch using symmetry line "
                        f"(angle={symmetry_line.angle_degrees:.1f}°)"
                    )
                else:
                    # Fallback to simple horizontal flip if no symmetry line found
                    flipped_original = cv2.flip(original_patch_det, 1)
                    flipped_mask = cv2.flip(region_mask_det, 1)
                    flipped_bg = cv2.flip(bg_patch_det, 1) if bg_patch_det is not None else None
                    logging.debug(
                        f"Track {logo.track_id}: No symmetry line, using simple horizontal flip"
                    )
                
                # Convert to frame_test space if input is flipped
                if pipeline_config.input_is_flipped:
                    flipped_patch = cv2.flip(flipped_original, 1)
                    original_patch = cv2.flip(original_patch_det, 1)
                    region_mask = cv2.flip(flipped_mask, 1)
                    bg_patch = cv2.flip(flipped_bg, 1) if flipped_bg is not None else None
                else:
                    flipped_patch = flipped_original
                    original_patch = original_patch_det
                    region_mask = flipped_mask
                    bg_patch = flipped_bg
                
                # Store cache with all metadata - always update with latest detection
                # flipped_patch: The properly flipped patch (using symmetry line)
                # original_patch: The unflipped original (for reference)
                # region_mask: Matches the flipped_patch
                logo.last_unoccluded_flipped_patch = flipped_patch
                logo.last_unoccluded_patch = original_patch
                logo.last_unoccluded_mask = region_mask
                logo.last_unoccluded_bbox = (x, y, w, h)  # Tracked bbox for reference
                logo.last_unoccluded_inpainted_bg = bg_patch
                logo.cache_frame_idx = frame_idx
                logo.cache_original_size = (patch_h, patch_w)
                logo.cache_mask_source = mask_source  # Track where the mask came from
                
                logging.info(
                    f"Track {logo.track_id}: Updated cache at frame {frame_idx} "
                    f"conf={best_region.detection_confidence:.2f}, occlusion={occlusion_ratio*100:.1f}%, "
                    f"size={patch_w}x{patch_h}, mask={mask_source}"
                )
            
            t_viz_start = time.time()
            # Pass hand mask to visualization for drawing
            viz_frame = tracker.draw_tracks(frame_test, hand_mask=full_frame_hand_mask)
            
            if tracker.pose_data_dir is not None:
                pose = tracker._load_pose(frame_idx - 1)
                viz_frame = tracker.draw_pose(viz_frame, pose, is_flipped=pipeline_config.input_is_flipped)
            
            timing_breakdown['visualization'] = time.time() - t_viz_start
            
            t_clean_start = time.time()
            
            # Clean frame generation
            if pipeline_config.input_is_flipped:
                clean_frame = cv2.flip(result.flipped_image, 1)
            else:
                clean_frame = result.flipped_image.copy()
            
            logging.debug(f"Frame {frame_idx}: Pipeline processed {len(result.regions)} regions")
            
            # Reset per-frame flags for all tracked logos
            for logo in tracked_logos:
                logo.was_pasted_from_cache = False
            
            # Step 2: Handle hand-occluded logos
            # IMPORTANT: Only paste cached logo if the logo was NOT detected by YOLO this frame.
            # If frames_since_detection == 0, the pipeline already flipped the logo in result.flipped_image,
            # so pasting again would cause "double pasting" artifacts.
            for logo in tracked_logos:
                if logo.state != "confirmed":
                    continue
                
                # Skip if logo was detected by YOLO this frame - pipeline already handled the flip
                if logo.frames_since_detection == 0:
                    continue
                
                if logo.is_hand_occluded:
                    # Use bbox stabilizer if available to reduce jitter
                    raw_bbox = logo.bbox
                    if logo.bbox_stabilizer is not None:
                        x, y, w, h = logo.bbox_stabilizer.stabilize(raw_bbox)
                        logging.debug(
                            f"Track {logo.track_id}: Stabilized bbox {raw_bbox} -> ({x},{y},{w},{h})"
                        )
                    else:
                        x, y, w, h = raw_bbox
                    
                    h_img, w_img = clean_frame.shape[:2]
                    
                    # Check if we have valid cached references FIRST
                    if logo.last_unoccluded_flipped_patch is None:
                        logging.warning(f"Track {logo.track_id}: No cached flipped patch")
                        continue
                    
                    if logo.frames_since_unoccluded > tracker.max_frames_hand_occluded:
                        logging.warning(
                            f"Track {logo.track_id}: Cache too old ({logo.frames_since_unoccluded} frames)"
                        )
                        continue
                    
                    # USE THE CACHED PATCH'S ORIGINAL SIZE - don't resize to tracked bbox!
                    # This prevents squeezing/stretching when tracker estimates different size
                    cached_h, cached_w = logo.last_unoccluded_flipped_patch.shape[:2]
                    
                    # Center the cached patch on the tracked bbox center
                    tracked_cx = x + w // 2
                    tracked_cy = y + h // 2
                    
                    # Calculate paste position using cached size, centered on tracked position
                    paste_x = tracked_cx - cached_w // 2
                    paste_y = tracked_cy - cached_h // 2
                    
                    # Clip to image bounds
                    x1 = max(0, paste_x)
                    y1 = max(0, paste_y)
                    x2 = min(w_img, paste_x + cached_w)
                    y2 = min(h_img, paste_y + cached_h)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Calculate the portion of cached patch to use (if clipped at edges)
                    src_x1 = x1 - paste_x  # How much was clipped on left
                    src_y1 = y1 - paste_y  # How much was clipped on top
                    src_x2 = src_x1 + (x2 - x1)
                    src_y2 = src_y1 + (y2 - y1)
                    
                    target_h, target_w = y2 - y1, x2 - x1
                    
                    logging.info(
                        f"Track {logo.track_id}: Hand occluded - "
                        f"has_patch={logo.last_unoccluded_flipped_patch is not None}, "
                        f"has_mask={logo.last_unoccluded_mask is not None}, "
                        f"has_bg={logo.last_unoccluded_inpainted_bg is not None}, "
                        f"mask_source={logo.cache_mask_source}, "
                        f"cached_frame={logo.cache_frame_idx}, "
                        f"cached_size={cached_w}x{cached_h}, "
                        f"paste_at=({x1},{y1}), "
                        f"frames_since={logo.frames_since_unoccluded}"
                    )
                    
                    # Use the PIPELINE-FLIPPED patch - it has the correct flip direction
                    # (based on symmetry line / perspective) already applied.
                    # The mask was also generated to match the flipped patch.
                    
                    if logo.last_unoccluded_flipped_patch is not None:
                        # Use the already-correctly-flipped patch from pipeline
                        # Crop to the portion that's visible (if clipped at image edges)
                        patch = logo.last_unoccluded_flipped_patch[src_y1:src_y2, src_x1:src_x2].copy()
                        
                        # Mask should match the flipped patch (no additional flip needed)
                        if logo.last_unoccluded_mask is not None:
                            logo_mask = logo.last_unoccluded_mask[src_y1:src_y2, src_x1:src_x2].copy()
                        else:
                            logo_mask = None
                        
                        logging.debug(
                            f"Track {logo.track_id}: Using PIPELINE-FLIPPED patch "
                            f"(correct flip direction, {cached_w}x{cached_h}, visible={target_w}x{target_h})"
                        )
                    elif logo.last_unoccluded_patch is not None:
                        # Fallback: simple horizontal flip of original
                        source_patch = logo.last_unoccluded_patch
                        cropped_patch = source_patch[src_y1:src_y2, src_x1:src_x2]
                        patch = cv2.flip(cropped_patch, 1)
                        
                        # Also flip the mask to match
                        if logo.last_unoccluded_mask is not None:
                            logo_mask = logo.last_unoccluded_mask[src_y1:src_y2, src_x1:src_x2].copy()
                            logo_mask = cv2.flip(logo_mask, 1)
                        else:
                            logo_mask = None
                        
                        logging.debug(
                            f"Track {logo.track_id}: Using ORIGINAL with simple h-flip (fallback) "
                            f"{cached_w}x{cached_h}, visible={target_w}x{target_h}"
                        )
                    else:
                        logging.warning(f"Track {logo.track_id}: No cached patch available")
                        continue
                    
                    # Create a "valid pixels" mask - exclude black corners from warpAffine
                    # Black pixels (0,0,0) at corners are artifacts from the reflection transform
                    if patch.ndim == 3:
                        # Check for near-black pixels (threshold to handle compression artifacts)
                        valid_pixels_mask = np.any(patch > 5, axis=2).astype(np.uint8) * 255
                    else:
                        valid_pixels_mask = (patch > 5).astype(np.uint8) * 255
                    
                    # Prepare mask for blending - use logo mask but also exclude invalid (black) pixels
                    if logo_mask is not None:
                        # Ensure mask is binary
                        if logo_mask.max() > 1:
                            _, logo_mask = cv2.threshold(logo_mask, 127, 255, cv2.THRESH_BINARY)
                        
                        # Combine with valid pixels mask to exclude black corners
                        logo_mask = cv2.bitwise_and(logo_mask, valid_pixels_mask)
                        
                        # Apply slight erosion to avoid edge artifacts
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        logo_mask = cv2.erode(logo_mask, kernel, iterations=1)
                        
                        # Smooth edges for better blending (feathering)
                        logo_mask = cv2.GaussianBlur(logo_mask, (5, 5), 0)
                    else:
                        # No logo mask - use valid pixels mask with feathered edges
                        logo_mask = valid_pixels_mask.copy()
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        logo_mask = cv2.erode(logo_mask, kernel, iterations=1)
                        logo_mask = cv2.GaussianBlur(logo_mask, (7, 7), 0)
                    
                    # First, put the INPAINTED BACKGROUND in the logo region
                    # This removes the original (unflipped) logo
                    if logo.last_unoccluded_inpainted_bg is not None:
                        bg_cropped = logo.last_unoccluded_inpainted_bg[src_y1:src_y2, src_x1:src_x2]
                        if bg_cropped.shape[:2] == (target_h, target_w):
                            # Create valid bg mask (exclude black corners from warpAffine)
                            if bg_cropped.ndim == 3:
                                valid_bg_mask = np.any(bg_cropped > 5, axis=2).astype(np.uint8) * 255
                            else:
                                valid_bg_mask = (bg_cropped > 5).astype(np.uint8) * 255
                            
                            # Blend inpainted background where it's valid
                            bg_alpha = valid_bg_mask.astype(np.float32) / 255.0
                            if len(bg_alpha.shape) == 2:
                                bg_alpha = bg_alpha[:, :, np.newaxis]
                            
                            current_pixels = clean_frame[y1:y2, x1:x2].astype(np.float32)
                            bg_blended = (bg_alpha * bg_cropped.astype(np.float32) + 
                                         (1.0 - bg_alpha) * current_pixels).astype(np.uint8)
                            clean_frame[y1:y2, x1:x2] = bg_blended
                            logging.debug(f"Track {logo.track_id}: Applied inpainted background")
                    
                    # Now paste the FLIPPED LOGO on top of the inpainted background
                    # Alpha blending
                    alpha = logo_mask.astype(np.float32) / 255.0
                    if len(alpha.shape) == 2:
                        alpha = alpha[:, :, np.newaxis]
                    
                    # Blend: result = alpha * flipped_logo + (1-alpha) * inpainted_background
                    current_bg = clean_frame[y1:y2, x1:x2].astype(np.float32)
                    patch_float = patch.astype(np.float32)
                    blended = (alpha * patch_float + (1.0 - alpha) * current_bg).astype(np.uint8)
                    clean_frame[y1:y2, x1:x2] = blended
                    
                    # Mark this logo as pasted from cache (for viz drawing)
                    logo.was_pasted_from_cache = True
                    
                    visible_pixels = int(np.sum(alpha) / 3) if len(alpha.shape) == 3 else int(np.sum(alpha))
                    logging.info(
                        f"Track {logo.track_id}: Pasted cached logo at ({x1},{y1})-({x2},{y2}) "
                        f"size={target_w}x{target_h}, visible={visible_pixels}px"
                    )
            
            # Restore original hand pixels
            if full_frame_hand_mask is not None and np.count_nonzero(full_frame_hand_mask) > 0:
                hand_bool = full_frame_hand_mask > 127
                # Restore hand pixels from frame_test (original pixels in test space)
                clean_frame[hand_bool] = frame_test[hand_bool]
                logging.debug(f"Frame {frame_idx}: Restored {np.count_nonzero(hand_bool)} hand pixels")
            
            # Generate inpainted frame - also work in frame_test space
            if hasattr(result, 'inpainted_background') and result.inpainted_background is not None:
                # Convert inpainted_background to frame_test space
                if pipeline_config.input_is_flipped:
                    inpainted_frame = cv2.flip(result.inpainted_background, 1)
                else:
                    inpainted_frame = result.inpainted_background.copy()
                
                # Restore hand pixels (already in frame_test space)
                if full_frame_hand_mask is not None and np.count_nonzero(full_frame_hand_mask) > 0:
                    hand_bool = full_frame_hand_mask > 127
                    inpainted_frame[hand_bool] = frame_test[hand_bool]
            else:
                # No inpainted_background from pipeline - likely YOLO didn't detect anything
                # But tracker may have valid tracked logos that need inpainting
                inpainted_frame = frame_test.copy()
                
                # Check if we have tracked logos that need inpainting
                tracked_regions_to_inpaint = []
                for logo in tracked_logos:
                    if logo.state != "confirmed":
                        continue
                    
                    # Only inpaint if YOLO didn't detect this logo (frames_since_detection > 0)
                    # AND we have a valid cached mask to use
                    if logo.frames_since_detection == 0:
                        continue  # YOLO detected it, pipeline already handled it
                    
                    # We need a mask to know what to inpaint
                    if logo.last_unoccluded_mask is None and logo.current_mask is None:
                        logging.debug(
                            f"Track {logo.track_id}: Skipping inpainting - no mask available"
                        )
                        continue
                    
                    x, y, w, h = logo.bbox
                    tracked_regions_to_inpaint.append({
                        'logo': logo,
                        'bbox': (x, y, w, h),
                        'mask': logo.last_unoccluded_mask if logo.last_unoccluded_mask is not None else logo.current_mask
                    })
                
                if tracked_regions_to_inpaint:
                    logging.info(
                        f"Frame {frame_idx}: YOLO detected nothing, but have {len(tracked_regions_to_inpaint)} "
                        f"tracked logo(s) to inpaint"
                    )
                    
                    # Create aggregated inpaint mask for all tracked regions
                    h_img, w_img = inpainted_frame.shape[:2]
                    aggregated_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    
                    for region_info in tracked_regions_to_inpaint:
                        logo = region_info['logo']
                        x, y, w, h = region_info['bbox']
                        mask = region_info['mask']
                        
                        # Clamp bbox to image bounds
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(w_img, x + w), min(h_img, y + h)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        target_h, target_w = y2 - y1, x2 - x1
                        
                        # Resize mask to current bbox size
                        if mask.shape[:2] != (target_h, target_w):
                            resized_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        else:
                            resized_mask = mask.copy()
                        
                        # Ensure mask is binary
                        if resized_mask.max() > 1:
                            _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
                        
                        # Add margin to mask for better inpainting
                        margin = max(0, pipeline_config.inpaint_mask_margin)
                        if margin > 0:
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin*2+1, margin*2+1))
                            resized_mask = cv2.dilate(resized_mask, kernel, iterations=1)
                        
                        # Place in aggregated mask
                        aggregated_mask[y1:y2, x1:x2] = np.maximum(
                            aggregated_mask[y1:y2, x1:x2], resized_mask
                        )
                        
                        logging.debug(
                            f"Track {logo.track_id}: Added to inpaint mask at ({x1},{y1})-({x2},{y2}), "
                            f"mask_pixels={np.count_nonzero(resized_mask)}"
                        )
                    
                    # Expand mask for inpainting (like pipeline does)
                    expansion = max(0, pipeline_config.inpaint_expansion)
                    if expansion > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion*2+1, expansion*2+1))
                        working_mask = cv2.dilate(aggregated_mask, kernel, iterations=1)
                    else:
                        working_mask = aggregated_mask
                    
                    # Add hand pixels near logo to mask (like pipeline does for two-pass)
                    if full_frame_hand_mask is not None and np.count_nonzero(full_frame_hand_mask) > 0:
                        buffer = max(0, pipeline_config.hand_exclusion_buffer)
                        if buffer > 0:
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer*2+1, buffer*2+1))
                            hand_exclusion = cv2.dilate(full_frame_hand_mask, kernel, iterations=1)
                        else:
                            hand_exclusion = full_frame_hand_mask.copy()
                        
                        # Expand logo mask to find nearby hands
                        logo_expanded = cv2.dilate(
                            working_mask,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                (pipeline_config.inpaint_radius*2 + buffer*2 + 1,
                                 pipeline_config.inpaint_radius*2 + buffer*2 + 1)),
                            iterations=1
                        )
                        
                        hand_near_logo = cv2.bitwise_and(hand_exclusion, logo_expanded)
                        hand_pixels_near = np.count_nonzero(hand_near_logo)
                        
                        if hand_pixels_near > 0:
                            logging.debug(
                                f"Frame {frame_idx}: Adding {hand_pixels_near} hand pixels to inpaint mask"
                            )
                            # Save original hand region
                            original_hand_pixels = inpainted_frame.copy()
                            # Add hand pixels to inpaint mask
                            working_mask = cv2.bitwise_or(working_mask, hand_near_logo)
                    
                    # Perform inpainting
                    mask_pixels = np.count_nonzero(working_mask)
                    if mask_pixels > 0:
                        if pipeline._lama_inpainter is not None:
                            # Use LaMa ML inpainting
                            try:
                                from PIL import Image as PILImage
                                source_rgb = cv2.cvtColor(inpainted_frame, cv2.COLOR_BGR2RGB)
                                source_pil = PILImage.fromarray(source_rgb)
                                mask_pil = PILImage.fromarray(working_mask)
                                
                                result_pil = pipeline._lama_inpainter(source_pil, mask_pil)
                                result_rgb = np.array(result_pil)
                                inpainted_frame = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                                
                                logging.info(
                                    f"Frame {frame_idx}: LaMa inpainted {mask_pixels} pixels for tracked logos"
                                )
                            except Exception as e:
                                logging.warning(f"LaMa inpainting failed: {e}, using cv2.inpaint")
                                radius = max(1, pipeline_config.inpaint_radius)
                                inpainted_frame = cv2.inpaint(inpainted_frame, working_mask, radius, cv2.INPAINT_TELEA)
                        else:
                            # Use cv2.inpaint
                            radius = max(1, pipeline_config.inpaint_radius)
                            inpainted_frame = cv2.inpaint(inpainted_frame, working_mask, radius, cv2.INPAINT_TELEA)
                            logging.info(
                                f"Frame {frame_idx}: cv2.inpaint inpainted {mask_pixels} pixels for tracked logos"
                            )
                        
                        # Restore original hand pixels after inpainting
                        if full_frame_hand_mask is not None and np.count_nonzero(full_frame_hand_mask) > 0:
                            hand_bool = full_frame_hand_mask > 127
                            inpainted_frame[hand_bool] = frame_test[hand_bool]
                    else:
                        logging.debug(f"Frame {frame_idx}: No pixels to inpaint")
                else:
                    logging.debug(f"Frame {frame_idx}: No inpainted_background and no tracked logos to inpaint")
            
            timing_breakdown['clean_processing'] = time.time() - t_clean_start
            
            # Write outputs
            t_write_start = time.time()
            out_viz.write(viz_frame)
            out_clean.write(clean_frame)
            out_inpainted.write(inpainted_frame)
            timing_breakdown['write'] = time.time() - t_write_start
            
            # Timing statistics with detailed breakdown
            elapsed = time.time() - start_time
            frame_times.append(elapsed)
            
            # Print detailed timing every 10 frames or when frame takes >1s
            if frame_idx % 10 == 0 or elapsed > 1.0:
                breakdown_str = " | ".join([f"{k}:{v*1000:.0f}ms" for k, v in timing_breakdown.items()])
                print(f"[TIMING] Frame {frame_idx}: {elapsed:.3f}s | dets={num_detections} | {breakdown_str}")
            else:
                print(f"[TIMING] Frame {frame_idx}: {elapsed:.3f}s | dets={num_detections}")
            
            if frame_idx % 30 == 0:
                mean_time = np.mean(frame_times[-30:])
                std_time = np.std(frame_times[-30:])
                logging.info(f"Progress: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%) | Avg: {mean_time:.3f}s ± {std_time:.3f}s")
    finally:
        cap.release()
        out_viz.release()
        out_clean.release()
        out_inpainted.release()
    
    # Print timing statistics
    if frame_times:
        mean_time = np.mean(frame_times)
        std_time = np.std(frame_times)
        min_time = np.min(frame_times)
        max_time = np.max(frame_times)
        total_time = np.sum(frame_times)
        
        logging.info("="*60)
        logging.info("Processing complete!")
        logging.info(f"Total frames processed: {len(frame_times)}")
        logging.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
        logging.info(f"Mean frame time: {mean_time:.3f}s ± {std_time:.3f}s")
        logging.info(f"Min/Max frame time: {min_time:.3f}s / {max_time:.3f}s")
        logging.info(f"Estimated FPS: {1/mean_time:.2f}")
        logging.info("="*60)
    
    logging.info(f"Visualization saved to: {output_viz_path}")
    logging.info(f"Clean output saved to: {output_clean_path}")
    logging.info(f"Inpainted output saved to: {output_inpainted_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Track and flip logos in video using optical flow"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--output-viz",
        type=Path,
        required=True,
        help="Output path for visualization video (with bboxes and traces)"
    )
    parser.add_argument(
        "--output-clean",
        type=Path,
        required=True,
        help="Output path for clean video (flipped logos only)"
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=PROJECT_ROOT / "runs" / "train" / "openlogo_training3" / "weights" / "best.pt",
        help="YOLOv7 weights path"
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.5,
        help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device for inference (0 for GPU, cpu for CPU)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Disable EasyOCR text detection completely"
    )
    parser.add_argument(
        "--grabcut-iters",
        type=int,
        default=5,
        help="GrabCut iterations for mask refinement (default: 5, try 7-15 for better quality)"
    )
    parser.add_argument(
        "--min-mask-quality",
        type=float,
        default=0.05,
        help="Minimum mask coverage ratio for quality validation (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--use-sam-masker",
        action="store_true",
        help="Use MobileSAM for refined masks before falling back to GrabCut.",
    )
    parser.add_argument(
        "--sam-model",
        type=Path,
        help="Path to MobileSAM weights (default: mobile_sam.pt).",
    )
    parser.add_argument(
        "--sam-device",
        default=None,
        help="Device identifier for SAM (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--sam-imgsz",
        type=int,
        default=512,
        help="MobileSAM predictor resolution (default: 512).",
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
    parser.add_argument(
        "--input-is-flipped",
        action="store_true",
        help="Input video is already flipped (testing mode: will flip input frame to simulate flipped video).",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        help="Directory containing bodypose JSON files (enables pose-enhanced tracking).",
    )
    parser.add_argument(
        "--no-pose-prediction",
        action="store_false",
        dest="use_pose_prediction",
        default=True,
        help="Disable pose-based motion prediction (only use optical flow).",
    )
    parser.add_argument(
        "--disable-clip",
        action="store_true",
        help="Disable CLIP-based appearance embeddings for tracking.",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B/32",
        help="CLIP model name for logo embeddings.",
    )
    parser.add_argument(
        "--clip-device",
        default=None,
        help="Device for CLIP (e.g., cuda, cuda:1, or cpu).",
    )
    parser.add_argument(
        "--disable-optical-flow",
        action="store_true",
        help="Disable optical flow tracking (only use detection-based tracking).",
    )
    parser.add_argument(
        "--use-fine-grid-tracking",
        action="store_true",
        help="Enable fine-grained grid-based tracking for occlusion detection.",
    )
    parser.add_argument(
        "--grid-cell-size",
        type=int,
        default=10,
        help="Grid cell size in pixels for fine-grained tracking (default: 10).",
    )
    parser.add_argument(
        "--sam-per-frame",
        action="store_true",
        help="Run SAM every frame for maximum mask quality (slower but most precise).",
    )
    parser.add_argument(
        "--inpaint-radius",
        type=int,
        default=7,
        help="Inpainting radius in pixels (default: 7, try 10-15 for better edge coverage).",
    )
    parser.add_argument(
        "--inpaint-expansion",
        type=int,
        default=15,
        help="Inpainting mask expansion in pixels (default: 15, higher = more context for edges).",
    )
    parser.add_argument(
        "--inpaint-mask-margin",
        type=int,
        default=0,
        help="Expand logo mask by N pixels before inpainting (default: 0, try 5-15 for larger removal area).",
    )
    parser.add_argument(
        "--disable-hand-detection",
        action="store_true",
        help="Disable MediaPipe hand detection for occlusion handling.",
    )
    parser.add_argument(
        "--hand-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for MediaPipe hand detection (default: 0.5).",
    )
    parser.add_argument(
        "--hand-exclusion-buffer",
        type=int,
        default=15,
        help="Pixels to expand hand mask for exclusion zone in inpainting (default: 15). "
             "Prevents sampling hand-colored pixels during logo inpainting.",
    )
    parser.add_argument(
        "--disable-two-pass-inpainting",
        action="store_true",
        help="Disable two-pass inpainting (neutralize hands first, then inpaint logo). "
             "Use if you want faster processing at the cost of potential hand color bleeding.",
    )
    parser.add_argument(
        "--no-inpaint-under-hand",
        action="store_true",
        help="Don't inpaint logo area under hand (legacy behavior). "
             "Default is to fully inpaint logo even under hand, then restore hand pixels.",
    )
    parser.add_argument(
        "--use-lama-inpainting",
        action="store_true",
        help="Use LaMa ML-based inpainting instead of cv2.inpaint. "
             "Better for textures like fabric. Requires: pip install simple-lama-inpainting",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(levelname)s] %(message)s",
    )

    # Create detection settings (matching best params from user)
    detection_settings = DetectionSettings(
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        yolo_iou=0.45,
        yolo_img_size=1280,
        yolo_device=args.device,
        use_easyocr=not args.disable_ocr,
        easyocr_langs=("en",),
        easyocr_conf=0.1,
        easyocr_min_size=1280,
        easyocr_max_size=1920,
        merge_gap=12,
    )

    # Create pipeline config (matching best params from user)
    pipeline_config = PipelineConfig(
        mask_padding=12,
        grabcut_iters=args.grabcut_iters,
        min_mask_quality=args.min_mask_quality,
        inpaint_radius=args.inpaint_radius,
        inpaint_expansion=args.inpaint_expansion,
        inpaint_mask_margin=args.inpaint_mask_margin,
        naturalise_mode="line",
        blend_mode="alpha",
        use_perspective_flip=args.use_perspective_flip,
        input_is_flipped=args.input_is_flipped,
        artifact_dir=None,  # No artifacts for video processing
        save_intermediates=False,
        bbox_json=None,
        detection=detection_settings,
        use_sam_masker=args.use_sam_masker,
        sam_model_path=args.sam_model,
        sam_device=args.sam_device,
        sam_image_size=args.sam_imgsz,
        # Hand-aware inpainting settings
        hand_exclusion_buffer=args.hand_exclusion_buffer,
        use_two_pass_inpainting=not args.disable_two_pass_inpainting,
        inpaint_full_logo_under_hand=not args.no_inpaint_under_hand,
        # ML-based inpainting
        use_lama_inpainting=args.use_lama_inpainting,
    )

    # Process video
    logging.info(f"Fine-grained grid tracking: {'enabled' if args.use_fine_grid_tracking else 'disabled'}")
    if args.use_fine_grid_tracking:
        logging.info(f"Grid cell size: {args.grid_cell_size}×{args.grid_cell_size} pixels")
    
    use_hand_detection = not args.disable_hand_detection
    logging.info(f"MediaPipe hand detection: {'enabled' if use_hand_detection else 'disabled'}")
    if use_hand_detection:
        logging.info(f"Hand detection confidence: {args.hand_detection_confidence}")
        logging.info(f"Hand exclusion buffer: {args.hand_exclusion_buffer}px")
        logging.info(f"Two-pass inpainting: {'enabled' if not args.disable_two_pass_inpainting else 'disabled'}")
        logging.info(f"Inpaint full logo under hand: {'yes' if not args.no_inpaint_under_hand else 'no'}")
    
    if args.sam_per_frame:
        logging.warning(
            "SAM per-frame mode enabled - this will be SLOW but provides maximum precision. "
            "Consider using default smart triggering mode for faster processing."
        )
    
    if args.use_lama_inpainting:
        logging.info("LaMa ML-based inpainting: ENABLED (better for fabric textures)")
    else:
        logging.info("LaMa ML-based inpainting: disabled (using cv2.inpaint)")
    
    process_video(
        input_path=args.input,
        output_viz_path=args.output_viz,
        output_clean_path=args.output_clean,
        output_inpainted_path=None,  # Auto-generated as inp.mp4
        detection_settings=detection_settings,
        pipeline_config=pipeline_config,
        pose_data_dir=args.pose_dir,
        use_pose_prediction=args.use_pose_prediction,
        use_clip_embeddings=not args.disable_clip,
        clip_model=args.clip_model,
        clip_device=args.clip_device,
        use_optical_flow=not args.disable_optical_flow,
        use_fine_grid_tracking=args.use_fine_grid_tracking,
        grid_cell_size=args.grid_cell_size,
        sam_per_frame=args.sam_per_frame,
        use_hand_detection=use_hand_detection,
        hand_detection_confidence=args.hand_detection_confidence,
    )


if __name__ == "__main__":
    main()
