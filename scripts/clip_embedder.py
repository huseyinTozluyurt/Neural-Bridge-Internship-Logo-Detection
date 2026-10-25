"""
Lightweight CLIP embedding wrapper used by the tracker for logo ReID.

This module hides the optional dependency on OpenAI's CLIP so that the rest of
the codebase can import `CLIPLogoEmbedder` without immediately pulling in heavy
packages.  If CLIP/torch are not available we simply disable appearance
matching and log a warning upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import logging

import numpy as np

try:  # pragma: no cover - requires optional dependency
    import torch
    import clip  # type: ignore
    from PIL import Image
    import cv2
except Exception:  # pragma: no cover - handled gracefully at runtime
    torch = None
    clip = None
    Image = None
    cv2 = None


BBox = Tuple[int, int, int, int]


@dataclass
class CLIPConfig:
    """Runtime options for CLIP-based embeddings."""

    model_name: str = "ViT-B/32"
    device: Optional[str] = None
    use_half_precision: bool = False


class CLIPLogoEmbedder:
    """
    Wraps OpenAI CLIP for extracting normalized embeddings from BGR frames.

    The class is intentionally tiny so that we can cache the model in the
    tracker and reuse it for every frame.
    """

    def __init__(self, config: Optional[CLIPConfig] = None) -> None:
        if clip is None or torch is None or Image is None or cv2 is None:
            raise ImportError(
                "CLIP dependencies (torch + clip + Pillow + OpenCV) are required "
                "for CLIPLogoEmbedder."
            )

        self.config = config or CLIPConfig()
        device = self.config.device
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, self.preprocess = clip.load(
            self.config.model_name, device=self.device, jit=False
        )
        self.model.eval()
        if (
            self.config.use_half_precision
            and self.device.startswith("cuda")
            and hasattr(self.model, "half")
        ):
            self.model = self.model.half()
        else:
            self.model = self.model.float()

    @torch.inference_mode()  # type: ignore[misc]
    def extract(self, frame_bgr: np.ndarray, bboxes: Sequence[BBox]) -> np.ndarray:
        """
        Compute ℓ2-normalized embeddings for each bbox in the provided frame.

        Returns:
            np.ndarray with shape (N, D). Empty detections return zeros.
        """
        if not bboxes:
            return np.zeros((0, 512), dtype=np.float32)

        patches: List[Optional[torch.Tensor]] = []
        valid_indices: List[int] = []

        for idx, (x, y, w, h) in enumerate(bboxes):
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = max(x1 + 1, int(x + w))
            y2 = max(y1 + 1, int(y + h))
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                patches.append(None)
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = self.preprocess(pil)
            patches.append(tensor)
            valid_indices.append(idx)

        if not valid_indices:
            return np.zeros((len(bboxes), 512), dtype=np.float32)

        batch = torch.stack([patches[i] for i in valid_indices]).to(self.device)
        if self.config.use_half_precision and self.device.startswith("cuda"):
            batch = batch.half()
        else:
            batch = batch.float()

        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.float().cpu().numpy()

        out = np.zeros((len(bboxes), feats.shape[1]), dtype=np.float32)
        for row, idx in enumerate(valid_indices):
            out[idx] = feats[row]
        return out


def create_clip_embedder(config: Optional[CLIPConfig] = None) -> Optional[CLIPLogoEmbedder]:
    """
    Helper that tries to construct an embedder and logs a warning if CLIP
    dependencies are missing.
    """
    try:
        return CLIPLogoEmbedder(config)
    except ImportError as exc:  # pragma: no cover - environment specific
        logging.warning("CLIP embeddings disabled: %s", exc)
        return None
