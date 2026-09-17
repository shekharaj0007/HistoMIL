"""Attention heatmap overlays for tumor localization."""

from __future__ import annotations

import cv2
import numpy as np


def overlay_heatmap(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h = heat.astype(np.float32)
    if h.max() > 0:
        h = h / h.max()
    h = cv2.GaussianBlur(h, (0, 0), sigmaX=24)
    if h.max() > 0:
        h = h / h.max()
    heat_u8 = np.clip(h * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    vis = (rgb.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha)
    return np.clip(vis, 0, 255).astype(np.uint8)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    return canvas
