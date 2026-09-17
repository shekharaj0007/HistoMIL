"""Tissue-only 256×256 tiling for gigapixel H&E slides."""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def is_tissue(tile_rgb: np.ndarray, sat_thresh: float = 20.0) -> bool:
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean()) > sat_thresh


def extract_tissue_tiles(
    image: np.ndarray,
    tile: int = 256,
    stride: int = 256,
) -> Iterator[tuple[np.ndarray, int, int]]:
    h, w = image.shape[:2]
    for y in range(0, max(h - tile, 0) + 1, stride):
        for x in range(0, max(w - tile, 0) + 1, stride):
            patch = image[y : y + tile, x : x + tile]
            if patch.shape[0] == tile and patch.shape[1] == tile and is_tissue(patch):
                yield patch, y, x
