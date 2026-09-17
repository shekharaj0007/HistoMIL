"""Tissue-only 256×256 tiling for gigapixel H&E slides."""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def tile_coords(h: int, w: int, tile: int = 256, stride: int = 256) -> list[tuple[int, int]]:
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys or ys[-1] != max(h - tile, 0):
        ys.append(max(h - tile, 0))
    if not xs or xs[-1] != max(w - tile, 0):
        xs.append(max(w - tile, 0))
    return [(y, x) for y in ys for x in xs]


def is_tissue(tile_rgb: np.ndarray, sat_thresh: float = 20.0) -> bool:
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean()) > sat_thresh


def extract_tissue_tiles(
    image: np.ndarray,
    tile: int = 256,
    stride: int = 256,
    sat_thresh: float = 20.0,
) -> Iterator[tuple[np.ndarray, int, int]]:
    h, w = image.shape[:2]
    for y, x in tile_coords(h, w, tile, stride):
        patch = image[y : y + tile, x : x + tile]
        if patch.shape[0] == tile and patch.shape[1] == tile and is_tissue(patch, sat_thresh):
            yield patch, y, x


def tissue_tile_coords(
    image: np.ndarray,
    tile: int = 256,
    stride: int = 256,
    sat_thresh: float = 20.0,
) -> list[tuple[int, int]]:
    return [(y, x) for _, y, x in extract_tissue_tiles(image, tile, stride, sat_thresh)]


def attention_heatmap(
    coords: list[tuple[int, int]],
    weights: np.ndarray,
    height: int,
    width: int,
    tile: int = 256,
) -> np.ndarray:
    """Place bag attention weights back onto the slide grid."""
    acc = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for (y, x), w in zip(coords, weights):
        acc[y : y + tile, x : x + tile] += float(w)
        count[y : y + tile, x : x + tile] += 1.0
    return acc / np.maximum(count, 1e-6)
