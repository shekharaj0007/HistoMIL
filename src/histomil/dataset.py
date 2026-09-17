"""Bags of 256×256 tissue tiles with slide-level labels only."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from histomil.tiling import tissue_tile_coords


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_index(split_dir: Path) -> list[dict]:
    return json.loads((split_dir / "index.json").read_text(encoding="utf-8"))


def _read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def normalize_tile(rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def augment_tile(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    img = rgb
    if rng.random() < 0.5:
        img = np.ascontiguousarray(np.fliplr(img))
    if rng.random() < 0.5:
        img = np.ascontiguousarray(np.flipud(img))
    k = rng.randint(0, 3)
    if k:
        img = np.ascontiguousarray(np.rot90(img, k))
    if rng.random() < 0.7:
        img = img.astype(np.float32)
        img *= rng.uniform(0.88, 1.12)
        img += rng.uniform(-10, 10)
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class SlideBagDataset(Dataset):
    """One bag = one slide. Pixel / instance labels are never used."""

    def __init__(
        self,
        records: list[dict],
        tile: int = 256,
        max_tiles: int = 32,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.tile = tile
        self.max_tiles = max_tiles
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        rgb = _read_rgb(rec["image"])
        coords = rec.get("tiles")
        if coords is None:
            coords = tissue_tile_coords(rgb, tile=self.tile)
        else:
            coords = [tuple(c) for c in coords]
        if not coords:
            raise RuntimeError(f"no tissue tiles in {rec['id']}")
        if len(coords) > self.max_tiles:
            coords = self.rng.sample(coords, self.max_tiles)
        patches = []
        for y, x in coords:
            patch = rgb[y : y + self.tile, x : x + self.tile]
            if self.augment:
                patch = augment_tile(patch, self.rng)
            patches.append(normalize_tile(patch))
        bag = torch.stack(patches, dim=0)
        label = torch.tensor(float(rec["label"]))
        return {
            "bag": bag,
            "label": label,
            "id": rec["id"],
            "coords": coords,
            "image": rec["image"],
            "mask": rec.get("mask"),
        }
