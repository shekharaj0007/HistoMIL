"""Build Camelyon-style bags: slide-level tumor/normal, no pixel labels for training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from histomil.tiling import tissue_tile_coords


def _noisy_ellipse(h: int, w: int, cx: float, cy: float, rx: float, ry: float, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    field = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2
    noise = cv2.GaussianBlur(rng.normal(0, 0.18, (h, w)).astype(np.float32), (0, 0), 12)
    return (field + noise) < 1.0


def _paint_nuclei(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, n: int, radius: tuple[int, int], dark: bool) -> None:
    ys, xs = np.where(mask)
    if ys.size == 0 or n <= 0:
        return
    pick = rng.integers(0, ys.size, size=n)
    color = (55, 20, 70) if dark else (120, 55, 130)
    for i in pick:
        r = int(rng.integers(radius[0], radius[1] + 1))
        cv2.circle(img, (int(xs[i]), int(ys[i])), r, color, -1, lineType=cv2.LINE_AA)


def synthesize_slide(
    size: int,
    rng: np.random.Generator,
    tumor: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """H&E-like slide: glass background, tissue, optional localized metastasis."""
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    img[..., 0] = 248
    img[..., 1] = 246
    img[..., 2] = 247

    cx, cy = size * rng.uniform(0.42, 0.58), size * rng.uniform(0.42, 0.58)
    rx, ry = size * rng.uniform(0.32, 0.42), size * rng.uniform(0.30, 0.40)
    tissue = _noisy_ellipse(size, size, cx, cy, rx, ry, rng)

    he = img.copy()
    he[tissue] = (
        np.array(
            [
                rng.integers(210, 235),
                rng.integers(155, 190),
                rng.integers(185, 215),
            ],
            dtype=np.uint8,
        )
    )
    grain = rng.integers(-12, 13, size=he.shape, dtype=np.int16)
    he = np.clip(he.astype(np.int16) + grain * tissue[..., None], 0, 255).astype(np.uint8)
    _paint_nuclei(he, tissue, rng, n=int(size * size * 0.00035), radius=(2, 4), dark=False)

    tumor_mask = np.zeros((size, size), dtype=np.uint8)
    if tumor:
        tx = cx + rng.uniform(-0.12, 0.12) * size
        ty = cy + rng.uniform(-0.12, 0.12) * size
        tumor_mask = _noisy_ellipse(
            size,
            size,
            tx,
            ty,
            size * rng.uniform(0.10, 0.16),
            size * rng.uniform(0.10, 0.16),
            rng,
        )
        tumor_mask = np.logical_and(tumor_mask, tissue)
        he[tumor_mask] = (
            np.array(
                [
                    rng.integers(150, 185),
                    rng.integers(70, 110),
                    rng.integers(130, 170),
                ],
                dtype=np.uint8,
            )
        )
        _paint_nuclei(he, tumor_mask, rng, n=int(tumor_mask.sum() * 0.012), radius=(3, 6), dark=True)

    he[~tissue] = img[~tissue]
    return he, (tumor_mask.astype(np.uint8) * 255)


def _write_split(records: list[dict], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--n-slides", type=int, default=48)
    p.add_argument("--size", type=int, default=1536)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    slide_dir = args.out / "slides"
    mask_dir = args.out / "masks"
    slide_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    n_tumor = args.n_slides // 2
    labels = [1] * n_tumor + [0] * (args.n_slides - n_tumor)
    py_rng.shuffle(labels)

    records: list[dict] = []
    for i, lab in enumerate(labels):
        rgb, mask = synthesize_slide(args.size, rng, tumor=bool(lab))
        sid = f"slide_{i:03d}"
        img_path = slide_dir / f"{sid}.png"
        mask_path = mask_dir / f"{sid}.png"
        Image.fromarray(rgb).save(img_path)
        Image.fromarray(mask).save(mask_path)
        tiles = tissue_tile_coords(rgb, tile=args.tile)
        records.append(
            {
                "id": sid,
                "image": str(img_path).replace("\\", "/"),
                "mask": str(mask_path).replace("\\", "/"),
                "label": int(lab),
                "tiles": tiles,
                "n_tiles": len(tiles),
            }
        )
        print(f"{sid} label={lab} tiles={len(tiles)}")

    py_rng.shuffle(records)
    n = len(records)
    n_test = max(6, n // 5)
    n_val = max(6, n // 6)
    test, rest = records[:n_test], records[n_test:]
    val, train = rest[:n_val], rest[n_val:]
    _write_split(train, args.out / "train")
    _write_split(val, args.out / "val")
    _write_split(test, args.out / "test")
    print(f"wrote train={len(train)} val={len(val)} test={len(test)} under {args.out}")


if __name__ == "__main__":
    main()
