"""Classify any H&E image: tissue tiles → slide logit + attention heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from histomil.evaluate import classify_slide, load_model
from histomil.tiling import attention_heatmap, tissue_tile_coords
from histomil.viz import overlay_heatmap, side_by_side


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=Path("outputs/histomil.pt"))
    p.add_argument("--out", type=Path, default=Path("outputs/infer"))
    p.add_argument("--encoder", choices=["compact", "resnet18"], default="compact")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--tile", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location=device)
    model = load_model(args.ckpt, device, args.encoder, args.dim)
    thresh = float(blob.get("threshold", 0.5))
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    coords = tissue_tile_coords(rgb, tile=args.tile)
    print(f"input {rgb.shape[1]}×{rgb.shape[0]}  tissue_tiles={len(coords)}")
    if not coords:
        raise SystemExit("no tissue tiles (HSV saturation filter dropped everything)")
    prob, attn = classify_slide(model, rgb, coords, device, tile=args.tile)
    heat = attention_heatmap(coords, attn, rgb.shape[0], rgb.shape[1], tile=args.tile)
    overlay = overlay_heatmap(rgb, heat)
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.out / "heatmap.png")
    Image.fromarray(side_by_side(rgb, overlay)).save(args.out / "panel.png")
    label = "tumor" if prob >= thresh else "normal"
    print({"prob_tumor": round(prob, 4), "label": label, "threshold": round(thresh, 3), "n_tiles": len(coords)})


if __name__ == "__main__":
    main()
