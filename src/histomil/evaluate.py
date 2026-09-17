"""Held-out slide AUC, ROC, and attention-heatmap localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from histomil.dataset import SlideBagDataset, load_index, normalize_tile
from histomil.metrics import accuracy, attention_in_tumor, roc_auc, roc_curve
from histomil.model import AttentionMIL
from histomil.tiling import attention_heatmap, is_tissue
from histomil.viz import overlay_heatmap, side_by_side


def load_model(ckpt: Path, device: torch.device, encoder: str, dim: int) -> AttentionMIL:
    blob = torch.load(ckpt, map_location=device)
    saved = blob.get("args", {})
    encoder = saved.get("encoder", encoder)
    dim = int(saved.get("dim", dim))
    model = AttentionMIL(dim=dim, encoder=encoder).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def classify_slide(
    model: AttentionMIL,
    rgb: np.ndarray,
    coords: list[tuple[int, int]],
    device: torch.device,
    tile: int = 256,
) -> tuple[float, np.ndarray]:
    if not coords:
        return 0.0, np.zeros((0,), dtype=np.float32)
    patches = [normalize_tile(rgb[y : y + tile, x : x + tile]) for y, x in coords]
    bag = torch.stack(patches, dim=0).to(device)
    logit, attn = model(bag)
    return float(torch.sigmoid(logit).item()), attn.detach().cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--ckpt", type=Path, default=Path("outputs/histomil.pt"))
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--encoder", choices=["compact", "resnet18"], default="compact")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--max-tiles", type=int, default=64)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location=device)
    model = load_model(args.ckpt, device, args.encoder, args.dim)
    thresh = float(blob.get("threshold", 0.5))
    records = load_index(args.data / args.split)
    loader = DataLoader(
        SlideBagDataset(records, max_tiles=args.max_tiles, augment=False),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    viz_dir = args.out / "overlays"
    viz_dir.mkdir(parents=True, exist_ok=True)
    assets = Path("assets")
    assets.mkdir(exist_ok=True)

    labels, probs, loc_scores = [], [], []
    for rec, batch in zip(records, tqdm(loader, desc="evaluate")):
        rgb = np.asarray(Image.open(rec["image"]).convert("RGB"))
        coords = [tuple(c) for c in rec["tiles"]]
        if len(coords) > args.max_tiles:
            coords = coords[: args.max_tiles]
        prob, attn = classify_slide(model, rgb, coords, device)
        labels.append(int(rec["label"]))
        probs.append(prob)
        heat = attention_heatmap(coords, attn, rgb.shape[0], rgb.shape[1])
        overlay = overlay_heatmap(rgb, heat)
        panel = side_by_side(rgb, overlay)
        Image.fromarray(panel).save(viz_dir / f"{rec['id']}_attn.png")
        if rec.get("mask"):
            mask = np.asarray(Image.open(rec["mask"]).convert("L"))
            tissue = np.zeros(mask.shape, dtype=np.uint8)
            for y, x in coords:
                tile = rgb[y : y + 256, x : x + 256]
                if is_tissue(tile):
                    tissue[y : y + 256, x : x + 256] = 1
            loc_scores.append(attention_in_tumor(heat, mask, tissue))

    y_true = np.asarray(labels)
    y_prob = np.asarray(probs)
    metrics = {
        "split": args.split,
        "n_slides": int(len(records)),
        "auc": roc_auc(y_true, y_prob),
        "accuracy": accuracy(y_true, y_prob, thresh),
        "threshold": thresh,
        "attention_in_tumor": float(np.mean(loc_scores)) if loc_scores else None,
        "labels": labels,
        "probs": [round(float(x), 4) for x in probs],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_roc(y_true, y_prob, args.out / "roc.png")

    overlays = sorted(viz_dir.glob("*_attn.png"))
    assets.mkdir(exist_ok=True)
    if overlays:
        Image.open(overlays[0]).save(assets / "attention_1.png")
    tumor = next((r for r in records if int(r["label"]) == 1), None)
    normal = next((r for r in records if int(r["label"]) == 0), None)
    if tumor:
        Image.open(viz_dir / f"{tumor['id']}_attn.png").save(assets / "attention_tumor.png")
    if normal:
        Image.open(viz_dir / f"{normal['id']}_attn.png").save(assets / "attention_normal.png")
    print(json.dumps({k: metrics[k] for k in ("split", "n_slides", "auc", "accuracy", "threshold", "attention_in_tumor")}, indent=2))


def _plot_roc(y_true: np.ndarray, y_score: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(fpr, tpr, color="#c05621", lw=2, label=f"AUC {roc_auc(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out slide ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
