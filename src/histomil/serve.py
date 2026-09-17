"""Cloud-style FastAPI /classify for a tiled H&E image."""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from histomil.evaluate import classify_slide, load_model
from histomil.tiling import tissue_tile_coords

app = FastAPI(title="HistoMIL", version="0.1.0")
_STATE: dict = {}


def _get_model() -> tuple[torch.nn.Module, torch.device]:
    if "model" not in _STATE:
        raise HTTPException(503, "model not loaded; start with python -m histomil.serve")
    return _STATE["model"], _STATE["device"]


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device": str(_STATE.get("device", "unloaded"))}


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> dict:
    model, device = _get_model()
    raw = await file.read()
    try:
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as exc:
        raise HTTPException(400, f"could not read image: {exc}") from exc
    t0 = time.perf_counter()
    coords = tissue_tile_coords(rgb)
    if not coords:
        raise HTTPException(400, "no tissue tiles found")
    prob, attn = classify_slide(model, rgb, coords, device)
    elapsed = (time.perf_counter() - t0) * 1000.0
    top = np.argsort(-attn)[:5]
    thresh = float(_STATE.get("threshold", 0.5))
    return {
        "prob_tumor": round(prob, 4),
        "label": "tumor" if prob >= thresh else "normal",
        "threshold": round(thresh, 3),
        "n_tiles": len(coords),
        "latency_ms": round(elapsed, 2),
        "ms_per_tile": round(elapsed / max(len(coords), 1), 3),
        "top_tiles": [{"y": int(coords[i][0]), "x": int(coords[i][1]), "attention": round(float(attn[i]), 5)} for i in top],
    }


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("outputs/histomil.pt"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--encoder", choices=["compact", "resnet18"], default="compact")
    p.add_argument("--dim", type=int, default=128)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location=device)
    _STATE["device"] = device
    _STATE["model"] = load_model(args.ckpt, device, args.encoder, args.dim)
    _STATE["threshold"] = float(blob.get("threshold", 0.5))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
