"""Export the tile encoder to ONNX and report ms/tile."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from histomil.evaluate import load_model
from histomil.model import TileEncoder


def _ms_per_tile(fn, n: int = 40, warmup: int = 8) -> float:
    dummy = torch.randn(1, 3, 256, 256)
    for _ in range(warmup):
        fn(dummy)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(dummy)
    return (time.perf_counter() - t0) * 1000.0 / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("outputs/histomil.pt"))
    p.add_argument("--out", type=Path, default=Path("outputs/tile_encoder.onnx"))
    p.add_argument("--encoder", choices=["compact", "resnet18"], default="compact")
    p.add_argument("--dim", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cpu")
    if args.ckpt.exists():
        mil = load_model(args.ckpt, device, args.encoder, args.dim)
        model = mil.encoder.eval()
    else:
        model = TileEncoder(args.dim).eval()
        print("checkpoint missing; exporting randomly initialized encoder")

    dummy = torch.randn(1, 3, 256, 256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["tile"],
        output_names=["embedding"],
        opset_version=17,
        dynamic_axes={"tile": {0: "batch"}},
    )
    print(f"exported {args.out}")

    def pt_step(x: torch.Tensor) -> None:
        with torch.no_grad():
            model(x)

    pt_ms = _ms_per_tile(pt_step)
    stats = {"pytorch_cpu_ms_per_tile": round(pt_ms, 3), "onnx": str(args.out)}

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])

        def ort_step(x: torch.Tensor) -> None:
            sess.run(None, {"tile": x.numpy()})

        stats["onnxruntime_cpu_ms_per_tile"] = round(_ms_per_tile(ort_step), 3)
    except Exception as exc:
        stats["onnxruntime"] = f"skipped: {exc}"

    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        gmodel = model.to(gpu).eval()
        gdummy = torch.randn(1, 3, 256, 256, device=gpu)
        for _ in range(8):
            gmodel(gdummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 40
        for _ in range(n):
            gmodel(gdummy)
        torch.cuda.synchronize()
        stats["pytorch_cuda_ms_per_tile"] = round((time.perf_counter() - t0) * 1000.0 / n, 3)

    (args.out.parent / "latency.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
