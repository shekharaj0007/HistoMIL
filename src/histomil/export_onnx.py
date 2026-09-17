"""Export tile encoder + attention pooling path to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from histomil.model import TileEncoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/tile_encoder.onnx"))
    args = p.parse_args()
    model = TileEncoder().eval()
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


if __name__ == "__main__":
    main()
