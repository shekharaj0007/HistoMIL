"""Train Attention-MIL for slide-level tumor vs normal."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from histomil.model import AttentionMIL


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", type=Path, default=Path("outputs/histomil.pt"))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMIL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dummy_bag = torch.randn(8, 3, 256, 256, device=device)
    logit, attn = model(dummy_bag)
    loss = F.binary_cross_entropy_with_logits(logit, torch.ones(1, device=device))
    loss.backward()
    opt.step()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "attn_example": attn.detach().cpu()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
