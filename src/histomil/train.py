"""Train gated Attention-MIL on slide-level bags (no pixel labels)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from histomil.dataset import SlideBagDataset, load_index
from histomil.metrics import roc_auc
from histomil.model import AttentionMIL


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_probs(model: AttentionMIL, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probs = [], []
    for batch in loader:
        bag = batch["bag"][0].to(device)
        logit, _ = model(bag)
        labels.append(float(batch["label"][0]))
        probs.append(float(torch.sigmoid(logit).item()))
    return np.asarray(labels), np.asarray(probs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-tiles", type=int, default=32)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--encoder", choices=["compact", "resnet18"], default="compact")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()

    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_ds = SlideBagDataset(load_index(args.data / "train"), max_tiles=args.max_tiles, augment=True, seed=args.seed)
    val_ds = SlideBagDataset(load_index(args.data / "val"), max_tiles=args.max_tiles, augment=False, seed=args.seed + 1)
    print(f"train slides={len(train_ds)}  val slides={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = AttentionMIL(dim=args.dim, encoder=args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_auc": []}
    best = -1.0
    best_loss = 1e9
    ckpt = args.out / "histomil.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in bar:
            bag = batch["bag"][0].to(device)
            y = batch["label"][0].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logit, _ = model(bag)
                loss = F.binary_cross_entropy_with_logits(logit.view(()), y.view(()))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            n += 1
            bar.set_postfix(loss=f"{running / max(n, 1):.4f}")
        sched.step()
        y_true, y_prob = collect_probs(model, val_loader, device)
        val_auc = roc_auc(y_true, y_prob)
        history["train_loss"].append(running / max(n, 1))
        history["val_auc"].append(val_auc)
        print(f"epoch {epoch}: loss={history['train_loss'][-1]:.4f}  val_auc={val_auc:.4f}")
        improved = val_auc > best or (abs(val_auc - best) < 1e-9 and history["train_loss"][-1] < best_loss)
        if improved:
            best = val_auc
            best_loss = history["train_loss"][-1]
            pos, neg = y_prob[y_true >= 0.5], y_prob[y_true < 0.5]
            if pos.size and neg.size and float(pos.min()) > float(neg.max()):
                thresh = 0.5 * (float(pos.min()) + float(neg.max()))
            else:
                thresh, thresh_acc = 0.5, -1.0
                for t in np.unique(y_prob):
                    acc = float(((y_true >= 0.5) == (y_prob >= t)).mean())
                    if acc > thresh_acc:
                        thresh_acc, thresh = acc, float(t)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "threshold": thresh,
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                },
                ckpt,
            )
            print(f"  saved {ckpt} (best val AUC {best:.4f}, thresh={thresh:.3f})")

    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _plot_curves(history, args.out / "training_curves.png")
    print(f"best val AUC={best:.4f}")


def _plot_curves(history: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(history["train_loss"], color="#1a365d")
    ax[0].set_title("Train loss (BCE)")
    ax[0].set_xlabel("epoch")
    ax[1].plot(history["val_auc"], color="#c05621")
    ax[1].set_title("Val AUC")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
