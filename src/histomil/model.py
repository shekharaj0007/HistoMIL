"""Gated Attention-MIL (Ilse et al.) with a ResNet-18 tile encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TileEncoder(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class AttentionMIL(nn.Module):
    """Bag of tiles → slide-level logit + attention heatmap weights."""

    def __init__(self, dim: int = 128, attn: int = 64) -> None:
        super().__init__()
        self.encoder = TileEncoder(dim)
        self.attn_v = nn.Linear(dim, attn)
        self.attn_u = nn.Linear(dim, attn)
        self.attn_c = nn.Linear(attn, 1)
        self.classifier = nn.Linear(dim, 1)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # bag: (n_tiles, 3, 256, 256)
        h = self.encoder(bag)
        a = self.attn_c(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h)))
        w = torch.softmax(a.squeeze(-1), dim=0)
        z = torch.sum(w.unsqueeze(-1) * h, dim=0)
        logit = self.classifier(z)
        return logit, w
