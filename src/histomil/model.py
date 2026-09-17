"""Gated Attention-MIL (Ilse et al.) with a compact ResNet-style tile encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


def _norm(ch: int) -> nn.GroupNorm:
    # GroupNorm is per-tile; BatchNorm over a bag of tiles breaks MIL eval.
    groups = 8 if ch >= 8 else 1
    return nn.GroupNorm(groups, ch)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            _norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                _norm(out_ch),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class TileEncoder(nn.Module):
    """Compact residual CNN: 256×256 RGB tile → `dim` embedding."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            _norm(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, dim, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class ResNet18Encoder(nn.Module):
    """Optional torchvision ResNet-18 tile encoder (swap-in)."""

    def __init__(self, dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, dim)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


def build_encoder(name: str = "compact", dim: int = 128, pretrained: bool = False) -> nn.Module:
    if name == "compact":
        return TileEncoder(dim)
    if name == "resnet18":
        return ResNet18Encoder(dim, pretrained=pretrained)
    raise ValueError(f"unknown encoder {name!r}; use compact|resnet18")


class AttentionMIL(nn.Module):
    """Bag of tiles → slide-level logit + gated attention weights."""

    def __init__(
        self,
        dim: int = 128,
        attn: int = 64,
        encoder: str = "compact",
        pretrained: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(encoder, dim, pretrained=pretrained)
        self.attn_v = nn.Linear(dim, attn)
        self.attn_u = nn.Linear(dim, attn)
        self.attn_c = nn.Linear(attn, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, 1)

    def encode(self, bag: torch.Tensor, chunk: int = 32) -> torch.Tensor:
        parts = [self.encoder(bag[i : i + chunk]) for i in range(0, bag.size(0), chunk)]
        return torch.cat(parts, dim=0)

    def attend(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.attn_c(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h)))
        w = torch.softmax(a.squeeze(-1), dim=0)
        z = torch.sum(w.unsqueeze(-1) * h, dim=0)
        logit = self.classifier(self.dropout(z)).squeeze(-1)
        return logit, w

    def forward(self, bag: torch.Tensor, chunk: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        # bag: (n_tiles, 3, 256, 256)
        h = self.encode(bag, chunk=chunk)
        return self.attend(h)
