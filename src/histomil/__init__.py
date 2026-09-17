"""HistoMIL: gated Attention-MIL for gigapixel histopathology WSIs."""

from .model import AttentionMIL, TileEncoder
from .tiling import extract_tissue_tiles

__all__ = ["AttentionMIL", "TileEncoder", "extract_tissue_tiles"]
