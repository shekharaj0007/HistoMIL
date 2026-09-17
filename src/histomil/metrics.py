"""Slide-level AUC / accuracy. No pixel labels required."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_hat = (np.asarray(y_prob) >= thresh).astype(int)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_hat).mean())


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    p = float(tps[-1]) if tps.size else 0.0
    n = float(fps[-1]) if fps.size else 0.0
    if p == 0 or n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    fpr = np.concatenate([[0.0], fps / n, [1.0]])
    tpr = np.concatenate([[0.0], tps / p, [1.0]])
    return fpr, tpr


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = roc_curve(y_true, y_score)
    return float(np.trapz(tpr, fpr))


def attention_in_tumor(
    attn_map: np.ndarray,
    tumor_mask: np.ndarray,
    tissue_mask: np.ndarray | None = None,
) -> float:
    """Fraction of attention mass that lands inside the hidden tumor region."""
    weights = attn_map.astype(np.float64)
    if tissue_mask is not None:
        weights = weights * (tissue_mask > 0)
    total = weights.sum()
    if total <= 0:
        return 0.0
    return float((weights * (tumor_mask > 0)).sum() / total)
