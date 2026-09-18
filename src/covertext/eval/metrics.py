"""Pure evaluation metrics. No encoder, detector, or model dependency."""

from __future__ import annotations

import math
import statistics
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def capacity_bpt(payload_bits: int, num_tokens: int) -> float:
    """Capacity in bits per token."""
    if num_tokens == 0:
        raise ValueError("num_tokens must be non-zero")
    return payload_bits / num_tokens


def decode_accuracy_bit(original: str, recovered: str) -> float:
    """Bit-level accuracy. Excess bits from a length mismatch count as errors."""
    n = max(len(original), len(recovered), 1)
    matches = sum(a == b for a, b in zip(original, recovered))
    return matches / n


def decode_accuracy_exact(original: str, recovered: str) -> bool:
    """Exact payload match."""
    return original == recovered


def fluency_loss(stego_ppl: float, clean_ppl: float) -> float:
    """Perplexity delta. Positive means the stego text is less fluent."""
    return stego_ppl - clean_ppl


def compute_perplexity(log_probs: list[float]) -> float:
    """Perplexity from per-token natural-log probabilities: exp(-mean(log_probs))."""
    if not log_probs:
        raise ValueError("log_probs must be non-empty")
    return math.exp(-statistics.mean(log_probs))


def compute_auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC. labels: 0 = clean, 1 = stego. scores: higher = more likely stego."""
    if len(set(labels)) < 2:
        warnings.warn(
            "AUROC is undefined when all labels are the same; returning 0.5",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_roc_curve(
    scores: list[float], labels: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve. Returns (fpr, tpr, thresholds)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds
