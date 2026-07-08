"""
metrics.py — Meta-XAI Evaluation Metrics

Implements the three primary evaluation metrics used in the benchmark:
    - HDR: Hallucination Detection Rate (recall of true hallucinations)
    - FPR: False Positive Rate (fraction of valid explanations incorrectly flagged)
    - ETS-human rho: Spearman correlation between ETS and human trust ratings
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from typing import List, Optional


def hallucination_detection_rate(
    y_true: List[bool],
    y_pred: List[bool],
) -> float:
    """Compute Hallucination Detection Rate (recall).

    HDR = TP / (TP + FN)

    Args:
        y_true: Ground-truth hallucination labels (True = hallucinated).
        y_pred: Predicted hallucination labels.

    Returns:
        HDR in [0, 1]. Returns 0.0 if no true positives exist.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def false_positive_rate(
    y_true: List[bool],
    y_pred: List[bool],
) -> float:
    """Compute False Positive Rate.

    FPR = FP / (FP + TN)

    Args:
        y_true: Ground-truth hallucination labels.
        y_pred: Predicted hallucination labels.

    Returns:
        FPR in [0, 1].
    """
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def ets_human_correlation(
    ets_scores: List[float],
    human_ratings: List[float],
) -> tuple[float, float]:
    """Compute Spearman rank correlation between ETS and human trust ratings.

    Args:
        ets_scores: List of ETS values in [0, 100].
        human_ratings: List of human-annotated trust scores (any ordinal scale).

    Returns:
        Tuple of (spearman_rho, p_value).
    """
    rho, pval = spearmanr(ets_scores, human_ratings)
    return float(rho), float(pval)


def compute_metrics_with_ci(
    y_true: List[bool],
    y_pred: List[bool],
    ets_scores: Optional[List[float]] = None,
    human_ratings: Optional[List[float]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute all evaluation metrics with 95% confidence intervals via bootstrap.

    Args:
        y_true: Ground-truth hallucination labels.
        y_pred: Predicted hallucination labels.
        ets_scores: ETS scores for correlation (optional).
        human_ratings: Human trust ratings (optional, paired with ets_scores).
        n_bootstrap: Number of bootstrap samples for CI estimation.
        seed: Random seed for reproducibility.

    Returns:
        Dict with metric names as keys and (mean, std, ci_low, ci_high) tuples as values.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    hdr_samples, fpr_samples = [], []
    rho_samples = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        hdr_samples.append(hallucination_detection_rate(yt, yp))
        fpr_samples.append(false_positive_rate(yt, yp))

        if ets_scores and human_ratings:
            es = [ets_scores[i] for i in idx]
            hr = [human_ratings[i] for i in idx]
            rho, _ = spearmanr(es, hr)
            rho_samples.append(rho if not np.isnan(rho) else 0.0)

    def _stats(samples):
        arr = np.array(samples)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }

    results = {
        "HDR": _stats(hdr_samples),
        "FPR": _stats(fpr_samples),
    }
    if rho_samples:
        results["ETS_human_rho"] = _stats(rho_samples)

    return results
