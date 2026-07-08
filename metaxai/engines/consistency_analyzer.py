"""
consistency_analyzer.py — Module 2: Consistency Analyzer

Computes three families of pairwise agreement metrics across all XAI explanation maps:
  - IoU (Intersection over Union) on binarized attribution maps
  - Cosine Similarity on flattened attribution vectors
  - Rank Correlation (Spearman rho) on feature importance rankings

This module is the heart of the Meta-XAI pipeline.

Research Specification: Meta-XAI v1.0, Section 3.4
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy.stats import spearmanr

import config
from metaxai.schemas import ConsistencyReport, ExplanationMap

logger = logging.getLogger(__name__)


class ConsistencyAnalyzer:
    """Module 2: Computes pairwise agreement across XAI explanation maps.

    Measures how consistently multiple XAI methods agree on which regions
    are important for a given prediction. Three complementary metrics are
    computed to capture spatial, directional, and ordinal agreement.

    Args:
        binarization_threshold: Threshold tau for converting attribution maps
            to binary masks for IoU computation. Default: config.BINARIZATION_THRESHOLD.
        top_k_features: Number of top features to compare for rank correlation.
            Default: config.TOP_K_FEATURES.
        w_iou: Weight for IoU metric in aggregate score.
        w_cosine: Weight for Cosine metric in aggregate score.
        w_rank: Weight for Rank Correlation metric in aggregate score.

    Example:
        >>> analyzer = ConsistencyAnalyzer()
        >>> report = analyzer.analyze(explanation_maps)
        >>> print(report.pairwise_agreement_score)
    """

    def __init__(
        self,
        binarization_threshold: float = config.BINARIZATION_THRESHOLD,
        top_k_features: int = config.TOP_K_FEATURES,
        w_iou: float = config.CONSISTENCY_W_IOU,
        w_cosine: float = config.CONSISTENCY_W_COSINE,
        w_rank: float = config.CONSISTENCY_W_RANK,
    ) -> None:
        self.tau = binarization_threshold
        self.top_k = top_k_features
        self.w_iou = w_iou
        self.w_cosine = w_cosine
        self.w_rank = w_rank
        logger.info(
            "ConsistencyAnalyzer initialized. tau=%.2f, top_k=%d", self.tau, self.top_k
        )

    # ── Metric: IoU ───────────────────────────────────────────────────────────

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
        """Compute IoU between two binarized attribution maps.

        Args:
            a: Attribution map A, shape (H, W), values in [0, 1].
            b: Attribution map B, shape (H, W), values in [0, 1].
            threshold: Binarization threshold tau.

        Returns:
            IoU score in [0, 1]. Returns 0.0 if union is empty.
        """
        a_bin = (a >= threshold).astype(bool)
        b_bin = (b >= threshold).astype(bool)
        intersection = np.logical_and(a_bin, b_bin).sum()
        union = np.logical_or(a_bin, b_bin).sum()
        return float(intersection / union) if union > 0 else 0.0

    # ── Metric: Cosine Similarity ──────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two flattened attribution vectors.

        Args:
            a: Attribution map A, any shape.
            b: Attribution map B, any shape (same number of elements as a).

        Returns:
            Cosine similarity in [-1, 1]. Clipped to [0, 1] since attributions
            are non-negative after normalization.
        """
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)

        # Resize to common shape if maps differ (should not happen in practice)
        min_len = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]

        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        cos = np.dot(a_flat, b_flat) / (norm_a * norm_b)
        return float(np.clip(cos, 0.0, 1.0))  # clip to [0,1] since maps are non-negative

    # ── Metric: Rank Correlation (Spearman) ───────────────────────────────────

    def _rank_corr(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Spearman rank correlation between top-K feature rankings.

        Measures ordinal agreement — whether both methods rank the same
        features as most important, regardless of magnitude.

        Args:
            a: Attribution map A, any shape.
            b: Attribution map B, any shape.

        Returns:
            Spearman rho in [-1, 1], normalized to [0, 1] via (rho + 1) / 2.
        """
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)

        # Use top-K indices from A to compare rankings
        min_len = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]

        k = min(self.top_k, min_len)
        top_k_idx = np.argsort(a_flat)[-k:]

        if len(top_k_idx) < 2:
            return 0.5  # undefined; return neutral

        rho, pval = spearmanr(a_flat[top_k_idx], b_flat[top_k_idx])
        if np.isnan(rho):
            # NaN occurs when one or both inputs are constant.
            # If both are constant and equal, they are perfectly correlated → 1.0.
            # If they differ in value but are constant, they are uncorrelated → 0.5.
            a_vals = a_flat[top_k_idx]
            b_vals = b_flat[top_k_idx]
            if np.allclose(a_vals, b_vals):
                return 1.0
            return 0.5
        return float((rho + 1.0) / 2.0)  # normalize [-1,1] → [0,1]

    # ── Pairwise Matrix Computation ───────────────────────────────────────────

    def _build_matrices(
        self, maps: List[ExplanationMap]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build pairwise metric matrices for all valid explanation maps.

        Args:
            maps: List of valid ExplanationMap objects.

        Returns:
            Tuple of (iou_matrix, cosine_matrix, rank_corr_matrix),
            each of shape (n, n) where n = len(maps).
        """
        n = len(maps)
        iou_mat = np.eye(n, dtype=np.float32)
        cos_mat = np.eye(n, dtype=np.float32)
        rank_mat = np.eye(n, dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                a = maps[i].attribution
                b = maps[j].attribution

                iou_val = self._iou(a, b, self.tau)
                cos_val = self._cosine(a, b)
                rank_val = self._rank_corr(a, b)

                iou_mat[i, j] = iou_mat[j, i] = iou_val
                cos_mat[i, j] = cos_mat[j, i] = cos_val
                rank_mat[i, j] = rank_mat[j, i] = rank_val

                logger.debug(
                    "Pair (%s, %s): IoU=%.3f, Cos=%.3f, Rank=%.3f",
                    maps[i].method, maps[j].method, iou_val, cos_val, rank_val,
                )

        return iou_mat, cos_mat, rank_mat

    # ── Aggregate Score ────────────────────────────────────────────────────────

    def _aggregate_score(
        self,
        iou_mat: np.ndarray,
        cos_mat: np.ndarray,
        rank_mat: np.ndarray,
    ) -> float:
        """Compute weighted aggregate pairwise agreement score.

        Uses the upper triangle (excluding diagonal) for averaging.
        Weights are learned from the calibration study (exp03).

        Args:
            iou_mat: Pairwise IoU matrix, shape (n, n).
            cos_mat: Pairwise cosine similarity matrix, shape (n, n).
            rank_mat: Pairwise rank correlation matrix, shape (n, n).

        Returns:
            pairwise_agreement_score in [0, 1].
        """
        n = iou_mat.shape[0]
        if n < 2:
            return 1.0  # single method — trivially consistent with itself

        # Extract upper triangle (excluding diagonal)
        iu = np.triu_indices(n, k=1)
        iou_mean = iou_mat[iu].mean()
        cos_mean = cos_mat[iu].mean()
        rank_mean = rank_mat[iu].mean()

        score = (
            self.w_iou * iou_mean
            + self.w_cosine * cos_mean
            + self.w_rank * rank_mean
        )
        return float(np.clip(score, 0.0, 1.0))

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, explanation_maps: List[ExplanationMap]) -> ConsistencyReport:
        """Compute consistency metrics across all valid XAI explanation maps.

        Args:
            explanation_maps: List of ExplanationMap objects. Invalid maps
                (is_valid=False) are filtered out before analysis.

        Returns:
            ConsistencyReport with pairwise metric matrices and aggregate score.

        Raises:
            ValueError: If fewer than 1 valid explanation map is provided.
        """
        valid_maps = [m for m in explanation_maps if m.is_valid]
        logger.info(
            "ConsistencyAnalyzer: %d valid maps (filtered %d invalid)",
            len(valid_maps),
            len(explanation_maps) - len(valid_maps),
        )

        if len(valid_maps) == 0:
            raise ValueError(
                "ConsistencyAnalyzer requires at least 1 valid ExplanationMap."
            )

        if len(valid_maps) == 1:
            logger.warning(
                "Only 1 valid method; returning identity matrices with score=1.0"
            )
            return ConsistencyReport(
                iou_matrix=np.array([[1.0]], dtype=np.float32),
                cosine_matrix=np.array([[1.0]], dtype=np.float32),
                rank_corr_matrix=np.array([[1.0]], dtype=np.float32),
                pairwise_agreement_score=1.0,
                method_names=[valid_maps[0].method],
            )

        iou_mat, cos_mat, rank_mat = self._build_matrices(valid_maps)
        score = self._aggregate_score(iou_mat, cos_mat, rank_mat)
        method_names = [m.method for m in valid_maps]

        logger.info(
            "ConsistencyAnalyzer complete. pairwise_agreement_score=%.4f", score
        )

        return ConsistencyReport(
            iou_matrix=iou_mat,
            cosine_matrix=cos_mat,
            rank_corr_matrix=rank_mat,
            pairwise_agreement_score=score,
            method_names=method_names,
        )
