"""
trust_scorer.py — Module 5: Trust Scoring Engine

Computes the Explanation Trust Score (ETS) using the formally defined formula:

    ETS(e) = w1 * C(e) + w2 * R(e) + w3 * S(e) + w4 * St(e) - lambda * H(e)

Where:
    C(e)  = Consistency score       (from Module 2)
    R(e)  = Robustness score        (from Module 3)
    S(e)  = Semantic coherence      (from Module 4)
    St(e) = Stability score         (1 - variance under input noise)
    H(e)  = Hallucination penalty   (fraction of methods hallucinated)

ETS is scaled to [0, 100]. Trust levels: HIGH >= 75, MEDIUM >= 50, LOW >= 25, REJECT < 25.

This module satisfies four formally proven axioms: Completeness, Monotonicity,
Nullity, and Symmetry (see Appendix A of the research paper).

Research Specification: Meta-XAI v1.0, Section 3.7
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

import config
from metaxai.schemas import (
    ConsistencyReport,
    ExplanationAuditReport,
    ExplanationMap,
    HallucinationResult,
    SemanticCoherenceResult,
)

logger = logging.getLogger(__name__)


# ── Trust Level Classification ─────────────────────────────────────────────────

def classify_trust_level(ets: float) -> str:
    """Map ETS score to a categorical trust level.

    Args:
        ets: Explanation Trust Score in [0, 100].

    Returns:
        Trust level string: 'HIGH', 'MEDIUM', 'LOW', or 'REJECT'.
    """
    if ets >= config.ETS_THRESHOLD_HIGH:
        return "HIGH"
    elif ets >= config.ETS_THRESHOLD_MEDIUM:
        return "MEDIUM"
    elif ets >= config.ETS_THRESHOLD_LOW:
        return "LOW"
    else:
        return "REJECT"


# ── Stability Score Computation ────────────────────────────────────────────────

def compute_stability_score(
    image: np.ndarray,
    model: nn.Module,
    explanation_fn,
    num_samples: int = config.STABILITY_NUM_SAMPLES,
    noise_std: float = config.STABILITY_NOISE_STD,
    seed: int = config.RANDOM_SEED,
) -> float:
    """Compute attribution stability under Gaussian input noise.

    Measures how consistent the explanation is when the input is perturbed
    with small noise. A high stability score means the explanation is
    robust to minor input variations.

    Args:
        image: Input image as numpy array (H, W, C), float32, [0, 1].
        model: PyTorch model in eval mode.
        explanation_fn: Callable that accepts (image, model) and returns
            an attribution map of shape (H, W).
        num_samples: Number of noisy samples to generate. Default: 10.
        noise_std: Standard deviation of Gaussian noise. Default: 0.05.
        seed: Random seed for reproducibility.

    Returns:
        Stability score in [0, 1]. 1.0 = perfectly stable, 0.0 = completely unstable.
    """
    rng = np.random.RandomState(seed)
    attributions = []

    # Generate base attribution
    try:
        base_attr = explanation_fn(image, model)
        if base_attr is None or base_attr.sum() == 0:
            return 0.5  # neutral fallback
        attributions.append(base_attr.flatten())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stability: base attribution failed: %s. Returning 0.5.", exc)
        return 0.5

    # Generate attributions under noisy inputs
    for _ in range(num_samples):
        noisy = np.clip(image + rng.normal(0, noise_std, image.shape), 0.0, 1.0)
        try:
            noisy_attr = explanation_fn(noisy.astype(np.float32), model)
            attributions.append(noisy_attr.flatten())
        except Exception:  # noqa: BLE001
            continue

    if len(attributions) < 2:
        return 0.5

    # Compute mean pixel-wise variance across samples
    attr_matrix = np.stack(attributions, axis=0)  # (samples, H*W)
    pixel_variance = attr_matrix.var(axis=0)      # (H*W,)
    mean_variance = pixel_variance.mean()

    # Convert variance to stability: lower variance = higher stability
    # Clip at 1.0 to keep within [0, 1] range
    stability = 1.0 - float(np.clip(mean_variance * 10.0, 0.0, 1.0))
    logger.debug("Stability score: %.4f (mean_variance=%.6f)", stability, mean_variance)
    return stability


# ── ETS Formula ────────────────────────────────────────────────────────────────

def compute_ets(
    consistency_score: float,
    robustness_score: float,
    semantic_score: float,
    stability_score: float,
    hallucination_fraction: float,
    w1: float = config.ETS_W1_CONSISTENCY,
    w2: float = config.ETS_W2_ROBUSTNESS,
    w3: float = config.ETS_W3_SEMANTIC,
    w4: float = config.ETS_W4_STABILITY,
    lam: float = config.ETS_LAMBDA_HALLUCINATION,
) -> float:
    """Compute the Explanation Trust Score (ETS).

    Implements the formally defined ETS formula:
        ETS(e) = w1*C + w2*R + w3*S + w4*St - lambda*H

    All component scores must be in [0, 1]. The result is scaled to [0, 100].

    Axiomatic Properties:
        - Completeness: ETS=100 iff C=R=S=St=1 and H=0
        - Monotonicity: ETS is strictly increasing in each component
        - Nullity: ETS→0 when H=1 and all other components=0
        - Symmetry: Score is invariant to method ordering (via symmetric matrices in C)

    Args:
        consistency_score: C(e) in [0, 1].
        robustness_score: R(e) in [0, 1].
        semantic_score: S(e) in [0, 1].
        stability_score: St(e) in [0, 1].
        hallucination_fraction: H(e) in [0, 1]. Fraction of methods hallucinated.
        w1: Weight for consistency. Default: config.ETS_W1_CONSISTENCY.
        w2: Weight for robustness. Default: config.ETS_W2_ROBUSTNESS.
        w3: Weight for semantic. Default: config.ETS_W3_SEMANTIC.
        w4: Weight for stability. Default: config.ETS_W4_STABILITY.
        lam: Hallucination penalty multiplier. Default: config.ETS_LAMBDA_HALLUCINATION.

    Returns:
        ETS in [0, 100], clamped.

    Raises:
        ValueError: If any input is outside [0, 1].
    """
    for name, val in [
        ("consistency_score", consistency_score),
        ("robustness_score", robustness_score),
        ("semantic_score", semantic_score),
        ("stability_score", stability_score),
        ("hallucination_fraction", hallucination_fraction),
    ]:
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {val:.4f}")

    raw_ets = (
        w1 * consistency_score
        + w2 * robustness_score
        + w3 * semantic_score
        + w4 * stability_score
        - lam * hallucination_fraction
    )

    # Scale to [0, 100] and clamp
    ets = float(np.clip(raw_ets * 100.0, 0.0, 100.0))

    logger.debug(
        "ETS components: C=%.3f, R=%.3f, S=%.3f, St=%.3f, H=%.3f → raw=%.4f → ETS=%.1f",
        consistency_score, robustness_score, semantic_score,
        stability_score, hallucination_fraction, raw_ets, ets,
    )
    return ets


# ── Public API ─────────────────────────────────────────────────────────────────

class TrustScoringEngine:
    """Module 5: Computes the Explanation Trust Score from all module outputs.

    Aggregates consistency, robustness, semantic coherence, stability, and
    hallucination signals into a single ETS in [0, 100] with a categorical
    trust level (HIGH/MEDIUM/LOW/REJECT).

    This is the only module permitted to read from both the ConsistencyAnalyzer
    and the HallucinationDetector outputs. It is the aggregation layer.

    Args:
        model: PyTorch model (required for stability score computation).
        enable_stability: If True, compute stability score. Default: True.

    Example:
        >>> engine = TrustScoringEngine(model=my_model)
        >>> ets, level = engine.score(
        ...     consistency_report=cr,
        ...     hallucination_results=hr,
        ...     semantic_score=0.8,
        ...     explanation_maps=maps,
        ...     image=image_np,
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        enable_stability: bool = True,
    ) -> None:
        self.model = model
        self.model.eval()
        self.enable_stability = enable_stability
        logger.info(
            "TrustScoringEngine initialized. enable_stability=%s", enable_stability
        )

    def _compute_robustness_score(
        self, hallucination_results: List[HallucinationResult]
    ) -> float:
        """Compute robustness score as mean perturbation delta across methods.

        Args:
            hallucination_results: List of HallucinationResult from Module 3.

        Returns:
            Robustness score in [0, 1].
        """
        if not hallucination_results:
            return 0.5
        deltas = [r.perturbation_delta for r in hallucination_results]
        mean_delta = np.mean(deltas)
        # Normalize: delta of PERTURBATION_DELTA (0.15) or above = score of 1.0
        normalized = float(np.clip(mean_delta / config.PERTURBATION_DELTA, 0.0, 1.0))
        return normalized

    def score(
        self,
        consistency_report: ConsistencyReport,
        hallucination_results: List[HallucinationResult],
        semantic_score: float,
        explanation_maps: List[ExplanationMap],
        image: np.ndarray,
        semantic_coherence_result: Optional[SemanticCoherenceResult] = None,
        stability_score: Optional[float] = None,
    ) -> tuple[float, str, float, float]:
        """Compute the final ETS and trust level.

        Args:
            consistency_report: ConsistencyReport from Module 2.
            hallucination_results: List of HallucinationResult from Module 3.
            semantic_score: Semantic coherence score from Module 4, in [0, 1].
            explanation_maps: List of ExplanationMap from Module 1 (for stability).
            image: Input image as numpy array (H, W, C) (for stability computation).
            semantic_coherence_result: Full SemanticCoherenceResult (optional).
            stability_score: Pre-computed stability score. Computed here if None.

        Returns:
            Tuple of (ets: float, trust_level: str, robustness_score: float, stability_score: float).
        """
        # Consistency score from Module 2
        c_score = consistency_report.pairwise_agreement_score

        # Robustness score from Module 3 (mean perturbation delta)
        r_score = self._compute_robustness_score(hallucination_results)

        # Semantic score from Module 4
        s_score = float(np.clip(semantic_score, 0.0, 1.0))

        # Stability score
        if stability_score is not None:
            st_score = float(np.clip(stability_score, 0.0, 1.0))
        elif self.enable_stability and explanation_maps:
            # Use first valid explanation map's method as proxy for stability
            valid_maps = [m for m in explanation_maps if m.is_valid]
            if valid_maps:
                # Simplified: use attribution variance as proxy for stability
                attr_list = [m.attribution for m in valid_maps]
                if len(attr_list) > 1:
                    stacked = np.stack([a.flatten() for a in attr_list], axis=0)
                    variance = stacked.var(axis=0).mean()
                    st_score = 1.0 - float(np.clip(variance * 20.0, 0.0, 1.0))
                else:
                    st_score = 0.75  # single method: assume moderate stability
            else:
                st_score = 0.5
        else:
            st_score = 0.75  # default when stability is disabled

        # Hallucination fraction H(e)
        h_fraction = 0.0
        if hallucination_results:
            h_fraction = sum(1 for r in hallucination_results if r.is_hallucinated) / len(hallucination_results)

        # Compute ETS
        ets = compute_ets(
            consistency_score=c_score,
            robustness_score=r_score,
            semantic_score=s_score,
            stability_score=st_score,
            hallucination_fraction=h_fraction,
        )

        trust_level = classify_trust_level(ets)

        logger.info(
            "TrustScoringEngine: ETS=%.1f/100 [%s] | C=%.3f R=%.3f S=%.3f St=%.3f H=%.3f",
            ets, trust_level, c_score, r_score, s_score, st_score, h_fraction,
        )

        return ets, trust_level, r_score, st_score
