"""
schemas.py — Meta-XAI Canonical Data Schemas

All inter-module communication uses these typed dataclasses.
No untyped dicts are permitted to cross module boundaries.

Research Specification: Meta-XAI v1.0, Section 3.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Explanation Generator Output ───────────────────────────────────────────────

@dataclass
class ExplanationMap:
    """Attribution map produced by a single XAI method.

    Args:
        method: Name of the XAI method. One of 'shap', 'lime', 'gradcam', 'intgrad'.
        attribution: Attribution map of shape (H, W), float32, min-max normalized to [0, 1].
        top_k_features: Indices of top-K contributing features (flattened pixel indices).
        computation_time_ms: Wall-clock time to generate this explanation, in milliseconds.
        metadata: Arbitrary method-specific metadata (e.g., SHAP expected value).
        is_valid: False if the method failed and returned an empty/fallback map.

    Raises:
        ValueError: If attribution is not a 2D float array.
    """

    method: str
    attribution: np.ndarray          # shape (H, W), float32, [0, 1] normalized
    top_k_features: List[int]        # flattened pixel indices of top-K attributions
    computation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True

    def __post_init__(self) -> None:
        """Validate the attribution map after construction."""
        if self.attribution is not None and self.attribution.ndim != 2:
            raise ValueError(
                f"ExplanationMap.attribution must be 2D, got shape {self.attribution.shape}"
            )
        if self.method not in {"shap", "lime", "gradcam", "intgrad", "attention", "__empty__"}:
            logger.warning("Unknown XAI method name: '%s'", self.method)

    @classmethod
    def empty(cls, method: str) -> "ExplanationMap":
        """Create an invalid/empty ExplanationMap as a fallback on method failure.

        Args:
            method: Name of the failing XAI method.

        Returns:
            An ExplanationMap with is_valid=False and zero attribution.
        """
        return cls(
            method=method,
            attribution=np.zeros((1, 1), dtype=np.float32),
            top_k_features=[],
            computation_time_ms=0.0,
            metadata={"error": "Method failed or timed out."},
            is_valid=False,
        )


# ── Consistency Analyzer Output ────────────────────────────────────────────────

@dataclass
class ConsistencyReport:
    """Pairwise agreement metrics across all XAI methods.

    Args:
        iou_matrix: Pairwise IoU scores. Shape (n_methods, n_methods), symmetric.
        cosine_matrix: Pairwise cosine similarity scores. Shape (n_methods, n_methods).
        rank_corr_matrix: Pairwise Spearman rank correlation. Shape (n_methods, n_methods).
        pairwise_agreement_score: Weighted aggregate of all three metrics, in [0, 1].
        method_names: Ordered list of method names corresponding to matrix rows/cols.
    """

    iou_matrix: np.ndarray             # shape (n_methods, n_methods)
    cosine_matrix: np.ndarray          # shape (n_methods, n_methods)
    rank_corr_matrix: np.ndarray       # shape (n_methods, n_methods)
    pairwise_agreement_score: float    # aggregate consistency in [0, 1]
    method_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate matrix shapes are consistent."""
        n = self.iou_matrix.shape[0]
        assert self.cosine_matrix.shape == (n, n), "cosine_matrix shape mismatch"
        assert self.rank_corr_matrix.shape == (n, n), "rank_corr_matrix shape mismatch"
        assert 0.0 <= self.pairwise_agreement_score <= 1.0, \
            f"pairwise_agreement_score must be in [0,1], got {self.pairwise_agreement_score}"


# ── Hallucination Detector Output ─────────────────────────────────────────────

@dataclass
class PerturbationResult:
    """Result of the perturbation test for a single explanation.

    Args:
        method: XAI method name.
        perturbation_delta: Confidence drop when top-K region is occluded.
        is_valid: True if confidence drop >= PERTURBATION_DELTA threshold.
    """

    method: str
    perturbation_delta: float   # change in model confidence after occlusion
    is_valid: bool              # True if delta >= threshold (explanation is causally valid)


@dataclass
class BoundaryResult:
    """Result of the segmentation boundary test for a single explanation.

    Args:
        method: XAI method name.
        boundary_overlap_ratio: Fraction of attribution mass inside the segmentation mask.
        is_valid: True if overlap >= BOUNDARY_OVERLAP_THRESHOLD.
    """

    method: str
    boundary_overlap_ratio: float  # fraction of attribution inside object mask
    is_valid: bool                 # True if ratio >= threshold


@dataclass
class HallucinationResult:
    """Combined hallucination detection result for a single XAI method.

    Args:
        method: XAI method name.
        is_hallucinated: True if BOTH perturbation test AND boundary test indicate failure.
        perturbation_delta: Confidence drop from perturbation test.
        boundary_overlap_ratio: Attribution overlap with segmentation mask.
        confidence: Detector's confidence in the hallucination verdict, in [0, 1].
    """

    method: str
    is_hallucinated: bool
    perturbation_delta: float         # from Strategy A
    boundary_overlap_ratio: float     # from Strategy B
    confidence: float                 # detector confidence in [0, 1]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"HallucinationResult.confidence must be in [0,1], got {self.confidence}")


# ── Semantic Coherence Output ──────────────────────────────────────────────────

@dataclass
class SemanticCoherenceResult:
    """LLM-grounded semantic validation result.

    Args:
        score: Semantic coherence score in [0, 1]. 1.0 = fully coherent.
        justification: Brief LLM-generated justification for the score.
        region_description: Description of the top-K attribution region queried.
        predicted_label: The predicted class label used in the query.
        model_used: Name of the LLM model used for this query.
    """

    score: float                 # coherence score in [0, 1]
    justification: str           # LLM justification text
    region_description: str      # description of queried region
    predicted_label: str
    model_used: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"SemanticCoherenceResult.score must be in [0,1], got {self.score}")


# ── Final Pipeline Output ─────────────────────────────────────────────────────

@dataclass
class ExplanationAuditReport:
    """The complete output of the Meta-XAI pipeline for a single prediction.

    This is the canonical deliverable of the system. It aggregates outputs
    from all five modules into a unified, structured audit.

    Args:
        image_id: Unique identifier for the input image.
        predicted_class: Model's predicted class label.
        model_confidence: Model's softmax confidence for the predicted class.
        explanation_maps: List of ExplanationMap objects from Module 1.
        consistency_report: ConsistencyReport from Module 2.
        hallucination_results: List of HallucinationResult from Module 3.
        semantic_coherence_score: LLM coherence score from Module 4, in [0, 1].
        semantic_coherence_result: Full SemanticCoherenceResult from Module 4.
        explanation_trust_score: ETS from Module 5, in [0, 100].
        trust_level: Human-readable trust classification.
        stability_score: Attribution stability under input noise, in [0, 1].
        robustness_score: Mean perturbation delta across methods, in [0, 1].
        audit_timestamp: ISO 8601 timestamp of when the audit was run.
        pipeline_version: Version string of the Meta-XAI pipeline used.
    """

    image_id: str
    predicted_class: str
    model_confidence: float
    explanation_maps: List[ExplanationMap]
    consistency_report: ConsistencyReport
    hallucination_results: List[HallucinationResult]
    semantic_coherence_score: float
    explanation_trust_score: float       # ETS in [0, 100]
    trust_level: str                     # 'HIGH' | 'MEDIUM' | 'LOW' | 'REJECT'
    stability_score: float = 0.0
    robustness_score: float = 0.0
    semantic_coherence_result: Optional[SemanticCoherenceResult] = None
    audit_timestamp: str = ""
    pipeline_version: str = "1.0.0"

    # Derived properties
    @property
    def num_hallucinated(self) -> int:
        """Count of XAI methods flagged as hallucinated."""
        return sum(1 for h in self.hallucination_results if h.is_hallucinated)

    @property
    def hallucination_fraction(self) -> float:
        """Fraction of XAI methods flagged as hallucinated. H(e) in ETS formula."""
        if not self.hallucination_results:
            return 0.0
        return self.num_hallucinated / len(self.hallucination_results)

    @property
    def valid_explanation_methods(self) -> List[str]:
        """List of XAI methods that produced valid (non-hallucinated) explanations."""
        return [h.method for h in self.hallucination_results if not h.is_hallucinated]

    def __post_init__(self) -> None:
        valid_trust_levels = {"HIGH", "MEDIUM", "LOW", "REJECT"}
        if self.trust_level not in valid_trust_levels:
            raise ValueError(
                f"trust_level must be one of {valid_trust_levels}, got '{self.trust_level}'"
            )
        if not 0.0 <= self.explanation_trust_score <= 100.0:
            raise ValueError(
                f"explanation_trust_score must be in [0, 100], got {self.explanation_trust_score}"
            )

    def summary(self) -> str:
        """Return a human-readable summary of the audit report.

        Returns:
            A formatted string summarizing key audit metrics.
        """
        hallucinated = [h.method for h in self.hallucination_results if h.is_hallucinated]
        return (
            f"=== Meta-XAI Audit Report ===\n"
            f"Image:          {self.image_id}\n"
            f"Predicted:      {self.predicted_class} (confidence: {self.model_confidence:.3f})\n"
            f"ETS:            {self.explanation_trust_score:.1f}/100 [{self.trust_level}]\n"
            f"Consistency:    {self.consistency_report.pairwise_agreement_score:.3f}\n"
            f"Robustness:     {self.robustness_score:.3f}\n"
            f"Semantic:       {self.semantic_coherence_score:.3f}\n"
            f"Stability:      {self.stability_score:.3f}\n"
            f"Hallucinated:   {hallucinated if hallucinated else 'None'}\n"
            f"Timestamp:      {self.audit_timestamp}\n"
        )
