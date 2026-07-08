"""
test_axioms.py — ETS Axiomatic Property Tests

Formally verifies that the ETS formula satisfies all four axioms:
    1. Completeness:   ETS=100 iff all components=1 and H=0
    2. Monotonicity:   Increasing any component cannot decrease ETS
    3. Nullity:        ETS→0 when H=1 and all positive components=0
    4. Symmetry:       Score is invariant to method ordering

These tests are required before paper submission as per Section 2.4.
"""

import pytest
import numpy as np
from metaxai.engines.trust_scorer import compute_ets, classify_trust_level
import config


class TestCompletenessAxiom:
    """Axiom 1: ETS=100 if and only if C=R=S=St=1 and H=0."""

    def test_completeness_all_perfect_yields_100(self):
        """ETS must equal 100 when all component scores are 1 and H=0."""
        ets = compute_ets(
            consistency_score=1.0,
            robustness_score=1.0,
            semantic_score=1.0,
            stability_score=1.0,
            hallucination_fraction=0.0,
        )
        assert abs(ets - 100.0) < 1e-4, f"Expected ETS=100, got {ets}"

    def test_completeness_converse_imperfect_yields_less_than_100(self):
        """Any deviation from perfect scores must yield ETS < 100."""
        # Drop consistency slightly
        ets = compute_ets(1.0 - 1e-4, 1.0, 1.0, 1.0, 0.0)
        assert ets < 100.0, f"Expected ETS < 100 for imperfect inputs, got {ets}"

    def test_completeness_nonzero_hallucination_reduces_ets(self):
        """Any non-zero hallucination fraction must reduce ETS below 100."""
        ets_perfect = compute_ets(1.0, 1.0, 1.0, 1.0, 0.0)
        ets_hallucinated = compute_ets(1.0, 1.0, 1.0, 1.0, 0.01)
        assert ets_hallucinated < ets_perfect


class TestMonotonicityAxiom:
    """Axiom 2: Increasing any component score cannot decrease ETS."""

    def _base_ets(self) -> float:
        return compute_ets(0.5, 0.5, 0.5, 0.5, 0.3)

    def test_monotonicity_consistency(self):
        """Increasing consistency score must not decrease ETS."""
        base = compute_ets(0.5, 0.5, 0.5, 0.5, 0.3)
        higher = compute_ets(0.8, 0.5, 0.5, 0.5, 0.3)
        assert higher >= base

    def test_monotonicity_robustness(self):
        """Increasing robustness score must not decrease ETS."""
        base = compute_ets(0.5, 0.5, 0.5, 0.5, 0.3)
        higher = compute_ets(0.5, 0.8, 0.5, 0.5, 0.3)
        assert higher >= base

    def test_monotonicity_semantic(self):
        """Increasing semantic score must not decrease ETS."""
        base = compute_ets(0.5, 0.5, 0.5, 0.5, 0.3)
        higher = compute_ets(0.5, 0.5, 0.9, 0.5, 0.3)
        assert higher >= base

    def test_monotonicity_stability(self):
        """Increasing stability score must not decrease ETS."""
        base = compute_ets(0.5, 0.5, 0.5, 0.5, 0.3)
        higher = compute_ets(0.5, 0.5, 0.5, 0.9, 0.3)
        assert higher >= base

    def test_monotonicity_hallucination_decreases_ets(self):
        """Increasing hallucination fraction must not increase ETS."""
        base = compute_ets(0.5, 0.5, 0.5, 0.5, 0.3)
        worse = compute_ets(0.5, 0.5, 0.5, 0.5, 0.8)
        assert worse <= base

    def test_monotonicity_strict_positive_derivative_for_components(self):
        """Verify partial derivative dETS/dC > 0."""
        eps = 0.01
        low = compute_ets(0.4, 0.5, 0.5, 0.5, 0.0)
        high = compute_ets(0.4 + eps, 0.5, 0.5, 0.5, 0.0)
        # Since weights are positive, ETS must strictly increase
        assert high > low, "ETS should strictly increase with consistency"


class TestNullityAxiom:
    """Axiom 3: ETS approaches 0 when H=1 and positive components are minimal."""

    def test_nullity_full_hallucination_with_zero_positives(self):
        """ETS must clamp to 0 when hallucination=1 and all other scores=0."""
        ets = compute_ets(
            consistency_score=0.0,
            robustness_score=0.0,
            semantic_score=0.0,
            stability_score=0.0,
            hallucination_fraction=1.0,
        )
        assert ets == 0.0, f"Expected ETS=0, got {ets}"

    def test_nullity_near_full_hallucination_gives_low_ets(self):
        """Near-full hallucination with moderate positive scores yields REJECT level."""
        ets = compute_ets(0.2, 0.2, 0.2, 0.2, 0.95)
        assert ets < config.ETS_THRESHOLD_LOW, \
            f"Expected REJECT-level ETS, got {ets}"

    def test_nullity_ets_clamped_at_zero(self):
        """ETS is always non-negative (never below 0)."""
        ets = compute_ets(0.0, 0.0, 0.0, 0.0, 1.0)
        assert ets >= 0.0


class TestSymmetryAxiom:
    """Axiom 4: ETS is invariant to the ordering of explanation methods."""

    def test_symmetry_iou_matrix_is_symmetric(self):
        """IoU matrix from ConsistencyAnalyzer must be symmetric."""
        from metaxai.engines.consistency_analyzer import ConsistencyAnalyzer
        from metaxai.schemas import ExplanationMap

        np.random.seed(42)
        maps = [
            ExplanationMap(
                method="shap",
                attribution=np.random.rand(8, 8).astype(np.float32),
                top_k_features=list(range(10)),
                computation_time_ms=100.0,
            ),
            ExplanationMap(
                method="lime",
                attribution=np.random.rand(8, 8).astype(np.float32),
                top_k_features=list(range(10)),
                computation_time_ms=200.0,
            ),
        ]

        analyzer = ConsistencyAnalyzer()
        report = analyzer.analyze(maps)

        assert np.allclose(report.iou_matrix, report.iou_matrix.T, atol=1e-6), \
            "IoU matrix must be symmetric"
        assert np.allclose(report.cosine_matrix, report.cosine_matrix.T, atol=1e-6), \
            "Cosine matrix must be symmetric"

    def test_symmetry_score_invariant_to_method_order(self):
        """Swapping method order must not change pairwise_agreement_score."""
        from metaxai.engines.consistency_analyzer import ConsistencyAnalyzer
        from metaxai.schemas import ExplanationMap

        # Use linearly-spaced arrays to guarantee well-defined Spearman (no constant inputs)
        h, w = 8, 8
        attr_a = np.linspace(0.0, 1.0, h * w).reshape(h, w).astype(np.float32)
        attr_b = np.linspace(0.1, 0.9, h * w).reshape(h, w).astype(np.float32)
        # Reverse b so they have different orderings (non-trivial correlation)
        attr_b = attr_b[::-1, :].copy()

        map_a = ExplanationMap("shap", attr_a, list(range(10)), 100.0)
        map_b = ExplanationMap("lime", attr_b, list(range(10)), 100.0)

        analyzer = ConsistencyAnalyzer()
        report_ab = analyzer.analyze([map_a, map_b])
        report_ba = analyzer.analyze([map_b, map_a])

        assert abs(report_ab.pairwise_agreement_score - report_ba.pairwise_agreement_score) < 1e-6, \
            f"pairwise_agreement_score must be invariant to method ordering. " \
            f"Got {report_ab.pairwise_agreement_score} vs {report_ba.pairwise_agreement_score}"

    def test_symmetry_ets_deterministic_same_inputs(self):
        """ETS must produce identical results for identical inputs."""
        ets_1 = compute_ets(0.7, 0.6, 0.8, 0.75, 0.1)
        ets_2 = compute_ets(0.7, 0.6, 0.8, 0.75, 0.1)
        assert ets_1 == ets_2, "ETS must be deterministic"


class TestTrustLevelClassification:
    """Test trust level classification thresholds."""

    def test_high_trust_threshold(self):
        assert classify_trust_level(75.0) == "HIGH"
        assert classify_trust_level(100.0) == "HIGH"

    def test_medium_trust_threshold(self):
        assert classify_trust_level(50.0) == "MEDIUM"
        assert classify_trust_level(74.9) == "MEDIUM"

    def test_low_trust_threshold(self):
        assert classify_trust_level(25.0) == "LOW"
        assert classify_trust_level(49.9) == "LOW"

    def test_reject_threshold(self):
        assert classify_trust_level(0.0) == "REJECT"
        assert classify_trust_level(24.9) == "REJECT"
