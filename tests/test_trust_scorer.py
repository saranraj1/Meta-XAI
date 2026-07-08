"""
test_trust_scorer.py — Unit Tests for Module 5: Trust Scoring Engine
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from metaxai.engines.trust_scorer import TrustScoringEngine, compute_ets, classify_trust_level
from metaxai.schemas import ConsistencyReport, HallucinationResult, ExplanationMap
import config


def make_consistency_report(score: float = 0.7) -> ConsistencyReport:
    n = 2
    mat = np.array([[1.0, score], [score, 1.0]], dtype=np.float32)
    return ConsistencyReport(
        iou_matrix=mat.copy(),
        cosine_matrix=mat.copy(),
        rank_corr_matrix=mat.copy(),
        pairwise_agreement_score=score,
        method_names=["shap", "lime"],
    )


def make_hallucination_results(
    methods=("shap", "lime"),
    hallucinated=(False, False),
    deltas=(0.2, 0.3),
    overlaps=(0.6, 0.7),
) -> list:
    results = []
    for m, h, d, o in zip(methods, hallucinated, deltas, overlaps):
        results.append(HallucinationResult(
            method=m,
            is_hallucinated=h,
            perturbation_delta=d,
            boundary_overlap_ratio=o,
            confidence=0.8 if h else 0.0,
        ))
    return results


class MockModel(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 10)


class TestComputeETS:

    def test_ets_range_always_0_to_100(self):
        """ETS must always be in [0, 100]."""
        for _ in range(20):
            ets = compute_ets(
                np.random.rand(), np.random.rand(), np.random.rand(),
                np.random.rand(), np.random.rand()
            )
            assert 0.0 <= ets <= 100.0

    def test_perfect_scores_yield_100(self):
        ets = compute_ets(1.0, 1.0, 1.0, 1.0, 0.0)
        assert abs(ets - 100.0) < 1e-4

    def test_zero_scores_yield_zero(self):
        ets = compute_ets(0.0, 0.0, 0.0, 0.0, 1.0)
        assert ets == 0.0

    def test_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_ets(1.5, 0.5, 0.5, 0.5, 0.0)

    def test_weights_sum_to_1(self):
        """Verify config weights sum to 1.0 (required for Completeness axiom)."""
        total = (config.ETS_W1_CONSISTENCY + config.ETS_W2_ROBUSTNESS +
                 config.ETS_W3_SEMANTIC + config.ETS_W4_STABILITY)
        assert abs(total - 1.0) < 1e-6

    def test_hallucination_penalty_reduces_score(self):
        base = compute_ets(0.8, 0.8, 0.8, 0.8, 0.0)
        penalized = compute_ets(0.8, 0.8, 0.8, 0.8, 0.5)
        assert penalized < base

    def test_ets_interpolation(self):
        """ETS with all 0.5 scores and no hallucination should be ~50."""
        ets = compute_ets(0.5, 0.5, 0.5, 0.5, 0.0)
        assert abs(ets - 50.0) < 1.0  # should be close to 50


class TestClassifyTrustLevel:

    def test_boundaries(self):
        assert classify_trust_level(75.0) == "HIGH"
        assert classify_trust_level(74.9) == "MEDIUM"
        assert classify_trust_level(50.0) == "MEDIUM"
        assert classify_trust_level(49.9) == "LOW"
        assert classify_trust_level(25.0) == "LOW"
        assert classify_trust_level(24.9) == "REJECT"
        assert classify_trust_level(0.0) == "REJECT"

    def test_extreme_values(self):
        assert classify_trust_level(100.0) == "HIGH"
        assert classify_trust_level(0.0) == "REJECT"


class TestTrustScoringEngine:

    def test_score_returns_tuple_of_four(self):
        """score() must return (ets, trust_level, robustness, stability)."""
        model = MockModel().eval()
        engine = TrustScoringEngine(model=model)
        cr = make_consistency_report(0.7)
        hr = make_hallucination_results()
        maps = [
            ExplanationMap("shap", np.random.rand(8, 8).astype(np.float32), list(range(5)), 50.0),
            ExplanationMap("lime", np.random.rand(8, 8).astype(np.float32), list(range(5)), 50.0),
        ]
        image = np.random.rand(8, 8, 3).astype(np.float32)
        result = engine.score(cr, hr, 0.7, maps, image)
        assert len(result) == 4
        ets, trust_level, robustness, stability = result
        assert 0.0 <= ets <= 100.0
        assert trust_level in {"HIGH", "MEDIUM", "LOW", "REJECT"}
        assert 0.0 <= robustness <= 1.0
        assert 0.0 <= stability <= 1.0

    def test_all_hallucinated_yields_low_ets(self):
        """All methods hallucinated should yield REJECT-level ETS."""
        model = MockModel().eval()
        engine = TrustScoringEngine(model=model)
        cr = make_consistency_report(0.1)
        hr = make_hallucination_results(hallucinated=(True, True), deltas=(0.0, 0.0), overlaps=(0.1, 0.1))
        maps = [ExplanationMap("shap", np.zeros((8, 8), dtype=np.float32), [], 0.0)]
        image = np.zeros((8, 8, 3), dtype=np.float32)
        ets, trust_level, _, _ = engine.score(cr, hr, 0.1, maps, image)
        assert ets < config.ETS_THRESHOLD_MEDIUM, f"Expected low ETS, got {ets}"

    def test_high_consistency_no_hallucinations_yields_high_ets(self):
        """High consistency and no hallucinations should yield HIGH/MEDIUM ETS."""
        model = MockModel().eval()
        engine = TrustScoringEngine(model=model)
        cr = make_consistency_report(0.95)
        hr = make_hallucination_results(hallucinated=(False, False), deltas=(0.4, 0.5), overlaps=(0.9, 0.85))
        maps = [
            ExplanationMap("shap", np.ones((8, 8), dtype=np.float32) * 0.8, list(range(5)), 100.0),
            ExplanationMap("lime", np.ones((8, 8), dtype=np.float32) * 0.7, list(range(5)), 100.0),
        ]
        image = np.random.rand(8, 8, 3).astype(np.float32)
        ets, trust_level, _, _ = engine.score(cr, hr, 0.9, maps, image)
        assert trust_level in {"HIGH", "MEDIUM"}, f"Expected HIGH or MEDIUM, got {trust_level} (ETS={ets})"
