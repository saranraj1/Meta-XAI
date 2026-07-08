"""
test_consistency_analyzer.py — Unit Tests for Module 2: Consistency Analyzer
"""
import numpy as np
import pytest
from metaxai.engines.consistency_analyzer import ConsistencyAnalyzer
from metaxai.schemas import ConsistencyReport, ExplanationMap


def make_map(method: str, pattern: str = "random", seed: int = 0) -> ExplanationMap:
    """Helper: create a synthetic ExplanationMap for testing."""
    rng = np.random.RandomState(seed)
    if pattern == "zeros":
        attr = np.zeros((16, 16), dtype=np.float32)
    elif pattern == "ones":
        attr = np.ones((16, 16), dtype=np.float32)
    elif pattern == "top_left":
        attr = np.zeros((16, 16), dtype=np.float32)
        attr[:8, :8] = 1.0
    elif pattern == "bottom_right":
        attr = np.zeros((16, 16), dtype=np.float32)
        attr[8:, 8:] = 1.0
    else:
        attr = rng.rand(16, 16).astype(np.float32)
    return ExplanationMap(
        method=method,
        attribution=attr,
        top_k_features=list(range(10)),
        computation_time_ms=50.0,
    )


class TestConsistencyAnalyzerBasics:

    def test_single_map_returns_identity_matrices(self):
        """Single map should return 1x1 identity matrices and score=1.0."""
        analyzer = ConsistencyAnalyzer()
        maps = [make_map("shap")]
        report = analyzer.analyze(maps)
        assert report.pairwise_agreement_score == 1.0
        assert report.iou_matrix.shape == (1, 1)
        assert report.iou_matrix[0, 0] == 1.0

    def test_identical_maps_yield_score_1(self):
        """Two identical attribution maps should yield perfect consistency."""
        analyzer = ConsistencyAnalyzer()
        attr = np.ones((16, 16), dtype=np.float32) * 0.7
        m1 = ExplanationMap("shap", attr.copy(), list(range(10)), 100.0)
        m2 = ExplanationMap("lime", attr.copy(), list(range(10)), 100.0)
        report = analyzer.analyze([m1, m2])
        assert report.pairwise_agreement_score > 0.95, \
            f"Identical maps should be nearly perfectly consistent, got {report.pairwise_agreement_score}"

    def test_orthogonal_maps_yield_low_score(self):
        """Maps with completely non-overlapping attributions should have low consistency."""
        analyzer = ConsistencyAnalyzer()
        m1 = make_map("shap", pattern="top_left")
        m2 = make_map("lime", pattern="bottom_right")
        report = analyzer.analyze([m1, m2])
        # IoU should be 0 for non-overlapping maps
        assert report.iou_matrix[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert report.pairwise_agreement_score < 0.5

    def test_report_has_correct_method_names(self):
        """ConsistencyReport must record method names in order."""
        analyzer = ConsistencyAnalyzer()
        m1 = make_map("shap")
        m2 = make_map("gradcam", seed=5)
        m3 = make_map("lime", seed=7)
        report = analyzer.analyze([m1, m2, m3])
        assert report.method_names == ["shap", "gradcam", "lime"]

    def test_report_matrices_are_symmetric(self):
        """All three metric matrices must be symmetric."""
        analyzer = ConsistencyAnalyzer()
        maps = [make_map(m, seed=i) for i, m in enumerate(["shap", "lime", "gradcam"])]
        report = analyzer.analyze(maps)
        assert np.allclose(report.iou_matrix, report.iou_matrix.T, atol=1e-5)
        assert np.allclose(report.cosine_matrix, report.cosine_matrix.T, atol=1e-5)
        assert np.allclose(report.rank_corr_matrix, report.rank_corr_matrix.T, atol=1e-5)

    def test_invalid_maps_are_filtered(self):
        """Invalid ExplanationMaps must be filtered before analysis."""
        from metaxai.schemas import ExplanationMap
        valid = make_map("shap")
        invalid = ExplanationMap.empty("lime")
        analyzer = ConsistencyAnalyzer()
        # Should run with only 1 valid map and return score=1.0
        report = analyzer.analyze([valid, invalid])
        assert report.pairwise_agreement_score == 1.0

    def test_pairwise_score_in_range(self):
        """pairwise_agreement_score must always be in [0, 1]."""
        analyzer = ConsistencyAnalyzer()
        for seed in range(10):
            maps = [make_map("shap", seed=seed), make_map("lime", seed=seed + 100)]
            report = analyzer.analyze(maps)
            assert 0.0 <= report.pairwise_agreement_score <= 1.0

    def test_raises_on_zero_valid_maps(self):
        """Must raise ValueError when no valid maps are provided."""
        analyzer = ConsistencyAnalyzer()
        invalid_maps = [ExplanationMap.empty("shap"), ExplanationMap.empty("lime")]
        with pytest.raises(ValueError, match="at least 1 valid"):
            analyzer.analyze(invalid_maps)


class TestConsistencyMetrics:

    def test_iou_perfect_overlap(self):
        """IoU should be 1.0 for identical binary masks."""
        from metaxai.engines.consistency_analyzer import ConsistencyAnalyzer
        analyzer = ConsistencyAnalyzer()
        ones = np.ones((8, 8), dtype=np.float32)
        assert analyzer._iou(ones, ones, threshold=0.5) == pytest.approx(1.0)

    def test_iou_zero_overlap(self):
        """IoU should be 0.0 for completely disjoint masks."""
        from metaxai.engines.consistency_analyzer import ConsistencyAnalyzer
        analyzer = ConsistencyAnalyzer()
        a = np.zeros((8, 8), dtype=np.float32)
        a[:4, :] = 1.0
        b = np.zeros((8, 8), dtype=np.float32)
        b[4:, :] = 1.0
        assert analyzer._iou(a, b, threshold=0.5) == pytest.approx(0.0)

    def test_cosine_identical_vectors(self):
        """Cosine similarity should be 1.0 for identical vectors."""
        from metaxai.engines.consistency_analyzer import ConsistencyAnalyzer
        analyzer = ConsistencyAnalyzer()
        v = np.random.rand(8, 8).astype(np.float32)
        assert analyzer._cosine(v, v) == pytest.approx(1.0, abs=1e-5)
