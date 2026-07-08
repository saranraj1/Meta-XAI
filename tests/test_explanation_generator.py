"""
test_explanation_generator.py — Unit Tests for Module 1: Explanation Generator Engine
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from metaxai.engines.explanation_generator import (
    ExplanationGeneratorEngine,
    _normalize_attribution,
    _get_top_k_features,
    _to_numpy_image,
)
from metaxai.schemas import ExplanationMap


# ── Mock Model ─────────────────────────────────────────────────────────────────

class MockClassifier(nn.Module):
    """Simple mock classifier for testing without real XAI dependencies."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


# ── Utility Function Tests ─────────────────────────────────────────────────────

class TestNormalizeAttribution:

    def test_output_in_range_0_1(self):
        """Normalized attribution must be in [0, 1]."""
        rng = np.random.RandomState(0)
        attr = rng.randn(16, 16).astype(np.float32)
        normalized = _normalize_attribution(attr)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_uniform_input_returns_zeros(self):
        """Uniform input (all same value) must return zeros."""
        attr = np.ones((8, 8), dtype=np.float32) * 5.0
        result = _normalize_attribution(attr)
        assert np.all(result == 0.0)

    def test_output_dtype_is_float32(self):
        """Output must be float32."""
        attr = np.random.rand(8, 8).astype(np.float64)
        normalized = _normalize_attribution(attr)
        assert normalized.dtype == np.float32

    def test_max_is_1_min_is_0(self):
        """After normalization, max should be 1.0 and min should be 0.0."""
        attr = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        normalized = _normalize_attribution(attr)
        assert normalized.max() == pytest.approx(1.0)
        assert normalized.min() == pytest.approx(0.0)


class TestGetTopKFeatures:

    def test_returns_correct_number_of_features(self):
        """Must return exactly k features (or fewer if attr has fewer pixels)."""
        attr = np.random.rand(8, 8).astype(np.float32)
        top_k = _get_top_k_features(attr, k=10)
        assert len(top_k) == 10

    def test_returns_valid_indices(self):
        """All returned indices must be valid flattened pixel indices."""
        h, w = 16, 16
        attr = np.random.rand(h, w).astype(np.float32)
        top_k = _get_top_k_features(attr, k=20)
        assert all(0 <= idx < h * w for idx in top_k)

    def test_top_features_have_highest_attribution(self):
        """Top-K features must correspond to the highest attribution values."""
        attr = np.zeros((4, 4), dtype=np.float32)
        attr[0, 0] = 1.0  # max value at flat index 0
        attr[3, 3] = 0.9  # second max at flat index 15
        top_k = _get_top_k_features(attr, k=2)
        assert 0 in top_k and 15 in top_k


class TestToNumpyImage:

    def test_pil_image_to_numpy(self):
        """PIL Image must be converted to float32 numpy array in [0, 1]."""
        pil_img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        result = _to_numpy_image(pil_img)
        assert result.dtype == np.float32
        assert result.shape == (32, 32, 3)
        assert result.max() <= 1.0

    def test_numpy_array_passthrough(self):
        """Numpy array in [0, 1] should pass through unchanged."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = _to_numpy_image(img)
        assert result.dtype == np.float32
        assert result.max() <= 1.0

    def test_uint8_numpy_normalized(self):
        """Uint8 numpy array must be normalized to [0, 1]."""
        img = np.ones((32, 32, 3), dtype=np.uint8) * 255
        result = _to_numpy_image(img)
        assert result.max() <= 1.0

    def test_torch_tensor_conversion(self):
        """PyTorch tensor (C, H, W) must be converted to (H, W, C) numpy float32."""
        tensor = torch.rand(3, 32, 32)
        result = _to_numpy_image(tensor)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float32

    def test_invalid_type_raises_type_error(self):
        """Non-image input types must raise TypeError."""
        with pytest.raises(TypeError):
            _to_numpy_image("not_an_image")


class TestExplanationGeneratorEngine:

    def test_engine_returns_list_of_explanation_maps(self):
        """Engine must return a list of ExplanationMap objects."""
        model = MockClassifier().eval()
        engine = ExplanationGeneratorEngine(model=model, methods=[])
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = engine.generate(img)
        assert isinstance(result, list)

    def test_unknown_method_is_skipped(self):
        """Unknown method names must be skipped without crashing."""
        model = MockClassifier().eval()
        engine = ExplanationGeneratorEngine(model=model, methods=["nonexistent_method"])
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = engine.generate(img)
        assert len(result) == 0  # skipped

    def test_empty_methods_list_returns_empty(self):
        """Empty methods list must return empty list."""
        model = MockClassifier().eval()
        engine = ExplanationGeneratorEngine(model=model, methods=[])
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = engine.generate(img)
        assert result == []

    def test_explanation_map_has_correct_schema(self):
        """ExplanationMap objects must have all required fields."""
        from metaxai.schemas import ExplanationMap as EM
        # Create a manual map to test schema validation
        attr = np.random.rand(8, 8).astype(np.float32)
        em = EM(
            method="shap",
            attribution=attr,
            top_k_features=[0, 1, 2],
            computation_time_ms=100.0,
        )
        assert em.method == "shap"
        assert em.attribution.shape == (8, 8)
        assert em.is_valid is True

    def test_empty_explanation_map_is_invalid(self):
        """ExplanationMap.empty() must produce an invalid map."""
        empty = ExplanationMap.empty("shap")
        assert empty.is_valid is False
        assert empty.method == "shap"

    def test_invalid_attribution_shape_raises(self):
        """3D attribution array must raise ValueError."""
        with pytest.raises(ValueError):
            ExplanationMap(
                method="shap",
                attribution=np.ones((3, 8, 8), dtype=np.float32),  # 3D, invalid
                top_k_features=[],
                computation_time_ms=0.0,
            )
