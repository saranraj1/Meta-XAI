"""
explanation_generator.py — Module 1: Explanation Generator Engine

This module wraps SHAP, LIME, Captum (Integrated Gradients, Grad-CAM),
and optionally Attention Maps for transformer-based models.

It is a thin adapter layer — it does NOT modify the underlying XAI methods.
Its sole responsibility is normalizing outputs into the ExplanationMap schema.

Research Specification: Meta-XAI v1.0, Section 3.3
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import config
from metaxai.schemas import ExplanationMap

logger = logging.getLogger(__name__)


# ── Internal Utilities ─────────────────────────────────────────────────────────

def _normalize_attribution(attribution: np.ndarray) -> np.ndarray:
    """Min-max normalize an attribution map to [0, 1].

    Args:
        attribution: Raw attribution map of any shape and range.

    Returns:
        Normalized attribution map in [0, 1] of the same shape.
    """
    min_val = attribution.min()
    max_val = attribution.max()
    if max_val - min_val < 1e-8:
        # All values are effectively equal — return uniform map
        return np.zeros_like(attribution, dtype=np.float32)
    return ((attribution - min_val) / (max_val - min_val)).astype(np.float32)


def _get_top_k_features(attribution: np.ndarray, k: int = config.TOP_K_FEATURES) -> List[int]:
    """Return flattened indices of the top-K attribution pixels.

    Args:
        attribution: 2D attribution map of shape (H, W).
        k: Number of top features to return.

    Returns:
        List of flattened pixel indices corresponding to the k highest attributions.
    """
    flat = attribution.flatten()
    k = min(k, len(flat))
    return np.argsort(flat)[-k:][::-1].tolist()


def _to_numpy_image(image: Image.Image | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert various image formats to a numpy array (H, W, C), float32, [0,1].

    Args:
        image: Input image as PIL Image, numpy array, or PyTorch tensor.

    Returns:
        Numpy array of shape (H, W, C), float32, values in [0, 1].

    Raises:
        TypeError: If the input type is not recognized.
    """
    if isinstance(image, Image.Image):
        return np.array(image).astype(np.float32) / 255.0
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]  # remove batch dim
        if arr.shape[0] in (1, 3):  # CHW → HWC
            arr = arr.transpose(1, 2, 0)
        return arr.astype(np.float32)
    if isinstance(image, np.ndarray):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


# ── SHAP Wrapper ───────────────────────────────────────────────────────────────

def _run_shap(
    image_np: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Compute SHAP DeepExplainer attribution for a single image.

    Args:
        image_np: Image as numpy array (H, W, C), float32, [0, 1].
        model: PyTorch model in eval mode.
        target_class: Target class index. If None, uses argmax of model output.

    Returns:
        Attribution map of shape (H, W), float32.

    Raises:
        ImportError: If the `shap` library is not installed.
        RuntimeError: If SHAP computation fails.
    """
    import shap  # lazy import — not required unless method is enabled

    # Create background dataset from dataset mean (simplified: random noise)
    background = np.random.randn(config.SHAP_BACKGROUND_SAMPLES, *image_np.shape).astype(np.float32)
    bg_tensor = torch.tensor(background.transpose(0, 3, 1, 2))  # NHWC → NCHW

    image_tensor = torch.tensor(image_np.transpose(2, 0, 1)).unsqueeze(0)  # HWC → 1CHW

    explainer = shap.DeepExplainer(model, bg_tensor)
    shap_values = explainer.shap_values(image_tensor)

    if isinstance(shap_values, list):
        if target_class is None:
            with torch.no_grad():
                logits = model(image_tensor)
                target_class = logits.argmax(dim=1).item()
        attribution = shap_values[target_class][0]  # (C, H, W)
    else:
        attribution = shap_values[0]

    if attribution.ndim == 3:
        attribution = np.abs(attribution).mean(axis=0)  # collapse channels

    return attribution.astype(np.float32)


# ── LIME Wrapper ───────────────────────────────────────────────────────────────

def _run_lime(
    image_np: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Compute LIME attribution for a single image.

    Args:
        image_np: Image as numpy array (H, W, C), float32, [0, 1].
        model: PyTorch model in eval mode.
        target_class: Target class index. If None, uses argmax of model output.

    Returns:
        Attribution map of shape (H, W), float32.

    Raises:
        ImportError: If the `lime` library is not installed.
    """
    from lime import lime_image  # lazy import
    from skimage.segmentation import mark_boundaries

    def predict_fn(images: np.ndarray) -> np.ndarray:
        """Prediction function compatible with LIME's interface."""
        tensors = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32)
        with torch.no_grad():
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    image_uint8 = (image_np * 255).astype(np.uint8)
    explainer = lime_image.LimeImageExplainer(random_state=config.RANDOM_SEED)
    explanation = explainer.explain_instance(
        image_uint8,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=config.LIME_NUM_SAMPLES,
        random_seed=config.RANDOM_SEED,
    )

    if target_class is None:
        target_class = explanation.top_labels[0]

    _, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=False,
        num_features=config.TOP_K_FEATURES,
        hide_rest=False,
    )
    return mask.astype(np.float32)


# ── Grad-CAM Wrapper ───────────────────────────────────────────────────────────

def _run_gradcam(
    image_np: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Compute Grad-CAM attribution using Captum.

    Args:
        image_np: Image as numpy array (H, W, C), float32, [0, 1].
        model: PyTorch model in eval mode.
        target_class: Target class index. If None, uses argmax of model output.

    Returns:
        Attribution map of shape (H, W), float32, upsampled to input resolution.

    Raises:
        ImportError: If `captum` is not installed.
        AttributeError: If model does not have the configured target layer.
    """
    from captum.attr import LayerGradCam  # lazy import

    image_tensor = torch.tensor(image_np.transpose(2, 0, 1)).unsqueeze(0)
    image_tensor.requires_grad_(True)

    # Navigate to configured target layer (e.g., 'layer4' for ResNet)
    target_layer = None
    for name, module in model.named_modules():
        if name == config.GRADCAM_TARGET_LAYER:
            target_layer = module
            break
    if target_layer is None:
        raise AttributeError(
            f"Target layer '{config.GRADCAM_TARGET_LAYER}' not found in model."
        )

    if target_class is None:
        with torch.no_grad():
            logits = model(image_tensor.detach())
            target_class = logits.argmax(dim=1).item()

    gc = LayerGradCam(model, target_layer)
    attribution = gc.attribute(image_tensor, target=target_class)
    attribution = attribution.detach().cpu().numpy()[0, 0]  # (1, 1, h', w') → (h', w')

    # Upsample to input resolution
    import torch.nn.functional as F
    h, w = image_np.shape[:2]
    att_tensor = torch.tensor(attribution).unsqueeze(0).unsqueeze(0)
    att_upsampled = F.interpolate(att_tensor, size=(h, w), mode="bilinear", align_corners=False)
    return att_upsampled.squeeze().numpy().astype(np.float32)


# ── Integrated Gradients Wrapper ───────────────────────────────────────────────

def _run_intgrad(
    image_np: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Compute Integrated Gradients attribution using Captum.

    Args:
        image_np: Image as numpy array (H, W, C), float32, [0, 1].
        model: PyTorch model in eval mode.
        target_class: Target class index. If None, uses argmax of model output.

    Returns:
        Attribution map of shape (H, W), float32.

    Raises:
        ImportError: If `captum` is not installed.
    """
    from captum.attr import IntegratedGradients  # lazy import

    image_tensor = torch.tensor(image_np.transpose(2, 0, 1)).unsqueeze(0)
    baseline = torch.zeros_like(image_tensor)

    if target_class is None:
        with torch.no_grad():
            logits = model(image_tensor)
            target_class = logits.argmax(dim=1).item()

    ig = IntegratedGradients(model)
    attribution = ig.attribute(
        image_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=config.INTGRAD_STEPS,
        return_convergence_delta=False,
    )

    attribution = attribution.detach().cpu().numpy()[0]  # (C, H, W)
    attribution = np.abs(attribution).mean(axis=0)       # collapse channels → (H, W)
    return attribution.astype(np.float32)


# ── Method Registry ────────────────────────────────────────────────────────────

_METHOD_REGISTRY: Dict[str, Callable] = {
    "shap": _run_shap,
    "lime": _run_lime,
    "gradcam": _run_gradcam,
    "intgrad": _run_intgrad,
}


# ── Public API ─────────────────────────────────────────────────────────────────

class ExplanationGeneratorEngine:
    """Module 1: Generates and normalizes attribution maps from multiple XAI methods.

    This engine wraps SHAP, LIME, Grad-CAM, and Integrated Gradients, normalizing
    their outputs into the canonical ExplanationMap schema. It does not modify the
    underlying XAI algorithms.

    Args:
        model: PyTorch model in eval mode.
        methods: List of XAI method names to run. Defaults to config.ENABLED_METHODS.
        target_class: Target class index for explanation. If None, uses model argmax.

    Example:
        >>> engine = ExplanationGeneratorEngine(model=my_model, methods=['shap', 'gradcam'])
        >>> maps = engine.generate(image)
        >>> for m in maps:
        ...     print(m.method, m.attribution.shape)
    """

    def __init__(
        self,
        model: nn.Module,
        methods: Optional[List[str]] = None,
        target_class: Optional[int] = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.methods = methods if methods is not None else config.ENABLED_METHODS
        self.target_class = target_class
        logger.info("ExplanationGeneratorEngine initialized with methods: %s", self.methods)

    def generate(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
    ) -> List[ExplanationMap]:
        """Generate attribution maps for all configured XAI methods.

        Args:
            image: Input image as PIL Image, numpy array (H,W,C), or PyTorch tensor.

        Returns:
            List of ExplanationMap objects (one per method). Invalid maps are included
            with is_valid=False so downstream modules can handle them gracefully.
        """
        image_np = _to_numpy_image(image)
        results: List[ExplanationMap] = []

        for method_name in self.methods:
            if method_name not in _METHOD_REGISTRY:
                logger.warning("Method '%s' not in registry. Skipping.", method_name)
                continue

            logger.debug("Running XAI method: %s", method_name)
            t_start = time.perf_counter()

            try:
                raw_attr = _METHOD_REGISTRY[method_name](
                    image_np, self.model, self.target_class
                )
                attr_normalized = _normalize_attribution(raw_attr)
                top_k = _get_top_k_features(attr_normalized)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                explanation_map = ExplanationMap(
                    method=method_name,
                    attribution=attr_normalized,
                    top_k_features=top_k,
                    computation_time_ms=elapsed_ms,
                    metadata={"target_class": self.target_class},
                    is_valid=True,
                )
                logger.info(
                    "Method '%s' completed in %.1f ms", method_name, elapsed_ms
                )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Method '%s' failed with exception: %s. Returning empty map.",
                    method_name,
                    exc,
                    exc_info=True,
                )
                explanation_map = ExplanationMap.empty(method_name)

            results.append(explanation_map)

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(
            "ExplanationGeneratorEngine: %d/%d methods succeeded.",
            valid_count,
            len(results),
        )
        return results
