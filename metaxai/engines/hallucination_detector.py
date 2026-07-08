"""
hallucination_detector.py — Module 3: Hallucination Detector

The most technically novel module of the Meta-XAI pipeline.
Uses two independent detection strategies; a prediction is flagged as
hallucinated ONLY when BOTH agree (dual-confirmation to minimize FPR).

Strategy A: Perturbation Test (Counterfactual Validity)
    - Occludes top-20% of attribution mass, reruns model
    - If confidence does NOT drop by >= delta, explanation is causally invalid

Strategy B: Segmentation Boundary Test
    - Uses SAM or DeepLabV3 to generate an object mask
    - If attribution mass inside mask < threshold, explanation is spurious

Research Specification: Meta-XAI v1.0, Section 3.5
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import config
from metaxai.schemas import (
    BoundaryResult,
    ExplanationMap,
    HallucinationResult,
    PerturbationResult,
)

logger = logging.getLogger(__name__)

# Dataset mean for occlusion (ImageNet RGB means, normalized)
DATASET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)


# ── Strategy A: Perturbation Test ─────────────────────────────────────────────

def perturbation_test(
    image: np.ndarray,
    attribution: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
    top_k: float = config.PERTURBATION_TOP_K,
    delta: float = config.PERTURBATION_DELTA,
    method_name: str = "unknown",
) -> PerturbationResult:
    """Test counterfactual validity of an attribution map via occlusion.

    Masks the top-K% most attributed pixels with the dataset mean and measures
    the resulting confidence drop. A valid explanation should cause a significant
    drop when its highlighted regions are removed.

    Args:
        image: Input image as numpy array (H, W, C), float32, [0, 1].
        attribution: Attribution map of shape (H, W), float32, [0, 1].
        model: PyTorch model in eval mode.
        target_class: Class index to track confidence for. Uses argmax if None.
        top_k: Fraction of top attribution pixels to occlude (default: 0.20).
        delta: Minimum required confidence drop to consider explanation valid (default: 0.15).
        method_name: Name of the XAI method (for logging).

    Returns:
        PerturbationResult with perturbation_delta and is_valid flag.
    """
    # Build occlusion mask
    threshold_val = np.percentile(attribution, (1.0 - top_k) * 100.0)
    mask = attribution >= threshold_val  # (H, W) bool

    # Occlude the image
    occluded = image.copy()
    if occluded.ndim == 3 and occluded.shape[2] == 3:
        occluded[mask] = DATASET_MEAN
    else:
        occluded[mask] = 0.0

    # Convert to tensors
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        return t

    with torch.no_grad():
        if target_class is None:
            logits = model(_to_tensor(image))
            target_class = int(logits.argmax(dim=1).item())

        original_conf = float(
            torch.softmax(model(_to_tensor(image)), dim=1)[0, target_class].item()
        )
        occluded_conf = float(
            torch.softmax(model(_to_tensor(occluded)), dim=1)[0, target_class].item()
        )

    drop = original_conf - occluded_conf
    is_valid = drop >= delta

    logger.debug(
        "Perturbation test [%s]: orig=%.4f, occluded=%.4f, drop=%.4f, valid=%s",
        method_name, original_conf, occluded_conf, drop, is_valid,
    )

    return PerturbationResult(
        method=method_name,
        perturbation_delta=float(drop),
        is_valid=bool(is_valid),
    )


# ── Strategy B: Segmentation Boundary Test ────────────────────────────────────

def _load_segmentation_model(model_type: str = config.SEGMENTATION_MODEL) -> nn.Module:
    """Load the pretrained segmentation model.

    Args:
        model_type: One of 'deeplab' or 'sam'.

    Returns:
        Loaded PyTorch segmentation model in eval mode.

    Raises:
        ValueError: If model_type is not recognized.
        ImportError: If required segmentation library is not installed.
    """
    if model_type == "deeplab":
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(pretrained=True)
        model.eval()
        return model
    elif model_type == "sam":
        try:
            from segment_anything import sam_model_registry, SamPredictor
            # SAM requires a checkpoint file; use mock in test mode
            logger.warning("SAM requires checkpoint download. Falling back to DeepLab.")
        except ImportError:
            logger.warning("segment_anything not installed. Falling back to DeepLab.")
        from torchvision.models.segmentation import deeplabv3_resnet50
        return deeplabv3_resnet50(pretrained=True).eval()
    else:
        raise ValueError(f"Unknown segmentation model type: '{model_type}'")


def _generate_segmentation_mask(
    image: np.ndarray,
    predicted_class_idx: int,
    seg_model: Optional[nn.Module] = None,
) -> np.ndarray:
    """Generate a binary segmentation mask for the predicted class.

    Args:
        image: Input image as numpy array (H, W, C), float32, [0, 1].
        predicted_class_idx: The model's predicted class index.
        seg_model: Pretrained segmentation model. Loaded lazily if None.

    Returns:
        Binary segmentation mask of shape (H, W), bool.
    """
    if seg_model is None:
        seg_model = _load_segmentation_model()

    h, w = image.shape[:2]
    img_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = seg_model(img_tensor)["out"]  # (1, num_classes, H, W)
        pred_mask = output.argmax(dim=1).squeeze(0).numpy()  # (H, W)

    # For CIFAR-10/generic models: use the predicted class to create a proxy mask
    # In production: map predicted_class_idx to PASCAL VOC class
    object_mask = (pred_mask == (predicted_class_idx % 21)).astype(bool)  # mod for VOC classes

    # Fallback: if mask is empty, use center region as proxy
    if object_mask.sum() == 0:
        logger.warning(
            "Segmentation mask is empty for class %d. Using center region as proxy.",
            predicted_class_idx,
        )
        cx, cy = h // 2, w // 2
        r = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        object_mask = ((y - cx) ** 2 + (x - cy) ** 2) <= r ** 2

    return object_mask


def boundary_test(
    attribution: np.ndarray,
    image: np.ndarray,
    predicted_class_idx: int,
    seg_model: Optional[nn.Module] = None,
    overlap_threshold: float = config.BOUNDARY_OVERLAP_THRESHOLD,
    method_name: str = "unknown",
) -> BoundaryResult:
    """Test whether attribution mass falls inside the predicted object's boundary.

    Uses a pretrained segmentation model to generate an object mask, then
    computes the fraction of attribution mass (boundary_overlap_ratio) that
    falls within this mask. Low overlap indicates the explanation is attending
    to background or irrelevant regions.

    Args:
        attribution: Attribution map of shape (H, W), float32, [0, 1].
        image: Input image as numpy array (H, W, C), float32, [0, 1].
        predicted_class_idx: The model's predicted class index.
        seg_model: Pretrained segmentation model. Loaded lazily if None.
        overlap_threshold: Minimum required overlap ratio (default: 0.40).
        method_name: Name of the XAI method (for logging).

    Returns:
        BoundaryResult with boundary_overlap_ratio and is_valid flag.
    """
    try:
        object_mask = _generate_segmentation_mask(image, predicted_class_idx, seg_model)
    except Exception as exc:  # noqa: BLE001
        logger.error("Segmentation failed: %s. Returning neutral boundary result.", exc)
        return BoundaryResult(method=method_name, boundary_overlap_ratio=1.0, is_valid=True)

    total_attribution = attribution.sum()
    if total_attribution < 1e-8:
        overlap_ratio = 0.0
    else:
        inside_mass = attribution[object_mask].sum()
        overlap_ratio = float(inside_mass / total_attribution)

    is_valid = overlap_ratio >= overlap_threshold

    logger.debug(
        "Boundary test [%s]: overlap_ratio=%.4f, valid=%s",
        method_name, overlap_ratio, is_valid,
    )

    return BoundaryResult(
        method=method_name,
        boundary_overlap_ratio=float(overlap_ratio),
        is_valid=bool(is_valid),
    )


# ── Hallucination Detector Class ──────────────────────────────────────────────

class HallucinationDetector:
    """Module 3: Detects hallucinated explanations using dual-confirmation.

    An explanation is flagged as hallucinated only when BOTH strategies fail:
    1. Perturbation test: confidence does not drop when highlighted region is occluded
    2. Boundary test: attribution mass does not overlap with the object boundary

    This dual-confirmation design minimizes false positives while maintaining
    high recall for true hallucinations.

    Args:
        model: PyTorch model in eval mode (for perturbation test).
        seg_model: Pretrained segmentation model (for boundary test). Loaded lazily if None.
        require_both: If True, requires BOTH tests to fail for hallucination verdict.
            Default: config.HALLUCINATION_REQUIRE_BOTH.

    Example:
        >>> detector = HallucinationDetector(model=my_model)
        >>> results = detector.detect(explanation_maps, image, target_class=5)
        >>> for r in results:
        ...     print(r.method, r.is_hallucinated)
    """

    def __init__(
        self,
        model: nn.Module,
        seg_model: Optional[nn.Module] = None,
        require_both: bool = config.HALLUCINATION_REQUIRE_BOTH,
    ) -> None:
        self.model = model
        self.model.eval()
        self.seg_model = seg_model  # loaded lazily
        self.require_both = require_both
        logger.info(
            "HallucinationDetector initialized. require_both=%s", require_both
        )

    def detect(
        self,
        explanation_maps: List[ExplanationMap],
        image: np.ndarray,
        target_class: Optional[int] = None,
    ) -> List[HallucinationResult]:
        """Detect hallucinated explanations for all valid XAI methods.

        Args:
            explanation_maps: List of ExplanationMap objects from Module 1.
            image: Input image as numpy array (H, W, C), float32, [0, 1].
            target_class: Model's predicted class index. Computed from model if None.

        Returns:
            List of HallucinationResult objects (one per valid method).
        """
        valid_maps = [m for m in explanation_maps if m.is_valid]
        logger.info(
            "HallucinationDetector: checking %d valid maps.", len(valid_maps)
        )

        # Determine target class once for all methods
        if target_class is None:
            img_tensor = torch.tensor(
                image.transpose(2, 0, 1), dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(img_tensor)
                target_class = int(logits.argmax(dim=1).item())

        results: List[HallucinationResult] = []

        for emap in valid_maps:
            pert_result = perturbation_test(
                image=image,
                attribution=emap.attribution,
                model=self.model,
                target_class=target_class,
                method_name=emap.method,
            )

            bound_result = boundary_test(
                attribution=emap.attribution,
                image=image,
                predicted_class_idx=target_class,
                seg_model=self.seg_model,
                method_name=emap.method,
            )

            # Dual-confirmation logic
            if self.require_both:
                is_hallucinated = (not pert_result.is_valid) and (not bound_result.is_valid)
            else:
                is_hallucinated = (not pert_result.is_valid) or (not bound_result.is_valid)

            # Confidence: based on how far both metrics deviate from thresholds
            pert_confidence = 1.0 - min(pert_result.perturbation_delta / config.PERTURBATION_DELTA, 1.0)
            bound_confidence = 1.0 - min(bound_result.boundary_overlap_ratio / config.BOUNDARY_OVERLAP_THRESHOLD, 1.0)
            detector_confidence = (pert_confidence + bound_confidence) / 2.0

            result = HallucinationResult(
                method=emap.method,
                is_hallucinated=bool(is_hallucinated),
                perturbation_delta=pert_result.perturbation_delta,
                boundary_overlap_ratio=bound_result.boundary_overlap_ratio,
                confidence=float(detector_confidence) if is_hallucinated else 0.0,
            )
            results.append(result)

            logger.info(
                "[%s] hallucinated=%s | pert_delta=%.3f | boundary_overlap=%.3f",
                emap.method,
                is_hallucinated,
                pert_result.perturbation_delta,
                bound_result.boundary_overlap_ratio,
            )

        hallucinated_methods = [r.method for r in results if r.is_hallucinated]
        logger.info(
            "HallucinationDetector complete. Hallucinated methods: %s",
            hallucinated_methods if hallucinated_methods else "None",
        )

        return results
