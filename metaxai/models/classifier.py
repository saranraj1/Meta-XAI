"""
classifier.py — Model Loader Wrapper

Provides a unified interface for loading PyTorch classifier models.
Supports pretrained torchvision models and custom checkpoint loading.
"""
from __future__ import annotations
import logging
from typing import Optional
import torch
import torch.nn as nn
import config

logger = logging.getLogger(__name__)

SUPPORTED_ARCHS = {
    "resnet18": "torchvision.models.resnet18",
    "resnet50": "torchvision.models.resnet50",
    "vgg16": "torchvision.models.vgg16",
    "efficientnet_b0": "torchvision.models.efficientnet_b0",
    "vit_b_16": "torchvision.models.vit_b_16",
}


def load_classifier(
    arch: str = config.MODEL_ARCH,
    num_classes: int = config.MODEL_NUM_CLASSES,
    pretrained: bool = config.MODEL_PRETRAINED,
    checkpoint: Optional[str] = None,
) -> nn.Module:
    """Load a PyTorch classifier model.

    Args:
        arch: Model architecture name. One of the keys in SUPPORTED_ARCHS.
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        checkpoint: Optional path to a custom .pth checkpoint file.
            If provided, loads from checkpoint instead of pretrained weights.

    Returns:
        PyTorch model in eval mode.

    Raises:
        ValueError: If arch is not in SUPPORTED_ARCHS.
        FileNotFoundError: If checkpoint path does not exist.
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported architecture '{arch}'. "
            f"Supported: {list(SUPPORTED_ARCHS.keys())}"
        )

    logger.info("Loading model: arch=%s, num_classes=%d, pretrained=%s", arch, num_classes, pretrained)

    import torchvision.models as models

    model_fn = getattr(models, arch)

    if pretrained and checkpoint is None:
        try:
            # torchvision >= 0.13 uses weights= API
            from torchvision.models import get_model_weights
            weights = get_model_weights(arch).DEFAULT
            model = model_fn(weights=weights)
        except (ImportError, AttributeError):
            model = model_fn(pretrained=True)
    else:
        model = model_fn(pretrained=False)

    # Replace final classifier head for custom num_classes
    if num_classes != 1000:
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, "heads"):
            # ViT
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)

    # Load custom checkpoint
    if checkpoint is not None:
        import os
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint from: %s", checkpoint)

    model.eval()
    logger.info("Model loaded successfully. Parameters: %.2fM",
                sum(p.numel() for p in model.parameters()) / 1e6)
    return model
