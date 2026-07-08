"""
metaxai/__init__.py — Meta-XAI Package Init

Exposes the top-level public API for the Meta-XAI library.
"""

from metaxai.pipeline import MetaXAIPipeline
from metaxai.schemas import (
    ConsistencyReport,
    ExplanationAuditReport,
    ExplanationMap,
    HallucinationResult,
    SemanticCoherenceResult,
)

__version__ = "1.0.0"
__author__ = "Meta-XAI Research Team"
__all__ = [
    "MetaXAIPipeline",
    "ExplanationMap",
    "ConsistencyReport",
    "HallucinationResult",
    "SemanticCoherenceResult",
    "ExplanationAuditReport",
]
