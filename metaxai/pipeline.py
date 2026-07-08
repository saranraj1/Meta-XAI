"""
pipeline.py — Meta-XAI Pipeline Orchestrator

The MetaXAIPipeline class orchestrates all five modules in sequence:
    1. Explanation Generator Engine
    2. Consistency Analyzer
    3. Hallucination Detector
    4. Semantic Coherence Module
    5. Trust Scoring Engine

Each module receives a defined input schema and produces a defined output schema.
The pipeline is stateless between modules — no hidden state is shared.

Research Specification: Meta-XAI v1.0, Section 3.1
"""

from __future__ import annotations

import datetime
import logging
from typing import List, Optional

import numpy as np
import torch.nn as nn
from PIL import Image

import config
from metaxai.engines.consistency_analyzer import ConsistencyAnalyzer
from metaxai.engines.explanation_generator import ExplanationGeneratorEngine
from metaxai.engines.hallucination_detector import HallucinationDetector
from metaxai.engines.semantic_coherence import SemanticCoherenceModule
from metaxai.engines.trust_scorer import TrustScoringEngine
from metaxai.schemas import ExplanationAuditReport

logger = logging.getLogger(__name__)

# Configure root logger
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


class MetaXAIPipeline:
    """The Meta-XAI Trustworthiness Auditing Pipeline.

    Orchestrates all five modules to produce a complete ExplanationAuditReport
    for any image-model-label combination. This is the primary entry point for
    the Meta-XAI framework.

    Args:
        model: PyTorch classifier model in eval mode.
        methods: List of XAI methods to run. Default: config.ENABLED_METHODS.
        enable_semantic_module: Whether to run the LLM semantic module. Default: True.
        use_mock_llm: If True, uses mock LLM for offline testing. Default: False.
        enable_stability: Whether to compute stability score. Default: True.
        target_class: Fixed target class for all audits. Uses model argmax if None.

    Example:
        >>> pipeline = MetaXAIPipeline(model=my_model, methods=['shap', 'lime', 'gradcam'])
        >>> report = pipeline.audit(image, label='dog')
        >>> print(report.summary())
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        model: nn.Module,
        methods: Optional[List[str]] = None,
        enable_semantic_module: bool = True,
        use_mock_llm: bool = False,
        enable_stability: bool = True,
        target_class: Optional[int] = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.methods = methods or config.ENABLED_METHODS
        self.target_class = target_class
        self.enable_semantic_module = enable_semantic_module

        # Initialize modules
        self._explanation_engine = ExplanationGeneratorEngine(
            model=model,
            methods=self.methods,
            target_class=target_class,
        )
        self._consistency_analyzer = ConsistencyAnalyzer()
        self._hallucination_detector = HallucinationDetector(model=model)
        self._semantic_module = SemanticCoherenceModule(use_mock=use_mock_llm)
        self._trust_engine = TrustScoringEngine(model=model, enable_stability=enable_stability)

        logger.info(
            "MetaXAIPipeline v%s initialized | methods=%s | semantic=%s | mock_llm=%s",
            self.VERSION, self.methods, enable_semantic_module, use_mock_llm,
        )

    def audit(
        self,
        image: Image.Image | np.ndarray,
        label: str,
        image_id: Optional[str] = None,
    ) -> ExplanationAuditReport:
        """Run the full Meta-XAI audit pipeline on a single image.

        Executes all five modules sequentially and returns a complete
        ExplanationAuditReport with ETS, trust level, and all intermediate results.

        Args:
            image: Input image as PIL Image or numpy array (H, W, C), float32, [0, 1].
            label: The model's predicted class label string (for semantic module).
            image_id: Optional unique identifier for the image. Auto-generated if None.

        Returns:
            ExplanationAuditReport containing all audit metrics and sub-reports.

        Raises:
            ValueError: If the image format is invalid.
            RuntimeError: If a critical module fails and cannot recover.
        """
        if image_id is None:
            image_id = f"img_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

        audit_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        logger.info("=== Meta-XAI Audit START | image_id=%s | label=%s ===", image_id, label)

        # ── Step 1: Explanation Generator ─────────────────────────────────────
        logger.info("[Module 1] Running Explanation Generator Engine...")
        explanation_maps = self._explanation_engine.generate(image)

        # Convert image to numpy for downstream modules
        if isinstance(image, Image.Image):
            image_np = np.array(image).astype(np.float32) / 255.0
        else:
            image_np = image.astype(np.float32)
            if image_np.max() > 1.0:
                image_np = image_np / 255.0

        # ── Step 2: Consistency Analyzer ──────────────────────────────────────
        logger.info("[Module 2] Running Consistency Analyzer...")
        try:
            consistency_report = self._consistency_analyzer.analyze(explanation_maps)
        except ValueError as exc:
            logger.error("ConsistencyAnalyzer failed: %s. Using degenerate report.", exc)
            import numpy as np
            consistency_report = type("obj", (object,), {
                "pairwise_agreement_score": 0.0,
                "iou_matrix": np.array([[0.0]]),
                "cosine_matrix": np.array([[0.0]]),
                "rank_corr_matrix": np.array([[0.0]]),
                "method_names": [],
            })()

        # ── Step 3: Hallucination Detector ────────────────────────────────────
        logger.info("[Module 3] Running Hallucination Detector...")
        hallucination_results = self._hallucination_detector.detect(
            explanation_maps=explanation_maps,
            image=image_np,
            target_class=self.target_class,
        )

        # ── Step 4: Semantic Coherence Module ─────────────────────────────────
        logger.info("[Module 4] Running Semantic Coherence Module...")
        if self.enable_semantic_module:
            semantic_score, semantic_result = self._semantic_module.evaluate_aggregate(
                explanation_maps=explanation_maps,
                label=label,
                image_shape=image_np.shape,
            )
        else:
            semantic_score = 0.5  # neutral default when module is disabled
            semantic_result = None
            logger.info("[Module 4] Semantic module disabled. Using neutral score=0.5")

        # ── Step 5: Trust Scoring Engine ──────────────────────────────────────
        logger.info("[Module 5] Running Trust Scoring Engine...")
        ets, trust_level, robustness_score, stability_score = self._trust_engine.score(
            consistency_report=consistency_report,
            hallucination_results=hallucination_results,
            semantic_score=semantic_score,
            explanation_maps=explanation_maps,
            image=image_np,
            semantic_coherence_result=semantic_result,
        )

        # ── Assemble Final Report ──────────────────────────────────────────────
        # Get model confidence
        import torch
        img_tensor = torch.tensor(image_np.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            if self.target_class is not None:
                model_confidence = float(probs[0, self.target_class].item())
                predicted_class_idx = self.target_class
            else:
                predicted_class_idx = int(probs.argmax(dim=1).item())
                model_confidence = float(probs[0, predicted_class_idx].item())

        report = ExplanationAuditReport(
            image_id=image_id,
            predicted_class=label,
            model_confidence=model_confidence,
            explanation_maps=explanation_maps,
            consistency_report=consistency_report,
            hallucination_results=hallucination_results,
            semantic_coherence_score=semantic_score,
            semantic_coherence_result=semantic_result,
            explanation_trust_score=ets,
            trust_level=trust_level,
            robustness_score=robustness_score,
            stability_score=stability_score,
            audit_timestamp=audit_timestamp,
            pipeline_version=self.VERSION,
        )

        logger.info("=== Meta-XAI Audit COMPLETE | ETS=%.1f/100 [%s] ===", ets, trust_level)
        return report

    def audit_batch(
        self,
        images: list,
        labels: list,
        image_ids: Optional[List[str]] = None,
    ) -> List[ExplanationAuditReport]:
        """Run audit on a batch of images.

        Args:
            images: List of images (PIL Image or numpy array).
            labels: List of predicted class label strings.
            image_ids: Optional list of image IDs.

        Returns:
            List of ExplanationAuditReport objects.

        Raises:
            ValueError: If images and labels have different lengths.
        """
        if len(images) != len(labels):
            raise ValueError(
                f"images and labels must have the same length. "
                f"Got {len(images)} images and {len(labels)} labels."
            )

        ids = image_ids or [None] * len(images)
        reports = []
        for i, (img, lbl, img_id) in enumerate(zip(images, labels, ids)):
            logger.info("Batch audit: processing image %d/%d", i + 1, len(images))
            report = self.audit(img, lbl, img_id)
            reports.append(report)

        return reports
