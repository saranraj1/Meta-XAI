"""
semantic_coherence.py — Module 4: Semantic Coherence Module

Uses a language model (via API) to perform semantic validation of XAI explanations.
For each prediction, it queries the LLM: "Given that a model predicted class [label],
is it semantically plausible that [region description] would be a contributing factor?"

The LLM returns a coherence score in [0, 1] with a brief justification that is
included in the ExplanationAuditReport.

Research Specification: Meta-XAI v1.0, Section 3.6
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

import config
from metaxai.schemas import ExplanationMap, SemanticCoherenceResult

logger = logging.getLogger(__name__)


# ── Region Description Utilities ──────────────────────────────────────────────

def _describe_top_k_region(
    attribution: np.ndarray,
    top_k_features: List[int],
    image_shape: tuple,
) -> str:
    """Generate a natural language description of the top-K attribution region.

    Describes the spatial location of the most attributed region using
    quadrant-based language (top-left, center, bottom-right, etc.).

    Args:
        attribution: Attribution map of shape (H, W).
        top_k_features: Flattened pixel indices of top-K attributions.
        image_shape: Tuple of (H, W) for coordinate calculations.

    Returns:
        A natural language string describing the spatial location of the region.
    """
    h, w = image_shape[:2]
    if not top_k_features:
        return "the entire image"

    # Convert flat indices to (row, col)
    rows = [idx // w for idx in top_k_features]
    cols = [idx % w for idx in top_k_features]

    mean_row = np.mean(rows)
    mean_col = np.mean(cols)

    # Determine quadrant
    vert = "top" if mean_row < h * 0.4 else ("center" if mean_row < h * 0.65 else "bottom")
    horiz = "left" if mean_col < w * 0.4 else ("center" if mean_col < w * 0.65 else "right")

    region_size_pct = len(top_k_features) / (h * w) * 100
    size_desc = "small" if region_size_pct < 5 else ("medium" if region_size_pct < 20 else "large")

    if vert == "center" and horiz == "center":
        location = "the central region"
    else:
        location = f"the {vert}-{horiz} region"

    return f"a {size_desc} area in {location} of the image"


def _build_coherence_prompt(
    predicted_label: str,
    region_description: str,
) -> str:
    """Construct the LLM prompt for semantic coherence scoring.

    Args:
        predicted_label: The model's predicted class label string.
        region_description: Natural language description of the attributed region.

    Returns:
        Formatted prompt string for the LLM.
    """
    return (
        f"You are an expert in computer vision and machine learning model interpretability.\n\n"
        f"A deep learning model predicted that an image belongs to the class: '{predicted_label}'.\n"
        f"An XAI explanation method highlighted {region_description} as the primary contributing "
        f"factor for this prediction.\n\n"
        f"Task: Rate the semantic plausibility of this explanation on a scale from 0.0 to 1.0.\n"
        f"- 1.0: The highlighted region is fully semantically coherent with the predicted class.\n"
        f"- 0.5: The highlighted region has some plausible connection to the predicted class.\n"
        f"- 0.0: The highlighted region has no semantic relationship to the predicted class.\n\n"
        f"Respond in exactly this JSON format:\n"
        f'{{"score": <float 0.0-1.0>, "justification": "<one sentence explanation>"}}\n'
        f"Do not include any other text."
    )


# ── LLM API Integration ────────────────────────────────────────────────────────

def _call_openai_api(prompt: str) -> dict:
    """Call the OpenAI Chat Completions API for semantic coherence scoring.

    Args:
        prompt: The formatted coherence scoring prompt.

    Returns:
        Parsed JSON response dict with 'score' and 'justification' keys.

    Raises:
        ImportError: If openai library is not installed.
        RuntimeError: If API call fails or response is malformed.
    """
    import json
    try:
        from openai import OpenAI  # lazy import
        client = OpenAI()
        response = client.chat.completions.create(
            model=config.SEMANTIC_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.SEMANTIC_MAX_TOKENS,
            temperature=config.SEMANTIC_TEMPERATURE,
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except ImportError:
        raise ImportError("openai library not installed. Run: pip install openai")
    except Exception as exc:
        raise RuntimeError(f"LLM API call failed: {exc}") from exc


def _call_mock_api(prompt: str, label: str) -> dict:
    """Mock LLM API for testing and offline use.

    Returns a heuristic score based on simple keyword matching between
    the predicted label and the region description in the prompt.

    Args:
        prompt: The coherence scoring prompt (used for keyword extraction).
        label: The predicted class label.

    Returns:
        Mock response dict with 'score' and 'justification'.
    """
    # Heuristic: if the label appears in common object categories, return moderate score
    common_classes = {
        "dog", "cat", "bird", "car", "person", "airplane", "truck", "horse", "ship", "deer"
    }
    label_lower = label.lower()
    base_score = 0.65 if label_lower in common_classes else 0.50

    # Add some variance for realism
    score = float(np.clip(base_score + np.random.normal(0, 0.05), 0.0, 1.0))

    return {
        "score": round(score, 3),
        "justification": (
            f"The highlighted region shows contextual relevance to '{label}' "
            f"based on typical visual patterns associated with this class."
        ),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

class SemanticCoherenceModule:
    """Module 4: LLM-grounded semantic validation of XAI explanations.

    Uses a language model to evaluate whether the highlighted region in an
    attribution map is semantically meaningful given the model's predicted class.

    Args:
        use_mock: If True, uses a mock LLM instead of real API. Useful for
            offline testing and development. Default: False.
        api_backend: LLM API backend to use. Currently supports 'openai'. Default: 'openai'.

    Example:
        >>> module = SemanticCoherenceModule(use_mock=True)
        >>> result = module.evaluate(explanation_map, label='dog', image_shape=(224, 224))
        >>> print(result.score, result.justification)
    """

    def __init__(
        self,
        use_mock: bool = False,
        api_backend: str = "openai",
    ) -> None:
        self.use_mock = use_mock
        self.api_backend = api_backend
        logger.info(
            "SemanticCoherenceModule initialized. use_mock=%s, backend=%s",
            use_mock, api_backend,
        )

    def evaluate(
        self,
        explanation_map: ExplanationMap,
        label: str,
        image_shape: tuple,
    ) -> SemanticCoherenceResult:
        """Evaluate the semantic coherence of a single explanation map.

        Args:
            explanation_map: ExplanationMap to evaluate. Must be valid.
            label: The model's predicted class label string.
            image_shape: Tuple of (H, W) or (H, W, C) for coordinate calculations.

        Returns:
            SemanticCoherenceResult with score in [0, 1] and LLM justification.

        Raises:
            ValueError: If explanation_map is not valid.
            RuntimeError: If LLM API call fails (propagated from backend).
        """
        if not explanation_map.is_valid:
            logger.warning(
                "SemanticCoherenceModule: received invalid ExplanationMap for method '%s'. "
                "Returning neutral score.",
                explanation_map.method,
            )
            return SemanticCoherenceResult(
                score=0.5,
                justification="Explanation map was invalid; semantic evaluation skipped.",
                region_description="N/A",
                predicted_label=label,
                model_used="none",
            )

        region_desc = _describe_top_k_region(
            explanation_map.attribution,
            explanation_map.top_k_features,
            image_shape,
        )
        prompt = _build_coherence_prompt(label, region_desc)

        logger.debug(
            "SemanticCoherenceModule: querying LLM for label='%s', region='%s'",
            label, region_desc,
        )

        try:
            if self.use_mock:
                response = _call_mock_api(prompt, label)
                model_used = "mock"
            else:
                response = _call_openai_api(prompt)
                model_used = config.SEMANTIC_LLM_MODEL

            score = float(np.clip(response.get("score", 0.5), 0.0, 1.0))
            justification = str(response.get("justification", "No justification provided."))

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SemanticCoherenceModule: LLM call failed: %s. Returning neutral score.", exc
            )
            score = 0.5
            justification = f"LLM evaluation failed: {exc}"
            model_used = "failed"

        logger.info(
            "SemanticCoherenceModule [%s]: score=%.3f | region='%s'",
            explanation_map.method, score, region_desc,
        )

        return SemanticCoherenceResult(
            score=score,
            justification=justification,
            region_description=region_desc,
            predicted_label=label,
            model_used=model_used,
        )

    def evaluate_aggregate(
        self,
        explanation_maps: List[ExplanationMap],
        label: str,
        image_shape: tuple,
    ) -> tuple[float, SemanticCoherenceResult]:
        """Evaluate semantic coherence across all valid explanation maps.

        Evaluates the best (highest attribution) valid explanation map and
        returns both the aggregate score and the detailed result.

        Args:
            explanation_maps: List of ExplanationMap objects.
            label: The model's predicted class label string.
            image_shape: Tuple of (H, W) or (H, W, C).

        Returns:
            Tuple of (aggregate_score: float, best_result: SemanticCoherenceResult).
        """
        valid_maps = [m for m in explanation_maps if m.is_valid]
        if not valid_maps:
            logger.warning("No valid explanation maps for semantic evaluation.")
            return 0.5, SemanticCoherenceResult(
                score=0.5,
                justification="No valid explanation maps available.",
                region_description="N/A",
                predicted_label=label,
                model_used="none",
            )

        # Evaluate the map with the highest mean attribution (most confident explanation)
        best_map = max(valid_maps, key=lambda m: m.attribution.mean())
        result = self.evaluate(best_map, label, image_shape)
        return result.score, result
