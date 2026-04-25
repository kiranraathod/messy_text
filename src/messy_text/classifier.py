"""Core classification logic for film/TV production stage detection."""

from __future__ import annotations

import os

from messy_text.models import ClassificationResult, ProductionStage
from messy_text.providers import call_llm

LOW_CONFIDENCE_THRESHOLD = 0.5
MAX_INPUT_CHARS = 2000


def classify(text: str) -> ClassificationResult:
    """Classify messy text into a production stage."""
    if not text or not text.strip():
        return ClassificationResult(
            reasoning="The input text is empty. No classification is possible.",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=1.0,
        )

    try:
        max_chars = int(os.environ.get("MESSY_TEXT_MAX_INPUT_CHARS", MAX_INPUT_CHARS))
        low_confidence_threshold = float(
            os.environ.get("MESSY_TEXT_CONFIDENCE_THRESHOLD", LOW_CONFIDENCE_THRESHOLD)
        )
    except ValueError as exc:
        raise ValueError(f"Invalid environment configuration: {exc}") from exc

    if max_chars <= 0:
        raise ValueError("max_input_chars must be greater than 0.")
    if not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be between 0.0 and 1.0.")

    normalized = text.strip()
    if len(normalized) > max_chars:
        raise ValueError(
            f"Input too long ({len(normalized)} chars). Maximum is {max_chars}."
        )

    result = call_llm(normalized)

    if result.confidence < low_confidence_threshold:
        return ClassificationResult(
            reasoning=f"Low confidence: {result.reasoning}",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=result.confidence,
        )

    return result
