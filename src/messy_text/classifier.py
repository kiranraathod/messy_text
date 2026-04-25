"""Core classification logic for film/TV production stage detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Callable
import os

from messy_text.errors import ConfigurationError, InputValidationError
from messy_text.models import ClassificationResult, ProductionStage

LLMClassifier = Callable[[str], ClassificationResult]
LOW_CONFIDENCE_THRESHOLD = 0.5
MAX_INPUT_CHARS = 2000


@dataclass(frozen=True)
class ClassifierConfig:
    """Runtime classifier settings resolved once at startup."""

    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    max_input_chars: int = MAX_INPUT_CHARS

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ConfigurationError(
                "low_confidence_threshold must be between 0.0 and 1.0."
            )
        if self.max_input_chars <= 0:
            raise ConfigurationError(
                "max_input_chars must be greater than 0."
            )


SYSTEM_PROMPT = """
You classify messy film and television production text into one production stage.

Return only a JSON object with these keys:
- reasoning: a brief explanation grounded only in the text
- stage: one of DEVELOPMENT, PRE_PRODUCTION, PRODUCTION, UNCLASSIFIABLE
- confidence: a float between 0.0 and 1.0

Rules:
- Keep the reasoning short and concrete.
- Do not invent facts that are not in the text.
- DEVELOPMENT: scripts, rights, pitching, financing, talent attachment, development hell, turnaround.
- PRE_PRODUCTION: greenlit and funded, hiring crew, scouting, or scheduling shoot dates.
- PRODUCTION: active filming, principal photography, cameras rolling, or shooting on set/location.
- UNCLASSIFIABLE: irrelevant text or not enough signal to place it in a stage.
- If multiple stages are mentioned, choose the latest chronological event.
- "Attached talent" by itself is DEVELOPMENT.
""".strip()


# Exact phrase shortcuts only. Everything else should be decided by the LLM.
_FAST_ROUTES: list[tuple[re.Pattern[str], ProductionStage, float, str]] = [
    (
        re.compile(r"\bprincipal\s+photography\b", re.IGNORECASE),
        ProductionStage.PRODUCTION,
        0.95,
        "The text explicitly mentions principal photography, indicating active filming.",
    ),
    (
        re.compile(r"\bcameras?\s+(?:are\s+)?roll(?:s|ing|ed)\b", re.IGNORECASE),
        ProductionStage.PRODUCTION,
        0.95,
        "The text mentions cameras rolling, indicating active filming.",
    ),
    (
        re.compile(r"\bcurrently\s+(?:filming|shooting)\b", re.IGNORECASE),
        ProductionStage.PRODUCTION,
        0.95,
        "The text states filming or shooting is currently happening.",
    ),
    (
        re.compile(r"\bday\s+\d+\s+of\s+(?:filming|shooting|production)\b", re.IGNORECASE),
        ProductionStage.PRODUCTION,
        0.95,
        "The text references a specific day of an active shoot.",
    ),
    (
        re.compile(r"\bofficially\s+greenl(?:it|ight(?:ed)?)\b", re.IGNORECASE),
        ProductionStage.PRE_PRODUCTION,
        0.95,
        "The text explicitly states the project is officially greenlit.",
    ),
    (
        re.compile(r"\bin\s+turnaround\b", re.IGNORECASE),
        ProductionStage.DEVELOPMENT,
        0.93,
        "The text explicitly says the project is in turnaround, indicating development.",
    ),
    (
        re.compile(r"\bseeking\s+(?:film\s+)?financing\b", re.IGNORECASE),
        ProductionStage.DEVELOPMENT,
        0.92,
        "The text mentions seeking financing, indicating development.",
    ),
    (
        re.compile(r"\bstart(?:ing|ed)\s+pre-?production\b", re.IGNORECASE),
        ProductionStage.PRE_PRODUCTION,
        0.93,
        "The text states pre-production is starting, indicating pre-production.",
    ),
    (
        re.compile(r"\bscout(?:ing|ed)\s+locations?\b", re.IGNORECASE),
        ProductionStage.PRE_PRODUCTION,
        0.92,
        "The text mentions scouting locations, indicating pre-production.",
    ),
]


def _fast_regex_router(text: str) -> ClassificationResult | None:
    """Return a result for exact high-confidence phrases, otherwise None."""
    matches = [
        (stage, confidence, reasoning)
        for pattern, stage, confidence, reasoning in _FAST_ROUTES
        if pattern.search(text)
    ]
    if len(matches) > 1:
        return None

    if matches:
        stage, confidence, reasoning = matches[0]
        return ClassificationResult(
            reasoning=reasoning,
            stage=stage,
            confidence=confidence,
        )
    return None


def load_classifier_config_from_env() -> ClassifierConfig:
    try:
        return ClassifierConfig(
            low_confidence_threshold=float(
                os.environ.get(
                    "MESSY_TEXT_CONFIDENCE_THRESHOLD",
                    LOW_CONFIDENCE_THRESHOLD,
                )
            ),
            max_input_chars=int(
                os.environ.get("MESSY_TEXT_MAX_INPUT_CHARS", MAX_INPUT_CHARS)
            ),
        )
    except (ValueError, ConfigurationError) as exc:
        raise ConfigurationError(
            f"Invalid classifier configuration: {exc}"
        ) from exc


def classify(
    text: str,
    *,
    llm_classifier: LLMClassifier | None = None,
    config: ClassifierConfig | None = None,
) -> ClassificationResult:
    """Classify messy text into a production stage."""
    if not text or not text.strip():
        return ClassificationResult(
            reasoning="The input text is empty. No classification is possible.",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=1.0,
        )

    classifier_config = config or ClassifierConfig()
    normalized = text.strip()
    if len(normalized) > classifier_config.max_input_chars:
        raise InputValidationError(
            "Input too long "
            f"({len(normalized)} chars). Maximum is {classifier_config.max_input_chars}."
        )

    fast_result = _fast_regex_router(normalized)
    if fast_result is not None:
        return fast_result

    if llm_classifier is None:
        raise ConfigurationError(
            "LLM classifier is not configured for ambiguous text."
        )

    result = llm_classifier(normalized)
    if result.confidence < classifier_config.low_confidence_threshold:
        return ClassificationResult(
            reasoning=f"Low confidence: {result.reasoning}",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=result.confidence,
        )

    return result
