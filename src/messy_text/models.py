"""Data models for the production stage classifier."""

from __future__ import annotations

import json
from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class ProductionStage(str, Enum):
    """Strict taxonomy of film/TV production stages."""

    DEVELOPMENT = "DEVELOPMENT"
    PRE_PRODUCTION = "PRE_PRODUCTION"
    PRODUCTION = "PRODUCTION"
    UNCLASSIFIABLE = "UNCLASSIFIABLE"


class ClassificationResult(BaseModel):
    """Structured output for a single classification.

    The JSON output order is: reasoning → stage → confidence,
    enforced by model_config field ordering.
    """

    reasoning: str = Field(
        description="A simple 1-2 sentence logical deduction identifying "
        "the chronological markers in the text.",
    )
    stage: ProductionStage = Field(
        description="The classified production stage.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    model_config = {"json_schema_extra": {"required": ["reasoning", "stage", "confidence"]}}

    @model_validator(mode="after")
    def _clamp_confidence(self) -> Self:
        """Safety clamp — pydantic already validates ge/le but this is belt-and-suspenders."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        return self

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to JSON with reasoning-first key order."""
        # Manually order keys to guarantee reasoning comes first
        ordered = {
            "reasoning": self.reasoning,
            "stage": self.stage.value,
            "confidence": self.confidence,
        }
        return json.dumps(ordered, indent=indent)
