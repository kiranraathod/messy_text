"""Data models for the production stage classifier."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ProductionStage(str, Enum):
    """Strict taxonomy of film/TV production stages."""

    DEVELOPMENT = "DEVELOPMENT"
    PRE_PRODUCTION = "PRE_PRODUCTION"
    PRODUCTION = "PRODUCTION"
    UNCLASSIFIABLE = "UNCLASSIFIABLE"


class ClassificationResult(BaseModel):
    """Structured output for a single successful classification."""

    reasoning: str = Field(
        description="A short explanation grounded in the text.",
    )
    stage: ProductionStage = Field(
        description="The classified production stage.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    reliable: bool = Field(
        default=False,
        description="True if confidence >= 0.85 (strong signal), False otherwise.",
    )
