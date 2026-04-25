"""Centralized configuration for the messy-text classifier."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Application configuration."""

    api_key: str
    model: str
    max_input_chars: int
    low_confidence_threshold: float
    batch_workers: int


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and validate configuration from the environment."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    model = os.environ.get("MESSY_TEXT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    try:
        max_chars = int(os.environ.get("MESSY_TEXT_MAX_INPUT_CHARS", "2000"))
        threshold = float(os.environ.get("MESSY_TEXT_CONFIDENCE_THRESHOLD", "0.5"))
        workers = int(os.environ.get("MESSY_TEXT_BATCH_WORKERS", "4"))
    except ValueError as exc:
        raise ValueError(f"Invalid environment configuration: {exc}") from exc

    if max_chars <= 0:
        raise ValueError("max_input_chars must be greater than 0.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be between 0.0 and 1.0.")
    if workers <= 0:
        raise ValueError("MESSY_TEXT_BATCH_WORKERS must be greater than 0.")

    return Config(
        api_key=api_key,
        model=model,
        max_input_chars=max_chars,
        low_confidence_threshold=threshold,
        batch_workers=workers,
    )
