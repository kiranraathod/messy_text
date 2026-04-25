"""Provider adapters for stage classification."""

from __future__ import annotations

import json
import os

from groq import Groq
from pydantic import ValidationError as PydanticValidationError

from messy_text.models import ClassificationResult

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

DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_client: Groq | None = None


def call_llm(text: str) -> ClassificationResult:
    """Classify text using the Groq LLM."""
    global _client

    model = os.environ.get("MESSY_TEXT_MODEL", DEFAULT_GROQ_MODEL)
    
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        _client = Groq(api_key=api_key, max_retries=3)

    try:
        response = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc

    try:
        raw = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("Groq response did not include message content.") from exc

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Groq response content was empty.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Groq response was not valid JSON.") from exc

    try:
        return ClassificationResult.model_validate(payload)
    except PydanticValidationError as exc:
        message = exc.errors()[0]["msg"]
        raise ValueError(
            f"Groq response did not match the expected schema: {message}"
        ) from exc


def _reset_client_for_testing() -> None:
    """Reset the global Groq client (useful for unit tests)."""
    global _client
    _client = None
