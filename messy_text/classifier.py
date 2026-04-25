"""Core classification logic for film/TV production stage detection."""

from __future__ import annotations

import os

from groq import Groq
from pydantic import ValidationError

from messy_text.models import ClassificationResult, ProductionStage

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


def classify(text: str) -> ClassificationResult:
    """Classify messy text into a production stage."""
    if not text or not text.strip():
        return ClassificationResult(
            reasoning="The input text is empty. No classification is possible.",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=1.0,
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    model = os.environ.get("MESSY_TEXT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    
    try:
        max_chars = int(os.environ.get("MESSY_TEXT_MAX_INPUT_CHARS", "2000"))
    except ValueError as exc:
        raise ValueError(f"Invalid MESSY_TEXT_MAX_INPUT_CHARS: {exc}") from exc

    normalized = text.strip()
    if len(normalized) > max_chars:
        raise ValueError(
            f"Input too long ({len(normalized)} chars). Maximum is {max_chars}."
        )

    client = Groq(api_key=api_key, max_retries=3)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": normalized},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc

    raw = response.choices[0].message.content

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Groq response content was empty.")

    try:
        return ClassificationResult.model_validate_json(raw)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"]
        raise ValueError(
            f"Groq response did not match the expected schema: {message}"
        ) from exc


