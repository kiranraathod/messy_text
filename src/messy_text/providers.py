"""Provider adapters for stage classification."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from groq import Groq
from pydantic import ValidationError as PydanticValidationError

from messy_text.classifier import SYSTEM_PROMPT
from messy_text.errors import ConfigurationError, ProviderError, ResponseFormatError
from messy_text.models import ClassificationResult

DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@dataclass
class GroqLLMClassifier:
    """Groq-backed classifier callable."""

    api_key: str | None = None
    model: str | None = None
    client: Groq | None = None

    def __post_init__(self) -> None:
        self.model = self.model or os.environ.get("MESSY_TEXT_MODEL", DEFAULT_GROQ_MODEL)

        if self.client is not None:
            return

        api_key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "GROQ_API_KEY environment variable is not set."
            )

        self.client = Groq(api_key=api_key)

    def __call__(self, text: str) -> ClassificationResult:
        raw_response = self._request_completion(text)
        return self._parse_response(raw_response)

    def _request_completion(self, text: str) -> str:
        last_error: ProviderError | None = None
        for delay in (0, 1, 2, 4):
            if delay:
                time.sleep(delay)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=256,
                )
                break
            except Exception as exc:
                last_error = ProviderError(f"Groq request failed: {exc}")
        else:
            if last_error is None:
                raise ProviderError("Retry exhausted but no error was captured.")
            raise last_error

        try:
            raw = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ResponseFormatError(
                "Groq response did not include message content."
            ) from exc

        if not isinstance(raw, str) or not raw.strip():
            raise ResponseFormatError("Groq response content was empty.")

        return raw

    def _parse_response(self, raw: str) -> ClassificationResult:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResponseFormatError("Groq response was not valid JSON.") from exc

        try:
            return ClassificationResult.model_validate(payload)
        except PydanticValidationError as exc:
            message = exc.errors()[0]["msg"]
            raise ResponseFormatError(
                f"Groq response did not match the expected schema: {message}"
            ) from exc
