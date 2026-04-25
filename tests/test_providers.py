"""Tests for the Groq provider adapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from messy_text.classifier import SYSTEM_PROMPT
from messy_text.errors import ConfigurationError, ProviderError, ResponseFormatError
from messy_text.models import ProductionStage
from messy_text.providers import GroqLLMClassifier


def _mock_groq_response(payload: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = payload
    return mock_response


class TestGroqLLMClassifier:
    """Tests for the Groq-backed LLM classifier."""

    def test_post_init_eagerly_creates_client(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        with patch("messy_text.providers.Groq", return_value=mock_client) as mock_groq:
            classifier = GroqLLMClassifier()

        assert classifier.client is mock_client
        mock_groq.assert_called_once_with(api_key="test-key")

    def test_post_init_resolves_model_once_from_environment(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("MESSY_TEXT_MODEL", "env-model")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "Test.",
                    "stage": "DEVELOPMENT",
                    "confidence": 0.8,
                }
            )
        )

        classifier = GroqLLMClassifier(client=mock_client)
        monkeypatch.setenv("MESSY_TEXT_MODEL", "changed-model")

        classifier("Some test text.")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "env-model"

    def test_development_classification(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The text mentions pitching and financing.",
                    "stage": "DEVELOPMENT",
                    "confidence": 0.9,
                }
            )
        )

        result = GroqLLMClassifier(client=mock_client)(
            "We're pitching the project to studios and seeking financing."
        )

        assert result.stage == ProductionStage.DEVELOPMENT
        assert result.confidence == 0.9

    def test_pre_production_classification(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The project is greenlit and crew is being hired.",
                    "stage": "PRE_PRODUCTION",
                    "confidence": 0.9,
                }
            )
        )

        result = GroqLLMClassifier(client=mock_client)(
            "The film has been greenlit and they're hiring department heads."
        )

        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_production_classification(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The text says filming started.",
                    "stage": "PRODUCTION",
                    "confidence": 0.95,
                }
            )
        )

        result = GroqLLMClassifier(client=mock_client)(
            "They started shooting the pilot episode in Brooklyn."
        )

        assert result.stage == ProductionStage.PRODUCTION

    def test_unclassifiable_classification(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The text is unrelated to production stages.",
                    "stage": "UNCLASSIFIABLE",
                    "confidence": 0.95,
                }
            )
        )

        result = GroqLLMClassifier(client=mock_client)(
            "Grilled salmon with lemon butter sauce."
        )

        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_api_called_with_correct_params(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "Test.",
                    "stage": "DEVELOPMENT",
                    "confidence": 0.8,
                }
            )
        )

        classifier = GroqLLMClassifier(client=mock_client, model="test-model")
        classifier("Some test text.")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.0
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "Some test text."}

    def test_missing_api_key_raises_configuration_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with pytest.raises(ConfigurationError):
            GroqLLMClassifier()

    def test_api_timeout_raises_provider_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Connection timed out")

        with patch("messy_text.providers.time.sleep"):
            with pytest.raises(ProviderError, match="Connection timed out"):
                GroqLLMClassifier(client=mock_client)("Some ambiguous text.")

    def test_request_retries_with_exponential_backoff(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            TimeoutError("try 1"),
            TimeoutError("try 2"),
            TimeoutError("try 3"),
            _mock_groq_response(
                json.dumps(
                    {
                        "reasoning": "The text mentions pitching and financing.",
                        "stage": "DEVELOPMENT",
                        "confidence": 0.9,
                    }
                )
            ),
        ]

        with patch("messy_text.providers.time.sleep") as mock_sleep:
            result = GroqLLMClassifier(client=mock_client)("Some ambiguous text.")

        assert result.stage == ProductionStage.DEVELOPMENT
        assert mock_client.chat.completions.create.call_count == 4
        assert [call.args[0] for call in mock_sleep.call_args_list] == [1, 2, 4]

    def test_malformed_json_raises_response_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "not valid json at all"
        )

        with pytest.raises(ResponseFormatError, match="not valid JSON"):
            GroqLLMClassifier(client=mock_client)("Some text.")

    def test_invalid_confidence_raises_response_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The text mentions financing.",
                    "stage": "DEVELOPMENT",
                    "confidence": 1.5,
                }
            )
        )

        with pytest.raises(ResponseFormatError, match="expected schema"):
            GroqLLMClassifier(client=mock_client)("Some text.")
