"""Tests for the Groq provider adapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from messy_text.models import ProductionStage
from messy_text.providers import SYSTEM_PROMPT, _reset_client_for_testing, call_llm


def _mock_groq_response(payload: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = payload
    return mock_response


class TestCallLLM:
    """Tests for the Groq-backed LLM call."""

    @pytest.fixture(autouse=True)
    def reset_client(self):
        _reset_client_for_testing()
        yield
        _reset_client_for_testing()

    def test_lazy_creates_client(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "Test.",
                    "stage": "DEVELOPMENT",
                    "confidence": 0.8,
                }
            )
        )

        with patch("messy_text.providers.Groq", return_value=mock_client) as mock_groq:
            call_llm("Some text")

        mock_groq.assert_called_once_with(api_key="test-key", max_retries=3)

    def test_development_classification(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The text mentions pitching and financing.",
                    "stage": "DEVELOPMENT",
                    "confidence": 0.9,
                }
            )
        )

        with patch("messy_text.providers.Groq", return_value=mock_client):
            result = call_llm("We're pitching the project to studios and seeking financing.")

        assert result.stage == ProductionStage.DEVELOPMENT
        assert result.confidence == 0.9

    def test_pre_production_classification(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The project is greenlit and crew is being hired.",
                    "stage": "PRE_PRODUCTION",
                    "confidence": 0.9,
                }
            )
        )

        with patch("messy_text.providers.Groq", return_value=mock_client):
            result = call_llm("The film has been greenlit and they're hiring department heads.")

        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_production_classification(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The text says filming started.",
                    "stage": "PRODUCTION",
                    "confidence": 0.95,
                }
            )
        )

        with patch("messy_text.providers.Groq", return_value=mock_client):
            result = call_llm("They started shooting the pilot episode in Brooklyn.")

        assert result.stage == ProductionStage.PRODUCTION

    def test_unclassifiable_classification(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The text is unrelated to production stages.",
                    "stage": "UNCLASSIFIABLE",
                    "confidence": 0.95,
                }
            )
        )

        with patch("messy_text.providers.Groq", return_value=mock_client):
            result = call_llm("Grilled salmon with lemon butter sauce.")

        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_api_called_with_correct_params(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("MESSY_TEXT_MODEL", "test-model")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "Test.",
                    "stage": "DEVELOPMENT",
                    "confidence": 0.8,
                }
            )
        )

        with patch("messy_text.providers.Groq", return_value=mock_client):
            call_llm("Some test text.")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.0
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "Some test text."}

    def test_missing_api_key_raises_configuration_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with pytest.raises(ValueError, match="GROQ_API_KEY environment variable is not set"):
            call_llm("Some text.")

    def test_api_timeout_raises_provider_error(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.side_effect = TimeoutError("Connection timed out")

        with patch("messy_text.providers.Groq", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Groq request failed"):
                call_llm("Some ambiguous text.")

    def test_malformed_json_raises_response_error(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "not valid json at all"
        )

        with patch("messy_text.providers.Groq", return_value=mock_client):
            with pytest.raises(ValueError, match="not valid JSON"):
                call_llm("Some text.")

    def test_invalid_confidence_raises_response_error(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The text mentions financing.",
                    "stage": "DEVELOPMENT",
                    "confidence": 1.5,
                }
            )
        )

        with patch("messy_text.providers.Groq", return_value=mock_client):
            with pytest.raises(ValueError, match="expected schema"):
                call_llm("Some text.")
