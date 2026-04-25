"""Tests for the core production stage classifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from messy_text.classifier import SYSTEM_PROMPT, classify, get_client
from messy_text.config import get_config
from messy_text.models import ProductionStage


def _mock_groq_response(payload: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = payload
    return mock_response


class TestClassifyPipeline:
    """Tests for the pure classification flow."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        get_config.cache_clear()
        get_client.cache_clear()
        yield
        get_config.cache_clear()
        get_client.cache_clear()

    def test_empty_string(self):
        result = classify("")
        assert result.stage == ProductionStage.UNCLASSIFIABLE
        assert result.confidence == 1.0

    def test_whitespace_only(self):
        result = classify("   \n\t  ")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

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

        with patch("messy_text.classifier.Groq", return_value=mock_client) as mock_groq:
            classify("Some text")

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

        with patch("messy_text.classifier.Groq", return_value=mock_client):
            result = classify("We're pitching the project to studios and seeking financing.")

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

        with patch("messy_text.classifier.Groq", return_value=mock_client):
            result = classify("The film has been greenlit and they're hiring department heads.")

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

        with patch("messy_text.classifier.Groq", return_value=mock_client):
            result = classify("They started shooting the pilot episode in Brooklyn.")

        assert result.stage == ProductionStage.PRODUCTION

    def test_low_confidence_llm_result_becomes_unclassifiable(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "The text might describe development, but the signal is weak.",
                    "stage": "DEVELOPMENT",
                    "confidence": 0.4,
                }
            )
        )

        with patch("messy_text.classifier.Groq", return_value=mock_client):
            result = classify("The project may still be seeking a path forward.")

        assert result.stage == ProductionStage.UNCLASSIFIABLE
        assert result.confidence == 0.4
        assert result.reasoning == (
            "Low confidence: The text might describe development, but the signal is weak."
        )

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

        with patch("messy_text.classifier.Groq", return_value=mock_client):
            classify("Some test text.")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.0
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "Some test text."}

    def test_api_timeout_raises_provider_error(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.side_effect = TimeoutError("Connection timed out")

        with patch("messy_text.classifier.Groq", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Groq request failed"):
                classify("Some ambiguous text.")

    def test_malformed_json_raises_response_error(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "not valid json at all"
        )

        with patch("messy_text.classifier.Groq", return_value=mock_client):
            with pytest.raises(ValueError, match="match the expected schema"):
                classify("Some text.")

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

        with patch("messy_text.classifier.Groq", return_value=mock_client):
            with pytest.raises(ValueError, match="expected schema"):
                classify("Some text.")

    def test_input_too_long_raises_input_validation_error(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("MESSY_TEXT_MAX_INPUT_CHARS", "10")
        with pytest.raises(
            ValueError,
            match=r"Input too long \(11 chars\)\. Maximum is 10\.",
        ):
            classify("x" * 11)

    @pytest.mark.parametrize(
        ("env_name", "value", "message"),
        [
            (
                "MESSY_TEXT_CONFIDENCE_THRESHOLD",
                "1.5",
                "low_confidence_threshold must be between 0.0 and 1.0.",
            ),
            (
                "MESSY_TEXT_CONFIDENCE_THRESHOLD",
                "-0.1",
                "low_confidence_threshold must be between 0.0 and 1.0.",
            ),
            (
                "MESSY_TEXT_MAX_INPUT_CHARS",
                "0",
                "max_input_chars must be greater than 0.",
            ),
            (
                "MESSY_TEXT_MAX_INPUT_CHARS",
                "abc",
                "Invalid environment configuration",
            ),
        ],
    )
    def test_classifier_config_validation(
        self,
        monkeypatch,
        env_name: str,
        value: str,
        message: str,
    ):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv(env_name, value)

        with pytest.raises(ValueError, match=message):
            classify("Some valid length string")


class TestOutputFormat:
    """Ensure successful JSON output matches the expected contract."""

    @pytest.fixture(autouse=True)
    def setup_mock_llm(self, monkeypatch):
        get_config.cache_clear()
        get_client.cache_clear()
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps(
                {
                    "reasoning": "Cameras rolling.",
                    "stage": "PRODUCTION",
                    "confidence": 0.95,
                }
            )
        )
        with patch("messy_text.classifier.Groq", return_value=mock_client):
            yield
        get_config.cache_clear()
        get_client.cache_clear()

    def test_json_has_three_keys(self):
        result = classify("Principal photography is underway.")
        output = json.loads(result.model_dump_json())
        assert set(output.keys()) == {"reasoning", "stage", "confidence"}

    def test_stage_is_valid_enum(self):
        result = classify("Cameras rolling on set.")
        output = json.loads(result.model_dump_json())
        valid_stages = {"DEVELOPMENT", "PRE_PRODUCTION", "PRODUCTION", "UNCLASSIFIABLE"}
        assert output["stage"] in valid_stages

    def test_confidence_in_range(self):
        result = classify("Cameras rolling on the new project.")
        assert 0.0 <= result.confidence <= 1.0

    def test_json_round_trip(self):
        result = classify("Currently filming on location.")
        output_str = result.model_dump_json(indent=2)
        parsed = json.loads(output_str)
        assert isinstance(parsed["reasoning"], str)
        assert isinstance(parsed["stage"], str)
        assert isinstance(parsed["confidence"], float)
