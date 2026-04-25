"""Tests for the core production stage classifier."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from messy_text.classifier import (
    MAX_INPUT_CHARS,
    classify,
)
from messy_text.models import ClassificationResult, ProductionStage


class TestClassifyPipeline:
    """Tests for the pure classification flow."""

    def test_empty_string(self):
        result = classify("")
        assert result.stage == ProductionStage.UNCLASSIFIABLE
        assert result.confidence == 1.0

    def test_whitespace_only(self):
        result = classify("   \n\t  ")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_fallback_to_llm(self):
        with patch("messy_text.classifier.call_llm") as mock_call:
            mock_call.return_value = ClassificationResult(
                reasoning="The text discusses script rewrites, a development activity.",
                stage=ProductionStage.DEVELOPMENT,
                confidence=0.85,
            )

            result = classify("The screenplay is being rewritten for the third time.")

            assert result.stage == ProductionStage.DEVELOPMENT
            mock_call.assert_called_once_with("The screenplay is being rewritten for the third time.")

    def test_temporal_override_uses_llm_result(self):
        with patch("messy_text.classifier.call_llm") as mock_call:
            mock_call.return_value = ClassificationResult(
                reasoning="The latest event is starting to shoot.",
                stage=ProductionStage.PRODUCTION,
                confidence=0.95,
            )

            result = classify("After 5 years in development, we finally started shooting today.")

            assert result.stage == ProductionStage.PRODUCTION

    def test_attached_talent_uses_llm_result(self):
        with patch("messy_text.classifier.call_llm") as mock_call:
            mock_call.return_value = ClassificationResult(
                reasoning="Attached talent alone indicates development.",
                stage=ProductionStage.DEVELOPMENT,
                confidence=0.85,
            )

            result = classify("Chris Hemsworth is attached to star in the upcoming action film.")

            assert result.stage == ProductionStage.DEVELOPMENT

    def test_low_confidence_llm_result_becomes_unclassifiable(self):
        with patch("messy_text.classifier.call_llm") as mock_call:
            mock_call.return_value = ClassificationResult(
                reasoning="The text might describe development, but the signal is weak.",
                stage=ProductionStage.DEVELOPMENT,
                confidence=0.4,
            )

            result = classify("The project may still be seeking a path forward.")

            assert result.stage == ProductionStage.UNCLASSIFIABLE
            assert result.confidence == 0.4
            assert result.reasoning == (
                "Low confidence: The text might describe development, but the signal is weak."
            )

    def test_input_too_long_raises_input_validation_error(self):
        with pytest.raises(
            ValueError,
            match=rf"Input too long \({MAX_INPUT_CHARS + 1} chars\)\. Maximum is {MAX_INPUT_CHARS}\.",
        ):
            classify("x" * (MAX_INPUT_CHARS + 1))

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
        monkeypatch.setenv(env_name, value)

        with pytest.raises(ValueError, match=message):
            classify("Some valid length string")


class TestOutputFormat:
    """Ensure successful JSON output matches the expected contract."""

    @pytest.fixture(autouse=True)
    def setup_mock_llm(self):
        with patch("messy_text.classifier.call_llm") as mock_call:
            mock_call.return_value = ClassificationResult(
                reasoning="Cameras rolling.",
                stage=ProductionStage.PRODUCTION,
                confidence=0.95,
            )
            yield

    def test_json_has_three_keys(self):
        result = classify("Principal photography is underway.")
        output = json.loads(result.to_json())
        assert set(output.keys()) == {"reasoning", "stage", "confidence"}

    def test_stage_is_valid_enum(self):
        result = classify("Cameras rolling on set.")
        output = json.loads(result.to_json())
        valid_stages = {"DEVELOPMENT", "PRE_PRODUCTION", "PRODUCTION", "UNCLASSIFIABLE"}
        assert output["stage"] in valid_stages

    def test_confidence_in_range(self):
        result = classify("Cameras rolling on the new project.")
        assert 0.0 <= result.confidence <= 1.0

    def test_json_round_trip(self):
        result = classify("Currently filming on location.")
        output_str = result.to_json()
        parsed = json.loads(output_str)
        assert isinstance(parsed["reasoning"], str)
        assert isinstance(parsed["stage"], str)
        assert isinstance(parsed["confidence"], float)
