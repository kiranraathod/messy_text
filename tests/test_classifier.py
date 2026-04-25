"""Tests for the core production stage classifier."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from messy_text.classifier import (
    ClassifierConfig,
    MAX_INPUT_CHARS,
    _fast_regex_router,
    classify,
    load_classifier_config_from_env,
)
from messy_text.errors import ConfigurationError, InputValidationError
from messy_text.models import ClassificationResult, ProductionStage


class TestFastRouter:
    """Tests for the exact-phrase fast router."""

    def test_principal_photography(self):
        result = _fast_regex_router("Principal photography began last Monday in Vancouver.")
        assert result is not None
        assert result.stage == ProductionStage.PRODUCTION

    def test_cameras_rolling(self):
        result = _fast_regex_router("Cameras are rolling on the new Marvel film in Atlanta.")
        assert result is not None
        assert result.stage == ProductionStage.PRODUCTION

    def test_currently_filming(self):
        result = _fast_regex_router("The show is currently filming its second season.")
        assert result is not None
        assert result.stage == ProductionStage.PRODUCTION

    def test_day_of_filming(self):
        result = _fast_regex_router("Day 42 of filming and the crew is exhausted.")
        assert result is not None
        assert result.stage == ProductionStage.PRODUCTION

    def test_officially_greenlit(self):
        result = _fast_regex_router("The series has been officially greenlit by the network.")
        assert result is not None
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_no_match_returns_none(self):
        result = _fast_regex_router("We're pitching the project to Netflix this week.")
        assert result is None

    def test_restaurant_menu_returns_none(self):
        result = _fast_regex_router("Grilled salmon with lemon butter sauce.")
        assert result is None

    def test_attached_talent_returns_none(self):
        result = _fast_regex_router("Chris Hemsworth is attached to star.")
        assert result is None

    def test_case_insensitive(self):
        result = _fast_regex_router("PRINCIPAL PHOTOGRAPHY has begun!")
        assert result is not None
        assert result.stage == ProductionStage.PRODUCTION

    def test_confidence_is_high(self):
        result = _fast_regex_router("Cameras rolling on set today.")
        assert result is not None
        assert result.confidence >= 0.9

    def test_additional_fast_routes(self):
        cases = [
            ("The project is in turnaround after studio changes.", ProductionStage.DEVELOPMENT),
            ("They are seeking film financing ahead of packaging.", ProductionStage.DEVELOPMENT),
            ("The team started pre-production this week.", ProductionStage.PRE_PRODUCTION),
            ("They scouted locations across Toronto.", ProductionStage.PRE_PRODUCTION),
        ]

        for text, expected_stage in cases:
            result = _fast_regex_router(text)
            assert result is not None
            assert result.stage == expected_stage

    def test_multiple_fast_route_matches_return_none(self):
        result = _fast_regex_router(
            "After being in turnaround, the team started pre-production last week."
        )

        assert result is None


class TestClassifyPipeline:
    """Tests for the pure classification flow."""

    def test_empty_string(self):
        result = classify("")
        assert result.stage == ProductionStage.UNCLASSIFIABLE
        assert result.confidence == 1.0

    def test_whitespace_only(self):
        result = classify("   \n\t  ")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_fast_route_skips_llm(self):
        llm_classifier = Mock(side_effect=AssertionError("LLM should not be called"))

        result = classify(
            "Principal photography began last Monday.",
            llm_classifier=llm_classifier,
        )

        assert result.stage == ProductionStage.PRODUCTION
        llm_classifier.assert_not_called()

    def test_fallback_to_llm(self):
        llm_classifier = Mock(
            return_value=ClassificationResult(
                reasoning="The text discusses script rewrites, a development activity.",
                stage=ProductionStage.DEVELOPMENT,
                confidence=0.85,
            )
        )

        result = classify(
            "The screenplay is being rewritten for the third time.",
            llm_classifier=llm_classifier,
        )

        assert result.stage == ProductionStage.DEVELOPMENT
        llm_classifier.assert_called_once_with(
            "The screenplay is being rewritten for the third time."
        )

    def test_temporal_override_uses_llm_result(self):
        llm_classifier = Mock(
            return_value=ClassificationResult(
                reasoning="The latest event is starting to shoot.",
                stage=ProductionStage.PRODUCTION,
                confidence=0.95,
            )
        )

        result = classify(
            "After 5 years in development, we finally started shooting today.",
            llm_classifier=llm_classifier,
        )

        assert result.stage == ProductionStage.PRODUCTION

    def test_attached_talent_uses_llm_result(self):
        llm_classifier = Mock(
            return_value=ClassificationResult(
                reasoning="Attached talent alone indicates development.",
                stage=ProductionStage.DEVELOPMENT,
                confidence=0.85,
            )
        )

        result = classify(
            "Chris Hemsworth is attached to star in the upcoming action film.",
            llm_classifier=llm_classifier,
        )

        assert result.stage == ProductionStage.DEVELOPMENT

    def test_ambiguous_text_without_llm_configuration_errors(self):
        with pytest.raises(ConfigurationError):
            classify("Studio executives are debating the project's next move.")

    def test_low_confidence_llm_result_becomes_unclassifiable(self):
        llm_classifier = Mock(
            return_value=ClassificationResult(
                reasoning="The text might describe development, but the signal is weak.",
                stage=ProductionStage.DEVELOPMENT,
                confidence=0.4,
            )
        )

        result = classify(
            "The project may still be seeking a path forward.",
            llm_classifier=llm_classifier,
            config=ClassifierConfig(),
        )

        assert result.stage == ProductionStage.UNCLASSIFIABLE
        assert result.confidence == 0.4
        assert result.reasoning == (
            "Low confidence: The text might describe development, but the signal is weak."
        )

    def test_input_too_long_raises_input_validation_error(self):
        with pytest.raises(
            InputValidationError,
            match=rf"Input too long \({MAX_INPUT_CHARS + 1} chars\)\. Maximum is {MAX_INPUT_CHARS}\.",
        ):
            classify(
                "x" * (MAX_INPUT_CHARS + 1),
                llm_classifier=Mock(),
                config=ClassifierConfig(),
            )

    def test_load_classifier_config_from_env_respects_runtime_env_overrides(self, monkeypatch):
        monkeypatch.setenv("MESSY_TEXT_MAX_INPUT_CHARS", "10")
        monkeypatch.setenv("MESSY_TEXT_CONFIDENCE_THRESHOLD", "0.7")
        config = load_classifier_config_from_env()

        assert config.max_input_chars == 10
        assert config.low_confidence_threshold == 0.7

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {"low_confidence_threshold": 1.5},
                "low_confidence_threshold must be between 0.0 and 1.0.",
            ),
            (
                {"max_input_chars": 0},
                "max_input_chars must be greater than 0.",
            ),
        ],
    )
    def test_classifier_config_validation_uses_field_names(self, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            ClassifierConfig(**kwargs)

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
        ],
    )
    def test_load_classifier_config_from_env_validates_bounds(
        self,
        monkeypatch,
        env_name: str,
        value: str,
        message: str,
    ):
        monkeypatch.setenv(env_name, value)

        with pytest.raises(ConfigurationError, match=message):
            load_classifier_config_from_env()

    def test_multi_signal_fast_route_text_falls_back_to_llm(self):
        llm_classifier = Mock(
            return_value=ClassificationResult(
                reasoning="The latest event is starting pre-production.",
                stage=ProductionStage.PRE_PRODUCTION,
                confidence=0.88,
            )
        )

        result = classify(
            "After being in turnaround, the team started pre-production last week.",
            llm_classifier=llm_classifier,
        )

        assert result.stage == ProductionStage.PRE_PRODUCTION
        llm_classifier.assert_called_once_with(
            "After being in turnaround, the team started pre-production last week."
        )


class TestOutputFormat:
    """Ensure successful JSON output matches the expected contract."""

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
