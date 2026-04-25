"""Test suite for the hybrid production stage classifier.

Tests are split into:
- TestFastRouter: Tests the regex pre-filter (no API calls).
- TestLLMClassifier: Tests the LLM path with mocked Groq responses.
- TestEmptyInput: Tests empty/whitespace handling.
- TestOutputFormat: Tests JSON output structure.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from messy_text.classifier import _fast_regex_router, _llm_classify, classify
from messy_text.models import ClassificationResult, ProductionStage


# ===================================================================
# Fast regex router tests (no API calls)
# ===================================================================

class TestFastRouter:
    """Tests for the ultra-fast regex pre-filter."""

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
        """Ambiguous text should fall through to LLM."""
        result = _fast_regex_router("We're pitching the project to Netflix this week.")
        assert result is None

    def test_restaurant_menu_returns_none(self):
        result = _fast_regex_router("Grilled salmon with lemon butter sauce.")
        assert result is None

    def test_attached_talent_returns_none(self):
        """Attached talent is ambiguous — should go to LLM, not fast-route."""
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


# ===================================================================
# LLM classifier tests (mocked Groq API)
# ===================================================================

def _mock_groq_response(reasoning: str, stage: str, confidence: float) -> MagicMock:
    """Create a mock Groq chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "reasoning": reasoning,
        "stage": stage,
        "confidence": confidence,
    })
    return mock_response


class TestLLMClassifier:
    """Tests for the LLM classification path with mocked API."""

    @patch("messy_text.classifier._get_groq_client")
    def test_development_classification(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "The text mentions pitching and seeking financing, which are development activities.",
            "DEVELOPMENT", 0.9,
        )
        mock_get_client.return_value = mock_client

        result = _llm_classify("We're pitching the project to studios and seeking financing.")
        assert result.stage == ProductionStage.DEVELOPMENT
        assert result.confidence == 0.9

    @patch("messy_text.classifier._get_groq_client")
    def test_pre_production_classification(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "The project is greenlit and crew is being hired, indicating pre-production.",
            "PRE_PRODUCTION", 0.9,
        )
        mock_get_client.return_value = mock_client

        result = _llm_classify("The film has been greenlit and they're hiring department heads.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    @patch("messy_text.classifier._get_groq_client")
    def test_production_classification(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "The text says filming started, indicating active production.",
            "PRODUCTION", 0.95,
        )
        mock_get_client.return_value = mock_client

        result = _llm_classify("They started shooting the pilot episode in Brooklyn.")
        assert result.stage == ProductionStage.PRODUCTION

    @patch("messy_text.classifier._get_groq_client")
    def test_unclassifiable(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "The text is about food and is irrelevant to film production.",
            "UNCLASSIFIABLE", 0.95,
        )
        mock_get_client.return_value = mock_client

        result = _llm_classify("Grilled salmon with lemon butter sauce.")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

    @patch("messy_text.classifier._get_groq_client")
    def test_temporal_override_dev_to_production(self, mock_get_client):
        """LLM should pick the latest chronological event."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "Multiple stages mentioned but the latest event is starting to shoot, indicating production.",
            "PRODUCTION", 0.95,
        )
        mock_get_client.return_value = mock_client

        result = _llm_classify(
            "After 5 years in development, we finally started shooting today."
        )
        assert result.stage == ProductionStage.PRODUCTION

    @patch("messy_text.classifier._get_groq_client")
    def test_attached_talent_is_development(self, mock_get_client):
        """Edge case: attached talent should be DEVELOPMENT per the system prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "Attached talent alone indicates development, not pre-production.",
            "DEVELOPMENT", 0.85,
        )
        mock_get_client.return_value = mock_client

        result = _llm_classify("Chris Hemsworth is attached to star in the upcoming action film.")
        assert result.stage == ProductionStage.DEVELOPMENT

    @patch("messy_text.classifier._get_groq_client")
    def test_api_called_with_correct_params(self, mock_get_client):
        """Verify the Groq API is called with json_object format and temperature 0."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "Test.", "DEVELOPMENT", 0.8,
        )
        mock_get_client.return_value = mock_client

        _llm_classify("Some test text.")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.0
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"] == "Some test text."


# ===================================================================
# Full classify() pipeline tests
# ===================================================================

class TestClassifyPipeline:
    """Tests for the full classify() pipeline."""

    def test_empty_string(self):
        result = classify("")
        assert result.stage == ProductionStage.UNCLASSIFIABLE
        assert result.confidence == 1.0

    def test_whitespace_only(self):
        result = classify("   \n\t  ")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_fast_route_skips_llm(self):
        """Fast-routed text should NOT trigger an API call."""
        result = classify("Principal photography began last Monday.")
        assert result.stage == ProductionStage.PRODUCTION
        # No mock needed — if it tried to call Groq without a key, it would error

    @patch("messy_text.classifier._get_groq_client")
    def test_fallback_to_llm(self, mock_get_client):
        """Text that doesn't match fast routes should go to LLM."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "The text discusses script rewrites, a development activity.",
            "DEVELOPMENT", 0.85,
        )
        mock_get_client.return_value = mock_client

        result = classify("The screenplay is being rewritten for the third time.")
        assert result.stage == ProductionStage.DEVELOPMENT


# ===================================================================
# Output format tests
# ===================================================================

class TestOutputFormat:
    """Ensure the JSON output format matches the spec exactly."""

    def test_json_has_three_keys(self):
        result = classify("Principal photography is underway.")
        output = json.loads(result.to_json())
        assert set(output.keys()) == {"reasoning", "stage", "confidence"}

    def test_reasoning_is_first_key(self):
        """The 'reasoning' key must appear FIRST in the JSON output."""
        result = classify("Principal photography is underway.")
        output_str = result.to_json()
        reasoning_pos = output_str.index('"reasoning"')
        stage_pos = output_str.index('"stage"')
        confidence_pos = output_str.index('"confidence"')
        assert reasoning_pos < stage_pos < confidence_pos

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
