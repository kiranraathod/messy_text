"""Comprehensive test suite for the production stage classifier.

Tests cover:
- Clear-cut cases for each stage
- Edge cases from the spec (attached talent, temporal overrides)
- UNCLASSIFIABLE detection
- Confidence scoring sanity
- JSON output format
"""

from __future__ import annotations

import json

import pytest

from messy_text.classifier import classify
from messy_text.models import ClassificationResult, ProductionStage


# ===================================================================
# DEVELOPMENT stage tests
# ===================================================================

class TestDevelopment:
    """Tests for text that should classify as DEVELOPMENT."""

    def test_script_and_financing(self):
        result = classify("The screenplay is being rewritten while producers seek financing.")
        assert result.stage == ProductionStage.DEVELOPMENT

    def test_pitching_project(self):
        result = classify("We're pitching the project to Netflix and HBO this week.")
        assert result.stage == ProductionStage.DEVELOPMENT

    def test_rights_optioned(self):
        result = classify("Sony has optioned the rights to the bestselling novel.")
        assert result.stage == ProductionStage.DEVELOPMENT

    def test_development_hell(self):
        result = classify(
            "The project has been stuck in development hell for over a decade."
        )
        assert result.stage == ProductionStage.DEVELOPMENT

    def test_turnaround(self):
        result = classify("After the director left, the movie went into turnaround at Warner Bros.")
        assert result.stage == ProductionStage.DEVELOPMENT

    def test_attached_talent_alone(self):
        """Edge case: 'attached talent' alone does NOT mean Pre-Production."""
        result = classify("Chris Hemsworth is attached to star in the upcoming action film.")
        assert result.stage == ProductionStage.DEVELOPMENT

    def test_shopping_script(self):
        result = classify("The writer is shopping the script around town.")
        assert result.stage == ProductionStage.DEVELOPMENT

    def test_draft_writing(self):
        result = classify("We just got the third draft of the screenplay back from the writer.")
        assert result.stage == ProductionStage.DEVELOPMENT

    def test_investor_pitch(self):
        result = classify("The producers met with investors to discuss funding for the indie film.")
        assert result.stage == ProductionStage.DEVELOPMENT

    def test_adaptation(self):
        result = classify("They acquired the rights and are adapting the graphic novel into a feature.")
        assert result.stage == ProductionStage.DEVELOPMENT


# ===================================================================
# PRE_PRODUCTION stage tests
# ===================================================================

class TestPreProduction:
    """Tests for text that should classify as PRE_PRODUCTION."""

    def test_greenlit(self):
        result = classify("The series has been officially greenlit by the network.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_location_scouting(self):
        result = classify("The team is scouting locations in Prague for the period drama.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_hiring_crew(self):
        result = classify("They've started hiring crew and department heads for the shoot.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_shoot_date_set(self):
        result = classify("Production is scheduled to begin filming in March 2027.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_series_order(self):
        result = classify("Amazon has given a series order for the new sci-fi show.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_storyboarding(self):
        result = classify("The director is storyboarding the action sequences with the DP.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_rehearsals(self):
        result = classify("Cast rehearsals are underway at Pinewood Studios.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_set_construction(self):
        result = classify("Set construction has begun on the studio backlot.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_table_read(self):
        result = classify("The cast did a table read of the pilot episode last Tuesday.")
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_straight_to_series(self):
        result = classify("HBO gave it a straight-to-series order, bypassing the pilot stage.")
        assert result.stage == ProductionStage.PRE_PRODUCTION


# ===================================================================
# PRODUCTION stage tests
# ===================================================================

class TestProduction:
    """Tests for text that should classify as PRODUCTION."""

    def test_principal_photography(self):
        result = classify("Principal photography began last Monday in Vancouver.")
        assert result.stage == ProductionStage.PRODUCTION

    def test_cameras_rolling(self):
        result = classify("Cameras are rolling on the new Marvel film in Atlanta.")
        assert result.stage == ProductionStage.PRODUCTION

    def test_currently_filming(self):
        result = classify("The show is currently filming its second season in London.")
        assert result.stage == ProductionStage.PRODUCTION

    def test_on_set(self):
        result = classify("The actors were spotted on set in downtown LA today.")
        assert result.stage == ProductionStage.PRODUCTION

    def test_day_of_filming(self):
        result = classify("Day 42 of filming and the crew is exhausted but pushing through.")
        assert result.stage == ProductionStage.PRODUCTION

    def test_wrapped(self):
        result = classify("That's a wrap! The movie wrapped principal photography today.")
        assert result.stage == ProductionStage.PRODUCTION

    def test_in_production(self):
        result = classify("The film is currently in production in New Zealand.")
        assert result.stage == ProductionStage.PRODUCTION

    def test_started_shooting(self):
        result = classify("They started shooting the pilot episode in Brooklyn.")
        assert result.stage == ProductionStage.PRODUCTION

    def test_set_photos_leaked(self):
        result = classify("Set photos from the new Batman film leaked on social media.")
        assert result.stage == ProductionStage.PRODUCTION

    def test_behind_the_scenes(self):
        result = classify("Behind-the-scenes footage shows the crew filming an intense chase scene.")
        assert result.stage == ProductionStage.PRODUCTION


# ===================================================================
# UNCLASSIFIABLE tests
# ===================================================================

class TestUnclassifiable:
    """Tests for text that should be UNCLASSIFIABLE."""

    def test_empty_string(self):
        result = classify("")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_whitespace_only(self):
        result = classify("   \n\t  ")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_restaurant_menu(self):
        result = classify("Grilled salmon with lemon butter sauce. Served with asparagus and rice.")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_weather_report(self):
        result = classify("Expect partly cloudy skies with a high of 72°F tomorrow.")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_random_gibberish(self):
        result = classify("asdfghjkl qwerty 12345 !!!???")
        assert result.stage == ProductionStage.UNCLASSIFIABLE

    def test_sports_news(self):
        result = classify("The Lakers defeated the Celtics 112-108 in overtime last night.")
        assert result.stage == ProductionStage.UNCLASSIFIABLE


# ===================================================================
# Edge case: Temporal override (latest event wins)
# ===================================================================

class TestTemporalOverrides:
    """The LATEST chronological event takes absolute precedence."""

    def test_dev_to_production_override(self):
        """'After 5 years in development, we finally started shooting today'
        should be PRODUCTION, not DEVELOPMENT."""
        result = classify(
            "After 5 years in development, we finally started shooting today."
        )
        assert result.stage == ProductionStage.PRODUCTION

    def test_dev_to_production_cameras_rolling(self):
        result = classify(
            "The project was stuck in development hell for ages, "
            "but cameras are now rolling in Atlanta."
        )
        assert result.stage == ProductionStage.PRODUCTION

    def test_dev_to_preprod_greenlit(self):
        result = classify(
            "After years of pitching and rewrites, the show has finally been greenlit."
        )
        assert result.stage == ProductionStage.PRE_PRODUCTION

    def test_mixed_signals_latest_wins(self):
        result = classify(
            "We wrote the script, scouted locations, and now filming is underway."
        )
        assert result.stage == ProductionStage.PRODUCTION

    def test_production_underway(self):
        result = classify(
            "Production has commenced on the highly anticipated sequel."
        )
        assert result.stage == ProductionStage.PRODUCTION


# ===================================================================
# Output format tests
# ===================================================================

class TestOutputFormat:
    """Ensure the JSON output format matches the spec exactly."""

    def test_json_has_three_keys(self):
        result = classify("The screenplay is in its fourth draft.")
        output = json.loads(result.to_json())
        assert set(output.keys()) == {"reasoning", "stage", "confidence"}

    def test_reasoning_is_first_key(self):
        """The 'reasoning' key must appear FIRST in the JSON output."""
        result = classify("Principal photography is underway.")
        output_str = result.to_json()
        # Find positions of keys
        reasoning_pos = output_str.index('"reasoning"')
        stage_pos = output_str.index('"stage"')
        confidence_pos = output_str.index('"confidence"')
        assert reasoning_pos < stage_pos < confidence_pos

    def test_stage_is_valid_enum(self):
        result = classify("Some random film text about a script rewrite.")
        output = json.loads(result.to_json())
        valid_stages = {"DEVELOPMENT", "PRE_PRODUCTION", "PRODUCTION", "UNCLASSIFIABLE"}
        assert output["stage"] in valid_stages

    def test_confidence_in_range(self):
        result = classify("Cameras rolling on the new project.")
        assert 0.0 <= result.confidence <= 1.0

    def test_reasoning_is_nonempty(self):
        result = classify("We're pitching the thriller to studios.")
        assert len(result.reasoning) > 0

    def test_json_round_trip(self):
        """Ensure JSON output can be parsed back into a valid dict."""
        result = classify("The film has been greenlit and crew is being hired.")
        output_str = result.to_json()
        parsed = json.loads(output_str)
        assert isinstance(parsed["reasoning"], str)
        assert isinstance(parsed["stage"], str)
        assert isinstance(parsed["confidence"], float)
