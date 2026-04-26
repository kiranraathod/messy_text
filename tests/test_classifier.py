import json
import os
from unittest import mock

import pytest
from pydantic import ValidationError

from messy_text.classifier import classify
from messy_text.models import ClassificationResult, ProductionStage
from messy_text.__main__ import main


@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to set up a clean environment for testing."""
    monkeypatch.setenv("GROQ_API_KEY", "test_key_123")
    monkeypatch.setenv("MESSY_TEXT_MAX_INPUT_CHARS", "2000")


@pytest.fixture
def mock_groq(mocker):
    """Fixture to mock the Groq API client."""
    mock_client = mocker.patch("messy_text.classifier.Groq")
    mock_instance = mock_client.return_value
    mock_completions = mock_instance.chat.completions.create
    return mock_completions


# --- 1. Input Validation & Edge Cases ---

def test_empty_input():
    # Empty input should return UNCLASSIFIABLE immediately without API call
    result = classify("   \n \t  ")
    assert result.stage == ProductionStage.UNCLASSIFIABLE
    assert result.confidence == 1.0
    assert result.reliable is True
    assert "empty" in result.reasoning.lower()


def test_input_too_long(mock_env):
    # Exceeding the 2000 char default limit
    long_text = "a" * 2001
    with pytest.raises(ValueError, match="Input too long"):
        classify(long_text)


def test_extreme_formatting(mock_env, mock_groq):
    # Testing that weird characters don't crash the pre-processing
    mock_groq.return_value.choices[0].message.content = json.dumps({
        "reasoning": "Valid inference.",
        "stage": "PRODUCTION",
        "confidence": 0.9
    })
    weird_text = "🎬🎥 " * 100 + "\x00\x01\x02\n\t"
    result = classify(weird_text)
    assert result.stage == ProductionStage.PRODUCTION
    assert mock_groq.called


# --- 2. Environment Configuration Resilience ---

def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY environment variable is not set"):
        classify("Some valid text")


def test_invalid_max_chars_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    monkeypatch.setenv("MESSY_TEXT_MAX_INPUT_CHARS", "not_a_number")
    with pytest.raises(ValueError, match="Invalid MESSY_TEXT_MAX_INPUT_CHARS"):
        classify("Some valid text")


# --- 3. API Communication Resilience ---

def test_api_network_failure(mock_env, mock_groq):
    # Simulating a timeout or connection issue raised by Groq client
    mock_groq.side_effect = Exception("Connection Timeout")
    with pytest.raises(RuntimeError, match="Groq request failed: Connection Timeout"):
        classify("Some valid text")


def test_api_empty_response(mock_env, mock_groq):
    # API returns blank content string
    mock_groq.return_value.choices[0].message.content = "   "
    with pytest.raises(ValueError, match="Groq response content was empty"):
        classify("Some valid text")


# --- 4. Schema & LLM Output Validation ---

def test_malformed_json_response(mock_env, mock_groq):
    # LLM hallucinates plain text instead of JSON
    mock_groq.return_value.choices[0].message.content = "This is not JSON at all."
    with pytest.raises(ValueError, match="Groq response did not match the expected schema"):
        classify("Some valid text")


def test_missing_required_fields(mock_env, mock_groq):
    # LLM forgets to include 'stage' and 'confidence'
    mock_groq.return_value.choices[0].message.content = json.dumps({
        "reasoning": "Missing stage and confidence."
    })
    with pytest.raises(ValueError, match="Groq response did not match the expected schema"):
        classify("Some valid text")


def test_extra_fields_forbidden(mock_env, mock_groq):
    # Tests that the newly added model_config extra="forbid" works
    # If the LLM hallucinates extra keys, Pydantic should reject it
    mock_groq.return_value.choices[0].message.content = json.dumps({
        "reasoning": "All fields present but has extra.",
        "stage": "DEVELOPMENT",
        "confidence": 0.8,
        "hallucinated_extra_field": "some value",
        "reliable": True  # The LLM is explicitly forbidden from generating this
    })
    with pytest.raises(ValueError, match="Groq response did not match the expected schema"):
        classify("Some valid text")


def test_invalid_stage_enum(mock_env, mock_groq):
    # LLM returns a stage that doesn't exist in our ProductionStage Enum
    mock_groq.return_value.choices[0].message.content = json.dumps({
        "reasoning": "Invalid stage returned.",
        "stage": "POST_PRODUCTION", 
        "confidence": 0.9
    })
    with pytest.raises(ValueError, match="Groq response did not match the expected schema"):
        classify("Some valid text")


def test_invalid_confidence_type(mock_env, mock_groq):
    # LLM returns a string instead of a float for confidence
    mock_groq.return_value.choices[0].message.content = json.dumps({
        "reasoning": "Confidence is a string.",
        "stage": "DEVELOPMENT",
        "confidence": "very high"
    })
    with pytest.raises(ValueError, match="Groq response did not match the expected schema"):
        classify("Some valid text")


def test_confidence_out_of_bounds(mock_env, mock_groq):
    # LLM hallucinates a confidence score greater than 1.0
    mock_groq.return_value.choices[0].message.content = json.dumps({
        "reasoning": "Confidence is > 1.0",
        "stage": "DEVELOPMENT",
        "confidence": 1.5
    })
    with pytest.raises(ValueError, match="Groq response did not match the expected schema"):
        classify("Some valid text")


# --- 5. Core Logic Validation ---

def test_reliable_flag_true(mock_env, mock_groq):
    # confidence >= 0.85 -> reliable = True
    mock_groq.return_value.choices[0].message.content = json.dumps({
        "reasoning": "High confidence inference.",
        "stage": "PRODUCTION",
        "confidence": 0.85
    })
    result = classify("Cameras are rolling on set today.")
    assert result.stage == ProductionStage.PRODUCTION
    assert result.confidence == 0.85
    assert result.reliable is True


def test_reliable_flag_false(mock_env, mock_groq):
    # confidence < 0.85 -> reliable = False
    mock_groq.return_value.choices[0].message.content = json.dumps({
        "reasoning": "Low confidence inference.",
        "stage": "DEVELOPMENT",
        "confidence": 0.84
    })
    result = classify("Maybe we will attach a director soon.")
    assert result.stage == ProductionStage.DEVELOPMENT
    assert result.confidence == 0.84
    assert result.reliable is False


# --- 6. CLI Wrapper Tests ---

def test_cli_empty_args(monkeypatch, capsys):
    # Running CLI without stdin or args
    monkeypatch.setattr("sys.argv", ["messy-text"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    
    exit_code = main()
    assert exit_code == 1
    
    captured = capsys.readouterr()
    assert "Usage: messy-text <text>" in captured.err


def test_cli_success(monkeypatch, capsys, mocker):
    # Running CLI with valid positional args
    monkeypatch.setattr("sys.argv", ["messy-text", "Cameras", "rolling!"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    
    # Mock classify to isolate CLI logic from env/API dependencies
    mock_classify = mocker.patch("messy_text.__main__.classify")
    mock_classify.return_value = ClassificationResult(
        reasoning="Testing CLI.",
        stage=ProductionStage.PRODUCTION,
        confidence=0.99,
        reliable=True
    )
    
    exit_code = main()
    assert exit_code == 0
    
    captured = capsys.readouterr()
    output_json = json.loads(captured.out)
    assert output_json["stage"] == "PRODUCTION"
    assert output_json["reliable"] is True


def test_cli_exception_handling(monkeypatch, capsys, mocker):
    # Simulate an error during classification (e.g. timeout, missing key)
    monkeypatch.setattr("sys.argv", ["messy-text", "trigger error"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    
    mock_classify = mocker.patch("messy_text.__main__.classify")
    mock_classify.side_effect = ValueError("Simulated missing API key failure")
    
    exit_code = main()
    assert exit_code == 1  # Should exit with error status
    
    captured = capsys.readouterr()
    output_json = json.loads(captured.out)
    assert output_json["error"] == "Simulated missing API key failure"
    assert output_json["error_type"] == "valueerror"  # The specific exception type
