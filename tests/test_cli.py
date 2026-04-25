"""Tests for CLI behavior and batch output."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

from messy_text.classifier import ClassifierConfig, MAX_INPUT_CHARS
from messy_text.cli import _build_llm_classifier, _run_batch, main
from messy_text.errors import ConfigurationError, ProviderError
from messy_text.models import ClassificationResult, ProductionStage


def test_build_llm_classifier_uses_find_dotenv():
    mock_classifier = MagicMock()

    with patch("messy_text.cli.find_dotenv", return_value="custom.env") as mock_find, patch(
        "messy_text.cli.load_dotenv"
    ) as mock_load, patch(
        "messy_text.cli.GroqLLMClassifier",
        return_value=mock_classifier,
    ) as mock_builder:
        result = _build_llm_classifier()

    assert result is mock_classifier
    mock_find.assert_called_once_with(usecwd=True)
    mock_load.assert_called_once_with("custom.env")
    mock_builder.assert_called_once_with()

def test_batch_uses_raw_text_for_non_jsonl_input(tmp_path, capsys):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("Principal photography began in Vancouver.\n", encoding="utf-8")

    exit_code = _run_batch(
        str(input_path),
        llm_classifier=lambda text: None,
        classifier_config=ClassifierConfig(),
    )
    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 0
    assert output["input"] == "Principal photography began in Vancouver."
    assert output["stage"] == "PRODUCTION"


def test_batch_preserves_full_input_text(tmp_path, capsys):
    long_text = "Crew updates " + ("x" * 160)
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"text": long_text}) + "\n",
        encoding="utf-8",
    )

    exit_code = _run_batch(
        str(input_path),
        llm_classifier=lambda text: ClassificationResult(
            reasoning="The text is ambiguous but film-related.",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=0.4,
        ),
        classifier_config=ClassifierConfig(),
    )
    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 0
    assert output["input"] == long_text


def test_batch_emits_explicit_error_rows(tmp_path, capsys):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"text": "We're pitching the project to studios."}) + "\n",
        encoding="utf-8",
    )

    def raise_provider_error(text: str) -> ClassificationResult:
        raise ProviderError("Connection timed out")

    exit_code = _run_batch(
        str(input_path),
        llm_classifier=raise_provider_error,
        classifier_config=ClassifierConfig(),
    )
    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 1
    assert output["input"] == "We're pitching the project to studios."
    assert output["error_type"] == "provider_error"
    assert output["error"] == "Connection timed out"


def test_batch_emits_input_validation_error_rows(tmp_path, capsys):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"text": "x" * (MAX_INPUT_CHARS + 1)}) + "\n",
        encoding="utf-8",
    )

    exit_code = _run_batch(
        str(input_path),
        llm_classifier=lambda text: ClassificationResult(
            reasoning="unused",
            stage=ProductionStage.DEVELOPMENT,
            confidence=0.9,
        ),
        classifier_config=ClassifierConfig(),
    )
    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 1
    assert output["error_type"] == "input_validation_error"


def test_main_emits_json_error_for_single_input(capsys):
    def raise_configuration_error(text: str) -> ClassificationResult:
        raise ConfigurationError("GROQ_API_KEY environment variable is not set.")

    with patch.object(sys, "argv", ["messy-text", "Ambiguous input"]), patch(
        "messy_text.cli._build_llm_classifier",
        return_value=raise_configuration_error,
    ), patch(
        "messy_text.cli.load_classifier_config_from_env",
        return_value=ClassifierConfig(),
    ):
        exit_code = main()

    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 1
    assert output["error_type"] == "configuration_error"
    assert output["error"] == "GROQ_API_KEY environment variable is not set."


def test_main_emits_json_error_when_build_classifier_fails(capsys):
    with patch.object(sys, "argv", ["messy-text", "Ambiguous input"]), patch(
        "messy_text.cli._build_llm_classifier",
        side_effect=ConfigurationError("GROQ_API_KEY environment variable is not set."),
    ):
        exit_code = main()

    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 1
    assert output["error_type"] == "configuration_error"
    assert output["error"] == "GROQ_API_KEY environment variable is not set."


def test_main_emits_json_error_when_config_load_fails(capsys):
    with patch.object(sys, "argv", ["messy-text", "Ambiguous input"]), patch(
        "messy_text.cli._build_llm_classifier",
        return_value=MagicMock(),
    ), patch(
        "messy_text.cli.load_classifier_config_from_env",
        side_effect=ConfigurationError(
            "MESSY_TEXT_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0."
        ),
    ):
        exit_code = main()

    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 1
    assert output["error_type"] == "configuration_error"
    assert output["error"] == "MESSY_TEXT_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0."


def test_batch_uses_configured_thread_pool_and_preserves_order(
    tmp_path,
    capsys,
    monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "alpha"}),
                json.dumps({"text": "beta"}),
                json.dumps({"text": "gamma"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, int] = {}

    class FakeExecutor:
        def __init__(self, *, max_workers: int):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, iterable):
            for item in iterable:
                yield func(item)

    monkeypatch.setenv("MESSY_TEXT_BATCH_WORKERS", "2")
    with patch("messy_text.cli.ThreadPoolExecutor", FakeExecutor):
        exit_code = _run_batch(
            str(input_path),
            llm_classifier=lambda text: ClassificationResult(
                reasoning=f"Handled {text}",
                stage=ProductionStage.DEVELOPMENT,
                confidence=0.9,
            ),
            classifier_config=ClassifierConfig(),
        )

    outputs = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]

    assert exit_code == 0
    assert captured["max_workers"] == 2
    assert [row["input"] for row in outputs] == ["alpha", "beta", "gamma"]
