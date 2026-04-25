"""Tests for CLI behavior and batch output."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from messy_text.cli import (
    _run_batch,
    main,
)
from messy_text.config import get_config
from messy_text.models import ClassificationResult, ProductionStage


@pytest.fixture(autouse=True)
def reset_config():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_batch_uses_raw_text_for_non_jsonl_input(tmp_path, capsys):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("Principal photography began in Vancouver.\n", encoding="utf-8")

    with patch("messy_text.cli.classify") as mock_classify:
        mock_classify.return_value = ClassificationResult(
            reasoning="Active filming.",
            stage=ProductionStage.PRODUCTION,
            confidence=0.95,
        )
        exit_code = _run_batch(str(input_path), batch_workers=4)
        
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

    with patch("messy_text.cli.classify") as mock_classify:
        mock_classify.return_value = ClassificationResult(
            reasoning="The text is ambiguous but film-related.",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=0.4,
        )
        exit_code = _run_batch(str(input_path), batch_workers=4)
        
    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 0
    assert output["input"] == long_text


def test_batch_emits_explicit_error_rows(tmp_path, capsys):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"text": "We're pitching the project to studios."}) + "\n",
        encoding="utf-8",
    )

    with patch("messy_text.cli.classify") as mock_classify:
        mock_classify.side_effect = RuntimeError("Connection timed out")
        exit_code = _run_batch(str(input_path), batch_workers=4)
        
    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 1
    assert output["input"] == "We're pitching the project to studios."
    assert output["error_type"] == "operational_error"
    assert output["error"] == "Connection timed out"


def test_main_emits_json_error_when_config_load_fails(capsys, monkeypatch):
    monkeypatch.setenv("MESSY_TEXT_BATCH_WORKERS", "abc")

    with patch.object(sys, "argv", ["messy-text", "--batch", "input.jsonl"]):
        exit_code = main()

    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 1
    assert output["error_type"] == "configuration_error"


def test_main_emits_json_error_for_single_input(capsys):
    with patch.object(sys, "argv", ["messy-text", "Ambiguous input"]), patch(
        "messy_text.cli.classify",
        side_effect=ValueError("GROQ_API_KEY environment variable is not set."),
    ):
        exit_code = main()

    output = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 1
    assert output["error_type"] == "operational_error"
    assert output["error"] == "GROQ_API_KEY environment variable is not set."


def test_batch_uses_configured_thread_pool_and_preserves_order(
    tmp_path,
    capsys,
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

        def submit(self, func, *args, **kwargs):
            future = MagicMock()
            future.result.return_value = func(*args, **kwargs)
            return future

    with patch("messy_text.cli.ThreadPoolExecutor", FakeExecutor), patch("messy_text.cli.classify") as mock_classify:
        mock_classify.return_value = ClassificationResult(
            reasoning="Handled",
            stage=ProductionStage.DEVELOPMENT,
            confidence=0.9,
        )
        exit_code = _run_batch(str(input_path), batch_workers=2)

    outputs = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]

    assert exit_code == 0
    assert captured["max_workers"] == 2
    assert [row["input"] for row in outputs] == ["alpha", "beta", "gamma"]

