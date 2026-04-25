"""CLI entrypoint for the messy-text classifier."""

from __future__ import annotations

import os
import json
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

from dotenv import find_dotenv, load_dotenv

from messy_text.classifier import (
    ClassifierConfig,
    LLMClassifier,
    classify,
    load_classifier_config_from_env,
)
from messy_text.errors import ConfigurationError, StageClassifierOperationalError
from messy_text.providers import GroqLLMClassifier

DEFAULT_BATCH_WORKERS = 4


def main() -> int:
    """Run the classifier on stdin or a provided argument."""
    try:
        llm_classifier = _build_llm_classifier()
        classifier_config = load_classifier_config_from_env()
        batch_workers = load_batch_workers_from_env()
    except StageClassifierOperationalError as exc:
        print(json.dumps(exc.to_dict()))
        return 1

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Usage: messy-text --batch <input.jsonl>", file=sys.stderr)
            return 1
        return _run_batch(
            sys.argv[2],
            llm_classifier,
            classifier_config,
            batch_workers,
        )

    if len(sys.argv) > 1:
        return _run_single(" ".join(sys.argv[1:]), llm_classifier, classifier_config)

    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if not text:
            return 0
        return _run_single(text, llm_classifier, classifier_config)

    return _run_interactive(llm_classifier, classifier_config)


def _build_llm_classifier() -> GroqLLMClassifier:
    load_dotenv(find_dotenv(usecwd=True))
    return GroqLLMClassifier()


def load_batch_workers_from_env() -> int:
    try:
        batch_workers = int(
            os.environ.get("MESSY_TEXT_BATCH_WORKERS", str(DEFAULT_BATCH_WORKERS))
        )
    except ValueError as exc:
        raise ConfigurationError(
            "MESSY_TEXT_BATCH_WORKERS must be an integer."
        ) from exc

    if batch_workers <= 0:
        raise ConfigurationError(
            "MESSY_TEXT_BATCH_WORKERS must be greater than 0."
        )

    return batch_workers


def _run_single(
    text: str,
    llm_classifier: LLMClassifier,
    classifier_config: ClassifierConfig,
) -> int:
    try:
        result = classify(
            text,
            llm_classifier=llm_classifier,
            config=classifier_config,
        )
    except StageClassifierOperationalError as exc:
        print(json.dumps(exc.to_dict()))
        return 1

    print(result.to_json())
    return 0


def _run_interactive(
    llm_classifier: LLMClassifier,
    classifier_config: ClassifierConfig,
) -> int:
    print("messy-text classifier - enter text to classify (Ctrl+C to exit):", file=sys.stderr)
    had_errors = False

    try:
        while True:
            print("\n> ", end="", file=sys.stderr, flush=True)
            line = input()
            if not line.strip():
                continue
            status = _run_single(line, llm_classifier, classifier_config)
            had_errors = had_errors or status != 0
    except (KeyboardInterrupt, EOFError):
        print("\nBye!", file=sys.stderr)

    return 1 if had_errors else 0


def _run_batch(
    filepath: str,
    llm_classifier: LLMClassifier,
    classifier_config: ClassifierConfig,
    batch_workers: int,
) -> int:
    had_errors = False
    with open(filepath, encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=batch_workers
    ) as executor:
        # GroqLLMClassifier is stateless after __post_init__; sharing across threads is safe.
        for output, has_error in executor.map(
            lambda entry: _classify_batch_entry(
                entry,
                llm_classifier,
                classifier_config,
            ),
            _iter_batch_entries(handle),
        ):
            had_errors = had_errors or has_error
            print(json.dumps(output))

    return 1 if had_errors else 0


def _extract_text(line: str) -> str:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return line

    if isinstance(record, dict):
        value = record.get("text", "")
        return value if isinstance(value, str) else str(value)

    return line


def _iter_batch_entries(handle: Iterable[str]) -> Iterator[tuple[int, str]]:
    for line_num, raw_line in enumerate(handle, 1):
        line = raw_line.strip()
        if not line:
            continue

        yield (line_num, _extract_text(line))


def _classify_batch_entry(
    entry: tuple[int, str],
    llm_classifier: LLMClassifier,
    classifier_config: ClassifierConfig,
) -> tuple[dict[str, object], bool]:
    line_num, text = entry
    try:
        result = classify(
            text,
            llm_classifier=llm_classifier,
            config=classifier_config,
        )
    except StageClassifierOperationalError as exc:
        return (
            {
                "line": line_num,
                "input": text,
                **exc.to_dict(),
            },
            True,
        )

    return (
        {
            "line": line_num,
            "input": text,
            **result.model_dump(mode="json"),
        },
        False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
