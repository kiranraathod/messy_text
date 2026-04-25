"""CLI entrypoint for the messy-text classifier."""

from __future__ import annotations

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from dotenv import find_dotenv, load_dotenv

from messy_text.classifier import classify

DEFAULT_BATCH_WORKERS = 4


def main() -> int:
    """Run the classifier on stdin or a provided argument."""
    load_dotenv(find_dotenv(usecwd=True))

    try:
        batch_workers = int(
            os.environ.get("MESSY_TEXT_BATCH_WORKERS", str(DEFAULT_BATCH_WORKERS))
        )
        if batch_workers <= 0:
            raise ValueError("MESSY_TEXT_BATCH_WORKERS must be greater than 0.")
    except Exception as exc:
        print(json.dumps({"error": str(exc), "error_type": "configuration_error"}))
        return 1

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Usage: messy-text --batch <input.jsonl>", file=sys.stderr)
            return 1
        return _run_batch(sys.argv[2], batch_workers)

    if len(sys.argv) > 1:
        return _run_single(" ".join(sys.argv[1:]))

    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if not text:
            return 0
        return _run_single(text)

    return _run_interactive()


def _run_single(text: str) -> int:
    try:
        result = classify(text)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "error_type": "operational_error"}))
        return 1

    print(result.to_json())
    return 0


def _run_interactive() -> int:
    print("messy-text classifier - enter text to classify (Ctrl+C to exit):", file=sys.stderr)
    had_errors = False

    try:
        while True:
            print("\n> ", end="", file=sys.stderr, flush=True)
            line = input()
            if not line.strip():
                continue
            status = _run_single(line)
            had_errors = had_errors or status != 0
    except (KeyboardInterrupt, EOFError):
        print("\nBye!", file=sys.stderr)

    return 1 if had_errors else 0


def _run_batch(filepath: str, batch_workers: int) -> int:
    had_errors = False
    
    def process_line(line_num: int, raw_line: str) -> tuple[dict[str, object], bool]:
        line = raw_line.strip()
        if not line:
            return ({}, False)

        try:
            record = json.loads(line)
            if isinstance(record, dict):
                val = record.get("text", "")
                text = val if isinstance(val, str) else str(val)
            else:
                text = line
        except json.JSONDecodeError:
            text = line

        try:
            result = classify(text)
            return (
                {
                    "line": line_num,
                    "input": text,
                    **result.model_dump(mode="json"),
                },
                False,
            )
        except Exception as exc:
            return (
                {
                    "line": line_num,
                    "input": text,
                    "error": str(exc),
                    "error_type": "operational_error",
                },
                True,
            )

    with open(filepath, encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=batch_workers
    ) as executor:
        lines = list(enumerate(handle, 1))
        
        futures = [
            executor.submit(process_line, line_num, raw_line)
            for line_num, raw_line in lines
        ]
        
        for future in futures:
            output, has_error = future.result()
            if output:  # Skip empty lines
                had_errors = had_errors or has_error
                print(json.dumps(output))

    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
