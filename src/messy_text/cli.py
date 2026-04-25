"""CLI entrypoint for the messy-text classifier."""

from __future__ import annotations

import json
import sys

from messy_text.classifier import classify


def main() -> None:
    """Run the classifier on stdin or a provided argument.

    Usage:
        messy-text "Some text about a film project..."
        echo "Some text" | messy-text
        messy-text --batch input.jsonl > output.jsonl
    """
    # Batch mode: process JSONL (one JSON object per line with a "text" key)
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Usage: messy-text --batch <input.jsonl>", file=sys.stderr)
            sys.exit(1)
        _run_batch(sys.argv[2])
        return

    # Single text from argument
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = classify(text)
        print(result.to_json())
        return

    # Read from stdin
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            result = classify(text)
            print(result.to_json())
            return

    # Interactive mode
    print("messy-text classifier — enter text to classify (Ctrl+C to exit):", file=sys.stderr)
    try:
        while True:
            print("\n> ", end="", file=sys.stderr, flush=True)
            line = input()
            if line.strip():
                result = classify(line)
                print(result.to_json())
    except (KeyboardInterrupt, EOFError):
        print("\nBye!", file=sys.stderr)


def _run_batch(filepath: str) -> None:
    """Process a JSONL file, classifying each line."""
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get("text", "")
            except json.JSONDecodeError:
                # Treat the line itself as raw text
                text = line

            result = classify(text)
            output = {
                "line": line_num,
                "input": text[:100],
                **json.loads(result.to_json()),
            }
            print(json.dumps(output))


if __name__ == "__main__":
    main()
