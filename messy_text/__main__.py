"""Minimal CLI for messy-text classifier."""

import json
import sys

from dotenv import load_dotenv
from messy_text.classifier import classify

def main() -> int:
    load_dotenv()  # one-liner, no find_dotenv gymnastics

    # stdin or single argument
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        text = " ".join(sys.argv[1:]).strip()

    if not text:
        print("Usage: messy-text <text>  or  pipe via stdin", file=sys.stderr)
        return 1

    try:
        result = classify(text)
        print(result.model_dump_json(indent=2))
        return 0
    except Exception as exc:  # keep one broad catch for CLI
        error_type = type(exc).__name__.lower()
        print(json.dumps({"error": str(exc), "error_type": error_type}))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
