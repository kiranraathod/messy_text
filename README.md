# messy-text

A simple hybrid classifier for messy film and television production text.

It returns:

- `stage`: `DEVELOPMENT`, `PRE_PRODUCTION`, `PRODUCTION`, or `UNCLASSIFIABLE`
- `confidence`: a float from `0.0` to `1.0`
- `reasoning`: a short explanation grounded in the text

## Taxonomy

StageDescription**DEVELOPMENT**Script/treatment work, rights, pitching, financing, talent attachment**PRE_PRODUCTION**Greenlit and funded, hiring crew, scouting, scheduling shoot dates**PRODUCTION**Active filming, principal photography, cameras rolling**UNCLASSIFIABLE**Irrelevant text or not enough signal

## Quick Start

```bash
# Install dependencies
uv sync

# Set your Groq API key
export GROQ_API_KEY=gsk_...     # Linux/Mac
$env:GROQ_API_KEY="gsk_..."     # PowerShell

# Classify a single text
uv run messy-text "After 5 years in development, we finally started shooting today."
```

## Example Output

```json
{
  "reasoning": "The latest chronological event is starting to shoot.",
  "stage": "PRODUCTION",
  "confidence": 0.95
}
```

## CLI Usage

```bash
# Single text argument
uv run messy-text "The screenplay is being rewritten while producers seek financing."

# Pipe from stdin
echo "Cameras are rolling in Atlanta." | uv run messy-text
```

## Architecture

The system is a pure, minimalist wrapper around the Groq API enforcing a strict JSON schema via Pydantic. It consists of only four files:

- `__main__.py`: Minimal CLI entrypoint supporting stdin or arguments.
- `classifier.py`: The orchestrator handling the system prompt and API execution.
- `models.py`: Pydantic models for strict output validation.
- `__init__.py`: Version export.

## Configuration

The application is configured entirely via environment variables.

| Env Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key for LLM requests |
| `MESSY_TEXT_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | LLM model used for classification |
| `MESSY_TEXT_MAX_INPUT_CHARS` | `2000` | Maximum input length before classification is rejected |
