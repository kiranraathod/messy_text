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
# Single text
uv run messy-text "The screenplay is being rewritten while producers seek financing."

# Pipe from stdin
echo "Cameras are rolling in Atlanta." | uv run messy-text

# Interactive mode
uv run messy-text

# Batch mode
uv run messy-text --batch input.jsonl > output.jsonl
```

## Batch Mode

Each line in the batch input should be a JSON object with a `"text"` key. Plain-text lines are also accepted and treated as raw input.

```jsonl
{"text": "The screenplay is in its fourth draft."}
{"text": "Principal photography began in Vancouver."}
Grilled salmon with lemon butter sauce.
```

Successful rows keep the classification payload:

```jsonl
{"line": 1, "input": "The screenplay is in its fourth draft.", "reasoning": "...", "stage": "DEVELOPMENT", "confidence": 0.9}
{"line": 2, "input": "Principal photography began in Vancouver.", "reasoning": "...", "stage": "PRODUCTION", "confidence": 0.95}
```

Operational failures are emitted explicitly instead of being misreported as `UNCLASSIFIABLE`:

```jsonl
{"line": 3, "input": "An ambiguous project update", "error": "GROQ_API_KEY environment variable is not set.", "error_type": "configuration_error"}
```

Supported `error_type` values are `configuration_error`, `provider_error`,
`response_format_error`, and `input_validation_error`.

## Architecture

```text
classify(text)
    |
    |- empty input -> UNCLASSIFIABLE
    |- exact phrase fast-path -> direct result
    \- LLM adapter -> Groq JSON classification
```

- The regex fast-path only handles a few exact phrases such as `principal photography` and `officially greenlit`.
- All broader policy and edge-case decisions live in the LLM prompt.
- Provider, configuration, and response-format failures are surfaced as explicit operational errors in the CLI.

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required for ambiguous text)* | Groq API key |
| `MESSY_TEXT_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Model override |
| `MESSY_TEXT_CONFIDENCE_THRESHOLD` | `0.5` | Confidence floor below which LLM results become `UNCLASSIFIABLE` |
| `MESSY_TEXT_MAX_INPUT_CHARS` | `2000` | Maximum input length before classification rejects the record |
| `MESSY_TEXT_BATCH_WORKERS` | `4` | Number of batch worker threads |

## Tests

```bash
uv run pytest -v
uv run pytest --cov
```
