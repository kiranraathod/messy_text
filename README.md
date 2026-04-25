# messy-text

A hybrid AI pipeline that classifies messy, unstructured film/TV industry text into exact production stages. Designed for batch processing at scale.

## Taxonomy

| Stage | Description |
|---|---|
| **DEVELOPMENT** | Script/treatment phase, rights acquisition, pitching, financing, talent attachment |
| **PRE_PRODUCTION** | Greenlit & funded, hiring crew, scouting locations, scheduling shoot dates |
| **PRODUCTION** | Cameras rolling, principal photography, active filming on set/location |
| **UNCLASSIFIABLE** | Irrelevant text or insufficient chronological markers |

## Quick Start

```bash
# Install
uv sync

# Set your Groq API key
export GROQ_API_KEY=gsk_...           # Linux/Mac
$env:GROQ_API_KEY="gsk_..."          # PowerShell

# Classify a single text
uv run messy-text "After 5 years in development, we finally started shooting today."
```

## Output Format

```json
{
  "reasoning": "The latest chronological event is starting to shoot, which takes precedence.",
  "stage": "PRODUCTION",
  "confidence": 0.95
}
```

## CLI Usage

```bash
# Single text (argument)
uv run messy-text "The screenplay is being rewritten while producers seek financing."

# Pipe from stdin
echo "Cameras are rolling in Atlanta." | uv run messy-text

# Interactive mode
uv run messy-text

# Batch mode — process thousands of records via JSONL
uv run messy-text --batch input.jsonl > output.jsonl
```

### Batch Processing (JSONL)

The `--batch` flag is designed for high-throughput pipeline use. Each line of the input file should be a JSON object with a `"text"` key:

```jsonl
{"text": "The screenplay is in its fourth draft."}
{"text": "Principal photography began in Vancouver."}
{"text": "Grilled salmon with lemon butter sauce."}
```

Output is one JSON result per line, suitable for piping into downstream systems:

```jsonl
{"line": 1, "input": "The screenplay is in its fourth draft.", "reasoning": "...", "stage": "DEVELOPMENT", "confidence": 0.9}
{"line": 2, "input": "Principal photography began in Vancouver.", "reasoning": "...", "stage": "PRODUCTION", "confidence": 0.95}
{"line": 3, "input": "Grilled salmon with lemon butter sauce.", "reasoning": "...", "stage": "UNCLASSIFIABLE", "confidence": 0.99}
```

## Architecture: Hybrid Classification

```
classify(text)
    │
    ├─ Empty? → UNCLASSIFIABLE (confidence: 1.0)
    │
    ├─ _fast_regex_router()          ← 5 patterns, zero API cost
    │   • "principal photography"     → PRODUCTION
    │   • "cameras rolling"           → PRODUCTION
    │   • "currently filming"         → PRODUCTION
    │   • "day N of filming"          → PRODUCTION
    │   • "officially greenlit"       → PRE_PRODUCTION
    │
    └─ _llm_classify()               ← Groq llama-4-scout (primary)
        • response_format: json_object
        • temperature: 0.0
        • Graceful error handling (see below)
```

### Why Hybrid?

- **Cost-efficient**: Obvious phrases like "principal photography" are resolved instantly by the regex router — no API call, no latency, no cost. The LLM is reserved for genuinely ambiguous text.
- **Scalable**: In a batch of 10,000 snippets, a significant portion will short-circuit through the fast router, dramatically reducing API calls and total processing time.
- **Reliable**: The LLM path wraps all API and JSON parsing operations in a `try/except` block. If Groq times out, rate-limits, returns a 500/503, or produces malformed JSON, the system gracefully falls back to `UNCLASSIFIABLE` with `confidence: 0.0` — the pipeline never crashes.

### Edge Case Rules

- **"Attached talent"** alone → `DEVELOPMENT` (not Pre-Production)
- **Multiple stages mentioned** → the **latest chronological event** wins
- **"Development hell" / "turnaround"** → remains `DEVELOPMENT`

## Project Structure

```
src/messy_text/
├── models.py       # Pydantic schema (reasoning → stage → confidence)
├── classifier.py   # Fast regex router + Groq LLM hybrid engine
└── cli.py          # CLI: single, stdin, batch (JSONL), interactive

tests/
└── test_classifier.py  # 29 tests (router, LLM mock, error handling, output format)
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key for LLM classification |
| `MESSY_TEXT_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Model override |

## Tests

```bash
uv run pytest -v          # 29 tests (router tests need no API key)
uv run pytest --cov       # with coverage
```


