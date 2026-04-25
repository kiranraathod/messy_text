# messy-text

A hybrid data pipeline that classifies messy, unstructured film/TV industry text into exact production stages using a fast regex router + Groq LLM.

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

# Set your Groq API key (required for LLM classification)
export GROQ_API_KEY=gsk_...           # Linux/Mac
$env:GROQ_API_KEY="gsk_..."          # PowerShell

# Classify a single text
uv run messy-text "After 5 years in development, we finally started shooting today."

# Interactive mode
uv run messy-text

# Batch mode (JSONL)
uv run messy-text --batch input.jsonl > output.jsonl

# Web UI
uv run messy-text-server
# → http://localhost:8080
```

## Output Format

```json
{
  "reasoning": "The text says filming started, indicating active production.",
  "stage": "PRODUCTION",
  "confidence": 0.95
}
```

## Architecture

```
classify(text)
    │
    ├─ Empty? → UNCLASSIFIABLE
    │
    ├─ _fast_regex_router()     ← 5 patterns, zero API cost
    │   "principal photography"  → PRODUCTION
    │   "cameras rolling"        → PRODUCTION
    │   "currently filming"      → PRODUCTION
    │   "day N of filming"       → PRODUCTION
    │   "officially greenlit"    → PRE_PRODUCTION
    │
    └─ _llm_classify()          ← Groq llama-4-scout (primary)
        response_format: json_object
        temperature: 0.0
```

### Edge Case Rules

- **"Attached talent"** alone → `DEVELOPMENT` (not Pre-Production)
- **Multiple stages mentioned** → the **latest chronological event** wins
- **"Development hell" / "turnaround"** → remains `DEVELOPMENT`

### File Structure

```
src/messy_text/
├── models.py      # Pydantic schema (reasoning → stage → confidence)
├── classifier.py  # Fast router + Groq LLM hybrid
├── cli.py         # CLI: single, stdin, batch, interactive
├── server.py      # HTTP server for web UI + /api/classify
└── web/
    └── index.html  # Dark-mode web interface
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `MESSY_TEXT_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Model override |
| `MESSY_TEXT_PORT` | `8080` | Web server port |

## Tests

```bash
uv run pytest -v          # 26 tests (router tests need no API key)
uv run pytest --cov       # with coverage
```

## License

MIT
