# 🎬 messy-text

A deterministic data pipeline that classifies messy, unstructured film/TV industry text into exact production stages.

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
  "reasoning": "Temporal override detected: explicit transition to active shooting. The latest chronological event indicates PRODUCTION.",
  "stage": "PRODUCTION",
  "confidence": 0.95
}
```

## Edge Case Rules

- **"Attached talent"** alone → `DEVELOPMENT` (not Pre-Production)
- **Multiple stages mentioned** → the **latest chronological event** wins
- **"Development hell" / "turnaround"** → remains `DEVELOPMENT`

## Architecture

```
src/messy_text/
├── models.py      # Pydantic schema (reasoning → stage → confidence)
├── classifier.py  # Weighted-signal engine + temporal overrides
├── cli.py         # CLI: single, stdin, batch, interactive
├── server.py      # Stdlib HTTP server for web UI + API
└── web/
    └── index.html  # Dark-mode web interface
```

### How Classification Works

1. **Signal Scan** — Text is matched against 60+ regex patterns across 3 stage banks, each with calibrated weights
2. **Temporal Override** — Explicit phrases like "finally started shooting" give absolute priority to the latest stage
3. **Weight Aggregation** — Per-stage weights are summed; if a later stage is within 60% of the leader's weight, it wins (latest-event tiebreak)
4. **Confidence Scoring** — Derived from the winner's weight dominance ratio

## Tests

```bash
uv run pytest -v          # 47 tests
uv run pytest --cov       # with coverage
```

## License

MIT
