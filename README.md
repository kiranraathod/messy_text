# messy-text

## Introduction

This project was built as a response to the AI Systems Engineer Final Assessment Task.
It evaluates the ability to apply AI in a practical and reliable way.

## What We're Building

At Location HQ, AI is used to interpret messy film and television production data and generate reliable signals. This system classifies production stage from unstructured, real-world text.

---

## The Task

Build a system using AI (or hybrid approaches) to classify production stage from messy text.

---

## Requirements

> Classify stage (development/pre-production/production), output confidence and short reasoning.

Every response returns four fields:

```json
{
  "stage": "PRODUCTION",
  "confidence": 0.98,
  "reasoning": "Explicit principal photography and cameras rolling = PRODUCTION.",
  "reliable": true
}
```

| Field | Type | Purpose |
|---|---|---|
| `stage` | enum | One of `DEVELOPMENT`, `PRE_PRODUCTION`, `PRODUCTION`, `UNCLASSIFIABLE` |
| `confidence` | float 0–1 | LLM's self-assessed certainty, anchored by explicit scoring guidance in the prompt |
| `reasoning` | string | One or two sentences referencing exact phrases from the input — never invented |
| `reliable` | bool | `true` if `confidence >= 0.85` — computed in Python, not asked of the LLM |

**Stage taxonomy:**

| Stage | What it covers |
|---|---|
| `DEVELOPMENT` | Script work, rights, pitching, financing, talent attachment, development hell, turnaround |
| `PRE_PRODUCTION` | Greenlit and funded, hiring key crew, location scouting, scheduling shoot dates |
| `PRODUCTION` | Active principal photography, cameras rolling, on-set activity, daily call sheets |
| `UNCLASSIFIABLE` | Post-production, marketing, release, or text with no clear stage signal |

---

## Constraints

> You may use LLMs. Keep it simple. Avoid unnecessary complexity.

The system is four files, zero unnecessary abstractions.

```
messy_text/
├── __init__.py       # version export
├── __main__.py       # CLI — stdin or argument, prints JSON, exits 0 or 1
├── classifier.py     # classify() — prompt, Groq call, validation, reliable gate
└── models.py         # ClassificationResult + ProductionStage Pydantic schema
```

---

## What We're Evaluating

### AI Usage

**Model:** `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API.

- Groq delivers sub-second inference for a 256-token response — appropriate for a near-real-time classification signal
- Llama 4 Scout handles informal, abbreviated, and typo-laden industry text reliably
- Groq's free tier means near-zero cost at prototype scale

The entire classification problem is delegated to the LLM via a structured system prompt. The LLM is not used for anything else — no summarisation, no entity extraction, no chaining.

Structured output is enforced at two layers:
1. `response_format={"type": "json_object"}` — the API rejects non-JSON before it reaches the application
2. `ClassificationResult.model_validate_json(raw)` — Pydantic validates schema, types, and `confidence` bounds on every response

```python
temperature=0.0   # deterministic — same input always produces the same classification
max_tokens=256    # tight bound — reasoning + JSON fits in ~100 tokens; 256 is generous
```

---

### Prompt Design

**Persona and task boundary:**
```
You are an expert film and television production coordinator. Your sole job is to
analyze messy, unstructured production text and classify it into exactly one stage.
```

**Anchored taxonomy** — each stage defined with explicit industry keywords, not vague descriptions.

**Rules for ambiguous cases:**
```
- If multiple stages appear, select the latest/most advanced chronological stage.
- "Talent attached" by itself = DEVELOPMENT.
- If no stage signals are present, assign UNCLASSIFIABLE with confidence < 0.5.
```

**Calibrated confidence scoring:**
```
>=0.85  strong, direct keywords or clear statements
0.5–0.84  reasonable inference from context
<0.5  genuinely ambiguous or conflicting signals
```

**8 few-shot examples** covering every stage and edge case: vague text, talent attachment, greenlit-and-scouting, active shoot, theatrical release, post-production, **and multi-stage transition** (dev hell → greenlit + scouting).

---

### Handling Uncertainty

Uncertainty is handled at four distinct layers:

**Layer 1 — Input guard** (before any API call)

Empty or whitespace-only inputs are short-circuited immediately. No token is spent.

```python
if not text or not text.strip():
    return ClassificationResult(
        reasoning="The input text is empty. No classification is possible.",
        stage=ProductionStage.UNCLASSIFIABLE,
        confidence=1.0,
        reliable=True,
    )
```

**Layer 2 — UNCLASSIFIABLE as a first-class stage**

The LLM is never forced to choose between three stages. `UNCLASSIFIABLE` is the correct answer for post-production, marketing, release content, or any text with no clear signal. This prevents hallucinated classifications on weak data.

**Layer 3 — Calibrated confidence**

The prompt anchors confidence ranges explicitly. A `confidence < 0.5` result means the LLM saw almost no signal — consumers should treat it as uncertain regardless of the returned stage.

**Layer 4 — `reliable` flag**

`confidence` is a float that requires interpretation. `reliable` makes that interpretation automatic and machine-readable:

```python
_RELIABILITY_THRESHOLD = 0.85

result = ClassificationResult.model_validate_json(raw)
return result.model_copy(update={"reliable": result.confidence >= _RELIABILITY_THRESHOLD})
```

`reliable` is computed in Python after validation — it cannot be hallucinated. A downstream pipeline can filter on `reliable: false` to route uncertain results for human review.

---

### Cost and Reliability Thinking

**Cost:**

| Control | Mechanism |
|---|---|
| Token spend per call | `max_tokens=256` — hard ceiling on every request |
| Runaway input | Input rejected before the API call if > 2000 chars |
| Model choice | Groq Llama 4 Scout — fast and cost-effective for classification |
| Empty input | No API call made for empty/whitespace inputs |

**Reliability:**

| Risk | Mitigation |
|---|---|
| Hung API request | `timeout=10.0` on the Groq client — blocked requests are killed cleanly |
| Transient API failure | `max_retries=3` built into the Groq client |
| Malformed LLM JSON | `response_format={"type": "json_object"}` enforced at the API level |
| Wrong schema or types | Pydantic `model_validate_json` validates every response |
| Silently bad classification | `reliable` flag surfaces low-confidence results |
| Missing API key | Explicit `ValueError` raised before any network call |

All errors surface as typed exceptions caught by the CLI and emitted as structured JSON — the process always exits with a meaningful signal, never a Python traceback.

---

## Submission

### Quick Start

```bash
# Install dependencies
uv sync

# Configure
cp .env.template .env
# fill in GROQ_API_KEY

# Classify a single text
uv run messy-text "Day 12 of principal photography. Cameras rolling on set in Toronto."
```

```json
{
  "reasoning": "Explicit principal photography and cameras rolling = PRODUCTION.",
  "stage": "PRODUCTION",
  "confidence": 0.98,
  "reliable": true
}
```

```bash
# Messy/noisy input — piped from stdin
echo "Scr1pt still bein re-written lol. financng talks r ongoing w/ 3 diff studios" | uv run messy-text
```

```json
{
  "reasoning": "Script rewrites and ongoing financing talks are DEVELOPMENT signals.",
  "stage": "DEVELOPMENT",
  "confidence": 0.88,
  "reliable": true
}
```

### Confidence Interpretation

| Range | Meaning | Recommended action |
|---|---|---|
| `0.9 – 1.0` | Strong, unambiguous signal | Trust the result (`reliable: true`) |
| `0.85 – 0.89` | Clear inference from context | Trust the result (`reliable: true`) |
| `0.5 – 0.84` | Reasonable but indirect inference | Use with caution (`reliable: false`) |
| `0.0 – 0.49` | Weak or contradictory signals | Route for human review (`reliable: false`) |

### Configuration

All configuration is via environment variables. Copy `.env.template` to `.env` to get started.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `MESSY_TEXT_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | LLM model for classification |
| `MESSY_TEXT_MAX_INPUT_CHARS` | `2000` | Input length limit — longer inputs are rejected |

---

## Next Step

Live review and system challenge discussion.
