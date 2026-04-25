"""Hybrid classification engine for film/TV production stage detection.

Architecture:
    1. _fast_regex_router() — Ultra-fast pre-filter for obvious cases.
       Catches exact high-confidence phrases to skip the LLM entirely.
    2. _llm_classify() — Primary classifier using Groq (llama-4-scout).
       Handles all ambiguous, messy, or multi-signal text.
"""

from __future__ import annotations

import json
import os
import re

from groq import Groq

from messy_text.models import ClassificationResult, ProductionStage


# ---------------------------------------------------------------------------
# System prompt — canonical taxonomy for the LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert data pipeline system for the film and television industry. Your task is to classify messy, unstructured text into an exact production stage.

CRITICAL INSTRUCTION: Keep the classification and reasoning simple. Avoid unnecessary complexity, over-analyzing the text, or making assumptions beyond what is explicitly stated. 
CRITICAL INSTRUCTION: You must output a valid JSON object. To ensure accuracy, you must generate the "reasoning" key FIRST, before you output the "stage" or "confidence" keys.

Taxonomy:
1. DEVELOPMENT: Exists on paper. Rights optioned, pitching, seeking financing, or attaching talent. (Note: "turnaround" or "development hell" remains DEVELOPMENT).
2. PRE_PRODUCTION: Officially "greenlit" and definitively funded. Hiring crew (department heads), scouting locations, setting shoot dates.
3. PRODUCTION: Active filming. Principal photography, cameras rolling, on set/location.
4. UNCLASSIFIABLE: Irrelevant to film, or lacks chronological markers.

Edge Case Rules:
- "Attached talent" alone is DEVELOPMENT, not Pre-Production.
- If multiple stages are mentioned (e.g., "After 5 years in development, we started shooting today"), the LATEST chronological event takes absolute precedence.

Output JSON exactly like this:
{
  "reasoning": "A simple 1-2 sentence logical deduction.",
  "stage": "DEVELOPMENT" | "PRE_PRODUCTION" | "PRODUCTION" | "UNCLASSIFIABLE",
  "confidence": 0.95
}
""".strip()

GROQ_MODEL = os.environ.get("MESSY_TEXT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


# ---------------------------------------------------------------------------
# Fast regex router — dumb, cheap, and obvious-only
# ---------------------------------------------------------------------------

# Each tuple: (compiled regex, stage, confidence, reasoning)
_FAST_ROUTES: list[tuple[re.Pattern[str], ProductionStage, float, str]] = [
    # --- PRODUCTION: unmistakable active-shooting phrases ---
    (re.compile(r"\bprincipal\s+photography\b", re.IGNORECASE),
     ProductionStage.PRODUCTION, 0.95,
     "The text explicitly mentions 'principal photography', indicating active filming."),

    (re.compile(r"\bcameras?\s+(?:are\s+)?roll(?:s|ing|ed)\b", re.IGNORECASE),
     ProductionStage.PRODUCTION, 0.95,
     "The text mentions cameras rolling, indicating active filming."),

    (re.compile(r"\bcurrently\s+(?:filming|shooting)\b", re.IGNORECASE),
     ProductionStage.PRODUCTION, 0.95,
     "The text states filming/shooting is currently happening."),

    (re.compile(r"\bday\s+\d+\s+of\s+(?:filming|shooting|production)\b", re.IGNORECASE),
     ProductionStage.PRODUCTION, 0.95,
     "The text references a specific day of an active shoot."),

    # --- PRE_PRODUCTION: definitive greenlight ---
    (re.compile(r"\bofficially\s+greenl(?:it|ight(?:ed)?)\b", re.IGNORECASE),
     ProductionStage.PRE_PRODUCTION, 0.95,
     "The text explicitly states the project is officially greenlit."),
]


def _fast_regex_router(text: str) -> ClassificationResult | None:
    """Ultra-fast pre-filter for obvious, high-confidence cases.

    Returns a ClassificationResult if an unmistakable phrase is found,
    or None if the text needs full LLM classification.
    """
    for pattern, stage, confidence, reasoning in _FAST_ROUTES:
        if pattern.search(text):
            return ClassificationResult(
                reasoning=reasoning,
                stage=stage,
                confidence=confidence,
            )
    return None


# ---------------------------------------------------------------------------
# LLM classifier — primary engine via Groq
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """Lazy-initialize the Groq client."""
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Set it to use LLM classification: export GROQ_API_KEY=gsk_..."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _llm_classify(text: str) -> ClassificationResult:
    """Send text to Groq's llama-4-scout for classification.

    Uses response_format={"type": "json_object"} to guarantee
    valid JSON output from the model. Catches API and parsing errors
    to ensure pipeline reliability.
    """
    client = _get_groq_client()

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        return ClassificationResult(
            reasoning=data.get("reasoning", "LLM reasoning missing."),
            stage=ProductionStage(data.get("stage", "UNCLASSIFIABLE")),
            confidence=float(data.get("confidence", 0.0)),
        )

    except Exception as e:
        # Catch network timeouts, rate limits, or JSON parsing errors
        return ClassificationResult(
            reasoning=f"System fallback due to LLM failure: {str(e)}",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(text: str) -> ClassificationResult:
    """Classify messy text into a production stage.

    Pipeline:
    1. Empty check → UNCLASSIFIABLE.
    2. Fast regex router → instant result for obvious phrases.
    3. LLM fallback → Groq llama-4-scout for everything else.
    """
    if not text or not text.strip():
        return ClassificationResult(
            reasoning="The input text is empty. No classification is possible.",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=1.0,
        )

    normalized = text.strip()

    # Step 1: Try fast regex router
    fast_result = _fast_regex_router(normalized)
    if fast_result is not None:
        return fast_result

    # Step 2: LLM classification
    return _llm_classify(normalized)
