"""Core classification logic for film/TV production stage detection."""

from __future__ import annotations

import os

from groq import Groq
from pydantic import ValidationError

from messy_text.models import ClassificationResult, ProductionStage

SYSTEM_PROMPT = """
You are an expert film and television production coordinator. Your sole job is to analyze messy, unstructured production text (emails, Slack messages, notes, call sheets, updates, etc.) and classify it into exactly one production stage.

Return ONLY a valid JSON object with these exact keys (no extra text, no markdown):
- reasoning: string - one short, concrete sentence (max 2 sentences) that references specific phrases or details directly from the input text. Never invent facts or use external knowledge.
- stage: string - exactly one of: "DEVELOPMENT", "PRE_PRODUCTION", "PRODUCTION", "UNCLASSIFIABLE"
- confidence: float - a number between 0.0 and 1.0

Stage definitions (apply strictly and chronologically):
- DEVELOPMENT: script writing/rewrites, rights acquisition, pitching, financing, talent/director attachment, development hell, turnaround.
- PRE_PRODUCTION: greenlit/funded, hiring key crew (DP, producers, etc.), location scouting, final casting, budgeting, scheduling shoot dates, prep/art department work.
- PRODUCTION: active principal photography, cameras rolling, shooting on set/location, daily call sheets, on-set activity, wrap reports.
- UNCLASSIFIABLE: post-production, marketing, distribution, release, or any text with no clear production-stage signals (too vague, irrelevant, or contradictory).

Critical rules for messy text:
- Interpret common industry shorthand/abbreviations: "dev", "pre-prod", "PP", "greenlight", "attached", "scouting locs", "principal photog", "cameras up", "rolling", "on set", "dailies", "wrap", "shoot day X", "call sheet", etc. Ignore typos and informal language.
- If multiple stages appear, select the latest/most advanced chronological stage mentioned.
- "Talent attached" by itself = DEVELOPMENT.
- Confidence scoring: >=0.85 for strong, direct keywords or clear statements; 0.5-0.84 for reasonable inference from context; <0.5 only for genuinely ambiguous or conflicting signals.
- Always ground the reasoning in the exact text provided. Never introduce facts or knowledge not present in the input text.
- If the input mentions multiple distinct projects (e.g. a sequel or spin-off), classify only the primary/main project being discussed.
- If no production-stage signals are present at all, assign UNCLASSIFIABLE with confidence < 0.5.

Examples (follow this output format and logic exactly):

Input: "Big week on the project. Lots happening, more to come!"
Output: {"reasoning": "No stage-specific language detected; text is too vague to classify.", "stage": "UNCLASSIFIABLE", "confidence": 0.35}

Input: "Just attached Tom Hanks to the lead role. Still working on financing with Warner Bros."
Output: {"reasoning": "Mentions attaching talent (Tom Hanks) and still seeking financing - matches DEVELOPMENT rules.", "stage": "DEVELOPMENT", "confidence": 0.85}

Input: "Greenlit! Hired the DP and scouting locations in Atlanta next week. Shoot starts in 3 weeks."
Output: {"reasoning": "Greenlit + hiring crew (DP) + scouting + scheduled shoot dates = PRE_PRODUCTION.", "stage": "PRE_PRODUCTION", "confidence": 0.95}

Input: "Day 12 of principal photography. Cameras rolling on set in Toronto. Call sheet attached."
Output: {"reasoning": "Explicit principal photography and cameras rolling = PRODUCTION.", "stage": "PRODUCTION", "confidence": 0.98}

Input: "The movie is now in theaters. Great reviews!"
Output: {"reasoning": "Text is about theatrical release/marketing - no production stage signals.", "stage": "UNCLASSIFIABLE", "confidence": 0.90}

Input: "Still polishing the script. No updates yet."
Output: {"reasoning": "Script polishing suggests DEVELOPMENT, but no corroborating signals (financing, talent, rights) are present.", "stage": "DEVELOPMENT", "confidence": 0.75}

Input: "Post-production wrapping up, VFX shots look great."
Output: {"reasoning": "Post-production content has no match in the defined stages.", "stage": "UNCLASSIFIABLE", "confidence": 0.92}

Now classify the user's text following the rules above.
""".strip()

# Threshold above which a result is considered reliably actionable.
_RELIABILITY_THRESHOLD = 0.85


def classify(text: str) -> ClassificationResult:
    """Classify messy text into a production stage."""
    if not text or not text.strip():
        return ClassificationResult(
            reasoning="The input text is empty. No classification is possible.",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=1.0,
            reliable=True,  # unambiguous special case — always reliable
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    model = os.environ.get("MESSY_TEXT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    try:
        max_chars = int(os.environ.get("MESSY_TEXT_MAX_INPUT_CHARS", "2000"))
    except ValueError as exc:
        raise ValueError(f"Invalid MESSY_TEXT_MAX_INPUT_CHARS: {exc}") from exc

    normalized = text.strip()
    if len(normalized) > max_chars:
        raise ValueError(
            f"Input too long ({len(normalized)} chars). Maximum is {max_chars}."
        )

    # timeout=10.0 prevents hung requests from blocking indefinitely.
    client = Groq(api_key=api_key, max_retries=3, timeout=10.0)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": normalized},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc

    raw = response.choices[0].message.content

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Groq response content was empty.")

    try:
        # LLM JSON never includes `reliable` — default=False covers that.
        # model_copy injects the correct value based on the returned confidence.
        result = ClassificationResult.model_validate_json(raw)
        return result.model_copy(update={"reliable": result.confidence >= _RELIABILITY_THRESHOLD})
    except ValidationError as exc:
        message = exc.errors()[0]["msg"]
        raise ValueError(
            f"Groq response did not match the expected schema: {message}"
        ) from exc
