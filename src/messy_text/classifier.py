"""Core classification engine for film/TV production stage detection.

Uses a weighted-signal approach: each stage has a set of lexical patterns
(keywords, phrases, regex) with associated weights. The text is scanned
for all signals, and the stage with the highest cumulative weight wins.

The "latest chronological event" rule is enforced by checking for
temporal override patterns that explicitly indicate a later stage
has been reached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from messy_text.models import ClassificationResult, ProductionStage


# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Signal:
    """A single lexical signal that indicates a production stage."""

    pattern: str  # regex pattern (case-insensitive matching)
    weight: float = 1.0
    description: str = ""


@dataclass
class StageSignals:
    """Collection of signals for a single production stage."""

    stage: ProductionStage
    signals: list[Signal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pattern banks — intentionally simple and readable
# ---------------------------------------------------------------------------

DEVELOPMENT_SIGNALS = StageSignals(
    stage=ProductionStage.DEVELOPMENT,
    signals=[
        # Script / writing
        Signal(r"\bscript\b", 1.0, "script mention"),
        Signal(r"\bscreenplay\b", 1.2, "screenplay"),
        Signal(r"\btreatment\b", 1.0, "treatment document"),
        Signal(r"\bteleplay\b", 1.0, "teleplay"),
        Signal(r"\brewrite\b", 0.8, "script rewrite"),
        Signal(r"\bdraft\b", 0.9, "script draft"),
        Signal(r"\bwriter(?:s)?\b", 0.6, "writer attached"),
        Signal(r"\bwriting\b", 0.7, "writing phase"),
        # Rights / IP
        Signal(r"\brights\b", 1.0, "rights acquisition"),
        Signal(r"\boption(?:ed|ing)?\b", 1.0, "optioning rights"),
        Signal(r"\bacquir(?:e|ed|ing)\b", 0.8, "acquiring IP"),
        Signal(r"\badapt(?:ation|ed|ing)\b", 0.8, "adaptation"),
        Signal(r"\bIP\b", 0.7, "intellectual property"),
        # Financing / pitching
        Signal(r"\bpitch(?:ed|ing)?\b", 1.2, "pitching"),
        Signal(r"\bfinancing\b", 1.1, "seeking financing"),
        Signal(r"\bfund(?:ing|ed)?\b", 0.7, "funding discussion"),
        Signal(r"\bbudget\b", 0.5, "budget mention"),
        Signal(r"\binvestor(?:s)?\b", 0.9, "investor discussion"),
        Signal(r"\bpackag(?:e|ed|ing)\b", 0.9, "packaging project"),
        # Talent attachment (development indicator per edge-case rules)
        Signal(r"\battach(?:ed|ing|ment)?\b", 0.8, "talent attachment"),
        Signal(r"\bin\s+talks?\b", 0.7, "in talks"),
        Signal(r"\bset\s+to\s+(?:star|direct|produce)\b", 0.7, "talent set to join"),
        Signal(r"\beye(?:d|ing)\s+to\b", 0.6, "eyeing talent"),
        # Development limbo
        Signal(r"\bturnaround\b", 1.3, "project in turnaround"),
        Signal(r"\bdevelopment\s+hell\b", 1.5, "development hell"),
        Signal(r"\bin\s+development\b", 1.3, "explicitly in development"),
        Signal(r"\bshelved\b", 0.9, "project shelved"),
        # Studio / network interest
        Signal(r"\bshop(?:ping|ped)\b", 0.9, "shopping the project"),
        Signal(r"\bpilot\s+order\b", 0.5, "pilot order — could be pre-prod"),
        Signal(r"\bput\s+pilot\b", 0.6, "put pilot"),
        Signal(r"\bpilot\s+script\b", 1.0, "pilot script"),
    ],
)

PRE_PRODUCTION_SIGNALS = StageSignals(
    stage=ProductionStage.PRE_PRODUCTION,
    signals=[
        # Greenlight / funding confirmed
        Signal(r"\bgreenl(?:it|ight(?:ed)?)\b", 1.5, "project greenlit"),
        Signal(r"\bofficially\s+(?:approved|confirmed|ordered)\b", 1.2, "official approval"),
        Signal(r"\bseries\s+order\b", 1.2, "series order"),
        Signal(r"\bstraight[\s-]+to[\s-]+series\b", 1.3, "straight to series"),
        Signal(r"\bpicked\s+up\b", 1.0, "picked up"),
        Signal(r"\bordered\b", 0.6, "ordered"),
        # Crew / logistics
        Signal(r"\bhir(?:e|ed|ing)\s+(?:crew|staff|department)\b", 1.2, "hiring crew"),
        Signal(r"\bscouting\s+location(?:s)?\b", 1.3, "location scouting"),
        Signal(r"\blocation\s+scout(?:ing)?\b", 1.3, "location scouting"),
        Signal(r"\brecce\b", 1.2, "recce"),
        Signal(r"\bcasting\b", 0.8, "casting — shared with dev"),
        Signal(r"\baudition(?:s|ing|ed)?\b", 0.9, "auditions"),
        Signal(r"\bcrew(?:ing)?\b", 0.6, "crew mention"),
        Signal(r"\bpre[\s-]*production\b", 1.5, "explicitly pre-production"),
        Signal(r"\bpre[\s-]*prod\b", 1.3, "pre-prod shorthand"),
        # Scheduling
        Signal(r"\bshoot\s+date\b", 1.3, "shoot date set"),
        Signal(r"\bstart(?:s|ing)?\s+(?:filming|shooting|production)\b", 1.2, "start of filming"),
        Signal(r"\bset\s+to\s+(?:begin|start)\s+(?:filming|shooting|production)\b", 1.3, "set to begin filming"),
        Signal(r"\bproduction\s+(?:begins?|starts?)\b", 1.0, "production begins — could be prod"),
        Signal(r"\bscheduled\s+to\s+(?:shoot|film|begin)\s*(?:filming|shooting|production)?\b", 1.8, "scheduled to shoot"),
        Signal(r"\b(?:begin|start)\s+(?:filming|shooting|production)\b", 1.3, "begin filming — future tense"),
        # Design / planning
        Signal(r"\bstoryboard(?:s|ed|ing)?\b", 1.1, "storyboarding"),
        Signal(r"\bshot\s+list\b", 1.0, "shot list"),
        Signal(r"\bproduction\s+design\b", 1.0, "production design"),
        Signal(r"\bcostume\s+design\b", 0.9, "costume design"),
        Signal(r"\bset\s+(?:construction|build)\b", 1.1, "set construction"),
        Signal(r"\brehears(?:als?|e|ed|ing)\b", 1.0, "rehearsals"),
        Signal(r"\btable\s+read\b", 0.9, "table read"),
    ],
)

PRODUCTION_SIGNALS = StageSignals(
    stage=ProductionStage.PRODUCTION,
    signals=[
        # Active shooting — strongest signals
        Signal(r"\bprincipal\s+photography\b", 2.0, "principal photography"),
        Signal(r"\bcameras?\s+roll(?:s|ing|ed)?\b", 1.8, "cameras rolling"),
        Signal(r"\bcurrently\s+(?:filming|shooting)\b", 1.8, "currently filming"),
        Signal(r"\bon\s+set\b", 1.2, "on set"),
        Signal(r"\bon\s+location\b", 1.0, "on location"),
        Signal(r"\bfilming\b", 1.0, "filming"),
        Signal(r"\bshooting\b", 1.0, "shooting"),
        Signal(r"\bin\s+production\b", 1.3, "in production"),
        Signal(r"\bstarted\s+(?:filming|shooting)\b", 1.5, "started filming"),
        Signal(r"\bbegan\s+(?:filming|shooting)\b", 1.5, "began filming"),
        Signal(r"\bwrap(?:s|ped|ping)?\b", 1.3, "wrap — end of shoot"),
        Signal(r"\bday\s+\d+\s+of\s+(?:filming|shooting|production)\b", 1.8, "day N of filming"),
        Signal(r"\bset\s+photo(?:s)?\b", 1.2, "set photos"),
        Signal(r"\bbehind[\s-]+the[\s-]+scenes?\b", 1.0, "BTS content"),
        Signal(r"\baction!\b", 1.5, "action call"),
        Signal(r"\bcut!\b", 1.0, "cut call"),
        Signal(r"\bclapper\s*board\b", 1.2, "clapperboard"),
        Signal(r"\bslate\b", 0.7, "slate"),
    ],
)


ALL_STAGE_SIGNALS = [DEVELOPMENT_SIGNALS, PRE_PRODUCTION_SIGNALS, PRODUCTION_SIGNALS]

# Stage priority for the "latest chronological event wins" rule.
# Higher number = later in the production timeline.
STAGE_CHRONOLOGICAL_ORDER: dict[ProductionStage, int] = {
    ProductionStage.DEVELOPMENT: 0,
    ProductionStage.PRE_PRODUCTION: 1,
    ProductionStage.PRODUCTION: 2,
}

# Temporal override patterns — phrases that explicitly signal
# a transition FROM an earlier stage TO a later one.
# These give absolute priority to the later stage.
TEMPORAL_OVERRIDES: list[tuple[re.Pattern[str], ProductionStage, str]] = [
    # "finally started shooting", "we began filming", etc.
    (re.compile(r"(?:finally|now|just|already|today)\s+(?:started|began|beginning|starting)\s+(?:filming|shooting|production|principal\s+photography)", re.IGNORECASE),
     ProductionStage.PRODUCTION, "explicit transition to active shooting"),
    # "cameras are now rolling"
    (re.compile(r"cameras?\s+(?:are\s+)?(?:now\s+)?roll(?:s|ing|ed)", re.IGNORECASE),
     ProductionStage.PRODUCTION, "cameras actively rolling"),
    # "shooting is underway"
    (re.compile(r"(?:filming|shooting|production|principal\s+photography)\s+(?:is|has)\s+(?:now\s+)?(?:underway|begun|started|commenced)", re.IGNORECASE),
     ProductionStage.PRODUCTION, "shooting underway"),
    # "has been greenlit", "officially greenlit" → pre-prod
    (re.compile(r"(?:has\s+been|officially|just|finally)\s+greenl(?:it|ight(?:ed)?)", re.IGNORECASE),
     ProductionStage.PRE_PRODUCTION, "explicit greenlight"),
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass
class MatchedSignal:
    """A signal that was found in the input text."""

    stage: ProductionStage
    pattern: str
    weight: float
    description: str
    match_text: str


def _scan_signals(text: str) -> list[MatchedSignal]:
    """Scan text against all signal banks and return matched signals."""
    matches: list[MatchedSignal] = []
    for stage_signals in ALL_STAGE_SIGNALS:
        for signal in stage_signals.signals:
            for m in re.finditer(signal.pattern, text, re.IGNORECASE):
                matches.append(
                    MatchedSignal(
                        stage=stage_signals.stage,
                        pattern=signal.pattern,
                        weight=signal.weight,
                        description=signal.description,
                        match_text=m.group(0),
                    )
                )
    return matches


def _check_temporal_overrides(text: str) -> tuple[ProductionStage | None, str]:
    """Check for temporal override patterns that indicate the latest stage.

    Returns the overriding stage and its reasoning, or (None, "") if no override.
    """
    best_stage: ProductionStage | None = None
    best_reason = ""
    best_order = -1

    for pattern, stage, reason in TEMPORAL_OVERRIDES:
        if pattern.search(text):
            order = STAGE_CHRONOLOGICAL_ORDER[stage]
            if order > best_order:
                best_stage = stage
                best_reason = reason
                best_order = order

    return best_stage, best_reason


def classify(text: str) -> ClassificationResult:
    """Classify messy text into a production stage.

    Algorithm:
    1. Normalize and scan for all matching signals.
    2. Check for temporal override patterns (latest-event-wins rule).
    3. If an override is found, use that stage with high confidence.
    4. Otherwise, sum weights per stage and pick the highest.
    5. If no signals match, return UNCLASSIFIABLE.

    Returns a ClassificationResult with reasoning, stage, and confidence.
    """
    if not text or not text.strip():
        return ClassificationResult(
            reasoning="The input text is empty. No classification is possible.",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=1.0,
        )

    normalized = text.strip()

    # Step 1: Scan all signals
    matches = _scan_signals(normalized)

    # Step 2: Check temporal overrides
    override_stage, override_reason = _check_temporal_overrides(normalized)

    if override_stage is not None:
        # The latest chronological event takes precedence
        return ClassificationResult(
            reasoning=f"Temporal override detected: {override_reason}. "
            f"The latest chronological event indicates {override_stage.value}.",
            stage=override_stage,
            confidence=0.95,
        )

    # Step 3: If no matches at all → UNCLASSIFIABLE
    if not matches:
        return ClassificationResult(
            reasoning="No film production signals were detected in the text.",
            stage=ProductionStage.UNCLASSIFIABLE,
            confidence=0.9,
        )

    # Step 4: Aggregate weights per stage
    stage_weights: dict[ProductionStage, float] = {
        ProductionStage.DEVELOPMENT: 0.0,
        ProductionStage.PRE_PRODUCTION: 0.0,
        ProductionStage.PRODUCTION: 0.0,
    }
    stage_descriptions: dict[ProductionStage, list[str]] = {
        ProductionStage.DEVELOPMENT: [],
        ProductionStage.PRE_PRODUCTION: [],
        ProductionStage.PRODUCTION: [],
    }

    for m in matches:
        stage_weights[m.stage] += m.weight
        if m.description not in stage_descriptions[m.stage]:
            stage_descriptions[m.stage].append(m.description)

    # Step 5: Pick the winner
    # If multiple stages have signals, apply the chronological tiebreak:
    # among stages with "close" scores, the later stage wins.
    sorted_stages = sorted(
        stage_weights.items(),
        key=lambda x: (x[1], STAGE_CHRONOLOGICAL_ORDER[x[0]]),
        reverse=True,
    )

    best_stage, best_weight = sorted_stages[0]
    runner_up_stage, runner_up_weight = sorted_stages[1] if len(sorted_stages) > 1 else (None, 0.0)

    # If the runner-up is a LATER stage and within 40% of the winner's weight,
    # the later stage takes precedence (latest-event-wins edge case)
    if (
        runner_up_stage is not None
        and runner_up_weight > 0
        and STAGE_CHRONOLOGICAL_ORDER[runner_up_stage] > STAGE_CHRONOLOGICAL_ORDER[best_stage]
        and runner_up_weight >= best_weight * 0.6
    ):
        best_stage = runner_up_stage
        best_weight = runner_up_weight

    # Compute confidence from weight distribution
    total_weight = sum(stage_weights.values())
    if total_weight > 0:
        dominance = best_weight / total_weight
        # Scale confidence: pure dominance (1.0) → 0.95, split (0.33) → 0.55
        confidence = round(0.55 + (dominance - 0.33) * (0.40 / 0.67), 2)
        confidence = max(0.4, min(0.95, confidence))
    else:
        confidence = 0.5

    # Build reasoning from matched descriptions
    top_signals = stage_descriptions[best_stage][:3]
    signals_text = ", ".join(top_signals)
    reasoning = f"Key signals detected: {signals_text}. These indicate {best_stage.value}."

    return ClassificationResult(
        reasoning=reasoning,
        stage=best_stage,
        confidence=confidence,
    )
