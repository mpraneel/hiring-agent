"""Grounded explanation of an already computed score.

The LLM is shown the computed breakdown and nothing else. It never sees the
resume, the job description, or any text it could mine for new claims, and it
never sees or produces the number's inputs. That is the whole point: an
explanation that cannot reach the source cannot justify a score it did not see,
and cannot quietly invent a skill the candidate never mentioned.

Failures propagate to the caller. The previous implementation caught its own
exceptions and returned the error string as the rationale, so the aggregator's
fallback never ran and raw provider exception text reached API clients.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..extraction.models import MatchExplanation
from ..extraction.provider import (
    StructuredProvider,
    StructuredProviderError,
    build_provider,
)
from ..normalize.normalizer import SkillNormalizer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You explain resume-to-job match scores to a recruiter.

You are given a score that has already been computed, together with the exact
skill lists it was computed from. Your job is to describe that result.

Rules:
- Reference only skills that appear in the lists you are given. Do not name any
  other skill, tool or technology, and do not speculate about what the candidate
  might also know.
- Do not restate the score as a different number, and do not argue that it
  should be higher or lower. The score is fixed.
- Write two or three sentences of rationale, in plain professional language.
- Give at most three suggestions. Every suggestion must name a skill from the
  missing must-have or missing nice-to-have list, and nothing else.
- If nothing is missing, return an empty suggestions list."""


@dataclass
class ScoreBreakdown:
    """Everything the explainer is allowed to know about a match."""

    score: float
    matched_must: List[str] = field(default_factory=list)
    matched_nice: List[str] = field(default_factory=list)
    missing_must: List[str] = field(default_factory=list)
    missing_nice: List[str] = field(default_factory=list)
    must_total: int = 0
    nice_total: int = 0
    extraction_path: str = "deterministic"

    @property
    def missing(self) -> List[str]:
        return list(self.missing_must) + list(self.missing_nice)

    def to_prompt(self) -> str:
        """Render the breakdown as the only context the model receives."""

        def render(label: str, values: List[str]) -> str:
            return f"{label}: {', '.join(values) if values else 'none'}"

        return "\n".join(
            [
                f"Match score: {self.score:.2f} ({self.score * 100:.0f} percent)",
                f"Must-have skills matched: {len(self.matched_must)} of {self.must_total}",
                f"Nice-to-have skills matched: {len(self.matched_nice)} of {self.nice_total}",
                "",
                render("Matched must-have", self.matched_must),
                render("Matched nice-to-have", self.matched_nice),
                render("Missing must-have", self.missing_must),
                render("Missing nice-to-have", self.missing_nice),
                "",
                f"Skills were extracted by the {self.extraction_path} path.",
            ]
        )


class LLMOverlay:
    """Produces a validated, grounded explanation of a score breakdown."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        structured_provider: Optional[StructuredProvider] = None,
        normalizer: Optional[SkillNormalizer] = None,
    ) -> None:
        self.normalizer = normalizer or SkillNormalizer()
        if structured_provider is not None:
            self._provider = structured_provider
        else:
            self._provider = build_provider(provider=provider, model=model)

    def explain(self, breakdown: ScoreBreakdown) -> MatchExplanation:
        """Return a validated explanation, or raise.

        Raising is deliberate. The aggregator decides what an unavailable
        explanation means for the response; this layer must not paper over it.
        """
        raw = self._provider.generate_json(
            SYSTEM_PROMPT, breakdown.to_prompt(), MatchExplanation
        )
        try:
            explanation = MatchExplanation.model_validate_json(raw)
        except Exception as exc:
            raise StructuredProviderError(f"explanation failed validation: {exc}") from exc

        explanation.suggestions = self._filter_suggestions(
            explanation.suggestions, breakdown
        )
        return explanation

    # -- grounding check --------------------------------------------------

    def _filter_suggestions(
        self, suggestions: List[str], breakdown: ScoreBreakdown
    ) -> List[str]:
        """Drop suggestions that name a skill which is not actually missing.

        Telling a candidate to add a skill they already listed, or one the job
        never asked for, is worse than saying nothing.
        """
        missing = {s.lower() for s in breakdown.missing}
        kept: List[str] = []

        for suggestion in suggestions:
            mentioned = self._skills_mentioned(suggestion)
            stray = mentioned - missing
            if stray:
                logger.warning(
                    "dropping suggestion naming skills that are not missing: %s",
                    ", ".join(sorted(stray)),
                )
                continue
            kept.append(suggestion)

        return kept[:3]

    def _skills_mentioned(self, text: str) -> set[str]:
        """Canonical skills named in a piece of text."""
        lowered = text.lower()
        found: set[str] = set()
        for canonical, variants in self.normalizer.ontology.items():
            for surface in (canonical, *variants):
                if self.normalizer.is_ambiguous(surface):
                    # Too short to identify from prose; the stoplist exists for
                    # exactly this reason.
                    continue
                pattern = r"(?<![a-z0-9])" + re.escape(surface.lower()) + r"(?![a-z0-9])"
                if re.search(pattern, lowered):
                    found.add(canonical)
                    break
        return found
