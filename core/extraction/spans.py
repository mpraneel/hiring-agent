"""Locating evidence spans inside a source document.

The UI highlights the text that justifies each extracted skill. Character offsets
are computed here, on the backend, against the exact normalized text the API
returns, so the frontend never has to search for a span and can never disagree
with the backend about where one is.

Both extraction paths produce spans:

* the deterministic parser knows which surface form it matched, so the span is
  that form's position,
* the LLM supplies an evidence string that is verbatim only after whitespace
  normalization, so it is located with a whitespace-flexible search.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from ..normalize.normalizer import SkillNormalizer
from ..schemas.models import SkillSpan


def normalize_source(text: str) -> str:
    """Collapse whitespace within each line while preserving line breaks.

    This is the text whose offsets the API reports, and the text the UI renders.
    Both sides therefore agree on every index.
    """
    if not text:
        return ""
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        lines.append(re.sub(r"[^\S\n]+", " ", raw).strip())
    return "\n".join(lines).strip()


def _flexible_pattern(needle: str) -> Optional[re.Pattern[str]]:
    """Match a phrase allowing any whitespace between its words."""
    words = [w for w in re.split(r"\s+", needle.strip()) if w]
    if not words:
        return None
    return re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)


def locate(source: str, needle: str) -> Optional[Tuple[int, int]]:
    """Return the offsets of needle in source, tolerating whitespace differences."""
    pattern = _flexible_pattern(needle)
    if pattern is None:
        return None
    match = pattern.search(source)
    return (match.start(), match.end()) if match else None


def locate_token(source: str, token: str) -> Optional[Tuple[int, int]]:
    """Locate a skill surface form on a word boundary.

    Boundaries are hand written rather than using \\b because skill names end in
    characters \\b treats as non-word: c++, c#, node.js.
    """
    escaped = re.escape(token)
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])", re.IGNORECASE
    )
    match = pattern.search(source)
    return (match.start(), match.end()) if match else None


def line_containing(source: str, start: int) -> str:
    """The full line that the given offset falls on, used as readable evidence."""
    if start < 0 or start > len(source):
        return ""
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", start)
    if line_end == -1:
        line_end = len(source)
    return source[line_start:line_end].strip()


def spans_for_skills(
    source: str,
    skills: Sequence[Tuple[str, Optional[str]]],
    normalizer: SkillNormalizer,
) -> List[SkillSpan]:
    """Build spans for deterministically extracted skills.

    Each skill is located by whichever of its surface forms actually appears in
    the document, preferring the longest so that "react native" wins over
    "react" when both could match.
    """
    result: List[SkillSpan] = []

    for skill, priority in skills:
        surfaces = [skill, *normalizer.get_skill_variants(skill)]
        # Longest first: a longer surface form is the more specific evidence.
        surfaces = sorted({s for s in surfaces if s}, key=len, reverse=True)

        located: Optional[Tuple[int, int]] = None
        for surface in surfaces:
            located = locate_token(source, surface)
            if located is not None:
                break

        if located is None:
            # The skill was extracted but its surface form is not findable in the
            # normalized text. Emit it without a span rather than dropping it, so
            # the score and the chip list stay consistent.
            result.append(
                SkillSpan(skill=skill, priority=priority, evidence="", start=None, end=None)
            )
            continue

        start, end = located
        result.append(
            SkillSpan(
                skill=skill,
                priority=priority,
                evidence=line_containing(source, start),
                start=start,
                end=end,
            )
        )

    return result


def spans_from_evidence(
    source: str,
    items: Sequence[Tuple[str, Optional[str], str]],
) -> List[SkillSpan]:
    """Build spans for LLM extracted skills from their evidence strings."""
    result: List[SkillSpan] = []
    for skill, priority, evidence in items:
        located = locate(source, evidence) if evidence else None
        if located is None:
            result.append(
                SkillSpan(
                    skill=skill, priority=priority, evidence=evidence, start=None, end=None
                )
            )
            continue
        start, end = located
        result.append(
            SkillSpan(
                skill=skill,
                priority=priority,
                evidence=source[start:end],
                start=start,
                end=end,
            )
        )
    return result
