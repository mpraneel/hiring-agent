"""Deterministic job description parsing.

The parser walks the JD one line at a time, which is the whole reason line
structure is preserved through cleaning. Each line gets a priority from, in
order of precedence:

1. an explicit nice-to-have cue on the line ("preferred", "bonus", "a plus"),
2. an explicit must-have cue on the line ("required", "must have", "minimum"),
3. the nearest preceding section heading ("Requirements", "Nice to have"),
4. must-have, as the default for a JD with no structure at all.

Nice cues are checked before must cues on purpose. "Preferred: experience with
Docker" contains both a nice cue and a must-ish phrase, and the nice reading is
the correct one. The previous implementation checked must-have first against
pattern lists that shared seven entries with the nice-to-have list, so every
preferred qualification was silently promoted to a hard requirement.
"""

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..extraction.spans import locate_token
from ..normalize.normalizer import SkillNormalizer
from ..schemas.models import ParsedJD, SkillSpan

# Headings that set the priority for the lines beneath them.
MUST_HEADINGS: frozenset[str] = frozenset(
    {
        "requirements",
        "required",
        "required skills",
        "required qualifications",
        "must have",
        "must haves",
        "must-have",
        "must-haves",
        "qualifications",
        "minimum qualifications",
        "basic qualifications",
        "essential",
        "essential skills",
        "what you need",
        "what you will need",
        "what you'll need",
        "who you are",
        "skills",
        "technical skills",
        "your experience",
        "experience",
    }
)

NICE_HEADINGS: frozenset[str] = frozenset(
    {
        "preferred",
        "preferred qualifications",
        "preferred skills",
        "nice to have",
        "nice to haves",
        "nice-to-have",
        "nice-to-haves",
        "bonus",
        "bonus points",
        "bonus points for",
        "desired",
        "desired skills",
        "pluses",
        "extra credit",
        "good to have",
        "additional qualifications",
    }
)

# Headings whose content is prose about the company, not about requirements.
# Lines under these are ignored unless they carry an explicit cue, which is the
# single biggest source of false positives removed in this phase.
IGNORED_HEADINGS: frozenset[str] = frozenset(
    {
        "about",
        "about us",
        "about the role",
        "about the team",
        "about the company",
        "the team",
        "the role",
        "our mission",
        "overview",
        "company overview",
        "what we offer",
        "what we offer you",
        "benefits",
        "perks",
        "perks and benefits",
        "compensation",
        "salary",
        "interview process",
        "hiring process",
        "how to apply",
        "apply",
        "equal opportunity",
        "eeo",
        "diversity",
        "location",
    }
)

# Explicit cues. The two sets are disjoint, unlike the lists they replace.
NICE_CUES: Tuple[str, ...] = (
    r"nice\s+to\s+have",
    r"nice-to-have",
    r"preferred",
    r"preferably",
    r"bonus",
    r"a\s+plus",
    r"is\s+a\s+plus",
    r"plus\s+if",
    r"advantage",
    r"advantageous",
    r"desirable",
    r"desired",
    r"ideally",
    r"would\s+be\s+great",
    r"would\s+be\s+nice",
    r"helpful",
    r"beneficial",
    r"exposure\s+to",
    r"good\s+to\s+have",
    r"extra\s+credit",
    r"optional",
)

MUST_CUES: Tuple[str, ...] = (
    r"must\s+have",
    r"must\s+be",
    r"must\b",
    r"required",
    r"requires",
    r"requirement",
    r"minimum",
    r"at\s+least",
    r"essential",
    r"necessary",
    r"mandatory",
    r"prerequisite",
    r"you\s+will\s+need",
    r"we\s+require",
    r"strong\b",
    r"proven\b",
    r"demonstrated",
    r"expert\b",
    r"\d+\+?\s*years?",
)

NICE_CUE_RE = re.compile("|".join(NICE_CUES), re.IGNORECASE)
MUST_CUE_RE = re.compile("|".join(MUST_CUES), re.IGNORECASE)

BULLET_PREFIX_RE = re.compile(r"^[\s•‣◦⁃∙\-\*→▶o]+(?=\S)")
NUMBERED_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[a-z][.)])\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HEADING_TRAILING_RE = re.compile(r"[\s:;.\-–—|]+$")


class JDParser:
    def __init__(self, normalizer: Optional[SkillNormalizer] = None) -> None:
        """Initialize the JD parser with a skill normalizer."""
        self.normalizer = normalizer or SkillNormalizer()
        self._match_index = self._build_match_index()

    # -- public API -------------------------------------------------------

    def parse_jd(self, jd_text: str) -> ParsedJD:
        """Parse job description text into structured requirements."""
        cleaned = self._clean_text(jd_text)
        lines = cleaned.split("\n")

        parsed_jd = ParsedJD()
        parsed_jd.title = self._extract_job_title(lines)

        must, nice, spans = self._extract_requirements(lines)
        parsed_jd.must_have_skills = must
        parsed_jd.nice_to_have_skills = nice
        parsed_jd.skills_norm = self.normalizer.normalize_skills(must + nice)

        # Offsets are reported against exactly the text the API returns, so the
        # UI can highlight evidence without searching for it.
        parsed_jd.source_text = cleaned
        parsed_jd.spans = spans
        return parsed_jd

    # -- cleaning ---------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace within each line while keeping line breaks.

        Collapsing all whitespace with a single re.sub, as the previous version
        did, destroyed every newline. Everything downstream splits on '\\n', so
        bullet extraction and title detection only ever saw one enormous line.
        """
        if not text:
            return ""

        text = HTML_TAG_RE.sub(" ", text)
        cleaned_lines: List[str] = []
        for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.replace("\t", " ")
            line = NUMBERED_PREFIX_RE.sub("", line)
            # Normalize any bullet glyph to "- " rather than dropping it.
            # Whether a line is a bullet is a signal the classifier needs: a
            # bullet under a requirements heading is a skill-list context, so
            # an ambiguous token such as "Go" may be read as a skill there.
            was_bullet = bool(BULLET_PREFIX_RE.match(line))
            line = BULLET_PREFIX_RE.sub("", line)
            line = re.sub(r"[  ]+", " ", line).strip()
            if was_bullet and line:
                line = "- " + line
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    # -- title ------------------------------------------------------------

    def _extract_job_title(self, lines: Iterable[str]) -> Optional[str]:
        """Take the first substantive line as the title.

        Real job descriptions lead with the title. That beats the previous
        capitalisation regexes, which had to fire against a lowercased,
        newline-free blob and so almost never matched.
        """
        for line in lines:
            candidate = line.strip()
            if not candidate or candidate.startswith("- "):
                continue
            if self._heading_key(candidate) is not None:
                continue
            words = candidate.split()
            if not 1 <= len(words) <= 9:
                continue
            if NICE_CUE_RE.search(candidate) or MUST_CUE_RE.search(candidate):
                continue
            if candidate.endswith((".", "!", "?")):
                continue
            return candidate
        return None

    # -- requirements -----------------------------------------------------

    def _extract_requirements(
        self, lines: List[str]
    ) -> Tuple[List[str], List[str], List[SkillSpan]]:
        """Split requirements into must-have and nice-to-have canonical skills.

        Evidence spans are recorded here rather than by searching the finished
        document, because only this loop knows which line a skill was actually
        matched on. Searching afterwards pointed "rest" at the phrase "the rest
        of the team" in the company blurb instead of at the REST API requirement
        it was really extracted from.
        """
        must: List[str] = []
        nice: List[str] = []
        located: Dict[str, Tuple[int, int, str]] = {}
        section: Optional[str] = None
        offset = 0

        for line in lines:
            line_start = offset
            offset += len(line) + 1  # the newline that join() puts back

            if not line:
                continue

            is_bullet = line.startswith("- ")
            body_offset = 2 if is_bullet else 0
            body = line[body_offset:].strip() if is_bullet else line
            if not body:
                continue

            heading = self._heading_key(body)
            if heading is not None:
                section = self._section_kind(heading)
                # A heading such as "Nice to have:" carries no skills itself.
                continue

            priority = self._line_priority(body, section)
            if priority is None:
                continue

            in_requirements = section in ("must", "nice")
            for skill, surface in self._extract_skills_from_line(
                body, bullet_in_requirements=is_bullet and in_requirements
            ):
                target = must if priority == "must" else nice
                if skill not in target:
                    target.append(skill)
                if skill not in located:
                    found = locate_token(body, surface)
                    if found is not None:
                        start = line_start + body_offset + found[0]
                        end = line_start + body_offset + found[1]
                        located[skill] = (start, end, line[body_offset:].strip())

        # A skill named in both sections is a hard requirement. Counting it in
        # both would also double its weight in the score.
        nice = [skill for skill in nice if skill not in must]

        spans: List[SkillSpan] = []
        for skill in must + nice:
            priority = "must" if skill in must else "nice"
            if skill in located:
                start, end, evidence = located[skill]
                spans.append(
                    SkillSpan(
                        skill=skill,
                        priority=priority,
                        evidence=evidence,
                        start=start,
                        end=end,
                    )
                )
            else:
                spans.append(SkillSpan(skill=skill, priority=priority))
        return must, nice, spans

    def _line_priority(self, line: str, section: Optional[str]) -> Optional[str]:
        """Decide whether a line describes a must-have or a nice-to-have."""
        if NICE_CUE_RE.search(line):
            return "nice"
        if MUST_CUE_RE.search(line):
            return "must"
        if section == "nice":
            return "nice"
        if section == "must":
            return "must"
        if section == "ignored":
            return None
        # No heading seen yet and no cue: treat it as a requirement rather than
        # lose it, since plenty of job descriptions have no structure at all.
        return "must"

    @staticmethod
    def _section_kind(heading: str) -> str:
        if heading in NICE_HEADINGS:
            return "nice"
        if heading in MUST_HEADINGS:
            return "must"
        if heading in IGNORED_HEADINGS:
            return "ignored"
        return "ignored"

    @staticmethod
    def _heading_key(line: str) -> Optional[str]:
        """Return the normalized heading name if this line is a known heading."""
        stripped = HEADING_TRAILING_RE.sub("", line.strip()).strip()
        if not stripped or len(stripped.split()) > 5:
            return None
        key = re.sub(r"\s+", " ", stripped.lower())
        if key in MUST_HEADINGS or key in NICE_HEADINGS or key in IGNORED_HEADINGS:
            return key
        return None

    # -- skill matching ---------------------------------------------------

    def _build_match_index(self) -> Dict[str, str]:
        """Map every searchable surface form to its canonical skill."""
        index: Dict[str, str] = {}
        for canonical, variants in self.normalizer.ontology.items():
            index[canonical.lower()] = canonical
            for variant in variants:
                index[variant.lower()] = canonical
        return index

    def _extract_skills_from_line(
        self, line: str, bullet_in_requirements: bool = False
    ) -> List[Tuple[str, str]]:
        """Find canonical skills in a line, with the surface form that matched.

        The surface form is returned so the caller can record where on the line
        the evidence actually sits.
        """
        lowered = line.lower()
        word_set = set(re.findall(r"[a-z0-9+#.\-]+", lowered))
        list_context = bullet_in_requirements or self._is_list_context(line)

        best: Dict[str, str] = {}
        for surface, canonical in self._match_index.items():
            if not self._surface_present(surface, lowered):
                continue
            if self.normalizer.is_ambiguous(surface) and not self._ambiguous_token_confirmed(
                surface, word_set, list_context
            ):
                continue
            # Prefer the longest matching surface form: it is the more specific
            # evidence, so "react native" beats "react".
            if canonical not in best or len(surface) > len(best[canonical]):
                best[canonical] = surface
        return list(best.items())

    @staticmethod
    def _surface_present(surface: str, lowered_line: str) -> bool:
        """Word-boundary containment that tolerates '+', '#' and '.' in names."""
        pattern = r"(?<![a-z0-9])" + re.escape(surface) + r"(?![a-z0-9])"
        return re.search(pattern, lowered_line) is not None

    def _ambiguous_token_confirmed(
        self, token: str, words: Set[str], list_context: bool
    ) -> bool:
        """Whether an ambiguous token may be read as a skill on this line.

        "Go language" and "REST API" are skills. "go over the roadmap" and
        "the rest of the team" are not, and word boundaries alone cannot tell
        them apart.
        """
        if list_context:
            return True
        return bool(self.normalizer.disambiguators_for(token) & words)

    @staticmethod
    def _is_list_context(line: str) -> bool:
        """Whether a line reads as a delimited list of short skill names."""
        parts = [p.strip() for p in re.split(r"[,;/|]", line) if p.strip()]
        if len(parts) < 3:
            return False
        return all(len(p.split()) <= 4 for p in parts)
