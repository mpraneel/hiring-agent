"""Deterministic resume parsing.

Section detection is the load bearing part. The previous implementation asked
whether a line *contained* a header word using re.search, so any sentence
mentioning "experience", "skills" or "projects" opened or closed a section. A
one line summary such as "Backend engineer who gained experience in distributed
systems" was enough to swallow the entire skills section, which is why resume
skill extraction returned nothing at all on realistic input.

A header is now a short line (five words or fewer) that matches a known header
name exactly, after stripping surrounding punctuation and case.
"""

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pypdf

from ..normalize.normalizer import SkillNormalizer
from ..schemas.models import CandidateExperience, ParsedResume

# Canonical header names, mapped to the section they open.
SECTION_ALIASES: Dict[str, str] = {
    "skills": "skills",
    "skill": "skills",
    "technical skills": "skills",
    "technical expertise": "skills",
    "core skills": "skills",
    "core competencies": "skills",
    "technologies": "skills",
    "tech stack": "skills",
    "tools": "skills",
    "tools and technologies": "skills",
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "employment": "experience",
    "employment history": "experience",
    "work history": "experience",
    "career history": "experience",
    "education": "education",
    "academic": "education",
    "academic background": "education",
    "academics": "education",
    "qualifications": "education",
    "projects": "projects",
    "project": "projects",
    "personal projects": "projects",
    "selected projects": "projects",
    "achievements": "achievements",
    "awards": "achievements",
    "certifications": "certifications",
    "certificates": "certifications",
    "languages": "languages",
    "interests": "interests",
    "hobbies": "interests",
    "contact": "contact",
    "personal": "contact",
    "personal details": "contact",
    "profile": "summary",
    "summary": "summary",
    "professional summary": "summary",
    "objective": "summary",
    "about": "summary",
    "about me": "summary",
    "publications": "publications",
    "references": "references",
    "volunteering": "volunteering",
}

MAX_HEADER_WORDS = 5
HEADER_TRAILING_RE = re.compile(r"^[\s\-–—•*|:]+|[\s:;.\-–—|]+$")

# Delimiters that separate skills in a list. A hyphen is deliberately absent:
# splitting on it broke scikit-learn, ci-cd and c-sharp.
SKILL_DELIMITER_RE = re.compile(r"[,;•‣◦|/\n]|\s{2,}|\s+\|\s+")

SKILL_PREFIX_RE = re.compile(
    r"^(?:proficient\s+in|proficient\s+with|experience\s+with|experience\s+in|"
    r"knowledge\s+of|expertise\s+in|familiar\s+with|skilled\s+in|working\s+with)\s*",
    re.IGNORECASE,
)
SKILL_SUFFIX_RE = re.compile(
    r"\s*\(?\s*(?:\d+\+?\s*(?:years?|yrs?)|years?|yrs?|experience|proficient|expert|"
    r"advanced|intermediate|beginner|basic|fluent)\s*\)?$",
    re.IGNORECASE,
)

BULLET_MARKER_RE = re.compile(r"^[\s•‣◦⁃∙\-\*→▶]+(?=\S)")

MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_DATE = rf"(?:(?:{MONTHS})\.?\s*)?(?:\d{{4}}|\d{{1,2}}[/-]\d{{2,4}})"
_NOW = r"present|current|now|to\s+date|ongoing"

DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE})\s*(?:[-–—]|\bto\b|\buntil\b)\s*(?P<end>{_DATE}|{_NOW})",
    re.IGNORECASE,
)

# "Title at Company", "Title - Company", "Title | Company", "Title, Company"
ROLE_SEPARATOR_RE = re.compile(r"\s+(?:at|@|for|with)\s+|\s+[-–—|]\s+|,\s+", re.IGNORECASE)

DEGREE_RE = re.compile(
    r"\b(?:bachelor|master|ph\.?d|doctorate|associate|diploma|certificate|"
    r"b\.?s\.?c?|m\.?s\.?c?|b\.?a\.?|m\.?a\.?|b\.?eng|m\.?eng|mba)\b",
    re.IGNORECASE,
)


class ResumeParser:
    def __init__(self, normalizer: Optional[SkillNormalizer] = None) -> None:
        """Initialize the resume parser with a skill normalizer."""
        self.normalizer = normalizer or SkillNormalizer()
        self.email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        self.phone_pattern = (
            r"(\+?\d{1,2}[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b"
        )
        self.name_patterns = [
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+)+)\s*$",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+)+)\s*[|,\-–—]",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+)+)\s*(?:resume|cv)\b",
        ]

    # -- public API -------------------------------------------------------

    def parse_pdf(self, pdf_path: str) -> ParsedResume:
        """Parse a PDF resume into structured data."""
        return self._parse_text(self.extract_pdf_text(pdf_path))

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract the plain text of a PDF, for callers that parse it separately."""
        return self._extract_text_from_pdf(pdf_path)

    def parse_text(self, text: str) -> ParsedResume:
        """Parse resume text into structured data."""
        return self._parse_text(text)

    # -- extraction -------------------------------------------------------

    @staticmethod
    def _extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from a PDF, preserving page and line breaks."""
        chunks: List[str] = []
        try:
            with open(pdf_path, "rb") as handle:
                reader = pypdf.PdfReader(handle)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        chunks.append(page_text)
        except Exception as exc:
            raise ValueError(f"Failed to read PDF file: {exc}") from exc
        return "\n".join(chunks)

    def _parse_text(self, text: str) -> ParsedResume:
        lines = [line.strip() for line in (text or "").replace("\r", "\n").split("\n")]
        non_empty = [line for line in lines if line]

        parsed = ParsedResume()
        parsed.name = self._extract_name(non_empty)
        parsed.email = self._extract_email(text or "")
        parsed.phone = self._extract_phone(text or "")

        sections = self._split_sections(non_empty)

        skills_lines = sections.get("skills", [])
        if skills_lines:
            parsed.skills_raw = self._extract_skills(skills_lines)
            parsed.skills_norm = self.normalizer.normalize_skills(parsed.skills_raw)

        education_lines = sections.get("education", [])
        if education_lines:
            parsed.education = self._extract_education(education_lines)

        experience_lines = sections.get("experience", [])
        if experience_lines:
            parsed.experiences = self._extract_experiences(experience_lines)

        return parsed

    # -- sections ---------------------------------------------------------

    def _split_sections(self, lines: Sequence[str]) -> Dict[str, List[str]]:
        """Group lines by the section heading that precedes them."""
        sections: Dict[str, List[str]] = {}
        current: Optional[str] = None
        for line in lines:
            section = self._section_for_header(line)
            if section is not None:
                current = section
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(line)
        return sections

    @staticmethod
    def _normalize_header(line: str) -> Optional[str]:
        """Return a comparable header key, or None if the line cannot be a header."""
        stripped = HEADER_TRAILING_RE.sub("", line.strip())
        stripped = HEADER_TRAILING_RE.sub("", stripped).strip()
        if not stripped:
            return None
        if len(stripped.split()) > MAX_HEADER_WORDS:
            return None
        return re.sub(r"\s+", " ", stripped.lower())

    @classmethod
    def _section_for_header(cls, line: str) -> Optional[str]:
        key = cls._normalize_header(line)
        if key is None:
            return None
        return SECTION_ALIASES.get(key)

    def _is_section_header(self, line: str, target_sections: Sequence[str]) -> bool:
        """Whether this line is a heading for one of the target sections.

        Matching is exact after normalization, so prose that merely mentions a
        header word is not a header. Both the literal target names and the
        canonical section they map to are accepted, so callers can pass either
        ["skills"] or ["technical skills"].
        """
        key = self._normalize_header(line)
        if key is None:
            return False

        wanted: Set[str] = set()
        for target in target_sections:
            normalized = re.sub(r"\s+", " ", target.strip().lower())
            wanted.add(normalized)
            mapped = SECTION_ALIASES.get(normalized)
            if mapped:
                wanted.add(mapped)

        if key in wanted:
            return True
        mapped_key = SECTION_ALIASES.get(key)
        return mapped_key is not None and mapped_key in wanted

    # -- contact details --------------------------------------------------

    def _extract_name(self, lines: Sequence[str]) -> Optional[str]:
        for line in lines[:10]:
            if self._section_for_header(line) is not None:
                continue
            for pattern in self.name_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1).strip()
                    if 2 <= len(name.split()) <= 4:
                        return name
        return None

    def _extract_email(self, text: str) -> Optional[str]:
        match = re.search(self.email_pattern, text)
        return match.group(0) if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        match = re.search(self.phone_pattern, text)
        if not match:
            return None
        return "".join(part for part in match.groups() if part).strip()

    # -- skills -----------------------------------------------------------

    def _extract_skills(self, skills_lines: Sequence[str]) -> List[str]:
        """Split a skills section into individual skill strings.

        Order is preserved and duplicates removed, so downstream output and the
        eval harness stay stable between runs.
        """
        skills: List[str] = []
        seen: Set[str] = set()

        for line in skills_lines:
            body = BULLET_MARKER_RE.sub("", line)
            # A "Languages: Python, Go" style label is not itself a skill.
            if ":" in body:
                head, _, tail = body.partition(":")
                if len(head.split()) <= 4 and tail.strip():
                    body = tail

            for candidate in SKILL_DELIMITER_RE.split(body):
                skill = candidate.strip().strip(".")
                if not skill:
                    continue
                skill = SKILL_PREFIX_RE.sub("", skill)
                skill = SKILL_SUFFIX_RE.sub("", skill)
                skill = skill.strip().strip(".,;")
                if len(skill) <= 1 or len(skill.split()) > 5:
                    continue
                key = skill.lower()
                if key not in seen:
                    seen.add(key)
                    skills.append(skill)

        return skills

    # -- education --------------------------------------------------------

    @staticmethod
    def _extract_education(education_lines: Sequence[str]) -> List[str]:
        """Group education lines into one entry per degree."""
        entries: List[str] = []
        current: List[str] = []

        for line in education_lines:
            if DEGREE_RE.search(line) and current:
                entries.append(" ".join(current).strip())
                current = [line]
            else:
                current.append(line)

        if current:
            entries.append(" ".join(current).strip())
        return [entry for entry in entries if entry]

    # -- experience -------------------------------------------------------

    def _extract_experiences(self, experience_lines: Sequence[str]) -> List[CandidateExperience]:
        """Parse an experience section into roles with bullets."""
        experiences: List[CandidateExperience] = []
        current: Optional[CandidateExperience] = None
        bullets: List[str] = []

        for line in experience_lines:
            is_bullet = bool(BULLET_MARKER_RE.match(line))
            parsed_role = None if is_bullet else self._parse_role_line(line)

            if parsed_role is not None:
                if current is not None:
                    current.bullets = bullets
                    experiences.append(current)
                current = parsed_role
                bullets = []
                continue

            text = BULLET_MARKER_RE.sub("", line).strip()
            if text and current is not None:
                bullets.append(text)

        if current is not None:
            current.bullets = bullets
            experiences.append(current)

        return experiences

    @staticmethod
    def _parse_role_line(line: str) -> Optional[CandidateExperience]:
        """Parse "Senior Engineer at Acme Corp, 2019 - 2023" into its parts.

        Returns None when the line does not look like a role header, so the
        caller can treat it as continuation text instead.
        """
        remainder = line.strip()
        if not remainder or remainder.endswith((".", "!", "?")):
            return None

        start: Optional[str] = None
        end: Optional[str] = None
        date_match = DATE_RANGE_RE.search(remainder)
        if date_match:
            start = date_match.group("start").strip()
            end = date_match.group("end").strip()
            remainder = (
                remainder[: date_match.start()] + remainder[date_match.end() :]
            ).strip()
        remainder = remainder.strip(" ,;|-–—")

        if not remainder:
            return None

        title = remainder
        company: Optional[str] = None
        separator = ROLE_SEPARATOR_RE.search(remainder)
        if separator:
            title = remainder[: separator.start()].strip()
            company = remainder[separator.end() :].strip(" ,;|-–—") or None

        if not title or len(title.split()) > 6:
            return None
        # A role header needs either a company or a date range to be credible.
        if company is None and start is None:
            return None

        return CandidateExperience(title=title, company=company, start=start, end=end)
