"""LLM extraction with validation, one retry, and a deterministic fallback.

Control flow per document:

1. ask the provider for structured output,
2. validate it with Pydantic,
3. on failure, retry once with the validation error appended to the prompt,
4. on a second failure, fall back to the deterministic parser.

Nothing here produces or influences a score. The LLM only improves the inputs to
scoring, so the score stays reproducible.

After validation, two guardrails apply:

* every item's evidence span must appear in the source text after whitespace
  normalization, otherwise the item is dropped as a hallucination,
* every skill is normalized through the ontology, with unknown skills kept as-is.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ..normalize.normalizer import SkillNormalizer
from ..parsers.jd_parser import JDParser
from ..parsers.resume_parser import ResumeParser
from ..schemas.models import ParsedJD, ParsedResume
from .models import (
    ExtractedRequirement,
    ExtractedSkill,
    JDExtraction,
    ResumeExtraction,
)
from .provider import StructuredProvider, StructuredProviderError, build_provider
from .spans import normalize_source, spans_for_skills, spans_from_evidence

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

MAX_ATTEMPTS = 2

JD_SYSTEM_PROMPT = """You extract hiring requirements from job descriptions.

Rules:
- Only list skills, tools, languages and technologies. Do not list years of
  experience, degrees, soft requirements or job duties.
- priority is "must" for a hard requirement and "nice" for anything preferred,
  desired, a bonus or a plus.
- evidence must be copied verbatim from the job description, word for word. Do
  not paraphrase, do not fix typos, do not add quotation marks. A short phrase
  is better than a whole sentence.
- If a skill is mentioned as both required and preferred, use "must".
- If the document does not name a skill, do not invent one. An empty list is a
  correct answer for a document with no requirements."""

RESUME_SYSTEM_PROMPT = """You extract skills from resumes.

Rules:
- Only list skills, tools, languages and technologies the resume actually
  evidences, whether in a skills section or in experience bullets.
- evidence must be copied verbatim from the resume, word for word. Do not
  paraphrase, do not fix typos, do not add quotation marks. A short phrase is
  better than a whole sentence.
- Do not infer skills that are merely adjacent to what is written. If the resume
  says Docker, that is not evidence of Kubernetes.
- If the document names no skills, an empty list is the correct answer."""


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and lowercase.

    Evidence is compared in this form because providers reliably alter
    whitespace when copying a span, wrapping lines differently or turning a
    newline into a space, while still quoting the words faithfully.
    """
    return re.sub(r"\s+", " ", text or "").strip().lower()


class LLMExtractor:
    """Extracts structured skills and requirements, with a deterministic fallback."""

    def __init__(
        self,
        provider: Optional[StructuredProvider] = None,
        normalizer: Optional[SkillNormalizer] = None,
        jd_parser: Optional[JDParser] = None,
        resume_parser: Optional[ResumeParser] = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.normalizer = normalizer or SkillNormalizer()
        self.jd_parser = jd_parser or JDParser(normalizer=self.normalizer)
        self.resume_parser = resume_parser or ResumeParser(normalizer=self.normalizer)
        self.max_attempts = max_attempts
        self._provider = provider
        self._provider_error: Optional[str] = None
        if self._provider is None:
            try:
                self._provider = build_provider()
            except StructuredProviderError as exc:
                # A missing key is a normal deployment state, not a crash. Every
                # extraction then takes the deterministic path.
                logger.info("LLM extraction unavailable, using deterministic path: %s", exc)
                self._provider_error = str(exc)

    @property
    def available(self) -> bool:
        """Whether a provider is configured at all."""
        return self._provider is not None

    # -- public API -------------------------------------------------------

    def extract_jd(self, jd_text: str) -> JDExtraction:
        """Extract requirements from a job description, LLM path only.

        Raises on exhaustion rather than falling back, so callers that want the
        raw LLM result (the eval harness in llm mode) can see a real failure.
        """
        extraction = self._attempt(JDExtraction, JD_SYSTEM_PROMPT, jd_text)
        extraction.requirements = self._clean_requirements(extraction.requirements, jd_text)
        return extraction

    def extract_resume(self, resume_text: str) -> ResumeExtraction:
        """Extract skills from a resume, LLM path only."""
        extraction = self._attempt(ResumeExtraction, RESUME_SYSTEM_PROMPT, resume_text)
        extraction.skills = self._clean_skills(extraction.skills, resume_text)
        return extraction

    def parse_jd(self, jd_text: str, mode: str = "hybrid") -> ParsedJD:
        """Produce a ParsedJD in the requested mode, recording the path taken."""
        if mode == "deterministic" or not self.available:
            parsed = self.jd_parser.parse_jd(jd_text)
            parsed.extraction_path = "deterministic"
            return parsed

        try:
            extraction = self.extract_jd(jd_text)
        except (StructuredProviderError, ValidationError) as exc:
            logger.warning("LLM JD extraction failed, falling back: %s", exc)
            parsed = self.jd_parser.parse_jd(jd_text)
            parsed.extraction_path = "fallback"
            return parsed

        llm_parsed = self._jd_from_extraction(extraction, jd_text)
        if mode == "llm":
            llm_parsed.extraction_path = "llm"
            return llm_parsed

        deterministic = self.jd_parser.parse_jd(jd_text)
        merged = self._merge_jd(llm_parsed, deterministic)
        merged.extraction_path = "hybrid"
        return merged

    def parse_resume(self, resume_text: str, mode: str = "hybrid") -> ParsedResume:
        """Produce a ParsedResume in the requested mode, recording the path taken."""
        if mode == "deterministic" or not self.available:
            parsed = self.resume_parser.parse_text(resume_text)
            parsed.extraction_path = "deterministic"
            return parsed

        try:
            extraction = self.extract_resume(resume_text)
        except (StructuredProviderError, ValidationError) as exc:
            logger.warning("LLM resume extraction failed, falling back: %s", exc)
            parsed = self.resume_parser.parse_text(resume_text)
            parsed.extraction_path = "fallback"
            return parsed

        deterministic = self.resume_parser.parse_text(resume_text)
        llm_skills = [item.skill for item in extraction.skills]

        if mode == "llm":
            parsed = ParsedResume(
                name=deterministic.name,
                email=deterministic.email,
                phone=deterministic.phone,
                skills_raw=[item.evidence for item in extraction.skills],
                skills_norm=self.normalizer.normalize_skills(llm_skills),
                education=deterministic.education,
                experiences=deterministic.experiences,
                source_text=deterministic.source_text,
                spans=spans_from_evidence(
                    deterministic.source_text,
                    [(item.skill, None, item.evidence) for item in extraction.skills],
                ),
            )
            parsed.extraction_path = "llm"
            return parsed

        merged_skills = self.normalizer.normalize_skills(
            llm_skills + deterministic.skills_norm
        )
        # Prefer the LLM's own evidence where it has some, and fall back to
        # locating the surface form for skills only the regex path found.
        llm_spans = spans_from_evidence(
            deterministic.source_text,
            [(item.skill, None, item.evidence) for item in extraction.skills],
        )
        covered = {span.skill for span in llm_spans if span.start is not None}
        remaining = [s for s in merged_skills if s not in covered]
        merged = ParsedResume(
            name=deterministic.name,
            email=deterministic.email,
            phone=deterministic.phone,
            skills_raw=deterministic.skills_raw,
            skills_norm=merged_skills,
            education=deterministic.education,
            experiences=deterministic.experiences,
            source_text=deterministic.source_text,
            spans=[s for s in llm_spans if s.start is not None]
            + spans_for_skills(
                deterministic.source_text,
                [(skill, None) for skill in remaining],
                self.normalizer,
            ),
        )
        merged.extraction_path = "hybrid"
        return merged

    # -- attempt loop -----------------------------------------------------

    def _attempt(self, schema: Type[ModelT], system: str, document: str) -> ModelT:
        """Call the provider, validating and retrying once on a schema error."""
        if self._provider is None:
            raise StructuredProviderError(
                self._provider_error or "no LLM provider configured"
            )

        user = f"Document:\n\n{document}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            prompt = user
            if last_error is not None:
                prompt = (
                    f"{user}\n\n"
                    f"Your previous response was rejected by schema validation:\n"
                    f"{last_error}\n\n"
                    f"Return JSON that satisfies the schema exactly."
                )
            raw = self._provider.generate_json(system, prompt, schema)
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "LLM output failed validation on attempt %d of %d: %s",
                    attempt,
                    self.max_attempts,
                    exc,
                )

        assert last_error is not None
        raise last_error

    # -- guardrails -------------------------------------------------------

    def _clean_requirements(
        self, requirements: List[ExtractedRequirement], source: str
    ) -> List[ExtractedRequirement]:
        """Drop hallucinated items, normalize skills, de-duplicate, prefer must."""
        haystack = normalize_whitespace(source)
        by_skill: dict[str, ExtractedRequirement] = {}

        for item in requirements:
            if not self._evidence_supported(item.evidence, haystack):
                logger.info(
                    "dropping requirement %r: evidence not found in source", item.skill
                )
                continue
            canonical = self.normalizer.normalize_skill(item.skill)
            if not canonical:
                continue
            existing = by_skill.get(canonical)
            if existing is None:
                by_skill[canonical] = item.model_copy(update={"skill": canonical})
            elif existing.priority == "nice" and item.priority == "must":
                # A skill named as both required and preferred is required.
                by_skill[canonical] = item.model_copy(update={"skill": canonical})

        return list(by_skill.values())

    def _clean_skills(
        self, skills: List[ExtractedSkill], source: str
    ) -> List[ExtractedSkill]:
        """Drop hallucinated items, normalize skills, de-duplicate."""
        haystack = normalize_whitespace(source)
        by_skill: dict[str, ExtractedSkill] = {}

        for item in skills:
            if not self._evidence_supported(item.evidence, haystack):
                logger.info("dropping skill %r: evidence not found in source", item.skill)
                continue
            canonical = self.normalizer.normalize_skill(item.skill)
            if not canonical or canonical in by_skill:
                continue
            by_skill[canonical] = item.model_copy(update={"skill": canonical})

        return list(by_skill.values())

    @staticmethod
    def _evidence_supported(evidence: str, normalized_source: str) -> bool:
        """Whether an evidence span really occurs in the source document."""
        needle = normalize_whitespace(evidence)
        if not needle:
            return False
        return needle in normalized_source

    # -- assembly ---------------------------------------------------------

    def _jd_from_extraction(self, extraction: JDExtraction, source: str) -> ParsedJD:
        must = [r.skill for r in extraction.requirements if r.priority == "must"]
        nice = [r.skill for r in extraction.requirements if r.priority == "nice"]
        nice = [s for s in nice if s not in must]
        normalized = normalize_source(source)
        return ParsedJD(
            title=extraction.title,
            must_have_skills=must,
            nice_to_have_skills=nice,
            skills_norm=self.normalizer.normalize_skills(must + nice),
            source_text=normalized,
            spans=spans_from_evidence(
                normalized,
                [
                    (r.skill, r.priority, r.evidence)
                    for r in extraction.requirements
                    if r.skill in must or r.skill in nice
                ],
            ),
        )

    def _merge_jd(self, primary: ParsedJD, secondary: ParsedJD) -> ParsedJD:
        """Union both extractions, with the primary winning priority conflicts."""
        must = list(primary.must_have_skills)
        for skill in secondary.must_have_skills:
            # The primary calling something nice-to-have beats the secondary
            # calling it a hard requirement.
            if skill not in must and skill not in primary.nice_to_have_skills:
                must.append(skill)

        nice = list(primary.nice_to_have_skills)
        for skill in secondary.nice_to_have_skills:
            if skill not in nice and skill not in must:
                nice.append(skill)
        nice = [s for s in nice if s not in must]

        priorities = {skill: "must" for skill in must}
        priorities.update({skill: "nice" for skill in nice})

        # Keep the primary's located spans, then locate anything only the
        # secondary found, so every chip can be traced back to the document.
        spans = [
            span.model_copy(update={"priority": priorities.get(span.skill, span.priority)})
            for span in primary.spans
            if span.start is not None and span.skill in priorities
        ]
        covered = {span.skill for span in spans}
        source = primary.source_text or secondary.source_text
        spans += spans_for_skills(
            source,
            [(skill, priorities[skill]) for skill in priorities if skill not in covered],
            self.normalizer,
        )

        return ParsedJD(
            title=primary.title or secondary.title,
            must_have_skills=must,
            nice_to_have_skills=nice,
            skills_norm=self.normalizer.normalize_skills(must + nice),
            source_text=source,
            spans=spans,
        )
