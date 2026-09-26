"""Pydantic models for parsed documents and match results."""

from typing import List, Optional

from pydantic import BaseModel, Field


class CandidateExperience(BaseModel):
    """One role from a resume's experience section."""

    title: str
    company: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    """A resume after extraction.

    ``skills_raw`` holds the skill strings exactly as the document spelled them.
    ``skills_norm`` holds canonical ontology names, and is the only field the
    scorer reads. Skills the ontology does not recognize are carried through to
    ``skills_norm`` unchanged rather than dropped.
    """

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills_raw: List[str] = Field(default_factory=list)
    skills_norm: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    experiences: List[CandidateExperience] = Field(default_factory=list)


class ParsedJD(BaseModel):
    """A job description after extraction.

    ``must_have_skills`` and ``nice_to_have_skills`` hold **canonical** skill
    names, not raw document text, because the scorer intersects them directly
    with ``ParsedResume.skills_norm``. They were previously named with a "_raw"
    suffix, which invited exactly the wrong thing to be written into them.

    The two lists are disjoint. A skill named in both sections of a JD is kept
    as must-have only, otherwise it would also be counted twice by the weighting.
    """

    title: Optional[str] = None
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    skills_norm: List[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    """The outcome of scoring one resume against one job description.

    ``match_score`` is computed deterministically from the extracted skills and
    is never adjusted by an LLM, so the same inputs always produce the same
    number. ``llm_rationale`` and ``suggestions`` explain that number; they
    cannot change it.
    """

    match_score: float = Field(ge=0.0, le=1.0)
    matched_skills: List[str]
    missing_skills: List[str]
    nice_matches: List[str]
    baseline_score: float = Field(ge=0.0, le=1.0)
    llm_rationale: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
