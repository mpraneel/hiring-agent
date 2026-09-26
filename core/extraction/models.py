"""Schemas for LLM extraction output.

Every field an LLM fills is validated here before anything downstream sees it.
The evidence spans exist so a claim can be checked against the source document:
an extracted skill whose evidence does not actually appear in the text is a
hallucination, and is dropped rather than scored.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Priority = Literal["must", "nice"]
ExtractionPath = Literal["llm", "deterministic", "fallback", "hybrid"]


class ExtractedRequirement(BaseModel):
    """One requirement found in a job description."""

    skill: str = Field(description="The skill name, as short as possible, e.g. 'PostgreSQL'.")
    priority: Priority = Field(
        description="'must' for a hard requirement, 'nice' for a preferred or bonus one."
    )
    evidence: str = Field(
        description="A short span copied verbatim from the job description that states "
        "this requirement. It must appear in the source text exactly."
    )


class ExtractedSkill(BaseModel):
    """One skill found in a resume."""

    skill: str = Field(description="The skill name, as short as possible, e.g. 'FastAPI'.")
    evidence: str = Field(
        description="A short span copied verbatim from the resume that evidences this "
        "skill. It must appear in the source text exactly."
    )


class JDExtraction(BaseModel):
    """Everything the LLM extracted from one job description."""

    title: Optional[str] = None
    requirements: List[ExtractedRequirement] = Field(default_factory=list)


class ResumeExtraction(BaseModel):
    """Everything the LLM extracted from one resume."""

    skills: List[ExtractedSkill] = Field(default_factory=list)


class MatchExplanation(BaseModel):
    """The grounded explanation of an already computed score.

    The LLM never sees or produces the number. It is shown the computed
    breakdown and asked to describe it, so it cannot justify a score it did not
    see, and it cannot move one.
    """

    rationale: str = Field(description="Two or three sentences explaining the match.")
    suggestions: List[str] = Field(
        default_factory=list,
        max_length=3,
        description="At most three concrete suggestions, each naming a missing skill.",
    )
