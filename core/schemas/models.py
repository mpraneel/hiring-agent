from pydantic import BaseModel
from typing import List, Optional


class CandidateExperience(BaseModel):
    title: str
    company: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    bullets: List[str] = []


class ParsedResume(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills_raw: List[str] = []
    skills_norm: List[str] = []
    education: List[str] = []
    experiences: List[CandidateExperience] = []


class ParsedJD(BaseModel):
    title: Optional[str] = None
    must_haves_raw: List[str] = []
    nice_to_haves_raw: List[str] = []
    skills_norm: List[str] = []


class MatchResult(BaseModel):
    match_score: float  # 0 to 1
    matched_skills: List[str]
    missing_skills: List[str]
    nice_matches: List[str]
    baseline_score: float
    llm_rationale: Optional[str] = None
    suggestions: List[str] = []  # optional bullet rewrites or learning resources
