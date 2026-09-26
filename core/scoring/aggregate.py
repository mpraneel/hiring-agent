"""Combines deterministic scoring with a grounded LLM explanation.

The score is computed here and is never adjusted afterwards. The explanation
layer is strictly additive: if it fails, the response carries the same number
with llm_status set to "unavailable" and no rationale.
"""

from __future__ import annotations

import logging
from typing import Optional

from .. import config
from ..schemas.models import ExtractionPath, MatchResult, ParsedJD, ParsedResume
from .baseline_scorer import BaselineScorer
from .llm_overlay import LLMOverlay, ScoreBreakdown

logger = logging.getLogger(__name__)


def combine_extraction_paths(jd_path: str, resume_path: str) -> ExtractionPath:
    """Summarize the two document paths into one value for the response.

    A fallback anywhere dominates, because a degraded run should not be
    advertised as a clean one.
    """
    if "fallback" in (jd_path, resume_path):
        return "fallback"
    if jd_path == resume_path:
        return jd_path  # type: ignore[return-value]
    return "hybrid"


class MatchAggregator:
    def __init__(
        self,
        llm_provider: Optional[str] = None,
        enable_llm: bool = True,
        llm_model: Optional[str] = None,
        llm_overlay: Optional[LLMOverlay] = None,
    ) -> None:
        """Initialize the match aggregator."""
        self.baseline_scorer = BaselineScorer()
        self.llm_overlay: Optional[LLMOverlay] = llm_overlay
        self.enable_llm = enable_llm and (
            llm_overlay is not None or config.llm_enabled()
        )

        if self.enable_llm and self.llm_overlay is None:
            try:
                self.llm_overlay = LLMOverlay(provider=llm_provider, model=llm_model)
            except Exception as exc:
                logger.warning("LLM overlay disabled: %s", exc)
                self.enable_llm = False

    # -- scoring ----------------------------------------------------------

    def build_breakdown(self, resume: ParsedResume, jd: ParsedJD) -> ScoreBreakdown:
        """Compute the deterministic breakdown that drives both score and prose."""
        detail = self.baseline_scorer.get_detailed_breakdown(resume, jd)
        return ScoreBreakdown(
            score=detail["baseline_score"],
            matched_must=sorted(detail["matched_must_have_skills"]),
            matched_nice=sorted(detail["matched_nice_to_have_skills"]),
            missing_must=sorted(detail["missing_must_have_skills"]),
            missing_nice=sorted(detail["missing_nice_to_have_skills"]),
            must_total=detail["must_have_total"],
            nice_total=detail["nice_to_have_total"],
            extraction_path=combine_extraction_paths(
                jd.extraction_path, resume.extraction_path
            ),
        )

    def match_resume_to_jd(self, resume: ParsedResume, jd: ParsedJD) -> MatchResult:
        """Score a resume against a job description and explain the result."""
        breakdown = self.build_breakdown(resume, jd)

        result = MatchResult(
            match_score=breakdown.score,
            baseline_score=breakdown.score,
            matched_skills=breakdown.matched_must + breakdown.matched_nice,
            missing_skills=breakdown.missing,
            nice_matches=breakdown.matched_nice,
            llm_rationale=None,
            suggestions=[],
            extraction_path=breakdown.extraction_path,
            llm_status="disabled",
        )

        if not (self.enable_llm and self.llm_overlay):
            return result

        try:
            explanation = self.llm_overlay.explain(breakdown)
        except Exception as exc:
            # Log the detail, tell the client only that it is unavailable. Raw
            # provider exception text has no business in an API response.
            logger.warning("LLM explanation unavailable: %s", exc)
            result.llm_status = "unavailable"
            return result

        result.llm_rationale = explanation.rationale
        result.suggestions = list(explanation.suggestions)
        result.llm_status = "ok"
        return result

    # -- analysis ---------------------------------------------------------

    def get_detailed_analysis(self, resume: ParsedResume, jd: ParsedJD) -> dict:
        """Full breakdown plus, when available, a grounded explanation."""
        baseline_breakdown = self.baseline_scorer.get_detailed_breakdown(resume, jd)
        breakdown = self.build_breakdown(resume, jd)

        analysis = {
            "resume_info": {
                "name": resume.name,
                "email": resume.email,
                "phone": resume.phone,
                "total_skills": len(resume.skills_norm),
                "skills": resume.skills_norm,
                "education_count": len(resume.education),
                "experience_count": len(resume.experiences),
                "extraction_path": resume.extraction_path,
            },
            "jd_info": {
                "title": jd.title,
                "must_have_skills_count": len(jd.must_have_skills),
                "nice_to_have_skills_count": len(jd.nice_to_have_skills),
                "must_have_skills": jd.must_have_skills,
                "nice_to_have_skills": jd.nice_to_have_skills,
                "extraction_path": jd.extraction_path,
            },
            "baseline_analysis": baseline_breakdown,
            "skill_gaps": self.baseline_scorer.get_skill_gaps(resume, jd),
            "skill_overlap": self.baseline_scorer.get_skill_overlap(resume, jd),
            "extraction_path": breakdown.extraction_path,
            "llm_enabled": self.enable_llm,
            "llm_status": "disabled",
            "llm_insights": None,
        }

        if not (self.enable_llm and self.llm_overlay):
            return analysis

        try:
            explanation = self.llm_overlay.explain(breakdown)
        except Exception as exc:
            logger.warning("LLM explanation unavailable: %s", exc)
            analysis["llm_status"] = "unavailable"
            return analysis

        analysis["llm_status"] = "ok"
        analysis["llm_insights"] = {
            "rationale": explanation.rationale,
            "suggestions": list(explanation.suggestions),
        }
        return analysis

    def get_match_summary(self, resume: ParsedResume, jd: ParsedJD) -> dict:
        """A compact summary of one match."""
        result = self.match_resume_to_jd(resume, jd)
        return {
            "score": result.match_score,
            "score_percentage": result.match_score * 100,
            "matched_skills_count": len(result.matched_skills),
            "missing_skills_count": len(result.missing_skills),
            "nice_matches_count": len(result.nice_matches),
            "has_critical_gaps": any(
                skill in jd.must_have_skills for skill in result.missing_skills
            ),
            "extraction_path": result.extraction_path,
            "llm_status": result.llm_status,
        }

    def compare_multiple_resumes(
        self, resumes: list[ParsedResume], jd: ParsedJD
    ) -> list[dict]:
        """Rank several resumes against one job description, best first."""
        results = []
        for index, resume in enumerate(resumes):
            match_result = self.match_resume_to_jd(resume, jd)
            results.append(
                {
                    "resume_index": index,
                    "candidate_name": resume.name or f"Resume {index + 1}",
                    "match_result": match_result,
                }
            )
        results.sort(key=lambda item: item["match_result"].match_score, reverse=True)
        return results
