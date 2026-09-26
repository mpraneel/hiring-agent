import logging
from typing import Optional

from .. import config
from ..schemas.models import ParsedResume, ParsedJD, MatchResult
from .baseline_scorer import BaselineScorer
from .llm_overlay import LLMOverlay

logger = logging.getLogger(__name__)


class MatchAggregator:
    def __init__(
        self,
        llm_provider: Optional[str] = None,
        enable_llm: bool = True,
        llm_model: Optional[str] = None,
    ):
        """Initialize the match aggregator."""
        self.baseline_scorer = BaselineScorer()
        self.llm_overlay: Optional[LLMOverlay] = None
        self.enable_llm = enable_llm and config.llm_enabled()

        if self.enable_llm:
            try:
                self.llm_overlay = LLMOverlay(provider=llm_provider, model=llm_model)
            except Exception as exc:
                logger.warning("LLM overlay disabled: %s", exc)
                self.enable_llm = False
    
    def match_resume_to_jd(self, resume: ParsedResume, jd: ParsedJD) -> MatchResult:
        """
        Match a resume to a job description and return comprehensive results.
        
        Args:
            resume: Parsed resume data
            jd: Parsed job description data
            
        Returns:
            MatchResult with score, matched/missing skills, and optional LLM insights
        """
        # Calculate baseline score
        baseline_score, matched_skills, missing_skills, nice_matches = self.baseline_scorer.calculate_score(resume, jd)
        
        # Create initial match result
        match_result = MatchResult(
            match_score=baseline_score,
            baseline_score=baseline_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            nice_matches=nice_matches,
            llm_rationale=None,
            suggestions=[]
        )
        
        # Add LLM overlay if enabled
        if self.enable_llm and self.llm_overlay:
            try:
                # Generate rationale
                rationale = self.llm_overlay.generate_rationale(resume, jd, match_result)
                match_result.llm_rationale = rationale
                
                # Generate suggestions
                suggestions = self.llm_overlay.generate_suggestions(resume, jd, match_result)
                match_result.suggestions = suggestions
                
            except Exception as e:
                # If LLM fails, continue without it
                logger.warning("LLM overlay failed: %s", e)
                match_result.llm_rationale = "LLM analysis unavailable"
                match_result.suggestions = []
        
        return match_result
    
    def get_detailed_analysis(self, resume: ParsedResume, jd: ParsedJD) -> dict:
        """
        Get detailed analysis including breakdowns and insights.
        
        Returns:
            Dictionary with comprehensive analysis data
        """
        # Get baseline breakdown
        baseline_breakdown = self.baseline_scorer.get_detailed_breakdown(resume, jd)
        
        # Get skill gaps analysis
        skill_gaps = self.baseline_scorer.get_skill_gaps(resume, jd)
        
        # Get skill overlap analysis
        skill_overlap = self.baseline_scorer.get_skill_overlap(resume, jd)
        
        # Create comprehensive analysis
        analysis = {
            'resume_info': {
                'name': resume.name,
                'email': resume.email,
                'phone': resume.phone,
                'total_skills': len(resume.skills_norm),
                'skills': resume.skills_norm,
                'education_count': len(resume.education),
                'experience_count': len(resume.experiences)
            },
            'jd_info': {
                'title': jd.title,
                'must_have_skills_count': len(jd.must_haves_raw),
                'nice_to_have_skills_count': len(jd.nice_to_haves_raw),
                'must_have_skills': jd.must_haves_raw,
                'nice_to_have_skills': jd.nice_to_haves_raw
            },
            'baseline_analysis': baseline_breakdown,
            'skill_gaps': skill_gaps,
            'skill_overlap': skill_overlap,
            'llm_enabled': self.enable_llm
        }
        
        # Add LLM insights if available
        if self.enable_llm and self.llm_overlay:
            try:
                # Create a temporary match result for LLM analysis
                temp_result = MatchResult(
                    match_score=baseline_breakdown['baseline_score'],
                    baseline_score=baseline_breakdown['baseline_score'],
                    matched_skills=baseline_breakdown['matched_must_have_skills'] + baseline_breakdown['matched_nice_to_have_skills'],
                    missing_skills=baseline_breakdown['missing_must_have_skills'] + baseline_breakdown['missing_nice_to_have_skills'],
                    nice_matches=baseline_breakdown['matched_nice_to_have_skills']
                )
                
                rationale = self.llm_overlay.generate_rationale(resume, jd, temp_result)
                suggestions = self.llm_overlay.generate_suggestions(resume, jd, temp_result)
                
                analysis['llm_insights'] = {
                    'rationale': rationale,
                    'suggestions': suggestions
                }
                
            except Exception as e:
                analysis['llm_insights'] = {
                    'rationale': f"LLM analysis failed: {str(e)}",
                    'suggestions': []
                }
        
        return analysis
    
    def get_match_summary(self, resume: ParsedResume, jd: ParsedJD) -> dict:
        """
        Get a concise summary of the match.
        
        Returns:
            Dictionary with key match metrics
        """
        match_result = self.match_resume_to_jd(resume, jd)
        
        return {
            'score': match_result.match_score,
            'score_percentage': match_result.match_score * 100,
            'matched_skills_count': len(match_result.matched_skills),
            'missing_skills_count': len(match_result.missing_skills),
            'nice_matches_count': len(match_result.nice_matches),
            'has_critical_gaps': len([s for s in match_result.missing_skills if s in jd.must_haves_raw]) > 0,
            'llm_available': match_result.llm_rationale is not None and match_result.llm_rationale != "LLM analysis unavailable"
        }
    
    def compare_multiple_resumes(self, resumes: list[ParsedResume], jd: ParsedJD) -> list[dict]:
        """
        Compare multiple resumes against a single job description.
        
        Args:
            resumes: List of parsed resumes
            jd: Parsed job description
            
        Returns:
            List of match results sorted by score (highest first)
        """
        results = []
        
        for i, resume in enumerate(resumes):
            match_result = self.match_resume_to_jd(resume, jd)
            
            results.append({
                'resume_index': i,
                'candidate_name': resume.name or f"Resume {i+1}",
                'match_result': match_result,
                'summary': self.get_match_summary(resume, jd)
            })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['match_result'].match_score, reverse=True)
        
        return results
