from typing import List, Tuple
from ..schemas.models import ParsedResume, ParsedJD


class BaselineScorer:
    def __init__(self, must_have_weight: float = 2.0, nice_to_have_weight: float = 1.0):
        """Initialize the baseline scorer with configurable weights."""
        self.must_have_weight = must_have_weight
        self.nice_to_have_weight = nice_to_have_weight
    
    def calculate_score(self, resume: ParsedResume, jd: ParsedJD) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Calculate baseline score and return matched/missing skills.
        
        Returns:
            Tuple of (score, matched_skills, missing_skills, nice_matches)
        """
        # Get normalized skills from both resume and JD
        resume_skills = set(resume.skills_norm)
        jd_must_have_skills = set(jd.must_haves_raw)
        jd_nice_to_have_skills = set(jd.nice_to_haves_raw)
        
        # Calculate matches
        matched_must_have = resume_skills.intersection(jd_must_have_skills)
        matched_nice_to_have = resume_skills.intersection(jd_nice_to_have_skills)
        
        # Calculate missing skills
        missing_must_have = jd_must_have_skills - resume_skills
        missing_nice_to_have = jd_nice_to_have_skills - resume_skills
        
        # Calculate total weighted scores
        total_must_have_weight = len(jd_must_have_skills) * self.must_have_weight
        total_nice_to_have_weight = len(jd_nice_to_have_skills) * self.nice_to_have_weight
        total_weight = total_must_have_weight + total_nice_to_have_weight
        
        if total_weight == 0:
            return 0.0, [], [], []
        
        # Calculate weighted score
        must_have_score = len(matched_must_have) * self.must_have_weight
        nice_to_have_score = len(matched_nice_to_have) * self.nice_to_have_weight
        total_score = must_have_score + nice_to_have_score
        
        # Normalize score to 0-1 range
        baseline_score = total_score / total_weight
        
        # Prepare return values
        matched_skills = list(matched_must_have) + list(matched_nice_to_have)
        missing_skills = list(missing_must_have) + list(missing_nice_to_have)
        nice_matches = list(matched_nice_to_have)
        
        return baseline_score, matched_skills, missing_skills, nice_matches
    
    def get_detailed_breakdown(self, resume: ParsedResume, jd: ParsedJD) -> dict:
        """Get detailed breakdown of the scoring."""
        resume_skills = set(resume.skills_norm)
        jd_must_have_skills = set(jd.must_haves_raw)
        jd_nice_to_have_skills = set(jd.nice_to_haves_raw)
        
        # Calculate matches
        matched_must_have = resume_skills.intersection(jd_must_have_skills)
        matched_nice_to_have = resume_skills.intersection(jd_nice_to_have_skills)
        
        # Calculate missing skills
        missing_must_have = jd_must_have_skills - resume_skills
        missing_nice_to_have = jd_nice_to_have_skills - resume_skills
        
        # Calculate scores
        must_have_score = len(matched_must_have) * self.must_have_weight
        nice_to_have_score = len(matched_nice_to_have) * self.nice_to_have_weight
        total_score = must_have_score + nice_to_have_score
        
        total_must_have_weight = len(jd_must_have_skills) * self.must_have_weight
        total_nice_to_have_weight = len(jd_nice_to_have_skills) * self.nice_to_have_weight
        total_weight = total_must_have_weight + total_nice_to_have_weight
        
        baseline_score = total_score / total_weight if total_weight > 0 else 0.0
        
        return {
            'baseline_score': baseline_score,
            'must_have_matches': len(matched_must_have),
            'must_have_total': len(jd_must_have_skills),
            'nice_to_have_matches': len(matched_nice_to_have),
            'nice_to_have_total': len(jd_nice_to_have_skills),
            'must_have_score': must_have_score,
            'nice_to_have_score': nice_to_have_score,
            'total_score': total_score,
            'total_weight': total_weight,
            'matched_must_have_skills': list(matched_must_have),
            'matched_nice_to_have_skills': list(matched_nice_to_have),
            'missing_must_have_skills': list(missing_must_have),
            'missing_nice_to_have_skills': list(missing_nice_to_have),
            'resume_skills': list(resume_skills),
            'jd_must_have_skills': list(jd_must_have_skills),
            'jd_nice_to_have_skills': list(jd_nice_to_have_skills)
        }
    
    def get_skill_gaps(self, resume: ParsedResume, jd: ParsedJD) -> dict:
        """Analyze skill gaps between resume and job description."""
        resume_skills = set(resume.skills_norm)
        jd_must_have_skills = set(jd.must_haves_raw)
        jd_nice_to_have_skills = set(jd.nice_to_haves_raw)
        
        # Calculate gaps
        missing_must_have = jd_must_have_skills - resume_skills
        missing_nice_to_have = jd_nice_to_have_skills - resume_skills
        
        # Calculate coverage percentages
        must_have_coverage = len(resume_skills.intersection(jd_must_have_skills)) / len(jd_must_have_skills) if jd_must_have_skills else 0.0
        nice_to_have_coverage = len(resume_skills.intersection(jd_nice_to_have_skills)) / len(jd_nice_to_have_skills) if jd_nice_to_have_skills else 0.0
        
        return {
            'missing_must_have_skills': list(missing_must_have),
            'missing_nice_to_have_skills': list(missing_nice_to_have),
            'must_have_coverage': must_have_coverage,
            'nice_to_have_coverage': nice_to_have_coverage,
            'total_missing_skills': len(missing_must_have) + len(missing_nice_to_have),
            'critical_gaps': len(missing_must_have),
            'nice_to_have_gaps': len(missing_nice_to_have)
        }
    
    def get_skill_overlap(self, resume: ParsedResume, jd: ParsedJD) -> dict:
        """Analyze skill overlap between resume and job description."""
        resume_skills = set(resume.skills_norm)
        jd_must_have_skills = set(jd.must_haves_raw)
        jd_nice_to_have_skills = set(jd.nice_to_haves_raw)
        
        # Calculate overlaps
        must_have_overlap = resume_skills.intersection(jd_must_have_skills)
        nice_to_have_overlap = resume_skills.intersection(jd_nice_to_have_skills)
        
        # Calculate percentages
        must_have_overlap_pct = len(must_have_overlap) / len(jd_must_have_skills) if jd_must_have_skills else 0.0
        nice_to_have_overlap_pct = len(nice_to_have_overlap) / len(jd_nice_to_have_skills) if jd_nice_to_have_skills else 0.0
        
        return {
            'must_have_overlap': list(must_have_overlap),
            'nice_to_have_overlap': list(nice_to_have_overlap),
            'must_have_overlap_percentage': must_have_overlap_pct,
            'nice_to_have_overlap_percentage': nice_to_have_overlap_pct,
            'total_overlap': len(must_have_overlap) + len(nice_to_have_overlap)
        }
