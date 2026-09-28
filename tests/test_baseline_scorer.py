"""Tests for the deterministic baseline scorer.

The scorer is the one component the spec records as already working, so these
are regression guards rather than bug reproductions. The score must stay
reproducible and explainable: no LLM influence and no randomness.
"""

import pytest

from core.schemas.models import ParsedJD, ParsedResume
from core.scoring.baseline_scorer import BaselineScorer


@pytest.fixture
def scorer() -> BaselineScorer:
    return BaselineScorer()


def make_pair(resume_skills, must, nice):
    return (
        ParsedResume(skills_norm=list(resume_skills)),
        ParsedJD(must_have_skills=list(must), nice_to_have_skills=list(nice)),
    )


def test_perfect_match_scores_one(scorer):
    resume, jd = make_pair(["python", "docker"], ["python"], ["docker"])
    score, _, missing, _ = scorer.calculate_score(resume, jd)
    assert score == pytest.approx(1.0)
    assert missing == []


def test_no_match_scores_zero(scorer):
    resume, jd = make_pair(["java"], ["python"], ["docker"])
    score, matched, _, _ = scorer.calculate_score(resume, jd)
    assert score == pytest.approx(0.0)
    assert matched == []


def test_empty_job_description_scores_zero(scorer):
    resume, jd = make_pair(["python"], [], [])
    assert scorer.calculate_score(resume, jd)[0] == pytest.approx(0.0)


def test_must_have_is_weighted_double_a_nice_to_have(scorer):
    """One must-have out of one must plus one nice is 2 of 3 total weight."""
    resume, jd = make_pair(["python"], ["python"], ["docker"])
    must_only = scorer.calculate_score(resume, jd)[0]

    resume, jd = make_pair(["docker"], ["python"], ["docker"])
    nice_only = scorer.calculate_score(resume, jd)[0]

    assert must_only == pytest.approx(2 / 3)
    assert nice_only == pytest.approx(1 / 3)
    assert must_only > nice_only


def test_weights_are_configurable():
    resume, jd = make_pair(["python"], ["python"], ["docker"])
    scorer = BaselineScorer(must_have_weight=3.0, nice_to_have_weight=1.0)
    assert scorer.calculate_score(resume, jd)[0] == pytest.approx(3 / 4)


def test_score_is_always_within_unit_range(scorer):
    resume, jd = make_pair(["python", "docker", "aws"], ["python", "go"], ["docker"])
    score = scorer.calculate_score(resume, jd)[0]
    assert 0.0 <= score <= 1.0


def test_score_is_deterministic_across_calls(scorer):
    resume, jd = make_pair(["python", "docker"], ["python", "go"], ["docker", "aws"])
    scores = {scorer.calculate_score(resume, jd)[0] for _ in range(5)}
    assert len(scores) == 1


def test_extra_resume_skills_do_not_inflate_the_score(scorer):
    lean, jd = make_pair(["python"], ["python"], [])
    padded, _ = make_pair(["python"] + [f"skill{i}" for i in range(20)], ["python"], [])
    assert scorer.calculate_score(lean, jd)[0] == scorer.calculate_score(padded, jd)[0]


def test_missing_skills_lists_both_categories(scorer):
    resume, jd = make_pair(["python"], ["python", "go"], ["docker"])
    _, _, missing, _ = scorer.calculate_score(resume, jd)
    assert set(missing) == {"go", "docker"}


def test_nice_matches_only_contains_nice_to_haves(scorer):
    resume, jd = make_pair(["python", "docker"], ["python"], ["docker"])
    _, _, _, nice = scorer.calculate_score(resume, jd)
    assert nice == ["docker"]


def test_detailed_breakdown_counts_agree_with_score(scorer):
    resume, jd = make_pair(["python", "docker"], ["python", "go"], ["docker", "aws"])
    breakdown = scorer.get_detailed_breakdown(resume, jd)
    assert breakdown["must_have_matches"] == 1
    assert breakdown["must_have_total"] == 2
    assert breakdown["nice_to_have_matches"] == 1
    assert breakdown["nice_to_have_total"] == 2
    assert breakdown["baseline_score"] == pytest.approx(scorer.calculate_score(resume, jd)[0])


def test_skill_gaps_reports_critical_gaps(scorer):
    resume, jd = make_pair(["python"], ["python", "go"], ["docker"])
    gaps = scorer.get_skill_gaps(resume, jd)
    assert gaps["critical_gaps"] == 1
    assert gaps["must_have_coverage"] == pytest.approx(0.5)


def test_skill_gaps_handles_empty_jd(scorer):
    resume, jd = make_pair(["python"], [], [])
    gaps = scorer.get_skill_gaps(resume, jd)
    assert gaps["must_have_coverage"] == pytest.approx(0.0)
    assert gaps["total_missing_skills"] == 0


def test_skill_overlap_percentages(scorer):
    resume, jd = make_pair(["python", "docker"], ["python", "go"], ["docker"])
    overlap = scorer.get_skill_overlap(resume, jd)
    assert overlap["must_have_overlap_percentage"] == pytest.approx(0.5)
    assert overlap["nice_to_have_overlap_percentage"] == pytest.approx(1.0)
    assert overlap["total_overlap"] == 2
