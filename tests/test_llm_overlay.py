"""Tests for the grounded explanation layer and the aggregator around it.

The central guarantee: the explainer sees the computed breakdown and nothing
else. If it could reach the resume or the job description it could mine them for
claims the score never accounted for.
"""

import json

import pytest

from core.extraction.provider import StructuredProviderError
from core.schemas.models import ParsedJD, ParsedResume
from core.scoring.aggregate import MatchAggregator, combine_extraction_paths
from core.scoring.llm_overlay import LLMOverlay, ScoreBreakdown


class ScriptedProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_json(self, system, user, schema):
        self.prompts.append((system, user))
        if not self.responses:
            raise AssertionError("provider called more often than scripted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def explanation(rationale="Strong on the must-haves.", suggestions=()):
    return json.dumps({"rationale": rationale, "suggestions": list(suggestions)})


def make_overlay(*responses) -> tuple[LLMOverlay, ScriptedProvider]:
    provider = ScriptedProvider(*responses)
    return LLMOverlay(structured_provider=provider), provider


BREAKDOWN = ScoreBreakdown(
    score=0.6,
    matched_must=["python", "docker"],
    matched_nice=["redis"],
    missing_must=["postgresql"],
    missing_nice=["kubernetes"],
    must_total=3,
    nice_total=2,
    extraction_path="hybrid",
)


# ---------------------------------------------------------------------------
# The breakdown is the only context
# ---------------------------------------------------------------------------


def test_prompt_contains_the_computed_numbers():
    overlay, provider = make_overlay(explanation())
    overlay.explain(BREAKDOWN)
    _, user = provider.prompts[0]
    assert "0.60" in user
    assert "2 of 3" in user
    assert "postgresql" in user


def test_prompt_contains_no_document_text():
    """The explainer must not be able to quote the resume or the JD.

    ScoreBreakdown is the only argument explain() takes, so document text has no
    route into the prompt. This asserts the property end to end rather than
    trusting the signature, and the user prompt is the only place it could land.
    """
    overlay, provider = make_overlay(explanation())
    overlay.explain(BREAKDOWN)
    user = provider.prompts[0][1].lower()
    for leak in ("jane doe", "@example.com", "senior backend engineer", "bachelor"):
        assert leak not in user, f"breakdown prompt leaked {leak!r}"
    # The prompt is short by construction: counts and skill names only.
    assert len(user) < 600


def test_explain_accepts_only_a_breakdown():
    """A signature guard: adding a document argument should be a deliberate act."""
    import inspect

    params = list(inspect.signature(LLMOverlay.explain).parameters)
    assert params == ["self", "breakdown"]


def test_prompt_records_the_extraction_path():
    overlay, provider = make_overlay(explanation())
    overlay.explain(BREAKDOWN)
    assert "hybrid" in provider.prompts[0][1]


def test_system_prompt_forbids_naming_other_skills():
    overlay, provider = make_overlay(explanation())
    overlay.explain(BREAKDOWN)
    assert "only skills that appear in the lists" in provider.prompts[0][0].lower()


def test_breakdown_missing_is_must_then_nice():
    assert BREAKDOWN.missing == ["postgresql", "kubernetes"]


def test_empty_lists_render_as_none():
    empty = ScoreBreakdown(score=0.0)
    assert "Matched must-have: none" in empty.to_prompt()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_valid_explanation_is_returned():
    overlay, _ = make_overlay(explanation("Good fit.", ["Add PostgreSQL experience."]))
    result = overlay.explain(BREAKDOWN)
    assert result.rationale == "Good fit."
    assert result.suggestions == ["Add PostgreSQL experience."]


def test_malformed_json_raises_rather_than_returning_error_text():
    """The old version returned the exception string as the rationale."""
    overlay, _ = make_overlay("not json")
    with pytest.raises(StructuredProviderError):
        overlay.explain(BREAKDOWN)


def test_missing_rationale_field_raises():
    overlay, _ = make_overlay(json.dumps({"suggestions": []}))
    with pytest.raises(StructuredProviderError):
        overlay.explain(BREAKDOWN)


def test_provider_error_propagates():
    overlay, _ = make_overlay(StructuredProviderError("network down"))
    with pytest.raises(StructuredProviderError):
        overlay.explain(BREAKDOWN)


def test_more_than_three_suggestions_is_rejected_by_the_schema():
    """max_length=3 is enforced by Pydantic, not trimmed silently."""
    overlay, _ = make_overlay(
        json.dumps(
            {
                "rationale": "ok",
                "suggestions": [
                    "Add PostgreSQL experience.",
                    "Add Kubernetes experience.",
                    "Mention PostgreSQL tuning.",
                    "Mention Kubernetes operators.",
                ],
            }
        )
    )
    with pytest.raises(StructuredProviderError):
        overlay.explain(BREAKDOWN)


# ---------------------------------------------------------------------------
# Suggestion grounding
# ---------------------------------------------------------------------------


def test_suggestion_naming_a_missing_skill_is_kept():
    overlay, _ = make_overlay(explanation(suggestions=["Add PostgreSQL experience."]))
    assert overlay.explain(BREAKDOWN).suggestions == ["Add PostgreSQL experience."]


def test_suggestion_naming_a_skill_that_is_not_missing_is_dropped():
    """Docker is already matched, so telling the candidate to add it is wrong."""
    overlay, _ = make_overlay(explanation(suggestions=["Add Docker experience."]))
    assert overlay.explain(BREAKDOWN).suggestions == []


def test_suggestion_naming_an_unrelated_skill_is_dropped():
    overlay, _ = make_overlay(explanation(suggestions=["Learn Rust and Haskell."]))
    assert overlay.explain(BREAKDOWN).suggestions == []


def test_grounded_and_ungrounded_suggestions_are_separated():
    overlay, _ = make_overlay(
        explanation(
            suggestions=[
                "Add PostgreSQL experience.",
                "Pick up Rust as well.",
                "Mention Kubernetes work.",
            ]
        )
    )
    assert overlay.explain(BREAKDOWN).suggestions == [
        "Add PostgreSQL experience.",
        "Mention Kubernetes work.",
    ]


def test_dropped_suggestion_is_logged(caplog):
    overlay, _ = make_overlay(explanation(suggestions=["Add Docker experience."]))
    with caplog.at_level("WARNING"):
        overlay.explain(BREAKDOWN)
    assert "not missing" in caplog.text


def test_generic_suggestion_naming_no_skill_is_kept():
    overlay, _ = make_overlay(explanation(suggestions=["Quantify your impact."]))
    assert overlay.explain(BREAKDOWN).suggestions == ["Quantify your impact."]


# ---------------------------------------------------------------------------
# Extraction path combination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("jd_path", "resume_path", "expected"),
    [
        ("llm", "llm", "llm"),
        ("deterministic", "deterministic", "deterministic"),
        ("hybrid", "hybrid", "hybrid"),
        ("fallback", "llm", "fallback"),
        ("llm", "fallback", "fallback"),
        ("llm", "deterministic", "hybrid"),
    ],
)
def test_paths_combine(jd_path, resume_path, expected):
    """A fallback anywhere dominates, so a degraded run is never advertised clean."""
    assert combine_extraction_paths(jd_path, resume_path) == expected


# ---------------------------------------------------------------------------
# Aggregator behaviour
# ---------------------------------------------------------------------------


class StubOverlay:
    def __init__(self, fail=False):
        self.fail = fail

    def explain(self, breakdown):
        if self.fail:
            raise RuntimeError("provider exploded: api-key-xyz in message")
        from core.extraction.models import MatchExplanation

        return MatchExplanation(rationale="Looks good.", suggestions=["Add PostgreSQL."])


def make_pair():
    resume = ParsedResume(skills_norm=["python", "docker"], extraction_path="llm")
    jd = ParsedJD(
        must_have_skills=["python", "postgresql"],
        nice_to_have_skills=["docker"],
        extraction_path="llm",
    )
    return resume, jd


def test_llm_disabled_reports_disabled_status():
    aggregator = MatchAggregator(enable_llm=False)
    result = aggregator.match_resume_to_jd(*make_pair())
    assert result.llm_status == "disabled"
    assert result.llm_rationale is None


def test_llm_success_reports_ok_status():
    aggregator = MatchAggregator(llm_overlay=StubOverlay())
    result = aggregator.match_resume_to_jd(*make_pair())
    assert result.llm_status == "ok"
    assert result.llm_rationale == "Looks good."


def test_llm_failure_reports_unavailable_and_keeps_the_score():
    aggregator = MatchAggregator(llm_overlay=StubOverlay(fail=True))
    baseline = MatchAggregator(enable_llm=False).match_resume_to_jd(*make_pair())
    result = aggregator.match_resume_to_jd(*make_pair())

    assert result.llm_status == "unavailable"
    assert result.llm_rationale is None
    assert result.suggestions == []
    assert result.match_score == baseline.match_score


def test_llm_failure_leaks_no_exception_text():
    aggregator = MatchAggregator(llm_overlay=StubOverlay(fail=True))
    serialized = aggregator.match_resume_to_jd(*make_pair()).model_dump_json()
    assert "api-key-xyz" not in serialized
    assert "provider exploded" not in serialized


def test_match_score_always_equals_baseline_score():
    """Decision 1: the LLM never moves the number."""
    for overlay in (None, StubOverlay(), StubOverlay(fail=True)):
        aggregator = MatchAggregator(llm_overlay=overlay, enable_llm=overlay is not None)
        result = aggregator.match_resume_to_jd(*make_pair())
        assert result.match_score == result.baseline_score


def test_result_records_the_extraction_path():
    aggregator = MatchAggregator(enable_llm=False)
    assert aggregator.match_resume_to_jd(*make_pair()).extraction_path == "llm"


def test_breakdown_is_built_from_the_scorer():
    aggregator = MatchAggregator(enable_llm=False)
    breakdown = aggregator.build_breakdown(*make_pair())
    assert breakdown.matched_must == ["python"]
    assert breakdown.missing_must == ["postgresql"]
    assert breakdown.matched_nice == ["docker"]
    assert breakdown.must_total == 2
    assert breakdown.score == pytest.approx((2 + 1) / (2 * 2 + 1))


def test_detailed_analysis_reports_status_and_path():
    analysis = MatchAggregator(llm_overlay=StubOverlay()).get_detailed_analysis(*make_pair())
    assert analysis["llm_status"] == "ok"
    assert analysis["extraction_path"] == "llm"
    assert analysis["llm_insights"]["rationale"] == "Looks good."


def test_detailed_analysis_on_llm_failure_has_no_insights():
    analysis = MatchAggregator(llm_overlay=StubOverlay(fail=True)).get_detailed_analysis(
        *make_pair()
    )
    assert analysis["llm_status"] == "unavailable"
    assert analysis["llm_insights"] is None
    assert "provider exploded" not in json.dumps(analysis)


def test_compare_multiple_resumes_sorts_by_score():
    jd = ParsedJD(must_have_skills=["python", "docker"])
    weak = ParsedResume(skills_norm=["python"])
    strong = ParsedResume(skills_norm=["python", "docker"])
    ranked = MatchAggregator(enable_llm=False).compare_multiple_resumes([weak, strong], jd)
    assert ranked[0]["match_result"].match_score > ranked[1]["match_result"].match_score
