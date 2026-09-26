"""Tests for LLM structured extraction.

Every provider call is a stub. The four control flow cases the spec calls for
are covered explicitly: valid output, schema-invalid output then a valid retry,
two failures then the deterministic fallback, and hallucinated evidence dropped.
"""

import json

import pytest
from pydantic import ValidationError

from core.extraction.llm_extractor import LLMExtractor, normalize_whitespace
from core.extraction.models import JDExtraction, ResumeExtraction
from core.extraction.provider import StructuredProviderError

JD_TEXT = """Senior Backend Engineer

Requirements
- Strong Python and PostgreSQL
- Comfortable with Docker

Preferred Qualifications
- Experience with Kubernetes
"""

RESUME_TEXT = """Alex Morgan

Skills
Python, FastAPI, PostgreSQL, Docker

Experience
Senior Engineer at Acme, 2020 - 2024
- Tuned PostgreSQL queries
"""


class ScriptedProvider:
    """Returns queued responses in order, recording the prompts it received."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def generate_json(self, system: str, user: str, schema) -> str:
        self.prompts.append((system, user))
        if not self.responses:
            raise AssertionError("provider called more times than the test scripted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RaisingProvider:
    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0

    def generate_json(self, system: str, user: str, schema) -> str:
        self.calls += 1
        raise self.exc


def jd_payload(*requirements) -> str:
    return json.dumps(
        {
            "title": "Senior Backend Engineer",
            "requirements": [
                {"skill": s, "priority": p, "evidence": e} for s, p, e in requirements
            ],
        }
    )


def resume_payload(*skills) -> str:
    return json.dumps({"skills": [{"skill": s, "evidence": e} for s, e in skills]})


def make_extractor(provider) -> LLMExtractor:
    return LLMExtractor(provider=provider)


# ---------------------------------------------------------------------------
# Evidence normalization
# ---------------------------------------------------------------------------


def test_whitespace_normalization_collapses_and_lowercases():
    assert normalize_whitespace("  Strong   Python\nand\tDocker ") == "strong python and docker"


def test_whitespace_normalization_of_empty_input():
    assert normalize_whitespace("") == ""
    assert normalize_whitespace(None) == ""


# ---------------------------------------------------------------------------
# Case 1: valid output
# ---------------------------------------------------------------------------


def test_valid_jd_output_is_accepted():
    provider = ScriptedProvider(
        jd_payload(
            ("Python", "must", "Strong Python and PostgreSQL"),
            ("Kubernetes", "nice", "Experience with Kubernetes"),
        )
    )
    extraction = make_extractor(provider).extract_jd(JD_TEXT)
    by_skill = {r.skill: r.priority for r in extraction.requirements}
    assert by_skill == {"python": "must", "kubernetes": "nice"}
    assert len(provider.prompts) == 1


def test_valid_resume_output_is_accepted():
    provider = ScriptedProvider(
        resume_payload(("FastAPI", "Python, FastAPI, PostgreSQL, Docker"))
    )
    extraction = make_extractor(provider).extract_resume(RESUME_TEXT)
    assert [s.skill for s in extraction.skills] == ["fastapi"]


def test_skills_are_normalized_through_the_ontology():
    provider = ScriptedProvider(
        jd_payload(("Postgres", "must", "Strong Python and PostgreSQL"))
    )
    extraction = make_extractor(provider).extract_jd(JD_TEXT)
    assert [r.skill for r in extraction.requirements] == ["postgresql"]


def test_unknown_skills_are_kept_as_is():
    provider = ScriptedProvider(
        jd_payload(("Weirdlang", "must", "Strong Python and PostgreSQL"))
    )
    extraction = make_extractor(provider).extract_jd(JD_TEXT)
    assert [r.skill for r in extraction.requirements] == ["weirdlang"]


def test_evidence_matching_tolerates_whitespace_differences():
    """Providers rewrap lines when copying spans; the words are what matter."""
    provider = ScriptedProvider(
        jd_payload(("Python", "must", "Strong    Python\n   and PostgreSQL"))
    )
    extraction = make_extractor(provider).extract_jd(JD_TEXT)
    assert [r.skill for r in extraction.requirements] == ["python"]


def test_a_skill_claimed_both_ways_becomes_must():
    provider = ScriptedProvider(
        jd_payload(
            ("Docker", "nice", "Comfortable with Docker"),
            ("Docker", "must", "Comfortable with Docker"),
        )
    )
    extraction = make_extractor(provider).extract_jd(JD_TEXT)
    assert [(r.skill, r.priority) for r in extraction.requirements] == [("docker", "must")]


# ---------------------------------------------------------------------------
# Case 2: schema-invalid output, then a valid retry
# ---------------------------------------------------------------------------


def test_invalid_priority_is_retried_once_and_then_accepted():
    bad = json.dumps(
        {
            "title": "x",
            "requirements": [
                {"skill": "Python", "priority": "REQUIRED", "evidence": "Strong Python"}
            ],
        }
    )
    good = jd_payload(("Python", "must", "Strong Python and PostgreSQL"))
    provider = ScriptedProvider(bad, good)

    extraction = make_extractor(provider).extract_jd(JD_TEXT)
    assert [r.skill for r in extraction.requirements] == ["python"]
    assert len(provider.prompts) == 2


def test_the_retry_prompt_carries_the_validation_error():
    bad = json.dumps({"requirements": [{"skill": "Python"}]})
    good = jd_payload(("Python", "must", "Strong Python and PostgreSQL"))
    provider = ScriptedProvider(bad, good)

    make_extractor(provider).extract_jd(JD_TEXT)
    retry_prompt = provider.prompts[1][1]
    assert "rejected by schema validation" in retry_prompt
    assert "evidence" in retry_prompt


def test_malformed_json_is_retried():
    provider = ScriptedProvider(
        "this is not json at all",
        jd_payload(("Python", "must", "Strong Python and PostgreSQL")),
    )
    extraction = make_extractor(provider).extract_jd(JD_TEXT)
    assert [r.skill for r in extraction.requirements] == ["python"]
    assert len(provider.prompts) == 2


# ---------------------------------------------------------------------------
# Case 3: two failures, then the deterministic fallback
# ---------------------------------------------------------------------------


def test_two_schema_failures_raise_from_extract_jd():
    bad = json.dumps({"requirements": [{"skill": "Python"}]})
    provider = ScriptedProvider(bad, bad)
    with pytest.raises(ValidationError):
        make_extractor(provider).extract_jd(JD_TEXT)
    assert len(provider.prompts) == 2


def test_parse_jd_falls_back_after_two_failures():
    bad = json.dumps({"requirements": [{"skill": "Python"}]})
    parsed = make_extractor(ScriptedProvider(bad, bad)).parse_jd(JD_TEXT, mode="hybrid")
    assert parsed.extraction_path == "fallback"
    assert "python" in parsed.must_have_skills


def test_parse_resume_falls_back_after_two_failures():
    parsed = make_extractor(ScriptedProvider("{}x", "{}x")).parse_resume(
        RESUME_TEXT, mode="hybrid"
    )
    assert parsed.extraction_path == "fallback"
    assert "python" in parsed.skills_norm


def test_provider_transport_failure_falls_back():
    provider = RaisingProvider(StructuredProviderError("network down"))
    parsed = make_extractor(provider).parse_jd(JD_TEXT, mode="llm")
    assert parsed.extraction_path == "fallback"
    assert "python" in parsed.must_have_skills
    # A transport error is not retried by the extractor.
    assert provider.calls == 1


def test_no_provider_configured_uses_the_deterministic_path():
    extractor = LLMExtractor(provider=None)
    if extractor.available:
        pytest.skip("a real provider is configured in this environment")
    parsed = extractor.parse_jd(JD_TEXT, mode="hybrid")
    assert parsed.extraction_path == "deterministic"


# ---------------------------------------------------------------------------
# Case 4: hallucinated evidence is dropped
# ---------------------------------------------------------------------------


def test_requirement_with_absent_evidence_is_dropped():
    provider = ScriptedProvider(
        jd_payload(
            ("Python", "must", "Strong Python and PostgreSQL"),
            ("Rust", "must", "Five years of Rust in production"),
        )
    )
    extraction = make_extractor(provider).extract_jd(JD_TEXT)
    skills = [r.skill for r in extraction.requirements]
    assert "python" in skills
    assert "rust" not in skills


def test_resume_skill_with_absent_evidence_is_dropped():
    provider = ScriptedProvider(
        resume_payload(
            ("FastAPI", "Python, FastAPI, PostgreSQL, Docker"),
            ("Kubernetes", "Ran the platform on Kubernetes"),
        )
    )
    extraction = make_extractor(provider).extract_resume(RESUME_TEXT)
    skills = [s.skill for s in extraction.skills]
    assert "fastapi" in skills
    assert "kubernetes" not in skills


def test_empty_evidence_is_dropped():
    provider = ScriptedProvider(jd_payload(("Python", "must", "")))
    assert make_extractor(provider).extract_jd(JD_TEXT).requirements == []


def test_paraphrased_evidence_is_dropped():
    """A plausible paraphrase is still not what the document says."""
    provider = ScriptedProvider(
        jd_payload(("Python", "must", "The role requires solid Python skills"))
    )
    assert make_extractor(provider).extract_jd(JD_TEXT).requirements == []


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def test_deterministic_mode_never_calls_the_provider():
    provider = ScriptedProvider()  # any call raises
    parsed = make_extractor(provider).parse_jd(JD_TEXT, mode="deterministic")
    assert parsed.extraction_path == "deterministic"
    assert provider.prompts == []


def test_llm_mode_uses_only_the_llm_result():
    provider = ScriptedProvider(jd_payload(("Python", "must", "Strong Python")))
    parsed = make_extractor(provider).parse_jd(JD_TEXT, mode="llm")
    assert parsed.extraction_path == "llm"
    assert parsed.must_have_skills == ["python"]
    # The deterministic parser also finds docker and postgresql; llm mode must not.
    assert "docker" not in parsed.must_have_skills


def test_hybrid_mode_unions_both_extractions():
    provider = ScriptedProvider(jd_payload(("Weirdlang", "must", "Strong Python")))
    parsed = make_extractor(provider).parse_jd(JD_TEXT, mode="hybrid")
    assert parsed.extraction_path == "hybrid"
    assert "weirdlang" in parsed.must_have_skills  # from the LLM
    assert "docker" in parsed.must_have_skills  # from the deterministic parser


def test_hybrid_mode_lets_the_llm_win_a_priority_conflict():
    """The deterministic parser calls Docker a must; the LLM says nice."""
    provider = ScriptedProvider(jd_payload(("Docker", "nice", "Comfortable with Docker")))
    parsed = make_extractor(provider).parse_jd(JD_TEXT, mode="hybrid")
    assert "docker" in parsed.nice_to_have_skills
    assert "docker" not in parsed.must_have_skills


def test_hybrid_lists_are_always_disjoint():
    provider = ScriptedProvider(
        jd_payload(
            ("Python", "nice", "Strong Python and PostgreSQL"),
            ("Kubernetes", "must", "Experience with Kubernetes"),
        )
    )
    parsed = make_extractor(provider).parse_jd(JD_TEXT, mode="hybrid")
    assert set(parsed.must_have_skills) & set(parsed.nice_to_have_skills) == set()


def test_hybrid_resume_mode_unions_skills():
    provider = ScriptedProvider(resume_payload(("Weirdlang", "Tuned PostgreSQL queries")))
    parsed = make_extractor(provider).parse_resume(RESUME_TEXT, mode="hybrid")
    assert parsed.extraction_path == "hybrid"
    assert "weirdlang" in parsed.skills_norm
    assert "python" in parsed.skills_norm


def test_llm_resume_mode_keeps_deterministic_contact_details():
    """Contact details are a solved regex problem; do not pay an LLM for them."""
    provider = ScriptedProvider(resume_payload(("FastAPI", "Python, FastAPI")))
    parsed = make_extractor(provider).parse_resume(RESUME_TEXT, mode="llm")
    assert parsed.extraction_path == "llm"
    assert parsed.name == "Alex Morgan"
    assert parsed.skills_norm == ["fastapi"]


def test_extraction_records_a_valid_path_value():
    provider = ScriptedProvider(jd_payload(("Python", "must", "Strong Python")))
    parsed = make_extractor(provider).parse_jd(JD_TEXT, mode="llm")
    assert parsed.extraction_path in ("llm", "deterministic", "fallback", "hybrid")


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def test_the_document_is_included_in_the_prompt():
    provider = ScriptedProvider(jd_payload(("Python", "must", "Strong Python")))
    make_extractor(provider).extract_jd(JD_TEXT)
    system, user = provider.prompts[0]
    assert "verbatim" in system
    assert "Senior Backend Engineer" in user


def test_the_jd_prompt_forbids_inventing_skills():
    provider = ScriptedProvider(jd_payload(("Python", "must", "Strong Python")))
    make_extractor(provider).extract_jd(JD_TEXT)
    assert "do not invent" in provider.prompts[0][0].lower()
