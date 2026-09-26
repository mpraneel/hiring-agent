"""Tests for job description parsing and must/nice classification."""

import pytest

from core.parsers.jd_parser import JDParser


@pytest.fixture
def parser() -> JDParser:
    return JDParser()


# --------------------------------------------------------------------------
# Behaviour that already works
# --------------------------------------------------------------------------


def test_empty_jd_yields_no_requirements(parser):
    parsed = parser.parse_jd("")
    assert parsed.must_haves_raw == []
    assert parsed.nice_to_haves_raw == []
    assert parsed.skills_norm == []


def test_required_cue_puts_skill_in_must_have(parser):
    parsed = parser.parse_jd("Required: strong Python and Docker experience.")
    assert "python" in parsed.must_haves_raw
    assert "docker" in parsed.must_haves_raw


def test_skills_norm_contains_canonical_names(parser):
    parsed = parser.parse_jd("Required: experience with k8s and JS.")
    assert "kubernetes" in parsed.skills_norm
    assert "javascript" in parsed.skills_norm


def test_no_skill_appears_in_both_lists(parser, sample_jd):
    """A skill counted twice would be double weighted by the scorer."""
    parsed = parser.parse_jd(sample_jd)
    overlap = set(parsed.must_haves_raw) & set(parsed.nice_to_haves_raw)
    assert overlap == set()


# --------------------------------------------------------------------------
# Line structure is destroyed by _clean_text (Phase 2)
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Phase 2: _clean_text collapses newlines, so bullet and title logic sees one line",
    strict=True,
)
def test_clean_text_preserves_line_structure(parser, sample_jd):
    cleaned = parser._clean_text(sample_jd)
    assert cleaned.count("\n") >= 5


def test_clean_text_collapses_runs_of_spaces_within_a_line(parser):
    """Collapsing runs of spaces must survive the Phase 2 newline fix."""
    cleaned = parser._clean_text("Requirements:\n-   Python     and    Docker\n")
    assert "Python and Docker" in cleaned or "python and docker" in cleaned


# --------------------------------------------------------------------------
# must / nice classification (Phase 2)
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Phase 2: must-have and nice-to-have pattern lists overlap, must-have wins",
    strict=True,
)
def test_preferred_experience_with_is_nice_to_have(parser):
    parsed = parser.parse_jd("Preferred: experience with Docker.")
    assert "docker" in parsed.nice_to_haves_raw
    assert "docker" not in parsed.must_haves_raw


@pytest.mark.xfail(
    reason="Phase 2: bullets under a Preferred heading are all forced into must-have",
    strict=True,
)
def test_bullets_inherit_preceding_section_heading(parser, sample_jd):
    parsed = parser.parse_jd(sample_jd)
    assert "python" in parsed.must_haves_raw
    assert "docker" in parsed.must_haves_raw
    assert "kubernetes" in parsed.nice_to_haves_raw
    assert "react" in parsed.nice_to_haves_raw


@pytest.mark.xfail(
    reason="Phase 2: nice-to-have extraction is dead because bullets are never seen",
    strict=True,
)
def test_nice_to_have_section_is_not_empty(parser, sample_jd):
    parsed = parser.parse_jd(sample_jd)
    assert parsed.nice_to_haves_raw != []


def test_skill_in_both_sections_is_must_have_only(parser):
    """Passes vacuously today because nice-to-have is always empty.

    It becomes a real assertion once Phase 2 makes nice-to-have extraction
    work, which is exactly when the double-counting bug could appear.
    """
    jd = "Requirements:\n- Docker required\n\nNice to have:\n- Docker at scale\n"
    parsed = parser.parse_jd(jd)
    assert "docker" in parsed.must_haves_raw
    assert "docker" not in parsed.nice_to_haves_raw


@pytest.mark.xfail(
    reason="Phase 2: bonus and plus cues are not honoured over must-have cues",
    strict=True,
)
def test_bonus_cue_is_nice_to_have(parser):
    parsed = parser.parse_jd("Knowledge of GraphQL is a bonus.")
    assert "graphql" in parsed.nice_to_haves_raw
    assert "graphql" not in parsed.must_haves_raw


# --------------------------------------------------------------------------
# False positives from short or generic tokens (Phase 2)
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Phase 2: bare 'rest' matches ordinary English on a word boundary",
    strict=True,
)
def test_rest_of_the_team_does_not_yield_rest_skill(parser):
    parsed = parser.parse_jd("Required: You will collaborate with the rest of the team.")
    assert "rest" not in parsed.must_haves_raw
    assert "rest" not in parsed.nice_to_haves_raw


@pytest.mark.xfail(
    reason="Phase 2: 'Requirements' heading matches the business analyst variant",
    strict=True,
)
def test_requirements_heading_does_not_yield_business_analyst(parser, sample_jd):
    parsed = parser.parse_jd(sample_jd)
    assert "business analyst" not in parsed.must_haves_raw
    assert "business analyst" not in parsed.nice_to_haves_raw


@pytest.mark.xfail(
    reason="Phase 2: ambiguous tokens need a skill-list context or a disambiguating neighbour",
    strict=True,
)
@pytest.mark.parametrize(
    ("text", "unwanted"),
    [
        ("Required: We will go over the roadmap together.", "go"),
        ("Required: Please find attached the team charter.", "find"),
        ("Required: A design-led culture is important to us.", "figma"),
    ],
)
def test_ambiguous_tokens_do_not_fire_in_prose(parser, text, unwanted):
    parsed = parser.parse_jd(text)
    assert unwanted not in parsed.must_haves_raw + parsed.nice_to_haves_raw


def test_disambiguated_short_tokens_still_resolve(parser):
    """The Phase 2 stoplist must not throw away disambiguated mentions."""
    parsed = parser.parse_jd("Required: Go language experience and REST API design.")
    combined = parsed.must_haves_raw + parsed.nice_to_haves_raw
    assert "go" in combined
    assert "rest" in combined


# --------------------------------------------------------------------------
# Titles and dead code
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Phase 2: the title is truncated because the whole JD is one line",
    strict=True,
)
def test_job_title_is_the_first_line(parser, sample_jd):
    parsed = parser.parse_jd(sample_jd)
    assert parsed.title is not None
    assert "backend engineer" in parsed.title.lower()


@pytest.mark.xfail(reason="Phase 2: dead code should be removed", strict=True)
@pytest.mark.parametrize("name", ["_extract_skills_from_text", "_skill_in_text"])
def test_unused_helpers_are_removed(parser, name):
    assert not hasattr(parser, name)
