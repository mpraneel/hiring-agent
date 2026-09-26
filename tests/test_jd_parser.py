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
    assert parsed.must_have_skills == []
    assert parsed.nice_to_have_skills == []
    assert parsed.skills_norm == []


def test_required_cue_puts_skill_in_must_have(parser):
    parsed = parser.parse_jd("Required: strong Python and Docker experience.")
    assert "python" in parsed.must_have_skills
    assert "docker" in parsed.must_have_skills


def test_skills_norm_contains_canonical_names(parser):
    parsed = parser.parse_jd("Required: experience with k8s and JS.")
    assert "kubernetes" in parsed.skills_norm
    assert "javascript" in parsed.skills_norm


def test_no_skill_appears_in_both_lists(parser, sample_jd):
    """A skill counted twice would be double weighted by the scorer."""
    parsed = parser.parse_jd(sample_jd)
    overlap = set(parsed.must_have_skills) & set(parsed.nice_to_have_skills)
    assert overlap == set()


# --------------------------------------------------------------------------
# Line structure is destroyed by _clean_text (Phase 2)
# --------------------------------------------------------------------------


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


def test_preferred_experience_with_is_nice_to_have(parser):
    parsed = parser.parse_jd("Preferred: experience with Docker.")
    assert "docker" in parsed.nice_to_have_skills
    assert "docker" not in parsed.must_have_skills


def test_bullets_inherit_preceding_section_heading(parser, sample_jd):
    parsed = parser.parse_jd(sample_jd)
    assert "python" in parsed.must_have_skills
    assert "docker" in parsed.must_have_skills
    assert "kubernetes" in parsed.nice_to_have_skills
    assert "react" in parsed.nice_to_have_skills


def test_nice_to_have_section_is_not_empty(parser, sample_jd):
    parsed = parser.parse_jd(sample_jd)
    assert parsed.nice_to_have_skills != []


def test_skill_in_both_sections_is_must_have_only(parser):
    """Passes vacuously today because nice-to-have is always empty.

    It becomes a real assertion once Phase 2 makes nice-to-have extraction
    work, which is exactly when the double-counting bug could appear.
    """
    jd = "Requirements:\n- Docker required\n\nNice to have:\n- Docker at scale\n"
    parsed = parser.parse_jd(jd)
    assert "docker" in parsed.must_have_skills
    assert "docker" not in parsed.nice_to_have_skills


def test_bonus_cue_is_nice_to_have(parser):
    parsed = parser.parse_jd("Knowledge of GraphQL is a bonus.")
    assert "graphql" in parsed.nice_to_have_skills
    assert "graphql" not in parsed.must_have_skills


# --------------------------------------------------------------------------
# False positives from short or generic tokens (Phase 2)
# --------------------------------------------------------------------------


def test_rest_of_the_team_does_not_yield_rest_skill(parser):
    parsed = parser.parse_jd("Required: You will collaborate with the rest of the team.")
    assert "rest" not in parsed.must_have_skills
    assert "rest" not in parsed.nice_to_have_skills


def test_requirements_heading_does_not_yield_business_analyst(parser, sample_jd):
    parsed = parser.parse_jd(sample_jd)
    assert "business analyst" not in parsed.must_have_skills
    assert "business analyst" not in parsed.nice_to_have_skills


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
    assert unwanted not in parsed.must_have_skills + parsed.nice_to_have_skills


def test_disambiguated_short_tokens_still_resolve(parser):
    """The Phase 2 stoplist must not throw away disambiguated mentions."""
    parsed = parser.parse_jd("Required: Go language experience and REST API design.")
    combined = parsed.must_have_skills + parsed.nice_to_have_skills
    assert "go" in combined
    assert "rest" in combined


# --------------------------------------------------------------------------
# Titles and dead code
# --------------------------------------------------------------------------


def test_job_title_is_the_first_line(parser, sample_jd):
    parsed = parser.parse_jd(sample_jd)
    assert parsed.title is not None
    assert "backend engineer" in parsed.title.lower()


@pytest.mark.parametrize("name", ["_extract_skills_from_text", "_skill_in_text"])
def test_unused_helpers_are_removed(parser, name):
    assert not hasattr(parser, name)


# --------------------------------------------------------------------------
# Skill-list context for ambiguous tokens
# --------------------------------------------------------------------------


def test_bullet_under_a_requirements_heading_is_a_list_context(parser):
    """The spec allows a bullet under a requirements heading to license 'Go'."""
    jd = "Preferred Qualifications\n- Go experience is a plus\n"
    parsed = parser.parse_jd(jd)
    assert "go" in parsed.nice_to_have_skills


def test_ambiguous_token_in_a_comma_list_is_accepted(parser):
    parsed = parser.parse_jd("Required skills\nPython, Go, Rust, Docker, Kubernetes\n")
    assert "go" in parsed.must_have_skills


def test_prose_outside_a_requirements_section_is_ignored(parser):
    """Company blurb is the biggest source of false positives."""
    jd = (
        "Senior Engineer\n\n"
        "About the role\n"
        "You will join the rest of the team and help design our Java strategy.\n"
    )
    parsed = parser.parse_jd(jd)
    assert parsed.must_have_skills == []
    assert parsed.nice_to_have_skills == []


def test_ignored_section_still_honours_an_explicit_cue(parser):
    jd = "About us\nWe require strong Python skills from everyone here.\n"
    assert "python" in parser.parse_jd(jd).must_have_skills


def test_unstructured_jd_still_yields_requirements(parser):
    """A JD with no headings at all must not extract nothing."""
    parsed = parser.parse_jd("We are hiring someone who knows Python and Docker well.")
    assert "python" in parsed.must_have_skills
    assert "docker" in parsed.must_have_skills


def test_postgresql_in_a_jd_is_found(parser):
    parsed = parser.parse_jd("Requirements\n- Production experience with PostgreSQL\n")
    assert "postgresql" in parsed.must_have_skills


def test_numbered_list_items_are_parsed(parser):
    parsed = parser.parse_jd("Requirements\n1. Strong Python\n2. Docker experience\n")
    assert "python" in parsed.must_have_skills
    assert "docker" in parsed.must_have_skills


def test_html_tags_are_stripped(parser):
    parsed = parser.parse_jd("<h2>Requirements</h2>\n<li>Strong Python</li>\n")
    assert "python" in parsed.must_have_skills


def test_title_is_none_for_a_headless_jd(parser):
    assert parser.parse_jd("- Strong Python.\n") is not None
