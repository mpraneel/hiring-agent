"""Tests for resume text and PDF parsing."""

import pytest

from core.parsers.resume_parser import ResumeParser


@pytest.fixture
def parser() -> ResumeParser:
    return ResumeParser()


# --------------------------------------------------------------------------
# Behaviour that already works
# --------------------------------------------------------------------------


def test_extracts_email(parser, sample_resume_text):
    assert parser.parse_text(sample_resume_text).email == "jane.doe@example.com"


def test_extracts_phone(parser, sample_resume_text):
    phone = parser.parse_text(sample_resume_text).phone
    assert phone is not None
    assert "555" in phone


def test_extracts_name(parser, sample_resume_text):
    assert parser.parse_text(sample_resume_text).name == "Jane Doe"


def test_missing_contact_details_are_none(parser):
    parsed = parser.parse_text("Skills\nPython\n")
    assert parsed.email is None
    assert parsed.phone is None


def test_parse_pdf_reads_a_real_pdf(parser, make_pdf, sample_resume_lines, tmp_path):
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(make_pdf(sample_resume_lines))
    parsed = parser.parse_pdf(str(pdf_path))
    assert parsed.email == "jane.doe@example.com"
    assert parsed.name == "Jane Doe"


def test_parse_pdf_raises_on_a_non_pdf(parser, tmp_path):
    bad = tmp_path / "not.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(Exception):
        parser.parse_pdf(str(bad))


# --------------------------------------------------------------------------
# Section detection is far too loose (Phase 2)
# --------------------------------------------------------------------------


def test_prose_mentioning_experience_is_not_a_section_header(parser):
    line = "Backend engineer who gained experience in distributed systems."
    assert parser._is_section_header(line, ["skills"]) is False


def test_skills_are_found_despite_a_summary_paragraph(parser, sample_resume_text):
    parsed = parser.parse_text(sample_resume_text)
    assert "python" in parsed.skills_norm
    assert "docker" in parsed.skills_norm


def test_long_line_is_never_a_section_header(parser):
    long_line = "Skills developed across six years of backend and platform engineering work"
    assert parser._is_section_header(long_line, ["skills"]) is False


@pytest.mark.parametrize("header", ["SKILLS", "Skills:", "  skills  ", "Skills -"])
def test_header_variants_are_recognised(parser, header):
    """Stricter Phase 2 matching must still accept real header spellings."""
    assert parser._is_section_header(header, ["skills"]) is True


# --------------------------------------------------------------------------
# Skill splitting (Phase 2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("skill", ["scikit-learn", "ci-cd", "c-sharp"])
def test_hyphenated_skills_are_not_split(parser, skill):
    extracted = parser._extract_skills(["Python, " + skill + ", Docker"])
    assert skill in [s.lower() for s in extracted]


def test_comma_separated_skills_are_split(parser):
    extracted = [s.lower() for s in parser._extract_skills(["Python, Docker, React"])]
    assert {"python", "docker", "react"} <= set(extracted)


# --------------------------------------------------------------------------
# Experience fields (Phase 2)
# --------------------------------------------------------------------------


def test_experience_captures_company_and_dates(parser):
    text = "\n".join(
        [
            "Experience",
            "Senior Engineer at Acme Corp, 2019 - 2023",
            "- Built payment services",
            "",
            "Education",
            "BSc Computer Science",
        ]
    )
    experiences = parser.parse_text(text).experiences
    assert experiences, "no experiences parsed"
    first = experiences[0]
    assert first.company is not None
    assert first.start is not None
    assert first.end is not None


def test_experience_bullets_are_captured(parser):
    text = "\n".join(
        [
            "Experience",
            "Senior Engineer at Acme Corp, 2019 - 2023",
            "- Built payment services in Python",
            "- Led a team of four",
        ]
    )
    experiences = parser.parse_text(text).experiences
    assert experiences
    assert len(experiences[0].bullets) == 2
