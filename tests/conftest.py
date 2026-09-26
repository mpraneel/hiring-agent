"""Shared test configuration.

The LLM environment is neutralised here, at import time and before any module
under test is imported, so that a developer's local .env can never cause the
suite to construct a real provider client or reach the network. python-dotenv
does not overwrite variables that are already set, so seeding them as empty
strings is enough to shadow a real .env.
"""

import os

os.environ["LLM_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = ""

import sys
from pathlib import Path
from typing import Callable, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.pdf_builder import build_pdf  # noqa: E402


@pytest.fixture(autouse=True)
def _no_llm_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee every test starts with the LLM disabled."""
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "")


@pytest.fixture
def make_pdf() -> Callable[[Sequence[str]], bytes]:
    """Return a callable that builds a one page PDF from lines of text."""
    return build_pdf


SAMPLE_RESUME_LINES = [
    "Jane Doe",
    "jane.doe@example.com | (555) 123-4567",
    "",
    "Summary",
    "Backend engineer who gained experience in distributed systems.",
    "",
    "Skills",
    "Python, scikit-learn, Docker, React, Git",
    "",
    "Experience",
    "Senior Engineer at Acme Corp",
    "- Built payment services in Python",
    "",
    "Education",
    "Bachelor of Science in Computer Science",
]

SAMPLE_JD = """Senior Backend Engineer

Requirements:
- 5+ years of experience with Python
- Strong knowledge of Docker

Preferred Qualifications:
- Experience with Kubernetes
- Familiarity with React
"""


@pytest.fixture
def sample_resume_lines() -> list[str]:
    return list(SAMPLE_RESUME_LINES)


@pytest.fixture
def sample_resume_text() -> str:
    return "\n".join(SAMPLE_RESUME_LINES)


@pytest.fixture
def sample_jd() -> str:
    return SAMPLE_JD
