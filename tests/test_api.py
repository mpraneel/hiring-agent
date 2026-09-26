"""Tests for the FastAPI endpoints.

No test in this module may reach the network. The LLM is either disabled via
the autouse credential fixture in conftest, or replaced with a stub whose
methods are fully under the test's control.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.extraction.models import MatchExplanation


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def resume_upload(make_pdf, sample_resume_lines):
    """A multipart file tuple carrying a valid one page PDF resume."""
    return ("resume.pdf", io.BytesIO(make_pdf(sample_resume_lines)), "application/pdf")


class StubOverlay:
    """Stand-in for LLMOverlay with no network access.

    Records the breakdown it was handed so a test can assert what the
    explanation layer is allowed to see.
    """

    def __init__(
        self,
        rationale="Solid overlap on the must-have skills.",
        suggestions=None,
        fail=False,
    ):
        self.rationale = rationale
        self.suggestions = (
            suggestions if suggestions is not None else ["Add a Kubernetes bullet."]
        )
        self.fail = fail
        self.calls = 0
        self.seen_breakdowns = []

    def explain(self, breakdown):
        self.calls += 1
        self.seen_breakdowns.append(breakdown)
        if self.fail:
            raise RuntimeError("provider exploded: secret-key-abc123 leaked in message")
        return MatchExplanation(rationale=self.rationale, suggestions=list(self.suggestions))


@pytest.fixture
def with_stub_llm(monkeypatch):
    """Install a stub overlay on the app's aggregator and return it."""

    def install(**kwargs):
        stub = StubOverlay(**kwargs)
        import api.main as main

        monkeypatch.setattr(main.match_aggregator, "llm_overlay", stub)
        monkeypatch.setattr(main.match_aggregator, "enable_llm", True)
        return stub

    return install


# --------------------------------------------------------------------------
# Informational endpoints
# --------------------------------------------------------------------------


def test_root_reports_configuration(client):
    body = client.get("/").json()
    assert body["status"] == "running"
    assert body["llm_provider"] in ("openai", "gemini")
    assert "llm_model" in body


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["/api/v1/match", "/api/v1/analyze"])
def test_endpoint_accepts_a_pdf_and_a_job_description(client, resume_upload, sample_jd, endpoint):
    response = client.post(
        endpoint,
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    )
    assert response.status_code == 200
    assert "request_id" in response.json()


def test_match_response_shape(client, resume_upload, sample_jd):
    body = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    ).json()
    for field in (
        "match_score",
        "baseline_score",
        "matched_skills",
        "missing_skills",
        "nice_matches",
        "llm_rationale",
        "suggestions",
        "request_id",
    ):
        assert field in body, f"missing field {field}"
    assert 0.0 <= body["match_score"] <= 1.0


def test_match_score_equals_baseline_when_llm_is_disabled(client, resume_upload, sample_jd):
    """Decision 1: the LLM never moves the number."""
    body = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    ).json()
    assert body["match_score"] == body["baseline_score"]
    assert body["llm_rationale"] is None


def test_request_ids_are_unique_per_request(client, resume_upload, sample_jd, make_pdf, sample_resume_lines):
    first = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    ).json()["request_id"]
    second = client.post(
        "/api/v1/match",
        files={"resume_pdf": ("resume.pdf", io.BytesIO(make_pdf(sample_resume_lines)), "application/pdf")},
        data={"job_description": sample_jd},
    ).json()["request_id"]
    assert first != second


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_non_pdf_upload_is_rejected(client, sample_jd):
    response = client.post(
        "/api/v1/match",
        files={"resume_pdf": ("resume.txt", io.BytesIO(b"plain text"), "text/plain")},
        data={"job_description": sample_jd},
    )
    assert response.status_code == 400


def test_empty_job_description_is_rejected(client, resume_upload):
    response = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": "   "},
    )
    assert response.status_code == 400


def test_missing_job_description_is_a_422(client, resume_upload):
    response = client.post("/api/v1/match", files={"resume_pdf": resume_upload})
    assert response.status_code == 422


def test_missing_both_resume_inputs_is_a_400(client, sample_jd):
    """Either input satisfies the requirement, so neither is a validation 400."""
    response = client.post("/api/v1/match", data={"job_description": sample_jd})
    assert response.status_code == 400
    assert "resume" in response.text.lower()


def test_supplying_both_resume_inputs_is_a_400(client, resume_upload, sample_jd, sample_resume_text):
    response = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd, "resume_text": sample_resume_text},
    )
    assert response.status_code == 400


def test_oversized_pdf_is_rejected(client, sample_jd):
    padded = b"%PDF-1.4\n" + b"0" * (6 * 1024 * 1024)
    response = client.post(
        "/api/v1/match",
        files={"resume_pdf": ("big.pdf", io.BytesIO(padded), "application/pdf")},
        data={"job_description": sample_jd},
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
# LLM behaviour, always stubbed
# --------------------------------------------------------------------------


def test_llm_rationale_is_returned_when_available(client, resume_upload, sample_jd, with_stub_llm):
    stub = with_stub_llm()
    body = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    ).json()
    assert body["llm_rationale"] == stub.rationale
    assert body["suggestions"] == ["Add a Kubernetes bullet."]
    assert stub.calls == 1


def test_llm_does_not_change_the_score(client, resume_upload, sample_jd, with_stub_llm, make_pdf, sample_resume_lines):
    without = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    ).json()

    with_stub_llm()
    with_llm = client.post(
        "/api/v1/match",
        files={"resume_pdf": ("resume.pdf", io.BytesIO(make_pdf(sample_resume_lines)), "application/pdf")},
        data={"job_description": sample_jd},
    ).json()

    assert with_llm["match_score"] == without["match_score"]
    assert with_llm["baseline_score"] == without["baseline_score"]


def test_llm_failure_still_returns_a_score(client, resume_upload, sample_jd, with_stub_llm):
    with_stub_llm(fail=True)
    response = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    )
    assert response.status_code == 200
    assert 0.0 <= response.json()["match_score"] <= 1.0


def test_llm_failure_does_not_leak_exception_text(client, resume_upload, sample_jd, with_stub_llm):
    with_stub_llm(fail=True)
    body = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    ).json()
    serialized = json.dumps(body)
    assert "secret-key-abc123" not in serialized
    assert "provider exploded" not in serialized
    assert body["llm_rationale"] is None


def test_llm_status_is_reported(client, resume_upload, sample_jd, with_stub_llm):
    with_stub_llm(fail=True)
    body = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    ).json()
    assert body["llm_status"] == "unavailable"


def test_extraction_path_is_reported(client, resume_upload, sample_jd):
    body = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    ).json()
    assert body["extraction_path"] in ("llm", "deterministic", "fallback")


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def test_internal_error_returns_a_request_id(client, resume_upload, sample_jd, monkeypatch):
    import api.main as main

    def boom(*args, **kwargs):
        raise RuntimeError("database on fire at 10.0.0.5")

    monkeypatch.setattr(main.match_aggregator, "match_resume_to_jd", boom)
    response = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    )
    assert response.status_code == 500
    assert "database on fire" not in response.text


def test_handled_error_reports_the_same_id_it_logged(client, resume_upload, sample_jd, monkeypatch, caplog):
    """The in-handler path already threads its request_id through correctly."""
    import api.main as main

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(main.match_aggregator, "match_resume_to_jd", boom)
    with caplog.at_level("ERROR"):
        response = client.post(
            "/api/v1/match",
            files={"resume_pdf": resume_upload},
            data={"job_description": sample_jd},
        )
    detail = response.json()["detail"]
    assert detail["request_id"] in caplog.text


@pytest.mark.asyncio
async def test_global_handler_reuses_the_request_id():
    """The bug the spec records lives in the global handler, not the route.

    The route body threads its own id through, so the only way to observe the
    defect is to hand the handler a request that already carries an id.
    """
    import json as _json
    from types import SimpleNamespace

    import api.main as main

    request = SimpleNamespace(state=SimpleNamespace(request_id="known-id-1234"))
    response = await main.global_exception_handler(request, RuntimeError("boom"))
    assert _json.loads(response.body)["request_id"] == "known-id-1234"


# --------------------------------------------------------------------------
# Phase 6 surface, not built yet
# --------------------------------------------------------------------------


def test_resume_text_is_accepted_instead_of_a_pdf(client, sample_resume_text, sample_jd):
    response = client.post(
        "/api/v1/match",
        data={"resume_text": sample_resume_text, "job_description": sample_jd},
    )
    assert response.status_code == 200


@pytest.mark.xfail(reason="Phase 6: examples endpoint does not exist yet", strict=True)
def test_examples_endpoint_returns_sample_pairs(client):
    response = client.get("/api/v1/examples")
    assert response.status_code == 200
    examples = response.json()["examples"]
    assert len(examples) >= 2
    for example in examples:
        assert example["resume_text"].strip()
        assert example["job_description"].strip()
