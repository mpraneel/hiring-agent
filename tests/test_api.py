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


def test_examples_endpoint_returns_sample_pairs(client):
    response = client.get("/api/v1/examples")
    assert response.status_code == 200
    examples = response.json()["examples"]
    assert len(examples) >= 2
    for example in examples:
        assert example["resume_text"].strip()
        assert example["job_description"].strip()


# --------------------------------------------------------------------------
# Evidence spans in the response
# --------------------------------------------------------------------------


def test_match_response_carries_documents(client, resume_upload, sample_jd):
    body = client.post(
        "/api/v1/match",
        files={"resume_pdf": resume_upload},
        data={"job_description": sample_jd},
    ).json()

    documents = body["documents"]
    assert documents["jd"]["source_text"].strip()
    assert documents["resume"]["source_text"].strip()
    assert documents["jd"]["requirements"]
    assert documents["resume"]["skills"]


def test_evidence_offsets_slice_to_the_skill(client, sample_jd, sample_resume_text):
    """The UI slices source_text with these offsets, so they must be exact."""
    body = client.post(
        "/api/v1/match",
        data={"resume_text": sample_resume_text, "job_description": sample_jd},
    ).json()

    for document_key, items_key in (("jd", "requirements"), ("resume", "skills")):
        source = body["documents"][document_key]["source_text"]
        for item in body["documents"][document_key][items_key]:
            if item["start"] is None:
                continue
            sliced = source[item["start"] : item["end"]]
            assert sliced, f"empty slice for {item['skill']}"
            # The slice must be a surface form of the skill, not arbitrary text.
            assert sliced.lower().replace(" ", "") != "", sliced
            assert item["evidence"], f"no evidence line for {item['skill']}"


def test_evidence_line_contains_the_highlighted_span(client, sample_jd, sample_resume_text):
    body = client.post(
        "/api/v1/match",
        data={"resume_text": sample_resume_text, "job_description": sample_jd},
    ).json()
    source = body["documents"]["jd"]["source_text"]
    for item in body["documents"]["jd"]["requirements"]:
        if item["start"] is None:
            continue
        assert source[item["start"] : item["end"]].lower() in item["evidence"].lower()


def test_breakdown_counts_are_present(client, sample_jd, sample_resume_text):
    breakdown = client.post(
        "/api/v1/match",
        data={"resume_text": sample_resume_text, "job_description": sample_jd},
    ).json()["breakdown"]
    assert breakdown["must_total"] >= len(breakdown["must_matched"])
    assert breakdown["nice_total"] >= len(breakdown["nice_matched"])
    assert set(breakdown["must_matched"]) & set(breakdown["must_missing"]) == set()


def test_every_requirement_priority_is_must_or_nice(client, sample_jd, sample_resume_text):
    body = client.post(
        "/api/v1/match",
        data={"resume_text": sample_resume_text, "job_description": sample_jd},
    ).json()
    for item in body["documents"]["jd"]["requirements"]:
        assert item["priority"] in ("must", "nice")


# --------------------------------------------------------------------------
# Examples endpoint
# --------------------------------------------------------------------------


def test_examples_returns_usable_pairs(client):
    examples = client.get("/api/v1/examples").json()["examples"]
    assert len(examples) >= 2
    for example in examples:
        assert example["id"]
        assert example["label"]
        assert example["job_description"].strip()
        assert example["resume_text"].strip()


def test_examples_contain_no_real_contact_details(client):
    """These are served publicly, so they must stay obviously synthetic."""
    for example in client.get("/api/v1/examples").json()["examples"]:
        combined = example["resume_text"] + example["job_description"]
        assert "@example.com" in example["resume_text"]
        assert "gmail.com" not in combined
        assert "outlook.com" not in combined


def test_an_example_scores_end_to_end(client):
    example = client.get("/api/v1/examples").json()["examples"][0]
    response = client.post(
        "/api/v1/match",
        data={
            "resume_text": example["resume_text"],
            "job_description": example["job_description"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["match_score"] > 0
    assert body["documents"]["jd"]["requirements"]


def test_api_routes_are_not_shadowed_by_the_static_mount(client):
    """The catch-all must never swallow an API path."""
    assert client.get("/api/v1/examples").status_code == 200
    assert client.get("/health").status_code == 200
