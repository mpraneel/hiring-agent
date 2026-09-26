"""FastAPI application for the hiring agent."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from typing import Any, Callable, Optional, TypeVar

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core import config
from core.extraction.llm_extractor import LLMExtractor
from core.scoring.aggregate import MatchAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = FastAPI(
    title="Hiring Agent API",
    description="Resume-to-job description matcher with auditable scoring",
    version="1.0.0",
)

# Development only. In production the built frontend is served from this same
# origin, so no cross-origin request is involved.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_provider = config.get_provider()
llm_model = config.get_model(llm_provider)
extraction_mode = config.get_extraction_mode()

extractor = LLMExtractor()
match_aggregator = MatchAggregator(llm_provider=llm_provider, llm_model=llm_model)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Give every request an id that error handlers can report back.

    Without this the global handler had to mint a fresh id, so the id a client
    was told to quote never appeared in the logs for the failing request.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Informational endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/info")
@app.get("/")
async def root() -> dict:
    """API metadata and effective configuration."""
    return {
        "message": "Hiring Agent API",
        "version": "1.0.0",
        "status": "running",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_enabled": config.llm_enabled(),
        "extraction_mode": extraction_mode,
    }


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_available": config.llm_enabled(),
        "extraction_mode": extraction_mode,
    }


# ---------------------------------------------------------------------------
# Shared request handling
# ---------------------------------------------------------------------------


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def _bad_request(detail: str, request_id: str) -> HTTPException:
    return HTTPException(
        status_code=400, detail={"error": detail, "request_id": request_id}
    )


async def _read_resume_text(
    resume_pdf: Optional[UploadFile],
    resume_text: Optional[str],
    request_id: str,
) -> str:
    """Resolve the resume to plain text from exactly one of the two inputs."""
    has_pdf = resume_pdf is not None and bool(resume_pdf.filename)
    has_text = bool(resume_text and resume_text.strip())

    if has_pdf and has_text:
        raise _bad_request(
            "Provide either resume_pdf or resume_text, not both", request_id
        )
    if not has_pdf and not has_text:
        raise _bad_request("Either resume_pdf or resume_text is required", request_id)

    if has_text:
        assert resume_text is not None
        return resume_text

    assert resume_pdf is not None
    if not (resume_pdf.filename or "").lower().endswith(".pdf"):
        raise _bad_request("File must be a PDF", request_id)

    content = await resume_pdf.read()
    # resume_pdf.size is None for some clients, so the length of what we read is
    # the only reliable check. Reading first also bounds memory the same way.
    if len(content) > MAX_UPLOAD_BYTES:
        raise _bad_request("File size must be less than 5MB", request_id)
    if not content:
        raise _bad_request("Uploaded file is empty", request_id)

    temp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        return extractor.resume_parser.extract_pdf_text(temp_path)
    except ValueError as exc:
        raise _bad_request(f"Could not read the PDF: {exc}", request_id) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


async def _parse_inputs(
    request: Request,
    resume_pdf: Optional[UploadFile],
    resume_text: Optional[str],
    job_description: str,
):
    """Validate inputs and run extraction, shared by both POST endpoints."""
    request_id = _request_id(request)

    if not job_description or not job_description.strip():
        raise _bad_request("Job description cannot be empty", request_id)

    text = await _read_resume_text(resume_pdf, resume_text, request_id)
    if not text.strip():
        raise _bad_request("No text could be extracted from the resume", request_id)

    logger.info("request %s: extracting with mode %s", request_id, extraction_mode)
    parsed_resume = extractor.parse_resume(text, mode=extraction_mode)
    parsed_jd = extractor.parse_jd(job_description, mode=extraction_mode)
    return request_id, parsed_resume, parsed_jd, text


T = TypeVar("T")


def _handle_errors(request_id: str, work: Callable[[], T]) -> T:
    """Run work, converting anything unexpected into a clean 500."""
    try:
        return work()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("request %s failed: %s", request_id, exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "request_id": request_id,
                "message": "An unexpected error occurred while processing the request",
            },
        ) from exc


# ---------------------------------------------------------------------------
# Matching endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/match")
async def match_resume_to_jd(
    request: Request,
    job_description: str = Form(...),
    resume_pdf: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
) -> JSONResponse:
    """Score a resume against a job description.

    Exactly one of resume_pdf or resume_text is required.
    """
    request_id, parsed_resume, parsed_jd, _ = await _parse_inputs(
        request, resume_pdf, resume_text, job_description
    )

    result = _handle_errors(
        request_id, lambda: match_aggregator.match_resume_to_jd(parsed_resume, parsed_jd)
    )

    return JSONResponse(
        status_code=200,
        content={
            "match_score": result.match_score,
            "baseline_score": result.baseline_score,
            "matched_skills": result.matched_skills,
            "missing_skills": result.missing_skills,
            "nice_matches": result.nice_matches,
            "llm_rationale": result.llm_rationale,
            "suggestions": result.suggestions,
            "extraction_path": result.extraction_path,
            "llm_status": result.llm_status,
            "request_id": request_id,
        },
    )


@app.post("/api/v1/analyze")
async def detailed_analysis(
    request: Request,
    job_description: str = Form(...),
    resume_pdf: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
) -> JSONResponse:
    """Full breakdown of a resume-to-job match."""
    request_id, parsed_resume, parsed_jd, _ = await _parse_inputs(
        request, resume_pdf, resume_text, job_description
    )

    analysis = _handle_errors(
        request_id,
        lambda: match_aggregator.get_detailed_analysis(parsed_resume, parsed_jd),
    )
    analysis["request_id"] = request_id
    return JSONResponse(status_code=200, content=analysis)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Last resort handler. Reuses the failing request's id."""
    request_id = _request_id(request) if hasattr(request, "state") else str(uuid.uuid4())
    logger.exception("unhandled exception in request %s: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "message": "An unexpected error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
