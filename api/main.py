from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import uuid
import logging
from typing import Optional

from core import config
from core.parsers.resume_parser import ResumeParser
from core.parsers.jd_parser import JDParser
from core.scoring.aggregate import MatchAggregator
from core.schemas.models import MatchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hiring Agent API",
    description="Resume-to-Job Description Matcher API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
resume_parser = ResumeParser()
jd_parser = JDParser()

# Resolve LLM settings from the environment
llm_provider = config.get_provider()
llm_model = config.get_model(llm_provider)

match_aggregator = MatchAggregator(llm_provider=llm_provider, llm_model=llm_model)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Hiring Agent API",
        "version": "1.0.0",
        "status": "running",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_enabled": config.llm_enabled()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_available": config.llm_enabled()
    }


@app.post("/api/v1/match")
async def match_resume_to_jd(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Match a resume PDF to a job description.
    
    Args:
        resume_pdf: PDF file containing the resume
        job_description: Text description of the job
        
    Returns:
        JSON response with match results
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing match request {request_id}")
    
    try:
        # Validate inputs
        if not resume_pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        if not job_description.strip():
            raise HTTPException(status_code=400, detail="Job description cannot be empty")
        
        # Check file size (max 5MB)
        if resume_pdf.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size must be less than 5MB")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await resume_pdf.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Parse resume
            logger.info(f"Parsing resume for request {request_id}")
            parsed_resume = resume_parser.parse_pdf(temp_file_path)
            
            # Parse job description
            logger.info(f"Parsing job description for request {request_id}")
            parsed_jd = jd_parser.parse_jd(job_description)
            
            # Perform matching
            logger.info(f"Performing matching for request {request_id}")
            match_result = match_aggregator.match_resume_to_jd(parsed_resume, parsed_jd)
            
            # Prepare response
            response_data = {
                "match_score": match_result.match_score,
                "baseline_score": match_result.baseline_score,
                "matched_skills": match_result.matched_skills,
                "missing_skills": match_result.missing_skills,
                "nice_matches": match_result.nice_matches,
                "llm_rationale": match_result.llm_rationale,
                "suggestions": match_result.suggestions,
                "request_id": request_id
            }
            
            logger.info(f"Successfully completed request {request_id}")
            return JSONResponse(content=response_data, status_code=200)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "request_id": request_id,
                "message": "An unexpected error occurred while processing the request"
            }
        )


@app.post("/api/v1/analyze")
async def detailed_analysis(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Get detailed analysis of resume-to-job description match.
    
    Args:
        resume_pdf: PDF file containing the resume
        job_description: Text description of the job
        
    Returns:
        JSON response with detailed analysis
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing analysis request {request_id}")
    
    try:
        # Validate inputs
        if not resume_pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        if not job_description.strip():
            raise HTTPException(status_code=400, detail="Job description cannot be empty")
        
        # Check file size (max 5MB)
        if resume_pdf.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size must be less than 5MB")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await resume_pdf.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Parse resume and JD
            parsed_resume = resume_parser.parse_pdf(temp_file_path)
            parsed_jd = jd_parser.parse_jd(job_description)
            
            # Get detailed analysis
            analysis = match_aggregator.get_detailed_analysis(parsed_resume, parsed_jd)
            analysis["request_id"] = request_id
            
            logger.info(f"Successfully completed analysis request {request_id}")
            return JSONResponse(content=analysis, status_code=200)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing analysis request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "request_id": request_id,
                "message": "An unexpected error occurred while processing the request"
            }
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    request_id = str(uuid.uuid4())
    logger.error(f"Unhandled exception in request {request_id}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
