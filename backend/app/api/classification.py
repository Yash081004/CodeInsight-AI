"""
CodeInsight AI - Classification API

Endpoint for logic error classification.
"""

from fastapi import APIRouter, HTTPException
from backend.app.schemas import ClassificationRequest, ClassificationResult
from backend.app.services.classification_service import analyze_code
from backend.app.services.history_service import get_history_service
from backend.app.config import get_settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classification", tags=["classification"])


@router.post("/analyze", response_model=ClassificationResult)
async def analyze(request: ClassificationRequest):
    """
    Classify a Python code snippet into one of five error categories.
    
    Uses fine-tuned CodeBERT when available, falls back to TF-IDF baseline.
    Returns the predicted label, probabilities, and deterministic feedback.
    """
    try:
        settings = get_settings()
        result = analyze_code(
            code=request.code,
            max_code_length=settings.max_code_length,
            problem_statement=request.problem_statement,
            function_context=request.function_context,
            test_cases=[tc.model_dump() for tc in request.test_cases] if request.test_cases else None,
            run_tests=request.run_tests,
        )
        
        # Record in history
        try:
            history = get_history_service(settings.db_path)
            history.add_entry(
                analysis_type="classification",
                code=request.code,
                result_summary=f"{result['friendly_label']} ({result['confidence']:.2%})",
                model_used=result["model_type"],
            )
        except Exception as e:
            logger.warning(f"Failed to record history: {e}")
        
        return ClassificationResult(**result)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
