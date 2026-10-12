"""
CodeInsight AI - Similarity API

Endpoint for code similarity analysis.
"""

from fastapi import APIRouter, HTTPException
from backend.app.schemas import SimilarityRequest, SimilarityResult
from backend.app.services.similarity_service import analyze_similarity
from backend.app.services.history_service import get_history_service
from backend.app.config import get_settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/similarity", tags=["similarity"])


@router.post("/analyze", response_model=SimilarityResult)
async def analyze(request: SimilarityRequest):
    """
    Analyze semantic similarity between two Python code snippets.
    
    Uses CodeBERT embeddings (when available) and TF-IDF baseline.
    The similarity score is NOT a calibrated probability of semantic equivalence.
    """
    try:
        settings = get_settings()
        result = analyze_similarity(
            code_a=request.code_a,
            code_b=request.code_b,
            threshold=request.threshold,
            max_code_length=settings.max_code_length,
        )
        
        # Record in history
        try:
            history = get_history_service(settings.db_path)
            score_str = f"CodeBERT: {result['codebert_score']}" if result['codebert_score'] else "N/A"
            tfidf_str = f"TF-IDF: {result['tfidf_score']}" if result['tfidf_score'] else "N/A"
            history.add_entry(
                analysis_type="similarity",
                code=request.code_a + "\n---\n" + request.code_b,
                result_summary=f"{score_str}, {tfidf_str}",
                model_used=result["decision_model"],
            )
        except Exception as e:
            logger.warning(f"Failed to record history: {e}")
        
        return SimilarityResult(**result)
    except Exception as e:
        logger.error(f"Similarity analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
