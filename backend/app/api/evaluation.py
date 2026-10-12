"""
CodeInsight AI - Evaluation API

Endpoints for model evaluation results.
"""

import json
from fastapi import APIRouter
from pathlib import Path
from backend.app.schemas import EvaluationSummary, EvaluationMetrics
from backend.app.config import get_settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _load_json(path: Path) -> dict:
    """Load a JSON file safely."""
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return {}


@router.get("/summary", response_model=EvaluationSummary)
async def evaluation_summary():
    """Get evaluation results from saved metrics files."""
    settings = get_settings()
    metrics_dir = settings.artifacts_path / "metrics"
    
    result = EvaluationSummary(
        classification_available=False,
        similarity_available=False,
    )
    
    # Load CodeBERT classifier metrics
    codebert_metrics_path = metrics_dir / "codebert_classification_metrics.json"
    codebert_data = _load_json(codebert_metrics_path)
    if codebert_data:
        try:
            result.codebert_metrics = EvaluationMetrics(**codebert_data)
            result.classification_available = True
        except Exception as e:
            logger.warning(f"Invalid CodeBERT metrics format: {e}")
    
    # Load baseline classifier metrics
    baseline_metrics_path = metrics_dir / "baseline_classification_metrics.json"
    baseline_data = _load_json(baseline_metrics_path)
    if baseline_data:
        try:
            result.baseline_metrics = EvaluationMetrics(**baseline_data)
            result.classification_available = True
        except Exception as e:
            logger.warning(f"Invalid baseline metrics format: {e}")
    
    # Load similarity metrics
    similarity_metrics_path = metrics_dir / "similarity_metrics.json"
    sim_data = _load_json(similarity_metrics_path)
    if sim_data:
        result.similarity_metrics = sim_data
        result.similarity_available = True
    
    # Load confusion matrix data
    cm_path = metrics_dir / "confusion_matrix.json"
    cm_data = _load_json(cm_path)
    if cm_data:
        result.confusion_matrix = cm_data
    
    return result


@router.get("/confusion-matrix")
async def confusion_matrix():
    """Get confusion matrix data for visualization."""
    settings = get_settings()
    
    # Try CodeBERT confusion matrix first
    cm_path = settings.artifacts_path / "metrics" / "confusion_matrix.json"
    data = _load_json(cm_path)
    
    if not data:
        # Try baseline confusion matrix
        cm_path = settings.artifacts_path / "metrics" / "baseline_confusion_matrix.json"
        data = _load_json(cm_path)
    
    if not data:
        return {"available": False, "message": "No confusion matrix data available. Run model evaluation first."}
    
    return {"available": True, "data": data}
