"""
CodeInsight AI - Health API

Provides system health and component availability information.
"""

import torch
from fastapi import APIRouter
from backend.app.config import get_settings
from backend.app.schemas import HealthResponse
from backend.app.models.codebert_encoder import get_encoder
from backend.app.models.error_classifier import get_classifier

router = APIRouter(tags=["health"])


@router.get("/ready")
async def readiness_check():
    """
    Readiness: data and at least one inference path (classifier or baseline) available.
    Does not load heavy models.
    """
    settings = get_settings()
    dataset_ok = (settings.classification_data_path / "dataset.csv").exists()
    baseline_ok = (
        (settings.baselines_model_path / "classification_logreg.joblib").exists()
        and (settings.baselines_model_path / "classification_tfidf_vectorizer.joblib").exists()
    )
    classifier = get_classifier(settings.classifier_model_path, settings.device)
    classifier_ok = classifier.is_trained
    ready = dataset_ok and (baseline_ok or classifier_ok)
    return {
        "ready": ready,
        "dataset_available": dataset_ok,
        "classifier_checkpoint": classifier_ok,
        "baseline_models": baseline_ok,
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns system status including model availability.
    Does NOT trigger model loading - only checks current state.
    """
    settings = get_settings()
    
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check CodeBERT encoder status (without loading)
    encoder = get_encoder(settings.codebert_model_name, settings.device)
    codebert_available = encoder.is_loaded or encoder.is_available
    
    # Check classifier status (without loading)
    classifier = get_classifier(settings.classifier_model_path, settings.device)
    classifier_available = classifier.is_trained
    
    # Check baseline availability
    baseline_available = (
        (settings.baselines_model_path / "classification_logreg.joblib").exists()
        and (settings.baselines_model_path / "classification_tfidf_vectorizer.joblib").exists()
    )
    
    # Check dataset availability
    dataset_available = (
        (settings.classification_data_path / "dataset.csv").exists()
    )
    
    # Determine statuses
    if classifier_available:
        classifier_status = "trained"
    elif baseline_available:
        classifier_status = "baseline_only"
    else:
        classifier_status = "not_trained"
    
    if codebert_available:
        similarity_status = "codebert_ready"
    elif baseline_available:
        similarity_status = "baseline_only"
    else:
        similarity_status = "unavailable"
    
    return HealthResponse(
        status="ok",
        device=device,
        codebert_available=codebert_available,
        classifier_available=classifier_available,
        baseline_available=baseline_available,
        dataset_available=dataset_available,
        classifier_status=classifier_status,
        similarity_status=similarity_status,
    )
