"""
CodeInsight AI - Classification Service

Orchestrates error classification using CodeBERT classifier and baseline.
"""

import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import logging

from backend.app.models.error_classifier import get_classifier
from backend.app.services.code_validator import validate_python_code
from backend.app.services.feedback_service import get_feedback, get_friendly_label
from backend.app.config import (
    get_settings, ERROR_LABELS, LABEL_TO_ID, ID_TO_LABEL, FRIENDLY_LABELS
)

logger = logging.getLogger(__name__)

# Baseline model singletons
_baseline_classifier = None
_baseline_vectorizer = None
_baseline_loaded = False


def _load_baseline() -> bool:
    """Load TF-IDF + Logistic Regression baseline if available."""
    global _baseline_classifier, _baseline_vectorizer, _baseline_loaded
    
    if _baseline_loaded:
        return _baseline_classifier is not None
    
    settings = get_settings()
    clf_path = settings.baselines_model_path / "classification_logreg.joblib"
    vec_path = settings.baselines_model_path / "classification_tfidf_vectorizer.joblib"
    
    if clf_path.exists() and vec_path.exists():
        try:
            _baseline_classifier = joblib.load(clf_path)
            _baseline_vectorizer = joblib.load(vec_path)
            _baseline_loaded = True
            logger.info("Baseline classifier loaded.")
            return True
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
    
    _baseline_loaded = True
    return False


def _predict_baseline(code: str) -> Optional[Dict]:
    """Predict using the TF-IDF + Logistic Regression baseline."""
    if not _load_baseline() or _baseline_classifier is None or _baseline_vectorizer is None:
        return None
    
    try:
        features = _baseline_vectorizer.transform([code])
        predicted_id = int(_baseline_classifier.predict(features)[0])
        proba = _baseline_classifier.predict_proba(features)[0]
        
        predicted_label = ID_TO_LABEL.get(predicted_id, "correct")
        probabilities = {
            ID_TO_LABEL[i]: float(proba[i]) 
            for i in range(len(proba))
        }
        
        return {
            "predicted_label": predicted_label,
            "probabilities": probabilities,
            "confidence": float(max(proba)),
        }
    except Exception as e:
        logger.error(f"Baseline prediction error: {e}")
        return None


def analyze_code(code: str, max_code_length: int = 512) -> Dict[str, Any]:
    """
    Full classification analysis pipeline.
    
    Tries CodeBERT fine-tuned classifier first, falls back to baseline.
    """
    settings = get_settings()
    
    # Validate code
    is_valid, error_msg = validate_python_code(code)
    
    if not is_valid:
        return {
            "predicted_label": "unknown",
            "friendly_label": "Unknown",
            "probabilities": {},
            "confidence": 0.0,
            "feedback": f"Cannot analyze: {error_msg}",
            "model_type": "none",
            "model_info": "Code validation failed",
            "code_valid": False,
            "code_error": error_msg,
        }
    
    # Try CodeBERT fine-tuned classifier first
    classifier = get_classifier(
        model_path=settings.classifier_model_path,
        device=settings.device,
    )
    
    result = None
    model_type = "none"
    model_info = "No model available"
    
    if classifier.is_loaded or classifier.load():
        result = classifier.predict(code, max_length=max_code_length)
        if result:
            model_type = "codebert_finetuned"
            model_info = f"Fine-tuned CodeBERT ({settings.codebert_model_name})"
    
    # Fall back to baseline
    if result is None:
        result = _predict_baseline(code)
        if result:
            model_type = "baseline"
            model_info = "TF-IDF + Logistic Regression Baseline"
    
    # If no model is available
    if result is None:
        return {
            "predicted_label": "unknown",
            "friendly_label": "Unknown",
            "probabilities": {},
            "confidence": 0.0,
            "feedback": (
                "No trained classifier is available. Train the baseline or "
                "fine-tune CodeBERT to enable classification. "
                "See the training documentation for instructions."
            ),
            "model_type": "unavailable",
            "model_info": "No trained model available",
            "code_valid": True,
            "code_error": None,
        }
    
    label = result["predicted_label"]
    
    return {
        "predicted_label": label,
        "friendly_label": get_friendly_label(label),
        "probabilities": result["probabilities"],
        "confidence": result["confidence"],
        "feedback": get_feedback(label),
        "model_type": model_type,
        "model_info": model_info,
        "code_valid": True,
        "code_error": None,
    }
