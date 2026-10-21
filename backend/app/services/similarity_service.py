"""
CodeInsight AI - Similarity Service

Orchestrates similarity analysis using CodeBERT embeddings and TF-IDF baseline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

from backend.app.config import get_settings
from backend.app.models.codebert_encoder import get_encoder
from backend.app.services.code_validator import validate_python_code
from backend.app.services.similarity_policy import (
    DEFAULT_TF_CONFIRM,
    DEFAULT_TF_LOW,
    decide_from_scores,
)

logger = logging.getLogger(__name__)

# Module-level TF-IDF components
_tfidf_vectorizer = None
_tfidf_loaded = False
_hybrid_params_cache: Optional[Tuple[float, float]] = None


def _load_hybrid_params() -> Tuple[float, float]:
    """Load val-tuned mid-band params from artifacts when present."""
    global _hybrid_params_cache
    if _hybrid_params_cache is not None:
        return _hybrid_params_cache
    settings = get_settings()
    path = settings.artifacts_path / "metrics" / "similarity_hybrid_policy.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _hybrid_params_cache = (
                float(data.get("tf_low", DEFAULT_TF_LOW)),
                float(data.get("tf_confirm", DEFAULT_TF_CONFIRM)),
            )
            return _hybrid_params_cache
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Could not load hybrid policy params: %s", e)
    _hybrid_params_cache = (DEFAULT_TF_LOW, DEFAULT_TF_CONFIRM)
    return _hybrid_params_cache


def _load_tfidf_vectorizer() -> bool:
    """Load the saved TF-IDF vectorizer if available."""
    global _tfidf_vectorizer, _tfidf_loaded
    
    if _tfidf_loaded:
        return _tfidf_vectorizer is not None
    
    settings = get_settings()
    vec_path = settings.baselines_model_path / "similarity_tfidf_vectorizer.joblib"
    
    if vec_path.exists():
        try:
            _tfidf_vectorizer = joblib.load(vec_path)
            _tfidf_loaded = True
            logger.info("TF-IDF similarity vectorizer loaded.")
            return True
        except Exception as e:
            logger.error(f"Failed to load TF-IDF vectorizer: {e}")
    
    _tfidf_loaded = True  # Prevent repeated load attempts
    return False


def compute_tfidf_similarity(code_a: str, code_b: str) -> Optional[float]:
    """
    Compute TF-IDF cosine similarity between two code snippets.
    
    If a pre-fitted vectorizer is available, uses it.
    Otherwise, fits a temporary vectorizer on just the two snippets
    (which has known limitations - documented in API response).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    try:
        if _load_tfidf_vectorizer() and _tfidf_vectorizer is not None:
            vectors = _tfidf_vectorizer.transform([code_a, code_b])
        else:
            # Fallback: fit on just the two snippets
            # This is limited because the vocabulary is only from these two samples
            vec = TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                lowercase=True,
                max_features=5000,
            )
            vectors = vec.fit_transform([code_a, code_b])
        
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(sim)
    except Exception as e:
        logger.error(f"TF-IDF similarity error: {e}")
        return None


def analyze_similarity(
    code_a: str,
    code_b: str,
    threshold: Optional[float] = None,
    max_code_length: int = 512,
) -> Dict[str, Any]:
    """
    Full similarity analysis pipeline.
    
    Returns results from both CodeBERT and TF-IDF methods.
    """
    settings = get_settings()
    
    if threshold is None:
        threshold = settings.similarity_threshold
    
    # Validate both code snippets
    a_valid, a_error = validate_python_code(code_a)
    b_valid, b_error = validate_python_code(code_b)
    
    result = {
        "code_a_valid": a_valid,
        "code_b_valid": b_valid,
        "code_a_error": a_error if not a_valid else None,
        "code_b_error": b_error if not b_valid else None,
        "threshold": threshold,
        "codebert_available": False,
        "codebert_score": None,
        "tfidf_available": True,
        "tfidf_score": None,
        "is_similar": None,
        "decision_model": "none",
        "explanation": "",
        "model_info": "",
    }
    
    # TF-IDF baseline (works even if code has syntax errors, since it's text-based)
    tfidf_score = compute_tfidf_similarity(code_a, code_b)
    if tfidf_score is not None:
        result["tfidf_score"] = round(tfidf_score, 4)
        result["tfidf_available"] = True
    
    # CodeBERT similarity
    encoder = get_encoder(
        model_name=settings.codebert_model_name,
        device=settings.device,
    )
    
    if encoder.is_loaded or encoder.load():
        result["codebert_available"] = True
        codebert_score = encoder.compute_similarity(code_a, code_b, max_length=max_code_length)
        if codebert_score is not None:
            result["codebert_score"] = round(codebert_score, 4)
    
    tf_low, tf_confirm = _load_hybrid_params()
    decision, policy_meta = decide_from_scores(
        codebert_score=result["codebert_score"],
        tfidf_score=result["tfidf_score"],
        threshold=threshold,
        tf_low=tf_low,
        tf_confirm=tf_confirm,
    )

    primary_score = (
        result["codebert_score"] if result["codebert_score"] is not None else result["tfidf_score"]
    )
    result["decision_score"] = primary_score
    result["decision_policy"] = policy_meta.get("decision_policy")
    result["review_recommended"] = decision.review_recommended
    result["score_gap"] = policy_meta.get("score_gap")

    if primary_score is not None:
        result["is_similar"] = decision.is_similar
        result["decision_model"] = decision.decision_model
        result["explanation"] = (
            f"{decision.reason} "
            f"CodeBERT={result['codebert_score']}, TF-IDF={result['tfidf_score']}, "
            f"user threshold={threshold:.2f}. "
            "Cosine similarity does not prove identical runtime behavior on all inputs."
        )
    else:
        result["is_similar"] = None
        result["decision_model"] = "none"
        result["explanation"] = (
            "Unable to compute similarity. Please check that the models are available "
            "and the code snippets are valid."
        )
    
    # Model info
    parts = []
    if result["codebert_available"]:
        parts.append(f"CodeBERT ({settings.codebert_model_name})")
    parts.append("TF-IDF Baseline")
    result["model_info"] = " + ".join(parts)
    
    return result
