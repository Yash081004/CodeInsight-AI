"""
CodeInsight AI - Similarity Service

Orchestrates similarity analysis using CodeBERT embeddings and TF-IDF baseline.
"""

import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import logging

from backend.app.models.codebert_encoder import get_encoder
from backend.app.services.code_validator import validate_python_code
from backend.app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level TF-IDF components
_tfidf_vectorizer = None
_tfidf_loaded = False


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
    
    # Decision logic
    primary_score = result["codebert_score"] if result["codebert_score"] is not None else result["tfidf_score"]
    
    if primary_score is not None:
        result["is_similar"] = primary_score >= threshold
        result["decision_model"] = "codebert" if result["codebert_score"] is not None else "tfidf"
        
        # Generate cautious explanation
        if result["is_similar"]:
            result["explanation"] = (
                f"The code representations show high similarity (score: {primary_score:.4f}, "
                f"threshold: {threshold}). This may indicate shared functionality, but the result "
                "alone does not establish plagiarism or prove that both programs are semantically equivalent."
            )
        else:
            result["explanation"] = (
                f"The code representations show lower similarity (score: {primary_score:.4f}, "
                f"threshold: {threshold}). The programs may implement different logic, but structural "
                "differences (variable names, formatting) can also affect this score."
            )
    else:
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
