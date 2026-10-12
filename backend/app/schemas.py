"""
CodeInsight AI - Pydantic Schemas

Request and response models for all API endpoints.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    codebert_available: bool
    classifier_available: bool
    baseline_available: bool
    dataset_available: bool
    classifier_status: str  # "not_trained", "trained", "demo_mode"
    similarity_status: str  # "codebert_ready", "baseline_only", "unavailable"


# ─── Similarity ──────────────────────────────────────────────────────────────

class SimilarityRequest(BaseModel):
    code_a: str = Field(..., min_length=1, max_length=10000)
    code_b: str = Field(..., min_length=1, max_length=10000)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @field_validator("code_a", "code_b")
    @classmethod
    def code_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Code cannot be empty or whitespace only")
        return v


class SimilarityResult(BaseModel):
    codebert_score: Optional[float] = None
    codebert_available: bool
    tfidf_score: Optional[float] = None
    tfidf_available: bool
    threshold: float
    is_similar: Optional[bool] = None
    decision_model: str  # which model made the decision
    explanation: str
    model_info: str
    code_a_valid: bool
    code_b_valid: bool
    code_a_error: Optional[str] = None
    code_b_error: Optional[str] = None


# ─── Classification ─────────────────────────────────────────────────────────

class ClassificationRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)
    
    @field_validator("code")
    @classmethod
    def code_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Code cannot be empty or whitespace only")
        return v


class ClassificationResult(BaseModel):
    predicted_label: str
    friendly_label: str
    probabilities: Dict[str, float]
    confidence: float
    feedback: str
    model_type: str  # "codebert_finetuned", "baseline", "demo"
    model_info: str
    code_valid: bool
    code_error: Optional[str] = None
    warning: str = (
        "Confidence represents the model's relative prediction scores. "
        "It is not a guarantee that the predicted category is correct."
    )


# ─── Dataset ────────────────────────────────────────────────────────────────

class DatasetSummary(BaseModel):
    classification_available: bool
    similarity_available: bool
    total_classification_samples: int = 0
    total_similarity_pairs: int = 0
    class_distribution: Dict[str, int] = {}
    problem_types: List[str] = []
    split_counts: Dict[str, int] = {}
    similarity_split_counts: Dict[str, int] = {}


class DatasetSample(BaseModel):
    sample_id: str
    problem_type: str
    code: str
    error_label: str
    friendly_label: str
    mutation_type: Optional[str] = None
    mutation_description: Optional[str] = None
    split: str
    source_template_id: Optional[str] = None


class DatasetSamplesResponse(BaseModel):
    samples: List[DatasetSample]
    total: int
    page: int
    page_size: int
    total_pages: int


# ─── Evaluation ──────────────────────────────────────────────────────────────

class EvaluationMetrics(BaseModel):
    model_type: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_metrics: Dict[str, Dict[str, float]]
    sample_count: int
    evaluated_at: Optional[str] = None


class EvaluationSummary(BaseModel):
    classification_available: bool
    similarity_available: bool
    codebert_metrics: Optional[EvaluationMetrics] = None
    baseline_metrics: Optional[EvaluationMetrics] = None
    similarity_metrics: Optional[Dict] = None
    confusion_matrix: Optional[Dict] = None


# ─── History ─────────────────────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    id: int
    timestamp: str
    analysis_type: str  # "similarity" or "classification"
    code_hash: str
    result_summary: str
    model_used: str
    status: str


class HistoryResponse(BaseModel):
    entries: List[HistoryEntry]
    total: int
    page: int
    page_size: int


# ─── General ─────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class GenerateDatasetRequest(BaseModel):
    confirm: bool = Field(
        ..., description="Must be true to confirm dataset generation"
    )
    force: bool = Field(
        False, description="If true, overwrite existing dataset"
    )


class GenerateDatasetResponse(BaseModel):
    status: str
    total_samples: int
    class_distribution: Dict[str, int]
    message: str


class TrainBaselineRequest(BaseModel):
    confirm: bool = Field(..., description="Must be true to start training")
    model_type: str = Field("classification", description="classification or similarity")


class TrainBaselineResponse(BaseModel):
    status: str
    model_type: str
    metrics: Optional[Dict] = None
    message: str
