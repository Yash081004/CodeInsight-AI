"""
CodeInsight AI - Datasets API

Endpoints for dataset exploration and generation.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from backend.app.schemas import (
    DatasetSummary, DatasetSample, DatasetSamplesResponse,
    GenerateDatasetRequest, GenerateDatasetResponse,
)
from backend.app.config import get_settings, FRIENDLY_LABELS
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["datasets"])


def _load_classification_df() -> Optional[pd.DataFrame]:
    """Load classification dataset CSV."""
    settings = get_settings()
    path = settings.classification_data_path / "dataset.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None


def _load_similarity_df() -> Optional[pd.DataFrame]:
    """Load similarity dataset CSV."""
    settings = get_settings()
    path = settings.similarity_data_path / "pairs.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to load similarity dataset: {e}")
        return None


@router.get("/summary", response_model=DatasetSummary)
async def dataset_summary():
    """Get dataset summary with actual counts from files."""
    clf_df = _load_classification_df()
    sim_df = _load_similarity_df()
    
    result = DatasetSummary(
        classification_available=clf_df is not None,
        similarity_available=sim_df is not None,
    )
    
    if clf_df is not None:
        result.total_classification_samples = len(clf_df)
        result.class_distribution = clf_df["error_label"].value_counts().to_dict()
        if "problem_type" in clf_df.columns:
            result.problem_types = sorted(clf_df["problem_type"].unique().tolist())
        if "split" in clf_df.columns:
            result.split_counts = clf_df["split"].value_counts().to_dict()
    
    if sim_df is not None:
        result.total_similarity_pairs = len(sim_df)
        if "split" in sim_df.columns:
            result.similarity_split_counts = sim_df["split"].value_counts().to_dict()
    
    return result


@router.get("/samples", response_model=DatasetSamplesResponse)
async def dataset_samples(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    label: Optional[str] = None,
    problem_type: Optional[str] = None,
    split: Optional[str] = None,
    search: Optional[str] = None,
):
    """Get paginated, filterable dataset samples."""
    df = _load_classification_df()
    
    if df is None:
        return DatasetSamplesResponse(
            samples=[], total=0, page=page, page_size=page_size, total_pages=0
        )
    
    # Apply filters
    if label:
        df = df[df["error_label"] == label]
    if problem_type:
        df = df[df["problem_type"] == problem_type]
    if split and "split" in df.columns:
        df = df[df["split"] == split]
    if search:
        df = df[df["code"].str.contains(search, case=False, na=False)]
    
    total = len(df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]
    
    samples = []
    for _, row in page_df.iterrows():
        samples.append(DatasetSample(
            sample_id=str(row.get("sample_id", "")),
            problem_type=str(row.get("problem_type", "")),
            code=str(row.get("code", "")),
            error_label=str(row.get("error_label", "")),
            friendly_label=FRIENDLY_LABELS.get(str(row.get("error_label", "")), ""),
            mutation_type=str(row.get("mutation_type", "")) if pd.notna(row.get("mutation_type")) else None,
            mutation_description=str(row.get("mutation_description", "")) if pd.notna(row.get("mutation_description")) else None,
            split=str(row.get("split", "")),
            source_template_id=str(row.get("source_template_id", "")) if pd.notna(row.get("source_template_id")) else None,
        ))
    
    return DatasetSamplesResponse(
        samples=samples,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.post("/generate", response_model=GenerateDatasetResponse)
async def generate_dataset(request: GenerateDatasetRequest):
    """
    Trigger dataset generation.
    Requires explicit confirmation.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Dataset generation requires confirm=true"
        )
    
    settings = get_settings()
    
    # Check if dataset exists and force is not set
    existing = settings.classification_data_path / "dataset.csv"
    if existing.exists() and not request.force:
        raise HTTPException(
            status_code=409,
            detail="Dataset already exists. Set force=true to regenerate."
        )
    
    try:
        # Import and run the dataset generator
        import sys
        sys.path.insert(0, str(settings.data_path.parent))
        from training.generate_dataset import generate_full_dataset
        
        result = generate_full_dataset(
            output_dir=settings.classification_data_path,
            similarity_dir=settings.similarity_data_path,
        )
        
        return GenerateDatasetResponse(
            status="completed",
            total_samples=result["total_samples"],
            class_distribution=result["class_distribution"],
            message=f"Generated {result['total_samples']} samples successfully.",
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Dataset generation module not found. Run 'python -m training.generate_dataset' manually."
        )
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Dataset generation failed: {str(e)}"
        )
