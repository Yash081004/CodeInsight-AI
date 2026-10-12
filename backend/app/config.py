"""
CodeInsight AI - Backend Configuration

Centralizes all configuration using pydantic-settings.
Values are loaded from environment variables or .env file.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


# Project root is one level above backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""
    
    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    
    # Model
    codebert_model_name: str = "microsoft/codebert-base"
    max_code_length: int = 512
    similarity_threshold: float = 0.8
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Paths (relative to project root)
    data_dir: str = "./data"
    models_dir: str = "./models"
    artifacts_dir: str = "./artifacts"
    
    # Database
    database_url: str = "sqlite:///./codeinsight.db"
    
    # Training
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 5
    seed: int = 42
    
    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir
    
    @property
    def models_path(self) -> Path:
        return PROJECT_ROOT / self.models_dir
    
    @property
    def artifacts_path(self) -> Path:
        return PROJECT_ROOT / self.artifacts_dir
    
    @property
    def classification_data_path(self) -> Path:
        return self.data_path / "classification"
    
    @property
    def similarity_data_path(self) -> Path:
        return self.data_path / "similarity"
    
    @property
    def classifier_model_path(self) -> Path:
        return self.models_path / "codebert_classifier"
    
    @property
    def baselines_model_path(self) -> Path:
        return self.models_path / "baselines"
    
    @property
    def db_path(self) -> Path:
        """Get the SQLite database file path."""
        return PROJECT_ROOT / "codeinsight.db"


# Label constants - single source of truth
ERROR_LABELS = [
    "correct",
    "wrong_initialization",
    "off_by_one_loop",
    "wrong_operator",
    "wrong_condition",
]

LABEL_TO_ID = {label: i for i, label in enumerate(ERROR_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(ERROR_LABELS)}

# User-friendly label names
FRIENDLY_LABELS = {
    "correct": "Correct Code",
    "wrong_initialization": "Wrong Initialization",
    "off_by_one_loop": "Off-by-One Loop",
    "wrong_operator": "Wrong Operator",
    "wrong_condition": "Wrong Condition",
}

# Feedback templates - deterministic, no LLM needed
FEEDBACK_TEMPLATES = {
    "correct": (
        "The model predicts that the submitted program belongs to the correct-code category. "
        "Verify the output against additional test cases before concluding that the program is fully correct."
    ),
    "wrong_initialization": (
        "Review the initial values assigned to variables and accumulators. "
        "Check whether the starting value matches the problem requirements."
    ),
    "off_by_one_loop": (
        "Review loop boundaries and range endpoints. "
        "Check whether the first and last required elements are included exactly as intended."
    ),
    "wrong_operator": (
        "Review arithmetic, comparison, and assignment operators. "
        "Confirm that the operator matches the intended algorithm."
    ),
    "wrong_condition": (
        "Review conditional expressions and Boolean logic. "
        "Check comparison operators, logical operators, and boundary conditions."
    ),
}


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
