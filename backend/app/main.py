"""
CodeInsight AI - FastAPI Application

Main application entry point. Registers all API routes and middleware.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.api import health, similarity, classification, datasets, evaluation, history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="CodeInsight AI",
        description=(
            "Deep Learning-Based Semantic Code Similarity Detection "
            "and Logic Error Classification for Student Python Programs"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": "An unexpected error occurred."},
        )
    
    # Register routes
    app.include_router(health.router, prefix="/api")
    app.include_router(similarity.router, prefix="/api")
    app.include_router(classification.router, prefix="/api")
    app.include_router(datasets.router, prefix="/api")
    app.include_router(evaluation.router, prefix="/api")
    app.include_router(history.router, prefix="/api")
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("CodeInsight AI backend starting up...")
        logger.info(f"CORS origins: {settings.cors_origins_list}")
        logger.info(f"Data dir: {settings.data_path}")
        logger.info(f"Models dir: {settings.models_path}")
        
        # Ensure directories exist
        settings.data_path.mkdir(parents=True, exist_ok=True)
        settings.classification_data_path.mkdir(parents=True, exist_ok=True)
        settings.similarity_data_path.mkdir(parents=True, exist_ok=True)
        settings.models_path.mkdir(parents=True, exist_ok=True)
        settings.baselines_model_path.mkdir(parents=True, exist_ok=True)
        settings.artifacts_path.mkdir(parents=True, exist_ok=True)
        (settings.artifacts_path / "metrics").mkdir(parents=True, exist_ok=True)
        (settings.artifacts_path / "figures").mkdir(parents=True, exist_ok=True)
        
        logger.info("CodeInsight AI backend ready.")
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
