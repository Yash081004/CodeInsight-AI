"""
CodeInsight AI - History API

Endpoints for analysis history.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from backend.app.schemas import HistoryResponse, HistoryEntry
from backend.app.services.history_service import get_history_service
from backend.app.config import get_settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
async def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    analysis_type: Optional[str] = None,
):
    """Get paginated analysis history."""
    settings = get_settings()
    history = get_history_service(settings.db_path)
    result = history.get_entries(
        page=page,
        page_size=page_size,
        analysis_type=analysis_type,
    )
    
    entries = [HistoryEntry(**entry) for entry in result["entries"]]
    
    return HistoryResponse(
        entries=entries,
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
    )


@router.post("/clear")
async def clear_history(confirm: bool = Query(...)):
    """Clear all analysis history. Requires confirm=true."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Clearing history requires confirm=true")
    
    settings = get_settings()
    history = get_history_service(settings.db_path)
    success = history.clear()
    
    if success:
        return {"status": "cleared", "message": "Analysis history has been cleared."}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear history.")
