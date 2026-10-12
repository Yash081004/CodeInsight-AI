"""
CodeInsight AI - History Service

SQLite-based analysis history storage.
Stores only metadata and code hashes by default, not full source code.
"""

import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class HistoryService:
    """Local SQLite-based analysis history."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_table()
    
    def _ensure_table(self):
        """Create the history table if it doesn't exist."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    result_summary TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'completed'
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to create history table: {e}")
    
    @staticmethod
    def _hash_code(code: str) -> str:
        """Create a SHA-256 hash of the code for privacy."""
        return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    
    def add_entry(
        self,
        analysis_type: str,
        code: str,
        result_summary: str,
        model_used: str,
        status: str = "completed",
    ) -> Optional[int]:
        """Add a new history entry. Returns the entry ID."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                """INSERT INTO analysis_history 
                   (timestamp, analysis_type, code_hash, result_summary, model_used, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    analysis_type,
                    self._hash_code(code),
                    result_summary,
                    model_used,
                    status,
                ),
            )
            entry_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return entry_id
        except Exception as e:
            logger.error(f"Failed to add history entry: {e}")
            return None
    
    def get_entries(
        self,
        page: int = 1,
        page_size: int = 20,
        analysis_type: Optional[str] = None,
    ) -> Dict:
        """Get paginated history entries."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            
            # Count query
            count_sql = "SELECT COUNT(*) FROM analysis_history"
            params = []
            if analysis_type:
                count_sql += " WHERE analysis_type = ?"
                params.append(analysis_type)
            
            total = conn.execute(count_sql, params).fetchone()[0]
            
            # Data query
            data_sql = "SELECT * FROM analysis_history"
            if analysis_type:
                data_sql += " WHERE analysis_type = ?"
            data_sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
            
            offset = (page - 1) * page_size
            params.extend([page_size, offset])
            
            rows = conn.execute(data_sql, params).fetchall()
            conn.close()
            
            entries = [dict(row) for row in rows]
            
            return {
                "entries": entries,
                "total": total,
                "page": page,
                "page_size": page_size,
            }
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return {"entries": [], "total": 0, "page": page, "page_size": page_size}
    
    def clear(self) -> bool:
        """Clear all history entries."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("DELETE FROM analysis_history")
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False


# Singleton
_history_instance: Optional[HistoryService] = None


def get_history_service(db_path: Path) -> HistoryService:
    global _history_instance
    if _history_instance is None:
        _history_instance = HistoryService(db_path)
    return _history_instance
