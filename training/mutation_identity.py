"""
Stable mutation IDs linking queue entries, generator variants, and optional CSV rows.

IDs are deterministic from code hash (not from label alone).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MUTATION_ID_VERSION = "v1"


def normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", (code or "").strip())


def code_hash_sha256(code: str) -> str:
    return hashlib.sha256(normalize_code(code).encode()).hexdigest()


def stable_mutation_id(
    *,
    template_id: str,
    error_label: str,
    mutation_type: str,
    code_hash: str,
) -> str:
    """Unambiguous ID: version + template + taxonomy + code hash prefix."""
    h = code_hash.lower()[:16]
    safe_tid = re.sub(r"[^a-zA-Z0-9_]", "_", template_id)
    safe_mut = re.sub(r"[^a-zA-Z0-9_]", "_", mutation_type or "unknown")
    safe_lab = re.sub(r"[^a-zA-Z0-9_]", "_", error_label or "unknown")
    return f"{MUTATION_ID_VERSION}.{safe_tid}.{safe_lab}.{safe_mut}.{h}"


def stable_mutation_id_from_parts(parts: Dict[str, Any]) -> str:
    return stable_mutation_id(
        template_id=str(parts["template_id"]),
        error_label=str(parts["error_label"]),
        mutation_type=str(parts.get("mutation_type") or ""),
        code_hash=str(parts["code_hash_sha256"]),
    )


def lookup_by_id(registry: Dict[str, Any], mutation_id: str) -> Optional[Dict[str, Any]]:
    for rec in registry.get("records") or []:
        if rec.get("mutation_id") == mutation_id:
            return rec
    return None


def registry_checksum(registry: Dict[str, Any]) -> str:
    payload = json.dumps(registry.get("records") or [], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
