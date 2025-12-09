"""
Utility helpers for embedder package.
"""

from typing import Any, Dict
import numpy as np
import hashlib


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = vec.astype("float32")
    n = np.linalg.norm(v)
    if n == 0 or np.isnan(n):
        return v
    return v / (n + 1e-12)


def l2_normalize_batch(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / (norms + 1e-12)


def stable_entity_cache_key(entity: Dict[str, Any]) -> str:
    """Produce a deterministic cache key for an entity.

    Use canonical id fields (character_id / location_id), entity_version,
    and a short hash of the appearance/description to be safe.
    """
    eid = (
        entity.get("character_id")
        or entity.get("location_id")
        or entity.get("id")
        or "unknown"
    )
    version = entity.get("entity_version") or entity.get("version") or 0
    text = (entity.get("appearance") or entity.get("description") or "").strip()
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{eid}_v{int(version)}_{h}"
