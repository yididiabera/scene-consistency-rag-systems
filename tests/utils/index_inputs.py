"""Helpers to build embed-ready inputs for indexing tests."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, "src")
from dataset import DatasetPreparer
from embedder import ClipEmbedder


def _map_id_to_image(dir_path: str, id_key: str) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    base = Path(dir_path)
    if not base.exists():
        return mapping
    for p in sorted(base.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            _id = item.get(id_key)
            if _id:
                mapping[_id] = item.get("canonical_image_path")
    return mapping


def _attach_embeddings(
    docs: List[Dict],
    id_field: str,
    image_map: Dict[str, Optional[str]],
    embedder: ClipEmbedder,
) -> List[Dict]:
    if not docs:
        return []

    texts = [d["text"] for d in docs]
    text_embs = embedder.embed_text_batch(texts)

    paths = [image_map.get(d["entity_id"]) for d in docs]

    # Only embed valid image paths; fall back to text-only fusion when missing
    valid_image_paths = [p for p in paths if p]
    img_embs = None
    if valid_image_paths:
        img_batch = embedder.embed_image_batch(valid_image_paths)
        # Map from document index to corresponding image embedding
        img_embs = {}
        img_idx = 0
        for idx, p in enumerate(paths):
            if p:
                img_embs[idx] = img_batch[img_idx]
                img_idx += 1

    out: List[Dict] = []
    for idx, doc in enumerate(docs):
        text_emb = text_embs[idx]
        image_emb = img_embs[idx] if img_embs and paths[idx] else None
        fused = embedder.fuse(text_emb, image_emb, alpha=0.5)
        out.append(
            {
                "chunk_id": doc["chunk_id"],
                id_field: doc["entity_id"],
                "text": doc["text"],
                "image_path": paths[idx],
                "embedding": fused.astype(np.float32),
                "metadata": doc.get("metadata", {}),
            }
        )
    return out


def build_embed_ready_inputs(
    characters_dir: str = "data/characters",
    locations_dir: str = "data/locations",
    relationships_dir: Optional[str] = None,
    embedder: Optional[ClipEmbedder] = None,
) -> Dict[str, object]:
    dp = DatasetPreparer()
    dataset = dp.prepare(characters_dir, locations_dir, relationships_dir)
    embedder = embedder or ClipEmbedder()

    char_map = _map_id_to_image(characters_dir, "character_id")
    loc_map = _map_id_to_image(locations_dir, "location_id")

    characters = _attach_embeddings(
        dataset.get("characters", []), "character_id", char_map, embedder
    )
    locations = _attach_embeddings(
        dataset.get("locations", []), "location_id", loc_map, embedder
    )

    return {
        "embedder": embedder,
        "characters": characters,
        "locations": locations,
    }


__all__ = [
    "build_embed_ready_inputs",
]
