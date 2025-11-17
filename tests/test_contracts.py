#!/usr/bin/env python3
"""
Indexing Validation — Data Preparation Contracts
Verifies that inputs passed to index building have the required structure.
We derive embed-ready inputs from DatasetPreparer output using ClipEmbedder.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

import sys

sys.path.insert(0, "src")
from dataset import DatasetPreparer  # noqa: E402
from embedder import ClipEmbedder  # noqa: E402
from rich.console import Console

console = Console()
SUMMARY: Optional[Dict[str, bool]] = None


def _map_id_to_image(dir_path: str, id_key: str) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    base = Path(dir_path)
    if not base.exists():
        return mapping
    for p in sorted(base.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            items = obj if isinstance(obj, list) else [obj]
            for it in items:
                _id = it.get(id_key)
                mapping[_id] = it.get("canonical_image_path")
        except Exception:
            # Ignore malformed
            continue
    return mapping


def _attach_embeddings(
    docs: List[Dict],
    id_field: str,
    image_map: Dict[str, Optional[str]],
    embedder: ClipEmbedder,
) -> List[Dict]:
    # Batch text first
    texts = [d["text"] for d in docs]
    text_embs = embedder.embed_text_batch(texts)

    # Batch images where available (filter out empty paths)
    paths = [image_map.get(d["entity_id"]) for d in docs]
    valid_paths = [(i, p) for i, p in enumerate(paths) if p]
    img_embs_list = (
        embedder.embed_image_batch([p for _, p in valid_paths]) if valid_paths else []
    )
    img_embs_map = (
        {i: emb for (i, _), emb in zip(valid_paths, img_embs_list)}
        if valid_paths
        else {}
    )

    out: List[Dict] = []
    for i, d in enumerate(docs):
        t = text_embs[i]
        i_emb = img_embs_map.get(i) if paths[i] else None
        fused = embedder.fuse(t, i_emb, alpha=0.5)
        out.append(
            {
                "chunk_id": d["chunk_id"],
                id_field: d["entity_id"],
                "text": d["text"],
                "image_path": paths[i],
                "embedding": fused,
                "metadata": d.get("metadata", {}),
            }
        )
    return out


@pytest.fixture(scope="module")
def ctx():
    dp = DatasetPreparer()
    embedder = ClipEmbedder()
    dataset = dp.prepare()
    char_map = _map_id_to_image("data/characters", "character_id")
    loc_map = _map_id_to_image("data/locations", "location_id")

    chars = _attach_embeddings(
        dataset["characters"], "character_id", char_map, embedder
    )
    locs = _attach_embeddings(dataset["locations"], "location_id", loc_map, embedder)

    return {
        "embedder": embedder,
        "characters": chars,
        "locations": locs,
    }


def test_data_contracts_embed_ready(ctx):
    embedder: ClipEmbedder = ctx["embedder"]
    chars: List[Dict] = ctx["characters"]
    locs: List[Dict] = ctx["locations"]

    # Existence
    assert len(chars) > 0
    assert len(locs) > 0

    # Unique chunk_id
    all_ids = [x["chunk_id"] for x in chars] + [x["chunk_id"] for x in locs]
    assert len(all_ids) == len(set(all_ids))

    # Field presence and types
    def check_items(items: List[Dict], id_field: str):
        for it in items:
            assert "chunk_id" in it and isinstance(it["chunk_id"], str)
            assert id_field in it and isinstance(it[id_field], str)
            assert "text" in it and isinstance(it["text"], str) and len(it["text"]) > 0
            assert "embedding" in it and isinstance(it["embedding"], np.ndarray)
            assert it["embedding"].dtype == np.float32
            assert it["embedding"].shape == (embedder.embed_dim,)
            # Norm ~ 1.0
            assert np.allclose(np.linalg.norm(it["embedding"]), 1.0, atol=1e-4)
            assert "metadata" in it and isinstance(it["metadata"], dict)
            # image_path may be None or str
            assert (it.get("image_path") is None) or isinstance(
                it.get("image_path"), str
            )

    check_items(chars, "character_id")
    check_items(locs, "location_id")


def _compute_summary(ctx) -> Dict[str, bool]:
    embedder: ClipEmbedder = ctx["embedder"]
    chars: List[Dict] = ctx["characters"]
    locs: List[Dict] = ctx["locations"]

    ok_nonempty = len(chars) > 0 and len(locs) > 0
    ids = [x["chunk_id"] for x in chars] + [x["chunk_id"] for x in locs]
    ok_unique_ids = len(ids) == len(set(ids))

    def all_ok(items: List[Dict], id_field: str) -> bool:
        try:
            for it in items:
                if not (
                    isinstance(it["chunk_id"], str) and isinstance(it[id_field], str)
                ):
                    return False
                if not (isinstance(it["text"], str) and len(it["text"]) > 0):
                    return False
                e = it["embedding"]
                if not (
                    isinstance(e, np.ndarray)
                    and e.dtype == np.float32
                    and e.shape == (embedder.embed_dim,)
                ):
                    return False
                if not np.allclose(np.linalg.norm(e), 1.0, atol=1e-4):
                    return False
                if not isinstance(it.get("metadata", {}), dict):
                    return False
            return True
        except Exception:
            return False

    ok_chars = all_ok(chars, "character_id")
    ok_locs = all_ok(locs, "location_id")

    ready = ok_nonempty and ok_unique_ids and ok_chars and ok_locs
    return {
        "nonempty": ok_nonempty,
        "unique_ids": ok_unique_ids,
        "chars": ok_chars,
        "locs": ok_locs,
        "ready": ready,
    }


def _print_checklist(printer, s: Dict[str, bool], colored: bool = True):
    def mark(ok: bool) -> str:
        if colored:
            return "[green]✔[/green]" if ok else "[red]✗[/red]"
        return "[✔]" if ok else "[✗]"

    printer("\nIndexing Step 1 — Data Contracts Checklist")
    printer(f"  {mark(s['nonempty'])} Non-empty character/location inputs")
    printer(f"  {mark(s['unique_ids'])} Unique chunk_id across all items")
    printer(
        f"  {mark(s['chars'])} Characters: text + normalized embeddings + metadata + IDs"
    )
    printer(
        f"  {mark(s['locs'])} Locations: text + normalized embeddings + metadata + IDs"
    )
    printer(f"  {mark(s['ready'])} READY for indexing inputs\n")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    yield
    # no-op, just to ensure hook order doesn't hide our terminal summary


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus):
    # Build the same ctx to compute summary (quick and deterministic due to caching)
    dp = DatasetPreparer()
    embedder = ClipEmbedder()
    dataset = dp.prepare()
    char_map = _map_id_to_image("data/characters", "character_id")
    loc_map = _map_id_to_image("data/locations", "location_id")

    chars = _attach_embeddings(
        dataset["characters"], "character_id", char_map, embedder
    )
    locs = _attach_embeddings(dataset["locations"], "location_id", loc_map, embedder)

    s = _compute_summary(
        {
            "embedder": embedder,
            "characters": chars,
            "locations": locs,
        }
    )
    _print_checklist(
        lambda msg: terminalreporter.write_line(msg, yellow=True), s, colored=True
    )
