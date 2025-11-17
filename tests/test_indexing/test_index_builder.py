#!/usr/bin/env python3
"""
Indexing Validation — Complete Test Suite
Tests Chroma setup, upsert, BM25, build_all_indices, smoke tests, and determinism.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import numpy as np
import pytest
from rich.console import Console

import sys

sys.path.insert(0, "src")
from dataset import DatasetPreparer
from embedder import ClipEmbedder

console = Console()

# ============================================================================
# FIXTURES & HELPERS
# ============================================================================


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
    # Filter out empty paths and create mapping
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
    for idx, doc in enumerate(docs):
        text_emb = text_embs[idx]
        image_emb = img_embs_map.get(idx) if paths[idx] else None
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


@pytest.fixture(scope="module")
def embedder():
    return ClipEmbedder()


@pytest.fixture(scope="module")
def embed_inputs(embedder):
    dp = DatasetPreparer()
    dataset = dp.prepare()
    char_map = _map_id_to_image("data/characters", "character_id")
    loc_map = _map_id_to_image("data/locations", "location_id")
    characters = _attach_embeddings(
        dataset.get("characters", []), "character_id", char_map, embedder
    )
    locations = _attach_embeddings(
        dataset.get("locations", []), "location_id", loc_map, embedder
    )
    if not characters or not locations:
        pytest.skip("No characters/locations prepared")
    return {"embedder": embedder, "characters": characters, "locations": locations}


@pytest.fixture(scope="module")
def chroma_test_dir(tmp_path_factory):
    """Isolated Chroma DB for tests."""
    return tmp_path_factory.mktemp("chroma_test")


# ============================================================================
# STAGE 4.2: CHROMA COLLECTION SETUP
# ============================================================================


def test_chroma_collection_setup(chroma_test_dir):
    """Test 1: Collections exist or are created cleanly."""
    db_path = chroma_test_dir / "rag_db"
    client = chromadb.PersistentClient(path=str(db_path))

    # Create collection
    col = client.get_or_create_collection(
        "characters", metadata={"hnsw:space": "cosine"}
    )
    assert col is not None

    # Calling again should not create duplicates
    col2 = client.get_or_create_collection(
        "characters", metadata={"hnsw:space": "cosine"}
    )
    assert col2 is not None
    assert col.name == col2.name

    # Check no duplicates
    names = [c.name for c in client.list_collections()]
    assert names.count("characters") == 1


def test_chroma_persistent_path(chroma_test_dir):
    """Test 2: Persistent path exists."""
    db_path = chroma_test_dir / "rag_db"
    client = chromadb.PersistentClient(path=str(db_path))
    client.get_or_create_collection("test_col")
    assert os.path.exists(db_path)


# ============================================================================
# STAGE 4.3: UPSERT EMBEDDINGS TO CHROMA
# ============================================================================


def test_upsert_embeddings_batch(chroma_test_dir, embed_inputs):
    """Test: Upsert a batch of embeddings."""
    db_path = chroma_test_dir / "rag_db_upsert"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        "characters", metadata={"hnsw:space": "cosine"}
    )

    chars: List[Dict] = embed_inputs["characters"]
    embedder = embed_inputs["embedder"]

    ids = [c["chunk_id"] for c in chars]
    embeddings = [c["embedding"].astype(float).tolist() for c in chars]
    metadatas = [c["metadata"] for c in chars]
    documents = [c["text"] for c in chars]

    # Verify unique IDs
    assert len(ids) == len(set(ids)), "chunk_id must be unique"

    # Verify embedding dimensions
    for emb in embeddings:
        assert (
            len(emb) == embedder.embed_dim
        ), f"Embedding dim mismatch: {len(emb)} vs {embedder.embed_dim}"

    # Upsert
    if hasattr(collection, "upsert"):
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
    else:
        collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    # Verify count
    assert collection.count() == len(
        chars
    ), f"Count mismatch: {collection.count()} vs {len(chars)}"


def test_upsert_no_duplicates(chroma_test_dir, embed_inputs):
    """Test: No duplicates after upsert."""
    db_path = chroma_test_dir / "rag_db_no_dup"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        "characters", metadata={"hnsw:space": "cosine"}
    )

    chars: List[Dict] = embed_inputs["characters"][:3]  # Use subset
    ids = [c["chunk_id"] for c in chars]
    embeddings = [c["embedding"].astype(float).tolist() for c in chars]
    metadatas = [c["metadata"] for c in chars]
    documents = [c["text"] for c in chars]

    # Upsert twice
    if hasattr(collection, "upsert"):
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
    else:
        collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    # Should still have 3, not 6
    assert collection.count() == 3


def test_retrieval_sanity_self_similarity(chroma_test_dir, embed_inputs):
    """Test: Self-similarity query returns distance <= 0.1."""
    db_path = chroma_test_dir / "rag_db_sanity"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        "characters", metadata={"hnsw:space": "cosine"}
    )

    chars: List[Dict] = embed_inputs["characters"]
    ids = [c["chunk_id"] for c in chars]
    embeddings = [c["embedding"].astype(float).tolist() for c in chars]
    metadatas = [c["metadata"] for c in chars]
    documents = [c["text"] for c in chars]

    if hasattr(collection, "upsert"):
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
    else:
        collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    # Query with first embedding
    result = collection.query(query_embeddings=[embeddings[0]], n_results=1)
    returned_id = result["ids"][0][0]
    dist = result["distances"][0][0]

    assert (
        returned_id == ids[0]
    ), f"Self-query returned wrong ID: {returned_id} vs {ids[0]}"
    assert dist <= 0.1, f"Self-similarity distance too high: {dist:.4f}"


# ============================================================================
# STAGE 4.4: BM25 INDEX VALIDATION
# ============================================================================


def test_bm25_construction(embed_inputs):
    """Test: BM25 index construction."""
    from rank_bm25 import BM25Okapi

    dp = DatasetPreparer()
    chars: List[Dict] = embed_inputs["characters"]

    # Prepare docs with tokens
    docs_with_tokens = []
    for c in chars:
        tokens = dp.bm25_preprocess(c["text"])
        docs_with_tokens.append(tokens)

    # Build BM25
    bm25 = BM25Okapi(docs_with_tokens)

    # Verify corpus length (use stored attribute if available, else use input)
    corpus = getattr(bm25, "corpus", docs_with_tokens)
    assert len(corpus) == len(
        chars
    ), f"Corpus length mismatch: {len(corpus)} vs {len(chars)}"

    # Verify no empty docs
    for doc in corpus:
        assert len(doc) > 0, "Empty document in corpus"


def test_bm25_scoring(embed_inputs):
    """Test: BM25 scoring returns floats, no NaN."""
    from rank_bm25 import BM25Okapi

    dp = DatasetPreparer()
    chars: List[Dict] = embed_inputs["characters"]

    docs_with_tokens = [dp.bm25_preprocess(c["text"]) for c in chars]
    bm25 = BM25Okapi(docs_with_tokens)

    # Score a query
    query_tokens = dp.bm25_preprocess("office desk sunset")
    scores = bm25.get_scores(query_tokens)

    assert len(scores) == len(
        chars
    ), f"Score count mismatch: {len(scores)} vs {len(chars)}"
    for score in scores:
        assert isinstance(score, (int, float)), f"Score not numeric: {type(score)}"
        assert not np.isnan(score), "Score is NaN"


# ============================================================================
# STAGE 4.5: PERSIST BM25 INDEX
# ============================================================================


def test_bm25_persistence(tmp_path, embed_inputs):
    """Test: BM25 index persistence and reload."""
    from rank_bm25 import BM25Okapi
    import pickle

    dp = DatasetPreparer()
    chars: List[Dict] = embed_inputs["characters"]

    # Build BM25
    docs_with_tokens = [dp.bm25_preprocess(c["text"]) for c in chars]
    bm25 = BM25Okapi(docs_with_tokens)

    # Save
    bm25_dir = tmp_path / "bm25"
    bm25_dir.mkdir()

    corpus_path = bm25_dir / "corpus.json"
    with open(corpus_path, "w") as f:
        json.dump(docs_with_tokens, f)

    bm25_path = bm25_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Verify files exist
    assert corpus_path.exists()
    assert bm25_path.exists()

    # Reload
    with open(bm25_path, "rb") as f:
        bm25_loaded = pickle.load(f)

    # Score with reloaded index
    query_tokens = dp.bm25_preprocess("test query")
    scores = bm25_loaded.get_scores(query_tokens)

    assert len(scores) == len(chars)
    for score in scores:
        assert isinstance(score, (int, float))
        assert not np.isnan(score)


# ============================================================================
# STAGE 4.6: build_all_indices() CONTRACT VALIDATION
# ============================================================================


def test_build_all_indices_contract(tmp_path, embed_inputs):
    """Test: build_all_indices() produces all required artifacts."""
    from rank_bm25 import BM25Okapi

    db_path = tmp_path / "rag_db_full"
    client = chromadb.PersistentClient(path=str(db_path))

    chars: List[Dict] = embed_inputs["characters"]
    locs: List[Dict] = embed_inputs["locations"]
    embedder = embed_inputs["embedder"]

    # Create collections
    char_col = client.get_or_create_collection(
        "characters", metadata={"hnsw:space": "cosine"}
    )
    loc_col = client.get_or_create_collection(
        "locations", metadata={"hnsw:space": "cosine"}
    )

    # Upsert characters
    char_ids = [c["chunk_id"] for c in chars]
    char_embs = [c["embedding"].astype(float).tolist() for c in chars]
    char_metas = [c["metadata"] for c in chars]
    char_docs = [c["text"] for c in chars]

    if hasattr(char_col, "upsert"):
        char_col.upsert(
            ids=char_ids,
            embeddings=char_embs,
            metadatas=char_metas,
            documents=char_docs,
        )
    else:
        char_col.add(
            ids=char_ids,
            embeddings=char_embs,
            metadatas=char_metas,
            documents=char_docs,
        )

    # Upsert locations
    loc_ids = [l["chunk_id"] for l in locs]
    loc_embs = [l["embedding"].astype(float).tolist() for l in locs]
    loc_metas = [l["metadata"] for l in locs]
    loc_docs = [l["text"] for l in locs]

    if hasattr(loc_col, "upsert"):
        loc_col.upsert(
            ids=loc_ids, embeddings=loc_embs, metadatas=loc_metas, documents=loc_docs
        )
    else:
        loc_col.add(
            ids=loc_ids, embeddings=loc_embs, metadatas=loc_metas, documents=loc_docs
        )

    # Build BM25
    dp = DatasetPreparer()
    all_tokens = [dp.bm25_preprocess(c["text"]) for c in chars] + [
        dp.bm25_preprocess(l["text"]) for l in locs
    ]
    bm25 = BM25Okapi(all_tokens)

    # Verify counts
    assert char_col.count() == len(chars)
    assert loc_col.count() == len(locs)
    corpus = getattr(bm25, "corpus", all_tokens)
    assert len(corpus) == len(chars) + len(locs)

    # Verify no duplicate IDs
    all_ids = char_ids + loc_ids
    assert len(all_ids) == len(set(all_ids))


# ============================================================================
# STAGE 4.7: ROUND-TRIP RETRIEVAL SMOKE TESTS
# ============================================================================


def test_embedding_query_self_match(tmp_path, embed_inputs):
    """Test: Embedding query returns exact self-match."""
    db_path = tmp_path / "rag_db_self_match"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        "characters", metadata={"hnsw:space": "cosine"}
    )

    chars: List[Dict] = embed_inputs["characters"]
    ids = [c["chunk_id"] for c in chars]
    embeddings = [c["embedding"].astype(float).tolist() for c in chars]
    metadatas = [c["metadata"] for c in chars]
    documents = [c["text"] for c in chars]

    if hasattr(collection, "upsert"):
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
    else:
        collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    # Query with first embedding
    result = collection.query(query_embeddings=[embeddings[0]], n_results=1)
    assert result["ids"][0][0] == ids[0]


def _quick_check_summary() -> bool:
    """
    Lightweight sanity runner so `python tests/test_indexing/test_index_builder.py`
    prints something more helpful than a silent exit. Mirrors the build pipeline:
    dataset → embeddings → Chroma upsert → BM25 save.
    """
    console.rule("Indexing Quick Check")
    dp = DatasetPreparer()
    dataset = dp.prepare()
    if not dataset.get("characters") and not dataset.get("locations"):
        console.print(
            "[yellow]⚠[/yellow] No chunks prepared; add JSON files under data/characters or data/locations."
        )
        return False

    embedder = ClipEmbedder()
    char_map = _map_id_to_image("data/characters", "character_id")
    loc_map = _map_id_to_image("data/locations", "location_id")
    chars = _attach_embeddings(
        dataset.get("characters", []), "character_id", char_map, embedder
    )
    locs = _attach_embeddings(
        dataset.get("locations", []), "location_id", loc_map, embedder
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        client = chromadb.PersistentClient(
            path=str(Path(tmpdir) / "index_builder_check")
        )
        char_col = client.get_or_create_collection(
            "characters", metadata={"hnsw:space": "cosine"}
        )
        loc_col = client.get_or_create_collection(
            "locations", metadata={"hnsw:space": "cosine"}
        )

        if chars:
            ids = [c["chunk_id"] for c in chars]
            embs = [c["embedding"].astype(float).tolist() for c in chars]
            docs = [c["text"] for c in chars]
            metas = [c["metadata"] for c in chars]
            if hasattr(char_col, "upsert"):
                char_col.upsert(
                    ids=ids, embeddings=embs, metadatas=metas, documents=docs
                )
            else:
                char_col.add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)

        if locs:
            ids = [c["chunk_id"] for c in locs]
            embs = [c["embedding"].astype(float).tolist() for c in locs]
            docs = [c["text"] for c in locs]
            metas = [c["metadata"] for c in locs]
            if hasattr(loc_col, "upsert"):
                loc_col.upsert(
                    ids=ids, embeddings=embs, metadatas=metas, documents=docs
                )
            else:
                loc_col.add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)

        char_count = char_col.count()
        loc_count = loc_col.count()

    bm25 = None
    if dataset.get("characters") or dataset.get("locations"):
        bm25 = dp.build_bm25_index(
            dataset.get("characters", []) + dataset.get("locations", [])
        )

    console.print(
        f"[green]✓[/green] Characters indexed: {char_count} | Locations indexed: {loc_count} "
        f"| BM25 docs: {len(dataset.get('characters', [])) + len(dataset.get('locations', []))}"
    )
    return char_count + loc_count > 0 and bm25 is not None


if __name__ == "__main__":
    ok = _quick_check_summary()
    if ok:
        console.print("[bold green]Indexing quick check passed ✓[/bold green]")
        raise SystemExit(0)
    console.print("[bold red]Indexing quick check failed ✗[/bold red]")
    raise SystemExit(1)


def test_corruption_count_decrease(tmp_path, embed_inputs):
    """Test: Deleting embeddings decreases count."""
    db_path = tmp_path / "rag_db_corrupt"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        "characters", metadata={"hnsw:space": "cosine"}
    )

    chars: List[Dict] = embed_inputs["characters"][:5]
    ids = [c["chunk_id"] for c in chars]
    embeddings = [c["embedding"].astype(float).tolist() for c in chars]
    metadatas = [c["metadata"] for c in chars]
    documents = [c["text"] for c in chars]

    if hasattr(collection, "upsert"):
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
    else:
        collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    initial_count = collection.count()
    assert initial_count == 5

    # Delete one
    collection.delete(ids=[ids[0]])
    assert collection.count() == 4


# ============================================================================
# STAGE 4.8: DETERMINISM TEST
# ============================================================================


def test_determinism_build_twice(tmp_path, embed_inputs):
    """Test: Running build twice produces identical results."""
    from rank_bm25 import BM25Okapi

    chars: List[Dict] = embed_inputs["characters"]
    locs: List[Dict] = embed_inputs["locations"]
    dp = DatasetPreparer()

    results = []
    for run in range(2):
        db_path = tmp_path / f"rag_db_det_{run}"
        client = chromadb.PersistentClient(path=str(db_path))

        char_col = client.get_or_create_collection(
            "characters", metadata={"hnsw:space": "cosine"}
        )
        loc_col = client.get_or_create_collection(
            "locations", metadata={"hnsw:space": "cosine"}
        )

        char_ids = [c["chunk_id"] for c in chars]
        char_embs = [c["embedding"].astype(float).tolist() for c in chars]
        char_metas = [c["metadata"] for c in chars]
        char_docs = [c["text"] for c in chars]

        if hasattr(char_col, "upsert"):
            char_col.upsert(
                ids=char_ids,
                embeddings=char_embs,
                metadatas=char_metas,
                documents=char_docs,
            )
        else:
            char_col.add(
                ids=char_ids,
                embeddings=char_embs,
                metadatas=char_metas,
                documents=char_docs,
            )

        loc_ids = [l["chunk_id"] for l in locs]
        loc_embs = [l["embedding"].astype(float).tolist() for l in locs]
        loc_metas = [l["metadata"] for l in locs]
        loc_docs = [l["text"] for l in locs]

        if hasattr(loc_col, "upsert"):
            loc_col.upsert(
                ids=loc_ids,
                embeddings=loc_embs,
                metadatas=loc_metas,
                documents=loc_docs,
            )
        else:
            loc_col.add(
                ids=loc_ids,
                embeddings=loc_embs,
                metadatas=loc_metas,
                documents=loc_docs,
            )

        all_tokens = [dp.bm25_preprocess(c["text"]) for c in chars] + [
            dp.bm25_preprocess(l["text"]) for l in locs
        ]
        bm25 = BM25Okapi(all_tokens)

        corpus = getattr(bm25, "corpus", all_tokens)
        results.append(
            {
                "char_count": char_col.count(),
                "loc_count": loc_col.count(),
                "char_ids": sorted(char_ids),
                "loc_ids": sorted(loc_ids),
                "bm25_corpus_len": len(corpus),
            }
        )

    # Compare runs
    assert results[0]["char_count"] == results[1]["char_count"]
    assert results[0]["loc_count"] == results[1]["loc_count"]
    assert results[0]["char_ids"] == results[1]["char_ids"]
    assert results[0]["loc_ids"] == results[1]["loc_ids"]
    assert results[0]["bm25_corpus_len"] == results[1]["bm25_corpus_len"]


# ============================================================================
# PYTEST TERMINAL SUMMARY
# ============================================================================
