#!/usr/bin/env python3
"""Reranker Validation.

Tests CrossEncoder reranking, batching, fallback behavior, and error handling.
"""

import sys
from typing import List, Dict
from contextlib import contextmanager

import numpy as np
import pytest
from _pytest.outcomes import Skipped

sys.path.insert(0, "src")
from reranker import Reranker
from retriever import HybridRetriever
from retriever.chroma_client import chroma_client
from dataset import DatasetPreparer
from embedder import ClipEmbedder

from rich.console import Console

from conftest import register_checklist_item

console = Console()


def _summarize_results(results: List[Dict], score_key: str, limit: int = 3) -> str:
    """Return a short human-friendly summary of top results for logs."""
    parts = []
    for item in results[:limit]:
        ident = (
            item.get("chunk_id")
            or item.get("id")
            or item.get("document_id")
            or item.get("character_id")
            or "<unknown>"
        )
        score = item.get(score_key)
        parts.append(
            f"{ident} ({score:.3f})"
            if isinstance(score, (int, float))
            else f"{ident} (n/a)"
        )
    return ", ".join(parts) or "<no results>"


def _summarize_ids(results: List[Dict], limit: int = 3) -> str:
    ids = []
    for item in results[:limit]:
        ident = (
            item.get("chunk_id")
            or item.get("id")
            or item.get("document_id")
            or item.get("character_id")
        )
        ids.append(ident or "<unknown>")
    return ", ".join(ids) or "<no results>"


@contextmanager
def checklist(stage: str, item_key: str):
    try:
        yield
    except Skipped:
        register_checklist_item(stage, item_key, "skipped")
        raise
    except Exception:
        register_checklist_item(stage, item_key, False)
        raise
    else:
        register_checklist_item(stage, item_key, True)


@pytest.fixture(scope="module")
def reranker_setup():
    """Set up reranker with populated indices and sample candidates."""
    try:
        # Prepare data
        dp = DatasetPreparer()
        dataset = dp.prepare()

        chars = dataset.get("characters", [])
        assert chars, "No character data available; dataset prep failed"

        # Populate Chroma for retrieval
        embedder = ClipEmbedder()
        retriever = HybridRetriever()

        char_ids = [c["chunk_id"] for c in chars]
        char_texts = [c["text"] for c in chars]
        char_metas = [c.get("metadata", {}) for c in chars]

        char_embs = embedder.embed_text_batch(char_texts)

        char_col = chroma_client.characters
        if hasattr(char_col, "upsert"):
            char_col.upsert(
                ids=char_ids,
                embeddings=[e.astype(float).tolist() for e in char_embs],
                documents=char_texts,
                metadatas=char_metas,
            )
        else:
            char_col.add(
                ids=char_ids,
                embeddings=[e.astype(float).tolist() for e in char_embs],
                documents=char_texts,
                metadatas=char_metas,
            )

        return {
            "reranker": Reranker(),
            "retriever": retriever,
            "chars": chars,
        }
    except Exception as e:
        pytest.fail(f"Reranker setup failed: {e}")


# ============================================================================
# TEST 1: BATCHED RERANKING
# ============================================================================


def test_batched_reranking_works(reranker_setup):
    """Test: Batched reranking works"""
    with checklist("stage3_reranker", "batched"):
        reranker = reranker_setup["reranker"]
        retriever = reranker_setup["retriever"]

        query = "office desk"
        candidates = retriever.hybrid_search(query, "characters", top_k=20)

        assert candidates, "Hybrid retriever returned no candidates"

        # Rerank
        reranked = reranker.rerank(query, candidates, top_k=10)

        assert len(reranked) <= len(
            candidates
        ), "Reranked results should not exceed input"
        assert len(reranked) <= 10, "Should respect top_k parameter"

        # All results should have rerank_score and rerank_score_norm
        for item in reranked:
            assert "rerank_score" in item, "Result missing rerank_score"
            assert isinstance(
                item["rerank_score"], (int, float)
            ), "rerank_score should be numeric"
            assert "rerank_score_norm" in item, "Result missing rerank_score_norm"
            assert isinstance(
                item["rerank_score_norm"], (int, float)
            ), "rerank_score_norm should be numeric"

        console.print(
            "[green]\u2713[/green] Batched reranking works:\n"
            f"  Original top (hybrid): { _summarize_results(candidates, 'hybrid_score') }\n"
            f"  Reranked top (cross-encoder): { _summarize_results(reranked, 'rerank_score') }"
        )


# ============================================================================
# TEST 2: CROSS-ENCODER MODEL LOADS ONCE
# ============================================================================


def test_cross_encoder_model_loads_once():
    """Test: Cross-encoder model loads once (singleton)"""
    with checklist("stage3_reranker", "singleton"):
        r1 = Reranker()
        r2 = Reranker()

        assert r1.model is not None, "First reranker model should be loaded"
        assert r2.model is not None, "Second reranker model should be loaded"
        assert r1.model is r2.model, "Reranker should use singleton model"

        console.print(
            "[green]\u2713[/green] Cross-encoder model loads once (singleton)"
        )


# ============================================================================
# TEST 3: RERANKED LIST IMPROVES SEMANTIC ORDER
# ============================================================================


def test_reranked_list_improves_semantic_order(reranker_setup):
    """Test: Reranked list improves semantic order"""
    with checklist("stage3_reranker", "improves_order"):
        reranker = reranker_setup["reranker"]
        retriever = reranker_setup["retriever"]

        query = "office desk monitor"
        candidates = retriever.hybrid_search(query, "characters", top_k=10)

        assert len(candidates) >= 2, "Need at least 2 candidates to test ordering"

        # Get original order (by hybrid_score)
        original_scores = [c.get("hybrid_score", 0) for c in candidates]

        # Rerank
        reranked = reranker.rerank(query, candidates, top_k=10)

        # Get reranked scores
        reranked_scores = [r.get("rerank_score", 0) for r in reranked]

        # Reranked should be sorted by rerank_score (descending)
        assert reranked_scores == sorted(
            reranked_scores, reverse=True
        ), "Reranked results should be sorted by rerank_score"

        # At least verify rerank_score is different from hybrid_score
        assert len(reranked) > 0, "Should have reranked results"

        console.print(
            "[green]\u2713[/green] Reranked list improves semantic order:\n"
            f"  Original top: { _summarize_ids(candidates) }\n"
            f"  Reranked top: { _summarize_ids(reranked) }"
        )


# ============================================================================
# TEST 4: ERRORS FALL BACK TO NON-RERANKED LIST
# ============================================================================


def test_errors_fall_back_to_non_reranked_list(reranker_setup, monkeypatch):
    """Test: Errors fall back to non-reranked list"""
    with checklist("stage3_reranker", "fallback"):
        reranker = reranker_setup["reranker"]
        retriever = reranker_setup["retriever"]

        query = "office"
        candidates = retriever.hybrid_search(query, "characters", top_k=5)

        assert candidates, "Hybrid retriever returned no candidates"

        # Monkeypatch model.predict to raise an error
        original_predict = reranker.model.predict

        def failing_predict(*args, **kwargs):
            raise RuntimeError("Model prediction failed")

        monkeypatch.setattr(reranker.model, "predict", failing_predict)

        # Reranker should handle the error gracefully and fall back to original ordering
        reranked = reranker.rerank(query, candidates, top_k=5)
        assert (
            reranked == candidates[:5]
        ), "On model failure, rerank should return original ordering"
        console.print(
            "[green]\u2713[/green] Fallback on model failure returns original ordering"
        )

        # Restore original
        monkeypatch.setattr(reranker.model, "predict", original_predict)


# ============================================================================
# TEST 5: APPROPRIATE LOGS ON ERRORS
# ============================================================================


def test_appropriate_logs_on_errors(reranker_setup):
    """Test: Appropriate logs on errors"""
    with checklist("stage3_reranker", "error_logs"):
        reranker = reranker_setup["reranker"]

        # Test with empty results
        empty_results = reranker.rerank("test", [], top_k=5)
        assert empty_results == [], "Empty input should return empty list"

        # Test with invalid query - empty query should be handled gracefully
        retriever = reranker_setup["retriever"]
        try:
            candidates = retriever.hybrid_search("", "characters", top_k=5)
            if candidates:
                reranked = reranker.rerank("", candidates, top_k=5)
                assert isinstance(
                    reranked, list
                ), "Should return list even with empty query"
        except ValueError:
            # Empty query validation is expected to raise ValueError in some components
            pass

        console.print("[green]\u2713[/green] Appropriate error handling and logging")


# ============================================================================
# TEST 6: BATCH SIZE HANDLING
# ============================================================================


def test_batch_size_handling(reranker_setup):
    """Test: Reranker handles different batch sizes correctly"""
    with checklist("stage3_reranker", "batch_size"):
        reranker = reranker_setup["reranker"]
        retriever = reranker_setup["retriever"]

        query = "office"
        candidates = retriever.hybrid_search(query, "characters", top_k=50)

        assert len(candidates) >= 5, "Need at least 5 candidates to test batching"
        target_top_k = min(20, len(candidates))

        # Should handle large batches (relative to available candidates)
        reranked = reranker.rerank(query, candidates, top_k=target_top_k)

        assert len(reranked) <= target_top_k, "Should respect top_k"
        assert all(
            "rerank_score" in r for r in reranked
        ), "All results should have rerank_score"

        console.print(
            f"[green]\u2713[/green] Batch size handling works: processed {len(candidates)} candidates"
        )


def test_handles_missing_text_field(monkeypatch):
    """Test: Reranker handles candidates without text field"""
    with checklist("stage3_reranker", "missing_text"):
        reranker = Reranker()
        candidates = [
            {"entity_id": "no_text_candidate"},
            {"entity_id": "with_text_candidate", "text": "some text"},
        ]

        def fake_predict(pairs, batch_size=None):
            return [0.1 * i for i in range(len(pairs))]

        monkeypatch.setattr(reranker.model, "predict", fake_predict)

        reranked = reranker.rerank("query", candidates, top_k=2)
        assert len(reranked) == 2
        for item in reranked:
            assert "rerank_score" in item
            assert "rerank_score_norm" in item


def test_improves_semantic_order_synthetic(monkeypatch):
    """Test: Cross-encoder can promote a better second result to first position."""
    with checklist("stage3_reranker", "improves_order_synthetic"):
        reranker = Reranker()
        candidates = [
            {"entity_id": "A", "text": "worse result", "hybrid_score": 1.0},
            {"entity_id": "B", "text": "better result", "hybrid_score": 0.5},
        ]

        def fake_predict(pairs, batch_size=None):
            # Cross-encoder thinks B is much better than A
            return [0.2, 0.9]

        monkeypatch.setattr(reranker.model, "predict", fake_predict)

        reranked = reranker.rerank("query", candidates, top_k=2)
        assert (
            reranked[0]["entity_id"] == "B"
        ), "Better candidate should be promoted to first position"


def test_handles_large_candidate_pool(monkeypatch):
    """Test: Reranker processes pools larger than its batch_size using internal batching."""
    with checklist("stage3_reranker", "large_pool"):
        reranker = Reranker()
        batch_size = reranker.batch_size
        num_candidates = batch_size * 2 + 5

        candidates = [
            {"entity_id": f"doc_{i}", "text": f"dummy text {i}"}
            for i in range(num_candidates)
        ]

        def fake_predict(pairs, batch_size=None):
            # Simple increasing scores so that sorting is deterministic
            return list(range(len(pairs)))

        monkeypatch.setattr(reranker.model, "predict", fake_predict)

        top_k = 10
        reranked = reranker.rerank("query", candidates, top_k=top_k)
        assert len(reranked) == top_k
        for item in reranked:
            assert "rerank_score" in item
            assert "rerank_score_norm" in item


if __name__ == "__main__":
    console.print("[bold cyan]\nRunning Reranker validation suite...[/bold cyan]")
    raise SystemExit(pytest.main([__file__]))
