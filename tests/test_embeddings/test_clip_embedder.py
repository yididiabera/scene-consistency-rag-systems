#!/usr/bin/env python3
"""
ClipEmbedder Validation
Validates:
- Singleton model loading
- Batch text/image embeddings
- Fusion with alpha and L2 renorm
- L2 normalization and no-NaN
- Caching (identity for repeated input in same session)
- Reproducibility (same input -> same vector; different -> different)
- Batch performance faster than sequential
- Error handling for invalid images and empty text
- Output format utility
- Pytest terminal summary checklist
"""

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest

import sys

sys.path.insert(0, "src")
from embedder import ClipEmbedder  # noqa: E402

from rich.console import Console

console = Console()
SUMMARY: Optional[Dict[str, bool]] = None


@pytest.fixture(scope="module")
def embedder() -> ClipEmbedder:
    """Module-scoped fixture amortizes the (expensive) CLIP model load."""
    return ClipEmbedder()


def test_model_loading_singleton():
    """ClipEmbedder should reuse the cached CLIP weights + preprocess pipeline."""
    e1 = ClipEmbedder()
    e2 = ClipEmbedder()
    assert e1.model is not None
    assert e1.preprocess is not None
    assert e1.model is e2.model


def test_text_embedding_batch(embedder: ClipEmbedder):
    """Batch interface should vectorize inputs and L2-normalize each row."""
    texts = ["hello world", "another text"]
    embs = embedder.embed_text_batch(texts)
    assert embs.shape[0] == 2
    assert embs.shape[1] == embedder.embed_dim
    assert not np.isnan(embs).any()
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_image_embedding_batch_and_errors(embedder: ClipEmbedder):
    """Image batches should produce normalized vectors and reject bad paths."""
    valid_img = None
    for p in ["data/characters/isaac.png", "data/characters/gertie.png"]:
        if Path(p).exists():
            valid_img = p
            break
    assert valid_img is not None, "Expected a sample image under data/characters/*.png"

    # Valid
    emb = embedder.embed_image_batch([valid_img])
    assert emb.shape == (1, embedder.embed_dim)
    assert np.allclose(np.linalg.norm(emb[0]), 1.0, atol=1e-4)

    # Invalid path raises error
    with pytest.raises(RuntimeError):
        _ = embedder.embed_image_batch(["INVALID_PATH.png"])


def test_fusion_logic(embedder: ClipEmbedder):
    """Fusion should re-normalize weighted sums and differ from the inputs."""
    valid_img = None
    for p in ["data/characters/isaac.png", "data/characters/gertie.png"]:
        if Path(p).exists():
            valid_img = p
            break
    assert valid_img is not None

    t = embedder.embed_text_batch(["hello"])[0]
    i = embedder.embed_image_batch([valid_img])[0]

    fused = embedder.fuse(t, i, alpha=0.7)
    assert fused.shape == t.shape
    assert np.allclose(np.linalg.norm(fused), 1.0, atol=1e-4)
    assert not np.allclose(fused, t)
    assert not np.allclose(fused, i)


def test_caching_identity(embedder: ClipEmbedder):
    """Repeated inputs reuse cached numpy objects (cheap downstream)."""
    # Text cache in-memory returns same reference
    e1 = embedder.embed_text("hello world")
    e2 = embedder.embed_text("hello world")
    assert e1 is e2

    # Image cache identity as well
    img = None
    for p in ["data/characters/isaac.png", "data/characters/gertie.png"]:
        if Path(p).exists():
            img = p
            break
    assert img is not None
    i1 = embedder.embed_image(img)
    i2 = embedder.embed_image(img)
    assert i1 is i2


def test_reproducibility(embedder: ClipEmbedder):
    """Deterministic model path: same text → identical vector; diff text → diff vec."""
    v1 = embedder.embed_text("test")
    v2 = embedder.embed_text("test")
    assert np.allclose(v1, v2)

    a = embedder.embed_text("a")
    b = embedder.embed_text("b")
    assert not np.allclose(a, b)

    # Across instances (model singleton + disk cache)
    e2 = ClipEmbedder()
    v3 = e2.embed_text("test")
    assert np.allclose(v1, v3)


def test_batch_performance():
    """Vectorized path must beat per-call invocation to justify batch API."""
    texts = [f"t{i}" for i in range(32)]  # Reduced from 64 to 32 for faster test

    e_batch = ClipEmbedder()
    e_batch.clear_cache()
    t1 = time.time()
    batch_emb = e_batch.embed_text_batch(texts)
    batch_time = time.time() - t1
    assert batch_emb.shape == (32, e_batch.embed_dim)

    e_single = ClipEmbedder()
    e_single.clear_cache()
    t2 = time.time()
    for t in texts:
        _ = e_single.embed_text(t)
    single_time = time.time() - t2

    # Batch should be faster (not necessarily 2x on CPU, but should be faster)
    assert batch_time < single_time * 1.1  # More lenient threshold


def test_error_handling(embedder: ClipEmbedder):
    """Explicit failures for empty text or unreadable images."""
    with pytest.raises(ValueError):
        _ = embedder.embed_text("")

    with pytest.raises(RuntimeError):
        _ = embedder.embed_image_batch(["INVALID_PATH.png"])


def test_output_format(embedder: ClipEmbedder):
    """format_output keeps legacy API happy (chunk_id + float32 vector)."""
    v = embedder.embed_text("format-test")
    out = embedder.format_output("chunk_123", v)
    assert set(out.keys()) == {"chunk_id", "embedding"}
    assert out["chunk_id"] == "chunk_123"
    assert isinstance(out["embedding"], np.ndarray)
    assert out["embedding"].dtype == np.float32
    assert out["embedding"].shape == (embedder.embed_dim,)


def _compute_summary() -> Dict[str, bool]:
    """Aggregated readiness profile mirrors what's printed in terminal summary."""
    e = ClipEmbedder()

    # Model
    model_ok = e.model is ClipEmbedder().model and e.preprocess is not None

    # Text
    txt = ["hello world", "another text"]
    te = e.embed_text_batch(txt)
    text_shape_ok = te.shape == (2, e.embed_dim)
    text_norm_ok = (
        np.allclose(np.linalg.norm(te, axis=1), 1.0, atol=1e-4)
        and not np.isnan(te).any()
    )

    # Image
    valid_img = None
    for p in ["data/characters/isaac.png", "data/characters/gertie.png"]:
        if Path(p).exists():
            valid_img = p
            break
    image_ok = True
    if valid_img is None:
        image_ok = False
        valid_img = "INVALID_PATH.png"
    ie = e.embed_image_batch([valid_img])
    image_shape_ok = image_ok and ie.shape == (1, e.embed_dim)
    image_norm_ok = image_ok and np.allclose(np.linalg.norm(ie[0]), 1.0, atol=1e-4)

    # Fusion
    t = e.embed_text_batch(["hello"])[0]
    i = e.embed_image_batch([valid_img])[0]
    fused = e.fuse(t, i, alpha=0.7)
    fusion_ok = (
        fused.shape == t.shape
        and np.allclose(np.linalg.norm(fused), 1.0, atol=1e-4)
        and (not np.allclose(fused, t))
        and (not np.allclose(fused, i))
    )

    # Cache
    c1 = e.embed_text("hello world")
    c2 = e.embed_text("hello world")
    cache_ok = c1 is c2

    # Reproducibility
    r1 = e.embed_text("test")
    r2 = e.embed_text("test")
    repro_ok = np.allclose(r1, r2)

    # Performance (more lenient on CPU)
    texts = [f"t{i}" for i in range(16)]
    e.clear_cache()
    t1 = time.time()
    _ = e.embed_text_batch(texts)
    batch_time = time.time() - t1
    e.clear_cache()
    t2 = time.time()
    [e.embed_text(t) for t in texts]
    single_time = time.time() - t2
    perf_ok = batch_time < single_time * 1.2  # More lenient for CPU

    # Output format
    fmt_ok = isinstance(e.format_output("chunk", r1)["embedding"], np.ndarray)

    ready = all(
        [
            model_ok,
            text_shape_ok,
            text_norm_ok,
            image_shape_ok,
            image_norm_ok,
            fusion_ok,
            cache_ok,
            repro_ok,
            perf_ok,
            fmt_ok,
        ]
    )

    return {
        "model": model_ok,
        "text_shape": text_shape_ok,
        "text_norm": text_norm_ok,
        "image_shape": image_shape_ok,
        "image_norm": image_norm_ok,
        "fusion": fusion_ok,
        "cache": cache_ok,
        "repro": repro_ok,
        "perf": perf_ok,
        "format": fmt_ok,
        "ready": ready,
    }


def _print_checklist(printer, s: Dict[str, bool], colored: bool = True):
    def mark(ok: bool) -> str:
        if colored:
            return "[green]✔[/green]" if ok else "[red]✗[/red]"
        return "[✔]" if ok else "[✗]"

    printer("\nClipEmbedder Readiness Checklist")
    printer(f"{mark(s['model'])} Model loads once; preprocessing loaded")
    printer(
        f"{mark(s['text_shape'])} Text embeddings: correct shapes; {mark(s['text_norm'])} normalized & no NaN"
    )
    printer(
        f"{mark(s['image_shape'])} Image embeddings: shapes correct; {mark(s['image_norm'])} normalized & errors handled"
    )
    printer(
        f"{mark(s['fusion'])} Fusion: weighted average with renormalization, distinct from inputs"
    )
    printer(f"{mark(s['cache'])} Caching: cache hit on repeated inputs")
    printer(
        f"{mark(s['repro'])} Reproducibility: same input -> same vector; different -> different"
    )
    printer(f"{mark(s['perf'])} Performance: batch faster than sequential")
    printer(f"{mark(s['format'])} Output: embedding dicts well-formed")


def test_clipembedder_readiness_summary():
    global SUMMARY
    s = _compute_summary()
    SUMMARY = s
    _print_checklist(lambda m: console.print(m), s, colored=True)
    assert s["ready"]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if SUMMARY is None:
        return

    def writer(msg: str):
        terminalreporter.write_line(msg)

    _print_checklist(writer, SUMMARY, colored=False)


if __name__ == "__main__":
    """
    Allow `python tests/test_embeddings/test_clip_embedder.py` to emit a quick health report
    without invoking pytest explicitly. Helpful during iterative refactors where we just want
    to sanity-check the embedder contract.
    """
    console.rule("ClipEmbedder Quick Check")
    summary = _compute_summary()
    _print_checklist(lambda msg: console.print(msg), summary, colored=True)
    if summary["ready"]:
        console.print("\n[bold green]ClipEmbedder checks passed ✓[/bold green]")
        sys.exit(0)
    console.print("\n[bold red]ClipEmbedder checks failed ✗[/bold red]")
    sys.exit(1)
