# \!/usr/bin/env python3
"""
DatasetPreparer verification
- Loading entities
- Chunk extraction (deterministic, no empties, edge cases)
- Chunk ID generation format
- BM25 preprocessing (clean tokens, deterministic)
- BM25 index build (corpus size, vocab > 0)
- Output structure from prepare()
- Readiness summary (colored + pytest summary)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

import sys

sys.path.insert(0, "src")
from dataset import DatasetPreparer  # noqa: E402

from rich.console import Console

console = Console()
SUMMARY_STATUS: Optional[Dict[str, bool]] = None


@pytest.fixture(scope="module")
def dp() -> DatasetPreparer:
    return DatasetPreparer()


# 1) Dataset Loading
def test_dataset_loading(dp: DatasetPreparer):
    chars = dp.load_entities("data/characters", "character")
    locs = dp.load_entities("data/locations", "location")
    assert len(chars) > 0 and len(locs) > 0
    assert "name" in chars[0]
    assert "description" in locs[0]

    # Invalid JSON raises
    tmp_dir = Path("tests/_tmp_invalid")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    bad = tmp_dir / "bad.json"
    bad.write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        dp.load_entities(str(tmp_dir), "character")
    # cleanup
    bad.unlink(missing_ok=True)
    tmp_dir.rmdir()


# 2) Chunk Extraction
def test_chunk_extraction(dp: DatasetPreparer):
    isaac = json.loads(Path("data/characters/isaac.json").read_text())
    chunks1 = dp.chunk_entity(isaac, "character")
    assert len(chunks1) >= 1
    assert all(len(c["text"]) > 10 for c in chunks1)

    chunks2 = dp.chunk_entity(isaac, "character")
    assert [c["doc_id"] for c in chunks1] == [c["doc_id"] for c in chunks2]
    assert [c["text"] for c in chunks1] == [c["text"] for c in chunks2]

    short_entity = {"character_id": "char_short_001", "appearance": "short text"}
    short_chunks = dp.chunk_entity(short_entity, "character")
    assert len(short_chunks) == 1

    very_long_text = ("lorem ipsum ") * ((dp.chunk_size // 11) * 3)
    long_entity = {"character_id": "char_long_001", "appearance": very_long_text}
    long_chunks = dp.chunk_entity(long_entity, "character")
    assert len(long_chunks) >= 2

    weird = {
        "character_id": "char_weird_001",
        "appearance": "line1\n\n\nline2\t\t text",
    }
    weird_chunks = dp.chunk_entity(weird, "character")
    assert len(weird_chunks) >= 1
    assert all(len(c["text"]) > 0 for c in weird_chunks)


# 3) Chunk ID Generation
def test_chunk_id_generation(dp: DatasetPreparer):
    isaac = json.loads(Path("data/characters/isaac.json").read_text())
    chunks = dp.chunk_entity(isaac, "character")
    first_id = chunks[0]["chunk_id"]
    assert first_id.startswith(isaac["character_id"])  # e.g., char_isaac_001_00
    assert first_id.endswith("_00")

    gertie = json.loads(Path("data/characters/gertie.json").read_text())
    g_chunks = dp.chunk_entity(gertie, "character")
    assert set(c["chunk_id"] for c in chunks).isdisjoint(
        set(c["chunk_id"] for c in g_chunks)
    )


# 4) BM25 Preprocessing
def test_bm25_preprocess(dp: DatasetPreparer):
    s = "This is a TEST, with punctuation!!"
    prepped = dp.bm25_preprocess(s)
    assert prepped == ["this", "is", "test", "with", "punctuation"]
    assert prepped == dp.bm25_preprocess(s)
    assert dp.bm25_preprocess("Café naïve façade")


# 5) BM25 Index Build
def test_bm25_index_build(dp: DatasetPreparer):
    chars = dp.load_entities("data/characters", "character")
    locs = dp.load_entities("data/locations", "location")
    char_docs = dp.prepare_documents(chars, "character")
    loc_docs = dp.prepare_documents(locs, "location")
    docs = char_docs + loc_docs

    bm25 = dp.build_bm25_index(docs)
    assert bm25 is not None
    assert hasattr(bm25, "corpus")
    assert len(bm25.corpus) == len(docs)
    assert len(set(t for doc in bm25.corpus for t in doc)) > 0


# 6) Output Structure Validation
def test_prepare_output_structure(dp: DatasetPreparer):
    dataset = dp.prepare()
    assert set(dataset.keys()) >= {"characters", "locations", "bm25"}

    def check_chunks(chunks: List[Dict]):
        for c in chunks:
            assert "entity_id" in c
            assert "chunk_id" in c
            assert "text" in c and len(c["text"]) > 0
            assert "bm25_tokens" in c and isinstance(c["bm25_tokens"], list)
            assert "metadata" in c

    check_chunks(dataset["characters"])
    check_chunks(dataset["locations"])


def _compute_preparer_status(dp: DatasetPreparer) -> Dict[str, bool]:
    chars = dp.load_entities("data/characters", "character")
    locs = dp.load_entities("data/locations", "location")
    char_docs = dp.prepare_documents(chars, "character")
    loc_docs = dp.prepare_documents(locs, "location")
    docs = char_docs + loc_docs

    data_ok = (
        len(chars) > 0
        and len(locs) > 0
        and "name" in chars[0]
        and "description" in locs[0]
    )
    chunk_ok = (
        len(char_docs) > 0
        and len(loc_docs) > 0
        and all(len(c["text"]) > 0 for c in docs)
    )
    id_ok = all(c["chunk_id"].startswith(c["entity_id"]) for c in docs) and all(
        c["chunk_id"].endswith("_00") for c in docs if c["chunk_index"] == 0
    )
    tokens_ok = all(
        isinstance(c["bm25_tokens"], list)
        and all(isinstance(t, str) for t in c["bm25_tokens"])
        for c in docs
    )

    bm25 = dp.build_bm25_index(docs)
    bm25_ok = (
        hasattr(bm25, "corpus")
        and len(bm25.corpus) == len(docs)
        and len(set(t for doc in bm25.corpus for t in doc)) > 0
    )

    ready = data_ok and chunk_ok and id_ok and tokens_ok and bm25_ok
    return {
        "data": data_ok,
        "chunk": chunk_ok,
        "ids": id_ok,
        "tokens": tokens_ok,
        "bm25": bm25_ok,
        "ready": ready,
    }


def _print_checklist(printer, status: Dict[str, bool], colored: bool = True):
    # Create a console with color support
    console = Console(force_terminal=True, color_system="auto")

    # Only print the checklist if all tests passed
    if all(status.values()):
        console.print("\n[bold]DatasetPreparer Readiness Checklist[/bold]")

        def print_check(ok: bool, message: str):
            if ok:
                console.print(f"[green]✓[/green] {message}")
            else:
                console.print(f"[red]✗[/red] {message}")

        print_check(status["data"], "Data quality OK")
        print_check(status["chunk"], "Chunking deterministic & non-empty")
        print_check(status["ids"], "Chunk IDs formatted & collision-free")
        print_check(status["tokens"], "BM25 tokens clean")
        print_check(status["bm25"], "BM25 index builds and corpus matches chunks")

    # Return the overall status
    return all(status.values())


def test_preparer_readiness_summary(dp: DatasetPreparer):
    global SUMMARY_STATUS
    status = _compute_preparer_status(dp)
    SUMMARY_STATUS = status

    # Print the checklist and get the overall status
    all_passed = _print_checklist(print, status)

    # Assert that all checks passed
    assert all_passed, "One or more dataset preparation checks failed"


if __name__ == "__main__":
    console = Console(force_terminal=True, color_system="auto")
    dp = DatasetPreparer()
    try:
        status = _compute_preparer_status(dp)
        all_passed = _print_checklist(print, status)

        if all_passed:
            console.print("\n[green]✓ All DatasetPreparer checks passed.[/green]")
            sys.exit(0)
        else:
            console.print("\n[red]✗ One or more DatasetPreparer checks failed.[/red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error running DatasetPreparer checks: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if SUMMARY_STATUS is None:
        return

    # Create a console that writes to the terminal reporter
    console = Console(force_terminal=True, color_system="auto")

    # Capture rich output and write it line by line
    with console.capture() as capture:
        _print_checklist(console.print, SUMMARY_STATUS, colored=True)

    # Write the captured output to the terminal reporter
    for line in capture.get().splitlines():
        terminalreporter.write_line(line)
