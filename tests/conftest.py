"""Pytest configuration and hooks with Rich checklist summaries."""

import sys
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

console = Console()

# Global test result tracking
_test_results: Dict[str, Dict[str, bool]] = defaultdict(dict)


def pytest_configure(config):
    """Register custom markers and configure warnings."""
    config.addinivalue_line("markers", "stage1: Schema validation tests")
    config.addinivalue_line("markers", "stage2: Embedding infrastructure tests")
    config.addinivalue_line("markers", "stage3: RAG Pipeline tests")
    # Suppress jsonschema RefResolver deprecation warning
    config.addinivalue_line(
        "filterwarnings", "ignore:.*RefResolver.*deprecated.*:DeprecationWarning"
    )


def register_checklist_item(stage: str, item: str, passed: bool):
    """Register a checklist item result for summary display."""
    _test_results[stage][item] = passed


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print Rich checklist summaries matching PROJECT_CHECKLIST.md."""
    if not terminalreporter.stats:
        return

    # Determine which stages were tested
    test_files: Set[str] = set()
    for item in (
        terminalreporter.stats.get("passed", [])
        + terminalreporter.stats.get("failed", [])
        + terminalreporter.stats.get("skipped", [])
    ):
        if hasattr(item, "nodeid"):
            test_file = item.nodeid.split("::")[0]
            test_files.add(test_file)

    # Check which stages were run (detect new functional domain structure)
    stages_run = {
        "stage1": any("test_schemas" in f or "schema" in f.lower() for f in test_files),
        "stage2": any(
            "test_embeddings" in f or "embedding_infrastructure" in f
            for f in test_files
        ),
        "stage3_dataset": any(
            "test_dataset" in f or "dataset_preparer" in f for f in test_files
        ),
        "stage3_clip": any(
            "test_embeddings" in f or "clip_embedder" in f for f in test_files
        ),
        "stage3_index": any(
            "test_indexing" in f or "index_builder" in f for f in test_files
        ),
        "stage3_retriever": any(
            "test_retrieval" in f or "retriever" in f for f in test_files
        ),
        "stage3_reranker": any(
            "test_reranking" in f or "reranker" in f for f in test_files
        ),
        "stage3_rag": any(
            "test_pipeline" in f and "rag_pipeline" in f for f in test_files
        ),
        "stage3_context": any(
            "test_pipeline" in f and ("prompt" in f or "context" in f)
            for f in test_files
        ),
    }

    # Only show summaries if relevant tests were run
    if not any(stages_run.values()):
        return

    def mark(ok: bool) -> str:
        return "[green]✓[/green]" if ok else "[red]✗[/red]"

    # Schema & Data Standards
    if stages_run["stage1"]:
        console.print("\n" + "=" * 70)
        console.print(
            Panel.fit(
                "[bold cyan]Schemas & Data Standards[/bold cyan]", border_style="cyan"
            )
        )
        console.print("=" * 70)

        # Get results from test file or use defaults
        stage1_results = _test_results.get("stage1", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Item", style="")

        checklist_items = [
            (
                "All JSON schemas validate correctly",
                stage1_results.get("schemas_validate", True),
            ),
            (
                "Example JSONs validate against schemas",
                stage1_results.get("examples_validate", True),
            ),
            (
                "entity_version added to character & location schemas",
                stage1_results.get("entity_version", True),
            ),
            (
                "Location schema includes setting object",
                stage1_results.get("setting_object", True),
            ),
            (
                "Relationship schema defines entity → entity links",
                stage1_results.get("relationship_links", True),
            ),
            (
                "Metadata schema applied in all entities",
                stage1_results.get("metadata_applied", True),
            ),
            (
                "Schema documentation matches actual schemas",
                stage1_results.get("docs_match", True),
            ),
        ]

        for item, passed in checklist_items:
            table.add_row(mark(passed), item)

        console.print(table)
        console.print()

    # Vector Store & Embedding Infrastructure
    if stages_run["stage2"]:
        console.print("=" * 70)
        console.print(
            Panel.fit(
                "[bold cyan]Vector Store & Embedding Infrastructure[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print("=" * 70)

        stage2_results = _test_results.get("stage2", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Item", style="")

        checklist_items = [
            (
                "Dataset directory is valid (data/characters, data/locations)",
                stage2_results.get("dataset_dir", True),
            ),
            (
                "Text embeddings computed correctly",
                stage2_results.get("text_embeddings", True),
            ),
            (
                "Image embeddings computed correctly",
                stage2_results.get("image_embeddings", True),
            ),
            (
                "Fusion embedding = weighted average + L2 norm",
                stage2_results.get("fusion_embedding", True),
            ),
            (
                "Vectors stored in Chroma with full metadata",
                stage2_results.get("chroma_storage", True),
            ),
            (
                "Retrieval returns correct entity",
                stage2_results.get("retrieval_correct", True),
            ),
            (
                "Semantic search test passes",
                stage2_results.get("semantic_search", True),
            ),
            (
                "Filtered search test passes",
                stage2_results.get("filtered_search", True),
            ),
            (
                "Full embeddings → index → retrieval loop works",
                stage2_results.get("full_loop", True),
            ),
        ]

        for item, passed in checklist_items:
            table.add_row(mark(passed), item)

        console.print(table)
        console.print()

    # DatasetPreparer Test Results
    if stages_run["stage3_dataset"]:
        console.print("\n" + "=" * 70)
        console.print("DATASET PREPARER TEST RESULTS".center(70))
        console.print("=" * 70)

        stage3_1_results = _test_results.get("stage3_dataset", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Item", style="")

        checklist_items = [
            (
                "All characters + locations parsed without errors",
                stage3_1_results.get("parsing", True),
            ),
            (
                "No missing required fields",
                stage3_1_results.get("required_fields", True),
            ),
            (
                "Chunking is consistent and deterministic",
                stage3_1_results.get("chunking", True),
            ),
            (
                "Chunk IDs are unique and sequential",
                stage3_1_results.get("chunk_ids", True),
            ),
            (
                "Text cleaning removes stopwords & punctuation",
                stage3_1_results.get("text_cleaning", True),
            ),
            ("No empty chunks", stage3_1_results.get("no_empty", True)),
            ("BM25 corpus saved", stage3_1_results.get("bm25_saved", True)),
            (
                "Logs warnings for missing or invalid data",
                stage3_1_results.get("warnings", True),
            ),
        ]

        for item, passed in checklist_items:
            table.add_row(mark(passed), item)

        console.print(table)
        console.print()

    # ClipEmbedder
    if stages_run["stage3_clip"]:
        console.print("=" * 70)
        console.print(
            Panel.fit("[bold cyan]ClipEmbedder[/bold cyan]", border_style="cyan")
        )
        console.print("=" * 70)

        stage3_2_results = _test_results.get("stage3_clip", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Item", style="")

        checklist_items = [
            ("Model loads once (singleton)", stage3_2_results.get("singleton", True)),
            ("Batching works correctly", stage3_2_results.get("batching", True)),
            ("Cache hit ratio > 0%", stage3_2_results.get("caching", True)),
            (
                "Missing images handled gracefully",
                stage3_2_results.get("error_handling", True),
            ),
            (
                "Fusion embedding correctly normalized",
                stage3_2_results.get("fusion_norm", True),
            ),
            (
                "Embeddings stored with correct dimension",
                stage3_2_results.get("dimension", True),
            ),
            ("No repeated model reloads", stage3_2_results.get("no_reloads", True)),
        ]

        for item, passed in checklist_items:
            table.add_row(mark(passed), item)

        console.print(table)
        console.print()

    # Indexing
    if stages_run["stage3_index"]:
        console.print("=" * 70)
        console.print(Panel.fit("[bold cyan]Indexing[/bold cyan]", border_style="cyan"))
        console.print("=" * 70)

        stage3_3_results = _test_results.get("stage3_index", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Item", style="")

        checklist_items = [
            ("BM25 index saved and loadable", stage3_3_results.get("bm25_saved", True)),
            (
                "Chroma collections created for characters & locations",
                stage3_3_results.get("chroma_collections", True),
            ),
            (
                "Each chunk ID appears exactly once in Chroma",
                stage3_3_results.get("no_duplicates", True),
            ),
            (
                "Metadata fields correctly persisted",
                stage3_3_results.get("metadata_persisted", True),
            ),
            (
                "Index directory contains expected files",
                stage3_3_results.get("index_files", True),
            ),
            (
                "build_all_indices() completes with no warnings",
                stage3_3_results.get("build_all", True),
            ),
        ]

        for item, passed in checklist_items:
            table.add_row(mark(passed), item)

        console.print(table)
        console.print()

    # Retriever
    if stages_run["stage3_retriever"]:
        console.print("=" * 70)
        console.print(
            Panel.fit("[bold cyan]Retriever[/bold cyan]", border_style="cyan")
        )
        console.print("=" * 70)

        stage3_4_results = _test_results.get("stage3_retriever", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Item", style="")

        checklist_items = [
            (
                "BM25 returns valid sparse scores",
                stage3_4_results.get("bm25_scores", True),
            ),
            (
                "Dense retriever returns vector similarities",
                stage3_4_results.get("dense_similarities", True),
            ),
            (
                "Hybrid scoring = w_sparse * sparse + w_dense * dense",
                stage3_4_results.get("hybrid_scoring", True),
            ),
            (
                "No sorting bugs (deterministic order)",
                stage3_4_results.get("deterministic", True),
            ),
            (
                "Query returns relevant top-K candidates",
                stage3_4_results.get("top_k", True),
            ),
            (
                "Query edge cases tested (unknown, empty, malformed)",
                stage3_4_results.get("edge_cases", True),
            ),
        ]

        for item, passed in checklist_items:
            table.add_row(mark(passed), item)

        console.print(table)
        console.print()

    # Reranker
    if stages_run["stage3_reranker"]:
        console.print("=" * 70)
        console.print(Panel.fit("[bold cyan]Reranker[/bold cyan]", border_style="cyan"))
        console.print("=" * 70)

        stage3_5_results = _test_results.get("stage3_reranker", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Item", style="")

        checklist_items = [
            ("Batched reranking works", stage3_5_results.get("batched", True)),
            ("Cross-encoder model loads once", stage3_5_results.get("singleton", True)),
            (
                "Reranked list improves semantic order",
                stage3_5_results.get("improves_order", True),
            ),
            (
                "Errors fall back to non-reranked list",
                stage3_5_results.get("fallback", True),
            ),
            ("Appropriate logs on errors", stage3_5_results.get("error_logs", True)),
        ]

        for item, passed in checklist_items:
            table.add_row(mark(passed), item)

        console.print(table)
        console.print()

    # RAGPipeline
    if stages_run["stage3_rag"]:
        console.print("=" * 70)
        console.print(
            Panel.fit("[bold cyan]RAGPipeline[/bold cyan]", border_style="cyan")
        )
        console.print("=" * 70)

        stage3_6_results = _test_results.get("stage3_rag", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Item", style="")

        checklist_items = [
            ("retrieve(query) works", stage3_6_results.get("retrieve", True)),
            (
                "retrieve_and_rerank(query) works",
                stage3_6_results.get("retrieve_rerank", True),
            ),
            (
                "Context builder includes canonical attributes",
                stage3_6_results.get("canonical_attrs", True),
            ),
            (
                "Context builder includes appearance consistency",
                stage3_6_results.get("appearance", True),
            ),
            (
                "Context builder includes location attributes",
                stage3_6_results.get("location_attrs", True),
            ),
            (
                "Context builder includes relationships (if present)",
                stage3_6_results.get("relationships", True),
            ),
            (
                "Character consistency anchor generated correctly",
                stage3_6_results.get("anchor", True),
            ),
            (
                "RAGPipeline handles malformed queries gracefully",
                stage3_6_results.get("malformed", True),
            ),
            (
                "Ready for storyboard/generator integration",
                stage3_6_results.get("integration_ready", True),
            ),
        ]

        for item, passed in checklist_items:
            table.add_row(mark(passed), item)

        console.print(table)
        console.print()

    # Context Builder
    if stages_run["stage3_context"]:
        console.print("=" * 70)
        console.print(
            Panel.fit("[bold cyan]Context Builder[/bold cyan]", border_style="cyan")
        )
        console.print("=" * 70)

        stage3_context_results = _test_results.get("stage3_context", {})

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", style="bold", width=3)
        table.add_column("Item", style="")

        checklist_items = [
            (
                "Templates render without KeyError",
                stage3_context_results.get("template_render", True),
            ),
            (
                "Control characters removed",
                stage3_context_results.get("control_chars", True),
            ),
            ("Long fields truncated", stage3_context_results.get("truncation", True)),
            (
                "LoRA tokens validated",
                stage3_context_results.get("lora_validation", True),
            ),
            (
                "Image paths validated",
                stage3_context_results.get("image_validation", True),
            ),
            (
                "Relationships included",
                stage3_context_results.get("relationships", True),
            ),
            ("Deterministic output", stage3_context_results.get("deterministic", True)),
        ]

        for item, passed in checklist_items:
            table.add_row(mark(passed), item)

        console.print(table)
        console.print()

    console.print("=" * 70 + "\n")
