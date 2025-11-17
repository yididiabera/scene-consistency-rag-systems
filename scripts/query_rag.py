#!/usr/bin/env python3
"""
Query the RAG system via the RAGPipeline using hybrid retrieval + reranking.

Usage:
  python scripts/query_rag.py --query "male protagonist athletic build" --collection characters --topk 5

Notes:
- Requires dense indices in ChromaDB (run scripts/build_dense.py first).
- Requires BM25 index (run scripts/build_bm25.py first).
"""

import argparse
import sys
from rich.console import Console
from rich.table import Table

sys.path.append("src")
from pipeline import RAGPipeline  # noqa: E402

console = Console()


def main(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline()

    results = pipeline.query(
        query_text=args.query,
        collection=args.collection,
        top_k_retrieval=args.topk_retrieval,
        top_k_rerank=args.topk,
        where=None,
    )

    if not results:
        console.print("[yellow]No results returned[/yellow]")
        return 0

    # Also print a plain summary for scripting use
    table = Table(title="Query Results")
    table.add_column("#", style="dim", width=3)
    table.add_column("Entity ID", style="cyan")
    table.add_column("Hybrid", style="yellow", justify="right")
    table.add_column("Rerank", style="green", justify="right")

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.get("entity_id", ""),
            f"{r.get('hybrid_score', 0):.3f}",
            f"{r.get('rerank_score', 0):.3f}",
        )

    console.print(table)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument(
        "--collection",
        default="characters",
        choices=["characters", "locations", "relationships"],
        help="Target collection",
    )
    parser.add_argument(
        "--topk_retrieval", type=int, default=20, help="Top-K before reranking"
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-K after reranking")
    ns = parser.parse_args()
    raise SystemExit(main(ns))
