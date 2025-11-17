#!/usr/bin/env python3
"""
Full Pipeline Demo: RAGPipeline end-to-end
Demonstrates: build_indices() → query() with hybrid retrieval + reranking
"""

import sys

sys.path.insert(0, "src")

from pipeline import RAGPipeline
from retriever.chroma_client import chroma_client
from rich.console import Console

console = Console()


def main():
    console.print("\n[bold cyan]═══ Full RAG Pipeline Demo ═══[/bold cyan]\n")

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Load sample data
    console.print("[bold]Step 1: Loading sample entities[/bold]")
    characters = pipeline.load_json_data("data/characters/isaac.json")
    locations = pipeline.load_json_data("data/locations/loc_office_001.json")

    if not characters or not locations:
        console.print("[red]✗[/red] Failed to load sample data")
        return

    # Build indices
    console.print("\n[bold]Step 2: Building indices (BM25 + Chroma)[/bold]")
    pipeline.build_indices(characters=characters, locations=locations, rebuild=True)

    # Verify indices
    char_count = chroma_client.get_collection_count("characters")
    loc_count = chroma_client.get_collection_count("locations")
    console.print(f"\n[green]✓[/green] Indices built:")
    console.print(f"  Characters: {char_count}")
    console.print(f"  Locations: {loc_count}")

    # Run queries
    console.print("\n[bold]Step 3: Running queries[/bold]\n")

    queries = [
        ("office", "characters"),
        ("desk", "characters"),
        ("window", "locations"),
    ]

    for query_text, collection in queries:
        console.print(
            f"\n[bold cyan]Query:[/bold cyan] '{query_text}' (collection: {collection})"
        )
        console.print("─" * 60)

        results = pipeline.query(
            query_text=query_text,
            collection=collection,
            top_k_retrieval=5,
            top_k_rerank=3,
        )

        if results:
            console.print(f"\n[green]✓[/green] Found {len(results)} result(s):\n")
            for i, result in enumerate(results, 1):
                entity_id = result.get("entity_id", "N/A")
                text = result.get("text", "")[:100]
                hybrid = result.get("hybrid_score", 0)
                rerank = result.get("rerank_score", 0)

                console.print(f"  [{i}] {entity_id}")
                console.print(f"      Text: {text}...")
                console.print(f"      Hybrid: {hybrid:.3f} | Rerank: {rerank:.3f}\n")
        else:
            console.print("[yellow]⚠[/yellow] No results found\n")

    console.print("\n[bold cyan]═══ Demo Complete ═══[/bold cyan]\n")
    console.print(
        "[green]✓[/green] Full pipeline (build_indices + query) executed successfully!"
    )


if __name__ == "__main__":
    main()
