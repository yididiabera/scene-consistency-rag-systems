"""
RAG Pipeline
Orchestrates the full retrieval-augmented generation workflow
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from config import cfg
from dataset import DatasetPreparer
from embedder import ClipEmbedder
from retriever.chroma_client import chroma_client
from retriever import HybridRetriever
from reranker import Reranker

console = Console()


class RAGPipeline:
    """Main RAG pipeline orchestrator."""

    def __init__(self):
        self.dataset_prep = DatasetPreparer()
        self.embedder = ClipEmbedder()
        self._retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[Reranker] = None
        console.print(Panel("[green]RAG Pipeline Initialized[/green]", expand=False))

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load entity data from JSON file."""
        path = Path(file_path)
        if not path.exists():
            console.print(f"[red]✗[/red] File not found: {file_path}")
            return []

        with open(path, "r") as f:
            data = json.load(f)

        # Handle both single objects and arrays
        if isinstance(data, dict):
            data = [data]

        console.print(f"[green]✓[/green] Loaded {len(data)} entities from {file_path}")
        return data

    @staticmethod
    def _entity_lookup(
        entities: List[Dict[str, Any]], id_key: str
    ) -> Dict[str, Dict[str, Any]]:
        """Build a lookup map of entity id → entity payload."""
        return {entity[id_key]: entity for entity in entities if entity.get(id_key)}

    def _index_documents(
        self,
        collection_name: str,
        docs: List[Dict[str, Any]],
        entity_lookup: Dict[str, Dict[str, Any]],
        entity_type: str,
        include_image: bool = True,
    ) -> int:
        """
        Embed chunk texts (plus optional canonical images) and upsert the payloads into Chroma.
        Minimal implementation: batch text embeddings, reuse cached image embeddings per entity.
        """
        if not docs:
            return 0

        texts = [doc["text"] for doc in docs]
        text_embs = self.embedder.embed_text_batch(texts)
        image_cache: Dict[str, Optional[Any]] = {}

        ids, embeddings, documents, metadatas = [], [], [], []
        for idx, doc in enumerate(docs):
            entity_id = doc.get("entity_id")
            entity = entity_lookup.get(entity_id, {})

            image_emb = None
            if include_image and entity:
                if entity_id not in image_cache:
                    path = entity.get("canonical_image_path")
                    image_cache[entity_id] = (
                        self.embedder.embed_image(path) if path else None
                    )
                image_emb = image_cache[entity_id]

            fused = self.embedder.fuse(text_embs[idx], image_emb)
            ids.append(doc["chunk_id"])
            embeddings.append(fused.astype(float).tolist())
            documents.append(doc["text"])

            tags_value = doc.get("tags", [])
            if isinstance(tags_value, list):
                tags_value = ",".join(str(t) for t in tags_value)
            else:
                tags_value = str(tags_value or "")

            metadatas.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "chunk_index": doc.get("chunk_index"),
                    "tags": tags_value,
                }
            )

        chroma_client.add_documents(
            collection_name, ids, embeddings, documents, metadatas
        )
        return len(ids)

    def build_indices(
        self,
        characters: List[Dict] = None,
        locations: List[Dict] = None,
        relationships: List[Dict] = None,
        rebuild: bool = False,
    ):
        """
        Build BM25 and ChromaDB indices from entity data.

        Args:
            characters: List of character dictionaries
            locations: List of location dictionaries
            relationships: List of relationship dictionaries
            rebuild: If True, rebuild indices from scratch
        """
        console.print("\n[bold cyan]═══ Building Indices ═══[/bold cyan]\n")

        # Reset collections if rebuilding
        if rebuild:
            console.print("[yellow]⟳[/yellow] Rebuilding indices...")
            chroma_client.reset_collection("characters")
            chroma_client.reset_collection("locations")
            chroma_client.reset_collection("relationships")

        all_documents = []

        characters = characters or []
        locations = locations or []
        relationships = relationships or []

        char_docs = (
            self.dataset_prep.prepare_documents(characters, "character")
            if characters
            else []
        )
        loc_docs = (
            self.dataset_prep.prepare_documents(locations, "location")
            if locations
            else []
        )
        rel_docs = (
            self.dataset_prep.prepare_documents(relationships, "relationship")
            if relationships
            else []
        )

        all_documents.extend(char_docs)
        all_documents.extend(loc_docs)
        all_documents.extend(rel_docs)

        char_lookup = (
            self._entity_lookup(characters, "character_id") if characters else {}
        )
        loc_lookup = self._entity_lookup(locations, "location_id") if locations else {}
        rel_lookup = (
            self._entity_lookup(relationships, "relationship_id")
            if relationships
            else {}
        )

        char_count = self._index_documents(
            "characters", char_docs, char_lookup, "character", include_image=True
        )
        loc_count = self._index_documents(
            "locations", loc_docs, loc_lookup, "location", include_image=True
        )
        rel_count = self._index_documents(
            "relationships", rel_docs, rel_lookup, "relationship", include_image=False
        )

        # Build BM25 index
        bm25 = self.dataset_prep.build_bm25_index(all_documents)
        self.dataset_prep.save_bm25_index(bm25, all_documents)

        # Print summary
        table = Table(title="Index Summary")
        table.add_column("Collection", style="cyan")
        table.add_column("Count", style="green")

        table.add_row(
            "Characters", str(chroma_client.get_collection_count("characters"))
        )
        table.add_row("Locations", str(chroma_client.get_collection_count("locations")))
        table.add_row(
            "Relationships", str(chroma_client.get_collection_count("relationships"))
        )
        table.add_row("BM25 Documents", str(len(all_documents)))

        console.print(table)
        console.print("\n[green]✓ Indices built successfully[/green]\n")

    def query(
        self,
        query_text: str,
        collection: str = "characters",
        top_k_retrieval: int = None,
        top_k_rerank: int = None,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system.

        Args:
            query_text: Query string
            collection: Collection to search ("characters", "locations", "relationships")
            top_k_retrieval: Number of results to retrieve (before reranking)
            top_k_rerank: Number of final results (after reranking)
            where: Metadata filter

        Returns:
            List of ranked results with scores
        """
        # Handle malformed queries gracefully
        if not query_text or not query_text.strip():
            console.print(
                "[yellow]⚠[/yellow] Empty query provided, returning no results"
            )
            return []

        top_k_retrieval = top_k_retrieval or cfg.get("top_k_retrieval", 20)
        top_k_rerank = top_k_rerank or cfg.get("top_k_rerank", 5)

        console.print(f"\n[bold cyan]═══ Query: '{query_text}' ═══[/bold cyan]\n")

        retriever = self._get_retriever()
        reranker = self._get_reranker()

        # Step 1: Hybrid retrieval
        try:
            results = retriever.hybrid_search(
                query=query_text,
                collection_name=collection,
                top_k=top_k_retrieval,
                where=where,
            )
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Query processing failed: {e}")
            return []

        if not results:
            console.print("[yellow]⚠[/yellow] No results found")
            return []

        # Step 2: Rerank
        reranked_results = reranker.rerank(query_text, results, top_k=top_k_rerank)

        # Display results
        self._display_results(reranked_results)

        return reranked_results

    def _display_results(self, results: List[Dict[str, Any]]):
        """Display query results in a nice table."""
        if not results:
            return

        table = Table(title="Top Results")
        table.add_column("#", style="dim", width=3)
        table.add_column("Entity ID", style="cyan")
        table.add_column("Text Preview", style="white", no_wrap=False)
        table.add_column("Hybrid", style="yellow", justify="right")
        table.add_column("Rerank", style="green", justify="right")

        for i, result in enumerate(results[:10], 1):
            text_preview = result.get("text", "")[:80] + "..."
            hybrid_score = f"{result.get('hybrid_score', 0):.3f}"
            rerank_score = f"{result.get('rerank_score', 0):.3f}"

            table.add_row(
                str(i),
                result.get("entity_id", ""),
                text_preview,
                hybrid_score,
                rerank_score,
            )

        console.print(table)
