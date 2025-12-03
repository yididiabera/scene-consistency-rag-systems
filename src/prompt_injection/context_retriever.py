"""
ContextRetriever
----------------
Step 2 of the Injection Pipeline.

Responsibility:
Take the Entity IDs found in Step 1, and query the RAG Pipeline
using the FULL prompt to find the most relevant context chunks.

Output:
A dictionary mapping Entity IDs to their retrieved text chunks.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import your existing pipeline
# Using relative import to avoid path issues
from pipeline import RAGPipeline
from .entity_extractor import EntityExtractionResult

@dataclass
class RetrievedContext:
    """
    Holds the raw text chunks found for each entity.
    """
    character_context: Dict[str, List[str]]  # { "char_isaac_001": ["He wears a white t-shirt..."] }
    location_context: Dict[str, List[str]]   # { "loc_office_001": ["The office is dimly lit..."] }
    character_scores: Optional[Dict[str, Dict[str, Any]]] = None
    location_scores: Optional[Dict[str, Dict[str, Any]]] = None

class ContextRetriever:
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Args:
            rag_pipeline: An initialized instance of your RAGPipeline.
                          (Do not instantiate it here to avoid reloading models).
        """
        self.rag = rag_pipeline

    def retrieve(self, prompt: str, entities: EntityExtractionResult) -> RetrievedContext:
        """
        The Core Logic:
        1. Iterate over every found entity ID.
        2. Query the RAG system using the *Original Prompt* as the query vector.
        3. Apply a Strict Filter for that specific ID.
        """
        char_data: Dict[str, List[str]] = {}
        loc_data: Dict[str, List[str]] = {}
        char_scores: Dict[str, Dict[str, Any]] = {}
        loc_scores: Dict[str, Dict[str, Any]] = {}

        # 1. Retrieve Characters
        for char_id in entities.characters:
            results = self.rag.query(
                query_text=prompt,          # <--- REVIEWER REQUEST: Use full prompt context
                collection="characters",
                top_k_retrieval=3,          # Fetch top 3 relevant chunks
                top_k_rerank=1,             # Rerank and keep only the BEST one
                where={"entity_id": char_id} # <--- CRITICAL: Strict ID filtering
            )

            if not results:
                continue

            # Extract text content, supporting both 'text' and legacy 'document' keys
            chunks: List[str] = []
            for r in results:
                text_value = r.get("text") or r.get("document")
                if not text_value:
                    continue
                chunks.append(text_value)

            if not chunks:
                continue

            char_data[char_id] = chunks

            # Capture score metadata for the top reranked result
            best = results[0]
            metadata = best.get("metadata") or {}
            chunk_index = best.get("chunk_index")
            if chunk_index is None and isinstance(metadata, dict):
                chunk_index = metadata.get("chunk_index")

            char_scores[char_id] = {
                "bm25_score": best.get("bm25_score"),
                "dense_score": best.get("dense_score"),
                "hybrid_score": best.get("hybrid_score"),
                "rerank_score": best.get("rerank_score"),
                "rerank_score_norm": best.get("rerank_score_norm"),
                "collection": "characters",
                "chunk_index": chunk_index,
                "chunk_id": best.get("chunk_id"),
                "doc_id": best.get("doc_id"),
            }

        # 2. Retrieve Locations
        for loc_id in entities.locations:
            results = self.rag.query(
                query_text=prompt,
                collection="locations",
                top_k_retrieval=3,
                top_k_rerank=1,
                where={"entity_id": loc_id}  # <--- Use entity_id for locations (matches index metadata)
            )

            if not results:
                continue

            chunks: List[str] = []
            for r in results:
                text_value = r.get("text") or r.get("document")
                if not text_value:
                    continue
                chunks.append(text_value)

            if not chunks:
                continue

            loc_data[loc_id] = chunks

            best = results[0]
            metadata = best.get("metadata") or {}
            chunk_index = best.get("chunk_index")
            if chunk_index is None and isinstance(metadata, dict):
                chunk_index = metadata.get("chunk_index")

            loc_scores[loc_id] = {
                "bm25_score": best.get("bm25_score"),
                "dense_score": best.get("dense_score"),
                "hybrid_score": best.get("hybrid_score"),
                "rerank_score": best.get("rerank_score"),
                "rerank_score_norm": best.get("rerank_score_norm"),
                "collection": "locations",
                "chunk_index": chunk_index,
                "chunk_id": best.get("chunk_id"),
                "doc_id": best.get("doc_id"),
            }

        return RetrievedContext(
            character_context=char_data,
            location_context=loc_data,
            character_scores=char_scores or None,
            location_scores=loc_scores or None,
        )

# ======================================================================
# Convenience Entry Point
# ======================================================================

def retrieve_context(
    prompt: str,
    entities: EntityExtractionResult,
    pipeline_instance: RAGPipeline
) -> RetrievedContext:
    """
    Functional interface.
    """
    retriever = ContextRetriever(pipeline_instance)
    return retriever.retrieve(prompt, entities)