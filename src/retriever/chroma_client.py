"""
ChromaDB Client
Manages vector storage for dense retrieval
"""

import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from config import cfg

console = Console()


class ChromaClient:
    """ChromaDB client for vector storage and retrieval."""

    def __init__(self):
        chroma_path = Path(cfg.get("chroma_store_path", "data/chroma_store"))
        chroma_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(chroma_path))

        # Get or create collections
        collections_config = cfg.get("collections", {})
        self.characters = self.client.get_or_create_collection(
            name=collections_config.get("characters", "characters")
        )
        self.locations = self.client.get_or_create_collection(
            name=collections_config.get("locations", "locations")
        )
        self.relationships = self.client.get_or_create_collection(
            name=collections_config.get("relationships", "relationships")
        )

        console.print(f"[green]✓[/green] ChromaDB client initialized at {chroma_path}")

    def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ):
        """Add documents to a collection."""
        collection = getattr(self, collection_name)

        collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

        console.print(
            f"[green]✓[/green] Added {len(ids)} documents to {collection_name}"
        )

    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Query a collection with vector similarity."""
        collection = getattr(self, collection_name)

        results = collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=where
        )

        return results

    def get_collection_count(self, collection_name: str) -> int:
        """Get number of documents in a collection."""
        collection = getattr(self, collection_name)
        return collection.count()

    def reset_collection(self, collection_name: str):
        """Delete and recreate a collection."""
        collection_config_name = cfg.get("collections", {}).get(
            collection_name, collection_name
        )

        try:
            self.client.delete_collection(name=collection_config_name)
        except Exception:
            # Collection may not exist yet, which is fine - we're about to create it
            pass

        collection = self.client.get_or_create_collection(name=collection_config_name)
        setattr(self, collection_name, collection)

        console.print(f"[green]✓[/green] Reset collection: {collection_name}")


# Global ChromaDB client
chroma_client = ChromaClient()
