"""
Reranker
Uses CrossEncoder for final relevance scoring
"""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from config import cfg

console = Console()

# Singleton CrossEncoder model
_cross_encoder = None


def get_cross_encoder():
    """Load CrossEncoder model once and return cached instance."""
    global _cross_encoder

    if _cross_encoder is None:
        model_name = cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Loading CrossEncoder ({model_name})...", total=None)
            _cross_encoder = CrossEncoder(model_name)

        console.print(f"[green]✓[/green] CrossEncoder loaded: {model_name}")

    else:
        console.print("[green]✓[/green] CrossEncoder loaded from cache")

    return _cross_encoder


class Reranker:
    """Reranks retrieval results using CrossEncoder."""

    def __init__(self):
        self.model = get_cross_encoder()
        self.batch_size = cfg.get("reranker_batch_size", 32)
        console.print("[green]✓[/green] Reranker initialized")

    def rerank(
        self, query: str, results: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using CrossEncoder.

        Args:
            query: Query text
            results: List of retrieval results
            top_k: Number of top results to return

        Returns:
            Reranked results with rerank_score
        """
        if not results:
            return []

        pairs = [(query, result.get("text", "")) for result in results]
        try:
            rerank_scores = self.model.predict(pairs, batch_size=self.batch_size)
        except Exception as exc:
            console.print(
                f"[yellow]⚠[/yellow] Reranking failed, falling back to original ordering: {exc}"
            )
            return results[:top_k]

        scores = [float(s) for s in rerank_scores]
        if scores:
            if len(scores) == 1:
                norm_scores = [1.0]
            else:
                s_min = min(scores)
                s_max = max(scores)
                denom = s_max - s_min
                if denom <= 1e-12:
                    norm_scores = [0.0 for _ in scores]
                else:
                    norm_scores = [(s - s_min) / denom for s in scores]
        else:
            norm_scores = []

        for result, score, norm in zip(results, scores, norm_scores):
            result["rerank_score"] = score
            result["rerank_score_norm"] = norm

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        console.print(
            f"[cyan]→[/cyan] Reranked {len(results)} results, returning top-{top_k}"
        )
        return reranked[:top_k]
