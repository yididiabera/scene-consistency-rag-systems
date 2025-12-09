# Scene Consistency RAG System

Multimodal Retrieval-Augmented Generation (RAG) stack for keeping anime characters, locations, and relationships visually consistent across generated video scenes. The system ingests curated JSON datasets, builds BM25 + CLIP indices, performs hybrid retrieval, reranks candidates, and feeds consistency-aware prompts to downstream generators.

---

## Overview

- **Purpose**: Guarantee scene and character continuity for long-form video synthesis.
- **Scope**: Dataset validation, chunking, multimodal embeddings, hybrid retrieval, reranking, and pipeline orchestration.
- **Implementation**: Minimal Python modules under `src/` with rich logging, pytest coverage, and documentation that mirrors production behaviors.

### High-Level Data Flow

```
Dataset JSON + canonical images
        ↓
DatasetPreparer  (chunking → BM25 tokens → saved sparse index)
        ↓
ClipEmbedder    (batch text/image → α-fusion → L2 norm → cache)
        ↓
Index Builder   (RAGPipeline.build_indices → Chroma collections + BM25 pickle)
        ↓
HybridRetriever (BM25 + dense search, min–max fusion)
        ↓
CrossEncoder Reranker (sentence-transformer reranking)
        ↓
### Prompt Injection Pipeline
```
SceneConsistencyEngine
        ↓
EntityExtractor (extracts entities from narrative)
        ↓
ContextRetriever (queries RAG for canonical descriptions)
        ↓
ShotEnricher (merges narrative + RAG context + metadata)
        ↓
EnrichedShot JSON (ready for downstream generation)
```

See `docs/ARCHITECTURE.md` for diagrams and component handoffs.

---

## Installation & Environment

```bash
git clone <repository-url>
cd scene-consistency-rag-systems

python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU note**: CLIP and CrossEncoder models automatically select CUDA when available; otherwise they fall back to CPU.

---

## Directory Map (abridged)

| Path | Description |
|------|-------------|
| `src/config/` | YAML-backed configuration singleton (`cfg`) |
| `src/dataset/` | `DatasetPreparer` (chunking, BM25 build/load) |
| `src/embedder/` | `ClipEmbedder`, backend loader, normalization utils |
| `src/retriever/` | Chroma client + `HybridRetriever` |
| `src/reranker/` | Cross-encoder reranker (`CrossEncoderReranker`) |
| `src/pipeline/` | `RAGPipeline` orchestrator |
| `src/prompt_injection/` | `SceneConsistencyEngine` & prompt injection logic |
| `schemas/` | JSON Schema contracts for data ingestion |
| `docs/` | Architecture, schema, and design notes |
| `tests/` | Pytest suites grouped by stage |
| `data/` | Example entities, canonical images, persisted BM25/Chroma stores |

Run `tree -L 2` for the full layout.

---

## Usage

### Build Indices from Curated Data

```python
import sys
sys.path.insert(0, "src")  # ensure modules resolve

from pipeline import RAGPipeline

pipeline = RAGPipeline()
# Load all data from directories
# ... (see demo_pipeline.py for full loading logic)

pipeline.build_indices(
    characters=characters,
    locations=locations,
    rebuild=True  # clears Chroma + regenerates BM25
)
```

### Demo Pipeline

```bash
python demo_pipeline.py
```

Shows end-to-end RAG consistency engine workflow:
1. Entity extraction from shot descriptions
2. Context retrieval using RAG
3. Shot enrichment with canonical entity descriptions
4. EnrichedShot JSON output ready for video generation

---

## Configuration

All currently used YAML config keys live in `src/config/default_config.yaml`.
You can optionally create `configs/config.yaml` to override these values; if it does
not exist, the default config is used.

| Key | Purpose | Default |
|-----|---------|---------|
| `clip_model` | CLIP backbone passed to `load_clip_model` | `"ViT-B/32"` |
| `bm25_weight` | Sparse score weight inside `HybridRetriever` | `0.3` |
| `top_k_retrieval` | Default `HybridRetriever` candidate count | `20` |
| `top_k_rerank` | Final results returned by `RAGPipeline.query` | `5` |
| `bm25_index_path` | Pickle target for saved BM25 state | `data/bm25_index.pkl` |
| `chroma_store_path` | On-disk storage for Chroma collections | `data/chroma_store` |
| `embed_cache_dir` | Directory for optional on-disk embedding cache | `data/embed_cache` |
| `max_cache_size` | Max items in LRU memory cache (prevents OOM) | `1000` |
| `reranker_model` | Sentence-Transformers cross encoder name | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` |
| `reranker_batch_size` | Batch size used by the CrossEncoder reranker | `32` |
| `chunk_size` | Maximum characters per chunk in DatasetPreparer | `500` |
| `chunk_overlap` | Overlap between chunks in DatasetPreparer | `50` |

Update the YAML and restart processes to pick up new settings.

---

## Testing

```bash
pytest tests/ -v

# focus on a stage
pytest tests/test_embeddings/test_clip_embedder.py -v
pytest tests/test_retrieval/test_hybrid_retriever.py -k hybrid
```

Highlights:
- `tests/test_embeddings/test_clip_embedder.py` exercises batching, caching identity, alpha fusion, and CLI quick checks.
- `tests/test_indexing/test_index_builder.py` validates the dataset→index bridge (now handled inside `RAGPipeline.build_indices`).
- `tests/test_retrieval/test_hybrid_retriever.py` covers BM25 scores, dense fallback, normalization, and metadata filters.
- `tests/test_pipeline/` ensures the full RAG loop produces deterministic outputs.

See `tests/README.md` for a domain-by-domain overview of the suite.

---

## Documentation Suite

- `docs/ARCHITECTURE.md` — Deep dive into the end-to-end RAG system (mermaid diagrams, fusion math, caching strategy).
- `docs/SCHEMA_DOCUMENTATION.md` — Contract for character/location/relationship JSON payloads.
- `tests/README.md` — How the test suite is organized by domain and how to run subsets.

---

## Troubleshooting Tips

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: clip` | `pip install 'clip @ git+https://github.com/openai/CLIP.git'` or `pip install open_clip_torch` |
| `BM25 index not loaded` warning | Run `pipeline.build_indices(..., rebuild=True)` to persist `data/bm25_index.pkl`. |
| Chroma collection errors | Remove `data/chroma_store/` and rebuild indices. |
| Slow reranking | Reduce `cfg["top_k_retrieval"]` or switch to a lighter `reranker_model`. |