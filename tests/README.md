# Test Suite Documentation

## Overview

This test suite validates the Scene Consistency RAG Systems project. Tests are organized by **functional domain** rather than by stage, making them easier to find, maintain, and extend. Each test domain corresponds to a major component of the system architecture.

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures and Rich checklist summaries
├── test_schemas/                        # Schema validation tests
│   ├── test_character_schema.py
│   ├── test_location_schema.py
│   ├── test_relationship_schema.py
│   └── test_metadata_schema.py
├── test_dataset/                        # Dataset preparation tests
│   └── test_dataset_preparer_basic.py
├── test_embeddings/                     # Embedding tests
│   ├── test_clip_embedder.py
│   └── test_embedding_infrastructure.py
├── test_indexing/                       # Index building tests
│   └── test_index_builder.py
├── test_retrieval/                      # Retrieval tests
│   └── test_hybrid_retriever.py
├── test_reranking/                      # Reranking tests
│   └── test_cross_encoder_reranker.py
├── test_pipeline/                       # Pipeline integration tests
│   ├── test_rag_pipeline_end_to_end.py
│   └── test_prompt_builder.py
├── test_utils/                          # Utility tests
└── utils/                               # Test utilities
    ├── schema_validator.py
    └── index_inputs.py
```

## What We Are Testing

### test_schemas/ — Schema Validation

**Purpose**: Validate JSON schemas and ensure data conforms to expected structure.

**What we test**:
- Character schema validation
- Location schema validation
- Relationship schema validation
- Metadata schema validation
- ID format compliance
- Required fields presence
- Entity versioning

**What we expect**:
- All JSON files validate against their schemas
- No validation errors
- Proper ID formats (`char_*_###`, `loc_*_###`)
- Valid metadata with timestamps and version numbers
- All required fields present

**Key tests**:
- `test_character_files_validate_against_schema()` - Validates all character JSONs
- `test_location_files_validate_against_schema()` - Validates all location JSONs
- `test_metadata_applied_in_all_entities()` - Validates metadata structure

---

### test_dataset/ — Dataset Preparation

**Purpose**: Test dataset loading, chunking, and BM25 preprocessing.

**What we test**:
- Entity loading from directories
- Text chunking (deterministic, no empties)
- Chunk ID generation (unique, sequential)
- BM25 preprocessing (stopword removal, punctuation)
- BM25 index building and persistence
- Output structure validation

**What we expect**:
- All entities parsed without errors
- Chunks are consistent and deterministic
- Chunk IDs are unique and sequential
- Text cleaning removes stopwords and punctuation
- No empty chunks
- BM25 index saved and loadable

**Key tests**:
- `test_dataset_loading()` - Loads characters and locations
- `test_chunk_extraction()` - Tests chunking logic
- `test_bm25_preprocess()` - Tests text preprocessing
- `test_bm25_index_build()` - Tests BM25 index construction

---

### test_embeddings/ — Embedding Generation

**Purpose**: Test CLIP embedding generation, fusion, and caching.

**What we test**:
- Singleton model loading (CLIP loads once)
- Batch text embedding
- Batch image embedding
- Fusion logic (weighted average + L2 normalization)
- Caching (in-memory and disk)
- Error handling for missing images
- Embedding normalization and dimensions

**What we expect**:
- Model loads once and is reused (singleton)
- Batch operations work correctly
- Cache improves performance
- Missing images handled gracefully
- All embeddings are L2-normalized
- Embeddings have correct dimensions (512)

**Key tests**:
- `test_model_loading_singleton()` - Verifies singleton pattern
- `test_text_embedding_batch()` - Tests batch text embedding
- `test_image_embedding_batch_and_errors()` - Tests image embedding with error handling
- `test_fusion_logic()` - Tests fusion formula
- `test_caching_identity()` - Tests caching behavior

---

### test_indexing/ — Index Building

**Purpose**: Test ChromaDB indexing and BM25 persistence.

**What we test**:
- ChromaDB collection setup and persistence
- Embedding upsertion and deduplication
- BM25 index construction and persistence
- `build_all_indices()` orchestrator
- Metadata persistence
- Determinism and corruption handling

**What we expect**:
- BM25 index saved and loadable
- Chroma collections created for characters & locations
- Each chunk ID appears exactly once
- Metadata fields correctly persisted
- Index directory contains expected files
- `build_all_indices()` completes without warnings

**Key tests**:
- `test_chroma_collection_setup()` - Tests ChromaDB setup
- `test_upsert_embeddings_batch()` - Tests embedding insertion
- `test_upsert_no_duplicates()` - Tests deduplication
- `test_bm25_persistence()` - Tests BM25 save/load
- `test_build_all_indices_contract()` - Tests orchestrator

---

### test_retrieval/ — Hybrid Retrieval

**Purpose**: Test sparse (BM25) and dense (vector) retrieval, plus fusion.

**What we test**:
- BM25 sparse retrieval
- Dense vector search
- Hybrid fusion scoring
- Candidate deduplication
- Metadata filtering
- Performance benchmarks
- Edge case handling

**What we expect**:
- BM25 returns valid sparse scores
- Dense retriever returns vector similarities
- Hybrid scoring = w_sparse * sparse + w_dense * dense
- No sorting bugs (deterministic order)
- Query returns relevant top-K candidates
- Edge cases handled gracefully

**Key tests**:
- `test_bm25_finds_relevant_docs()` - Tests BM25 retrieval
- `test_dense_search_self_match()` - Tests dense search
- `test_hybrid_fusion_changes_order()` - Tests hybrid fusion
- `test_metadata_filtering()` - Tests metadata filtering
- `test_performance_bm25()` - Performance benchmarks

---

### test_reranking/ — Reranking

**Purpose**: Test CrossEncoder reranking and error handling.

**What we test**:
- Batched reranking with CrossEncoder
- Singleton model loading
- Semantic order improvement
- Error handling and fallback
- Batch size handling

**What we expect**:
- Batched reranking works correctly
- Cross-encoder model loads once
- Reranked list improves semantic order
- Errors fall back to non-reranked list
- Appropriate logs on errors

**Key tests**:
- `test_batched_reranking_works()` - Tests batch reranking
- `test_cross_encoder_model_loads_once()` - Tests singleton
- `test_reranked_list_improves_semantic_order()` - Tests improvement
- `test_errors_fall_back_to_non_reranked_list()` - Tests fallback

---

### test_pipeline/ — Pipeline Integration

**Purpose**: Test end-to-end RAG pipeline and prompt building.

**What we test**:
- End-to-end pipeline build and query
- Retrieve and rerank functionality
- Context builder integration
- Character consistency anchor generation
- Malformed query handling
- Storyboard/generator integration readiness
- Prompt assembly and sanitization

**What we expect**:
- `retrieve(query)` works
- `retrieve_and_rerank(query)` works
- Context builder includes all required attributes
- Character consistency anchor generated correctly
- Malformed queries handled gracefully
- Ready for storyboard/generator integration

**Key tests**:
- `test_rag_pipeline_build_and_query()` - End-to-end test
- `test_retrieve_query()` - Tests retrieval
- `test_retrieve_and_rerank()` - Tests reranking integration
- `test_character_consistency_anchor()` - Tests anchor generation
- `test_malformed_query_handling()` - Tests error handling

---

## How to Run Tests

### Run All Tests

```bash
pytest tests/ -v
```

This will:
- Run all test files
- Display Rich checklist summaries for each component tested
- Show ✓ (green) for passed items and ✗ (red) for failed items

### Run Tests for a Specific Domain

```bash
# Schema validation
pytest tests/test_schemas/ -v

# Dataset preparation
pytest tests/test_dataset/ -v

# Embeddings
pytest tests/test_embeddings/ -v

# Indexing
pytest tests/test_indexing/ -v

# Retrieval
pytest tests/test_retrieval/ -v

# Reranking
pytest tests/test_reranking/ -v

# Pipeline
pytest tests/test_pipeline/ -v
```

### Run a Specific Test File

```bash
pytest tests/test_embeddings/test_clip_embedder.py -v
```

### Run a Specific Test Function

```bash
pytest tests/test_pipeline/test_rag_pipeline_end_to_end.py::test_rag_pipeline_build_and_query -v
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Tests Quietly (Summary Only)

```bash
pytest tests/ -q
```

## Test Output

After running tests, you'll see:

1. **Standard pytest output**: Test execution results with pass/fail status
2. **Rich checklist summaries**: Formatted tables showing checklist items with ✓/✗ marks
3. **Component panels**: Each component tested displays a panel with its checklist

Example output:
```
======================================================================
======================================================================
 Schemas & Data Standards 
======================================================================
  ✓      All JSON schemas validate correctly                   
  ✓      Example JSONs validate against schemas                
  ...
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- **`rag_pipeline`**: RAGPipeline instance for testing
- **`retriever_setup`**: Retriever with populated indices
- **`reranker_setup`**: Reranker with sample candidates
- **`context_builder`**: ContextBuilder instance
- **`prompt_assembler`**: PromptAssembler instance
- **`chroma_test_dir`**: Temporary ChromaDB directory
- **`embed_inputs`**: Sample embedding inputs

## Test Utilities

The `utils/` directory contains shared utilities:

- **`schema_validator.py`**: JSON schema validation functions
  - `validate_file()`: Validate a single JSON file
  - `validate_dir()`: Validate all JSON files in a directory
  - `load_json()`: Load JSON file with error handling

- **`index_inputs.py`**: Test data generation helpers

## Prerequisites

- Python 3.12+
- All dependencies from `requirements.txt`
- CLIP model (will download on first use)
- CrossEncoder model (will download on first use)
- Test data in `data/characters/` and `data/locations/`

## Troubleshooting

### Tests Fail with "CLIP not available"
- Install CLIP: `pip install 'clip @ git+https://github.com/openai/CLIP.git'`
- Or install OpenCLIP: `pip install open_clip_torch>=2.24.0`

### Tests Fail with "No module named 'tests'"
- Ensure you're running from project root
- `conftest.py` should add project root to `sys.path` automatically

### ChromaDB Errors
- Clear `data/chroma_store/` directory if corrupted
- Tests use temporary directories for isolation

### Missing Test Data
- Ensure `data/characters/` and `data/locations/` contain JSON files
- Character images should exist at paths specified in JSON

## Benefits of This Structure

1. **Modular**: Each component tested independently
2. **Maintainable**: Easy to find and update tests
3. **Scalable**: Add new tests without clutter
4. **Debuggable**: Failures immediately point to subsystem
5. **Intuitive**: Structure mirrors architecture

## Maintenance

When adding new features:

1. **Add tests** to the appropriate domain folder
2. **Register checklist items** using `register_checklist_item()` from `conftest.py`
3. **Update this README** if adding new test domains or major changes
4. **Run full test suite** before committing: `pytest tests/ -v`

## Test Statistics

- **Total Test Files**: 13
- **Total Tests**: ~87 (varies by data availability)
- **Coverage**: All major components and checklist items
- **Execution Time**: ~40-50 seconds for full suite

## Checklist Coverage

The Rich panels printed from `tests/conftest.py` summarize coverage for:

- **Schemas & Data Standards** (schema validation tests)
- **DatasetPreparer & Index Inputs** (dataset preparation tests)
- **Embedding Infrastructure** (ClipEmbedder and related tests)
- **Indexing & Chroma Integration** (index builder tests)
- **Hybrid Retrieval** (BM25 + dense fusion tests)
- **Reranker** (CrossEncoder-based reranking tests)
- **Context Builder** (anchor generation and sanitization tests)
- **RAG Pipeline** (end-to-end retrieval → rerank → anchor → prompt tests)

Each checklist row in these panels is backed by at least one real test that
calls `register_checklist_item(...)`, so green checks reflect real coverage
rather than defaults.

## Notes

- Tests use pytest fixtures for setup and teardown
- Some tests use monkeypatching to avoid loading heavy models
- Tests are designed to be fast and isolated
- Rich terminal summaries are provided via `pytest_terminal_summary` hooks
- Warning filters configured in `pytest.ini` suppress known deprecation warnings
