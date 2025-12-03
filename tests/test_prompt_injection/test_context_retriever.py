"""
Tests for ContextRetriever
---------------------------
Goal: Verify that we are asking the RAG system the right questions 
(using full context, not keywords).
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from prompt_injection.context_retriever import ContextRetriever, RetrievedContext
from prompt_injection.entity_extractor import EntityExtractionResult


class TestContextRetriever:
    """Test suite for ContextRetriever component."""

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create a mock RAGPipeline for testing."""
        mock = Mock()
        mock.query = Mock(return_value=[
            {
                "entity_id": "char_isaac_001",
                "document": "Isaac wears a white t-shirt and has short brown hair.",
                "hybrid_score": 0.95,
                "rerank_score": 0.98
            }
        ])
        return mock

    @pytest.fixture
    def retriever(self, mock_rag_pipeline):
        """Create a ContextRetriever instance with mocked pipeline."""
        return ContextRetriever(rag_pipeline=mock_rag_pipeline)

    @pytest.fixture
    def sample_entities(self):
        """Create sample extracted entities."""
        return EntityExtractionResult(
            characters=["char_isaac_001"],
            locations=["loc_office_001"],
            categories=[]
        )

    def test_retrieve_uses_full_prompt_as_query(self, retriever, sample_entities, mock_rag_pipeline):
        """
        Test: Mock the RAGPipeline. Assert that rag.query(query_text=...) 
        received the full original sentence, not just the entity name.
        """
        full_prompt = "Isaac monitors display lines of code scrolling across his office monitor"
        
        retriever.retrieve(full_prompt, sample_entities)
        
        # Verify the query was called with the full prompt
        mock_rag_pipeline.query.assert_called()
        call_args = mock_rag_pipeline.query.call_args_list[0]
        
        assert call_args[1]["query_text"] == full_prompt, \
            "Should use full prompt, not just entity name"

    def test_retrieve_applies_strict_id_filter(self, retriever, sample_entities, mock_rag_pipeline):
        """
        Test: Assert that rag.query(where=...) received {'entity_id': 'char_isaac_001'}.
        Verifies strict entity filtering.
        """
        full_prompt = "Isaac is working in his office"
        
        retriever.retrieve(full_prompt, sample_entities)
        
        # Check the first call (for character)
        call_args = mock_rag_pipeline.query.call_args_list[0]
        assert call_args[1]["where"] == {"entity_id": "char_isaac_001"}, \
            "Should apply strict entity_id filter"

    def test_retrieve_handles_rag_pipeline_returning_empty(self, sample_entities):
        """
        Test: If the RAG system finds no chunks, the dictionary returned 
        should be empty (or contain empty lists), but not crash.
        """
        # Create a mock that returns empty results
        mock_rag = Mock()
        mock_rag.query = Mock(return_value=[])
        
        retriever = ContextRetriever(rag_pipeline=mock_rag)
        result = retriever.retrieve("Some text", sample_entities)
        
        assert isinstance(result, RetrievedContext)
        assert result.character_context == {}
        assert result.location_context == {}

    def test_retrieve_groups_chunks_by_entity(self, mock_rag_pipeline):
        """
        Test: If multiple chunks are returned for Isaac, ensure they are 
        stored under his specific ID key in the result object.
        """
        # Configure mock to return multiple chunks
        mock_rag_pipeline.query = Mock(return_value=[
            {"entity_id": "char_isaac_001", "document": "Chunk 1"},
            {"entity_id": "char_isaac_001", "document": "Chunk 2"},
            {"entity_id": "char_isaac_001", "document": "Chunk 3"}
        ])
        
        retriever = ContextRetriever(rag_pipeline=mock_rag_pipeline)
        entities = EntityExtractionResult(
            characters=["char_isaac_001"],
            locations=[],
            categories=[]
        )
        
        result = retriever.retrieve("Test prompt", entities)
        
        assert "char_isaac_001" in result.character_context
        assert len(result.character_context["char_isaac_001"]) == 3
        assert "Chunk 1" in result.character_context["char_isaac_001"]
        assert "Chunk 2" in result.character_context["char_isaac_001"]
        assert "Chunk 3" in result.character_context["char_isaac_001"]

    def test_retrieve_handles_multiple_entities(self, mock_rag_pipeline):
        """
        Test: Multiple entities are queried separately with correct filters.
        """
        retriever = ContextRetriever(rag_pipeline=mock_rag_pipeline)
        entities = EntityExtractionResult(
            characters=["char_isaac_001", "char_gertie_001"],
            locations=["loc_office_001"],
            categories=[]
        )
        
        result = retriever.retrieve("Isaac and Gertie in the office", entities)
        
        # Should have made 3 calls (2 characters + 1 location)
        assert mock_rag_pipeline.query.call_count == 3

    def test_retrieve_uses_correct_collection_names(self, mock_rag_pipeline):
        """
        Test: Characters queried from 'characters' collection, 
        locations from 'locations' collection.
        """
        retriever = ContextRetriever(rag_pipeline=mock_rag_pipeline)
        entities = EntityExtractionResult(
            characters=["char_isaac_001"],
            locations=["loc_office_001"],
            categories=[]
        )
        
        retriever.retrieve("Test", entities)
        
        # Check collection names in calls
        calls = mock_rag_pipeline.query.call_args_list
        assert calls[0][1]["collection"] == "characters"
        assert calls[1][1]["collection"] == "locations"

    def test_retrieve_extracts_document_field(self, mock_rag_pipeline):
        """
        Test: Correctly extracts 'document' field from RAG results.
        """
        mock_rag_pipeline.query = Mock(return_value=[
            {"entity_id": "char_isaac_001", "document": "Test content", "score": 0.9}
        ])
        
        retriever = ContextRetriever(rag_pipeline=mock_rag_pipeline)
        entities = EntityExtractionResult(characters=["char_isaac_001"], locations=[], categories=[])
        
        result = retriever.retrieve("Test", entities)
        
        assert result.character_context["char_isaac_001"] == ["Test content"]

    def test_retrieve_skips_results_without_document_field(self, mock_rag_pipeline):
        """
        Test: Results without 'document' field are skipped gracefully.
        """
        mock_rag_pipeline.query = Mock(return_value=[
            {"entity_id": "char_isaac_001", "score": 0.9},  # No 'document' field
            {"entity_id": "char_isaac_001", "document": "Valid content"}
        ])
        
        retriever = ContextRetriever(rag_pipeline=mock_rag_pipeline)
        entities = EntityExtractionResult(characters=["char_isaac_001"], locations=[], categories=[])
        
        result = retriever.retrieve("Test", entities)
        
        # Should only have the valid content
        assert len(result.character_context["char_isaac_001"]) == 1
        assert result.character_context["char_isaac_001"][0] == "Valid content"
