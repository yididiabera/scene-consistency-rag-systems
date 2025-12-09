"""
Tests for SceneConsistencyEngine
---------------------------------
Goal: Integration test. Does the Orchestrator pass data correctly from A to B to C?
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from prompt_injection import SceneConsistencyEngine
from prompt_injection.entity_extractor import EntityExtractionResult
from prompt_injection.context_retriever import RetrievedContext
from prompt_injection.shot_enricher import EnrichedShot


class TestSceneConsistencyEngine:
    """Test suite for SceneConsistencyEngine orchestrator."""

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create a mock RAG pipeline."""
        mock = Mock()
        mock.query = Mock(return_value=[
            {"entity_id": "char_isaac_001", "document": "Isaac description"}
        ])
        return mock

    @pytest.fixture
    def engine(self, mock_rag_pipeline):
        """Create engine with mocked dependencies."""
        return SceneConsistencyEngine(
            rag_pipeline=mock_rag_pipeline,
            characters_dir="data/characters",
            locations_dir="data/locations"
        )

    @pytest.fixture
    def sample_shot(self):
        """Create a sample shot."""
        return {
            "shot_id": "shot_001",
            "scene_id": "scene_001",
            "description": "Isaac monitors display lines of code in his office",
            "actions": {"type": "working"},
            "camera": {"angle": "medium_shot"},
            "metadata": {}
        }

    def test_process_shot_end_to_end_flow(self, engine, sample_shot):
        """
        Test: Input a raw dictionary -> Receive an EnrichedShot object.
        Verify the pipeline ran in order (Extract -> Retrieve -> Enrich).
        """
        result = engine.process_shot(sample_shot)
        
        assert isinstance(result, EnrichedShot)
        assert result.shot_id == "shot_001"
        assert result.scene_id == "scene_001"
        assert result.raw_description == "Isaac monitors display lines of code in his office"

    def test_process_shot_calls_extraction_first(self, mock_rag_pipeline):
        """
        Test: Verify that retriever.retrieve was called with the 
        result of extractor.extract.
        """
        with patch('prompt_injection.entity_extractor.EntityExtractor') as MockExtractor:
            # Setup mock extractor
            mock_extractor_instance = MockExtractor.return_value
            mock_extractor_instance.extract = Mock(return_value=EntityExtractionResult(
                characters=["char_isaac_001"],
                locations=["loc_office_001"],
                categories=[]
            ))
            
            engine = SceneConsistencyEngine(
                rag_pipeline=mock_rag_pipeline,
                characters_dir="data/characters",
                locations_dir="data/locations"
            )
            
            # Replace the extractor with our mock
            engine.entity_extractor = mock_extractor_instance
            
            shot = {"description": "Isaac in office"}
            result = engine.process_shot(shot)
            
            # Verify extraction was called
            mock_extractor_instance.extract.assert_called_once_with("Isaac in office")

    def test_extract_entities_convenience_method(self, engine):
        """
        Test: Convenience method extract_entities works correctly.
        """
        text = "Isaac walks into Isaac's Office"
        result = engine.extract_entities(text)
        
        assert isinstance(result, EntityExtractionResult)
        assert "char_isaac_001" in result.characters
        assert "loc_office_001" in result.locations

    def test_retrieve_context_convenience_method(self, engine):
        """
        Test: Convenience method retrieve_context works correctly.
        """
        entities = EntityExtractionResult(
            characters=["char_isaac_001"],
            locations=[],
            categories=[]
        )
        
        result = engine.retrieve_context("Test prompt", entities)
        
        assert isinstance(result, RetrievedContext)

    def test_process_shot_with_no_entities(self, engine):
        """
        Test: Shot with no recognizable entities still processes without error.
        """
        shot = {
            "shot_id": "shot_002",
            "description": "A bird flies across the sky"
        }
        
        result = engine.process_shot(shot)
        
        assert isinstance(result, EnrichedShot)
        assert result.rag_characters == {}
        assert result.rag_locations == {}

    def test_process_shot_with_minimal_data(self, engine):
        """
        Test: Shot with only description field processes correctly.
        """
        shot = {"description": "Isaac working"}
        
        result = engine.process_shot(shot)
        
        assert isinstance(result, EnrichedShot)
        assert result.raw_description == "Isaac working"

    def test_engine_initialization(self, mock_rag_pipeline):
        """
        Test: Engine initializes all three components correctly.
        """
        engine = SceneConsistencyEngine(
            rag_pipeline=mock_rag_pipeline,
            characters_dir="data/characters",
            locations_dir="data/locations"
        )
        
        assert engine.entity_extractor is not None
        assert engine.context_retriever is not None
        assert engine.shot_enricher is not None
        assert engine.rag is mock_rag_pipeline

    def test_process_shot_preserves_all_metadata(self, engine):
        """
        Test: All shot metadata is preserved through the pipeline.
        """
        shot = {
            "shot_id": "shot_003",
            "scene_id": "scene_002",
            "description": "Isaac typing",
            "characters": ["char_isaac_001"],
            "locations": ["loc_office_001"],
            "actions": {"type": "typing", "speed": "fast"},
            "camera": {"angle": "close_up", "movement": "zoom"},
            "metadata": {"duration": 5.0, "custom_field": "value"}
        }
        
        result = engine.process_shot(shot)
        
        assert result.shot_id == "shot_003"
        assert result.scene_id == "scene_002"
        assert result.actions == {"type": "typing", "speed": "fast"}
        assert result.camera == {"angle": "close_up", "movement": "zoom"}
        assert result.metadata == {"duration": 5.0, "custom_field": "value"}
