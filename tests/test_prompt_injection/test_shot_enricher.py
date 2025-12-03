"""
Tests for ShotEnricher
----------------------
Goal: Verify that we successfully merge the RAG data into the Shot object.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from prompt_injection.shot_enricher import ShotEnricher, EnrichedShot
from prompt_injection.context_retriever import RetrievedContext


class TestShotEnricher:
    """Test suite for ShotEnricher component."""

    @pytest.fixture
    def enricher(self):
        """Create a ShotEnricher instance."""
        return ShotEnricher()

    @pytest.fixture
    def sample_shot(self):
        """Create a sample shot dictionary."""
        return {
            "shot_id": "shot_001",
            "scene_id": "scene_001",
            "description": "Isaac monitors display lines of code",
            "characters": ["char_isaac_001"],
            "locations": ["loc_office_001"],
            "actions": {"type": "working", "intensity": "focused"},
            "camera": {"angle": "medium_shot", "movement": "static"},
            "metadata": {"duration": 3.5, "timestamp": "00:01:23"}
        }

    @pytest.fixture
    def sample_context(self):
        """Create sample retrieved context."""
        return RetrievedContext(
            character_context={
                "char_isaac_001": [
                    "Isaac wears a white t-shirt.",
                    "He has short brown hair."
                ]
            },
            location_context={
                "loc_office_001": [
                    "The office has wooden shelves.",
                    "Large window with city view."
                ]
            }
        )

    def test_enrich_preserves_original_shot_metadata(self, enricher, sample_shot, sample_context):
        """
        Test: Ensure shot_id, camera, and actions from the input 
        are present in the output EnrichedShot.
        """
        result = enricher.enrich(sample_shot, sample_context)
        
        assert isinstance(result, EnrichedShot)
        assert result.shot_id == "shot_001"
        assert result.scene_id == "scene_001"
        assert result.actions == {"type": "working", "intensity": "focused"}
        assert result.camera == {"angle": "medium_shot", "movement": "static"}
        assert result.metadata == {"duration": 3.5, "timestamp": "00:01:23"}

    def test_enrich_flattens_text_chunks(self, enricher, sample_shot, sample_context):
        """
        Test: If ContextRetriever returned ["Chunk 1", "Chunk 2"], 
        the EnrichedShot should contain the single string "Chunk 1 Chunk 2".
        """
        result = enricher.enrich(sample_shot, sample_context)
        
        # Check character context is flattened
        assert "char_isaac_001" in result.rag_characters
        char_text = result.rag_characters["char_isaac_001"]
        assert "Isaac wears a white t-shirt. He has short brown hair." == char_text
        
        # Check location context is flattened
        assert "loc_office_001" in result.rag_locations
        loc_text = result.rag_locations["loc_office_001"]
        assert "The office has wooden shelves. Large window with city view." == loc_text

    def test_enrich_handles_missing_context_gracefully(self, enricher, sample_shot):
        """
        Test: If the RetrievedContext is empty, the EnrichedShot should 
        simply have empty dictionaries for rag_characters, without error.
        """
        empty_context = RetrievedContext(
            character_context={},
            location_context={}
        )
        
        result = enricher.enrich(sample_shot, empty_context)
        
        assert isinstance(result, EnrichedShot)
        assert result.rag_characters == {}
        assert result.rag_locations == {}
        assert result.shot_id == "shot_001"  # Other fields still present

    def test_enrich_handles_missing_shot_fields(self, enricher, sample_context):
        """
        Test: Shot with missing optional fields doesn't crash.
        """
        minimal_shot = {
            "description": "A simple shot"
        }
        
        result = enricher.enrich(minimal_shot, sample_context)
        
        assert isinstance(result, EnrichedShot)
        assert result.shot_id is None
        assert result.scene_id is None
        assert result.characters == []
        assert result.locations == []
        assert result.actions is None
        assert result.camera is None
        assert result.metadata == {}

    def test_enrich_preserves_raw_description(self, enricher, sample_shot, sample_context):
        """
        Test: Original description is preserved in raw_description field.
        """
        result = enricher.enrich(sample_shot, sample_context)
        
        assert result.raw_description == "Isaac monitors display lines of code"

    def test_enrich_handles_single_chunk(self, enricher, sample_shot):
        """
        Test: Single chunk (not a list) is handled correctly.
        """
        single_chunk_context = RetrievedContext(
            character_context={"char_isaac_001": ["Single chunk"]},
            location_context={}
        )
        
        result = enricher.enrich(sample_shot, single_chunk_context)
        
        assert result.rag_characters["char_isaac_001"] == "Single chunk"

    def test_enrich_strips_whitespace(self, enricher, sample_shot):
        """
        Test: Whitespace is properly stripped when joining chunks.
        """
        context_with_whitespace = RetrievedContext(
            character_context={
                "char_isaac_001": ["  Text with spaces  ", "  Another chunk  "]
            },
            location_context={}
        )
        
        result = enricher.enrich(sample_shot, context_with_whitespace)
        
        char_text = result.rag_characters["char_isaac_001"]
        # Should be joined and stripped
        assert not char_text.startswith(" ")
        assert not char_text.endswith(" ")

    def test_enrich_maintains_character_list(self, enricher, sample_shot, sample_context):
        """
        Test: Original character list is preserved.
        """
        result = enricher.enrich(sample_shot, sample_context)
        
        assert result.characters == ["char_isaac_001"]

    def test_enrich_maintains_location_list(self, enricher, sample_shot, sample_context):
        """
        Test: Original location list is preserved.
        """
        result = enricher.enrich(sample_shot, sample_context)
        
        assert result.locations == ["loc_office_001"]
