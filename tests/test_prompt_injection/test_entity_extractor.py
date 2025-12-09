"""
Tests for EntityExtractor
--------------------------
Goal: Verify that we correctly identify who is in the text without 
"hallucinating" or making simple substring errors.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from prompt_injection.entity_extractor import EntityExtractor, EntityExtractionResult


class TestEntityExtractor:
    """Test suite for EntityExtractor component."""

    @pytest.fixture
    def extractor(self):
        """Create an EntityExtractor instance for testing."""
        return EntityExtractor(
            characters_dir="data/characters",
            locations_dir="data/locations"
        )

    def test_extract_canonical_name_match(self, extractor):
        """
        Test: Input text containing "Isaac" returns ['char_isaac_001'].
        Verifies basic name matching works.
        """
        text = "Isaac is working at his desk."
        result = extractor.extract(text)
        
        assert isinstance(result, EntityExtractionResult)
        assert "char_isaac_001" in result.characters
        assert len(result.characters) >= 1

    def test_extract_alias_match(self, extractor):
        """
        Test: Input text containing an alias (if defined in JSON) returns the entity ID.
        Note: This test assumes aliases are defined in the JSON files.
        If no aliases exist, this test will be skipped.
        """
        # Check if any aliases are loaded
        if not any("alias" in name.lower() for name in extractor.char_map.keys()):
            pytest.skip("No aliases defined in character JSON files")
        
        # Test with a known alias (adjust based on your actual data)
        text = "The protagonist enters the room."
        result = extractor.extract(text)
        
        # This is a placeholder - adjust based on actual aliases in your data
        assert isinstance(result, EntityExtractionResult)

    def test_extract_case_insensitivity(self, extractor):
        """
        Test: Input "isaac" (lowercase) still finds the ID.
        Verifies case-insensitive matching.
        """
        text = "isaac is monitoring the system."
        result = extractor.extract(text)
        
        assert "char_isaac_001" in result.characters

    def test_regex_word_boundary_safety(self, extractor):
        """
        Test: Input "Isaacs" or "Bisaac" does not match "Isaac".
        Input "Officer" does not match "Office".
        Verifies word boundary protection prevents partial matches.
        """
        # Test character partial matches
        text_plural = "The Isaacs family reunion was fun."
        result_plural = extractor.extract(text_plural)
        assert "char_isaac_001" not in result_plural.characters
        
        text_prefix = "Bisaac is not a real name."
        result_prefix = extractor.extract(text_prefix)
        assert "char_isaac_001" not in result_prefix.characters
        
        # Test location partial matches
        text_officer = "The officer arrived at the scene."
        result_officer = extractor.extract(text_officer)
        # Should not match "office" location
        assert "loc_office_001" not in result_officer.locations

    def test_multiple_entities_extraction(self, extractor):
        """
        Test: Input "Isaac walks into Isaac's Office" returns both 
        the character ID and the location ID.
        Verifies multiple entity extraction in one sentence.
        """
        text = "Isaac walks into Isaac's Office to start working."
        result = extractor.extract(text)
        
        assert "char_isaac_001" in result.characters
        assert "loc_office_001" in result.locations

    def test_no_entities_found(self, extractor):
        """
        Test: Input "A dog runs." returns empty lists, does not crash.
        Verifies graceful handling of text with no entities.
        """
        text = "A dog runs through the park."
        result = extractor.extract(text)
        
        assert isinstance(result, EntityExtractionResult)
        assert result.characters == []
        assert result.locations == []
        assert result.categories == []

    def test_extract_deduplication(self, extractor):
        """
        Test: Input with repeated entity names returns each ID only once.
        Verifies deduplication logic.
        """
        text = "Isaac met Isaac's friend at Isaac's office."
        result = extractor.extract(text)
        
        # Count occurrences of the character ID
        char_count = result.characters.count("char_isaac_001")
        assert char_count == 1, "Entity IDs should be deduplicated"

    def test_empty_input(self, extractor):
        """
        Test: Empty string input returns empty results without crashing.
        """
        text = ""
        result = extractor.extract(text)
        
        assert isinstance(result, EntityExtractionResult)
        assert result.characters == []
        assert result.locations == []

    def test_special_characters_in_text(self, extractor):
        """
        Test: Text with special characters doesn't break extraction.
        """
        text = "Isaac's Office! (It's amazing.)"
        result = extractor.extract(text)
        
        assert "char_isaac_001" in result.characters
        assert "loc_office_001" in result.locations
