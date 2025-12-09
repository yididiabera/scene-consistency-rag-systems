"""
EntityExtractor
---------------
Extracts characters, locations, and categories from a free-text prompt.

This module uses deterministic lookup tables built from your existing
JSON entity files. It uses compiled Regex to ensure exact word matching
(avoiding partial matches like 'Dan' inside 'Dancing').
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import json
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Dataclasses

@dataclass
class EntityExtractionResult:
    characters: List[str]
    locations: List[str]
    categories: List[str]

class EntityExtractor:
    """
    Loads character + location registries and performs deterministic
    regex-based lookup.
    """

    def __init__(
        self,
        characters_dir: str = "data/characters",
        locations_dir: str = "data/locations"
    ):
        self.characters_dir = Path(characters_dir)
        self.locations_dir = Path(locations_dir)

        # Build maps: { "name": "id", "alias": "id" }
        self.char_map = self._build_lookup_map(self.characters_dir)
        self.loc_map = self._build_lookup_map(self.locations_dir)

        # Pre-compile regex patterns for O(1) matching speed
        self.char_regex = self._compile_regex(self.char_map.keys())
        self.loc_regex = self._compile_regex(self.loc_map.keys())

    def _build_lookup_map(self, directory: Path) -> Dict[str, str]:
        """
        Scans JSON files and builds a flat map of:
        "lowercase name" -> "entity_id"
        "lowercase alias" -> "entity_id"
        """
        lookup = {}
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return lookup

        for file in directory.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle both character_id and location_id field names
                entity_id = data.get("character_id") or data.get("location_id") or data.get("entity_id")
                name = data.get("name")
                aliases = data.get("aliases", []) # Optional field

                if not entity_id or not name:
                    continue

                # Index the canonical name
                lookup[name.lower().strip()] = entity_id

                # Index any aliases
                for alias in aliases:
                    if alias:
                        lookup[alias.lower().strip()] = entity_id

            except Exception as e:
                logger.error(f"Error loading {file}: {e}", exc_info=True)

        return lookup

    def _compile_regex(self, terms: Set[str]) -> Optional[re.Pattern]:
        """
        Creates a single optimized regex pattern for all terms.
        Pattern: \b(term1|term2|term3)\b
        Sorted by length desc to ensure "Isaac Smith" matches before "Isaac".
        """
        if not terms:
            return None

        # Sort by length descending to match longest phrases first
        sorted_terms = sorted(terms, key=len, reverse=True)

        # Escape special regex characters in names
        escaped_terms = [re.escape(t) for t in sorted_terms]

        # Join with OR (|) and wrap in word boundaries (\b)
        pattern_str = r'\b(' + '|'.join(escaped_terms) + r')\b'
        return re.compile(pattern_str, re.IGNORECASE)


    def extract(self, text: str) -> EntityExtractionResult:
        """
        Scans text using pre-compiled regex.
        Returns deduplicated lists of IDs.
        """
        # Characters
        found_char_ids = set()
        if self.char_regex:
            matches = self.char_regex.findall(text)
            for m in matches:
                # Look up the ID using the lowercase match
                found_char_ids.add(self.char_map[m.lower()])

        # Locations
        found_loc_ids = set()
        if self.loc_regex:
            matches = self.loc_regex.findall(text)
            for m in matches:
                found_loc_ids.add(self.loc_map[m.lower()])

        return EntityExtractionResult(
            characters=list(found_char_ids),
            locations=list(found_loc_ids),
            categories=[]
        )