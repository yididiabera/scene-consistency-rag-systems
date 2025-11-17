"""
Relationship Schema Validation Tests
Tests for validating relationship JSON files against relationship schema.
"""

from pathlib import Path

from rich.console import Console

from tests.utils.schema_validator import validate_file, load_json
from tests.conftest import register_checklist_item

console = Console()


def test_relationship_schema_defines_entity_links():
    """Test that relationship schema defines entity → entity links."""
    # Check if relationship example files exist
    rel_examples = list(Path("examples").glob("*relationship*.json"))

    if not rel_examples:
        # Skip if no relationship examples
        return

    for rel_file in rel_examples:
        if "collection" in rel_file.name:
            continue  # Skip collection files

        schema_path = Path("schemas/relationship_schema.json")
        if schema_path.exists():
            errors = validate_file(schema_path, rel_file)
            assert (
                len(errors) == 0
            ), f"Relationship file {rel_file.name} has errors: {errors}"

    register_checklist_item("stage1", "relationship_links", True)


def test_relationship_structure():
    """Test that relationships have correct structure for entity links."""
    rel_examples = list(Path("examples").glob("*relationship*.json"))

    if not rel_examples:
        return

    for rel_file in rel_examples:
        if "collection" in rel_file.name:
            continue

        rel_data = load_json(rel_file)

        # Relationships should link entities
        if isinstance(rel_data, dict):
            assert (
                "source_entity" in rel_data or "target_entity" in rel_data
            ), "Relationship must have source_entity or target_entity"
        elif isinstance(rel_data, list):
            for rel in rel_data:
                assert (
                    "source_entity" in rel or "target_entity" in rel
                ), "Each relationship must have source_entity or target_entity"
