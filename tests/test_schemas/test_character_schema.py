"""
Character Schema Validation Tests
Tests for validating character JSON files against character schema.
"""

from pathlib import Path

from rich.console import Console

from tests.utils.schema_validator import validate_dir, load_json
from tests.conftest import register_checklist_item

console = Console()


def test_character_files_validate_against_schema():
    """Test that all character JSON files validate against character schema."""
    total, total_errors, errors_by_file = validate_dir(
        Path("schemas/character_schema.json"), Path("data/characters")
    )

    if total_errors > 0:
        console.print(
            f"[red]✗[/red] {total_errors} validation error(s) in {len(errors_by_file)} file(s)"
        )
        for fname, errs in errors_by_file.items():
            console.print(f"  [red]{fname}:[/red]")
            for err in errs:
                console.print(f"    • {err}")

    # Register result BEFORE assertions so failures are recorded
    register_checklist_item(
        "stage1", "schemas_validate", total_errors == 0 and total > 0
    )
    
    # Now assert (will fail test if errors found)
    assert (
        total_errors == 0
    ), f"Found {total_errors} validation errors in character files"
    assert total > 0, "No character files found"


def test_character_schema_required_fields():
    """Test that character files contain all required fields."""
    characters = [load_json(p) for p in sorted(Path("data/characters").glob("*.json"))]

    required_fields = [
        "character_id",
        "name",
        "canonical_image_path",
        "lora_trigger_word",
        "appearance",
    ]

    for char in characters:
        missing = [field for field in required_fields if field not in char]
        assert (
            not missing
        ), f"Character {char.get('character_id', 'unknown')} missing fields: {missing}"


def test_character_id_format():
    """Test that character IDs follow the correct format."""
    characters = [load_json(p) for p in sorted(Path("data/characters").glob("*.json"))]

    import re

    pattern = r"^char_[a-z_]+_\d{3}$"

    for char in characters:
        char_id = char.get("character_id", "")
        assert re.match(pattern, char_id), f"Invalid character_id format: {char_id}"


def test_character_entity_version():
    """Test that characters have entity_version field."""
    characters = [load_json(p) for p in sorted(Path("data/characters").glob("*.json"))]

    for char in characters:
        version = char.get("entity_version")
        assert isinstance(
            version, int
        ), f"entity_version must be integer, got {type(version)}"
        assert version >= 1, f"entity_version must be >= 1, got {version}"

    register_checklist_item("stage1", "entity_version", True)
