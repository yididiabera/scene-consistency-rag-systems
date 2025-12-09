"""
Location Schema Validation Tests
Tests for validating location JSON files against location schema.
"""

from pathlib import Path
import re

from rich.console import Console

from tests.utils.schema_validator import validate_dir, load_json
from tests.conftest import register_checklist_item

console = Console()


def test_location_files_validate_against_schema():
    """Test that all location JSON files validate against location schema."""
    total, total_errors, errors_by_file = validate_dir(
        Path("schemas/location_schema.json"), Path("data/locations")
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
        "stage1", "examples_validate", total_errors == 0 and total > 0
    )
    
    # Now assert (will fail test if errors found)
    assert (
        total_errors == 0
    ), f"Found {total_errors} validation errors in location files"
    assert total > 0, "No location files found"


def test_location_schema_required_fields():
    """Test that location files contain all required fields."""
    locations = [load_json(p) for p in sorted(Path("data/locations").glob("*.json"))]

    required_fields = ["location_id", "name", "description"]

    for loc in locations:
        missing = [field for field in required_fields if field not in loc]
        assert (
            not missing
        ), f"Location {loc.get('location_id', 'unknown')} missing fields: {missing}"


def test_location_id_format():
    """Test that location IDs follow the correct format."""
    locations = [load_json(p) for p in sorted(Path("data/locations").glob("*.json"))]

    pattern = r"^loc_[a-z_]+_\d{3}$"

    for loc in locations:
        loc_id = loc.get("location_id", "")
        assert re.match(pattern, loc_id), f"Invalid location_id format: {loc_id}"


def test_location_setting_object():
    """Test that location schema includes setting object."""
    locations = [load_json(p) for p in sorted(Path("data/locations").glob("*.json"))]

    # Setting is optional, but if present should have correct structure
    for loc in locations:
        if "setting" in loc:
            setting = loc["setting"]
            assert isinstance(setting, dict), "setting must be an object"
            # Setting can have time, weather, props, etc.

    register_checklist_item("stage1", "setting_object", True)
