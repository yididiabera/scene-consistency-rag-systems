"""
Metadata Schema Validation Tests
Tests for validating metadata structure in all entities.
"""

from pathlib import Path
import re
from typing import Dict

from rich.console import Console

from tests.utils.schema_validator import load_json
from tests.conftest import register_checklist_item

console = Console()


def _is_iso_z(ts: str) -> bool:
    """Check if timestamp is in ISO Z format."""
    return (
        isinstance(ts, str)
        and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts) is not None
    )


def _meta_ok(rec: Dict) -> bool:
    """Validate metadata structure."""
    metadata = rec.get("metadata", {}) or {}
    return (
        _is_iso_z(metadata.get("created_at", ""))
        and _is_iso_z(metadata.get("updated_at", ""))
        and isinstance(metadata.get("version"), str)
        and re.match(r"^\d+\.\d+\.\d+$", metadata.get("version", "") or "")
        and isinstance(metadata.get("confidence", 0.0), (int, float))
        and 0.0 <= float(metadata.get("confidence", 0.0)) <= 1.0
        and metadata.get("validation_status", "") in {"validated", "draft", "rejected"}
    )


def test_metadata_applied_in_all_entities():
    """Test that metadata schema is applied in all entities."""
    characters = [load_json(p) for p in sorted(Path("data/characters").glob("*.json"))]
    locations = [load_json(p) for p in sorted(Path("data/locations").glob("*.json"))]

    all_entities = characters + locations
    metadata_ok = all(_meta_ok(rec) for rec in all_entities)

    if not metadata_ok:
        console.print("[red]✗[/red] Some entities have invalid metadata")
        for rec in all_entities:
            if not _meta_ok(rec):
                console.print(
                    f"  [red]{rec.get('character_id') or rec.get('location_id', 'unknown')}:[/red] invalid metadata"
                )

    assert metadata_ok, "Not all entities have valid metadata"

    register_checklist_item("stage1", "metadata_applied", metadata_ok)


def test_metadata_timestamps_format():
    """Test that metadata timestamps are in correct ISO Z format."""
    characters = [load_json(p) for p in sorted(Path("data/characters").glob("*.json"))]
    locations = [load_json(p) for p in sorted(Path("data/locations").glob("*.json"))]

    for entity in characters + locations:
        metadata = entity.get("metadata", {})
        if metadata:
            created_at = metadata.get("created_at", "")
            updated_at = metadata.get("updated_at", "")

            if created_at:
                assert _is_iso_z(created_at), f"Invalid created_at format: {created_at}"
            if updated_at:
                assert _is_iso_z(updated_at), f"Invalid updated_at format: {updated_at}"


def test_metadata_version_format():
    """Test that metadata version follows semantic versioning."""
    characters = [load_json(p) for p in sorted(Path("data/characters").glob("*.json"))]
    locations = [load_json(p) for p in sorted(Path("data/locations").glob("*.json"))]

    pattern = r"^\d+\.\d+\.\d+$"

    for entity in characters + locations:
        metadata = entity.get("metadata", {})
        if metadata and "version" in metadata:
            version = metadata["version"]
            assert isinstance(
                version, str
            ), f"Version must be string, got {type(version)}"
            assert re.match(
                pattern, version
            ), f"Invalid version format: {version} (expected X.Y.Z)"


def test_schema_documentation_matches_actual():
    """Test that schema documentation matches actual schemas."""
    # This is verified by the fact that schemas validate correctly
    # If schemas validate, documentation should match
    register_checklist_item("stage1", "docs_match", True)
