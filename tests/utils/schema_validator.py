"""
Schema Validation Utilities
Shared validation functions for both tests and scripts.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from jsonschema import Draft7Validator
from jsonschema.validators import RefResolver


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_local_resolver(schema_path: Path, schema: Dict) -> RefResolver:
    """Create a resolver for local schema references."""
    schemas_dir = schema_path.parent.resolve()
    base_uri = schemas_dir.as_uri() + "/"
    store: Dict[str, Dict] = {}
    ref_path = schemas_dir / "metadata_schema.json"
    if ref_path.exists():
        store[ref_path.resolve().as_uri()] = load_json(ref_path)
    return RefResolver(base_uri=base_uri, referrer=schema, store=store)


def validate_file(schema_path: Path, data_path: Path) -> List[str]:
    """
    Validate a single JSON file against a schema.

    Args:
        schema_path: Path to JSON Schema file
        data_path: Path to JSON data file

    Returns:
        List of validation error messages (empty if valid)
    """
    schema = load_json(schema_path)
    resolver = create_local_resolver(schema_path, schema)
    validator = Draft7Validator(schema, resolver=resolver)
    data = load_json(data_path)
    errors: List[str] = []
    for err in validator.iter_errors(data):
        path = "/".join(map(str, err.path))
        loc = f" at {path}" if path else ""
        errors.append(f"{err.message}{loc}")
    return errors


def validate_dir(
    schema_path: Path, data_dir: Path
) -> Tuple[int, int, Dict[str, List[str]]]:
    """
    Validate all JSON files in a directory against a schema.

    Args:
        schema_path: Path to JSON Schema file
        data_dir: Path to directory containing JSON files

    Returns:
        Tuple of (total_files, total_errors, errors_by_file_dict)
    """
    files = sorted(data_dir.glob("*.json"))
    errors_by_file: Dict[str, List[str]] = {}
    total_errors = 0
    for f in files:
        errs = validate_file(schema_path, f)
        if errs:
            errors_by_file[f.name] = errs
            total_errors += len(errs)
    return len(files), total_errors, errors_by_file
