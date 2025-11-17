#!/usr/bin/env python3
"""
Schema Validation Script
Validates JSON files against JSON Schema definitions.

Usage:
    python scripts/validate_schemas.py --schema schemas/character_schema.json --data examples/character_example.json
    python scripts/validate_schemas.py --schema schemas/character_schema.json --data data/characters
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

# Import shared validation functions
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).parent.parent))
from tests.utils.schema_validator import validate_file, validate_dir

console = Console()


def print_summary(
    title: str, total: int, total_errors: int, errors_by_file: Dict[str, List[str]]
):
    """Print validation summary."""
    console.rule(f"[bold cyan]{title}")
    if total == 0:
        console.print("[yellow]No JSON files found[/yellow]")
        return
    if total_errors == 0:
        console.print(f"[green]✓[/green] All {total} file(s) valid")
        return
    console.print(
        f"[red]✗[/red] {total_errors} error(s) found across {len(errors_by_file)} file(s)"
    )
    table = Table(title="Validation Errors")
    table.add_column("File", style="cyan")
    table.add_column("Error", style="red")
    for fname, errs in errors_by_file.items():
        for e in errs:
            table.add_row(fname, e)
    console.print(table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate JSON files against JSON Schema definitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single file
  python scripts/validate_schemas.py --schema schemas/character_schema.json --data examples/character_example.json

  # Validate all files in a directory
  python scripts/validate_schemas.py --schema schemas/character_schema.json --data data/characters

  # Validate location data
  python scripts/validate_schemas.py --schema schemas/location_schema.json --data data/locations
        """,
    )
    parser.add_argument(
        "--schema", type=str, required=True, help="Path to JSON Schema file"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to JSON data file or directory containing JSON files",
    )

    args = parser.parse_args()

    schema_path = Path(args.schema)
    data_path = Path(args.data)

    if not schema_path.exists():
        console.print(f"[red]✗[/red] Schema file not found: {schema_path}")
        return 1

    if not data_path.exists():
        console.print(f"[red]✗[/red] Data path not found: {data_path}")
        return 1

    if data_path.is_file():
        # Validate single file
        errors = validate_file(schema_path, data_path)
        if errors:
            console.print(f"[red]✗[/red] Validation failed for {data_path.name}")
            for err in errors:
                console.print(f"  [red]•[/red] {err}")
            return 1
        else:
            console.print(f"[green]✓[/green] {data_path.name} is valid")
            return 0
    elif data_path.is_dir():
        # Validate directory
        total, total_errors, errors_by_file = validate_dir(schema_path, data_path)
        print_summary(f"Validation: {data_path}", total, total_errors, errors_by_file)
        return 0 if total_errors == 0 else 1
    else:
        console.print(f"[red]✗[/red] Invalid data path: {data_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
