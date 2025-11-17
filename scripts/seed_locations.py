#!/usr/bin/env python3
"""
Seed at least one location JSON into data/locations using schema v1.1.
If a file already exists, it will be overwritten only with --force.

Usage:
  python scripts/seed_locations.py [--force]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()

DATA_DIR = Path("data/locations")
SCHEMA_VERSION = "1.1.0"


def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def build_office_location() -> dict:
    return {
        "location_id": "loc_office_001",
        "name": "Isaac's Office",
        "description": "warmly lit office interior, wooden shelves, desk with monitor, large window with city skyline view",
        "type": "indoor",
        "tags": ["urban", "modern"],
        "setting": {
            "location": "office",
            "time_of_day": "daytime",
            "weather": "sunny",
            "props": ["desk", "laptop", "chair"],
        },
        "entity_version": 1,
        "metadata": {
            "source": "seed_script",
            "created_at": iso_now(),
            "updated_at": iso_now(),
            "version": SCHEMA_VERSION,
            "confidence": 0.9,
            "validation_status": "validated",
        },
    }


def main(force: bool = False) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "loc_office_001.json"

    if out_path.exists() and not force:
        console.print(
            f"[yellow]⚠ {out_path} exists. Use --force to overwrite.[/yellow]"
        )
        return 0

    record = build_office_location()
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    console.print(f"[green]✓[/green] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(force=args.force))
