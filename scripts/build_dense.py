#!/usr/bin/env python3
"""
Build dense (CLIP) indices in ChromaDB from data JSONs.

This script:
- Loads characters/locations/relationships JSONs from data/
- Uses RAGPipeline to generate embeddings and upsert into ChromaDB
- Optionally rebuilds (resets) Chroma collections

Usage:
  python scripts/build_dense.py [--characters data/characters] [--locations data/locations] [--relationships data/relationships] [--rebuild]

Notes:
- Requires CLIP and torch installed. Install CLIP with:
  pip install git+https://github.com/openai/CLIP.git
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console

sys.path.append("src")
from pipeline import RAGPipeline  # noqa: E402

console = Console()


def load_jsons(pattern: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in glob.glob(pattern):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            out.extend(obj if isinstance(obj, list) else [obj])
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to load {p}: {e}")
    return out


def main(ns: argparse.Namespace) -> int:
    pipeline = RAGPipeline()

    characters = (
        load_jsons(str(Path(ns.characters) / "*.json")) if ns.characters else []
    )
    locations = load_jsons(str(Path(ns.locations) / "*.json")) if ns.locations else []
    relationships = (
        load_jsons(str(Path(ns.relationships) / "*.json")) if ns.relationships else []
    )

    if not any([characters, locations, relationships]):
        console.print(
            "[yellow]⚠ No input JSONs found. Provide --characters/--locations/--relationships."
        )
        return 1

    pipeline.build_indices(
        characters=characters or None,
        locations=locations or None,
        relationships=relationships or None,
        rebuild=ns.rebuild,
    )

    console.print("[green]✓[/green] Dense indices built in ChromaDB")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build dense CLIP indices into ChromaDB from data JSONs"
    )
    parser.add_argument(
        "--characters", default="data/characters", help="Characters dir"
    )
    parser.add_argument("--locations", default="data/locations", help="Locations dir")
    parser.add_argument(
        "--relationships", default="data/relationships", help="Relationships dir"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reset Chroma collections before building",
    )
    args = parser.parse_args()
    raise SystemExit(main(args))
