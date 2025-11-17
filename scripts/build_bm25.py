#!/usr/bin/env python3
"""
Build BM25 index from data JSONs (characters, locations, relationships).

This script:
- Loads JSON files from data/characters, data/locations, data/relationships
- Uses DatasetPreparer to chunk and tokenize
- Builds and saves BM25 index to the configured path

Usage:
  python scripts/build_bm25.py [--characters data/characters] [--locations data/locations] [--relationships data/relationships]

Notes:
- Does not require CLIP or sentence-transformers.
- Uses Rich for logging.
"""

import argparse
import glob
import json
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console

import sys

sys.path.append("src")
from dataset import DatasetPreparer  # noqa: E402

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


def main(args: argparse.Namespace) -> int:
    dp = DatasetPreparer()

    characters = (
        load_jsons(str(Path(args.characters) / "*.json")) if args.characters else []
    )
    locations = (
        load_jsons(str(Path(args.locations) / "*.json")) if args.locations else []
    )
    relationships = (
        load_jsons(str(Path(args.relationships) / "*.json"))
        if args.relationships
        else []
    )

    char_docs = dp.prepare_documents(characters, "character") if characters else []
    loc_docs = dp.prepare_documents(locations, "location") if locations else []
    rel_docs = (
        dp.prepare_documents(relationships, "relationship") if relationships else []
    )

    all_docs = char_docs + loc_docs + rel_docs
    if not all_docs:
        console.print(
            "[yellow]⚠ No documents found. Ensure your data directories contain JSON files."
        )
        return 1

    bm25 = dp.build_bm25_index(all_docs)
    dp.save_bm25_index(bm25, all_docs)

    console.print(
        f"[green]✓[/green] Done. Docs={len(all_docs)} | Characters={len(char_docs)} | Locations={len(loc_docs)} | Relationships={len(rel_docs)}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 index from data JSONs")
    parser.add_argument(
        "--characters", default="data/characters", help="Characters dir"
    )
    parser.add_argument("--locations", default="data/locations", help="Locations dir")
    parser.add_argument(
        "--relationships",
        default="data/relationships",
        help="Relationships dir (optional)",
    )
    ns = parser.parse_args()
    raise SystemExit(main(ns))
