#!/usr/bin/env python3
"""
Generate character JSON files from data/characters/*.txt + images using schema v1.1.
- Reads appearance text from <name>.txt
- Finds canonical image (png/jpg)
- Writes <name>.json with required fields

Usage:
  python scripts/generate_character_jsons.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.console import Console

console = Console()

DATA_DIR = Path("data/characters")
SCHEMA_VERSION = "1.1.0"


def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def rename_if_needed(stem: str) -> str:
    # Correct common name typos
    corrections = {"issac": "isaac"}
    return corrections.get(stem.lower(), stem.lower())


def find_image(stem: str) -> Optional[Path]:
    # prefer png then jpg
    candidates = [DATA_DIR / f"{stem}.png", DATA_DIR / f"{stem}.jpg"]
    for p in candidates:
        if p.exists():
            return p
    return None


def ensure_renamed(original_stem: str, corrected_stem: str):
    if original_stem == corrected_stem:
        return
    # Rename files if exist
    for ext in (".png", ".jpg", ".txt"):
        src = DATA_DIR / f"{original_stem}{ext}"
        dst = DATA_DIR / f"{corrected_stem}{ext}"
        if src.exists() and not dst.exists():
            src.rename(dst)
            console.print(f"[yellow]↪[/yellow] Renamed {src.name} -> {dst.name}")


def build_character_json(stem: str, appearance_text: str, image_path: Path) -> dict:
    name_title = stem.replace("_", " ").title()
    character_id = f"char_{stem}_001"
    lora_trigger = f"<lora:{stem}_v1:1.0>"

    return {
        "character_id": character_id,
        "name": name_title,
        "canonical_image_path": str(image_path.as_posix()),
        "lora_trigger_word": lora_trigger,
        "appearance": appearance_text.strip(),
        "tags": [],
        "entity_version": 1,
        "metadata": {
            "source": "auto_from_txt",
            "created_at": iso_now(),
            "updated_at": iso_now(),
            "version": SCHEMA_VERSION,
            "confidence": 0.7,
            "validation_status": "draft",
        },
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = list(DATA_DIR.glob("*.txt"))
    if not txt_files:
        console.print("[yellow]⚠ No .txt files found in data/characters/[/yellow]")
        return 0

    created = 0
    skipped = 0

    for txt in sorted(txt_files):
        original_stem = txt.stem
        stem = rename_if_needed(original_stem)
        ensure_renamed(original_stem, stem)

        # Recompute path after potential rename
        txt = DATA_DIR / f"{stem}.txt"
        if not txt.exists():
            console.print(
                f"[yellow]⚠ Skipping {original_stem}: missing .txt after rename"
            )
            skipped += 1
            continue

        image = find_image(stem)
        if image is None:
            console.print(f"[yellow]⚠ Skipping {stem}: no image (.png/.jpg) found")
            skipped += 1
            continue

        appearance = txt.read_text(encoding="utf-8")
        record = build_character_json(stem, appearance, image)

        out_path = DATA_DIR / f"{stem}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        created += 1
        console.print(f"[green]✓[/green] Wrote {out_path}")

    console.print(f"[bold]Summary:[/bold] created={created}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
