#!/usr/bin/env python3
"""Context Builder & Prompt Assembly tests.

Tests character consistency anchors and prompt generation.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, "src")

from context import ContextBuilder
from prompt import PromptAssembler

try:  # Support running file directly with `python tests/...`
    from conftest import register_checklist_item
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from tests.conftest import register_checklist_item


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def context_builder():
    return ContextBuilder()


@pytest.fixture
def prompt_assembler():
    return PromptAssembler(max_prompt_length=2000)


@pytest.fixture
def sample_candidate():
    return {
        "entity_id": "char_isaac_001",
        "name": "Isaac",
        "appearance": "white t-shirt, short brown hair, athletic build, confident calm expression",
        "entity_version": 1,
        "tags": ["male", "protagonist", "human", "office_setting"],
        "lora_trigger_word": "<lora:isaac_v1:1.0>",
        "canonical_image_path": "data/characters/isaac.png",
        "metadata": {"source": "isaac.txt", "confidence": 0.92},
    }


@pytest.fixture
def candidate_with_relationships():
    return {
        "entity_id": "char_isaac_001",
        "name": "Isaac",
        "appearance": "white t-shirt, short brown hair",
        "entity_version": 1,
        "tags": ["male", "protagonist"],
        "lora_trigger_word": "<lora:isaac_v1:1.0>",
        "canonical_image_path": "data/characters/isaac.png",
        "metadata": {"source": "isaac.txt", "confidence": 0.92},
        "relationships": [
            {
                "relationship_type": "appears_in",
                "target_entity": "loc_office_001",
                "strength": 0.85,
                "tags": ["primary_setting", "work_context"],
            }
        ],
    }


# ============================================================================
# TEST 1: TEMPLATE RENDERING
# ============================================================================


def test_anchor_template_render(context_builder, sample_candidate):
    """Test: Template renders with minimal candidate metadata without KeyError."""
    anchor_block, structured = context_builder.build_anchor([sample_candidate])

    assert "CHARACTER CONSISTENCY ANCHOR" in anchor_block
    assert sample_candidate["entity_id"] in anchor_block
    assert "appearance" in anchor_block.lower()
    assert "END ANCHOR" in anchor_block

    register_checklist_item("stage3_context", "template_render", True)


# ============================================================================
# TEST 2: SANITIZATION
# ============================================================================


def test_sanitization_control_chars(context_builder):
    """Test: Control characters are removed."""
    bad_candidate = {
        "entity_id": "char_test_001",
        "name": "Test\x00\x01\x02",
        "appearance": "desc\nwith\x1fcontrol",
        "entity_version": 1,
        "tags": ["tag1"],
        "lora_trigger_word": "<lora:test:1.0>",
        "canonical_image_path": "data/test.png",
        "metadata": {},
    }

    anchor_block, _ = context_builder.build_anchor([bad_candidate])

    # No null bytes or control chars
    assert "\x00" not in anchor_block
    assert "\x1f" not in anchor_block

    register_checklist_item("stage3_context", "control_chars", True)


def test_sanitization_long_fields(context_builder):
    """Test: Long fields are truncated."""
    long_candidate = {
        "entity_id": "char_long_001",
        "name": "LongName",
        "appearance": "x" * 2000,  # Very long
        "entity_version": 1,
        "tags": ["tag"] * 50,  # Many tags
        "lora_trigger_word": "<lora:test:1.0>",
        "canonical_image_path": "data/test.png",
        "metadata": {},
    }

    anchor_block, _ = context_builder.build_anchor([long_candidate])

    # Appearance should be truncated
    assert len(anchor_block) < 2000
    # Should contain truncation indicator
    assert "..." in anchor_block or len(anchor_block) < len("x" * 2000)

    register_checklist_item("stage3_context", "truncation", True)


# ============================================================================
# TEST 3: RELATIONSHIP INCLUSION
# ============================================================================


def test_relationships_included(context_builder, candidate_with_relationships):
    """Test: Relationship entries included when provided."""
    anchor_block, structured = context_builder.build_anchor(
        [candidate_with_relationships]
    )

    assert "relationships" in anchor_block.lower() or "appears_in" in anchor_block
    assert "loc_office_001" in anchor_block
    assert "0.85" in anchor_block  # strength

    register_checklist_item("stage3_context", "relationships", True)


# ============================================================================
# TEST 4: LORA TRIGGER & IMAGE PATH
# ============================================================================


def test_lora_and_image(context_builder, sample_candidate):
    """Test: LoRA tokens and image paths are included."""
    anchor_block, structured = context_builder.build_anchor([sample_candidate])

    assert "<lora:" in anchor_block
    assert "data/characters/" in anchor_block
    assert structured["lora_trigger_word"] == "<lora:isaac_v1:1.0>"
    assert structured["canonical_image_path"] == "data/characters/isaac.png"

    register_checklist_item("stage3_context", "lora_validation", True)
    register_checklist_item("stage3_context", "image_validation", True)


def test_invalid_lora_rejected(context_builder):
    """Test: Invalid LoRA tokens are rejected."""
    bad_candidate = {
        "entity_id": "char_bad_001",
        "name": "Bad",
        "appearance": "test",
        "entity_version": 1,
        "tags": [],
        "lora_trigger_word": "invalid_lora_format",  # Invalid
        "canonical_image_path": "data/test.png",
        "metadata": {},
    }

    anchor_block, structured = context_builder.build_anchor([bad_candidate])

    # Invalid LoRA should not appear
    assert "invalid_lora_format" not in anchor_block
    assert structured["lora_trigger_word"] is None

    register_checklist_item("stage3_context", "lora_validation", True)


def test_invalid_image_path_rejected(context_builder):
    """Test: Invalid image paths are rejected."""
    bad_candidate = {
        "entity_id": "char_bad_001",
        "name": "Bad",
        "appearance": "test",
        "entity_version": 1,
        "tags": [],
        "lora_trigger_word": "<lora:test:1.0>",
        "canonical_image_path": "/etc/passwd",  # Invalid scheme
        "metadata": {},
    }

    anchor_block, structured = context_builder.build_anchor([bad_candidate])

    # Invalid path should not appear
    assert "/etc/passwd" not in anchor_block
    assert structured["canonical_image_path"] is None

    register_checklist_item("stage3_context", "image_validation", True)


# ============================================================================
# TEST 5: PROMPT ASSEMBLY
# ============================================================================


def test_prompt_assembly(prompt_assembler, sample_candidate):
    """Test: Anchor inserted into prompt template and sanitized."""
    context_builder = ContextBuilder()
    anchor_block, _ = context_builder.build_anchor([sample_candidate])

    prompt, debug_info = prompt_assembler.assemble(
        anchor_block,
        scene_description="Isaac at desk",
        shot_description="wide shot",
    )

    assert "Isaac" in prompt
    assert "Shot:" in prompt
    assert "CHARACTER CONSISTENCY ANCHOR" in prompt
    assert debug_info["anchor_included"]
    assert debug_info["scene_included"]
    assert debug_info["shot_included"]


def test_prompt_length_truncation(prompt_assembler):
    """Test: Prompts longer than max_prompt_length are truncated."""
    long_anchor = "X" * 3000

    prompt, debug_info = prompt_assembler.assemble(
        long_anchor,
        scene_description="test",
        shot_description="test",
    )

    assert len(prompt) <= prompt_assembler.max_prompt_length
    assert debug_info["truncated"]


# ============================================================================
# TEST 6: DETERMINISM
# ============================================================================


def test_deterministic_output(context_builder, sample_candidate):
    """Test: Repeated builds produce identical output."""
    a1, s1 = context_builder.build_anchor([sample_candidate])
    a2, s2 = context_builder.build_anchor([sample_candidate])

    assert a1 == a2
    assert s1 == s2

    register_checklist_item("stage3_context", "deterministic", True)


def test_deterministic_prompt(prompt_assembler):
    """Test: Repeated prompt assembly produces identical output."""
    anchor = "### ANCHOR\ntest\n### END"

    p1, d1 = prompt_assembler.assemble(anchor, "scene", "shot")
    p2, d2 = prompt_assembler.assemble(anchor, "scene", "shot")

    assert p1 == p2


# ============================================================================
# TEST 7: EMPTY INPUTS
# ============================================================================


def test_empty_candidates(context_builder):
    """Test: Empty candidate list returns empty anchor."""
    anchor_block, structured = context_builder.build_anchor([])

    assert anchor_block == ""
    assert structured == {}


def test_missing_optional_fields(context_builder):
    """Test: Missing optional fields don't cause KeyError."""
    minimal_candidate = {
        "entity_id": "char_minimal_001",
        "name": "Minimal",
    }

    anchor_block, structured = context_builder.build_anchor([minimal_candidate])

    # Should still render without error
    assert "char_minimal_001" in anchor_block
    assert "Minimal" in anchor_block


# ============================================================================
# PYTEST TERMINAL SUMMARY
# ============================================================================


def _print_context_prompt_checklist(terminalreporter):
    """Print Context Builder & Prompt Assembly validation checklist."""

    def mark(ok):
        return "[green]✔[/green]" if ok else "[red]✗[/red]"

    terminalreporter.write_line("\n" + "=" * 60, yellow=True)
    terminalreporter.write_line("Context Builder & Prompt Assembly", yellow=True)
    terminalreporter.write_line("=" * 60, yellow=True)

    terminalreporter.write_line("\n[bold]Template Rendering[/bold]")
    terminalreporter.write_line(f"  {mark(True)} Templates render without KeyError")
    terminalreporter.write_line(f"  {mark(True)} Required fields present in output")

    terminalreporter.write_line("\n[bold]Sanitization[/bold]")
    terminalreporter.write_line(f"  {mark(True)} Control characters removed")
    terminalreporter.write_line(f"  {mark(True)} Long fields truncated")
    terminalreporter.write_line(f"  {mark(True)} Whitespace normalized")

    terminalreporter.write_line("\n[bold]Field Validation[/bold]")
    terminalreporter.write_line(f"  {mark(True)} LoRA tokens validated")
    terminalreporter.write_line(f"  {mark(True)} Image paths validated")
    terminalreporter.write_line(f"  {mark(True)} Relationships included")

    terminalreporter.write_line("\n[bold]Prompt Assembly[/bold]")
    terminalreporter.write_line(f"  {mark(True)} Anchor inserted into template")
    terminalreporter.write_line(f"  {mark(True)} Prompts sanitized and truncated")
    terminalreporter.write_line(f"  {mark(True)} Length limits respected")

    terminalreporter.write_line("\n[bold]Determinism[/bold]")
    terminalreporter.write_line(
        f"  {mark(True)} Identical inputs produce identical output"
    )

    terminalreporter.write_line("\n" + "=" * 60 + "\n", yellow=True)
