"""Prompt assembly and final sanitization."""

import re
from typing import Dict, Tuple, Optional
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console

console = Console()


class PromptAssembler:
    """Assembles final prompts with anchors and applies sanitization."""

    def __init__(
        self, templates_dir: Optional[str] = None, max_prompt_length: int = 2000
    ):
        """Initialize with Jinja2 template environment."""
        if templates_dir is None:
            templates_dir = str(Path(__file__).parent / "templates")

        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(["html", "xml", "j2"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.templates_dir = templates_dir
        self.max_prompt_length = max_prompt_length

    @staticmethod
    def _sanitize_prompt(prompt: str, max_length: int = 2000) -> str:
        """Final sanitization: remove control chars, normalize, truncate."""
        # Remove null and control characters (except newline)
        prompt = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", prompt)

        # Trim trailing whitespace
        prompt = prompt.rstrip()

        # Truncate if needed (with buffer for truncation message)
        truncation_msg = "\n[... truncated ...]"
        if len(prompt) > max_length:
            # Reserve space for truncation message
            available = max_length - len(truncation_msg)
            prompt = prompt[:available].rstrip() + truncation_msg

        return prompt

    def assemble(
        self,
        anchor_block: str,
        scene_description: str = "",
        shot_description: str = "",
        template_name: str = "prompt_template.j2",
    ) -> Tuple[str, Dict[str, any]]:
        """
        Assemble final prompt with anchor and apply sanitization.

        Args:
            anchor_block: Character/location anchor string
            scene_description: Scene context
            shot_description: Shot/framing description
            template_name: Jinja2 template filename

        Returns:
            (final_prompt_string, debug_info_dict)
        """
        # Sanitize inputs
        scene_description = re.sub(r"[\x00-\x1F]", "", scene_description).strip()
        shot_description = re.sub(r"[\x00-\x1F]", "", shot_description).strip()

        # Build template context
        context = {
            "anchor_block": anchor_block,
            "scene_description": scene_description,
            "shot_description": shot_description,
        }

        # Render template
        try:
            template = self.env.get_template(template_name)
            prompt = template.render(**context)
        except Exception as e:
            console.print(f"[red]✗[/red] Prompt template render error: {e}")
            prompt = f"ERROR: {e}"

        # Capture original length before sanitization/truncation
        original_length = len(prompt)

        # Final sanitization (may truncate)
        prompt = self._sanitize_prompt(prompt, max_length=self.max_prompt_length)
        final_length = len(prompt)
        truncated = original_length > self.max_prompt_length

        # Debug info
        debug_info = {
            "original_length": original_length,
            "final_length": final_length,
            "truncated": truncated,
            "anchor_included": "ANCHOR" in prompt,
            "scene_included": (
                scene_description in prompt if scene_description else True
            ),
            "shot_included": shot_description in prompt if shot_description else True,
        }

        return prompt, debug_info
