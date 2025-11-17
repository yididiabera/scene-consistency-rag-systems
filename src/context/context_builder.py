"""Context builder for character consistency anchors."""

import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from rich.console import Console

console = Console()


class ContextBuilder:
    """Builds character/location consistency anchors from retrieval results."""

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize with Jinja2 template environment."""
        if templates_dir is None:
            templates_dir = str(Path(__file__).parent / "templates")

        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=False,  # We handle escaping manually
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.templates_dir = templates_dir

    @staticmethod
    def _sanitize_string(value: str, max_length: int = 500) -> str:
        """
        Remove control chars, normalize whitespace, and truncate a string.

        Args:
            value: Input string (or value to convert to string).
            max_length: Maximum length before truncation (default 500).

        Returns:
            Sanitized string with control characters removed, whitespace normalized,
            and length capped at max_length with "..." appended if truncated.
        """
        if not isinstance(value, str):
            value = str(value)

        # Remove null and control characters (except newline, which we'll normalize)
        value = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", value)

        # Normalize whitespace: replace multiple spaces/newlines with single space
        value = re.sub(r"\s+", " ", value)

        # Trim
        value = value.strip()

        # Truncate if needed
        if len(value) > max_length:
            value = value[:max_length].rstrip() + "..."

        return value

    @staticmethod
    def _sanitize_list(items: List[str], max_items: int = 10) -> List[str]:
        """
        Sanitize and limit a list of items.

        Args:
            items: List of strings (or single item to wrap in a list).
            max_items: Maximum number of items to return (default 10).

        Returns:
            List of sanitized strings, limited to max_items length.
            Each item is sanitized using _sanitize_string() with max_length=100.
        """
        if not isinstance(items, list):
            items = [items]

        sanitized = [
            ContextBuilder._sanitize_string(item, max_length=100) for item in items
        ]
        return sanitized[:max_items]

    @staticmethod
    def _validate_image_path(path: Optional[str]) -> Optional[str]:
        """
        Validate image path against allowed schemes.

        Args:
            path: Image path string to validate.

        Returns:
            The path if it matches an allowed scheme (data/, http://, https://, s3://, gs://),
            or None if invalid. Logs a warning for invalid schemes.
        """
        if not path:
            return None

        path = str(path).strip()

        allowed_schemes = ("data/", "http://", "https://", "s3://", "gs://")
        if any(path.startswith(scheme) for scheme in allowed_schemes):
            return path

        console.print(f"[yellow]⚠[/yellow] Invalid image path scheme: {path}")
        return None

    @staticmethod
    def _validate_lora_token(token: Optional[str]) -> Optional[str]:
        """
        Validate LoRA token format.

        Args:
            token: LoRA token string to validate (expected format: <lora:name:weight>).

        Returns:
            The token if it matches the LoRA format regex, or None if invalid.
            Logs a warning for invalid formats.
        """
        if not token:
            return None

        token = str(token).strip()

        # LoRA format: <lora:name:weight>
        if re.match(r"^<lora:[a-zA-Z0-9_\-]+:[0-9.]+>$", token):
            return token

        console.print(f"[yellow]⚠[/yellow] Invalid LoRA token format: {token}")
        return None

    def build_anchor(
        self,
        candidates: List[Dict[str, Any]],
        template_name: str = "character_anchor.j2",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build character consistency anchor from retrieval candidates.

        Args:
            candidates: List of retrieval result dicts
            template_name: Jinja2 template filename

        Returns:
            (anchor_block_string, structured_anchor_dict)
        """
        if not candidates:
            return "", {}

        # Use first candidate (highest relevance)
        candidate = candidates[0]

        # Extract and sanitize fields
        entity_id = self._sanitize_string(
            candidate.get("entity_id", ""), max_length=100
        )
        name = self._sanitize_string(
            candidate.get("name", candidate.get("entity_id", "")), max_length=100
        )
        appearance = self._sanitize_string(
            candidate.get("appearance", candidate.get("text", "")), max_length=400
        )
        entity_version = candidate.get("entity_version", 1)
        tags = self._sanitize_list(candidate.get("tags", []), max_items=10)
        lora_trigger = self._validate_lora_token(candidate.get("lora_trigger_word"))
        canonical_image = self._validate_image_path(
            candidate.get("canonical_image_path")
        )
        metadata = candidate.get("metadata", {})

        # Extract relationships if present
        relationships = []
        if "relationships" in candidate and isinstance(
            candidate["relationships"], list
        ):
            for rel in candidate["relationships"][:5]:  # Limit to 5 relationships
                rel_dict = {
                    "relationship_type": self._sanitize_string(
                        rel.get("relationship_type", ""), max_length=50
                    ),
                    "target_entity": self._sanitize_string(
                        rel.get("target_entity", ""), max_length=100
                    ),
                    "strength": rel.get("strength", 0.5),
                    "tags": self._sanitize_list(rel.get("tags", []), max_items=5),
                }
                relationships.append(rel_dict)

        # Build template context
        context = {
            "entity_id": entity_id,
            "name": name,
            "appearance": appearance,
            "entity_version": entity_version,
            "tags": tags,
            "lora_trigger_word": lora_trigger,
            "canonical_image_path": canonical_image,
            "metadata": metadata,
            "relationships": relationships,
        }

        # Render template
        try:
            template = self.env.get_template(template_name)
            anchor_block = template.render(**context)
        except Exception as e:
            console.print(f"[red]✗[/red] Template render error: {e}")
            anchor_block = f"# ERROR: {e}"

        # Structured output for downstream use
        structured_anchor = {
            "entity_id": entity_id,
            "name": name,
            "lora_trigger_word": lora_trigger,
            "canonical_image_path": canonical_image,
            "tags": tags,
            "relationships": relationships,
        }

        return anchor_block, structured_anchor
