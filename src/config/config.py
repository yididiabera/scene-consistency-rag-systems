"""
Configuration Management
Loads and validates YAML configuration with singleton pattern
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from rich.console import Console

console = Console()


class Config:
    """Singleton configuration manager."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str = "configs/config.yaml") -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self._config is not None:
            return self._config

        path = Path(config_path)
        if not path.exists():
            fallback = Path(__file__).with_name("default_config.yaml")
            if fallback.exists():
                console.print(
                    f"[yellow]⚠[/yellow] Config file not found at {config_path}; using default_config.yaml"
                )
                path = fallback
            else:
                console.print(f"[red]✗[/red] Config file not found: {config_path}")
                raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r") as f:
            self._config = yaml.safe_load(f)

        console.print(f"[green]✓[/green] Configuration loaded from {config_path}")
        return self._config

    def get(self, key: str, default=None):
        """Get configuration value by key."""
        if self._config is None:
            self.load()
        return self._config.get(key, default)

    def __getitem__(self, key: str):
        """Allow dictionary-style access."""
        if self._config is None:
            self.load()
        return self._config[key]


# Global config instance
cfg = Config()
