"""
Configuration manager for global settings like LLM API keys.

Stores encrypted configuration in ~/.supyagent/config/ that is shared
across all agents and projects.
"""

import getpass
import json
import os
import re
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Common LLM provider API key names
KNOWN_LLM_KEYS = {
    "OPENAI_API_KEY": "OpenAI (GPT-4, GPT-3.5)",
    "ANTHROPIC_API_KEY": "Anthropic (Claude)",
    "GOOGLE_API_KEY": "Google (Gemini)",
    "AZURE_API_KEY": "Azure OpenAI",
    "AZURE_API_BASE": "Azure OpenAI endpoint",
    "COHERE_API_KEY": "Cohere",
    "HUGGINGFACE_API_KEY": "Hugging Face",
    "REPLICATE_API_KEY": "Replicate",
    "TOGETHER_API_KEY": "Together AI",
    "GROQ_API_KEY": "Groq",
    "MISTRAL_API_KEY": "Mistral AI",
    "PERPLEXITY_API_KEY": "Perplexity AI",
    "OPENROUTER_API_KEY": "OpenRouter",
    "DEEPSEEK_API_KEY": "DeepSeek",
    "FIREWORKS_API_KEY": "Fireworks AI",
    "STABILITY_API_KEY": "Stability.ai (Stable Diffusion 3)",
    "FAL_API_KEY": "FAL AI (Flux, Bria)",
    "OLLAMA_API_BASE": "Ollama (local) base URL",
}


class ConfigManager:
    """
    Manages global configuration including LLM API keys.

    Configuration is stored encrypted in ~/.supyagent/config/
    and automatically loaded into environment variables when
    agents are run.

    Directory structure:
        ~/.supyagent/config/.key     # Encryption key
        ~/.supyagent/config/keys.enc # Encrypted API keys
    """

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize the config manager.

        Args:
            base_dir: Base directory for config storage.
                      Defaults to ~/.supyagent/config/
        """
        if base_dir is None:
            base_dir = Path.home() / ".supyagent" / "config"

        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._fernet = self._get_fernet()
        self._cache: dict[str, str] | None = None

    def _get_fernet(self) -> Fernet:
        """Get or create the encryption key."""
        key_file = self.base_dir / ".key"

        if key_file.exists():
            key = key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            try:
                key_file.chmod(0o600)
            except OSError:
                pass

        return Fernet(key)

    def _keys_path(self) -> Path:
        """Get path to the encrypted keys file."""
        return self.base_dir / "keys.enc"

    def _load_keys(self) -> dict[str, str]:
        """Load and decrypt stored keys."""
        if self._cache is not None:
            return self._cache

        path = self._keys_path()
        if not path.exists():
            self._cache = {}
            return {}

        try:
            encrypted = path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            keys = json.loads(decrypted)
            self._cache = keys
            return keys
        except (InvalidToken, json.JSONDecodeError):
            self._cache = {}
            return {}

    def _save_keys(self, keys: dict[str, str]) -> None:
        """Encrypt and save keys."""
        encrypted = self._fernet.encrypt(json.dumps(keys).encode())
        path = self._keys_path()
        path.write_bytes(encrypted)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        self._cache = keys

    def get(self, name: str) -> str | None:
        """
        Get a config value.

        Checks environment first, then stored config.

        Args:
            name: Key name

        Returns:
            Value or None
        """
        # Environment takes precedence
        if name in os.environ:
            return os.environ[name]

        keys = self._load_keys()
        return keys.get(name)

    def set(self, name: str, value: str) -> None:
        """
        Set a config value.

        Args:
            name: Key name
            value: Key value
        """
        keys = self._load_keys()
        keys[name] = value
        self._save_keys(keys)

    def delete(self, name: str) -> bool:
        """
        Delete a stored key.

        Args:
            name: Key name

        Returns:
            True if deleted, False if not found
        """
        keys = self._load_keys()
        if name in keys:
            del keys[name]
            self._save_keys(keys)
            return True
        return False

    def list_keys(self) -> list[str]:
        """List all stored key names."""
        return list(self._load_keys().keys())

    def load_into_environment(self) -> int:
        """
        Load all stored keys into environment variables.

        Only sets variables that aren't already in the environment.

        Returns:
            Number of keys loaded
        """
        keys = self._load_keys()
        loaded = 0

        for name, value in keys.items():
            if name not in os.environ:
                os.environ[name] = value
                loaded += 1

        return loaded

    def set_interactive(self, name: str | None = None) -> bool:
        """
        Interactively prompt user to set a key.

        Args:
            name: Key name, or None to show a menu of common keys

        Returns:
            True if key was set
        """
        if name is None:
            # Show menu of common keys
            console.print()
            console.print("[bold]Common LLM API Keys:[/bold]")
            console.print()

            items = list(KNOWN_LLM_KEYS.items())
            for i, (key, desc) in enumerate(items, 1):
                status = "[green]✓[/green]" if self.get(key) else "[dim]○[/dim]"
                console.print(f"  {status} [{i}] {key}")
                console.print(f"      [dim]{desc}[/dim]")

            console.print()
            console.print("  [0] Enter custom key name")
            console.print()

            choice = input("Select key to set (number or name): ").strip()

            if choice == "0":
                name = input("Enter key name: ").strip()
                if not name:
                    return False
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(items):
                    name = items[idx][0]
                else:
                    console.print("[red]Invalid selection[/red]")
                    return False
            else:
                # Treat as key name
                name = choice.upper()

        # Get the value
        description = KNOWN_LLM_KEYS.get(name, "API key")
        console.print()
        console.print(
            Panel(
                f"[bold]{name}[/bold]\n{description}",
                title="🔑 Set API Key",
                border_style="blue",
            )
        )

        value = getpass.getpass("Enter value (or press Enter to cancel): ")

        if not value:
            console.print("[dim]Cancelled[/dim]")
            return False

        self.set(name, value)
        console.print(f"[green]✓[/green] Saved {name}")
        return True

    def set_from_file(self, file_path: str | Path) -> int:
        """
        Load keys from a .env file.

        File format (one per line):
            KEY_NAME=value
            # comments are ignored
            export KEY_NAME=value  # export prefix is stripped

        Args:
            file_path: Path to .env file

        Returns:
            Number of keys imported
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        imported = 0
        pattern = re.compile(r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.+)$")

        with open(path) as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                match = pattern.match(line)
                if match:
                    name, value = match.groups()

                    # Strip quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]

                    self.set(name, value)
                    imported += 1

        return imported

    def show_status(self) -> None:
        """Display current configuration status."""
        keys = self._load_keys()

        if not keys:
            console.print("[dim]No API keys configured[/dim]")
            console.print()
            console.print("Run [cyan]supyagent config set[/cyan] to add keys")
            return

        table = Table(title="Configured API Keys")
        table.add_column("Key", style="cyan")
        table.add_column("Provider")
        table.add_column("Status")

        for name in sorted(keys.keys()):
            provider = KNOWN_LLM_KEYS.get(name, "Custom")
            # Show if it's overridden by environment
            if name in os.environ and os.environ[name] != keys[name]:
                status = "[yellow]env override[/yellow]"
            else:
                status = "[green]stored[/green]"

            table.add_row(name, provider, status)

        console.print(table)
        console.print()
        console.print(f"[dim]Config location: {self.base_dir}[/dim]")


# Global config manager instance
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config() -> int:
    """
    Load global config into environment.

    Call this at the start of any agent execution.

    Returns:
        Number of keys loaded
    """
    return get_config_manager().load_into_environment()
