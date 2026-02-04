"""
Credential manager for secure storage and retrieval of API keys and tokens.

Credentials are encrypted using Fernet (AES-128-CBC) and stored per-agent.
"""

import getpass
import json
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()


class CredentialManager:
    """
    Manages secure credential storage and retrieval.

    Features:
    - Encrypted storage using Fernet (AES-128-CBC)
    - Per-agent credential isolation
    - Environment variable fallback
    - Interactive prompting with hidden input
    - Persistence choice for users

    Directory structure:
        .supyagent/credentials/.key        # Encryption key (600 permissions)
        .supyagent/credentials/<agent>.enc # Encrypted credentials
    """

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize the credential manager.

        Args:
            base_dir: Base directory for credential storage
        """
        if base_dir is None:
            base_dir = Path(".supyagent/credentials")
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._fernet = self._get_fernet()
        self._cache: dict[str, dict[str, str]] = {}

    def _get_fernet(self) -> Fernet:
        """Get or create the encryption key."""
        key_file = self.base_dir / ".key"

        if key_file.exists():
            key = key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            # Set restrictive permissions (owner read/write only)
            try:
                key_file.chmod(0o600)
            except OSError:
                pass  # Windows doesn't support chmod the same way

        return Fernet(key)

    def _cred_path(self, agent: str) -> Path:
        """Get the path to an agent's credential file."""
        return self.base_dir / f"{agent}.enc"

    def _load_credentials(self, agent: str) -> dict[str, str]:
        """
        Load and decrypt credentials for an agent.

        Args:
            agent: Agent name

        Returns:
            Dict of credential name -> value
        """
        if agent in self._cache:
            return self._cache[agent]

        path = self._cred_path(agent)
        if not path.exists():
            self._cache[agent] = {}
            return {}

        try:
            encrypted = path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            creds = json.loads(decrypted)
            self._cache[agent] = creds
            return creds
        except (InvalidToken, json.JSONDecodeError):
            # Corrupted or invalid file
            self._cache[agent] = {}
            return {}

    def _save_credentials(self, agent: str, creds: dict[str, str]) -> None:
        """
        Encrypt and save credentials.

        Args:
            agent: Agent name
            creds: Dict of credentials to save
        """
        encrypted = self._fernet.encrypt(json.dumps(creds).encode())
        path = self._cred_path(agent)
        path.write_bytes(encrypted)
        # Set restrictive permissions
        try:
            path.chmod(0o600)
        except OSError:
            pass
        self._cache[agent] = creds

    def get(self, agent: str, name: str) -> str | None:
        """
        Get a credential value.

        Checks in order:
        1. Environment variables
        2. Stored credentials

        Args:
            agent: Agent name
            name: Credential name (e.g., "OPENAI_API_KEY")

        Returns:
            Credential value or None if not found
        """
        # First check environment
        if name in os.environ:
            return os.environ[name]

        # Then check stored credentials
        creds = self._load_credentials(agent)
        return creds.get(name)

    def set(
        self,
        agent: str,
        name: str,
        value: str,
        persist: bool = True,
    ) -> None:
        """
        Set a credential value.

        Args:
            agent: Agent name
            name: Credential name
            value: Credential value
            persist: If True, save to encrypted storage. If False, only set in environment.
        """
        if persist:
            creds = self._load_credentials(agent)
            creds[name] = value
            self._save_credentials(agent, creds)
        else:
            # Session-only: just set in environment
            os.environ[name] = value

    def has(self, agent: str, name: str) -> bool:
        """
        Check if a credential exists.

        Args:
            agent: Agent name
            name: Credential name

        Returns:
            True if credential exists
        """
        return self.get(agent, name) is not None

    def prompt_for_credential(
        self,
        name: str,
        description: str,
        service: str | None = None,
    ) -> tuple[str, bool] | None:
        """
        Interactively prompt user for a credential.

        Displays a nice panel and uses hidden input for the value.

        Args:
            name: Credential name
            description: Why this credential is needed
            service: Optional service name

        Returns:
            Tuple of (value, should_persist) or None if user skipped
        """
        console.print()

        # Build the panel content
        service_line = f"\nService: [cyan]{service}[/cyan]" if service else ""
        content = f"[bold]{name}[/bold]{service_line}\n\n{description}"

        console.print(
            Panel(
                content,
                title="🔑 Credential Required",
                border_style="yellow",
            )
        )

        # Get the value with hidden input
        value = getpass.getpass("Enter value (or press Enter to skip): ")

        if not value:
            console.print("[dim]Skipped[/dim]")
            return None

        # Ask about persistence
        persist = Confirm.ask("Save for future sessions?", default=True)

        return value, persist

    def list_credentials(self, agent: str) -> list[str]:
        """
        List stored credential names for an agent.

        Args:
            agent: Agent name

        Returns:
            List of credential names
        """
        creds = self._load_credentials(agent)
        return list(creds.keys())

    def delete(self, agent: str, name: str) -> bool:
        """
        Delete a stored credential.

        Args:
            agent: Agent name
            name: Credential name

        Returns:
            True if deleted, False if not found
        """
        creds = self._load_credentials(agent)
        if name in creds:
            del creds[name]
            self._save_credentials(agent, creds)
            return True
        return False

    def get_all_for_tools(self, agent: str) -> dict[str, str]:
        """
        Get all credentials for tool execution.

        Combines environment variables with stored credentials
        (environment takes precedence).

        Args:
            agent: Agent name

        Returns:
            Dict of all available credentials
        """
        # Start with stored credentials
        creds = dict(self._load_credentials(agent))

        # Override with environment variables (they take precedence)
        for name in creds:
            if name in os.environ:
                creds[name] = os.environ[name]

        return creds
