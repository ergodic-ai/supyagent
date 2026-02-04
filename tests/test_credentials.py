"""
Tests for credential management.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from supyagent.core.credentials import CredentialManager


class TestCredentialManagerInit:
    """Tests for CredentialManager initialization."""

    def test_creates_directory(self, temp_dir):
        """Test that credential manager creates its directory."""
        cred_dir = temp_dir / "creds"
        mgr = CredentialManager(base_dir=cred_dir)

        assert cred_dir.exists()
        assert (cred_dir / ".key").exists()

    def test_generates_encryption_key(self, temp_dir):
        """Test that encryption key is generated."""
        cred_dir = temp_dir / "creds"
        mgr = CredentialManager(base_dir=cred_dir)

        key_file = cred_dir / ".key"
        assert key_file.exists()
        assert len(key_file.read_bytes()) > 0

    def test_reuses_existing_key(self, temp_dir):
        """Test that existing key is reused."""
        cred_dir = temp_dir / "creds"

        # Create first manager (generates key)
        mgr1 = CredentialManager(base_dir=cred_dir)
        key1 = (cred_dir / ".key").read_bytes()

        # Create second manager (should reuse key)
        mgr2 = CredentialManager(base_dir=cred_dir)
        key2 = (cred_dir / ".key").read_bytes()

        assert key1 == key2


class TestCredentialStorage:
    """Tests for storing and retrieving credentials."""

    def test_set_and_get_credential(self, temp_dir):
        """Test setting and getting a credential."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        mgr.set("test-agent", "API_KEY", "secret123")
        value = mgr.get("test-agent", "API_KEY")

        assert value == "secret123"

    def test_get_nonexistent_credential(self, temp_dir):
        """Test getting a credential that doesn't exist."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        value = mgr.get("test-agent", "NONEXISTENT")
        assert value is None

    def test_credential_persists_across_instances(self, temp_dir):
        """Test that credentials persist across manager instances."""
        cred_dir = temp_dir / "creds"

        # Set credential
        mgr1 = CredentialManager(base_dir=cred_dir)
        mgr1.set("test-agent", "API_KEY", "secret123")

        # Get with new instance
        mgr2 = CredentialManager(base_dir=cred_dir)
        value = mgr2.get("test-agent", "API_KEY")

        assert value == "secret123"

    def test_credentials_are_encrypted(self, temp_dir):
        """Test that credentials are stored encrypted."""
        cred_dir = temp_dir / "creds"
        mgr = CredentialManager(base_dir=cred_dir)

        mgr.set("test-agent", "API_KEY", "secret123")

        # Read the raw file
        cred_file = cred_dir / "test-agent.enc"
        raw_content = cred_file.read_bytes()

        # Should not contain plaintext
        assert b"secret123" not in raw_content
        assert b"API_KEY" not in raw_content

    def test_multiple_credentials_per_agent(self, temp_dir):
        """Test storing multiple credentials for one agent."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        mgr.set("test-agent", "KEY1", "value1")
        mgr.set("test-agent", "KEY2", "value2")
        mgr.set("test-agent", "KEY3", "value3")

        assert mgr.get("test-agent", "KEY1") == "value1"
        assert mgr.get("test-agent", "KEY2") == "value2"
        assert mgr.get("test-agent", "KEY3") == "value3"

    def test_credentials_isolated_by_agent(self, temp_dir):
        """Test that credentials are isolated between agents."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        mgr.set("agent-a", "API_KEY", "secret-a")
        mgr.set("agent-b", "API_KEY", "secret-b")

        assert mgr.get("agent-a", "API_KEY") == "secret-a"
        assert mgr.get("agent-b", "API_KEY") == "secret-b"

    def test_update_existing_credential(self, temp_dir):
        """Test updating an existing credential."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        mgr.set("test-agent", "API_KEY", "old-value")
        mgr.set("test-agent", "API_KEY", "new-value")

        assert mgr.get("test-agent", "API_KEY") == "new-value"


class TestEnvironmentFallback:
    """Tests for environment variable fallback."""

    def test_env_var_takes_precedence(self, temp_dir):
        """Test that environment variables take precedence."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        # Set stored credential
        mgr.set("test-agent", "MY_VAR", "stored-value")

        # Set environment variable
        with patch.dict(os.environ, {"MY_VAR": "env-value"}):
            value = mgr.get("test-agent", "MY_VAR")

        assert value == "env-value"

    def test_falls_back_to_stored(self, temp_dir):
        """Test fallback to stored credential when env var not set."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        mgr.set("test-agent", "STORED_ONLY", "stored-value")
        value = mgr.get("test-agent", "STORED_ONLY")

        assert value == "stored-value"


class TestNonPersistentCredentials:
    """Tests for session-only credentials."""

    def test_non_persistent_sets_env_var(self, temp_dir):
        """Test that non-persistent credentials set environment variable."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        # Clear any existing env var
        os.environ.pop("TEMP_KEY", None)

        mgr.set("test-agent", "TEMP_KEY", "temp-value", persist=False)

        assert os.environ.get("TEMP_KEY") == "temp-value"

        # Clean up
        os.environ.pop("TEMP_KEY", None)

    def test_non_persistent_not_saved_to_disk(self, temp_dir):
        """Test that non-persistent credentials aren't saved."""
        cred_dir = temp_dir / "creds"
        mgr = CredentialManager(base_dir=cred_dir)

        mgr.set("test-agent", "TEMP_KEY", "temp-value", persist=False)

        # File shouldn't exist
        cred_file = cred_dir / "test-agent.enc"
        assert not cred_file.exists()

        # Clean up env
        os.environ.pop("TEMP_KEY", None)


class TestHasCredential:
    """Tests for checking credential existence."""

    def test_has_stored_credential(self, temp_dir):
        """Test has() with stored credential."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")
        mgr.set("test-agent", "API_KEY", "value")

        assert mgr.has("test-agent", "API_KEY") is True

    def test_has_env_credential(self, temp_dir):
        """Test has() with environment variable."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        with patch.dict(os.environ, {"ENV_KEY": "value"}):
            assert mgr.has("test-agent", "ENV_KEY") is True

    def test_has_nonexistent(self, temp_dir):
        """Test has() with nonexistent credential."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        assert mgr.has("test-agent", "NONEXISTENT") is False


class TestDeleteCredential:
    """Tests for deleting credentials."""

    def test_delete_credential(self, temp_dir):
        """Test deleting a credential."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        mgr.set("test-agent", "API_KEY", "value")
        assert mgr.has("test-agent", "API_KEY")

        result = mgr.delete("test-agent", "API_KEY")
        assert result is True
        assert mgr.has("test-agent", "API_KEY") is False

    def test_delete_nonexistent(self, temp_dir):
        """Test deleting a nonexistent credential."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        result = mgr.delete("test-agent", "NONEXISTENT")
        assert result is False


class TestListCredentials:
    """Tests for listing credentials."""

    def test_list_credentials(self, temp_dir):
        """Test listing credentials for an agent."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        mgr.set("test-agent", "KEY1", "value1")
        mgr.set("test-agent", "KEY2", "value2")

        creds = mgr.list_credentials("test-agent")

        assert set(creds) == {"KEY1", "KEY2"}

    def test_list_empty(self, temp_dir):
        """Test listing credentials when none exist."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        creds = mgr.list_credentials("test-agent")
        assert creds == []


class TestGetAllForTools:
    """Tests for get_all_for_tools()."""

    def test_combines_stored_and_env(self, temp_dir):
        """Test that get_all_for_tools combines sources."""
        mgr = CredentialManager(base_dir=temp_dir / "creds")

        mgr.set("test-agent", "STORED_KEY", "stored-value")

        with patch.dict(os.environ, {"ENV_KEY": "env-value", "STORED_KEY": "env-override"}):
            all_creds = mgr.get_all_for_tools("test-agent")

        # Env var should override stored
        assert all_creds["STORED_KEY"] == "env-override"


class TestPromptForCredential:
    """Tests for interactive credential prompting."""

    @patch("supyagent.core.credentials.getpass.getpass")
    @patch("supyagent.core.credentials.Confirm.ask")
    def test_prompt_returns_value_and_persist(self, mock_confirm, mock_getpass, temp_dir):
        """Test that prompt returns value and persist flag."""
        mock_getpass.return_value = "secret123"
        mock_confirm.return_value = True

        mgr = CredentialManager(base_dir=temp_dir / "creds")
        result = mgr.prompt_for_credential("API_KEY", "Test description")

        assert result == ("secret123", True)

    @patch("supyagent.core.credentials.getpass.getpass")
    def test_prompt_skip_returns_none(self, mock_getpass, temp_dir):
        """Test that empty input returns None."""
        mock_getpass.return_value = ""

        mgr = CredentialManager(base_dir=temp_dir / "creds")
        result = mgr.prompt_for_credential("API_KEY", "Test description")

        assert result is None

    @patch("supyagent.core.credentials.getpass.getpass")
    @patch("supyagent.core.credentials.Confirm.ask")
    def test_prompt_no_persist(self, mock_confirm, mock_getpass, temp_dir):
        """Test prompt with no persistence."""
        mock_getpass.return_value = "secret123"
        mock_confirm.return_value = False

        mgr = CredentialManager(base_dir=temp_dir / "creds")
        result = mgr.prompt_for_credential("API_KEY", "Test description")

        assert result == ("secret123", False)


class TestCorruptedCredentials:
    """Tests for handling corrupted credential files."""

    def test_handles_corrupted_file(self, temp_dir):
        """Test that corrupted files are handled gracefully."""
        cred_dir = temp_dir / "creds"
        cred_dir.mkdir(parents=True)

        # Create a corrupted credential file
        cred_file = cred_dir / "test-agent.enc"
        cred_file.write_text("not encrypted data")

        # Also need a valid key file
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        (cred_dir / ".key").write_bytes(key)

        mgr = CredentialManager(base_dir=cred_dir)

        # Should not raise, should return empty
        creds = mgr.list_credentials("test-agent")
        assert creds == []
