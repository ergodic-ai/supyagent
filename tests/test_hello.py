"""
Tests for the supyagent hello wizard.
"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from supyagent.cli.main import cli


def _make_mgr(tmp_path):
    from supyagent.core.config import ConfigManager

    return ConfigManager(base_dir=tmp_path / "config")


def _apply_patches(stack, mgr, extra_patches=None):
    """Apply common patches via an ExitStack."""
    stack.enter_context(patch("supyagent.core.config.get_config_manager", return_value=mgr))
    stack.enter_context(patch("supyagent.cli.hello.get_config_manager", return_value=mgr))
    if extra_patches:
        mocks = {}
        for p in extra_patches:
            m = stack.enter_context(p)
            mocks[p.attribute or ""] = m
        return mocks
    return {}


@patch("supyagent.core.config._config_manager", None)
class TestHelloCommand:
    """Tests for the 'hello' CLI command."""

    def test_hello_fresh_setup_skip_all(self, tmp_path):
        """Fresh setup, skip service + skip model + skip agent."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        # Input flow: n=skip service (integrations auto-skipped),
        # 2=anthropic, 1=sonnet, n=skip agent
        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value=""),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="n\n2\n1\nn\n",
                )

        assert result.exit_code == 0
        assert "Welcome to Supyagent" in result.output

    def test_hello_fresh_creates_dirs(self, tmp_path):
        """Wizard creates agents/ and powers/ directories."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value=""),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="n\n2\n1\nn\n",
                )

            assert result.exit_code == 0
            assert Path("agents").exists()
            assert Path("powers").exists()

    def test_hello_existing_state_continue(self, tmp_path):
        """Already set up, user chooses 'continue' -> exits early."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            Path("agents").mkdir()
            Path("powers").mkdir()
            Path("powers/shell.py").write_text("# tool")

            with ExitStack() as stack:
                _apply_patches(stack, mgr)
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="continue\n",
                )

        assert result.exit_code == 0
        assert "Already set up" in result.output
        assert "Nothing to do" in result.output

    def test_hello_existing_state_redo(self, tmp_path):
        """Already set up, user chooses 'redo' -> runs all steps."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            Path("agents").mkdir()
            Path("powers").mkdir()
            Path("powers/shell.py").write_text("# tool")

            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value=""),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="redo\nn\n2\n1\nn\n",
                )

        assert result.exit_code == 0
        assert "Step 1" in result.output
        assert "Step 2" in result.output

    def test_hello_service_connection_success(self, tmp_path):
        """Test service connection step with successful device auth."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        device_data = {
            "device_code": "dev_abc123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://app.supyagent.com/device",
            "expires_in": 900,
            "interval": 5,
        }

        mock_client = MagicMock()
        mock_client.list_integrations.return_value = []

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.request_device_code", return_value=device_data),
                    patch("supyagent.cli.hello.poll_for_token", return_value="sk_live_test"),
                    patch("supyagent.cli.hello.webbrowser"),
                    patch("supyagent.cli.hello.ServiceClient", return_value=mock_client),
                    patch("supyagent.cli.hello.getpass.getpass", return_value=""),
                ])
                mock_store = stack.enter_context(
                    patch("supyagent.cli.hello.store_service_credentials")
                )
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="y\ndone\n2\n1\nn\n",
                )

        assert result.exit_code == 0
        assert "ABCD-1234" in result.output
        assert "Connected to service" in result.output
        mock_store.assert_called_once()

    def test_hello_service_connection_skip(self, tmp_path):
        """Test skipping service connection."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value=""),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="n\n2\n1\nn\n",
                )

        assert result.exit_code == 0
        assert "connect anytime" in result.output

    def test_hello_service_unreachable(self, tmp_path):
        """Test graceful handling when service is unreachable."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch(
                        "supyagent.cli.hello.request_device_code",
                        side_effect=Exception("Connection refused"),
                    ),
                    patch("supyagent.cli.hello.getpass.getpass", return_value=""),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="y\n2\n1\nn\n",
                )

        assert result.exit_code == 0
        assert "Could not reach service" in result.output

    def test_hello_model_selection_openai(self, tmp_path):
        """Test selecting OpenAI provider stores the key."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test-key"),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="n\n1\n1\nn\n",
                )

        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert mgr.get("OPENAI_API_KEY") == "sk-test-key"

    def test_hello_model_selection_anthropic(self, tmp_path):
        """Test selecting Anthropic provider."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-ant-test"),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="n\n2\n1\nn\n",
                )

        assert result.exit_code == 0
        assert "claude-sonnet-4-5" in result.output
        assert mgr.get("ANTHROPIC_API_KEY") == "sk-ant-test"

    def test_hello_model_custom(self, tmp_path):
        """Test custom model string entry."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="my-key-123"),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="n\n0\nollama/llama3\nMY_KEY\nn\n",
                )

        assert result.exit_code == 0
        assert mgr.get("MY_KEY") == "my-key-123"

    def test_hello_agent_creation(self, tmp_path):
        """Test agent creation step produces correct YAML."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="n\n2\n1\ny\nresearch assistant\nresearcher\n",
                )

            assert result.exit_code == 0
            agent_path = Path("agents/researcher.yaml")
            assert agent_path.exists()

            content = agent_path.read_text()
            assert "name: researcher" in content
            assert "research assistant" in content
            assert "service:" in content
            assert "enabled: false" in content

    def test_hello_agent_with_service(self, tmp_path):
        """Agent YAML has service.enabled: true when service is connected."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        from supyagent.core.service import SERVICE_API_KEY

        mgr.set(SERVICE_API_KEY, "sk_live_test")

        mock_client = MagicMock()
        mock_client.list_integrations.return_value = []

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.ServiceClient", return_value=mock_client),
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                result = runner.invoke(
                    cli,
                    ["hello"],
                    input="n\ndone\n2\n1\ny\nhelper\nhelper\n",
                )

            assert result.exit_code == 0
            agent_path = Path("agents/helper.yaml")
            assert agent_path.exists()
            content = agent_path.read_text()
            assert "enabled: true" in content


@patch("supyagent.core.config._config_manager", None)
class TestSetupAlias:
    """Test that 'setup' is an alias for 'hello'."""

    def test_setup_alias(self, tmp_path):
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value=""),
                ])
                result = runner.invoke(
                    cli,
                    ["setup"],
                    input="n\n2\n1\nn\n",
                )

        assert result.exit_code == 0
        assert "Welcome to Supyagent" in result.output


@patch("supyagent.core.config._config_manager", None)
class TestInitCommand:
    """Test that 'init' delegates to wizard or quick init."""

    def test_init_without_flags_runs_wizard(self, tmp_path):
        """'supyagent init' (no flags) runs the wizard."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value=""),
                ])
                result = runner.invoke(
                    cli,
                    ["init"],
                    input="n\n2\n1\nn\n",
                )

        assert result.exit_code == 0
        assert "Welcome to Supyagent" in result.output

    def test_init_quick_skips_wizard(self):
        """'supyagent init --quick' runs old behavior."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "--quick"])

            assert result.exit_code == 0
            assert "Initializing supyagent" in result.output
            assert "Welcome to Supyagent" not in result.output
            assert Path("agents").exists()
            assert Path("powers").exists()

    def test_init_force_skips_wizard(self):
        """'supyagent init --force' runs old behavior."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "--force"])

            assert result.exit_code == 0
            assert "Initializing supyagent" in result.output
            assert "Welcome to Supyagent" not in result.output

    def test_init_custom_tools_dir_skips_wizard(self):
        """'supyagent init -t my_tools' runs old behavior."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "-t", "my_tools"])

            assert result.exit_code == 0
            assert "Initializing supyagent" in result.output
            assert "Welcome to Supyagent" not in result.output
            assert Path("my_tools").exists()


@patch("supyagent.core.config._config_manager", None)
class TestIntegrationPolling:
    """Test integration connect polling behavior."""

    def test_integration_already_connected(self, tmp_path):
        """Already connected provider shows status."""
        from supyagent.cli.hello import _step_integrations
        from supyagent.core.config import ConfigManager
        from supyagent.core.service import SERVICE_API_KEY

        mgr = ConfigManager(base_dir=tmp_path / "config")
        mgr.set(SERVICE_API_KEY, "sk_live_test")

        mock_client = MagicMock()
        mock_client.list_integrations.return_value = [
            {"provider": "google", "status": "active", "services": ["gmail.read"]},
        ]

        with ExitStack() as stack:
            _apply_patches(stack, mgr, [
                patch("supyagent.cli.hello.ServiceClient", return_value=mock_client),
                patch("supyagent.cli.hello.Prompt.ask", return_value="done"),
            ])
            result = _step_integrations(service_connected=True, statuses={})

        assert "google" in result

    def test_integration_skipped_when_not_connected(self):
        """Integrations skipped when not connected to service."""
        from supyagent.cli.hello import _step_integrations

        result = _step_integrations(service_connected=False, statuses={})
        assert result == []
