"""
Tests for the supyagent hello wizard (new workspace-centric flow).

Uses questionary mocks to simulate interactive arrow-key selection.
"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from supyagent.cli.main import cli


def _make_mgr(tmp_path):
    from supyagent.core.config import ConfigManager

    return ConfigManager(base_dir=tmp_path / "config")


def _make_registry(tmp_path):
    from supyagent.core.model_registry import ModelRegistry

    return ModelRegistry(base_dir=tmp_path / "registry")


def _apply_patches(stack, mgr, registry=None, extra_patches=None):
    """Apply common patches via an ExitStack."""
    stack.enter_context(patch("supyagent.core.config.get_config_manager", return_value=mgr))
    stack.enter_context(patch("supyagent.cli.hello.get_config_manager", return_value=mgr))
    if registry is not None:
        stack.enter_context(
            patch("supyagent.cli.hello.get_model_registry", return_value=registry)
        )
    if extra_patches:
        mocks = {}
        for p in extra_patches:
            m = stack.enter_context(p)
            mocks[p.attribute or ""] = m
        return mocks
    return {}


def _mock_questionary_select(stack, return_values):
    """Mock questionary.select to return values from a list sequentially.

    Each call to questionary.select(...).ask() returns the next value.
    """
    mock_select = stack.enter_context(patch("supyagent.cli.hello.questionary.select"))
    ask_mock = MagicMock(side_effect=return_values)
    mock_select.return_value.ask = ask_mock
    return mock_select


def _mock_questionary_checkbox(stack, return_values):
    """Mock questionary.checkbox to return values from a list sequentially."""
    mock_checkbox = stack.enter_context(patch("supyagent.cli.hello.questionary.checkbox"))
    ask_mock = MagicMock(side_effect=return_values)
    mock_checkbox.return_value.ask = ask_mock
    return mock_checkbox


# Standard wizard flow selections:
# Step 1: skip service -> "n"
# Step 2: select Anthropic -> select Sonnet -> done adding -> (only 1 model, no default/role prompts)
# Step 3: env vars -> "done" (skip adding)
# Step 4: select "coding" profile
# Step 5: skip goals -> ""
# Step 6: select "yolo" execution mode, heartbeat -> "n"
# Summary: select "exit"

# questionary.select calls in order for standard wizard:
# 1. provider select: "Anthropic"
# 2. provider select loop (done): "done"
# 3. env vars action: "done"
# 4. workspace profile: "coding"
# 5. execution mode: "yolo"
# 6. summary "what's next": "exit"
STANDARD_SELECT_SEQUENCE = ["Anthropic", "done", "done", "coding", "yolo", "exit"]

# questionary.checkbox calls: one for model multi-select
STANDARD_CHECKBOX_SEQUENCE = [["anthropic/claude-sonnet-4-5-20250929"]]


@patch("supyagent.core.config._config_manager", None)
class TestHelloCommand:
    """Tests for the new 5-step 'hello' wizard."""

    def test_hello_welcome_banner(self, tmp_path):
        """Fresh setup shows welcome banner."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\n\nn\n",  # skip service, skip goals, no heartbeat
                )

        assert result.exit_code == 0
        assert "Welcome to Supyagent" in result.output

    def test_hello_creates_workspace(self, tmp_path):
        """Wizard creates .supyagent/, agents/, powers/, GOALS.md."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\n\nn\n",
                )

            assert result.exit_code == 0
            assert Path("agents").exists()
            assert Path("powers").exists()
            assert Path("GOALS.md").exists()
            assert Path(".supyagent/workspace.yaml").exists()

    def test_hello_service_skip(self, tmp_path):
        """Skipping service shows connect-later message."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\n\nn\n",
                )

        assert result.exit_code == 0
        assert "supyagent connect" in result.output

    def test_hello_service_connection_success(self, tmp_path):
        """Test service connection step with successful device auth."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

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
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.request_device_code", return_value=device_data),
                    patch("supyagent.cli.hello.poll_for_token", return_value="sk_live_test"),
                    patch("supyagent.cli.hello.webbrowser"),
                    patch("supyagent.cli.hello.ServiceClient", return_value=mock_client),
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                mock_store = stack.enter_context(
                    patch("supyagent.cli.hello.store_service_credentials")
                )
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    # yes connect, done integrations, skip goals
                    input="y\ndone\n\nn\n",  # yes connect, done integrations, skip goals, no heartbeat
                )

        assert result.exit_code == 0
        assert "ABCD-1234" in result.output
        assert "Connected to Supyagent Service" in result.output
        mock_store.assert_called_once()

    def test_hello_service_unreachable(self, tmp_path):
        """Graceful handling when service is unreachable."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch(
                        "supyagent.cli.hello.request_device_code",
                        side_effect=Exception("Connection refused"),
                    ),
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="y\n\nn\n",  # yes connect (fails), skip goals, no heartbeat
                )

        assert result.exit_code == 0
        assert "Could not reach service" in result.output

    def test_hello_model_registration(self, tmp_path):
        """Test that model setup registers model and sets default."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\n\nn\n",
                )

        assert result.exit_code == 0
        assert len(registry.list_models()) > 0
        assert registry.get_default() is not None

    def test_hello_coding_profile_agents(self, tmp_path):
        """Coding profile creates assistant, coder, planner agents."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\n\nn\n",
                )

            assert result.exit_code == 0
            assert Path("agents/assistant.yaml").exists()
            assert Path("agents/coder.yaml").exists()
            assert Path("agents/planner.yaml").exists()

    def test_hello_goals_with_content(self, tmp_path):
        """Test that user goals are saved to GOALS.md."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\nBuild a CLI tool\nn\n",  # skip service, type goals, no heartbeat
                )

            assert result.exit_code == 0
            goals = Path("GOALS.md").read_text()
            assert "Build a CLI tool" in goals

    def test_hello_existing_workspace_menu(self, tmp_path):
        """Already set up, user sees re-run menu with interactive selection."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            Path(".supyagent").mkdir()
            Path(".supyagent/workspace.yaml").write_text("name: test\nprofile: coding\n")
            Path("agents").mkdir()
            Path("powers").mkdir()
            Path("powers/shell.py").write_text("# tool")

            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry)
                # questionary.select for re-run menu: exit
                _mock_questionary_select(stack, ["exit"])
                result = runner.invoke(cli, ["hello", "--cli"])

        assert result.exit_code == 0

    def test_hello_quick_flag(self, tmp_path):
        """Test --quick flag runs non-interactive setup."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry)
                result = runner.invoke(cli, ["hello", "--quick"])

        assert result.exit_code == 0
        assert "Workspace initialized" in result.output
        assert "Welcome to Supyagent" not in result.output

    def test_hello_quick_creates_structure(self, tmp_path):
        """Quick wizard creates agents, powers, GOALS.md, workspace.yaml."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry)
                result = runner.invoke(cli, ["hello", "--quick"])

            assert result.exit_code == 0
            assert Path("agents").exists()
            assert Path("powers").exists()
            assert Path("GOALS.md").exists()
            assert Path(".supyagent/workspace.yaml").exists()

    def test_hello_shows_step_numbers(self, tmp_path):
        """Wizard shows step progression."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\n\nn\n",
                )

        assert result.exit_code == 0
        assert "Step 1" in result.output
        assert "Step 2" in result.output
        assert "Step 3" in result.output

    def test_hello_summary_panel(self, tmp_path):
        """Wizard ends with a summary panel."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\n\nn\n",
                )

        assert result.exit_code == 0
        assert "Workspace Ready" in result.output

    def test_hello_multiple_models_triggers_default_select(self, tmp_path):
        """When multiple models are registered, user is asked to choose default."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        # Select two providers worth of models, then done, default, roles, env vars, profile, mode, exit
        select_seq = [
            "Anthropic",  # pick anthropic
            "OpenAI",  # pick openai too
            "done",  # done adding
            "anthropic/claude-sonnet-4-5-20250929",  # default model
            None,  # fast role: skip
            None,  # smart role: skip
            None,  # reasoning role: skip
            "done",  # env vars: done
            "coding",  # workspace profile
            "yolo",  # execution mode
            "exit",  # summary
        ]
        checkbox_seq = [
            ["anthropic/claude-sonnet-4-5-20250929"],  # anthropic models
            ["gpt-4.1"],  # openai models
        ]

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, select_seq)
                _mock_questionary_checkbox(stack, checkbox_seq)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\n\nn\n",  # skip service, skip goals, no heartbeat
                )

        assert result.exit_code == 0
        assert len(registry.list_models()) == 2
        assert registry.get_default() == "anthropic/claude-sonnet-4-5-20250929"

    def test_hello_openrouter_models(self, tmp_path):
        """OpenRouter provider registers models correctly."""
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        # 2 models → triggers default select + 3 role assignments + env vars
        select_seq = [
            "OpenRouter",  # provider
            "done",  # done adding
            "openrouter/deepseek/deepseek-chat",  # default model
            None,  # fast role: skip
            None,  # smart role: skip
            None,  # reasoning role: skip
            "done",  # env vars: done
            "coding",  # workspace profile
            "yolo",  # execution mode
            "exit",  # summary
        ]
        checkbox_seq = [["openrouter/deepseek/deepseek-chat", "openrouter/google/gemini-2.5-flash"]]

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, select_seq)
                _mock_questionary_checkbox(stack, checkbox_seq)
                result = runner.invoke(
                    cli,
                    ["hello", "--cli"],
                    input="n\n\nn\n",
                )

        assert result.exit_code == 0
        models = registry.list_models()
        assert "openrouter/deepseek/deepseek-chat" in models
        assert "openrouter/google/gemini-2.5-flash" in models


@patch("supyagent.core.config._config_manager", None)
class TestSetupAlias:
    """Test that 'setup' is an alias for 'hello'."""

    def test_setup_alias(self, tmp_path):
        runner = CliRunner()
        mgr = _make_mgr(tmp_path)
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["setup", "--cli"],
                    input="n\n\nn\n",
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
        registry = _make_registry(tmp_path)

        with runner.isolated_filesystem():
            with ExitStack() as stack:
                _apply_patches(stack, mgr, registry, [
                    patch("supyagent.cli.hello.getpass.getpass", return_value="sk-test"),
                ])
                _mock_questionary_select(stack, STANDARD_SELECT_SEQUENCE)
                _mock_questionary_checkbox(stack, STANDARD_CHECKBOX_SEQUENCE)
                result = runner.invoke(
                    cli,
                    ["init"],
                    input="n\n\nn\n",  # init calls run_hello_wizard() directly (no browser)
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
class TestIntegrationInline:
    """Test integration connect behavior (now inline with service step)."""

    def test_offer_integrations_shows_providers(self, tmp_path):
        """Integration list is shown after service connection."""
        from supyagent.cli.hello import _offer_integrations

        mock_client = MagicMock()
        mock_client.list_integrations.return_value = [
            {"provider": "google", "status": "active", "services": ["gmail.read"]},
        ]

        mgr = _make_mgr(tmp_path)
        with ExitStack() as stack:
            _apply_patches(stack, mgr, extra_patches=[
                patch("supyagent.cli.hello.ServiceClient", return_value=mock_client),
                patch("supyagent.cli.hello.Prompt.ask", return_value="done"),
            ])
            # Should not raise; just returns
            _offer_integrations("sk_live_test", "https://app.supyagent.com")
