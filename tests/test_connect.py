"""
Tests for supyagent CLI connect/disconnect/status commands.
"""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from supyagent.cli.main import cli

# Imports in CLI commands are local (inside functions), so we patch
# at the source module: supyagent.core.service.*


@patch("supyagent.core.config._config_manager", None)
class TestConnectCommand:
    def test_connect_success(self):
        runner = CliRunner()

        device_data = {
            "device_code": "dev_abc123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://app.supyagent.com/device",
            "expires_in": 900,
            "interval": 5,
        }

        mock_client = MagicMock()
        mock_client.discover_tools.return_value = [
            {
                "type": "function",
                "function": {"name": "gmail_list_messages"},
                "metadata": {"provider": "google", "service": "gmail"},
            },
            {
                "type": "function",
                "function": {"name": "slack_send_message"},
                "metadata": {"provider": "slack", "service": "messages"},
            },
        ]

        with patch("supyagent.core.service.request_device_code", return_value=device_data), \
             patch("supyagent.core.service.poll_for_token", return_value="sk_live_new_key"), \
             patch("supyagent.core.service.store_service_credentials") as mock_store, \
             patch("supyagent.core.service.ServiceClient", return_value=mock_client), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["connect"])

        assert result.exit_code == 0
        assert "ABCD-1234" in result.output
        assert "Connected" in result.output
        assert "google" in result.output
        assert "slack" in result.output
        mock_store.assert_called_once_with("sk_live_new_key", None)

    def test_connect_with_custom_url(self):
        runner = CliRunner()

        device_data = {
            "device_code": "dev_abc",
            "user_code": "TEST-CODE",
            "verification_uri": "https://custom.example.com/device",
            "expires_in": 900,
            "interval": 5,
        }

        mock_client = MagicMock()
        mock_client.discover_tools.return_value = []

        with patch("supyagent.core.service.request_device_code", return_value=device_data) as mock_req, \
             patch("supyagent.core.service.poll_for_token", return_value="sk_live_key"), \
             patch("supyagent.core.service.store_service_credentials") as mock_store, \
             patch("supyagent.core.service.ServiceClient", return_value=mock_client), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["connect", "--url", "https://custom.example.com"])

        assert result.exit_code == 0
        mock_req.assert_called_once_with("https://custom.example.com")
        mock_store.assert_called_once_with("sk_live_key", "https://custom.example.com")

    def test_connect_service_unreachable(self):
        runner = CliRunner()

        with patch(
            "supyagent.core.service.request_device_code",
            side_effect=Exception("Connection refused"),
        ):
            result = runner.invoke(cli, ["connect"])

        assert result.exit_code != 0
        assert "Could not reach" in result.output

    def test_connect_denied(self):
        runner = CliRunner()

        device_data = {
            "device_code": "dev_abc",
            "user_code": "DENY-CODE",
            "verification_uri": "https://app.supyagent.com/device",
            "expires_in": 900,
            "interval": 5,
        }

        with patch("supyagent.core.service.request_device_code", return_value=device_data), \
             patch(
                 "supyagent.core.service.poll_for_token",
                 side_effect=PermissionError("denied"),
             ), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["connect"])

        assert result.exit_code != 0
        assert "denied" in result.output.lower()

    def test_connect_expired(self):
        runner = CliRunner()

        device_data = {
            "device_code": "dev_abc",
            "user_code": "EXP-CODE",
            "verification_uri": "https://app.supyagent.com/device",
            "expires_in": 900,
            "interval": 5,
        }

        with patch("supyagent.core.service.request_device_code", return_value=device_data), \
             patch(
                 "supyagent.core.service.poll_for_token",
                 side_effect=TimeoutError("expired"),
             ), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["connect"])

        assert result.exit_code != 0
        assert "expired" in result.output.lower()

    def test_connect_no_integrations(self):
        runner = CliRunner()

        device_data = {
            "device_code": "dev_abc",
            "user_code": "TEST-CODE",
            "verification_uri": "https://app.supyagent.com/device",
            "expires_in": 900,
            "interval": 5,
        }

        mock_client = MagicMock()
        mock_client.discover_tools.return_value = []

        with patch("supyagent.core.service.request_device_code", return_value=device_data), \
             patch("supyagent.core.service.poll_for_token", return_value="sk_live_key"), \
             patch("supyagent.core.service.store_service_credentials"), \
             patch("supyagent.core.service.ServiceClient", return_value=mock_client), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["connect"])

        assert result.exit_code == 0
        assert "Connected" in result.output
        assert "No integrations" in result.output


@patch("supyagent.core.config._config_manager", None)
class TestDisconnectCommand:
    def test_disconnect_when_connected(self):
        runner = CliRunner()

        with patch(
            "supyagent.core.service.clear_service_credentials", return_value=True
        ):
            result = runner.invoke(cli, ["disconnect"])

        assert result.exit_code == 0
        assert "Disconnected" in result.output

    def test_disconnect_when_not_connected(self):
        runner = CliRunner()

        with patch(
            "supyagent.core.service.clear_service_credentials", return_value=False
        ):
            result = runner.invoke(cli, ["disconnect"])

        assert result.exit_code == 0
        assert "Not currently connected" in result.output


@patch("supyagent.core.config._config_manager", None)
class TestStatusCommand:
    def test_status_not_connected(self, tmp_path):
        runner = CliRunner()

        from supyagent.core.config import ConfigManager

        mgr = ConfigManager(base_dir=tmp_path / "config")

        with patch("supyagent.cli.main.ConfigManager", return_value=mgr):
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Not connected" in result.output

    def test_status_connected_with_tools(self, tmp_path):
        runner = CliRunner()

        from supyagent.core.config import ConfigManager
        from supyagent.core.service import SERVICE_API_KEY

        mgr = ConfigManager(base_dir=tmp_path / "config")
        mgr.set(SERVICE_API_KEY, "sk_live_test")

        mock_client = MagicMock()
        mock_client.health_check.return_value = True
        mock_client.discover_tools.return_value = [
            {
                "type": "function",
                "function": {"name": "gmail_list_messages"},
                "metadata": {"provider": "google", "service": "gmail"},
            },
            {
                "type": "function",
                "function": {"name": "gmail_get_message"},
                "metadata": {"provider": "google", "service": "gmail"},
            },
            {
                "type": "function",
                "function": {"name": "slack_send_message"},
                "metadata": {"provider": "slack", "service": "messages"},
            },
        ]

        with patch("supyagent.cli.main.ConfigManager", return_value=mgr), \
             patch("supyagent.core.service.ServiceClient", return_value=mock_client):
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Connected" in result.output
        assert "google" in result.output
        assert "slack" in result.output
        assert "3 tools total" in result.output

    def test_status_service_unreachable(self, tmp_path):
        runner = CliRunner()

        from supyagent.core.config import ConfigManager
        from supyagent.core.service import SERVICE_API_KEY

        mgr = ConfigManager(base_dir=tmp_path / "config")
        mgr.set(SERVICE_API_KEY, "sk_live_test")

        mock_client = MagicMock()
        mock_client.health_check.return_value = False

        with patch("supyagent.cli.main.ConfigManager", return_value=mgr), \
             patch("supyagent.core.service.ServiceClient", return_value=mock_client):
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "not reachable" in result.output

    def test_status_connected_no_tools(self, tmp_path):
        runner = CliRunner()

        from supyagent.core.config import ConfigManager
        from supyagent.core.service import SERVICE_API_KEY

        mgr = ConfigManager(base_dir=tmp_path / "config")
        mgr.set(SERVICE_API_KEY, "sk_live_test")

        mock_client = MagicMock()
        mock_client.health_check.return_value = True
        mock_client.discover_tools.return_value = []

        with patch("supyagent.cli.main.ConfigManager", return_value=mgr), \
             patch("supyagent.core.service.ServiceClient", return_value=mock_client):
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "No integrations" in result.output
