"""Tests for the skills generation module."""

import json

from supyagent.cli.skills import (
    _build_provider_descriptions,
    _extract_action,
    _generate_example_args,
    _placeholder_for_type,
    generate_skill_md,
)


def _make_tool(name, description, provider, properties=None, required=None):
    """Helper to create a mock tool in OpenAI function-calling format."""
    params = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
        "metadata": {"provider": provider, "service": "test", "method": "POST", "path": "/test"},
    }


class TestExtractAction:
    def test_short_description(self):
        assert _extract_action("Send a message to a channel.") == "send a message to a channel"

    def test_long_description_truncated(self):
        desc = "Send a very detailed and comprehensive message to a Slack channel or user in the workspace for collaboration purposes."
        result = _extract_action(desc)
        assert len(result.split()) <= 5

    def test_empty_description(self):
        assert _extract_action("") == ""


class TestBuildProviderDescriptions:
    def test_single_provider(self):
        tools = [
            _make_tool("slack_send_message", "Send a message.", "slack"),
            _make_tool("slack_list_channels", "List channels.", "slack"),
        ]
        descs = _build_provider_descriptions({"slack": tools})
        assert "Slack" in descs["slack"]
        assert "send a message" in descs["slack"]
        assert "list channels" in descs["slack"]

    def test_caps_at_four_actions(self):
        tools = [_make_tool(f"tool_{i}", f"Action {i}.", "test") for i in range(6)]
        descs = _build_provider_descriptions({"test": tools})
        # Count commas to verify at most 4 items
        assert descs["test"].count(",") <= 3


class TestPlaceholderForType:
    def test_email_parameter(self):
        assert _placeholder_for_type("to", "string", {}) == "user@example.com"
        assert _placeholder_for_type("email", "string", {}) == "user@example.com"

    def test_channel_parameter(self):
        assert _placeholder_for_type("channel", "string", {}) == "C0123456789"

    def test_text_parameter(self):
        assert _placeholder_for_type("text", "string", {}) == "Hello world"

    def test_id_parameter(self):
        assert _placeholder_for_type("user_id", "string", {}) == "abc123"

    def test_enum_parameter(self):
        assert _placeholder_for_type("status", "string", {"enum": ["active", "inactive"]}) == "active"

    def test_integer(self):
        assert _placeholder_for_type("limit", "integer", {}) == 10

    def test_boolean(self):
        assert _placeholder_for_type("flag", "boolean", {}) is True

    def test_array(self):
        assert _placeholder_for_type("items", "array", {}) == []


class TestGenerateExampleArgs:
    def test_empty_params(self):
        assert _generate_example_args({}) == "{}"

    def test_required_only(self):
        params = {
            "properties": {
                "channel": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["channel", "text"],
        }
        result = json.loads(_generate_example_args(params))
        assert "channel" in result
        assert "text" in result

    def test_includes_one_optional(self):
        params = {
            "properties": {
                "channel": {"type": "string"},
                "text": {"type": "string"},
                "opt1": {"type": "string"},
                "opt2": {"type": "string"},
            },
            "required": ["channel", "text"],
        }
        result = json.loads(_generate_example_args(params))
        assert "channel" in result
        assert "text" in result
        optional_count = sum(1 for k in result if k not in ("channel", "text"))
        assert optional_count <= 1


class TestGenerateSkillMd:
    def test_frontmatter(self):
        tools = [_make_tool("slack_send_message", "Send a message.", "slack")]
        md = generate_skill_md(tools)
        assert md.startswith("---\n")
        assert "name: supy" in md
        assert "description:" in md
        assert "Slack" in md

    def test_provider_section(self):
        tools = [
            _make_tool(
                "slack_send_message",
                "Send a message to a Slack channel.",
                "slack",
                properties={"channel": {"type": "string", "description": "Channel ID"}, "text": {"type": "string", "description": "Message"}},
                required=["channel", "text"],
            )
        ]
        md = generate_skill_md(tools)
        assert "## Slack" in md
        assert "### slack_send_message" in md
        assert "| `channel` | string | yes |" in md
        assert "| `text` | string | yes |" in md
        assert "supyagent service run slack_send_message" in md

    def test_multiple_providers_sorted(self):
        tools = [
            _make_tool("slack_send", "Send.", "slack"),
            _make_tool("github_list", "List repos.", "github"),
        ]
        md = generate_skill_md(tools)
        github_pos = md.index("## GitHub")
        slack_pos = md.index("## Slack")
        assert github_pos < slack_pos  # alphabetical

    def test_empty_parameters(self):
        tools = [_make_tool("inbox_list", "List events.", "inbox")]
        md = generate_skill_md(tools)
        assert "supyagent service run inbox_list '{}'" in md

    def test_how_to_use_section(self):
        tools = [_make_tool("test_tool", "Test.", "test")]
        md = generate_skill_md(tools)
        assert "supyagent service run <tool_name>" in md
        assert '"ok": true' in md
