"""Tests for the skills generation module."""

import json
from pathlib import Path

from supyagent.cli.skills import (
    SKILL_FILE_PREFIX,
    _build_provider_descriptions,
    _extract_action,
    _generate_example_args,
    _placeholder_for_type,
    _skill_display_name,
    _skill_key,
    detect_ai_tool_folders,
    generate_skill_files,
    generate_skill_md,
    write_skills_to_dir,
)


def _make_tool(name, description, provider, properties=None, required=None, service="test"):
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
        "metadata": {
            "provider": provider,
            "service": service,
            "method": "POST",
            "path": "/test",
        },
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
        assert (
            _placeholder_for_type("status", "string", {"enum": ["active", "inactive"]}) == "active"
        )

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


class TestSkillKey:
    def test_google_splits_by_service(self):
        tool = _make_tool("gmail_send", "Send.", "google", service="gmail")
        assert _skill_key(tool) == "gmail"

    def test_microsoft_splits_by_service(self):
        tool = _make_tool("outlook_send", "Send.", "microsoft", service="mail")
        assert _skill_key(tool) == "mail"

    def test_other_providers_use_provider(self):
        tool = _make_tool("slack_send", "Send.", "slack", service="messages")
        assert _skill_key(tool) == "slack"

    def test_google_slides_splits_by_service(self):
        tool = _make_tool("slides_get_presentation", "Get presentation.", "google", service="slides")
        assert _skill_key(tool) == "slides"

    def test_google_falls_back_to_provider(self):
        tool = _make_tool("google_tool", "Do.", "google")
        # service defaults to "test" in _make_tool, so it returns "test"
        # But if service is missing, should fall back to provider
        tool["metadata"]["service"] = ""
        assert _skill_key(tool) == ""
        # With no service key at all, falls back to provider
        del tool["metadata"]["service"]
        assert _skill_key(tool) == "google"


class TestSkillDisplayName:
    def test_service_display_name(self):
        assert _skill_display_name("gmail") == "Gmail"
        assert _skill_display_name("calendar") == "Google Calendar"
        assert _skill_display_name("drive") == "Google Drive"
        assert _skill_display_name("slides") == "Google Slides"

    def test_provider_display_name(self):
        assert _skill_display_name("slack") == "Slack"
        assert _skill_display_name("hubspot") == "HubSpot"

    def test_unknown_key(self):
        assert _skill_display_name("some_thing") == "Some Thing"


class TestGenerateSkillFiles:
    def test_single_provider(self):
        tools = [_make_tool("slack_send", "Send a message.", "slack", service="messages")]
        files = generate_skill_files(tools)
        assert "supy-cloud-slack" in files
        assert len(files) == 1

    def test_google_split_by_service(self):
        tools = [
            _make_tool("gmail_send", "Send email.", "google", service="gmail"),
            _make_tool("calendar_list", "List events.", "google", service="calendar"),
        ]
        files = generate_skill_files(tools)
        assert "supy-cloud-gmail" in files
        assert "supy-cloud-calendar" in files
        assert len(files) == 2

    def test_non_google_grouped_by_provider(self):
        tools = [
            _make_tool("slack_send", "Send.", "slack", service="messages"),
            _make_tool("slack_list_channels", "List channels.", "slack", service="channels"),
        ]
        files = generate_skill_files(tools)
        assert "supy-cloud-slack" in files
        assert len(files) == 1
        # Both tools should be in the same file
        assert "slack_send" in files["supy-cloud-slack"]
        assert "slack_list_channels" in files["supy-cloud-slack"]

    def test_each_file_self_contained(self):
        tools = [
            _make_tool("slack_send", "Send.", "slack", service="messages"),
            _make_tool("gmail_send", "Send email.", "google", service="gmail"),
        ]
        files = generate_skill_files(tools)
        for dir_name, content in files.items():
            assert content.startswith("---\n"), f"{dir_name} missing frontmatter"
            assert "name: supy-" in content, f"{dir_name} missing name field"
            assert "description:" in content, f"{dir_name} missing description"
            assert "supyagent service run" in content, f"{dir_name} missing execution instructions"
            assert '"ok": true' in content, f"{dir_name} missing output format"

    def test_dir_naming(self):
        tools = [
            _make_tool("linear_list", "List issues.", "linear", service="issues"),
            _make_tool("gmail_send", "Send.", "google", service="gmail"),
            _make_tool("drive_list", "List files.", "google", service="drive"),
        ]
        files = generate_skill_files(tools)
        assert set(files.keys()) == {"supy-cloud-linear", "supy-cloud-gmail", "supy-cloud-drive"}

    def test_frontmatter_name_matches_key(self):
        tools = [_make_tool("slack_send", "Send.", "slack", service="messages")]
        files = generate_skill_files(tools)
        content = files["supy-cloud-slack"]
        assert "name: supy-slack" in content

    def test_display_name_in_header(self):
        tools = [_make_tool("gmail_send", "Send.", "google", service="gmail")]
        files = generate_skill_files(tools)
        assert "# Gmail" in files["supy-cloud-gmail"]

    def test_slides_produces_own_directory(self):
        tools = [
            _make_tool(
                "slides_get_presentation", "Get presentation.", "google", service="slides"
            ),
            _make_tool("slides_get_page", "Get page.", "google", service="slides"),
        ]
        files = generate_skill_files(tools)
        assert "supy-cloud-slides" in files
        content = files["supy-cloud-slides"]
        assert "# Google Slides" in content
        assert "slides_get_presentation" in content
        assert "slides_get_page" in content

    def test_multiple_providers_sorted(self):
        tools = [
            _make_tool("slack_send", "Send.", "slack", service="messages"),
            _make_tool("github_list", "List repos.", "github", service="repos"),
        ]
        files = generate_skill_files(tools)
        keys = list(files.keys())
        assert keys.index("supy-cloud-github") < keys.index("supy-cloud-slack")


class TestGenerateSkillMd:
    def test_frontmatter(self):
        tools = [_make_tool("slack_send_message", "Send a message.", "slack")]
        md = generate_skill_md(tools)
        assert md.startswith("---\n")
        assert "name: supy-" in md
        assert "description:" in md
        assert "Slack" in md

    def test_provider_section(self):
        tools = [
            _make_tool(
                "slack_send_message",
                "Send a message to a Slack channel.",
                "slack",
                properties={
                    "channel": {"type": "string", "description": "Channel ID"},
                    "text": {"type": "string", "description": "Message"},
                },
                required=["channel", "text"],
            )
        ]
        md = generate_skill_md(tools)
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
        github_pos = md.index("# GitHub")
        slack_pos = md.index("# Slack")
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


class TestDetectAiToolFolders:
    def test_detect_claude(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        detected = detect_ai_tool_folders(tmp_path)
        assert len(detected) == 1
        assert detected[0]["name"] == "Claude Code"

    def test_detect_multiple(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".cursor").mkdir()
        detected = detect_ai_tool_folders(tmp_path)
        assert len(detected) == 2
        names = {e["name"] for e in detected}
        assert names == {"Claude Code", "Cursor"}

    def test_detect_none(self, tmp_path):
        assert detect_ai_tool_folders(tmp_path) == []

    def test_detect_all(self, tmp_path):
        for folder in [".claude", ".cursor", ".agents", ".copilot", ".windsurf"]:
            (tmp_path / folder).mkdir()
        assert len(detect_ai_tool_folders(tmp_path)) == 5

    def test_ignores_files(self, tmp_path):
        """Files named like AI tool folders should not be detected."""
        (tmp_path / ".claude").write_text("not a directory")
        assert detect_ai_tool_folders(tmp_path) == []


class TestWriteSkillsToDir:
    def test_writes_skill_files(self, tmp_path):
        files = {"supy-cloud-slack": "# Slack\ncontent"}
        write_skills_to_dir(tmp_path, files)
        assert (tmp_path / "supy-cloud-slack" / "SKILL.md").exists()
        assert (tmp_path / "supy-cloud-slack" / "SKILL.md").read_text() == "# Slack\ncontent"

    def test_cleans_stale_dirs(self, tmp_path):
        (tmp_path / "supy-cloud-old").mkdir()
        (tmp_path / "supy-cloud-old" / "SKILL.md").write_text("old")
        files = {"supy-cloud-new": "# New"}
        write_skills_to_dir(tmp_path, files)
        assert not (tmp_path / "supy-cloud-old").exists()
        assert (tmp_path / "supy-cloud-new" / "SKILL.md").exists()

    def test_cleans_legacy_prefix(self, tmp_path):
        (tmp_path / "supy-legacy").mkdir()
        (tmp_path / "supy-legacy" / "SKILL.md").write_text("legacy")
        write_skills_to_dir(tmp_path, {})
        assert not (tmp_path / "supy-legacy").exists()

    def test_cleans_legacy_flat_files(self, tmp_path):
        (tmp_path / "supy.md").write_text("legacy")
        (tmp_path / "supy-old.md").write_text("legacy")
        write_skills_to_dir(tmp_path, {})
        assert not (tmp_path / "supy.md").exists()
        assert not (tmp_path / "supy-old.md").exists()

    def test_preserves_unrelated_files(self, tmp_path):
        (tmp_path / "unrelated.md").write_text("keep")
        (tmp_path / "other-dir").mkdir()
        files = {"supy-cloud-test": "# Test"}
        write_skills_to_dir(tmp_path, files)
        assert (tmp_path / "unrelated.md").exists()
        assert (tmp_path / "other-dir").exists()

    def test_creates_output_dir(self, tmp_path):
        output = tmp_path / "nested" / "skills"
        files = {"supy-cloud-test": "# Test"}
        write_skills_to_dir(output, files)
        assert (output / "supy-cloud-test" / "SKILL.md").exists()
