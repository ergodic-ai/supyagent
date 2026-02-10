"""
Skills generation for Claude Code Agent Skills format.

Generates .claude/skills/supy-{key}/SKILL.md files from connected service integrations.
"""

from __future__ import annotations

import json
from typing import Any

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "google": "Google",
    "slack": "Slack",
    "github": "GitHub",
    "discord": "Discord",
    "notion": "Notion",
    "twitter": "Twitter/X",
    "linkedin": "LinkedIn",
    "microsoft": "Microsoft 365",
    "telegram": "Telegram",
    "hubspot": "HubSpot",
    "whatsapp": "WhatsApp",
    "twilio": "Twilio",
    "resend": "Resend",
    "inbox": "AI Inbox",
}

# Providers whose tools should be split by service rather than grouped together.
SPLIT_BY_SERVICE_PROVIDERS: set[str] = {"google", "microsoft"}

# Display names for individual services (used when a provider is split).
SERVICE_DISPLAY_NAMES: dict[str, str] = {
    "gmail": "Gmail",
    "calendar": "Google Calendar",
    "drive": "Google Drive",
    "mail": "Outlook Mail",
}

# Prefix for generated skill directories — used to identify managed files for cleanup.
SKILL_FILE_PREFIX = "supy-cloud-"


def _skill_key(tool: dict) -> str:
    """Return the grouping key for a tool.

    Google and Microsoft tools are grouped by their service field (e.g. gmail, calendar).
    All other providers are grouped by provider name.
    """
    metadata = tool.get("metadata", {})
    provider = metadata.get("provider", "unknown")
    if provider in SPLIT_BY_SERVICE_PROVIDERS:
        return metadata.get("service", provider)
    return provider


def _skill_display_name(key: str) -> str:
    """Return a human-readable display name for a skill key."""
    if key in SERVICE_DISPLAY_NAMES:
        return SERVICE_DISPLAY_NAMES[key]
    if key in PROVIDER_DISPLAY_NAMES:
        return PROVIDER_DISPLAY_NAMES[key]
    return key.replace("_", " ").title()


def _extract_action(description: str) -> str:
    """Convert a tool description to a short action phrase.

    "Send a message to a Slack channel or user." -> "send messages"
    "List Slack channels in the workspace." -> "list channels"
    """
    if not description:
        return ""
    first_sentence = description.split(".")[0].strip().lower()
    if len(first_sentence) <= 50:
        return first_sentence
    words = first_sentence.split()
    return " ".join(words[:5]) if len(words) >= 3 else first_sentence


def _build_provider_descriptions(
    tools_by_provider: dict[str, list[dict]],
) -> dict[str, str]:
    """Build human-readable capability summary per provider from actual tools."""
    descriptions: dict[str, str] = {}
    for provider, tools in sorted(tools_by_provider.items()):
        display = PROVIDER_DISPLAY_NAMES.get(provider, provider.title())
        actions: list[str] = []
        for tool in tools:
            func = tool.get("function", {})
            action = _extract_action(func.get("description", ""))
            if action and action not in actions:
                actions.append(action)
        # Cap at 4 for conciseness
        if len(actions) > 4:
            actions = actions[:4]
        descriptions[provider] = f"{display} ({', '.join(actions)})" if actions else display
    return descriptions


def _build_group_description(display_name: str, tools: list[dict]) -> str:
    """Build a concise description for a single skill file's frontmatter."""
    actions: list[str] = []
    for tool in tools:
        func = tool.get("function", {})
        action = _extract_action(func.get("description", ""))
        if action and action not in actions:
            actions.append(action)
    if len(actions) > 4:
        actions = actions[:4]
    action_list = ", ".join(actions)
    return (
        f"Use supyagent to interact with {display_name}. "
        f"Available actions: {action_list}. "
        f"Use when the user asks to interact with {display_name}."
    )


def _placeholder_for_type(name: str, param_type: str, schema: dict) -> Any:
    """Generate a sensible placeholder value based on parameter name and type."""
    if "enum" in schema:
        return schema["enum"][0]

    if param_type == "string":
        n = name.lower()
        if "email" in n or n == "to" or n == "cc" or n == "bcc":
            return "user@example.com"
        if "channel" in n:
            return "C0123456789"
        if "phone" in n:
            return "+1234567890"
        if "url" in n:
            return "https://example.com"
        if "id" in n:
            return "abc123"
        if "text" in n or "body" in n or "message" in n or "content" in n:
            return "Hello world"
        if "subject" in n or "title" in n or "name" in n or "summary" in n:
            return "Example"
        if "query" in n or n == "q":
            return "search term"
        if "cursor" in n or "token" in n or "page_token" in n:
            return "..."
        return "..."

    type_defaults: dict[str, Any] = {
        "integer": 10,
        "number": 10.0,
        "boolean": True,
        "array": [],
        "object": {},
    }
    return type_defaults.get(param_type, "...")


def _generate_example_args(parameters: dict) -> str:
    """Generate placeholder JSON args from a parameter schema."""
    props = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    if not props:
        return "{}"

    example: dict[str, Any] = {}
    optional_added = 0
    for name, schema in props.items():
        if name not in required:
            if optional_added >= 1:
                continue
            optional_added += 1
        example[name] = _placeholder_for_type(name, schema.get("type", "string"), schema)

    return json.dumps(example)


def _render_tool_docs(tools: list[dict]) -> list[str]:
    """Render markdown documentation lines for a list of tools."""
    lines: list[str] = []
    for tool in tools:
        func = tool.get("function", {})
        tool_name = func.get("name", "unknown")
        tool_desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        lines.append(f"### {tool_name}")
        lines.append("")
        lines.append(tool_desc)
        lines.append("")

        if props:
            lines.append("| Parameter | Type | Required | Description |")
            lines.append("|-----------|------|----------|-------------|")
            for pname, pschema in props.items():
                ptype = pschema.get("type", "string")
                preq = "yes" if pname in required else "no"
                pdesc = pschema.get("description", "")
                lines.append(f"| `{pname}` | {ptype} | {preq} | {pdesc} |")
            lines.append("")

        example = _generate_example_args(params)
        lines.append("```bash")
        lines.append(f"supyagent service run {tool_name} '{example}'")
        lines.append("```")
        lines.append("")
    return lines


def generate_skill_files(tools: list[dict]) -> dict[str, str]:
    """Generate per-integration Claude Code skill files from a list of service tools.

    Each skill is a directory with a SKILL.md file inside, following the
    Claude Code Agent Skills directory convention.

    Args:
        tools: List of tools in OpenAI function-calling format with metadata.

    Returns:
        Dict mapping skill directory name (e.g. "supy-gmail") to SKILL.md content.
    """
    # Group by skill key
    tools_by_key: dict[str, list[dict]] = {}
    for tool in tools:
        key = _skill_key(tool)
        tools_by_key.setdefault(key, []).append(tool)

    files: dict[str, str] = {}
    for key in sorted(tools_by_key.keys()):
        group_tools = tools_by_key[key]
        display = _skill_display_name(key)
        description = _build_group_description(display, group_tools)

        lines: list[str] = []

        # YAML frontmatter
        lines.append("---")
        lines.append(f"name: supy-{key}")
        lines.append("description: >-")
        lines.append(f"  {description}")
        lines.append("---")
        lines.append("")

        # Header
        lines.append(f"# {display}")
        lines.append("")
        lines.append("Execute tools: `supyagent service run <tool_name> '<json>'`")
        lines.append("")
        lines.append(
            'Output: `{"ok": true, "data": ...}` on success, '
            '`{"ok": false, "error": "..."}` on failure.'
        )
        lines.append("")
        lines.append(
            "**Binary content:** For PDFs, images, and other non-text files, "
            "the response includes a `filePath` field. Use the Read tool on "
            "that path to view the file visually."
        )
        lines.append("")

        # Tool docs
        lines.extend(_render_tool_docs(group_tools))

        dir_name = f"{SKILL_FILE_PREFIX}{key}"
        files[dir_name] = "\n".join(lines)

    return files


def generate_skill_md(tools: list[dict]) -> str:
    """Generate a single combined Claude Code SKILL.md from a list of service tools.

    This is a backwards-compatible wrapper around generate_skill_files() that
    returns all skill content as a single string. Used for --stdout output.

    Args:
        tools: List of tools in OpenAI function-calling format with metadata.

    Returns:
        Combined SKILL.md content as a string.
    """
    files = generate_skill_files(tools)
    return "\n".join(content for content in files.values())
