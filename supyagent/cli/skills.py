"""
Skills generation for Claude Code Agent Skills format.

Generates .claude/skills/supy.md from connected service integrations.
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


def generate_skill_md(tools: list[dict]) -> str:
    """Generate a Claude Code SKILL.md from a list of service tools.

    Args:
        tools: List of tools in OpenAI function-calling format with metadata.

    Returns:
        Complete SKILL.md content as a string.
    """
    # Group by provider
    tools_by_provider: dict[str, list[dict]] = {}
    for tool in tools:
        provider = tool.get("metadata", {}).get("provider", "unknown")
        tools_by_provider.setdefault(provider, []).append(tool)

    # Build dynamic description
    provider_descs = _build_provider_descriptions(tools_by_provider)
    desc_parts = "; ".join(provider_descs.values())
    description = (
        f"Use supyagent to interact with cloud services. "
        f"Connected: {desc_parts}. "
        f"Use when the user asks to send messages, emails, or interact with connected services."
    )

    lines: list[str] = []

    # YAML frontmatter
    lines.append("---")
    lines.append("name: supy")
    lines.append("description: >-")
    lines.append(f"  {description}")
    lines.append("---")
    lines.append("")

    # Header
    lines.append("# Supyagent Cloud Integrations")
    lines.append("")
    lines.append("Execute tools: `supyagent service run <tool_name> '<json>'`")
    lines.append("")
    lines.append(
        'Output: `{"ok": true, "data": ...}` on success, '
        '`{"ok": false, "error": "..."}` on failure.'
    )
    lines.append("")

    # Provider sections
    for provider in sorted(tools_by_provider.keys()):
        provider_tools = tools_by_provider[provider]
        display = PROVIDER_DISPLAY_NAMES.get(provider, provider.title())

        lines.append("---")
        lines.append("")
        lines.append(f"## {display}")
        lines.append("")

        for tool in provider_tools:
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

    return "\n".join(lines)
