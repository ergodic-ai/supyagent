"""
Skills generation for AI coding assistant skill files.

Generates {tool_folder}/skills/supy-{key}/SKILL.md files from connected service integrations.
Supports multiple AI tools: Claude Code, Cursor, Codex, Copilot, Windsurf.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "google": "Google Workspace",
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
    "platform-search": "Platform Services - Search",
    "platform-multimodal": "Platform Services - Multimodal",
    "platform-compute": "Platform Services - Compute & Data",
    "platform-browser": "Platform Services - Browser",
}

# Providers whose tools should be split by service rather than grouped together.
SPLIT_BY_SERVICE_PROVIDERS: set[str] = {"google", "microsoft"}

# Display names for individual services (used when a provider is split).
SERVICE_DISPLAY_NAMES: dict[str, str] = {
    "gmail": "Gmail",
    "calendar": "Google Calendar",
    "drive": "Google Drive",
    "slides": "Google Slides",
    "sheets": "Google Sheets",
    "docs": "Google Docs",
    "mail": "Outlook Mail",
}

# Custom description overrides for skill keys that need keyword-rich descriptions
# instead of the auto-generated action list.
PROVIDER_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "platform-search": (
        "Use supyagent to search the web, find images, discover videos, read news, "
        "find local places, compare shopping results, and search academic papers. "
        "Powered by Google Search. Use when the user asks to search for information, "
        "look something up, find images or videos, get news, find nearby places or "
        "businesses, compare prices or products, or search for academic research and "
        "scholarly articles."
    ),
    "platform-multimodal": (
        "Use supyagent for OCR text extraction from images and PDFs, speech-to-text "
        "audio transcription, text-to-speech voice generation, AI image generation, "
        "AI video generation, and video understanding/analysis. Use when the user asks "
        "to extract text from a document or image, transcribe audio or video, convert "
        "text to speech, generate an image from a prompt, create a video, or analyze "
        "and understand video content."
    ),
    "platform-browser": (
        "Use supyagent to visit web pages, scrape websites, take page snapshots, "
        "search text on a page, and run an AI browser agent to interact with web pages. "
        "Use when the user asks to visit a web page, scrape a website, browse a URL, "
        "get page content, extract data from a website, or use a browser agent to "
        "interact with web pages."
    ),
    "platform-compute": (
        "Use supyagent to execute Python code in a sandbox, upload and host temporary files, "
        "and create/query SQLite databases. Use when the user asks to run code, compute something, "
        "process data, generate a chart or file, store structured data, query a database, or "
        "persist information across conversations."
    ),
}

# Prefix for generated skill directories — used to identify managed files for cleanup.
SKILL_FILE_PREFIX = "supy-cloud-"

# Registry of AI coding tools and their skills folder conventions.
AI_TOOL_FOLDERS: list[dict[str, str]] = [
    {"name": "Claude Code", "detect": ".claude", "skills": ".claude/skills"},
    {"name": "Cursor", "detect": ".cursor", "skills": ".cursor/skills"},
    {"name": "Codex", "detect": ".agents", "skills": ".agents/skills"},
    {"name": "Copilot", "detect": ".copilot", "skills": ".copilot/skills"},
    {"name": "Windsurf", "detect": ".windsurf", "skills": ".windsurf/skills"},
]


def detect_ai_tool_folders(root: Path) -> list[dict[str, str]]:
    """Return AI tool entries whose detection folder exists under root."""
    return [entry for entry in AI_TOOL_FOLDERS if (root / entry["detect"]).is_dir()]


def prompt_skill_output_dirs(
    root: Path,
    detected: list[dict[str, str]],
) -> list[Path]:
    """Interactive prompt for selecting which AI tool folders to populate with skills.

    Shows detected folders with a default of 'all'. If none detected, prompts for a
    custom path.

    Returns:
        List of absolute skills directory Paths.
    """
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console(stderr=True)

    if not detected:
        console.print(
            "[yellow]No AI tool folders detected[/yellow] "
            "(.claude, .cursor, .agents, .copilot, .windsurf)"
        )
        custom = Prompt.ask("Enter skills output path", default=".claude/skills")
        return [root / custom]

    console.print()
    console.print("[bold]Detected AI tool folders:[/bold]")
    console.print()
    for i, entry in enumerate(detected, 1):
        console.print(f"  [cyan]{i}[/cyan]. {entry['name']}  [dim]({entry['skills']}/)[/dim]")
    console.print()

    choice = Prompt.ask(
        "Select folders (comma-separated numbers, or 'a' for all)",
        default="a",
    )

    if choice.strip().lower() == "a":
        return [root / e["skills"] for e in detected]

    indices = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(detected):
                indices.append(idx)

    if not indices:
        console.print("[yellow]Invalid selection, defaulting to all.[/yellow]")
        return [root / e["skills"] for e in detected]

    return [root / detected[i]["skills"] for i in indices]


def write_skills_to_dir(output_dir: Path, skill_files: dict[str, str]) -> None:
    """Write skill files to a directory, cleaning up stale entries first."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.iterdir():
        if existing.is_dir() and (
            existing.name.startswith(SKILL_FILE_PREFIX)
            or existing.name.startswith("supy-")  # legacy prefix
        ):
            shutil.rmtree(existing)
        elif existing.is_file() and (
            existing.name == "supy.md"
            or existing.name.startswith("supy-")  # legacy flat files
        ):
            existing.unlink()
    for dir_name, content in skill_files.items():
        skill_dir = output_dir / dir_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content)


def resolve_output_dirs(
    output: str | None,
    select_all: bool,
    root: Path = Path("."),
) -> list[Path]:
    """Determine output directories based on CLI flags.

    Args:
        output: Explicit --output path, or None for auto-detection.
        select_all: If True, write to all detected folders without prompting.
        root: Project root directory to scan for AI tool folders.

    Returns:
        List of skills directory Paths to write to.
    """
    if output:
        return [Path(output)]

    detected = detect_ai_tool_folders(root)

    if select_all or not sys.stdin.isatty():
        if detected:
            return [root / e["skills"] for e in detected]
        return [root / ".claude/skills"]

    return prompt_skill_output_dirs(root, detected)


def _skill_key(tool: dict) -> str:
    """Return the grouping key for a tool.

    Google and Microsoft tools are grouped by their service field (e.g. gmail, calendar).
    Platform tools are split into search vs multimodal categories.
    All other providers are grouped by provider name.
    """
    metadata = tool.get("metadata", {})
    provider = metadata.get("provider", "unknown")
    if provider in SPLIT_BY_SERVICE_PROVIDERS:
        return metadata.get("service", provider)
    if provider == "platform":
        service = metadata.get("service", "")
        if service.startswith("platform_search"):
            return "platform-search"
        if service in ("platform_files", "platform_code", "platform_sqlite"):
            return "platform-compute"
        if service.startswith("platform_browser"):
            return "platform-browser"
        return "platform-multimodal"
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


def _build_group_description(display_name: str, tools: list[dict], key: str = "") -> str:
    """Build a concise description for a single skill file's frontmatter."""
    if key in PROVIDER_DESCRIPTION_OVERRIDES:
        return PROVIDER_DESCRIPTION_OVERRIDES[key]
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
        if n == "sql":
            return "SELECT * FROM table_name LIMIT 10"
        if n == "code":
            return "print('Hello world')"
        if n == "database":
            return "my_database"
        if n == "language":
            return "python"
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
        description = _build_group_description(display, group_tools, key=key)

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
