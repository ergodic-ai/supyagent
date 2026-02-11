"""
Interactive onboarding wizard for supyagent.

Walks users through workspace setup: service connection, model registration,
workspace profile selection, goal definition, and settings configuration.
"""

import getpass
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import questionary
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

from supyagent.core.config import get_config_manager
from supyagent.core.model_registry import get_model_registry
from supyagent.core.service import (
    DEFAULT_SERVICE_URL,
    SERVICE_API_KEY,
    SERVICE_URL,
    ServiceClient,
    poll_for_token,
    request_device_code,
    store_service_credentials,
)
from supyagent.core.workspace import (
    create_goals_file,
    initialize_workspace,
    is_workspace_initialized,
)
from supyagent.default_agents import (
    AGENT_ROLES,
    WORKSPACE_PROFILES,
    install_agent,
    install_workspace_agents,
)
from supyagent.default_tools import install_default_tools, list_default_tools

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PROVIDERS = {
    "OpenAI": {
        "key_name": "OPENAI_API_KEY",
        "models": [
            ("gpt-4.1", "GPT-4.1 (flagship)"),
            ("gpt-4.1-mini", "GPT-4.1 Mini (fast)"),
            ("gpt-4.1-nano", "GPT-4.1 Nano (cheapest)"),
            ("o3", "o3 (reasoning)"),
            ("o4-mini", "o4-mini (reasoning, fast)"),
        ],
    },
    "Anthropic": {
        "key_name": "ANTHROPIC_API_KEY",
        "models": [
            ("anthropic/claude-sonnet-4-5-20250929", "Claude Sonnet 4.5 (recommended)"),
            ("anthropic/claude-opus-4-6", "Claude Opus 4.6 (most capable)"),
            ("anthropic/claude-haiku-4-5-20251001", "Claude Haiku 4.5 (fast)"),
        ],
    },
    "OpenRouter": {
        "key_name": "OPENROUTER_API_KEY",
        "models": [
            ("openrouter/deepseek/deepseek-chat", "DeepSeek V3 (cheap)"),
            ("openrouter/deepseek/deepseek-r1", "DeepSeek R1 (reasoning)"),
            ("openrouter/google/gemini-2.5-flash", "Gemini 2.5 Flash (fast)"),
            ("openrouter/google/gemini-2.5-pro", "Gemini 2.5 Pro (powerful)"),
            ("openrouter/meta-llama/llama-4-maverick", "Llama 4 Maverick"),
            ("openrouter/meta-llama/llama-4-scout", "Llama 4 Scout (fast)"),
            ("openrouter/qwen/qwen3-235b-a22b", "Qwen3 235B (MoE)"),
            ("openrouter/mistralai/mistral-large-2512", "Mistral Large"),
        ],
    },
}

INTEGRATION_PROVIDERS = [
    ("google", "Google", "Gmail, Calendar, Drive"),
    ("slack", "Slack", "Messages, channels"),
    ("github", "GitHub", "Repos, issues, PRs"),
    ("discord", "Discord", "Servers, channels, messages"),
    ("notion", "Notion", "Pages, databases"),
    ("microsoft", "Microsoft 365", "Outlook, Calendar, OneDrive"),
    ("twitter", "Twitter/X", "Tweets, timeline"),
    ("linkedin", "LinkedIn", "Profile, posts"),
    ("hubspot", "HubSpot", "Contacts, deals, companies"),
    ("telegram", "Telegram", "Messages, chats"),
    ("whatsapp", "WhatsApp", "Business messaging"),
]

WIZARD_STEPS = [
    "Connect to Services",
    "Set Up Models",
    "Workspace Profile",
    "Define Goals",
    "Settings",
]

WIZARD_STYLE = questionary.Style([
    ("qmark", "fg:ansicyan bold"),
    ("question", "bold"),
    ("pointer", "fg:ansicyan bold"),
    ("highlighted", "fg:ansicyan bold"),
    ("selected", "fg:ansigreen"),
    ("separator", "fg:ansibrightblack"),
    ("instruction", "fg:ansibrightblack"),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_progress_table(statuses: dict[int, str]) -> Table:
    """Build a compact progress overview for all wizard steps."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(width=3)
    table.add_column(width=24)
    table.add_column(width=3)
    for i, name in enumerate(WIZARD_STEPS, 1):
        status = statuses.get(i, "pending")
        if status == "complete":
            icon = "[green]\u2713[/green]"
            style = ""
        elif status == "active":
            icon = "[cyan]>[/cyan]"
            style = "bold"
        else:
            icon = "[bright_black]\u25cb[/bright_black]"
            style = "bright_black"
        table.add_row(f" {i}.", f"[{style}]{name}[/{style}]" if style else name, icon)
    return table


def _step_header(step: int, title: str, statuses: dict[int, str] | None = None) -> None:
    """Print a styled step header with optional progress overview."""
    console.print()
    if statuses:
        console.print(_build_progress_table(statuses))
        console.print()
    console.print(Rule(f"[bold]Step {step}[/bold]  {title}", style="blue"))
    console.print()


def _detect_state() -> dict[str, Any]:
    """Detect existing setup state."""
    config_mgr = get_config_manager()
    registry = get_model_registry()

    agents_dir = Path("agents")
    powers_dir = Path("powers")

    has_agents_dir = agents_dir.exists()
    has_tools = powers_dir.exists() and any(
        f for f in powers_dir.glob("*.py") if f.name != "__init__.py"
    )
    agent_yamls = list(agents_dir.glob("*.yaml")) if has_agents_dir else []

    service_key = config_mgr.get(SERVICE_API_KEY)

    # Check for known LLM API keys (in config store)
    llm_keys = {}
    for provider_name, provider_info in MODEL_PROVIDERS.items():
        key_name = provider_info["key_name"]
        val = config_mgr.get(key_name)
        if val:
            llm_keys[key_name] = provider_name

    # Check for LLM API keys in environment (not yet imported)
    env_keys = {}
    for provider_name, provider_info in MODEL_PROVIDERS.items():
        key_name = provider_info["key_name"]
        if key_name not in llm_keys and os.environ.get(key_name):
            env_keys[key_name] = provider_name

    return {
        "has_agents_dir": has_agents_dir,
        "has_tools": has_tools,
        "agent_yamls": [f.stem for f in agent_yamls],
        "service_connected": service_key is not None,
        "llm_keys": llm_keys,
        "env_keys": env_keys,
        "is_workspace": is_workspace_initialized(),
        "is_setup": has_agents_dir and has_tools,
        "has_models": len(registry.list_models()) > 0,
        "has_goals": Path("GOALS.md").exists(),
    }


def _show_status_summary(state: dict[str, Any]) -> None:
    """Show a status summary of the current setup."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(width=3)
    table.add_column()

    def _row(done: bool, label: str) -> None:
        if done:
            table.add_row("[green]\u2713[/green]", label)
        else:
            table.add_row(
                "[bright_black]\u25cb[/bright_black]",
                f"[bright_black]{label}[/bright_black]",
            )

    _row(state["is_workspace"], "workspace initialized")
    _row(state["has_tools"], "tools installed")
    _row(state["service_connected"], "service connected")
    _row(state["has_models"], "models registered")
    _row(state["has_goals"], "GOALS.md defined")

    if state["agent_yamls"]:
        agents = ", ".join(state["agent_yamls"][:5])
        if len(state["agent_yamls"]) > 5:
            agents += f" +{len(state['agent_yamls']) - 5} more"
        table.add_row("[green]\u2713[/green]", f"agents: {agents}")
    else:
        _row(False, "no agents created")

    console.print(Panel(table, title="Current Setup", border_style="green"))


# ---------------------------------------------------------------------------
# Step 1: Connect to Services (kept from original, adapted step number)
# ---------------------------------------------------------------------------


def _step_service_connection(statuses: dict[int, str]) -> bool:
    """
    Step 1: Service authentication + integrations.

    Returns True if connected (newly or already), False if skipped.
    """
    statuses[1] = "active"
    _step_header(1, "Connect to Services", statuses)

    config_mgr = get_config_manager()
    existing_key = config_mgr.get(SERVICE_API_KEY)

    if existing_key:
        console.print("  [green]\u2713[/green] Already connected to Supyagent Service")
        if not Confirm.ask("  Reconnect?", default=False):
            statuses[1] = "complete"
            return True

    console.print(
        "  [grey62]Supyagent Service gives your agents access to real services:[/grey62]"
    )
    console.print(
        "  [grey62]  Gmail, Slack, GitHub, Discord, Google Calendar, and more.[/grey62]"
    )
    console.print()

    if not Confirm.ask("  Connect now?", default=True):
        console.print(
            "  [grey62]No problem. Run [cyan]supyagent connect[/cyan] anytime.[/grey62]"
        )
        statuses[1] = "complete"
        return False

    base_url = config_mgr.get(SERVICE_URL) or DEFAULT_SERVICE_URL

    try:
        console.print("  [grey62]Requesting device code...[/grey62]")
        device_data = request_device_code(base_url)
    except Exception as e:
        console.print(f"  [yellow]Could not reach service: {e}[/yellow]")
        console.print(
            "  [grey62]You can connect later with: supyagent connect[/grey62]"
        )
        statuses[1] = "complete"
        return False

    user_code = device_data["user_code"]
    device_code = device_data["device_code"]
    verification_uri = device_data.get("verification_uri") or f"{base_url}/device"
    if "localhost" in verification_uri and "localhost" not in base_url:
        verification_uri = f"{base_url}/device"
    expires_in = device_data.get("expires_in", 900)
    interval = device_data.get("interval", 5)

    console.print()
    console.print(
        Panel(
            f"\n[bold white on blue]  {user_code}  [/bold white on blue]\n",
            title="Your Code",
            border_style="cyan",
            padding=(1, 4),
        ),
        justify="center",
    )
    console.print(
        f"  Visit [link={verification_uri}][cyan]{verification_uri}[/cyan][/link] "
        "and enter the code above.",
    )

    try:
        webbrowser.open(verification_uri)
        console.print("  [grey62]Browser opened automatically.[/grey62]")
    except Exception:
        console.print(f"  [grey62]Open this URL: {verification_uri}[/grey62]")

    console.print("  [grey62]Waiting for authorization...[/grey62]")
    try:
        api_key = poll_for_token(
            base_url=base_url,
            device_code=device_code,
            interval=interval,
            expires_in=expires_in,
        )
    except (TimeoutError, PermissionError, Exception) as e:
        if isinstance(e, TimeoutError):
            console.print("  [yellow]Device code expired.[/yellow]")
        elif isinstance(e, PermissionError):
            console.print("  [yellow]Authorization denied.[/yellow]")
        else:
            console.print(f"  [yellow]Error: {e}[/yellow]")
        console.print(
            "  [grey62]You can connect later with: supyagent connect[/grey62]"
        )
        statuses[1] = "complete"
        return False

    store_service_credentials(api_key, base_url if base_url != DEFAULT_SERVICE_URL else None)
    console.print("  [green]\u2713[/green] Connected to Supyagent Service!")

    # Offer integrations inline
    _offer_integrations(api_key, base_url)

    statuses[1] = "complete"
    return True


def _offer_integrations(api_key: str, base_url: str) -> None:
    """Offer to connect integrations after service auth."""
    try:
        client = ServiceClient(api_key=api_key, base_url=base_url)
        current = client.list_integrations()
        client.close()
    except Exception:
        current = []

    connected_providers = {i["provider"] for i in current}

    console.print()
    console.print(
        "  [grey62]Connect integrations to give your agent access to external services.[/grey62]"
    )
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Service", width=16)
    table.add_column("Features", width=30)
    table.add_column("Status", width=16)
    for idx, (provider_id, name, desc) in enumerate(INTEGRATION_PROVIDERS, 1):
        if provider_id in connected_providers:
            status = "[green]\u2713 connected[/green]"
        else:
            status = "[bright_black]not connected[/bright_black]"
        table.add_row(str(idx), name, desc, status)
    console.print(table)

    console.print()
    console.print(
        "  [grey62]Enter a number to connect a service, or press Enter to continue.[/grey62]"
    )
    console.print()

    while True:
        try:
            choice = Prompt.ask("  Connect", default="done")
        except KeyboardInterrupt:
            console.print()
            break

        if choice.lower() in ("done", "d", "0", ""):
            break

        try:
            idx = int(choice)
            if idx < 1 or idx > len(INTEGRATION_PROVIDERS):
                console.print("  [yellow]Invalid number.[/yellow]")
                continue
        except ValueError:
            console.print("  [yellow]Enter a number or press Enter to continue.[/yellow]")
            continue

        provider_id, name, _ = INTEGRATION_PROVIDERS[idx - 1]

        if provider_id in connected_providers:
            console.print(f"  [grey62]{name} is already connected.[/grey62]")
            continue

        connect_url = f"{base_url}/integrations?connect={provider_id}"
        console.print(f"  [grey62]Opening browser to connect {name}...[/grey62]")
        try:
            webbrowser.open(connect_url)
        except Exception:
            console.print(f"  [grey62]Open this URL: {connect_url}[/grey62]")

        console.print(f"  [grey62]Waiting for {name} to connect (up to 5 min)...[/grey62]")
        deadline = time.time() + 300
        connected = False
        while time.time() < deadline:
            time.sleep(5)
            try:
                client = ServiceClient(api_key=api_key, base_url=base_url)
                updated = client.list_integrations()
                client.close()
                updated_providers = {i["provider"] for i in updated}
                if provider_id in updated_providers:
                    connected = True
                    connected_providers.add(provider_id)
                    break
            except Exception:
                pass

        if connected:
            console.print(f"  [green]\u2713[/green] {name} connected!")
        else:
            console.print(
                f"  [yellow]Timed out waiting for {name}.[/yellow]\n"
                f"  [grey62]You can finish connecting on the dashboard.[/grey62]"
            )


# ---------------------------------------------------------------------------
# Step 2: Set Up Models (NEW — model registry based)
# ---------------------------------------------------------------------------


def _step_model_setup(statuses: dict[int, str], env_keys: dict[str, str]) -> str | None:
    """
    Step 2: Model registration and role assignment.

    Returns the default model string, or None if skipped.
    """
    statuses[2] = "active"
    _step_header(2, "Set Up Models", statuses)

    config_mgr = get_config_manager()
    registry = get_model_registry()

    # Detect and offer to import environment API keys
    if env_keys:
        names = ", ".join(env_keys.values())
        key_list = ", ".join(env_keys.keys())
        console.print(f"  Found API keys in your environment: [cyan]{key_list}[/cyan]")
        if Confirm.ask(f"  Import {names} keys into supyagent?", default=True):
            for key_name, provider_name in env_keys.items():
                config_mgr.set(key_name, os.environ[key_name])
                console.print(f"  [green]\u2713[/green] Imported {key_name} ({provider_name})")
            console.print()

    # Show current model registry
    registered = registry.list_models()
    default_model = registry.get_default()
    roles = registry.list_roles()

    if registered:
        console.print("  [bold]Registered models:[/bold]")
        for m in registered:
            marker = " [cyan](default)[/cyan]" if m == default_model else ""
            console.print(f"    [green]\u2713[/green] {m}{marker}")

        if roles:
            role_parts = [f"{r}=[cyan]{m}[/cyan]" for r, m in roles.items()]
            console.print(f"  Roles: {', '.join(role_parts)}")

        console.print()

        if not Confirm.ask("  Add more models?", default=False):
            selected_model = default_model
            statuses[2] = "complete"
            if selected_model:
                config_mgr.set("DEFAULT_MODEL", selected_model)
            return selected_model
        console.print()
    else:
        console.print("  [bright_black]No models registered yet.[/bright_black]")
        console.print()

    selected_model = default_model

    # Interactive provider → model selection loop
    while True:
        # Build provider choices with key status indicators
        provider_choices = []
        for name, info in MODEL_PROVIDERS.items():
            has_key = config_mgr.get(info["key_name"]) is not None
            indicator = "\u2713" if has_key else " "
            provider_choices.append(
                Choice(f"[{indicator}] {name}", value=name)
            )
        provider_choices.append(Choice("    Custom model (LiteLLM string)", value="custom"))
        provider_choices.append(Choice("    Done adding models", value="done"))

        provider = questionary.select(
            "Select a provider:",
            choices=provider_choices,
            style=WIZARD_STYLE,
            instruction="(arrow keys to move, enter to select)",
        ).ask()

        if provider is None or provider == "done":
            break

        if provider == "custom":
            model_id = Prompt.ask("  Model ID (LiteLLM format)")
            if model_id:
                if not registry.check_api_key(model_id):
                    registry.ensure_api_key(model_id)
                registry.add(model_id)
                if not selected_model:
                    selected_model = model_id
                    registry.set_default(model_id)
                console.print(f"  [green]\u2713[/green] Registered [cyan]{model_id}[/cyan]")
            continue

        # Provider selected — ensure API key
        provider_info = MODEL_PROVIDERS[provider]
        key_name = provider_info["key_name"]

        if not config_mgr.get(key_name):
            console.print()
            value = getpass.getpass(f"  Enter {key_name}: ")
            if value:
                config_mgr.set(key_name, value)
                console.print(f"  [green]\u2713[/green] Saved {key_name}")
            else:
                console.print(f"  [yellow]No key provided. Skipping {provider}.[/yellow]")
                continue

        # Multi-select models from this provider
        model_choices = [
            Choice(desc, value=model_id, checked=(i == 0))
            for i, (model_id, desc) in enumerate(provider_info["models"])
        ]

        console.print()
        chosen_models = questionary.checkbox(
            f"Select {provider} models:",
            choices=model_choices,
            style=WIZARD_STYLE,
            instruction="(arrow keys, space to toggle, enter to confirm)",
        ).ask()

        if not chosen_models:
            continue

        for model_id in chosen_models:
            registry.add(model_id)
            console.print(f"  [green]\u2713[/green] Registered [cyan]{model_id}[/cyan]")
            if not selected_model:
                selected_model = model_id
                registry.set_default(model_id)
                console.print(f"  Set [cyan]{model_id}[/cyan] as default")

    # Set default if multiple models registered
    registered = registry.list_models()
    if len(registered) > 1 and selected_model:
        console.print()
        default_choice = questionary.select(
            "Which model should be the default?",
            choices=[Choice(m, value=m) for m in registered],
            default=selected_model,
            style=WIZARD_STYLE,
        ).ask()
        if default_choice:
            selected_model = default_choice
            registry.set_default(default_choice)

    # Offer role assignments if multiple models registered
    if len(registered) > 1:
        console.print()
        console.print(
            "  [grey62]Assign models to roles (agents reference roles, not models):[/grey62]"
        )
        for role_name in ("fast", "smart", "reasoning"):
            role_choices = [Choice(m, value=m) for m in registered]
            role_choices.append(Choice("Skip", value=None))

            choice = questionary.select(
                f"  {role_name} model:",
                choices=role_choices,
                default=None,
                style=WIZARD_STYLE,
            ).ask()

            if choice:
                registry.assign_role(role_name, choice)
                console.print(
                    f"  [green]\u2713[/green] {role_name} = [cyan]{choice}[/cyan]"
                )

    # Sync to old DEFAULT_MODEL config for backwards compat
    if selected_model:
        config_mgr.set("DEFAULT_MODEL", selected_model)

    statuses[2] = "complete"
    return selected_model


# ---------------------------------------------------------------------------
# Step 3: Workspace Profile (NEW — replaces "Create Agent")
# ---------------------------------------------------------------------------


def _step_workspace_profile(
    statuses: dict[int, str], model: str | None
) -> tuple[str, list[str]]:
    """
    Step 3: Choose workspace profile and install agents.

    Returns (profile_name, list_of_installed_agent_names).
    """
    statuses[3] = "active"
    _step_header(3, "Workspace Profile", statuses)

    # Install default tools first
    tools_path = Path("powers")
    if tools_path.exists() and any(
        f for f in tools_path.glob("*.py") if f.name != "__init__.py"
    ):
        console.print("  [bright_black]\u25cb powers/ already has tools[/bright_black]")
    else:
        count = install_default_tools(tools_path)
        console.print(f"  [green]\u2713[/green] Installed {count} default tools to powers/")

    # Show tool capabilities
    tools = list_default_tools()
    capabilities = []
    for tool in tools:
        name = tool["name"]
        cap_map = {
            "files": "read and write files",
            "web": "fetch web pages",
            "shell": "run shell commands",
            "edit": "edit code",
            "browser": "browse the web",
            "search": "search codebases",
            "find": "find files and directories",
            "patch": "apply multi-file patches",
            "git": "git operations",
            "plan": "create and track plans",
        }
        capabilities.append(cap_map.get(name, tool["description"].rstrip(".").lower()))

    if capabilities:
        cap_str = ", ".join(capabilities[:-1]) + ", and " + capabilities[-1]
        console.print(f"\n  Your agents can: [cyan]{cap_str}[/cyan]")

    console.print()

    # Interactive profile selection
    profile = questionary.select(
        "Select workspace profile:",
        choices=[
            Choice(
                "Coding — Assistant + Coder + Planner (Recommended)",
                value="coding",
            ),
            Choice("Automation — Assistant + Writer", value="automation"),
            Choice(
                "Full — All agents (Assistant + Coder + Planner + Writer)",
                value="full",
            ),
            Choice("Custom — Choose which agents to install", value="custom"),
        ],
        style=WIZARD_STYLE,
    ).ask()
    if profile is None:
        profile = "coding"

    agents_dir = Path("agents")
    agents_dir.mkdir(parents=True, exist_ok=True)
    installed_names = []

    if profile == "custom":
        # Interactive multi-select for agent roles
        role_list = list(AGENT_ROLES.items())
        agent_choices = [
            Choice(f"{role} — {meta['description']}", value=role, checked=(role == "assistant"))
            for role, meta in role_list
        ]

        console.print()
        picked_roles = questionary.checkbox(
            "Select agents to install:",
            choices=agent_choices,
            style=WIZARD_STYLE,
            instruction="(arrow keys, space to toggle, enter to confirm)",
        ).ask()

        if picked_roles:
            for role_name in picked_roles:
                path = install_agent(
                    role_name,
                    agents_dir,
                    model=model,
                    standalone=True,
                    force=False,
                )
                if path:
                    installed_names.append(role_name)
                    console.print(f"  [green]\u2713[/green] Created [cyan]{role_name}[/cyan]")
                else:
                    console.print(
                        f"  [bright_black]\u25cb {role_name} already exists[/bright_black]"
                    )
                    installed_names.append(role_name)
        else:
            console.print("  [yellow]No agents selected, installing Coding profile.[/yellow]")
            profile = "coding"

    if profile != "custom":
        paths = install_workspace_agents(profile, agents_dir, model=model)
        profile_roles = WORKSPACE_PROFILES[profile]
        for role in profile_roles:
            installed_names.append(role)
        if paths:
            for p in paths:
                console.print(f"  [green]\u2713[/green] Created [cyan]{p.stem}[/cyan]")
        else:
            console.print("  [bright_black]\u25cb Agents already exist[/bright_black]")

    # Show delegation info
    if profile == "coding":
        console.print()
        console.print("  [grey62]Delegation: assistant \u2192 planner \u2192 coder[/grey62]")
    elif profile == "full":
        console.print()
        console.print(
            "  [grey62]Delegation: assistant \u2192 planner \u2192 coder, assistant \u2192 writer[/grey62]"
        )

    statuses[3] = "complete"
    return profile, installed_names


# ---------------------------------------------------------------------------
# Step 4: Define Goals (NEW)
# ---------------------------------------------------------------------------


def _step_define_goals(statuses: dict[int, str]) -> str:
    """
    Step 4: Create GOALS.md with user-defined goals.

    Returns the goals text (may be empty).
    """
    statuses[4] = "active"
    _step_header(4, "Define Goals", statuses)

    console.print(
        "  [grey62]What do you want to achieve with this workspace?[/grey62]"
    )
    console.print(
        "  [grey62]This becomes your GOALS.md -- your agent reads it on every conversation.[/grey62]"
    )
    console.print()

    goals_text = ""
    try:
        goals_text = Prompt.ask(
            "  Enter your goals (or press Enter to skip)",
            default="",
        )
    except KeyboardInterrupt:
        console.print()

    # Create GOALS.md
    if goals_text.strip():
        # Format as bullet points if not already
        lines = goals_text.strip().split("\n")
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("- "):
                line = f"- {line}"
            formatted_lines.append(line)
        goals_text = "\n".join(formatted_lines)

    create_goals_file(goals_text)
    if goals_text.strip():
        console.print("  [green]\u2713[/green] Created GOALS.md with your goals")
    else:
        console.print("  [green]\u2713[/green] Created GOALS.md (edit it anytime)")

    statuses[4] = "complete"
    return goals_text


# ---------------------------------------------------------------------------
# Step 5: Settings (NEW)
# ---------------------------------------------------------------------------


def _step_settings(statuses: dict[int, str]) -> tuple[str, bool, str]:
    """
    Step 5: Configure execution mode and heartbeat.

    Returns (execution_mode, heartbeat_enabled, heartbeat_interval).
    """
    statuses[5] = "active"
    _step_header(5, "Settings", statuses)

    # Execution mode — interactive select
    execution_mode = questionary.select(
        "Execution mode:",
        choices=[
            Choice("YOLO — Agent runs commands freely (Recommended)", value="yolo"),
            Choice("Isolated — Agent runs in Docker container", value="isolated"),
        ],
        style=WIZARD_STYLE,
    ).ask()
    if execution_mode is None:
        execution_mode = "yolo"
    console.print(f"  [green]\u2713[/green] Execution mode: [cyan]{execution_mode}[/cyan]")

    # Heartbeat
    console.print()
    heartbeat_enabled = False
    heartbeat_interval = "5m"
    try:
        heartbeat_enabled = Confirm.ask(
            "  Enable heartbeat? (sleep \u2192 wake \u2192 execute \u2192 sleep)", default=False
        )
    except KeyboardInterrupt:
        console.print()

    if heartbeat_enabled:
        try:
            heartbeat_interval = Prompt.ask("  Heartbeat interval", default="5m")
        except KeyboardInterrupt:
            console.print()

        console.print(
            f"  [green]\u2713[/green] Heartbeat: every [cyan]{heartbeat_interval}[/cyan]"
        )
    else:
        console.print("  [bright_black]\u25cb Heartbeat disabled[/bright_black]")

    statuses[5] = "complete"
    return execution_mode, heartbeat_enabled, heartbeat_interval


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _step_summary(
    profile: str,
    installed_agents: list[str],
    model: str | None,
    service_connected: bool,
    execution_mode: str,
) -> bool:
    """
    Final summary and next steps.

    Returns True if user chose to start chatting.
    """
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(width=3)
    table.add_column()

    table.add_row("[green]\u2713[/green]", f"Workspace profile: [cyan]{profile}[/cyan]")

    if installed_agents:
        agents_str = ", ".join(f"[cyan]{a}[/cyan]" for a in installed_agents)
        table.add_row("[green]\u2713[/green]", f"Agents: {agents_str}")

    if model:
        table.add_row("[green]\u2713[/green]", f"Default model: [cyan]{model}[/cyan]")

    if service_connected:
        table.add_row("[green]\u2713[/green]", "Service connected")
    else:
        table.add_row(
            "[bright_black]\u25cb[/bright_black]",
            "[bright_black]Service not connected[/bright_black]",
        )

    table.add_row("[green]\u2713[/green]", f"Mode: [cyan]{execution_mode}[/cyan]")
    table.add_row("[green]\u2713[/green]", "GOALS.md created")

    console.print(Panel(table, title="Workspace Ready", border_style="green"))

    # Offer next action
    main_agent = "assistant" if "assistant" in installed_agents else (
        installed_agents[0] if installed_agents else None
    )

    next_choices = []
    if main_agent:
        next_choices.append(
            Choice(f"Start chatting with {main_agent}", value="chat")
        )
    next_choices.append(Choice("Exit (show next steps)", value="exit"))

    console.print()
    action = questionary.select(
        "What's next?",
        choices=next_choices,
        style=WIZARD_STYLE,
    ).ask()

    if action == "chat":
        return True

    # Show next steps
    console.print()
    console.print("[bold]Next steps:[/bold]")
    if main_agent:
        console.print(f"  Start chatting:     [cyan]supyagent chat {main_agent}[/cyan]")
        console.print("  Or just:            [cyan]supyagent[/cyan]")
    console.print("  Edit your goals:    [cyan]edit GOALS.md[/cyan]")
    console.print("  Manage models:      [cyan]supyagent models[/cyan]")
    if not service_connected:
        console.print("  Connect service:    [cyan]supyagent connect[/cyan]")
    console.print("  Check setup:        [cyan]supyagent doctor[/cyan]")
    console.print()
    return False


def _launch_chat(agent_name: str) -> None:
    """Launch supyagent chat as a replacement process."""
    console.print()
    console.print(f"  [bold]Starting chat with {agent_name}...[/bold]")
    console.print()

    try:
        os.execvp("supyagent", ["supyagent", "chat", agent_name])
    except OSError:
        subprocess.run([sys.executable, "-m", "supyagent.cli.main", "chat", agent_name])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_hello_wizard(quick: bool = False) -> None:
    """Run the interactive onboarding wizard."""

    if quick:
        _run_quick_wizard()
        return

    console.print()
    console.print(
        Panel(
            "[bold]Welcome to Supyagent![/bold]\n\n"
            "Let's set up your workspace. This takes about a minute.",
            border_style="cyan",
        )
    )

    # Detect existing state
    state = _detect_state()
    statuses: dict[int, str] = {}

    if state["is_workspace"]:
        _show_status_summary(state)
        console.print()

        action = questionary.select(
            "What would you like to do?",
            choices=[
                Choice("Manage models", value="models"),
                Choice("Connect service", value="service"),
                Choice("Change workspace profile", value="profile"),
                Choice("Edit goals", value="goals"),
                Choice("Start over (full wizard)", value="restart"),
                Choice("Exit", value="exit"),
            ],
            style=WIZARD_STYLE,
        ).ask()

        if action == "models":
            _step_model_setup(statuses, state.get("env_keys", {}))
            return
        elif action == "service":
            _step_service_connection(statuses)
            return
        elif action == "profile":
            registry = get_model_registry()
            default_model = registry.get_default()
            _step_workspace_profile(statuses, default_model)
            return
        elif action == "goals":
            _step_define_goals(statuses)
            return
        elif action == "restart":
            pass  # Fall through to full wizard
        else:
            return

    # Full wizard flow
    # Step 1: Connect to Services
    service_connected = state["service_connected"]
    try:
        service_connected = _step_service_connection(statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped service connection.[/grey62]")
        statuses[1] = "complete"

    # Step 2: Set Up Models
    model = None
    try:
        model = _step_model_setup(statuses, state.get("env_keys", {}))
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped model setup.[/grey62]")
        statuses[2] = "complete"

    # Step 3: Workspace Profile
    profile = "coding"
    installed_agents: list[str] = []
    try:
        profile, installed_agents = _step_workspace_profile(statuses, model)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped workspace profile.[/grey62]")
        statuses[3] = "complete"

    # Step 4: Define Goals
    try:
        _step_define_goals(statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped goals.[/grey62]")
        create_goals_file("")
        statuses[4] = "complete"

    # Step 5: Settings
    execution_mode = "yolo"
    heartbeat_enabled = False
    heartbeat_interval = "5m"
    try:
        execution_mode, heartbeat_enabled, heartbeat_interval = _step_settings(statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Using defaults.[/grey62]")
        statuses[5] = "complete"

    # Create workspace config (skip GOALS.md — step 4 already created it)
    from supyagent.core.workspace import (
        ExecutionConfig,
        HeartbeatConfig,
        WorkspaceConfig,
        save_workspace,
    )

    save_workspace(
        WorkspaceConfig(
            name=Path.cwd().name,
            profile=profile,
            execution=ExecutionConfig(mode=execution_mode),
            heartbeat=HeartbeatConfig(
                enabled=heartbeat_enabled,
                interval=heartbeat_interval,
            ),
        )
    )

    # Summary + offer to launch chat
    main_agent = "assistant" if "assistant" in installed_agents else (
        installed_agents[0] if installed_agents else None
    )
    should_chat = _step_summary(
        profile, installed_agents, model, service_connected, execution_mode
    )
    if should_chat and main_agent:
        _launch_chat(main_agent)


def _run_quick_wizard() -> None:
    """Non-interactive setup: init workspace with coding profile and sensible defaults."""
    config_mgr = get_config_manager()
    registry = get_model_registry()

    # Auto-import environment API keys
    imported = []
    for provider_name, provider_info in MODEL_PROVIDERS.items():
        key_name = provider_info["key_name"]
        if not config_mgr.get(key_name) and os.environ.get(key_name):
            config_mgr.set(key_name, os.environ[key_name])
            imported.append(provider_name)
    if imported:
        console.print(
            f"  [green]\u2713[/green] Imported API keys: {', '.join(imported)}"
        )

    # Determine model
    model = registry.get_default() or config_mgr.get("DEFAULT_MODEL")
    if not model:
        for provider_name, provider_info in MODEL_PROVIDERS.items():
            if config_mgr.get(provider_info["key_name"]):
                model = provider_info["models"][0][0]
                break
    if model:
        registry.set_default(model)
        config_mgr.set("DEFAULT_MODEL", model)
        console.print(f"  [green]\u2713[/green] Model: [cyan]{model}[/cyan]")
    else:
        console.print(
            "  [yellow]![/yellow] No API keys found. "
            "Run [cyan]supyagent config set[/cyan] to add one."
        )

    # Install default tools
    tools_path = Path("powers")
    if not (tools_path.exists() and any(
        f for f in tools_path.glob("*.py") if f.name != "__init__.py"
    )):
        install_default_tools(tools_path)
    console.print("  [green]\u2713[/green] Tools installed to powers/")

    # Install coding profile agents
    agents_dir = Path("agents")
    agents_dir.mkdir(parents=True, exist_ok=True)
    paths = install_workspace_agents("coding", agents_dir, model=model)
    if paths:
        agent_names = [p.stem for p in paths]
        console.print(
            f"  [green]\u2713[/green] Created agents: {', '.join(agent_names)}"
        )
    else:
        console.print("  [bright_black]\u25cb Agents already exist[/bright_black]")

    # Create GOALS.md and workspace config
    create_goals_file("")
    initialize_workspace(profile="coding", execution_mode="yolo")
    console.print("  [green]\u2713[/green] Workspace initialized")

    console.print()
    console.print("  Start chatting: [cyan]supyagent[/cyan]")
    console.print()
