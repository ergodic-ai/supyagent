"""
Interactive onboarding wizard for supyagent.

Walks users through project setup, service connection, AI model
configuration, and agent creation in one polished flow.
"""

import getpass
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

from supyagent.core.config import get_config_manager
from supyagent.core.service import (
    DEFAULT_SERVICE_URL,
    SERVICE_API_KEY,
    SERVICE_URL,
    ServiceClient,
    poll_for_token,
    request_device_code,
    store_service_credentials,
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
            ("gpt-4o", "GPT-4o (flagship)"),
            ("gpt-4o-mini", "GPT-4o Mini (fast, cheap)"),
            ("o3-mini", "o3-mini (reasoning)"),
        ],
    },
    "Anthropic": {
        "key_name": "ANTHROPIC_API_KEY",
        "models": [
            ("anthropic/claude-sonnet-4-5-20250929", "Claude Sonnet 4.5 (recommended)"),
            ("anthropic/claude-3-5-haiku-20241022", "Claude 3.5 Haiku (fast)"),
        ],
    },
    "Google": {
        "key_name": "GOOGLE_API_KEY",
        "models": [
            ("gemini/gemini-2.5-flash", "Gemini 2.5 Flash (fast)"),
            ("gemini/gemini-2.5-pro", "Gemini 2.5 Pro (powerful)"),
        ],
    },
    "DeepSeek": {
        "key_name": "DEEPSEEK_API_KEY",
        "models": [
            ("deepseek/deepseek-chat", "DeepSeek V3"),
            ("deepseek/deepseek-reasoner", "DeepSeek R1 (reasoning)"),
        ],
    },
    "OpenRouter": {
        "key_name": "OPENROUTER_API_KEY",
        "models": [
            ("openrouter/meta-llama/llama-4-maverick", "Llama 4 Maverick"),
            ("openrouter/google/gemini-2.5-flash", "Gemini 2.5 Flash (via OR)"),
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

# Steps are now 4 (merged service+integrations into one step)
WIZARD_STEPS = [
    "Project Setup",
    "Connect to Services",
    "Choose AI Model",
    "Create Your Agent",
]


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
            icon = "[green]✓[/green]"
            style = ""
        elif status == "active":
            icon = "[cyan]>[/cyan]"
            style = "bold"
        else:
            icon = "[bright_black]○[/bright_black]"
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
        "is_setup": has_agents_dir and has_tools,
    }


def _show_status_summary(state: dict[str, Any]) -> None:
    """Show a status summary of the current setup."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(width=3)
    table.add_column()

    def _row(done: bool, label: str) -> None:
        if done:
            table.add_row("[green]✓[/green]", label)
        else:
            table.add_row(
                "[bright_black]○[/bright_black]",
                f"[bright_black]{label}[/bright_black]",
            )

    _row(state["has_agents_dir"], "agents/ directory")
    _row(state["has_tools"], "tools installed")
    _row(state["service_connected"], "service connected")

    if state["llm_keys"]:
        providers = ", ".join(state["llm_keys"].values())
        table.add_row("[green]✓[/green]", f"API keys: {providers}")
    else:
        _row(False, "no LLM API keys configured")

    if state["agent_yamls"]:
        agents = ", ".join(state["agent_yamls"][:5])
        if len(state["agent_yamls"]) > 5:
            agents += f" +{len(state['agent_yamls']) - 5} more"
        table.add_row("[green]✓[/green]", f"agents: {agents}")
    else:
        _row(False, "no agents created")

    console.print(Panel(table, title="Current Setup", border_style="green"))


def _derive_agent_name(description: str) -> str:
    """Derive a sensible agent name from the project folder or description."""
    # Prefer the current directory name as a base
    folder = Path.cwd().name.lower().replace(" ", "-")
    # If the folder name is something generic, fall back to "assistant"
    generic_folders = {
        "desktop", "downloads", "documents", "home", "tmp", "temp",
        "projects", "code", "src", "dev", "work", "test",
    }
    if folder and folder not in generic_folders and folder.isidentifier():
        return folder

    return "assistant"


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------


def _step_project_init(statuses: dict[int, str]) -> None:
    """Step 1: Initialize project directories and default tools."""
    statuses[1] = "active"
    _step_header(1, "Project Setup", statuses)

    # Create agents directory
    agents_dir = Path("agents")
    if not agents_dir.exists():
        agents_dir.mkdir(parents=True)
        console.print("  [green]✓[/green] Created agents/")
    else:
        console.print("  [bright_black]○ agents/ already exists[/bright_black]")

    # Install default tools
    tools_path = Path("powers")
    if tools_path.exists() and any(
        f for f in tools_path.glob("*.py") if f.name != "__init__.py"
    ):
        console.print("  [bright_black]○ powers/ already has tools[/bright_black]")
    else:
        count = install_default_tools(tools_path)
        console.print(f"  [green]✓[/green] Installed {count} default tools to powers/")

    # Show capability summary instead of raw tool table
    tools = list_default_tools()
    capabilities = []
    for tool in tools:
        name = tool["name"]
        if name == "files":
            capabilities.append("read and write files")
        elif name == "web":
            capabilities.append("fetch web pages")
        elif name == "shell":
            capabilities.append("run shell commands")
        elif name == "edit":
            capabilities.append("edit code")
        elif name == "browser":
            capabilities.append("browse the web")
        elif name == "search":
            capabilities.append("search codebases")
        elif name == "find":
            capabilities.append("find files and directories")
        else:
            capabilities.append(tool["description"].rstrip(".").lower())

    if capabilities:
        cap_str = ", ".join(capabilities[:-1]) + ", and " + capabilities[-1]
        console.print(f"\n  Your agent can: [cyan]{cap_str}[/cyan]")

    statuses[1] = "complete"


def _step_service_and_integrations(statuses: dict[int, str]) -> bool:
    """
    Step 2: Service authentication + integrations (merged).

    Returns True if connected (newly or already), False if skipped.
    """
    statuses[2] = "active"
    _step_header(2, "Connect to Services", statuses)

    config_mgr = get_config_manager()
    existing_key = config_mgr.get(SERVICE_API_KEY)

    if existing_key:
        console.print("  [green]✓[/green] Already connected to Supyagent Service")
        if not Confirm.ask("  Reconnect?", default=False):
            statuses[2] = "complete"
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
        statuses[2] = "complete"
        return False

    base_url = config_mgr.get(SERVICE_URL) or DEFAULT_SERVICE_URL

    # Request device code
    try:
        console.print("  [grey62]Requesting device code...[/grey62]")
        device_data = request_device_code(base_url)
    except Exception as e:
        console.print(f"  [yellow]Could not reach service: {e}[/yellow]")
        console.print(
            "  [grey62]You can connect later with: supyagent connect[/grey62]"
        )
        statuses[2] = "complete"
        return False

    user_code = device_data["user_code"]
    device_code = device_data["device_code"]
    verification_uri = device_data.get("verification_uri") or f"{base_url}/device"
    if "localhost" in verification_uri and "localhost" not in base_url:
        verification_uri = f"{base_url}/device"
    expires_in = device_data.get("expires_in", 900)
    interval = device_data.get("interval", 5)

    # Show code and open browser
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

    # Poll for approval
    console.print("  [grey62]Waiting for authorization...[/grey62]")
    try:
        api_key = poll_for_token(
            base_url=base_url,
            device_code=device_code,
            interval=interval,
            expires_in=expires_in,
        )
    except TimeoutError:
        console.print("  [yellow]Device code expired.[/yellow]")
        console.print(
            "  [grey62]You can connect later with: supyagent connect[/grey62]"
        )
        statuses[2] = "complete"
        return False
    except PermissionError:
        console.print("  [yellow]Authorization denied.[/yellow]")
        console.print(
            "  [grey62]You can connect later with: supyagent connect[/grey62]"
        )
        statuses[2] = "complete"
        return False
    except Exception as e:
        console.print(f"  [yellow]Error: {e}[/yellow]")
        console.print(
            "  [grey62]You can connect later with: supyagent connect[/grey62]"
        )
        statuses[2] = "complete"
        return False

    store_service_credentials(api_key, base_url if base_url != DEFAULT_SERVICE_URL else None)
    console.print("  [green]✓[/green] Connected to Supyagent Service!")

    # Now offer integrations inline
    _offer_integrations(api_key, base_url)

    statuses[2] = "complete"
    return True


def _offer_integrations(api_key: str, base_url: str) -> None:
    """Offer to connect integrations after service auth (inline, no separate step)."""
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
            status = "[green]✓ connected[/green]"
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
            console.print(f"  [green]✓[/green] {name} connected!")
        else:
            console.print(
                f"  [yellow]Timed out waiting for {name}.[/yellow]\n"
                f"  [grey62]You can finish connecting on the dashboard.[/grey62]"
            )


def _step_model_selection(
    statuses: dict[int, str], env_keys: dict[str, str]
) -> str | None:
    """
    Step 3: AI model selection.

    Returns the selected model string, or None if skipped.
    """
    statuses[3] = "active"
    _step_header(3, "Choose an AI Model", statuses)

    config_mgr = get_config_manager()

    # Detect and offer to import environment variables
    if env_keys:
        names = ", ".join(env_keys.values())
        key_list = ", ".join(env_keys.keys())
        console.print(f"  Found API keys in your environment: [cyan]{key_list}[/cyan]")
        if Confirm.ask(f"  Import {names} keys into supyagent?", default=True):
            for key_name, provider_name in env_keys.items():
                config_mgr.set(key_name, os.environ[key_name])
                console.print(f"  [green]✓[/green] Imported {key_name} ({provider_name})")
            console.print()

    provider_names = list(MODEL_PROVIDERS.keys())

    # Show providers table
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Provider", width=14)
    table.add_column("Status", width=20)
    for idx, name in enumerate(provider_names, 1):
        info = MODEL_PROVIDERS[name]
        key_name = info["key_name"]
        has_key = config_mgr.get(key_name) is not None
        status = (
            "[green]✓ key configured[/green]"
            if has_key
            else "[bright_black]no key[/bright_black]"
        )
        table.add_row(str(idx), name, status)
    table.add_row("0", "Custom model", "[bright_black]LiteLLM string[/bright_black]")
    console.print(table)

    console.print()

    # Find a smart default: first provider with a key configured, else Anthropic
    default_idx = 2  # Anthropic
    for i, name in enumerate(provider_names, 1):
        if config_mgr.get(MODEL_PROVIDERS[name]["key_name"]) is not None:
            default_idx = i
            break

    # Select provider
    try:
        choice = Prompt.ask(
            "  Select provider",
            default=str(default_idx),
        )
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped model selection.[/grey62]")
        statuses[3] = "complete"
        return None

    if choice == "0":
        # Custom model
        model_id = Prompt.ask("  Model ID (LiteLLM format)")
        if not model_id:
            statuses[3] = "complete"
            return None
        key_name = Prompt.ask("  API key env var name (e.g. OPENAI_API_KEY)", default="")
        if key_name:
            existing = config_mgr.get(key_name)
            if existing:
                console.print(f"  [grey62]{key_name} already configured.[/grey62]")
            else:
                value = getpass.getpass(f"  Enter {key_name}: ")
                if value:
                    config_mgr.set(key_name, value)
                    console.print(f"  [green]✓[/green] Saved {key_name}")
        # Save as default model
        config_mgr.set("DEFAULT_MODEL", model_id)
        statuses[3] = "complete"
        return model_id

    # Accept both numbers and provider names (case-insensitive)
    idx = None
    try:
        idx = int(choice)
        if idx < 1 or idx > len(provider_names):
            idx = None
    except ValueError:
        choice_lower = choice.lower()
        for i, name in enumerate(provider_names, 1):
            if name.lower() == choice_lower or name.lower().startswith(choice_lower):
                idx = i
                break

    if idx is None:
        console.print(
            f"  [yellow]Invalid selection, using default "
            f"({provider_names[default_idx - 1]}).[/yellow]"
        )
        idx = default_idx

    provider_name = provider_names[idx - 1]
    provider_info = MODEL_PROVIDERS[provider_name]
    key_name = provider_info["key_name"]
    models = provider_info["models"]

    # Select model within provider (with "Other" option)
    console.print()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Model")
    for midx, (model_id, desc) in enumerate(models, 1):
        table.add_row(str(midx), desc)
    table.add_row("0", "[bright_black]Other (enter model ID)[/bright_black]")
    console.print(table)

    console.print()
    try:
        model_choice = Prompt.ask("  Select model", default="1")
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped.[/grey62]")
        statuses[3] = "complete"
        return None

    if model_choice == "0":
        # Custom model within this provider
        prefix = _provider_prefix(provider_name)
        model_id = Prompt.ask("  Model ID", default=prefix)
        if not model_id:
            statuses[3] = "complete"
            return None
        selected_model = model_id
    else:
        try:
            midx = int(model_choice)
            if midx < 1 or midx > len(models):
                midx = 1
        except ValueError:
            midx = 1
        selected_model = models[midx - 1][0]

    console.print(f"  Selected: [cyan]{selected_model}[/cyan]")

    # Ensure API key is set
    existing = config_mgr.get(key_name)
    if existing:
        console.print(f"  [grey62]{key_name} already configured.[/grey62]")
    else:
        console.print()
        value = getpass.getpass(f"  Enter {key_name}: ")
        if value:
            config_mgr.set(key_name, value)
            console.print(f"  [green]✓[/green] Saved {key_name}")
        else:
            console.print(
                f"  [yellow]No key provided.[/yellow] "
                f"Set it later: [cyan]supyagent config set {key_name}[/cyan]"
            )

    # Save as DEFAULT_MODEL globally
    config_mgr.set("DEFAULT_MODEL", selected_model)

    statuses[3] = "complete"
    return selected_model


def _provider_prefix(provider_name: str) -> str:
    """Return the LiteLLM model prefix for a provider."""
    prefixes = {
        "OpenAI": "",
        "Anthropic": "anthropic/",
        "Google": "gemini/",
        "DeepSeek": "deepseek/",
        "OpenRouter": "openrouter/",
    }
    return prefixes.get(provider_name, "")


def _step_create_agent(
    model: str | None, service_connected: bool, statuses: dict[int, str]
) -> str | None:
    """
    Step 4: Create an agent. No confirmation — just ask for name and go.

    Returns the agent name if created, None if skipped.
    """
    statuses[4] = "active"
    _step_header(4, "Create Your Agent", statuses)

    # Default name from project folder
    default_name = _derive_agent_name("")

    name = Prompt.ask("  Agent name", default=default_name)
    name = name.strip().replace(" ", "-").lower()

    if not name:
        name = "assistant"

    # Check if exists
    agent_path = Path("agents") / f"{name}.yaml"
    if agent_path.exists():
        if not Confirm.ask(f"  Agent '{name}' already exists. Overwrite?", default=False):
            statuses[4] = "complete"
            return None

    # Build template
    effective_model = model or "anthropic/claude-sonnet-4-5-20250929"

    template = f"""name: {name}
description: An interactive AI assistant
version: "1.0"
type: interactive

model:
  provider: {effective_model}
  temperature: 0.7
  max_tokens: 4096

system_prompt: |
  You are {name}, a helpful AI assistant.

  You have access to various tools via supypowers. Use them when needed
  to help accomplish tasks. Be concise, helpful, and accurate.

tools:
  allow:
    - "*"

will_create_tools: true

limits:
  max_tool_calls_per_turn: 10
"""

    Path("agents").mkdir(parents=True, exist_ok=True)
    agent_path.write_text(template)
    console.print(f"  [green]✓[/green] Created [cyan]agents/{name}.yaml[/cyan]")

    statuses[4] = "complete"
    return name


def _step_summary(
    state: dict[str, Any],
    service_connected: bool,
    model: str | None,
    agent_name: str | None,
) -> bool:
    """
    Final summary and next steps.

    Returns True if user chose to start chatting (caller should launch chat).
    """
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(width=3)
    table.add_column()

    table.add_row("[green]✓[/green]", "Project initialized (agents/ + powers/)")

    if service_connected:
        table.add_row("[green]✓[/green]", "Service connected")
    else:
        table.add_row(
            "[bright_black]○[/bright_black]",
            "[bright_black]Service not connected[/bright_black]",
        )

    if model:
        table.add_row("[green]✓[/green]", f"Model: [cyan]{model}[/cyan]")

    # Check all configured keys
    config_mgr = get_config_manager()
    configured = []
    for pname, pinfo in MODEL_PROVIDERS.items():
        if config_mgr.get(pinfo["key_name"]):
            configured.append(pname)
    if configured:
        table.add_row("[green]✓[/green]", f"API keys: {', '.join(configured)}")

    if agent_name:
        table.add_row("[green]✓[/green]", f"Agent: [cyan]{agent_name}[/cyan]")

    console.print(Panel(table, title="Setup Complete", border_style="green"))

    # Offer to start chatting immediately
    if agent_name:
        console.print()
        try:
            if Confirm.ask(
                f"  Start chatting with [cyan]{agent_name}[/cyan]?", default=True
            ):
                return True
        except KeyboardInterrupt:
            console.print()

    # Show next steps for anything they didn't do
    console.print()
    console.print("[bold]Next steps:[/bold]")
    if agent_name:
        console.print(f"  Start chatting:     [cyan]supyagent chat {agent_name}[/cyan]")
    else:
        console.print("  Create an agent:    [cyan]supyagent new myagent[/cyan]")
    if not service_connected:
        console.print("  Connect service:    [cyan]supyagent connect[/cyan]")
    console.print("  Configure API keys: [cyan]supyagent config set[/cyan]")
    console.print("  Check setup:        [cyan]supyagent doctor[/cyan]")
    console.print()
    return False


def _launch_chat(agent_name: str) -> None:
    """Launch supyagent chat as a replacement process."""
    console.print()
    console.print(f"  [bold]Starting chat with {agent_name}...[/bold]")
    console.print()

    # Replace this process with `supyagent chat <agent_name>`
    # This gives the user a clean chat experience
    try:
        os.execvp("supyagent", ["supyagent", "chat", agent_name])
    except OSError:
        # Fallback: run as subprocess
        subprocess.run([sys.executable, "-m", "supyagent.cli.main", "chat", agent_name])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_hello_wizard(quick: bool = False) -> None:
    """Run the interactive onboarding wizard."""

    # --quick mode: non-interactive setup with sensible defaults
    if quick:
        _run_quick_wizard()
        return

    console.print()
    console.print(
        Panel(
            "[bold]Welcome to Supyagent![/bold]\n\n"
            "Let's get you set up. This takes about a minute.",
            border_style="cyan",
        )
    )

    # Step 0: Detect existing state
    state = _detect_state()
    statuses: dict[int, str] = {}

    if state["is_setup"]:
        _show_status_summary(state)
        console.print()
        console.print("  [bold]What would you like to do?[/bold]")
        console.print("    1. Change AI model")
        console.print("    2. Connect service")
        console.print("    3. Create another agent")
        console.print("    4. Start over (full wizard)")
        console.print("    0. Exit")
        console.print()
        try:
            choice = Prompt.ask("  Select", default="0")
        except KeyboardInterrupt:
            console.print()
            return

        if choice == "1":
            model = _step_model_selection(statuses, state.get("env_keys", {}))
            if model:
                console.print(f"\n  [green]✓[/green] Default model set to [cyan]{model}[/cyan]")
            return
        elif choice == "2":
            _step_service_and_integrations(statuses)
            return
        elif choice == "3":
            agent_name = _step_create_agent(None, state["service_connected"], statuses)
            if agent_name:
                try:
                    if Confirm.ask(
                        f"\n  Start chatting with [cyan]{agent_name}[/cyan]?",
                        default=True,
                    ):
                        _launch_chat(agent_name)
                except KeyboardInterrupt:
                    console.print()
            return
        elif choice == "4":
            pass  # Fall through to full wizard
        else:
            return

    # Step 1: Project init
    try:
        _step_project_init(statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped project init.[/grey62]")
        statuses[1] = "complete"

    # Step 2: Service auth + integrations (merged)
    service_connected = state["service_connected"]
    try:
        service_connected = _step_service_and_integrations(statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped service connection.[/grey62]")
        statuses[2] = "complete"

    # Step 3: Model selection (with env var detection)
    model = None
    try:
        model = _step_model_selection(statuses, state.get("env_keys", {}))
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped model selection.[/grey62]")
        statuses[3] = "complete"

    # Step 4: Create agent (no confirmation prompt)
    agent_name = None
    try:
        agent_name = _step_create_agent(model, service_connected, statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped agent creation.[/grey62]")
        statuses[4] = "complete"

    # Summary + offer to launch chat
    should_chat = _step_summary(state, service_connected, model, agent_name)
    if should_chat and agent_name:
        _launch_chat(agent_name)


def _run_quick_wizard() -> None:
    """Non-interactive setup: init project, detect keys, create default agent."""
    config_mgr = get_config_manager()

    # Project init
    agents_dir = Path("agents")
    agents_dir.mkdir(parents=True, exist_ok=True)
    tools_path = Path("powers")
    if not (tools_path.exists() and any(
        f for f in tools_path.glob("*.py") if f.name != "__init__.py"
    )):
        install_default_tools(tools_path)

    console.print("  [green]✓[/green] Project initialized (agents/ + powers/)")

    # Auto-import environment API keys
    imported = []
    for provider_name, provider_info in MODEL_PROVIDERS.items():
        key_name = provider_info["key_name"]
        if not config_mgr.get(key_name) and os.environ.get(key_name):
            config_mgr.set(key_name, os.environ[key_name])
            imported.append(provider_name)
    if imported:
        console.print(
            f"  [green]✓[/green] Imported API keys: {', '.join(imported)}"
        )

    # Determine model: use DEFAULT_MODEL if set, else pick from available keys
    model = config_mgr.get("DEFAULT_MODEL")
    if not model:
        for provider_name, provider_info in MODEL_PROVIDERS.items():
            if config_mgr.get(provider_info["key_name"]):
                model = provider_info["models"][0][0]
                config_mgr.set("DEFAULT_MODEL", model)
                break
    if model:
        console.print(f"  [green]✓[/green] Model: [cyan]{model}[/cyan]")
    else:
        console.print(
            "  [yellow]![/yellow] No API keys found. "
            "Run [cyan]supyagent config set[/cyan] to add one."
        )

    # Create default agent
    name = _derive_agent_name("")
    agent_path = Path("agents") / f"{name}.yaml"
    if not agent_path.exists():
        effective_model = model or "anthropic/claude-sonnet-4-5-20250929"
        template = f"""name: {name}
description: An interactive AI assistant
version: "1.0"
type: interactive

model:
  provider: {effective_model}
  temperature: 0.7
  max_tokens: 4096

system_prompt: |
  You are {name}, a helpful AI assistant.

  You have access to various tools via supypowers. Use them when needed
  to help accomplish tasks. Be concise, helpful, and accurate.

tools:
  allow:
    - "*"

will_create_tools: true

limits:
  max_tool_calls_per_turn: 10
"""
        agent_path.write_text(template)
        console.print(f"  [green]✓[/green] Created agent: [cyan]{name}[/cyan]")
    else:
        console.print(f"  [bright_black]○[/bright_black] Agent '{name}' already exists")

    console.print()
    console.print(f"  Start chatting: [cyan]supyagent chat {name}[/cyan]")
    console.print()
