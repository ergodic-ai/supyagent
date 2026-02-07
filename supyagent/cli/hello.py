"""
Interactive onboarding wizard for supyagent.

Walks users through project setup, service connection, integration setup,
AI model configuration, and agent creation in one polished flow.
"""

import getpass
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

WIZARD_STEPS = [
    "Project Setup",
    "Connect to Service",
    "Connect Integrations",
    "Choose AI Model",
    "Create First Agent",
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

    # Check for known LLM API keys
    llm_keys = {}
    for provider_name, provider_info in MODEL_PROVIDERS.items():
        key_name = provider_info["key_name"]
        val = config_mgr.get(key_name)
        if val:
            llm_keys[key_name] = provider_name

    return {
        "has_agents_dir": has_agents_dir,
        "has_tools": has_tools,
        "agent_yamls": [f.stem for f in agent_yamls],
        "service_connected": service_key is not None,
        "llm_keys": llm_keys,
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
            table.add_row("[bright_black]○[/bright_black]", f"[bright_black]{label}[/bright_black]")

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

    # Show available tools
    console.print()
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Tool", style="cyan")
    table.add_column("Description")
    for tool in list_default_tools():
        table.add_row(tool["name"], tool["description"])
    console.print(table)

    statuses[1] = "complete"


def _step_service_auth(statuses: dict[int, str]) -> bool:
    """
    Step 2: Service authentication via device flow.

    Returns True if connected (newly or already), False if skipped.
    """
    statuses[2] = "active"
    _step_header(2, "Connect to Supyagent Service", statuses)

    config_mgr = get_config_manager()
    existing_key = config_mgr.get(SERVICE_API_KEY)

    if existing_key:
        console.print("  [green]✓[/green] Already connected to service")
        if not Confirm.ask("  Reconnect?", default=False):
            statuses[2] = "complete"
            return True

    console.print(
        "  [grey62]Supyagent Service gives your agents access to third-party\n"
        "  integrations like Gmail, Slack, GitHub, and more.[/grey62]"
    )
    console.print()

    if not Confirm.ask("  Connect to Supyagent Service?", default=True):
        console.print(
            "  [grey62]Skipped. You can connect later with: supyagent connect[/grey62]"
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
    # Ensure verification URI uses the correct host (server may return localhost in dev)
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
    console.print("  [green]✓[/green] Connected to service!")
    statuses[2] = "complete"
    return True


def _step_integrations(service_connected: bool, statuses: dict[int, str]) -> list[str]:
    """
    Step 3: Integration setup.

    Returns list of connected provider names.
    """
    statuses[3] = "active"
    _step_header(3, "Connect Integrations", statuses)

    if not service_connected:
        console.print(
            "  [grey62]Skipped -- not connected to service.\n"
            "  Run 'supyagent connect' first, then visit the dashboard.[/grey62]"
        )
        statuses[3] = "complete"
        return []

    # Get current integrations
    config_mgr = get_config_manager()
    api_key = config_mgr.get(SERVICE_API_KEY)
    base_url = config_mgr.get(SERVICE_URL) or DEFAULT_SERVICE_URL

    try:
        client = ServiceClient(api_key=api_key, base_url=base_url)
        current = client.list_integrations()
        client.close()
    except Exception:
        current = []

    connected_providers = {i["provider"] for i in current}

    console.print(
        "  [grey62]Connect third-party services to give your agents\n"
        "  access to Gmail, Slack, GitHub, and more.[/grey62]"
    )
    console.print()

    # Show integration table
    def _print_integration_table() -> None:
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

    _print_integration_table()

    console.print()
    console.print(
        "  [grey62]Enter a number to connect a service, or 'done' to continue.[/grey62]"
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
            console.print("  [yellow]Enter a number or 'done'.[/yellow]")
            continue

        provider_id, name, _ = INTEGRATION_PROVIDERS[idx - 1]

        if provider_id in connected_providers:
            console.print(f"  [grey62]{name} is already connected.[/grey62]")
            continue

        # Open dashboard to connect
        connect_url = f"{base_url}/integrations?connect={provider_id}"
        console.print(f"  [grey62]Opening browser to connect {name}...[/grey62]")
        try:
            webbrowser.open(connect_url)
        except Exception:
            console.print(f"  [grey62]Open this URL: {connect_url}[/grey62]")

        # Poll for completion
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

    statuses[3] = "complete"
    return list(connected_providers)


def _step_model_selection(statuses: dict[int, str]) -> str | None:
    """
    Step 4: AI model selection.

    Returns the selected model string, or None if skipped.
    """
    statuses[4] = "active"
    _step_header(4, "Choose an AI Model", statuses)

    config_mgr = get_config_manager()

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

    # Select provider
    try:
        choice = Prompt.ask(
            "  Select provider",
            default="2",  # Default to Anthropic
        )
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped model selection.[/grey62]")
        statuses[4] = "complete"
        return None

    if choice == "0":
        # Custom model
        model_id = Prompt.ask("  Model ID (LiteLLM format)")
        if not model_id:
            statuses[4] = "complete"
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
        statuses[4] = "complete"
        return model_id

    try:
        idx = int(choice)
        if idx < 1 or idx > len(provider_names):
            console.print("  [yellow]Invalid selection, using default.[/yellow]")
            idx = 2
    except ValueError:
        console.print("  [yellow]Invalid selection, using default.[/yellow]")
        idx = 2

    provider_name = provider_names[idx - 1]
    provider_info = MODEL_PROVIDERS[provider_name]
    key_name = provider_info["key_name"]
    models = provider_info["models"]

    # Select model within provider
    console.print()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Model")
    for midx, (model_id, desc) in enumerate(models, 1):
        table.add_row(str(midx), desc)
    console.print(table)

    console.print()
    try:
        model_choice = Prompt.ask("  Select model", default="1")
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped.[/grey62]")
        statuses[4] = "complete"
        return None

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

    statuses[4] = "complete"
    return selected_model


def _step_create_agent(
    model: str | None, service_connected: bool, statuses: dict[int, str]
) -> str | None:
    """
    Step 5: Create an agent.

    Returns the agent name if created, None if skipped.
    """
    statuses[5] = "active"
    _step_header(5, "Create Your First Agent", statuses)

    if not Confirm.ask("  Create an agent now?", default=True):
        console.print(
            "  [grey62]Skipped. Create one later: supyagent new <name>[/grey62]"
        )
        statuses[5] = "complete"
        return None

    # Get description
    description = Prompt.ask(
        "  What should your agent do?",
        default="A helpful assistant that can use tools",
    )

    # Derive name
    stop_words = {
        "a", "an", "the", "that", "which", "can", "will", "should", "is",
        "are", "my", "your", "this", "to", "and", "or", "for", "with",
        "do", "does", "be", "been", "have", "has",
    }
    words = description.lower().split()
    meaningful = [w for w in words if w not in stop_words and w.isalpha()]
    derived_name = meaningful[0] if meaningful else "assistant"

    name = Prompt.ask("  Agent name", default=derived_name)
    name = name.strip().replace(" ", "-").lower()

    if not name:
        name = "assistant"

    # Check if exists
    agent_path = Path("agents") / f"{name}.yaml"
    if agent_path.exists():
        if not Confirm.ask(f"  Agent '{name}' already exists. Overwrite?", default=False):
            statuses[5] = "complete"
            return None

    # Build template
    effective_model = model or "anthropic/claude-sonnet-4-5-20250929"
    service_enabled = "true" if service_connected else "false"

    template = f"""name: {name}
description: {description}
version: "1.0"
type: interactive

model:
  provider: {effective_model}
  temperature: 0.7
  max_tokens: 4096

system_prompt: |
  You are {name}, a helpful AI assistant.

  Your purpose: {description}

  You have access to various tools via supypowers. Use them when needed
  to help accomplish tasks. Be concise, helpful, and accurate.

service:
  enabled: {service_enabled}

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

    statuses[5] = "complete"
    return name


def _step_summary(
    state: dict[str, Any],
    service_connected: bool,
    model: str | None,
    agent_name: str | None,
) -> None:
    """Final summary and next steps."""
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

    # Next steps
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_hello_wizard() -> None:
    """Run the interactive onboarding wizard."""
    console.print()
    console.print(
        Panel(
            "[bold]Welcome to Supyagent![/bold]\n\n"
            "This wizard will help you set up your project, connect\n"
            "to services, configure an AI model, and create your first agent.",
            border_style="cyan",
        )
    )

    # Step 0: Detect existing state
    state = _detect_state()
    statuses: dict[int, str] = {}

    if state["is_setup"]:
        _show_status_summary(state)
        console.print()
        choice = Prompt.ask(
            "  Already set up. What would you like to do?",
            choices=["continue", "redo"],
            default="continue",
        )
        if choice == "continue":
            console.print(
                "  [grey62]Nothing to do. "
                "Run 'supyagent doctor' to check your setup.[/grey62]"
            )
            return

    # Step 1: Project init
    try:
        _step_project_init(statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped project init.[/grey62]")
        statuses[1] = "complete"

    # Step 2: Service auth
    service_connected = state["service_connected"]
    try:
        service_connected = _step_service_auth(statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped service connection.[/grey62]")
        statuses[2] = "complete"

    # Step 3: Integrations
    try:
        _step_integrations(service_connected, statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped integrations.[/grey62]")
        statuses[3] = "complete"

    # Step 4: Model selection
    model = None
    try:
        model = _step_model_selection(statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped model selection.[/grey62]")
        statuses[4] = "complete"

    # Step 5: Create agent
    agent_name = None
    try:
        agent_name = _step_create_agent(model, service_connected, statuses)
    except KeyboardInterrupt:
        console.print("\n  [grey62]Skipped agent creation.[/grey62]")
        statuses[5] = "complete"

    # Summary
    _step_summary(state, service_connected, model, agent_name)
