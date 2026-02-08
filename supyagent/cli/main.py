"""
CLI entry point for supyagent.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    from importlib.metadata import version as pkg_version

    _version = pkg_version("supyagent")
except Exception:
    _version = "0.2.7"

from supyagent.core.agent import Agent
from supyagent.core.config import ConfigManager, load_config
from supyagent.core.executor import ExecutionRunner
from supyagent.core.registry import AgentRegistry
from supyagent.core.session_manager import SessionManager
from supyagent.default_tools import install_default_tools, list_default_tools
from supyagent.models.agent_config import (
    AgentConfigError,
    AgentNotFoundError,
    load_agent_config,
    validate_agent_config,
)

console = Console()
console_err = Console(stderr=True)

_KNOWN_MODEL_PREFIXES = (
    "anthropic/",
    "openai/",
    "openrouter/",
    "google/",
    "azure/",
    "deepseek/",
    "groq/",
    "together_ai/",
    "replicate/",
    "ollama/",
    "bedrock/",
    "vertex_ai/",
    "mistral/",
    "cohere/",
    "fireworks_ai/",
    "huggingface/",
)


def _warn_model_provider(provider: str) -> None:
    """Print a warning if the model provider string looks invalid."""
    if "/" not in provider:
        console.print(
            f"[yellow]Warning:[/yellow] Model '{provider}' has no provider prefix."
        )
        console.print(
            "  Expected format: [cyan]provider/model-name[/cyan] "
            "(e.g., openrouter/google/gemini-2.5-flash)"
        )
    elif not any(provider.startswith(p) for p in _KNOWN_MODEL_PREFIXES):
        prefix = provider.split("/")[0]
        console.print(
            f"[yellow]Note:[/yellow] Provider prefix '{prefix}/' is not in the common list."
        )
        console.print(
            "  It may still work if LiteLLM supports it or you have a custom endpoint."
        )


@click.group()
@click.version_option(version=_version, prog_name="supyagent")
@click.option("--debug", is_flag=True, hidden=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, debug: bool):
    """
    Supyagent - LLM agents powered by supypowers.

    Create and interact with AI agents that can use tools.

    Quick start:

    \b
        supyagent hello           # Interactive setup wizard
        supyagent new myagent     # Create an agent
        supyagent chat myagent    # Start chatting
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    if debug:
        import logging

        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(levelname)s] %(name)s: %(message)s",
        )
        # Suppress noisy third-party loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.INFO)


@cli.command()
@click.option("--quick", "-q", is_flag=True, help="Non-interactive setup with sensible defaults")
def hello(quick: bool):
    """Interactive setup wizard -- the best way to get started."""
    from supyagent.cli.hello import run_hello_wizard

    run_hello_wizard(quick=quick)


# Register 'setup' as an alias for 'hello'
cli.add_command(hello, name="setup")


def _init_quick(tools_dir: str, force: bool) -> None:
    """Quick init without wizard (original init behavior)."""
    console.print("[bold]Initializing supyagent...[/bold]")
    console.print()

    # Create agents directory
    agents_dir = Path("agents")
    if not agents_dir.exists():
        agents_dir.mkdir(parents=True)
        console.print(f"  [green]✓[/green] Created {agents_dir}/")
    else:
        console.print(f"  [bright_black]○[/bright_black] {agents_dir}/ already exists")

    # Install default tools
    tools_path = Path(tools_dir)

    if force:
        # Remove and reinstall
        import shutil

        if tools_path.exists():
            shutil.rmtree(tools_path)

    if tools_path.exists() and any(tools_path.glob("*.py")):
        console.print(f"  [bright_black]○[/bright_black] {tools_dir}/ already has tools")
    else:
        count = install_default_tools(tools_path)
        console.print(
            f"  [green]✓[/green] Installed {count} default tools to {tools_dir}/"
        )

    # Show available tools
    console.print()
    console.print("[bold]Available tools:[/bold]")
    for tool in list_default_tools():
        console.print(f"  • [cyan]{tool['name']}[/cyan]: {tool['description']}")

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Configure your API key:  [cyan]supyagent config set[/cyan]")
    console.print("  2. Create an agent:         [cyan]supyagent new myagent[/cyan]")
    console.print("  3. Start chatting:          [cyan]supyagent chat myagent[/cyan]")
    console.print()
    console.print(
        "[dim]Tip:[/dim] Want your agent to send emails, post to Slack, or create GitHub issues?"
    )
    console.print(
        "     Run [cyan]supyagent connect[/cyan] to unlock 12+ cloud integrations."
    )


@cli.command()
@click.option(
    "--tools-dir",
    "-t",
    default="powers",
    help="Directory for tools (default: powers/)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing files",
)
@click.option(
    "--quick",
    "-q",
    is_flag=True,
    help="Quick init without wizard",
)
def init(tools_dir: str, force: bool, quick: bool):
    """
    Initialize supyagent. Runs setup wizard by default.

    Use --quick for fast init without the interactive wizard.

    \b
    Examples:
        supyagent init              # Interactive wizard
        supyagent init --quick      # Quick init (directories + tools only)
        supyagent init --force      # Quick init, overwrite existing
        supyagent init -t my_tools  # Quick init with custom tools dir
    """
    if quick or force or tools_dir != "powers":
        _init_quick(tools_dir, force)
    else:
        from supyagent.cli.hello import run_hello_wizard

        run_hello_wizard()


@cli.command()
@click.argument("name")
@click.option(
    "--type",
    "-t",
    "agent_type",
    type=click.Choice(["interactive", "execution"]),
    default="interactive",
    help="Type of agent to create",
)
@click.option(
    "--model",
    "-m",
    "model_provider",
    default=None,
    help="Model provider (e.g., 'openrouter/google/gemini-2.5-flash')",
)
@click.option(
    "--from",
    "from_agent",
    default=None,
    help="Clone from an existing agent (e.g., --from myagent)",
)
def new(name: str, agent_type: str, model_provider: str | None, from_agent: str | None):
    """
    Create a new agent from template.

    NAME is the agent name (will create agents/NAME.yaml)

    \b
    Examples:
        supyagent new myagent
        supyagent new myagent --model openrouter/google/gemini-2.5-flash
        supyagent new myagent --from existing-agent
        supyagent new worker --type execution
    """
    agents_dir = Path("agents")
    agents_dir.mkdir(exist_ok=True)

    agent_path = agents_dir / f"{name}.yaml"

    if agent_path.exists():
        if not click.confirm(f"Agent '{name}' already exists. Overwrite?"):
            return

    # Clone from existing agent
    if from_agent:
        source_path = agents_dir / f"{from_agent}.yaml"
        if not source_path.exists():
            console.print(f"[red]Error:[/red] Source agent '{from_agent}' not found")
            available = [f.stem for f in agents_dir.glob("*.yaml")]
            if available:
                console.print(f"  Available: {', '.join(available)}")
            return

        import yaml

        with open(source_path) as f:
            data = yaml.safe_load(f)

        # Update name and optionally model
        data["name"] = name
        if model_provider:
            data.setdefault("model", {})["provider"] = model_provider

        with open(agent_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        console.print(f"[green]✓[/green] Created agent: [cyan]{agent_path}[/cyan] (cloned from {from_agent})")
        if model_provider:
            console.print(f"  Model: [cyan]{model_provider}[/cyan]")
        console.print()
        console.print("Next steps:")
        console.print(f"  1. Edit [cyan]{agent_path}[/cyan] to customize")
        console.print(f"  2. Run: [cyan]supyagent chat {name}[/cyan]")
        return

    # Default model: CLI flag > config DEFAULT_MODEL > hardcoded fallback
    if not model_provider:
        try:
            cfg = ConfigManager()
            model_provider = cfg.get("DEFAULT_MODEL")
        except Exception:
            pass
    model = model_provider or "anthropic/claude-3-5-sonnet-20241022"
    _warn_model_provider(model)

    # Create template based on type
    if agent_type == "interactive":
        template = f"""name: {name}
description: An interactive AI assistant
version: "1.0"
type: interactive

model:
  provider: {model}
  temperature: 0.7
  max_tokens: 4096

system_prompt: |
  You are a helpful AI assistant named {name}.

  You have access to various tools via supypowers. Use them when needed
  to help accomplish tasks.

  Be concise, helpful, and accurate.

tools:
  allow:
    - "*"  # Allow all tools (customize as needed)

# Set to true to enable the agent to create new supypowers tools
will_create_tools: true

limits:
  max_tool_calls_per_turn: 10
"""
    else:
        template = f"""name: {name}
description: An execution agent for automated tasks
version: "1.0"
type: execution

model:
  provider: {model}
  temperature: 0.3  # Lower temperature for consistency
  max_tokens: 2048

system_prompt: |
  You are a task execution agent. Process the input and produce the output.
  Be precise and follow instructions exactly.
  Do not engage in conversation - just output the result.

tools:
  allow: []  # Execution agents often don't need tools

# Set to true to enable the agent to create new supypowers tools
will_create_tools: false

limits:
  max_tool_calls_per_turn: 5
"""

    agent_path.write_text(template)

    console.print(f"[green]✓[/green] Created agent: [cyan]{agent_path}[/cyan]")
    if model_provider:
        console.print(f"  Model: [cyan]{model_provider}[/cyan]")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Edit [cyan]{agent_path}[/cyan] to customize")
    if agent_type == "execution":
        console.print(f'  2. Run: [cyan]supyagent run {name} "your task"[/cyan]')
    else:
        console.print(f"  2. Run: [cyan]supyagent chat {name}[/cyan]")


@cli.command("list")
def list_agents():
    """List all available agents."""
    agents_dir = Path("agents")

    if not agents_dir.exists():
        console.print("[yellow]No agents directory found.[/yellow]")
        console.print("Create an agent with: [cyan]supyagent new <name>[/cyan]")
        return

    yaml_files = sorted(agents_dir.glob("*.yaml"))

    if not yaml_files:
        console.print("[yellow]No agents found.[/yellow]")
        console.print("Create an agent with: [cyan]supyagent new <name>[/cyan]")
        return

    console.print("[bold]Available agents:[/bold]\n")

    for yaml_file in yaml_files:
        name = yaml_file.stem
        try:
            config = load_agent_config(name)
            agent_type = f"[dim]({config.type})[/dim]"
            desc = (
                config.description[:50] + "..."
                if len(config.description) > 50
                else config.description
            )
            console.print(f"  [cyan]{name}[/cyan] {agent_type}")
            if desc:
                console.print(f"    [dim]{desc}[/dim]")
        except Exception as e:
            console.print(f"  [red]{name}[/red] [dim](invalid: {e})[/dim]")


@cli.command()
@click.argument("agent_name")
@click.option("--new", "new_session", is_flag=True, help="Start a new session")
@click.option("--session", "session_id", help="Resume a specific session by ID")
@click.option("--verbose", "-v", is_flag=True, help="Show tool call details and token usage")
@click.option("--dry-run", is_flag=True, help="Show system prompt and tool schemas, then exit")
def chat(agent_name: str, new_session: bool, session_id: str | None, verbose: bool, dry_run: bool):
    """
    Start an interactive chat session with an agent.

    AGENT_NAME is the name of the agent to chat with.

    By default, resumes the most recent session. Use --new to start fresh,
    or --session <id> to resume a specific session.
    """
    # Load global config (API keys) into environment
    load_config()

    # Load agent config
    try:
        config = load_agent_config(agent_name)
    except AgentNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("\nAvailable agents:")
        agents_dir = Path("agents")
        if agents_dir.exists():
            for f in agents_dir.glob("*.yaml"):
                console.print(f"  - {f.stem}")
        else:
            console.print(
                "  [dim](none - create one with 'supyagent new <name>')[/dim]"
            )
        sys.exit(1)

    # Warn about model provider issues early
    _warn_model_provider(config.model.provider)

    # Initialize session manager
    session_mgr = SessionManager()

    # Determine which session to use
    session = None
    if session_id:
        # Try alias first
        resolved_id = session_mgr.resolve_alias(agent_name, session_id)
        if resolved_id:
            session = session_mgr.load_session(agent_name, resolved_id)
        else:
            # Resume specific session (with prefix matching)
            session = session_mgr.load_session(agent_name, session_id)
            if not session:
                # Try prefix match
                all_sessions = session_mgr.list_sessions(agent_name)
                matches = [s for s in all_sessions if s.session_id.startswith(session_id)]
                if len(matches) == 1:
                    session = session_mgr.load_session(agent_name, matches[0].session_id)
                elif len(matches) > 1:
                    console.print(f"[yellow]Ambiguous prefix '{session_id}'. Matches:[/yellow]")
                    for m in matches:
                        console.print(f"  {m.session_id}: {m.title or '(untitled)'}")
                    sys.exit(1)
                else:
                    console.print(f"[red]Error:[/red] Session '{session_id}' not found")
                    console.print("\nAvailable sessions:")
                    for s in all_sessions:
                        console.print(f"  - {s.session_id}: {s.title or '(untitled)'}")
                    sys.exit(1)
    elif not new_session:
        # Try to resume current session
        session = session_mgr.get_current_session(agent_name)

        # If the session has only a user message with no assistant response,
        # it likely failed (e.g., auth error). Start fresh instead.
        if session and session.messages:
            has_assistant = any(m.type == "assistant" for m in session.messages)
            if not has_assistant and len(session.messages) <= 1:
                console.print(
                    "[dim]Previous session had no responses (possible error). Starting fresh.[/dim]"
                )
                session = None

    # Initialize agent
    try:
        agent = Agent(config, session=session, session_manager=session_mgr)
    except Exception as e:
        console.print(f"[red]Error initializing agent:[/red] {e}")
        sys.exit(1)

    # Dry-run: show config and exit
    if dry_run:
        from supyagent.models.agent_config import get_full_system_prompt

        full_prompt = get_full_system_prompt(
            config, **agent._system_prompt_kwargs()
        )
        console.print(f"\n[bold]Agent:[/bold] {config.name} ({config.type})")
        console.print(f"[bold]Model:[/bold] {config.model.provider}")
        console.print(f"[bold]Tools:[/bold] {len(agent.tools)} available\n")

        console.print("[bold]System Prompt:[/bold]")
        console.print(Panel(full_prompt, border_style="dim", expand=False))

        if agent.tools:
            console.print(f"\n[bold]Tool Schemas ({len(agent.tools)}):[/bold]")
            for tool in agent.tools:
                func = tool.get("function", {})
                name = func.get("name", "?")
                desc = func.get("description", "")
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                console.print(f"  [cyan]{name}[/cyan]: {desc}")
        return

    # Warn if supypowers tools are missing
    if not agent.supypowers_available:
        console.print()
        console.print(
            "[yellow]Warning:[/yellow] No supypowers tools found. "
            "Your agent only has built-in tools."
        )
        console.print(
            "  Check that supypowers is installed: "
            "[cyan]uv tool install supypowers[/cyan]"
        )
        console.print(
            "  Run [cyan]supyagent doctor[/cyan] to diagnose."
        )

    # Print welcome
    console.print()

    if session and session.messages:
        # Resuming existing session
        console.print(
            Panel(
                f"[bold]Resuming session[/bold] [cyan]{agent.session.meta.session_id}[/cyan]\n\n"
                f"[dim]{len(session.messages)} messages in history[/dim]\n"
                f"Model: [cyan]{config.model.provider}[/cyan]\n\n"
                f"[dim]Type /help for commands, /quit to exit[/dim]",
                title=f"💬 {config.name}",
                border_style="cyan",
            )
        )
    else:
        # New session
        console.print(
            Panel(
                f"[bold]New session[/bold] [cyan]{agent.session.meta.session_id}[/cyan]\n\n"
                f"[dim]{config.description or 'No description'}[/dim]\n"
                f"Model: [cyan]{config.model.provider}[/cyan]\n"
                f"Tools: [cyan]{len(agent.tools)} available[/cyan]\n\n"
                f"[dim]Type /help for commands, /quit to exit[/dim]",
                title=f"💬 {config.name}",
                border_style="green",
            )
        )
    console.print()

    # Chat state
    show_tokens = verbose
    debug_mode = verbose

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = console.input("[bold blue]You>[/bold blue] ")

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split()
                cmd = cmd_parts[0].lower() if cmd_parts else ""

                if cmd in ("quit", "exit", "q"):
                    console.print("[dim]Goodbye![/dim]")
                    break

                elif cmd in ("help", "h", "?"):
                    console.print(
                        "\n[bold]Available commands:[/bold]\n"
                        "  /help              Show this help\n"
                        "  /image <path> [msg] Send an image with optional message\n"
                        "  /tools             List available tools\n"
                        "  /creds [action]    Manage credentials (list|set|delete)\n"
                        "  /sessions          List all sessions\n"
                        "  /session <id>      Switch to another session\n"
                        "  /new               Start a new session\n"
                        "  /delete [id]       Delete a session (default: current)\n"
                        "  /rename <title>    Set a display title for the current session\n"
                        "  /history [n]       Show last n messages (default: 10)\n"
                        "  /context           Show context window usage and status\n"
                        "  /tokens            Toggle token usage display after each turn\n"
                        "  /debug [on|off]    Toggle verbose debug mode\n"
                        "  /summarize         Force context summarization\n"
                        "  /export [file]     Export conversation to markdown\n"
                        "  /model [name]      Show or change model\n"
                        "  /reload            Reload tools (pick up new/changed powers)\n"
                        "  /clear             Clear screen\n"
                        "  /quit              Exit the chat\n"
                    )
                    continue

                elif cmd == "tools":
                    tools = agent.get_available_tools()
                    if tools:
                        console.print("\n[bold]Available tools:[/bold]")
                        for tool in tools:
                            console.print(f"  - {tool}")
                        console.print()
                    else:
                        console.print("[dim]No tools available[/dim]")
                    continue

                elif cmd == "sessions":
                    all_sessions = session_mgr.list_sessions(agent_name)
                    if not all_sessions:
                        console.print("[dim]No sessions found[/dim]")
                    else:
                        table = Table(title="Sessions")
                        table.add_column("ID", style="cyan")
                        table.add_column("Title")
                        table.add_column("Msgs", style="dim", justify="right")
                        table.add_column("Updated", style="dim")
                        table.add_column("", style="green")

                        current_id = agent.session.meta.session_id
                        hidden_count = 0
                        for s in all_sessions:
                            marker = "← current" if s.session_id == current_id else ""
                            title = s.title or "(untitled)"
                            updated = s.updated_at.strftime("%Y-%m-%d %H:%M")
                            loaded = session_mgr.load_session(agent_name, s.session_id)
                            msg_count = len(loaded.messages) if loaded else 0
                            if msg_count < 2 and s.session_id != current_id:
                                hidden_count += 1
                                continue
                            table.add_row(s.session_id, title, str(msg_count), updated, marker)

                        console.print(table)
                        if hidden_count:
                            console.print(f"[dim]{hidden_count} empty session(s) hidden[/dim]")
                    continue

                elif cmd == "session":
                    if len(cmd_parts) < 2:
                        console.print("[yellow]Usage: /session <id>[/yellow]")
                        continue

                    target_id = cmd_parts[1]
                    new_sess = session_mgr.load_session(agent_name, target_id)
                    if not new_sess:
                        # Try prefix match
                        all_sessions = session_mgr.list_sessions(agent_name)
                        matches = [s for s in all_sessions if s.session_id.startswith(target_id)]
                        if len(matches) == 1:
                            target_id = matches[0].session_id
                            new_sess = session_mgr.load_session(agent_name, target_id)
                        elif len(matches) > 1:
                            console.print(f"[yellow]Ambiguous prefix '{target_id}'. Matches:[/yellow]")
                            for m in matches:
                                console.print(f"  {m.session_id}: {m.title or '(untitled)'}")
                            continue
                        else:
                            console.print(f"[red]Session '{target_id}' not found[/red]")
                            continue
                    if not new_sess:
                        console.print(f"[red]Session '{target_id}' not found[/red]")
                        continue

                    # Reinitialize agent with new session
                    agent = Agent(config, session=new_sess, session_manager=session_mgr)
                    session_mgr._set_current(agent_name, target_id)
                    console.print(f"[green]Switched to session {target_id}[/green]")
                    continue

                elif cmd == "new":
                    # Start a fresh session
                    agent = Agent(config, session=None, session_manager=session_mgr)
                    console.print(
                        f"[green]Started new session {agent.session.meta.session_id}[/green]"
                    )
                    continue

                elif cmd == "history":
                    n = 10
                    if len(cmd_parts) > 1:
                        try:
                            n = int(cmd_parts[1])
                        except ValueError:
                            pass

                    messages = agent.session.messages[-n:]
                    if not messages:
                        console.print("[dim]No messages in history[/dim]")
                    else:
                        console.print(f"\n[bold]Last {len(messages)} messages:[/bold]")
                        for msg in messages:
                            if msg.type == "user":
                                preview = (msg.content or "")[:80]
                                if len(msg.content or "") > 80:
                                    preview += "..."
                                console.print(f"  [blue]You:[/blue] {preview}")
                            elif msg.type == "assistant":
                                preview = (msg.content or "")[:80]
                                if len(msg.content or "") > 80:
                                    preview += "..."
                                console.print(
                                    f"  [green]{config.name}:[/green] {preview}"
                                )
                            elif msg.type == "tool_result":
                                console.print(f"  [dim]Tool: {msg.name}[/dim]")
                        console.print()
                    continue

                elif cmd == "context":
                    # Show context window status with trigger thresholds
                    from supyagent.core.tokens import count_tools_tokens

                    conversation_messages = [
                        m for m in agent.messages if m.get("role") != "system"
                    ]
                    status = agent.context_manager.get_trigger_status(
                        conversation_messages
                    )

                    console.print("\n[cyan]Context Status[/cyan]")
                    console.print(
                        f"  [dim]Context limit:[/dim] {agent.context_manager.context_limit:,} tokens"
                    )

                    # Tool definition tokens
                    tools_tokens = count_tools_tokens(agent.tools, agent.context_manager.model)
                    if tools_tokens > 0:
                        console.print(
                            f"  [dim]Tool definitions:[/dim] {tools_tokens:,} tokens "
                            f"({len(agent.tools)} tools)"
                        )

                    if agent.context_manager.summary:
                        summary = agent.context_manager.summary
                        console.print(
                            f"  [dim]Last summary:[/dim] {summary.messages_summarized} messages → "
                            f"{summary.token_count} tokens"
                        )
                        console.print(
                            f"  [dim]Created:[/dim] {summary.created_at.strftime('%Y-%m-%d %H:%M')}"
                        )
                    else:
                        console.print("  [dim]No summary yet[/dim]")

                    console.print(
                        "\n[cyan]Summarization Triggers (N messages OR K tokens)[/cyan]"
                    )

                    # Messages trigger (N)
                    msg_pct = int(status["messages_percent"])
                    msg_bar = "█" * (msg_pct // 5) + "░" * (20 - msg_pct // 5)
                    console.print(
                        f"  Messages: {status['messages_since_summary']:,} / "
                        f"{status['messages_threshold']:,} ({msg_pct}%)"
                    )
                    console.print(f"           [{msg_bar}]")

                    # Tokens trigger (K)
                    tok_pct = int(status["tokens_percent"])
                    tok_bar = "█" * (tok_pct // 5) + "░" * (20 - tok_pct // 5)
                    console.print(
                        f"  Tokens:   {status['total_tokens']:,} / "
                        f"{status['tokens_threshold']:,} ({tok_pct}%)"
                    )
                    console.print(f"           [{tok_bar}]")

                    if status["will_trigger"]:
                        console.print(
                            "\n  [yellow]⚡ Summarization will trigger on next message[/yellow]"
                        )
                    console.print()
                    continue

                elif cmd == "summarize":
                    # Force summarization
                    conversation_messages = [
                        m for m in agent.messages if m.get("role") != "system"
                    ]
                    if len(conversation_messages) < 4:
                        console.print(
                            "[yellow]Not enough messages to summarize[/yellow]"
                        )
                        continue

                    console.print("[dim]Generating context summary...[/dim]")
                    try:
                        summary = agent.context_manager.generate_summary(
                            conversation_messages
                        )
                        console.print(
                            f"[green]✓[/green] Summarized {summary.messages_summarized} messages "
                            f"({summary.token_count} tokens)"
                        )
                    except Exception as e:
                        console.print(f"[red]Error:[/red] {e}")
                    continue

                elif cmd == "reload":
                    old_count = len(agent.tools)
                    new_count = agent.reload_tools()
                    console.print(
                        f"[green]✓[/green] Reloaded tools: {new_count} "
                        f"(was {old_count})"
                    )
                    continue

                elif cmd == "clear":
                    console.clear()
                    continue

                elif cmd == "creds":
                    action = cmd_parts[1] if len(cmd_parts) > 1 else "list"
                    cred_name = cmd_parts[2] if len(cmd_parts) > 2 else None

                    if action == "list":
                        creds = agent.credential_manager.list_credentials(agent_name)
                        if not creds:
                            console.print("[dim]No stored credentials[/dim]")
                        else:
                            console.print("\n[bold]Stored credentials:[/bold]")
                            for c in creds:
                                console.print(f"  - {c}")
                            console.print()
                    elif action == "set" and cred_name:
                        result = agent.credential_manager.prompt_for_credential(
                            cred_name, "Manually setting credential"
                        )
                        if result:
                            value, persist = result
                            agent.credential_manager.set(
                                agent_name, cred_name, value, persist
                            )
                            console.print(
                                f"[green]Credential {cred_name} saved[/green]"
                            )
                        else:
                            console.print("[dim]Cancelled[/dim]")
                    elif action == "delete" and cred_name:
                        if agent.credential_manager.delete(agent_name, cred_name):
                            console.print(
                                f"[green]Credential {cred_name} deleted[/green]"
                            )
                        else:
                            console.print(
                                f"[yellow]Credential {cred_name} not found[/yellow]"
                            )
                    else:
                        console.print(
                            "[yellow]Usage: /creds [list|set|delete] [name][/yellow]"
                        )
                    continue

                elif cmd == "export":
                    filename = (
                        cmd_parts[1]
                        if len(cmd_parts) > 1
                        else f"{agent_name}_{agent.session.meta.session_id}.md"
                    )

                    lines = [f"# Conversation with {config.name}", ""]
                    prev_role = None
                    for msg in agent.session.messages:
                        if msg.type == "user":
                            lines.append(f"**You:** {msg.content}\n")
                            prev_role = "user"
                        elif msg.type == "assistant":
                            content = msg.content
                            if not content or not str(content).strip():
                                continue
                            if prev_role == "assistant":
                                lines.append(f"{content}\n")
                            else:
                                lines.append(f"**{config.name}:** {content}\n")
                            prev_role = "assistant"

                    with open(filename, "w") as f:
                        f.write("\n".join(lines))
                    console.print(f"[green]Exported to {filename}[/green]")
                    continue

                elif cmd == "model":
                    if len(cmd_parts) > 1:
                        new_model = cmd_parts[1]
                        agent.llm.change_model(new_model)
                        console.print(f"[green]Model changed to {new_model}[/green]")
                    else:
                        console.print(f"Current model: [cyan]{agent.llm.model}[/cyan]")
                    continue

                elif cmd == "tokens":
                    show_tokens = not show_tokens
                    state = "on" if show_tokens else "off"
                    console.print(f"[green]Token display {state}[/green]")
                    continue

                elif cmd == "debug":
                    if len(cmd_parts) > 1:
                        debug_mode = cmd_parts[1].lower() in ("on", "true", "1", "yes")
                    else:
                        debug_mode = not debug_mode
                    state = "on" if debug_mode else "off"
                    console.print(f"[green]Debug mode {state}[/green]")
                    if debug_mode:
                        import logging

                        logging.basicConfig(
                            level=logging.DEBUG,
                            format="[%(levelname)s] %(name)s: %(message)s",
                        )
                        logging.getLogger("httpx").setLevel(logging.WARNING)
                        logging.getLogger("httpcore").setLevel(logging.WARNING)
                        logging.getLogger("litellm").setLevel(logging.INFO)
                    continue

                elif cmd in ("rename", "title"):
                    if len(cmd_parts) < 2:
                        console.print("[yellow]Usage: /rename <title>[/yellow]")
                        console.print("[dim]Sets a display title for the current session[/dim]")
                        continue
                    new_title = " ".join(cmd_parts[1:])
                    agent.session.meta.title = new_title
                    session_mgr._update_meta(agent.session)
                    console.print(f"[green]Session title set to \"{new_title}\"[/green]")
                    continue

                elif cmd == "image":
                    if len(cmd_parts) < 2:
                        console.print("[yellow]Usage: /image <path> [message][/yellow]")
                        continue

                    from supyagent.utils.media import IMAGE_EXTENSIONS, wrap_with_image

                    image_path = Path(cmd_parts[1]).expanduser()
                    if not image_path.exists():
                        console.print(f"[red]File not found: {image_path}[/red]")
                        continue
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        console.print(f"[red]Unsupported image type: {image_path.suffix}[/red]")
                        console.print(f"[dim]Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}[/dim]")
                        continue

                    text_msg = " ".join(cmd_parts[2:]) if len(cmd_parts) > 2 else "Describe this image."
                    multimodal_content = wrap_with_image(text_msg, image_path)

                    console.print(f"[dim]Sending image: {image_path.name}[/dim]")
                    console.print()
                    console.print(f"[bold green]{config.name}>[/bold green]")

                    try:
                        collected_response = ""
                        for event_type, data in agent.send_message_stream(multimodal_content):
                            if event_type == "text":
                                click.echo(data, nl=False)
                                collected_response += data
                            elif event_type == "tool_start":
                                tool_name = data.get("name", "?")
                                console.print(f"\n  [dim]⚙ {tool_name}...[/dim]")
                            elif event_type == "tool_end":
                                result = data.get("result", {})
                                status = "✓" if result.get("ok") else "✗"
                                console.print(f"  [dim]  {status} done[/dim]")
                            elif event_type == "done":
                                pass
                        click.echo("")
                    except Exception as e:
                        console.print(f"\n[red]Error: {e}[/red]")
                    console.print()
                    continue

                elif cmd == "delete":
                    target_id = cmd_parts[1] if len(cmd_parts) > 1 else agent.session.meta.session_id
                    is_current = target_id == agent.session.meta.session_id

                    # Find title for confirmation
                    target_session = session_mgr.load_session(agent_name, target_id)
                    if not target_session:
                        # Try prefix match
                        all_sessions = session_mgr.list_sessions(agent_name)
                        matches = [s for s in all_sessions if s.session_id.startswith(target_id)]
                        if len(matches) == 1:
                            target_id = matches[0].session_id
                            is_current = target_id == agent.session.meta.session_id
                            target_session = session_mgr.load_session(agent_name, target_id)
                        elif len(matches) > 1:
                            console.print(f"[yellow]Ambiguous prefix '{target_id}'. Matches:[/yellow]")
                            for m in matches:
                                console.print(f"  {m.session_id}: {m.title or '(untitled)'}")
                            continue
                        else:
                            console.print(f"[red]Session '{target_id}' not found[/red]")
                            continue

                    title = target_session.meta.title or "(untitled)"
                    if not click.confirm(f"Delete session {target_id} (\"{title}\")?"):
                        continue

                    session_mgr.delete_session(agent_name, target_id)
                    console.print(f"[green]Deleted session {target_id}[/green]")

                    if is_current:
                        # Start a new session since we deleted the current one
                        agent = Agent(config, session=None, session_manager=session_mgr)
                        console.print(
                            f"[green]Started new session {agent.session.meta.session_id}[/green]"
                        )
                    continue

                else:
                    console.print(f"[yellow]Unknown command: /{cmd}[/yellow]")
                    continue

            # Send message to agent with streaming
            console.print()
            console.print(f"[bold green]{config.name}>[/bold green]")

            try:
                collected_response = ""
                in_reasoning = False
                for event_type, data in agent.send_message_stream(user_input):
                    if event_type == "text":
                        # If we were in reasoning, add newline before text
                        if in_reasoning:
                            click.echo("")  # End reasoning line
                            in_reasoning = False
                        # Print text as it streams
                        click.echo(data, nl=False)
                        collected_response += data
                    elif event_type == "reasoning":
                        # Show LLM reasoning/thinking if available
                        if not in_reasoning:
                            # Start of reasoning - show emoji
                            console.print("[magenta dim]💭 [/magenta dim]", end="")
                            in_reasoning = True
                        # Stream the reasoning content
                        console.print(f"[magenta dim]{data}[/magenta dim]", end="")
                    elif event_type == "tool_start":
                        # End reasoning if we were in it
                        if in_reasoning:
                            click.echo("")  # End reasoning line
                            in_reasoning = False
                        # Show tool being called with inputs
                        tool_name = data.get("name", "unknown")
                        console.print(f"\n[cyan]⚡ {tool_name}[/cyan]")
                        # Show tool inputs
                        if data.get("arguments"):
                            try:
                                args = json.loads(data["arguments"])
                                # Format args compactly on one line if simple
                                if len(args) <= 2 and all(
                                    isinstance(v, (str, int, bool))
                                    for v in args.values()
                                ):
                                    args_str = ", ".join(
                                        f"{k}={repr(v)}" for k, v in args.items()
                                    )
                                    console.print(f"[dim]   └─ {args_str}[/dim]")
                                else:
                                    args_str = json.dumps(args, indent=2)
                                    for line in args_str.split("\n"):
                                        console.print(f"[dim]   {line}[/dim]")
                            except json.JSONDecodeError:
                                pass
                    elif event_type == "tool_end":
                        # Show tool completed
                        result = data.get("result", {})
                        if result.get("ok", False):
                            console.print("[green]   ✓ done[/green]")
                        else:
                            error = result.get("error", "failed")
                            console.print(f"[red]   ✗ {error}[/red]")
                    elif event_type == "done":
                        # End reasoning if we were in it
                        if in_reasoning:
                            click.echo("")  # End reasoning line
                            in_reasoning = False
                        # Ensure newline after streaming
                        if collected_response:
                            click.echo("")  # Final newline

                # Show token usage if enabled
                if show_tokens:
                    try:
                        from supyagent.core.tokens import count_tools_tokens

                        conversation_messages = [
                            m for m in agent.messages if m.get("role") != "system"
                        ]
                        status = agent.context_manager.get_trigger_status(
                            conversation_messages
                        )
                        tools_tokens = count_tools_tokens(
                            agent.tools, agent.context_manager.model
                        )
                        ctx_limit = agent.context_manager.context_limit
                        total = status["total_tokens"] + tools_tokens
                        pct = int(total / ctx_limit * 100) if ctx_limit else 0
                        console.print(
                            f"\n  [dim]tokens: {status['total_tokens']:,} msgs + "
                            f"{tools_tokens:,} tools | "
                            f"context: {total:,} / {ctx_limit:,} ({pct}%)[/dim]"
                        )
                    except Exception:
                        pass

                # Show debug info if enabled
                if debug_mode:
                    try:
                        conversation_messages = [
                            m for m in agent.messages if m.get("role") != "system"
                        ]
                        console.print(
                            f"  [dim][DEBUG] Messages: {len(conversation_messages)} | "
                            f"Tools: {len(agent.tools)}[/dim]"
                        )
                    except Exception:
                        pass

                console.print()
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}\n")
                continue

        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/dim]")
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break


@cli.command()
@click.argument("agent_name")
@click.option("--search", "-s", "search_query", help="Search sessions by title keyword")
@click.option("--delete", "-d", "delete_id", help="Delete a specific session by ID (or prefix)")
@click.option("--delete-all", "delete_all", is_flag=True, help="Delete all sessions for this agent")
@click.option("--name", "-n", nargs=2, type=str, metavar="SESSION_ID ALIAS",
              help="Set a named alias for a session")
@click.option("--export", "-e", "export_id", help="Export a session (by ID, prefix, or alias)")
@click.option("--format", "-f", "export_fmt", type=click.Choice(["markdown", "json"]),
              default="markdown", help="Export format (default: markdown)")
@click.option("--all", "-a", "show_all", is_flag=True,
              help="Include empty sessions (< 2 messages)")
def sessions(
    agent_name: str,
    search_query: str | None,
    delete_id: str | None,
    delete_all: bool,
    name: tuple[str, str] | None,
    export_id: str | None,
    export_fmt: str,
    show_all: bool,
):
    """
    List, search, name, export, and manage sessions for an agent.

    \b
    Examples:
        supyagent sessions myagent
        supyagent sessions myagent --search "project"
        supyagent sessions myagent --name abc123 project-alpha
        supyagent sessions myagent --export project-alpha
        supyagent sessions myagent --export abc123 --format json
    """
    session_mgr = SessionManager()

    # Handle naming a session
    if name:
        sid, alias = name
        # Verify session exists (resolve prefix)
        session = session_mgr.load_session(agent_name, sid)
        if not session:
            all_sessions = session_mgr.list_sessions(agent_name)
            matches = [s for s in all_sessions if s.session_id.startswith(sid)]
            if len(matches) == 1:
                sid = matches[0].session_id
            else:
                console.print(f"[red]Session '{sid}' not found[/red]")
                return
        session_mgr.set_alias(agent_name, alias, sid)
        console.print(f"[green]✓[/green] Named session {sid} as [cyan]{alias}[/cyan]")
        return

    # Handle export
    if export_id:
        # Resolve alias or prefix
        resolved = session_mgr.resolve_alias(agent_name, export_id)
        if not resolved:
            # Try as direct ID or prefix
            session = session_mgr.load_session(agent_name, export_id)
            if session:
                resolved = export_id
            else:
                all_sessions = session_mgr.list_sessions(agent_name)
                matches = [s for s in all_sessions if s.session_id.startswith(export_id)]
                if len(matches) == 1:
                    resolved = matches[0].session_id
        if not resolved:
            console.print(f"[red]Session '{export_id}' not found[/red]")
            sys.exit(1)
        output = session_mgr.export_session(agent_name, resolved, fmt=export_fmt)
        if output:
            click.echo(output)
        else:
            console.print(f"[red]Failed to export session '{resolved}'[/red]")
            sys.exit(1)
        return

    # Handle single delete
    if delete_id:
        # Try exact match first, then prefix
        result = session_mgr.delete_session(agent_name, delete_id)
        if not result:
            all_sessions = session_mgr.list_sessions(agent_name)
            matches = [s for s in all_sessions if s.session_id.startswith(delete_id)]
            if len(matches) == 1:
                session_mgr.delete_session(agent_name, matches[0].session_id)
                title = matches[0].title or "(untitled)"
                console.print(f"[green]✓[/green] Deleted session {matches[0].session_id} (\"{title}\")")
            elif len(matches) > 1:
                console.print(f"[yellow]Ambiguous prefix '{delete_id}'. Matches:[/yellow]")
                for m in matches:
                    console.print(f"  {m.session_id}: {m.title or '(untitled)'}")
            else:
                console.print(f"[red]Session '{delete_id}' not found[/red]")
        else:
            console.print(f"[green]✓[/green] Deleted session {delete_id}")
        return

    # Handle delete-all
    if delete_all:
        count = session_mgr.delete_all_sessions(agent_name)
        if count == 0:
            console.print(f"[dim]No sessions to delete for '{agent_name}'[/dim]")
        else:
            console.print(f"[green]✓[/green] Deleted {count} session(s) for '{agent_name}'")
        return

    # Handle search
    if search_query:
        session_list = session_mgr.search_sessions(agent_name, query=search_query)
        if not session_list:
            console.print(f"[dim]No sessions matching '{search_query}' for '{agent_name}'[/dim]")
            return
    else:
        session_list = session_mgr.list_sessions(agent_name)

    if not session_list:
        console.print(f"[dim]No sessions found for '{agent_name}'[/dim]")
        console.print(
            f"\nStart a session with: [cyan]supyagent chat {agent_name}[/cyan]"
        )
        return

    # Get current session
    current = session_mgr.get_current_session(agent_name)
    current_id = current.meta.session_id if current else None

    title_text = f"Sessions for {agent_name}"
    if search_query:
        title_text += f" (search: '{search_query}')"

    # Load aliases for display
    aliases = session_mgr._load_aliases(agent_name)
    alias_by_id = {v: k for k, v in aliases.items()}

    table = Table(title=title_text)
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Title")
    table.add_column("Msgs", style="dim", justify="right")
    table.add_column("Created", style="dim")
    table.add_column("Updated", style="dim")
    table.add_column("", style="green")

    hidden = 0
    for s in session_list:
        marker = "← current" if s.session_id == current_id else ""
        title = s.title or "(untitled)"
        alias_name = alias_by_id.get(s.session_id, "")
        created = s.created_at.strftime("%Y-%m-%d %H:%M")
        updated = s.updated_at.strftime("%Y-%m-%d %H:%M")
        # Count messages
        loaded = session_mgr.load_session(agent_name, s.session_id)
        msg_count = len(loaded.messages) if loaded else 0
        # Hide empty sessions unless --all or it's the current session
        if not show_all and msg_count < 2 and s.session_id != current_id:
            hidden += 1
            continue
        table.add_row(s.session_id, alias_name, title, str(msg_count), created, updated, marker)

    console.print(table)
    if hidden:
        console.print(
            f"[dim]{hidden} empty session(s) hidden. Use --all to show them.[/dim]"
        )
    console.print()
    console.print(
        "[dim]Resume a session:[/dim] supyagent chat " + agent_name + " --session <id or name>"
    )


@cli.command()
@click.argument("agent_name")
def show(agent_name: str):
    """Show details about an agent."""
    try:
        config = load_agent_config(agent_name)
    except AgentNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print(f"\n[bold cyan]{config.name}[/bold cyan] v{config.version}")
    console.print(f"[dim]{config.description}[/dim]\n")

    console.print(f"[bold]Type:[/bold] {config.type}")
    console.print(f"[bold]Model:[/bold] {config.model.provider}")
    console.print(f"[bold]Temperature:[/bold] {config.model.temperature}")
    console.print(f"[bold]Max Tokens:[/bold] {config.model.max_tokens}")

    if config.tools.allow:
        console.print("\n[bold]Allowed Tools:[/bold]")
        for pattern in config.tools.allow:
            console.print(f"  - {pattern}")

    if config.tools.deny:
        console.print("\n[bold]Denied Tools:[/bold]")
        for pattern in config.tools.deny:
            console.print(f"  - {pattern}")

    if config.delegates:
        console.print("\n[bold]Delegates:[/bold]")
        for delegate in config.delegates:
            console.print(f"  - {delegate}")

    console.print("\n[bold]System Prompt:[/bold]")
    console.print(Panel(config.system_prompt, border_style="dim"))


@cli.command()
@click.argument("agent_name")
def validate(agent_name: str):
    """
    Validate an agent's configuration.

    Checks YAML syntax, required fields, model provider, delegate references,
    and tool permission patterns.

    \b
    Examples:
        supyagent validate myagent
        supyagent validate planner
    """
    agents_dir = Path("agents")

    # Step 1: Check file exists
    config_path = agents_dir / f"{agent_name}.yaml"
    if not config_path.exists():
        console.print(f"[red]  x[/red] File not found: {config_path}")
        available = [f.stem for f in agents_dir.glob("*.yaml")] if agents_dir.exists() else []
        if available:
            console.print(f"    Available agents: {', '.join(available)}")
        sys.exit(1)

    # Step 2: Parse YAML and Pydantic validation
    try:
        config = load_agent_config(agent_name, agents_dir)
    except AgentConfigError as e:
        console.print(f"\n[red]  x[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]  x[/red] Failed to parse YAML: {e}")
        sys.exit(1)

    console.print("[green]  ✓[/green] YAML syntax and required fields OK")

    # Step 3: Deep validation
    issues = validate_agent_config(config, agents_dir)

    if issues:
        for issue in issues:
            console.print(f"[yellow]  ![/yellow] {issue}")
    else:
        console.print("[green]  ✓[/green] Model, delegates, and tools OK")

    # Step 4: Check local tools
    from supyagent.core.tools import discover_tools as discover_sp_tools
    from supyagent.core.tools import filter_tools, supypowers_to_openai_tools

    sp_tools = discover_sp_tools()
    local_count = 0
    if sp_tools:
        openai_tools = supypowers_to_openai_tools(sp_tools)
        filtered = filter_tools(openai_tools, config.tools)
        local_count = len(filtered)
        console.print(f"[green]  ✓[/green] Local tools: {local_count} available")
    else:
        console.print("[dim]  -[/dim] No local tools (run 'supyagent init')")

    # Step 5: Check service connectivity
    service_count = 0
    if config.service.enabled:
        from supyagent.core.service import get_service_client

        client = get_service_client()
        if client:
            if client.health_check():
                svc_tools = client.discover_tools()
                service_count = len(svc_tools)
                console.print(
                    f"[green]  ✓[/green] Service: connected ({service_count} tools)"
                )
            else:
                console.print(f"[yellow]  ![/yellow] Service: unreachable at {client.base_url}")
            client.close()
        else:
            console.print(
                "[yellow]  ![/yellow] Service enabled but not connected. "
                "Run 'supyagent connect'"
            )

    # Step 6: Summary
    console.print(
        f"\n  Agent: [cyan]{config.name}[/cyan] ({config.type})"
        f"\n  Model: [cyan]{config.model.provider}[/cyan]"
        f"\n  Tools: {local_count} local, {service_count} service"
        f"\n  Patterns: {len(config.tools.allow)} allow, {len(config.tools.deny)} deny"
    )
    if config.delegates:
        console.print(f"  Delegates: {', '.join(config.delegates)}")

    if not issues:
        console.print("\n[green]  All checks passed.[/green]")


@cli.command()
def doctor():
    """
    Diagnose your supyagent setup.

    Checks supypowers installation, agents directory, API keys,
    default tools, sessions directory, and config encryption.

    \b
    Example:
        supyagent doctor
    """
    all_ok = True

    # 1. Check supypowers
    try:
        import subprocess

        result = subprocess.run(
            ["supypowers", "docs", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            console.print("[green]  ✓[/green] supypowers installed")
        else:
            console.print(
                f"[red]  x[/red] supypowers not working (exit code {result.returncode})"
            )
            all_ok = False
    except FileNotFoundError:
        console.print("[red]  x[/red] supypowers not installed")
        console.print("    Install: [cyan]uv tool install supypowers[/cyan]")
        all_ok = False
    except Exception as e:
        console.print(f"[yellow]  ![/yellow] supypowers check failed: {e}")

    # 2. Check agents directory
    agents_dir = Path("agents")
    if agents_dir.exists():
        agent_files = list(agents_dir.glob("*.yaml"))
        console.print(f"[green]  ✓[/green] agents/ directory found ({len(agent_files)} agents)")
    else:
        console.print("[yellow]  ![/yellow] agents/ directory not found")
        console.print("    Run: [cyan]supyagent init[/cyan]")
        all_ok = False

    # 3. Check API keys
    config_mgr = ConfigManager()
    stored_keys = config_mgr._load_keys()
    env_keys = [
        k for k in os.environ
        if any(p in k for p in ["API_KEY", "OPENAI", "ANTHROPIC", "OPENROUTER", "GOOGLE", "AZURE"])
    ]

    if stored_keys or env_keys:
        key_names = list(stored_keys.keys()) + [k for k in env_keys if k not in stored_keys]
        console.print(f"[green]  ✓[/green] API keys configured: {', '.join(key_names[:5])}")
        if len(key_names) > 5:
            console.print(f"    ... and {len(key_names) - 5} more")
    else:
        console.print("[yellow]  ![/yellow] No API keys configured")
        console.print("    Run: [cyan]supyagent config set[/cyan]")
        all_ok = False

    # 4. Check agents requiring keys
    if agents_dir.exists():
        for agent_file in agents_dir.glob("*.yaml"):
            try:
                cfg = load_agent_config(agent_file.stem)
                provider = cfg.model.provider.lower()
                needed = None
                if "anthropic" in provider:
                    needed = "ANTHROPIC_API_KEY"
                elif "openai/" in provider:
                    needed = "OPENAI_API_KEY"
                elif "openrouter" in provider:
                    needed = "OPENROUTER_API_KEY"
                elif "google" in provider or "gemini" in provider:
                    needed = "GOOGLE_API_KEY"

                if needed and needed not in stored_keys and needed not in os.environ:
                    console.print(
                        f"[yellow]  ![/yellow] {agent_file.stem} needs {needed} "
                        f"(model: {cfg.model.provider})"
                    )
                    all_ok = False
            except Exception:
                pass

    # 5. Check default tools
    tools_dir = Path("powers")
    if tools_dir.exists():
        tool_files = [f for f in tools_dir.glob("*.py") if f.name != "__init__.py"]
        console.print(f"[green]  ✓[/green] powers/ directory found ({len(tool_files)} tools)")
    else:
        console.print("[yellow]  ![/yellow] powers/ not found (no custom tools)")
        console.print("    Run: [cyan]supyagent init[/cyan]")

    # 6. Check sessions directory
    sessions_dir = Path(".supyagent/sessions")
    if sessions_dir.exists():
        console.print("[green]  ✓[/green] Sessions directory writable")
    else:
        console.print("[dim]  -[/dim] No sessions yet (will be created on first chat)")

    # 7. Check config encryption
    config_dir = Path.home() / ".supyagent" / "config"
    key_file = config_dir / "key.key"
    if key_file.exists():
        console.print("[green]  ✓[/green] Config encryption working")
    else:
        console.print("[dim]  -[/dim] Config encryption not yet initialized")

    # 8. Check agent registry
    registry = AgentRegistry()
    reg_stats = registry.stats()
    if reg_stats["total"] > 0:
        parts = []
        if reg_stats["active"]:
            parts.append(f"{reg_stats['active']} active")
        if reg_stats["completed"]:
            parts.append(f"{reg_stats['completed']} completed")
        if reg_stats["failed"]:
            parts.append(f"{reg_stats['failed']} failed")
        console.print(
            f"[green]  ✓[/green] Agent registry: {reg_stats['total']} entries "
            f"({', '.join(parts)})"
        )
        if reg_stats["total"] > 20:
            console.print(
                "    [yellow]Tip:[/yellow] Run [cyan]supyagent cleanup[/cyan] to prune stale entries"
            )
    else:
        console.print("[dim]  -[/dim] Agent registry empty")

    # Summary
    if all_ok:
        console.print("\n[green]  All checks passed.[/green]")
    else:
        console.print("\n[yellow]  Some issues found. See above for fixes.[/yellow]")


# =============================================================================
# Tools Commands
# =============================================================================

# Cloud tools shown when user isn't connected to supyagent service
_CLOUD_TOOL_PREVIEWS = [
    ("gmail__send_message", "Send an email via Gmail"),
    ("gmail__list_messages", "List recent emails from Gmail"),
    ("slack__post_message", "Post a message to a Slack channel"),
    ("slack__list_channels", "List available Slack channels"),
    ("github__create_issue", "Create a GitHub issue"),
    ("github__list_repos", "List GitHub repositories"),
    ("google_calendar__list_events", "List upcoming calendar events"),
    ("google_calendar__create_event", "Create a calendar event"),
    ("google_drive__list_files", "List files in Google Drive"),
    ("discord__send_message", "Send a message to a Discord channel"),
    ("notion__search", "Search across Notion pages"),
    ("telegram__send_message", "Send a Telegram message"),
]


def _show_cloud_tools_status() -> None:
    """Show cloud tools as connected or locked depending on service status."""
    try:
        from supyagent.core.service import get_service_client

        client = get_service_client()
        if client:
            # Connected — show discovered cloud tools count
            tools = client.discover_tools()
            if tools:
                console.print(f"\n[green]Cloud tools: {len(tools)} available[/green] (via supyagent service)")
                return
    except Exception:
        pass

    # Not connected — show preview of what's available
    console.print()
    console.print(
        "[dim]Cloud tools (run [cyan]supyagent connect[/cyan] to unlock):[/dim]"
    )
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="dim")
    for name, desc in _CLOUD_TOOL_PREVIEWS:
        table.add_row(f"  🔒 {name}", desc)
    console.print(table)
    console.print(f"  [dim]... and 50+ more across 12 providers[/dim]")


@cli.group("tools")
def tools_group():
    """Discover and manage supypowers tools."""
    pass


@tools_group.command("list")
@click.option("--agent", "-a", "agent_name", help="Filter by agent's tool permissions")
def tools_list(agent_name: str | None):
    """
    List all available tools.

    Shows all discovered supypowers tools. Use --agent to see only
    the tools available to a specific agent after permission filtering.

    \b
    Examples:
        supyagent tools list
        supyagent tools list --agent myagent
    """
    from supyagent.core.tools import discover_tools, filter_tools, supypowers_to_openai_tools

    sp_tools = discover_tools()

    if not sp_tools:
        console.print("[yellow]No supypowers tools found.[/yellow]")
        console.print()
        console.print("Possible causes:")
        console.print(
            "  - supypowers not installed: [cyan]uv tool install supypowers[/cyan]"
        )
        console.print("  - supypowers not on PATH: check [cyan]which supypowers[/cyan]")
        console.print(
            "  - No powers/ directory: run [cyan]supyagent init[/cyan]"
        )
        # Still show cloud tools below
        openai_tools = []
    else:
        openai_tools = supypowers_to_openai_tools(sp_tools)

        # Filter by agent permissions if specified
        if agent_name:
            try:
                config = load_agent_config(agent_name)
                openai_tools = filter_tools(openai_tools, config.tools)
                console.print(f"[dim]Tools available to '{agent_name}':[/dim]\n")
            except (AgentNotFoundError, AgentConfigError) as e:
                console.print(f"[red]Error:[/red] {e}")
                sys.exit(1)
        else:
            console.print("[dim]All discovered tools:[/dim]\n")

        table = Table()
        table.add_column("Tool", style="cyan")
        table.add_column("Description")

        for tool in openai_tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            # Truncate long descriptions
            if len(desc) > 70:
                desc = desc[:67] + "..."
            table.add_row(name, desc)

        console.print(table)
        console.print(f"\n[dim]{len(openai_tools)} local tools[/dim]")

    # Show cloud tools status
    _show_cloud_tools_status()


@tools_group.command("new")
@click.argument("name")
def tools_new(name: str):
    """
    Create a new tool from template.

    NAME is the tool name (will create powers/NAME.py).

    \b
    Example:
        supyagent tools new github_api
    """
    tools_dir = Path("powers")
    if not tools_dir.exists():
        console.print("[yellow]powers/ directory not found.[/yellow]")
        console.print("Run [cyan]supyagent init[/cyan] first.")
        return

    tool_path = tools_dir / f"{name}.py"
    if tool_path.exists():
        if not click.confirm(f"Tool '{name}' already exists. Overwrite?"):
            return

    # Generate class names from tool name
    class_name = "".join(w.capitalize() for w in name.split("_"))

    template = f'''# /// script
# dependencies = ["pydantic"]
# ///
"""
{name} - Custom supypowers tool.
"""
from pydantic import BaseModel, Field


class {class_name}Input(BaseModel):
    """{class_name} input parameters."""
    value: str = Field(..., description="Input value")


class {class_name}Output(BaseModel):
    """{class_name} output."""
    ok: bool
    data: str | None = None
    error: str | None = None


def {name}(input: {class_name}Input) -> {class_name}Output:
    """
    Describe what this tool does.

    Examples:
        >>> {name}({{"value": "test"}})
    """
    try:
        # Your implementation here
        return {class_name}Output(ok=True, data=f"Processed: {{input.value}}")
    except Exception as e:
        return {class_name}Output(ok=False, error=str(e))
'''

    tool_path.write_text(template)
    console.print(f"[green]  ✓[/green] Created [cyan]{tool_path}[/cyan]")
    console.print(f"\n  The tool will be available as [cyan]{name}__<function_name>[/cyan]")
    console.print("  Edit the file to add your implementation.")


@tools_group.command("test")
@click.argument("tool_name")
@click.argument("args_json", default="{}")
@click.option("--secrets", "-s", multiple=True, help="Secrets as KEY=VALUE or .env file")
def tools_test(tool_name: str, args_json: str, secrets: tuple[str, ...]):
    """
    Test a tool outside of an agent.

    TOOL_NAME is the full tool name (e.g., shell__run_command).
    ARGS_JSON is a JSON string of arguments.

    \b
    Examples:
        supyagent tools test shell__run_command '{"command": "echo hello"}'
        supyagent tools test files__read_file '{"path": "README.md"}'
    """
    from supyagent.core.tools import execute_tool

    if "__" not in tool_name:
        console.print("[red]Error:[/red] Tool name must be in 'script__function' format")
        console.print("  Example: shell__run_command, files__read_file")
        sys.exit(1)

    script, func = tool_name.split("__", 1)

    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON arguments: {e}")
        sys.exit(1)

    secrets_dict = parse_secrets(secrets)

    console.print(f"[dim]Running {tool_name}...[/dim]")
    result = execute_tool(script, func, args, secrets=secrets_dict)

    if result.get("ok"):
        console.print("[green]  ✓[/green] ok: true")
        data = result.get("data")
        if data is not None:
            if isinstance(data, str) and len(data) > 200:
                console.print(f"  data: {data[:200]}...")
            else:
                console.print(f"  data: {json.dumps(data, indent=2) if not isinstance(data, str) else data}")
    else:
        console.print("[red]  x[/red] ok: false")
        console.print(f"  error: {result.get('error', 'unknown')}")
        if result.get("error_type"):
            console.print(f"  error_type: {result['error_type']}")

    if result.get("duration_ms"):
        console.print(f"  duration: {result['duration_ms']}ms")


@cli.command()
def schema():
    """
    Show the full agent configuration schema.

    Displays all available YAML fields, their types, defaults, and descriptions.
    Use this as a reference when writing agent configuration files.
    """
    from supyagent.models.agent_config import (
        AgentConfig,
        ContextSettings,
        ModelConfig,
        SupervisorSettings,
        ToolPermissions,
    )

    def _print_model(model_cls: type, indent: int = 0) -> None:
        prefix = "  " * indent
        for name, field in model_cls.model_fields.items():
            field_type = str(field.annotation).replace("typing.", "")
            # Simplify common types
            for old, new in [
                ("list[str]", "list[str]"),
                ("<class 'str'>", "str"),
                ("<class 'int'>", "int"),
                ("<class 'float'>", "float"),
                ("<class 'bool'>", "bool"),
            ]:
                field_type = field_type.replace(old, new)

            default = field.default
            if default is not None and str(default) != "PydanticUndefined":
                default_str = f" [dim](default: {default})[/dim]"
            elif str(default) == "PydanticUndefined":
                default_str = " [red](required)[/red]"
            else:
                default_str = ""

            desc = ""
            if field.description:
                desc = f"\n{prefix}    [dim]# {field.description}[/dim]"

            console.print(f"{prefix}  [cyan]{name}[/cyan]{default_str}{desc}")

    console.print("\n[bold]Agent Configuration Schema[/bold]\n")

    console.print("[bold]Top-level fields:[/bold]")
    _print_model(AgentConfig)

    console.print("\n[bold]model:[/bold]")
    _print_model(ModelConfig, indent=1)

    console.print("\n[bold]tools:[/bold]")
    _print_model(ToolPermissions, indent=1)

    console.print("\n[bold]context:[/bold]")
    _print_model(ContextSettings, indent=1)

    console.print("\n[bold]supervisor:[/bold]")
    _print_model(SupervisorSettings, indent=1)

    console.print("\n[bold]limits (dict):[/bold]")
    console.print("    [cyan]max_tool_calls_per_turn[/cyan] [dim](default: 20)[/dim]")
    console.print("      [dim]# Max tool calls per user message[/dim]")
    console.print("    [cyan]circuit_breaker_threshold[/cyan] [dim](default: 3)[/dim]")
    console.print("      [dim]# Block a tool after N consecutive failures[/dim]")
    console.print()


def parse_secrets(secrets: tuple[str, ...]) -> dict[str, str]:
    """
    Parse secrets from KEY=VALUE pairs or .env files.

    Args:
        secrets: Tuple of "KEY=VALUE" strings or file paths

    Returns:
        Dict of secret key -> value
    """
    result: dict[str, str] = {}

    for secret in secrets:
        if "=" in secret:
            # KEY=VALUE format
            key, value = secret.split("=", 1)
            result[key.strip()] = value
        elif os.path.isfile(secret):
            # .env file format
            with open(secret) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        result[key.strip()] = value.strip()

    return result


@cli.command()
@click.argument("agent_name")
@click.argument("task", required=False)
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(),
    help="Read task from file (use '-' for stdin)",
)
@click.option(
    "--output",
    "-o",
    "output_format",
    type=click.Choice(["raw", "json", "markdown"]),
    default="raw",
    help="Output format",
)
@click.option(
    "--secrets",
    "-s",
    multiple=True,
    help="Secrets as KEY=VALUE or path to .env file",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Only output the result, no status messages",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show tool calls as they happen",
)
@click.option(
    "--no-stream",
    "no_stream",
    is_flag=True,
    help="Disable streaming (stream is on by default)",
)
def run(
    agent_name: str,
    task: str | None,
    input_file: str | None,
    output_format: str,
    secrets: tuple[str, ...],
    quiet: bool,
    verbose: bool,
    no_stream: bool,
):
    """
    Run an agent in execution mode (non-interactive).

    AGENT_NAME is the agent to run.
    TASK is the task description or JSON input (optional if using --input or stdin).

    \b
    Examples:
        supyagent run summarizer "Summarize this text..."
        supyagent run summarizer --input document.txt
        supyagent run summarizer --input document.txt --output json
        echo "text" | supyagent run summarizer
        supyagent run api-caller '{"endpoint": "/users"}' --secrets API_KEY=xxx
    """
    # Load global config (API keys) into environment
    load_config()

    # Load agent config
    try:
        config = load_agent_config(agent_name)
    except AgentNotFoundError as e:
        console_err.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Warn if using interactive agent in execution mode
    if config.type != "execution" and not quiet:
        console_err.print(
            f"[yellow]Note:[/yellow] '{agent_name}' is an interactive agent. "
            "Consider using 'chat' for interactive use."
        )

    # Parse secrets
    secrets_dict = parse_secrets(secrets)

    # Get task content
    task_content: str | dict[str, Any]

    if input_file:
        if input_file == "-":
            task_content = sys.stdin.read().strip()
        else:
            input_path = Path(input_file)
            if not input_path.exists():
                console_err.print(f"[red]Error:[/red] File not found: {input_file}")
                sys.exit(1)
            task_content = input_path.read_text().strip()
    elif task:
        # Try to parse as JSON, otherwise use as string
        try:
            task_content = json.loads(task)
        except json.JSONDecodeError:
            task_content = task
    else:
        # Check if there's stdin input
        if not sys.stdin.isatty():
            task_content = sys.stdin.read().strip()
        else:
            console_err.print(
                "[red]Error:[/red] No task provided. "
                "Use positional argument, --input, or pipe to stdin."
            )
            sys.exit(1)

    if not task_content:
        console_err.print("[red]Error:[/red] Empty task")
        sys.exit(1)

    # Run the agent
    runner = ExecutionRunner(config)

    if not quiet:
        console_err.print(f"[dim]Running {agent_name}...[/dim]")

    # State for tracking reasoning across callbacks
    progress_state = {"in_reasoning": False}

    # Progress callback for streaming output and tool display
    def on_progress(event_type: str, data: dict):
        if event_type == "tool_start" and not quiet:
            # End reasoning if we were in it
            if progress_state["in_reasoning"]:
                click.echo("")  # End reasoning line
                progress_state["in_reasoning"] = False
            tool_name = data.get("name", "unknown")
            console_err.print(f"[cyan]⚡ {tool_name}[/cyan]")
            # Always show tool inputs
            if data.get("arguments"):
                try:
                    args = json.loads(data["arguments"])
                    # Format args compactly on one line if simple
                    if len(args) <= 2 and all(
                        isinstance(v, (str, int, bool)) for v in args.values()
                    ):
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                        console_err.print(f"[dim]   └─ {args_str}[/dim]")
                    else:
                        args_str = json.dumps(args, indent=2)
                        for line in args_str.split("\n"):
                            console_err.print(f"[dim]   {line}[/dim]")
                except json.JSONDecodeError:
                    pass
        elif event_type == "tool_end" and not quiet:
            tool_name = data.get("name", "unknown")
            result_data = data.get("result", {})
            ok = result_data.get("ok", False)
            if ok:
                console_err.print("[green]   ✓ done[/green]")
            else:
                error = result_data.get("error", "unknown error")
                console_err.print(f"[red]   ✗ {error}[/red]")
        elif event_type == "reasoning" and verbose:
            # Show LLM reasoning/thinking if available
            if not progress_state["in_reasoning"]:
                # Start of reasoning - show emoji
                console_err.print("[magenta dim]💭 [/magenta dim]", end="")
                progress_state["in_reasoning"] = True
            # Stream the reasoning content
            console_err.print(
                f"[magenta dim]{data.get('content', '')}[/magenta dim]", end=""
            )
        elif event_type == "streaming" and not no_stream:
            # End reasoning if we were in it
            if progress_state["in_reasoning"]:
                click.echo("")  # End reasoning line
                progress_state["in_reasoning"] = False
            # Print streamed content directly
            click.echo(data.get("content", ""), nl=False)

    # Determine if we should use streaming
    use_stream = not no_stream and output_format == "raw"

    result = runner.run(
        task_content,
        secrets=secrets_dict,
        output_format=output_format,
        on_progress=on_progress if (verbose or use_stream) else None,
        stream=use_stream,
    )

    # Output result
    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    elif result["ok"]:
        # If we streamed, add a newline; otherwise print the result
        if use_stream:
            click.echo("")  # Final newline after streamed content
        else:
            click.echo(result["data"])
    else:
        console_err.print(f"[red]Error:[/red] {result['error']}")
        sys.exit(1)


@cli.command("exec")
@click.argument("agent_name")
@click.option("--task", "-t", required=True, help="Task for the agent to perform")
@click.option("--context", "-c", default="{}", help="JSON context from parent agent")
@click.option(
    "--output",
    "-o",
    "output_fmt",
    type=click.Choice(["json", "text"]),
    default="json",
)
@click.option(
    "--timeout", type=float, default=300, help="Max execution time in seconds"
)
@click.option(
    "--depth", type=int, default=0, help="Delegation depth (set by parent agent)"
)
def exec_agent(
    agent_name: str,
    task: str,
    context: str,
    output_fmt: str,
    timeout: float,
    depth: int,
):
    """
    Execute an agent as a subprocess (used internally for delegation).

    This command is primarily used by the ProcessSupervisor when a parent
    agent delegates to a child agent. It runs the agent and returns JSON output.

    \b
    Examples:
        supyagent exec researcher --task "Find papers on AI"
        supyagent exec summarizer --task "Summarize this text" --output json
    """
    # Enforce delegation depth limit
    from supyagent.core.registry import AgentRegistry

    if depth > AgentRegistry.MAX_DEPTH:
        error_msg = (
            f"Maximum delegation depth ({AgentRegistry.MAX_DEPTH}) exceeded "
            f"(depth={depth}). Cannot delegate further."
        )
        if output_fmt == "json":
            click.echo(json.dumps({"ok": False, "error": error_msg}))
        else:
            console_err.print(f"[red]Error:[/red] {error_msg}")
        sys.exit(1)

    # Set depth in environment so child agents can read it
    os.environ["SUPYAGENT_DELEGATION_DEPTH"] = str(depth)

    # Load global config (API keys) into environment
    load_config()

    # Parse context
    try:
        context_dict = json.loads(context)
    except json.JSONDecodeError:
        if output_fmt == "json":
            click.echo(json.dumps({"ok": False, "error": "Invalid context JSON"}))
        else:
            console_err.print("[red]Error:[/red] Invalid context JSON")
        sys.exit(1)

    # Load agent config
    try:
        config = load_agent_config(agent_name)
    except AgentNotFoundError:
        if output_fmt == "json":
            click.echo(
                json.dumps({"ok": False, "error": f"Agent '{agent_name}' not found"})
            )
        else:
            console_err.print(f"[red]Error:[/red] Agent '{agent_name}' not found")
        sys.exit(1)

    # Build full task with context
    full_task = _build_task_with_context(task, context_dict)

    # Run the agent
    try:
        if config.type == "execution":
            runner = ExecutionRunner(config)
            result = runner.run(full_task, output_format="json")
        else:
            agent = Agent(config)
            response = agent.send_message(full_task)
            result = {"ok": True, "data": response}

        if output_fmt == "json":
            click.echo(json.dumps(result))
        else:
            if result.get("ok"):
                click.echo(result.get("data", ""))
            else:
                console_err.print(
                    f"[red]Error:[/red] {result.get('error', 'Unknown error')}"
                )
                sys.exit(1)

    except Exception as e:
        if output_fmt == "json":
            click.echo(json.dumps({"ok": False, "error": str(e)}))
        else:
            console_err.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def _build_task_with_context(task: str, context: dict) -> str:
    """Build task string with context from parent agent."""
    parts = []

    if context.get("parent_agent"):
        parts.append(f"You are being called by the '{context['parent_agent']}' agent.")

    if context.get("parent_task"):
        parts.append(f"Parent's current task: {context['parent_task']}")

    if context.get("conversation_summary"):
        parts.append(f"\nConversation context:\n{context['conversation_summary']}")

    if context.get("relevant_facts"):
        parts.append("\nRelevant information:")
        for fact in context["relevant_facts"]:
            parts.append(f"- {fact}")

    if parts:
        parts.append(f"\n---\n\nYour task:\n{task}")
        return "\n".join(parts)

    return task


@cli.command()
@click.argument("agent_name")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(),
    help="Output file (default: stdout)",
)
@click.option(
    "--format",
    "-f",
    "input_format",
    type=click.Choice(["jsonl", "csv"]),
    default="jsonl",
    help="Input file format",
)
@click.option(
    "--secrets",
    "-s",
    multiple=True,
    help="Secrets as KEY=VALUE or path to .env file",
)
def batch(
    agent_name: str,
    input_file: str,
    output_file: str | None,
    input_format: str,
    secrets: tuple[str, ...],
):
    """
    Run an agent on multiple inputs from a file.

    \b
    Input formats:
        - jsonl: One JSON object per line
        - csv: CSV with headers, each row becomes a dict

    \b
    Examples:
        supyagent batch summarizer inputs.jsonl
        supyagent batch summarizer inputs.jsonl --output results.jsonl
        supyagent batch summarizer data.csv --format csv
    """
    # Load global config (API keys) into environment
    load_config()

    # Load agent config
    try:
        config = load_agent_config(agent_name)
    except AgentNotFoundError as e:
        console_err.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Parse secrets
    secrets_dict = parse_secrets(secrets)

    # Load inputs
    inputs: list[dict[str, Any] | str] = []

    if input_format == "jsonl":
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        inputs.append(json.loads(line))
                    except json.JSONDecodeError:
                        inputs.append(line)
    elif input_format == "csv":
        import csv

        with open(input_file) as f:
            reader = csv.DictReader(f)
            inputs = list(reader)

    if not inputs:
        console.print("[yellow]No inputs found in file[/yellow]")
        return

    # Process
    runner = ExecutionRunner(config)
    results: list[dict[str, Any]] = []
    total = len(inputs)

    for idx, item in enumerate(inputs, 1):
        # Show task separator on stderr
        task_preview = str(item.get("task", item) if isinstance(item, dict) else item)
        if len(task_preview) > 60:
            task_preview = task_preview[:57] + "..."
        console_err.print(f"[cyan]── Task {idx}/{total}: {task_preview} ──[/cyan]")

        result = runner.run(item, secrets=secrets_dict, output_format="json")
        results.append(result)

        status = "[green]✓[/green]" if result["ok"] else "[red]✗[/red]"
        console_err.print(f"  {status} done")

    # Count successes/failures
    successes = sum(1 for r in results if r["ok"])
    failures = len(results) - successes

    # Output
    output_content = "\n".join(json.dumps(r) for r in results)

    if output_file:
        with open(output_file, "w") as f:
            f.write(output_content + "\n")
        console_err.print(
            f"\n[green]✓[/green] Processed {len(results)} items "
            f"({successes} succeeded, {failures} failed)"
        )
        console_err.print(f"  Results written to [cyan]{output_file}[/cyan]")
    else:
        click.echo(output_content)


@cli.group(invoke_without_command=True)
@click.pass_context
def agents(ctx):
    """List agent instances or inspect agent configuration."""
    if ctx.invoked_subcommand is None:
        # Default: list instances (backward compatible)
        _agents_list()


def _agents_list():
    """List all registered agent instances."""
    registry = AgentRegistry()
    instances = registry.list_all()

    if not instances:
        console.print("[dim]No active agent instances[/dim]")
        return

    table = Table(title="Agent Instances")
    table.add_column("ID", style="cyan")
    table.add_column("Agent")
    table.add_column("Status")
    table.add_column("Parent")
    table.add_column("Created")

    for inst in instances:
        status_style = {
            "active": "green",
            "completed": "dim",
            "failed": "red",
        }.get(inst.status, "")

        parent = inst.parent_id if inst.parent_id else "-"
        created = inst.created_at.strftime("%Y-%m-%d %H:%M")

        table.add_row(
            inst.instance_id,
            inst.name,
            f"[{status_style}]{inst.status}[/{status_style}]",
            parent,
            created,
        )

    console.print(table)


@agents.command("list")
def agents_list_cmd():
    """List all registered agent instances."""
    _agents_list()


@agents.command("inspect")
@click.argument("agent_name")
def agents_inspect(agent_name: str):
    """
    Show the full assembled system prompt for an agent.

    Displays the base prompt plus all injected sections (tool creation,
    resilience, cloud service, thinking guidelines).

    \b
    Examples:
        supyagent agents inspect myagent
    """
    try:
        config = load_agent_config(agent_name)
    except AgentNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    from supyagent.models.agent_config import get_full_system_prompt

    # Show with both states for comparison
    console.print(f"\n[bold]Inspecting agent:[/bold] [cyan]{agent_name}[/cyan]")
    console.print(f"[dim]Model: {config.model.provider} | Type: {config.type}[/dim]\n")

    # Show which sections are injected
    sections = []
    sections.append(("Base system prompt", True))
    sections.append(("Tool creation instructions", config.will_create_tools))
    sections.append(("Thinking guidelines", True))
    sections.append(("Resilience instructions", "when supypowers unavailable"))
    sections.append(("Cloud service instructions", "when service not connected"))

    console.print("[bold]Injected sections:[/bold]")
    for name, active in sections:
        if active is True:
            console.print(f"  [green]✓[/green] {name}")
        elif active is False:
            console.print(f"  [dim]○ {name} (disabled)[/dim]")
        else:
            console.print(f"  [yellow]~[/yellow] {name} ({active})")

    # Show the full prompt as it would be assembled
    full_prompt = get_full_system_prompt(
        config, supypowers_available=True, has_service=False
    )
    console.print(f"\n[bold]Full system prompt[/bold] ({len(full_prompt)} chars):")
    console.print(Panel(full_prompt, border_style="dim", expand=False))


@cli.command()
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--validate-only", is_flag=True, help="Only validate, don't run")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
def orchestrate(workflow_file: str, validate_only: bool, output: str):
    """
    Run a multi-agent workflow from a YAML file.

    The workflow defines a sequence of steps, each handled by a specific agent.
    Steps can depend on outputs from previous steps using {{variable}} syntax.

    \b
    Example workflow (workflows/daily-report.yaml):
        name: daily-report
        steps:
          - agent: email-checker
            task: "Summarize important emails from today"
            output: email_summary
          - agent: report-writer
            task: "Write a daily report including: {{email_summary}}"
            depends_on: [email_summary]

    \b
    Examples:
        supyagent orchestrate workflows/daily-report.yaml
        supyagent orchestrate workflows/pipeline.yaml --validate-only
        supyagent orchestrate workflows/pipeline.yaml --output json
    """
    from supyagent.core.orchestrator import Workflow, run_workflow

    workflow_path = Path(workflow_file)

    try:
        workflow = Workflow.from_file(workflow_path)
    except Exception as e:
        console.print(f"[red]Error loading workflow:[/red] {e}")
        sys.exit(1)

    # Validate
    issues = workflow.validate()
    if issues:
        console.print("[yellow]Workflow validation issues:[/yellow]")
        for issue in issues:
            console.print(f"  [yellow]![/yellow] {issue}")
        if validate_only or any("not found" in i for i in issues):
            sys.exit(1)

    if validate_only:
        console.print(
            f"[green]✓[/green] Workflow '{workflow.name}' is valid "
            f"({len(workflow.steps)} steps)"
        )
        return

    console.print(
        f"[bold]Running workflow:[/bold] {workflow.name} "
        f"({len(workflow.steps)} steps)\n"
    )

    def on_step_start(i: int, agent: str, task: str):
        console.print(
            f"[cyan]Step {i + 1}/{len(workflow.steps)}:[/cyan] "
            f"[bold]{agent}[/bold]"
        )
        console.print(f"  [dim]{task[:100]}{'...' if len(task) > 100 else ''}[/dim]")

    def on_step_end(_i: int, _agent: str, result: dict):
        if result.get("ok"):
            data_preview = str(result.get("data", ""))[:100]
            console.print(f"  [green]✓[/green] {data_preview}")
        else:
            console.print(f"  [red]✗[/red] {result.get('error', 'failed')}")
        console.print()

    result = run_workflow(
        workflow,
        on_step_start=on_step_start if output == "text" else None,
        on_step_end=on_step_end if output == "text" else None,
    )

    if output == "json":
        click.echo(json.dumps(result, indent=2))
    elif result.get("ok"):
        console.print("[green]Workflow completed successfully[/green]")
        outputs = result.get("outputs", {})
        if outputs:
            console.print("\n[bold]Outputs:[/bold]")
            for key, value in outputs.items():
                preview = str(value)[:200]
                console.print(f"  {key}: {preview}")
    else:
        console.print(f"[red]Workflow failed:[/red] {result.get('error')}")
        sys.exit(1)


@cli.command()
@click.argument("task")
@click.option(
    "--planner",
    "-p",
    default="planner",
    help="Planning agent to use (default: planner)",
)
@click.option(
    "--new",
    "-n",
    "new_session",
    is_flag=True,
    help="Start a new session",
)
def plan(task: str, planner: str, new_session: bool):
    """
    Run a task through the planning agent for orchestration.

    The planning agent will break down the task and delegate to
    specialist agents as needed.

    \b
    Examples:
        supyagent plan "Build a web scraper for news articles"
        supyagent plan "Create a Python library for data validation"
        supyagent plan "Write a blog post about AI" --planner my-planner
    """
    # Load global config (API keys) into environment
    load_config()

    # Load planner config
    try:
        config = load_agent_config(planner)
    except AgentNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    if not config.delegates:
        console.print(
            f"[yellow]Warning:[/yellow] Agent '{planner}' has no delegates configured. "
            "It will handle the task directly."
        )

    # Show plan info
    console.print(
        Panel(
            f"[bold]Planning Agent:[/bold] {planner}\n"
            f"[bold]Delegates:[/bold] {', '.join(config.delegates) if config.delegates else 'None'}\n"
            f"[bold]Task:[/bold] {task}",
            title="🎯 Plan Execution",
            border_style="blue",
        )
    )
    console.print()

    # Create agent with registry for tracking
    registry = AgentRegistry()
    agent = Agent(config, registry=registry)

    # Execute the task
    try:
        response = agent.send_message(task)
        console.print(Markdown(response))
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

    # Show summary of agent activity
    children = registry.list_children(agent.instance_id) if agent.instance_id else []
    if children:
        console.print()
        console.print(
            Panel(
                "\n".join(f"• {c.name} [{c.status}]" for c in children),
                title="Delegated Agents",
                border_style="dim",
            )
        )


@cli.command()
@click.option(
    "--max-age",
    type=float,
    default=24,
    help="Prune active entries older than N hours (default: 24)",
)
@click.option(
    "--sessions",
    "-s",
    "clean_sessions",
    is_flag=True,
    help="Also delete empty sessions (< 2 messages)",
)
def cleanup(max_age: float, clean_sessions: bool):
    """Clean up stale registry entries and optionally empty sessions."""
    total_cleaned = 0

    # 1. Prune completed/failed instances
    registry = AgentRegistry()
    count = registry.cleanup_completed()
    if count:
        console.print(f"[green]✓[/green] Removed {count} completed/failed instance(s)")
        total_cleaned += count

    # 2. Prune stale active instances (old PIDs)
    stale = registry.prune_stale(max_age_hours=max_age)
    if stale:
        console.print(f"[green]✓[/green] Pruned {stale} stale instance(s) (older than {max_age}h)")
        total_cleaned += stale

    # 3. Optionally clean empty sessions
    if clean_sessions:
        session_mgr = SessionManager()
        agents_dir = Path("agents")
        empty_count = 0
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.yaml"):
                agent_name = agent_file.stem
                for meta in session_mgr.list_sessions(agent_name):
                    loaded = session_mgr.load_session(agent_name, meta.session_id)
                    if loaded and len(loaded.messages) < 2:
                        session_mgr.delete_session(agent_name, meta.session_id)
                        empty_count += 1
        if empty_count:
            console.print(f"[green]✓[/green] Deleted {empty_count} empty session(s)")
            total_cleaned += empty_count

    if total_cleaned == 0:
        console.print("[dim]Nothing to clean up[/dim]")


# =============================================================================
# Telemetry Commands
# =============================================================================


@cli.command()
@click.argument("agent_name")
@click.option("--session", "-s", "session_id", help="Session ID (default: all sessions)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def telemetry(agent_name: str, session_id: str | None, as_json: bool):
    """
    View telemetry data for an agent.

    Shows tool call frequency, timing, error rates, and token usage.

    \b
    Examples:
        supyagent telemetry myagent
        supyagent telemetry myagent --session abc123
        supyagent telemetry myagent --json
    """
    telemetry_dir = Path(".supyagent/telemetry") / agent_name

    if not telemetry_dir.exists():
        console.print(f"[dim]No telemetry data for '{agent_name}'[/dim]")
        return

    # Collect data from sessions
    if session_id:
        files = [telemetry_dir / f"{session_id}.jsonl"]
        if not files[0].exists():
            console.print(f"[red]Session '{session_id}' telemetry not found[/red]")
            return
    else:
        files = sorted(telemetry_dir.glob("*.jsonl"))

    if not files:
        console.print(f"[dim]No telemetry data for '{agent_name}'[/dim]")
        return

    # Parse all events
    tool_stats: dict[str, dict[str, Any]] = {}
    total_llm_calls = 0
    total_tokens = 0
    total_errors = 0
    total_turns = 0

    for f in files:
        if not f.exists():
            continue
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    etype = event.get("type")

                    if etype == "tool_call":
                        tool = event.get("tool", "?")
                        if tool not in tool_stats:
                            tool_stats[tool] = {
                                "calls": 0, "ok": 0, "errors": 0,
                                "total_ms": 0, "is_service": False,
                            }
                        tool_stats[tool]["calls"] += 1
                        tool_stats[tool]["total_ms"] += event.get("duration_ms", 0)
                        tool_stats[tool]["is_service"] = event.get("is_service", False)
                        if event.get("ok"):
                            tool_stats[tool]["ok"] += 1
                        else:
                            tool_stats[tool]["errors"] += 1
                    elif etype == "llm_call":
                        total_llm_calls += 1
                        total_tokens += event.get("input_tokens", 0) + event.get("output_tokens", 0)
                    elif etype == "error":
                        total_errors += 1
                    elif etype == "turn":
                        total_turns += 1
        except (json.JSONDecodeError, OSError):
            continue

    if as_json:
        click.echo(json.dumps({
            "agent": agent_name,
            "turns": total_turns,
            "llm_calls": total_llm_calls,
            "total_tokens": total_tokens,
            "errors": total_errors,
            "tools": tool_stats,
        }, indent=2))
        return

    # Display summary
    console.print(f"\n[bold]Telemetry for {agent_name}[/bold]")
    console.print(f"  Sessions: {len(files)}")
    console.print(f"  Turns: {total_turns}")
    console.print(f"  LLM calls: {total_llm_calls}")
    if total_tokens > 0:
        console.print(f"  Total tokens: {total_tokens:,}")
    console.print(f"  Errors: {total_errors}")
    console.print()

    if tool_stats:
        table = Table(title="Tool Usage")
        table.add_column("Tool", style="cyan")
        table.add_column("Calls", justify="right")
        table.add_column("OK", justify="right", style="green")
        table.add_column("Errors", justify="right", style="red")
        table.add_column("Avg (ms)", justify="right", style="dim")
        table.add_column("Type", style="dim")

        for name, stats in sorted(tool_stats.items(), key=lambda x: x[1]["calls"], reverse=True):
            avg_ms = stats["total_ms"] / stats["calls"] if stats["calls"] > 0 else 0
            tool_type = "service" if stats["is_service"] else "local"
            table.add_row(
                name,
                str(stats["calls"]),
                str(stats["ok"]),
                str(stats["errors"]),
                f"{avg_ms:.0f}",
                tool_type,
            )

        console.print(table)


# =============================================================================
# Process Management Commands
# =============================================================================


@cli.group()
def process():
    """Manage background processes (tools and agents)."""
    pass


@process.command("list")
@click.option(
    "--all", "-a", "include_all", is_flag=True, help="Include completed processes"
)
def process_list(include_all: bool):
    """List running background processes."""
    from supyagent.core.supervisor import get_supervisor

    supervisor = get_supervisor()
    processes = supervisor.list_processes(include_completed=include_all)

    if not processes:
        console.print("[dim]No running processes[/dim]")
        return

    table = Table(title="Background Processes")
    table.add_column("ID", style="cyan")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("PID")
    table.add_column("Started")

    for proc in processes:
        status = proc.get("status", "unknown")
        status_style = {
            "running": "green",
            "backgrounded": "yellow",
            "completed": "blue",
            "failed": "red",
            "killed": "red",
        }.get(status, "white")

        started = proc.get("started_at", "")
        if started:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                started = dt.strftime("%H:%M:%S")
            except (ValueError, AttributeError):
                pass

        table.add_row(
            proc.get("process_id", "?"),
            proc.get("process_type", "?"),
            f"[{status_style}]{status}[/{status_style}]",
            str(proc.get("pid", "-")),
            started,
        )

    console.print(table)

    # Show metadata for each process
    for proc in processes:
        meta = proc.get("metadata", {})
        if meta:
            proc_id = proc.get("process_id", "?")
            if "agent_name" in meta:
                console.print(f"  [dim]{proc_id}:[/dim] Agent: {meta['agent_name']}")
            elif "script" in meta:
                console.print(
                    f"  [dim]{proc_id}:[/dim] Tool: {meta['script']}__{meta.get('func', '?')}"
                )


@process.command("show")
@click.argument("process_id")
def process_show(process_id: str):
    """Show details of a specific process."""
    from supyagent.core.supervisor import get_supervisor

    supervisor = get_supervisor()
    proc = supervisor.get_process(process_id)

    if not proc:
        console.print(f"[red]Process {process_id} not found[/red]")
        return

    console.print(f"\n[bold]Process ID:[/bold] {proc['process_id']}")
    console.print(f"[bold]Type:[/bold] {proc['process_type']}")
    console.print(f"[bold]Status:[/bold] {proc['status']}")
    console.print(f"[bold]PID:[/bold] {proc.get('pid', 'N/A')}")
    console.print(f"[bold]Started:[/bold] {proc.get('started_at', 'N/A')}")

    if proc.get("completed_at"):
        console.print(f"[bold]Completed:[/bold] {proc['completed_at']}")

    if proc.get("exit_code") is not None:
        console.print(f"[bold]Exit Code:[/bold] {proc['exit_code']}")

    if proc.get("log_file"):
        console.print(f"[bold]Log File:[/bold] {proc['log_file']}")

    # Show command (truncated)
    cmd = proc.get("cmd", [])
    if cmd:
        cmd_str = " ".join(cmd[:5])
        if len(cmd) > 5:
            cmd_str += " ..."
        console.print(f"[bold]Command:[/bold] {cmd_str}")

    # Show metadata
    meta = proc.get("metadata", {})
    if meta:
        console.print("\n[bold]Metadata:[/bold]")
        for key, value in meta.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            console.print(f"  {key}: {value}")


@process.command("output")
@click.argument("process_id")
@click.option("--tail", "-n", default=50, help="Number of lines to show")
def process_output(process_id: str, tail: int):
    """Show output from a background process."""
    import asyncio

    from supyagent.core.supervisor import get_supervisor

    supervisor = get_supervisor()
    result = asyncio.run(supervisor.get_output(process_id, tail=tail))

    if result["ok"]:
        data = result.get("data", {})
        if isinstance(data, dict):
            output = data.get("output", "") or data.get("stdout", "")
            if output:
                click.echo(output)
            stderr = data.get("stderr", "")
            if stderr:
                console.print("\n[bold red]STDERR:[/bold red]")
                click.echo(stderr)
        else:
            click.echo(data)
    else:
        console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")


@process.command("kill")
@click.argument("process_id")
@click.option("--force", "-f", is_flag=True, help="Force kill without confirmation")
def process_kill(process_id: str, force: bool):
    """Kill a running background process."""
    import asyncio

    from supyagent.core.supervisor import get_supervisor

    if not force:
        if not click.confirm(f"Kill process {process_id}?"):
            return

    supervisor = get_supervisor()
    result = asyncio.run(supervisor.kill(process_id))

    if result["ok"]:
        console.print(f"[green]✓[/green] Process {process_id} killed")
    else:
        console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")


@process.command("cleanup")
def process_cleanup():
    """Remove completed/failed processes from tracking."""
    import asyncio

    from supyagent.core.supervisor import get_supervisor

    supervisor = get_supervisor()
    count = asyncio.run(supervisor._cleanup_completed())

    if count == 0:
        console.print("[dim]No completed processes to clean up[/dim]")
    else:
        console.print(f"[green]✓[/green] Cleaned up {count} process(es)")

    # Also clean old log files
    log_count = asyncio.run(supervisor.cleanup_old_logs())
    if log_count > 0:
        console.print(f"[green]✓[/green] Removed {log_count} old log file(s)")


# =============================================================================
# Config Commands
# =============================================================================


@cli.group()
def config():
    """Manage API keys and global configuration."""
    pass


@config.command("set")
@click.argument("key_name", required=False)
@click.option(
    "--value",
    "-v",
    help="Set value directly (use with caution - visible in shell history)",
)
def config_set(key_name: str | None, value: str | None):
    """
    Set an API key.

    If KEY_NAME is not provided, shows an interactive menu of common keys.

    \b
    Examples:
        supyagent config set                    # Interactive menu
        supyagent config set OPENAI_API_KEY     # Set specific key
        supyagent config set MY_KEY -v "value"  # Set with value (not recommended)
    """
    config_mgr = ConfigManager()

    if value:
        if not key_name:
            console.print("[red]Error:[/red] KEY_NAME required when using --value")
            sys.exit(1)
        config_mgr.set(key_name, value)
        console.print(f"[green]✓[/green] Saved {key_name}")
    else:
        config_mgr.set_interactive(key_name)


@config.command("list")
def config_list():
    """List all configured API keys."""
    config_mgr = ConfigManager()
    config_mgr.show_status()


@config.command("delete")
@click.argument("key_name")
def config_delete(key_name: str):
    """Delete a stored API key."""
    config_mgr = ConfigManager()

    if config_mgr.delete(key_name):
        console.print(f"[green]✓[/green] Deleted {key_name}")
    else:
        console.print(f"[yellow]Key not found:[/yellow] {key_name}")


@config.command("import")
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--filter",
    "-f",
    "key_filter",
    help="Only import keys matching this prefix (e.g., 'OPENAI')",
)
def config_import(file_path: str, key_filter: str | None):
    """
    Import API keys from a .env file.

    The file should contain KEY=VALUE pairs, one per line.
    Lines starting with # are ignored.

    \b
    Examples:
        supyagent config import .env
        supyagent config import secrets.env --filter OPENAI
    """
    config_mgr = ConfigManager()

    try:
        # If filter is specified, we need custom handling
        if key_filter:
            import re
            from pathlib import Path

            path = Path(file_path)
            pattern = re.compile(r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.+)$")
            imported = 0

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    match = pattern.match(line)
                    if match:
                        name, value = match.groups()
                        if name.startswith(key_filter.upper()):
                            if (value.startswith('"') and value.endswith('"')) or (
                                value.startswith("'") and value.endswith("'")
                            ):
                                value = value[1:-1]
                            config_mgr.set(name, value)
                            console.print(f"  [green]✓[/green] {name}")
                            imported += 1
        else:
            imported = config_mgr.set_from_file(file_path)

        if imported == 0:
            console.print("[yellow]No keys found in file[/yellow]")
        else:
            console.print(f"\n[green]✓[/green] Imported {imported} key(s)")

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@config.command("export")
@click.argument("file_path", type=click.Path())
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file")
def config_export(file_path: str, force: bool):
    """
    Export stored API keys to a .env file.

    \b
    Example:
        supyagent config export backup.env
    """
    config_mgr = ConfigManager()
    path = Path(file_path)

    if path.exists() and not force:
        console.print(f"[red]Error:[/red] File exists: {file_path}")
        console.print("Use --force to overwrite")
        sys.exit(1)

    keys = config_mgr._load_keys()

    if not keys:
        console.print("[yellow]No keys to export[/yellow]")
        return

    with open(path, "w") as f:
        f.write("# Supyagent API Keys\n")
        f.write("# Generated export - keep this file secure!\n\n")
        for name, value in sorted(keys.items()):
            f.write(f"{name}={value}\n")

    # Set restrictive permissions
    try:
        path.chmod(0o600)
    except OSError:
        pass

    console.print(f"[green]✓[/green] Exported {len(keys)} key(s) to {file_path}")


# =============================================================================
# Service Tool Commands (direct CLI access to cloud integrations)
# =============================================================================


@cli.group("service")
def service_group():
    """
    Use cloud integration tools directly from the CLI.

    Execute service tools (Gmail, Slack, GitHub, etc.) without running
    a full agent conversation loop.

    \b
    Examples:
        supyagent service tools                           # List available tools
        supyagent service tools --provider google         # Filter by provider
        supyagent service run gmail:send_message '{...}'  # Execute a tool
    """
    pass


@service_group.command("tools")
@click.option(
    "--provider",
    "-p",
    default=None,
    help="Filter tools by provider (e.g., google, slack, github)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def service_tools(provider: str | None, as_json: bool):
    """
    List available service tools.

    Shows all tools from connected integrations. Use --provider to filter
    by a specific provider.

    \b
    Examples:
        supyagent service tools
        supyagent service tools --provider google
        supyagent service tools --json
    """
    from supyagent.core.service import get_service_client

    client = get_service_client()
    if not client:
        console.print("[yellow]Not connected to service.[/yellow]")
        console.print("Run [cyan]supyagent connect[/cyan] to authenticate.")
        sys.exit(1)

    tools = client.discover_tools()
    client.close()

    if not tools:
        console.print("[dim]No tools available. Connect integrations on the dashboard.[/dim]")
        return

    # Filter by provider if specified
    if provider:
        tools = [
            t for t in tools
            if t.get("metadata", {}).get("provider", "").lower() == provider.lower()
        ]
        if not tools:
            console.print(f"[yellow]No tools found for provider '{provider}'.[/yellow]")
            return

    if as_json:
        # Output structured JSON for machine consumption
        output = []
        for tool in tools:
            meta = tool.get("metadata", {})
            func = tool.get("function", {})
            output.append({
                "name": f"{meta.get('provider', '?')}:{func.get('name', '?')}",
                "description": func.get("description", ""),
                "provider": meta.get("provider"),
                "service": meta.get("service"),
                "method": meta.get("method"),
                "parameters": func.get("parameters", {}),
            })
        click.echo(json.dumps(output, indent=2))
        return

    # Group by provider/service for table display
    table = Table(title="Service Tools")
    table.add_column("Tool", style="cyan")
    table.add_column("Description")
    table.add_column("Provider", style="dim")

    for tool in tools:
        meta = tool.get("metadata", {})
        func = tool.get("function", {})
        tool_name = func.get("name", "?")
        desc = func.get("description", "")
        # Truncate long descriptions
        if len(desc) > 60:
            desc = desc[:57] + "..."
        prov = meta.get("provider", "?")
        table.add_row(tool_name, desc, prov)

    console.print(table)
    console.print(f"\n[dim]{len(tools)} tools available[/dim]")
    console.print(
        "[dim]Run a tool:[/dim] supyagent service run <tool_name> '<json_args>'"
    )


@service_group.command("inbox")
@click.option(
    "--status",
    "-s",
    default=None,
    type=click.Choice(["unread", "read", "archived"], case_sensitive=False),
    help="Filter by status (default: all)",
)
@click.option(
    "--provider",
    "-p",
    default=None,
    help="Filter by provider (e.g., github, slack)",
)
@click.option("--limit", "-l", default=20, type=int, help="Max events to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def service_inbox(
    status: str | None, provider: str | None, limit: int, as_json: bool
):
    """
    View your AI inbox — incoming events from connected integrations.

    Shows recent webhook events (GitHub PRs, Slack messages, SMS, etc.)
    that your agents can act on.

    \b
    Examples:
        supyagent service inbox                        # List recent events
        supyagent service inbox --status unread         # Unread only
        supyagent service inbox --provider github       # GitHub events only
        supyagent service inbox --json                  # Machine-readable output
    """
    from supyagent.core.service import get_service_client

    client = get_service_client()
    if not client:
        console.print("[yellow]Not connected to service.[/yellow]")
        console.print("Run [cyan]supyagent connect[/cyan] to authenticate.")
        sys.exit(1)

    data = client.inbox_list(status=status, provider=provider, limit=limit)
    client.close()

    events = data.get("events", [])
    total = data.get("total", 0)

    if data.get("error"):
        console.print(f"[red]Error:[/red] {data['error']}")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(data, indent=2))
        return

    if not events:
        console.print("[grey62]No events in your inbox.[/grey62]")
        if not status:
            console.print(
                "[grey62]Configure webhooks on the dashboard to start receiving events.[/grey62]"
            )
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Status", width=8)
    table.add_column("Provider", style="cyan", width=10)
    table.add_column("Type", width=20)
    table.add_column("Summary")
    table.add_column("Time", style="bright_black", justify="right", width=10)

    for event in events:
        st = event.get("status", "?")
        st_style = (
            "[blue]unread[/blue]"
            if st == "unread"
            else "[bright_black]read[/bright_black]"
            if st == "read"
            else "[bright_black]archived[/bright_black]"
        )
        prov = event.get("provider", "?")
        etype = event.get("event_type", "?")
        summary = event.get("summary", "")
        if len(summary) > 60:
            summary = summary[:57] + "..."
        received = event.get("received_at", "")
        time_str = _format_relative_time(received) if received else ""

        table.add_row(st_style, prov, etype, summary, time_str)

    console.print(table)
    console.print(f"\n[grey62]{total} total event{'s' if total != 1 else ''}[/grey62]")


@service_group.command("inbox:get")
@click.argument("event_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def service_inbox_get(event_id: str, as_json: bool):
    """
    View a single inbox event with its full payload.

    The event is automatically marked as read when viewed.

    \b
    Examples:
        supyagent service inbox:get <event-id>
        supyagent service inbox:get <event-id> --json
    """
    from supyagent.core.service import get_service_client

    client = get_service_client()
    if not client:
        console.print("[yellow]Not connected to service.[/yellow]")
        console.print("Run [cyan]supyagent connect[/cyan] to authenticate.")
        sys.exit(1)

    event = client.inbox_get(event_id)
    client.close()

    if not event:
        console.print(f"[red]Error:[/red] Event not found: {event_id}")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(event, indent=2))
        return

    console.print(f"[bold]{event.get('summary', 'No summary')}[/bold]")
    console.print()
    console.print(f"  Provider:  [cyan]{event.get('provider', '?')}[/cyan]")
    console.print(f"  Type:      {event.get('event_type', '?')}")
    console.print(f"  Status:    {event.get('status', '?')}")
    console.print(f"  Received:  {event.get('received_at', '?')}")
    if event.get("provider_event_id"):
        console.print(f"  Event ID:  [bright_black]{event['provider_event_id']}[/bright_black]")
    console.print()
    console.print("[bold]Payload:[/bold]")
    payload_str = json.dumps(event.get("payload", {}), indent=2)
    console.print(f"[grey62]{payload_str}[/grey62]")


@service_group.command("inbox:archive")
@click.argument("event_id", required=False, default=None)
@click.option("--all", "archive_all", is_flag=True, help="Archive all events")
@click.option("--provider", "-p", default=None, help="When using --all, archive only this provider")
def service_inbox_archive(
    event_id: str | None, archive_all: bool, provider: str | None
):
    """
    Archive inbox events.

    Archive a single event by ID, or use --all to archive everything.

    \b
    Examples:
        supyagent service inbox:archive <event-id>
        supyagent service inbox:archive --all
        supyagent service inbox:archive --all --provider github
    """
    from supyagent.core.service import get_service_client

    if not event_id and not archive_all:
        console.print("[red]Error:[/red] Provide an event ID or use --all.")
        sys.exit(1)

    client = get_service_client()
    if not client:
        console.print("[yellow]Not connected to service.[/yellow]")
        console.print("Run [cyan]supyagent connect[/cyan] to authenticate.")
        sys.exit(1)

    if archive_all:
        count = client.inbox_archive_all(provider=provider)
        client.close()
        if count > 0:
            console.print(f"[green]Archived {count} event{'s' if count != 1 else ''}.[/green]")
        else:
            console.print("[grey62]No events to archive.[/grey62]")
    else:
        ok = client.inbox_archive(event_id)
        client.close()
        if ok:
            console.print("[green]Event archived.[/green]")
        else:
            console.print(f"[red]Error:[/red] Could not archive event {event_id}.")
            sys.exit(1)


def _format_relative_time(iso_str: str) -> str:
    """Format an ISO timestamp as a relative time string."""
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = (now - dt).total_seconds()
        minutes = int(diff // 60)
        hours = int(diff // 3600)
        days = int(diff // 86400)

        if minutes < 1:
            return "now"
        if minutes < 60:
            return f"{minutes}m"
        if hours < 24:
            return f"{hours}h"
        if days < 7:
            return f"{days}d"
        return dt.strftime("%b %d")
    except (ValueError, TypeError):
        return ""


@service_group.command("run")
@click.argument("tool_spec")
@click.argument("args_json", required=False, default="{}")
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(),
    help="Read arguments from JSON file (use '-' for stdin)",
)
def service_run(tool_spec: str, args_json: str, input_file: str | None):
    """
    Execute a service tool directly.

    TOOL_SPEC is the tool name (e.g., gmail_send_message or gmail:send_message).
    ARGS_JSON is the JSON arguments (optional if using --input).

    \b
    Examples:
        supyagent service run gmail_list_messages '{"max_results": 5}'
        supyagent service run slack:send_message '{"channel": "#general", "text": "Hello"}'
        supyagent service run github_list_repos
        echo '{"query": "test"}' | supyagent service run gmail_search_messages --input -
    """
    from supyagent.core.service import get_service_client

    client = get_service_client()
    if not client:
        console.print("[yellow]Not connected to service.[/yellow]")
        console.print("Run [cyan]supyagent connect[/cyan] to authenticate.")
        sys.exit(1)

    # Parse arguments
    if input_file:
        if input_file == "-":
            raw = sys.stdin.read().strip()
        else:
            input_path = Path(input_file)
            if not input_path.exists():
                console.print(f"[red]Error:[/red] File not found: {input_file}")
                sys.exit(1)
            raw = input_path.read_text().strip()
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON in input: {e}")
            sys.exit(1)
    else:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON arguments: {e}")
            sys.exit(1)

    # Normalize tool spec: support both "gmail:send_message" and "gmail_send_message"
    tool_name = tool_spec.replace(":", "_")

    # Discover tools to find metadata
    console_err.print("[dim]Discovering tools...[/dim]")
    tools = client.discover_tools()

    if not tools:
        console.print("[red]Error:[/red] No tools available. Connect integrations on the dashboard.")
        client.close()
        sys.exit(1)

    # Find the matching tool
    matched_tool = None
    for tool in tools:
        func = tool.get("function", {})
        if func.get("name") == tool_name:
            matched_tool = tool
            break

    if not matched_tool:
        # Try partial matching (provider:action without service prefix)
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            # Match suffix: "send_message" matches "gmail_send_message"
            if name.endswith(f"_{tool_name}") or name.endswith(f"_{tool_spec.replace(':', '_')}"):
                matched_tool = tool
                break

    if not matched_tool:
        console.print(f"[red]Error:[/red] Tool '{tool_spec}' not found.")
        console.print()
        # Suggest similar tools
        available = [t.get("function", {}).get("name", "") for t in tools]
        suggestions = [n for n in available if tool_name.split("_")[0] in n]
        if suggestions:
            console.print("[dim]Did you mean:[/dim]")
            for s in suggestions[:5]:
                console.print(f"  - {s}")
        else:
            console.print(f"[dim]Available tools ({len(available)}):[/dim]")
            for n in available[:10]:
                console.print(f"  - {n}")
            if len(available) > 10:
                console.print(f"  [dim]... and {len(available) - 10} more[/dim]")
        client.close()
        sys.exit(1)

    metadata = matched_tool.get("metadata", {})
    func_name = matched_tool.get("function", {}).get("name", tool_spec)

    console_err.print(
        f"[dim]Executing {func_name} "
        f"({metadata.get('method', '?')} {metadata.get('path', '?')})...[/dim]"
    )

    result = client.execute_tool(func_name, args, metadata)
    client.close()

    # Output result as JSON
    click.echo(json.dumps(result, indent=2))

    if not result.get("ok"):
        sys.exit(1)


# =============================================================================
# Service Connection Commands
# =============================================================================


@cli.command()
@click.option(
    "--url",
    default=None,
    help="Service URL (default: https://app.supyagent.com)",
)
def connect(url: str | None):
    """
    Connect to supyagent service for third-party integrations.

    Authenticates via device authorization flow: you'll receive a code
    to enter in your browser, then the CLI will receive an API key.

    \b
    Examples:
        supyagent connect
        supyagent connect --url https://custom.supyagent.com
    """
    import threading
    import time
    import webbrowser

    from rich.rule import Rule

    from supyagent.core.service import (
        DEFAULT_SERVICE_URL,
        ServiceClient,
        poll_for_token,
        request_device_code,
        store_service_credentials,
    )

    base_url = url or DEFAULT_SERVICE_URL

    # Step 1: Request device code
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Requesting device code...", total=None)
            device_data = request_device_code(base_url)
    except Exception as e:
        console.print(f"[red]Error:[/red] Could not reach service at {base_url}: {e}")
        sys.exit(1)

    user_code = device_data["user_code"]
    device_code = device_data["device_code"]
    verification_uri = device_data.get("verification_uri") or f"{base_url}/device"
    # Ensure verification URI uses the correct host (server may return localhost in dev)
    if "localhost" in verification_uri and "localhost" not in base_url:
        verification_uri = f"{base_url}/device"
    expires_in = device_data.get("expires_in", 900)
    interval = device_data.get("interval", 5)

    # Step 2: Show code and open browser
    console.print()
    console.print(Rule("Device Authorization", style="blue"))
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
    console.print()
    console.print(
        f"  Visit [link={verification_uri}][cyan]{verification_uri}[/cyan][/link] "
        "and enter the code above to authorize this device.",
    )
    console.print()

    try:
        webbrowser.open(verification_uri)
        console.print("[grey62]Browser opened automatically.[/grey62]")
    except Exception:
        console.print(
            f"[grey62]Open this URL in your browser: {verification_uri}[/grey62]"
        )

    # Step 3: Poll for approval with countdown
    console.print()
    result_container: dict[str, Any] = {"key": None, "error": None}

    def _poll() -> None:
        try:
            result_container["key"] = poll_for_token(
                base_url=base_url,
                device_code=device_code,
                interval=interval,
                expires_in=expires_in,
            )
        except Exception as exc:
            result_container["error"] = exc

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[countdown]}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Waiting for authorization...", total=None, countdown=""
            )
            start = time.time()
            while poll_thread.is_alive():
                remaining = max(0, expires_in - (time.time() - start))
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                progress.update(
                    task,
                    countdown=f"[bright_black]({minutes}:{seconds:02d} remaining)[/bright_black]",
                )
                time.sleep(0.5)

        poll_thread.join()
        if result_container["error"]:
            raise result_container["error"]
        api_key = result_container["key"]
    except TimeoutError:
        console.print("[red]Error:[/red] Device code expired. Please try again.")
        sys.exit(1)
    except PermissionError:
        console.print("[yellow]Authorization denied.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Step 4: Store credentials and show success
    store_service_credentials(api_key, base_url if url else None)
    console.print()
    console.print("[bold green]✓ Connected![/bold green]")
    console.print()

    # Step 5: Discover and show available tools
    try:
        client = ServiceClient(api_key=api_key, base_url=base_url)
        tools = client.discover_tools()
        client.close()

        if tools:
            # Group by provider
            providers: dict[str, list[str]] = {}
            for tool in tools:
                meta = tool.get("metadata", {})
                provider = meta.get("provider", "unknown")
                service = meta.get("service", "")
                if provider not in providers:
                    providers[provider] = []
                if service and service not in providers[provider]:
                    providers[provider].append(service)

            tool_table = Table(
                show_header=True, header_style="bold", box=None, padding=(0, 2)
            )
            tool_table.add_column("Provider", style="cyan")
            tool_table.add_column("Services")
            for provider, services in sorted(providers.items()):
                tool_table.add_row(
                    provider, ", ".join(services) if services else "-"
                )
            console.print(tool_table)
            console.print(
                f"[bright_black]{len(tools)} tools available[/bright_black]"
            )
        else:
            console.print(
                "[grey62]No integrations connected yet. "
                "Visit the dashboard to connect services.[/grey62]"
            )
    except Exception:
        console.print(
            "[grey62]Connected. Run 'supyagent status' to see available tools.[/grey62]"
        )


@cli.command()
def disconnect():
    """
    Disconnect from supyagent service.

    Removes the stored API key and service URL.
    """
    from supyagent.core.service import clear_service_credentials

    removed = clear_service_credentials()
    if removed:
        console.print("[green]Disconnected from service.[/green]")
    else:
        console.print("[grey62]Not currently connected to a service.[/grey62]")


@cli.command()
def status():
    """
    Show connection status and available service integrations.
    """
    from supyagent.core.service import (
        DEFAULT_SERVICE_URL,
        SERVICE_API_KEY,
        SERVICE_URL,
        ServiceClient,
    )

    config_mgr = ConfigManager()
    api_key = config_mgr.get(SERVICE_API_KEY)

    if not api_key:
        console.print("[yellow]Not connected to service.[/yellow]")
        console.print()
        console.print("Run [cyan]supyagent connect[/cyan] to authenticate.")
        return

    base_url = config_mgr.get(SERVICE_URL) or DEFAULT_SERVICE_URL
    console.print(f"[green]Connected[/green] to {base_url}")

    # Health check
    client = ServiceClient(api_key=api_key, base_url=base_url)
    reachable = client.health_check()

    if not reachable:
        console.print(f"[yellow]Service is not reachable at {base_url}[/yellow]")
        client.close()
        return

    console.print("[grey62]Service is reachable[/grey62]")
    console.print()

    # Discover tools
    tools = client.discover_tools()
    client.close()

    if not tools:
        console.print("[grey62]No integrations connected. Visit the dashboard to add services.[/grey62]")
        return

    # Group tools by provider/service
    providers: dict[str, dict[str, list[str]]] = {}
    for tool in tools:
        meta = tool.get("metadata", {})
        provider = meta.get("provider", "unknown")
        service = meta.get("service", "general")
        tool_name = tool.get("function", {}).get("name", "?")

        if provider not in providers:
            providers[provider] = {}
        if service not in providers[provider]:
            providers[provider][service] = []
        providers[provider][service].append(tool_name)

    table = Table(title="Available Service Tools")
    table.add_column("Provider", style="cyan")
    table.add_column("Service")
    table.add_column("Tools", style="grey62", justify="right")

    for provider in sorted(providers):
        for service in sorted(providers[provider]):
            tool_count = len(providers[provider][service])
            table.add_row(provider, service, str(tool_count))

    console.print(table)
    console.print(f"\n[bright_black]{len(tools)} tools total[/bright_black]")


@cli.command()
@click.option("--status", "-s", type=click.Choice(["unread", "read", "archived"]), default=None, help="Filter by status")
@click.option("--provider", "-p", default=None, help="Filter by provider (e.g. github, slack)")
@click.option("--limit", "-n", default=20, type=int, help="Number of events (default: 20)")
@click.option("--event-id", "-i", default=None, help="Get a specific event by ID")
@click.option("--archive", "-a", default=None, help="Archive a specific event by ID")
@click.option("--archive-all", is_flag=True, help="Archive all events")
def inbox(
    status: str | None,
    provider: str | None,
    limit: int,
    event_id: str | None,
    archive: str | None,
    archive_all: bool,
):
    """
    View and manage your AI Inbox.

    Shows webhook events from connected integrations (GitHub, Slack, Telegram, etc.).

    \b
    Examples:
        supyagent inbox                        # List unread events
        supyagent inbox -s read                # List read events
        supyagent inbox -p github              # GitHub events only
        supyagent inbox -i EVENT_ID            # View a specific event
        supyagent inbox -a EVENT_ID            # Archive an event
        supyagent inbox --archive-all          # Archive all events
    """
    from supyagent.core.service import get_service_client

    client = get_service_client()
    if not client:
        console.print("[yellow]Not connected to service.[/yellow]")
        console.print("Run [cyan]supyagent connect[/cyan] to authenticate.")
        return

    # Archive a specific event
    if archive:
        success = client.inbox_archive(archive)
        if success:
            console.print(f"[green]Archived[/green] event {archive[:12]}...")
        else:
            console.print(f"[red]Failed to archive[/red] event {archive}")
        return

    # Archive all
    if archive_all:
        count = client.inbox_archive_all(provider=provider)
        console.print(f"[green]Archived {count} event(s)[/green]")
        return

    # Get a specific event
    if event_id:
        event = client.inbox_get(event_id)
        if not event:
            console.print(f"[red]Event not found:[/red] {event_id}")
            return

        console.print(Panel(
            f"[bold]{event.get('provider', '?')}[/bold] / {event.get('event_type', '?')}\n"
            f"[grey62]{event.get('received_at', '')}[/grey62]\n\n"
            f"{event.get('summary', 'No summary')}\n\n"
            f"[grey62]Status: {event.get('status', '?')} | ID: {event.get('id', '?')}[/grey62]",
            title="Inbox Event",
            border_style="cyan",
        ))

        # Show payload
        payload = event.get("payload")
        if payload:
            console.print()
            console.print("[bright_black]Payload:[/bright_black]")
            console.print(json.dumps(payload, indent=2, default=str)[:2000])
        return

    # List events
    result = client.inbox_list(
        status=status or "unread",
        provider=provider,
        limit=limit,
    )

    events = result.get("events", [])
    total = result.get("total", 0)

    if not events:
        filter_desc = f" ({status or 'unread'})"
        if provider:
            filter_desc += f" from {provider}"
        console.print(f"[grey62]No events{filter_desc}.[/grey62]")
        return

    table = Table(title=f"Inbox ({total} total)")
    table.add_column("Provider", style="cyan", width=10)
    table.add_column("Type", width=22)
    table.add_column("Summary", ratio=1)
    table.add_column("When", style="grey62", width=18)
    table.add_column("ID", style="bright_black", width=12)

    for event in events:
        received = event.get("received_at", "")
        if received:
            # Show relative time or short date
            received = received[:16].replace("T", " ")

        table.add_row(
            event.get("provider", "?"),
            event.get("event_type", "?"),
            (event.get("summary", "") or "")[:60],
            received,
            (event.get("id", ""))[:12] + "...",
        )

    console.print(table)

    if result.get("has_more"):
        console.print(f"\n[bright_black]Showing {len(events)} of {total}. Use -n to show more.[/bright_black]")


@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
@click.option("--port", "-p", default=8000, type=int, help="Bind port (default: 8000)")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes")
@click.option(
    "--cors-origin",
    multiple=True,
    default=["*"],
    help="Allowed CORS origins (default: *)",
)
def serve(host: str, port: int, reload: bool, cors_origin: tuple[str, ...]):
    """
    Start the API server (Vercel AI SDK compatible).

    The server exposes a streaming chat endpoint compatible with the
    AI SDK useChat() hook, plus REST endpoints for agents, sessions,
    and tools.

    \b
    Examples:
        supyagent serve                           # localhost:8000
        supyagent serve --port 3001               # custom port
        supyagent serve --host 0.0.0.0            # all interfaces
        supyagent serve --cors-origin http://localhost:3000
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Error:[/red] Server dependencies not installed.\n\n"
            "Install them with:\n"
            "  pip install supyagent[serve]\n"
            "  # or: uv pip install fastapi uvicorn[standard]"
        )
        sys.exit(1)

    from supyagent.server.app import create_app

    app = create_app(cors_origins=list(cors_origin))

    console.print(f"[bold]supyagent server[/bold] starting on http://{host}:{port}")
    console.print(f"[dim]CORS origins: {', '.join(cors_origin)}[/dim]")
    console.print(f"[dim]Docs: http://{host}:{port}/docs[/dim]\n")

    uvicorn.run(
        app if not reload else "supyagent.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=reload,
    )


if __name__ == "__main__":
    cli()
