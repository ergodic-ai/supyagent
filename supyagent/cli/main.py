"""
CLI entry point for supyagent.
"""

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from typing import Any

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
        supyagent init            # Set up default tools
        supyagent config set      # Configure API keys
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
def init(tools_dir: str, force: bool):
    """
    Initialize supyagent in the current directory.

    This sets up:
    - Default tools in powers/ (shell commands, file operations)
    - agents/ directory for agent configurations

    \b
    Examples:
        supyagent init
        supyagent init --tools-dir my_tools
    """
    console.print("[bold]Initializing supyagent...[/bold]")
    console.print()

    # Create agents directory
    agents_dir = Path("agents")
    if not agents_dir.exists():
        agents_dir.mkdir(parents=True)
        console.print(f"  [green]✓[/green] Created {agents_dir}/")
    else:
        console.print(f"  [dim]○[/dim] {agents_dir}/ already exists")

    # Install default tools
    tools_path = Path(tools_dir)

    if force:
        # Remove and reinstall
        import shutil

        if tools_path.exists():
            shutil.rmtree(tools_path)

    if tools_path.exists() and any(tools_path.glob("*.py")):
        console.print(f"  [dim]○[/dim] {tools_dir}/ already has tools")
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

    # Default model
    model = model_provider or "anthropic/claude-3-5-sonnet-20241022"

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
def chat(agent_name: str, new_session: bool, session_id: str | None):
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
        console.print(f"\nAvailable agents:")
        agents_dir = Path("agents")
        if agents_dir.exists():
            for f in agents_dir.glob("*.yaml"):
                console.print(f"  - {f.stem}")
        else:
            console.print(
                "  [dim](none - create one with 'supyagent new <name>')[/dim]"
            )
        sys.exit(1)

    # Initialize session manager
    session_mgr = SessionManager()

    # Determine which session to use
    session = None
    if session_id:
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

    # Initialize agent
    try:
        agent = Agent(config, session=session, session_manager=session_mgr)
    except Exception as e:
        console.print(f"[red]Error initializing agent:[/red] {e}")
        sys.exit(1)

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
    show_tokens = False
    debug_mode = False

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
                        "  /rename <title>    Rename the current session\n"
                        "  /history [n]       Show last n messages (default: 10)\n"
                        "  /context           Show context window usage and status\n"
                        "  /tokens            Toggle token usage display after each turn\n"
                        "  /debug [on|off]    Toggle verbose debug mode\n"
                        "  /summarize         Force context summarization\n"
                        "  /export [file]     Export conversation to markdown\n"
                        "  /model [name]      Show or change model\n"
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
                    sessions = session_mgr.list_sessions(agent_name)
                    if not sessions:
                        console.print("[dim]No sessions found[/dim]")
                    else:
                        table = Table(title="Sessions")
                        table.add_column("ID", style="cyan")
                        table.add_column("Title")
                        table.add_column("Msgs", style="dim", justify="right")
                        table.add_column("Updated", style="dim")
                        table.add_column("", style="green")

                        current_id = agent.session.meta.session_id
                        for s in sessions:
                            marker = "← current" if s.session_id == current_id else ""
                            title = s.title or "(untitled)"
                            updated = s.updated_at.strftime("%Y-%m-%d %H:%M")
                            # Count messages
                            loaded = session_mgr.load_session(agent_name, s.session_id)
                            msg_count = str(len(loaded.messages)) if loaded else "?"
                            table.add_row(s.session_id, title, msg_count, updated, marker)

                        console.print(table)
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

                    console.print(f"\n[cyan]Context Status[/cyan]")
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
                        f"\n[cyan]Summarization Triggers (N messages OR K tokens)[/cyan]"
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
                            f"\n  [yellow]⚡ Summarization will trigger on next message[/yellow]"
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
                    for msg in agent.session.messages:
                        if msg.type == "user":
                            lines.append(f"**You:** {msg.content}\n")
                        elif msg.type == "assistant":
                            lines.append(f"**{config.name}:** {msg.content}\n")

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

                elif cmd == "rename":
                    if len(cmd_parts) < 2:
                        console.print("[yellow]Usage: /rename <new title>[/yellow]")
                        continue
                    new_title = " ".join(cmd_parts[1:])
                    agent.session.meta.title = new_title
                    session_mgr._update_meta(agent.session)
                    console.print(f"[green]Session renamed to \"{new_title}\"[/green]")
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
                            console.print(f"[green]   ✓ done[/green]")
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
def sessions(agent_name: str, search_query: str | None, delete_id: str | None, delete_all: bool):
    """List all sessions for an agent."""
    session_mgr = SessionManager()

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

    table = Table(title=title_text)
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Msgs", style="dim", justify="right")
    table.add_column("Created", style="dim")
    table.add_column("Updated", style="dim")
    table.add_column("", style="green")

    for s in session_list:
        marker = "← current" if s.session_id == current_id else ""
        title = s.title or "(untitled)"
        created = s.created_at.strftime("%Y-%m-%d %H:%M")
        updated = s.updated_at.strftime("%Y-%m-%d %H:%M")
        # Count messages
        loaded = session_mgr.load_session(agent_name, s.session_id)
        msg_count = str(len(loaded.messages)) if loaded else "?"
        table.add_row(s.session_id, title, msg_count, created, updated, marker)

    console.print(table)
    console.print()
    console.print(
        "[dim]Resume a session:[/dim] supyagent chat " + agent_name + " --session <id>"
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
        console.print(f"\n[bold]Allowed Tools:[/bold]")
        for pattern in config.tools.allow:
            console.print(f"  - {pattern}")

    if config.tools.deny:
        console.print(f"\n[bold]Denied Tools:[/bold]")
        for pattern in config.tools.deny:
            console.print(f"  - {pattern}")

    if config.delegates:
        console.print(f"\n[bold]Delegates:[/bold]")
        for delegate in config.delegates:
            console.print(f"  - {delegate}")

    console.print(f"\n[bold]System Prompt:[/bold]")
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

    console.print(f"[green]  ✓[/green] YAML syntax and required fields OK")

    # Step 3: Deep validation
    issues = validate_agent_config(config, agents_dir)

    if issues:
        for issue in issues:
            console.print(f"[yellow]  ![/yellow] {issue}")
    else:
        console.print(f"[green]  ✓[/green] Model, delegates, and tools OK")

    # Step 4: Summary
    console.print(
        f"\n  Agent: [cyan]{config.name}[/cyan] ({config.type})"
        f"\n  Model: [cyan]{config.model.provider}[/cyan]"
        f"\n  Tools: {len(config.tools.allow)} allow, {len(config.tools.deny)} deny"
    )
    if config.delegates:
        console.print(f"  Delegates: {', '.join(config.delegates)}")

    if not issues:
        console.print(f"\n[green]  All checks passed.[/green]")


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
            ["supypowers", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            console.print(f"[green]  ✓[/green] supypowers installed ({ver})")
        else:
            console.print(f"[red]  x[/red] supypowers not working (exit code {result.returncode})")
            all_ok = False
    except FileNotFoundError:
        console.print("[red]  x[/red] supypowers not installed")
        console.print("    Install: [cyan]pip install supypowers[/cyan]")
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
        console.print(f"[green]  ✓[/green] Sessions directory writable")
    else:
        console.print(f"[dim]  -[/dim] No sessions yet (will be created on first chat)")

    # 7. Check config encryption
    config_dir = Path.home() / ".supyagent" / "config"
    key_file = config_dir / "key.key"
    if key_file.exists():
        console.print(f"[green]  ✓[/green] Config encryption working")
    else:
        console.print(f"[dim]  -[/dim] Config encryption not yet initialized")

    # Summary
    if all_ok:
        console.print(f"\n[green]  All checks passed.[/green]")
    else:
        console.print(f"\n[yellow]  Some issues found. See above for fixes.[/yellow]")


# =============================================================================
# Tools Commands
# =============================================================================


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
        console.print("[yellow]No tools found.[/yellow]")
        console.print("Run [cyan]supyagent init[/cyan] to install default tools.")
        return

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
    console.print(f"\n[dim]{len(openai_tools)} tools total[/dim]")


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
    console.print(f"  Edit the file to add your implementation.")


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
        console.print(f"[red]Error:[/red] Tool name must be in 'script__function' format")
        console.print(f"  Example: shell__run_command, files__read_file")
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
        console.print(f"[green]  ✓[/green] ok: true")
        data = result.get("data")
        if data is not None:
            if isinstance(data, str) and len(data) > 200:
                console.print(f"  data: {data[:200]}...")
            else:
                console.print(f"  data: {json.dumps(data, indent=2) if not isinstance(data, str) else data}")
    else:
        console.print(f"[red]  x[/red] ok: false")
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

    console.print(f"\n[bold]model:[/bold]")
    _print_model(ModelConfig, indent=1)

    console.print(f"\n[bold]tools:[/bold]")
    _print_model(ToolPermissions, indent=1)

    console.print(f"\n[bold]context:[/bold]")
    _print_model(ContextSettings, indent=1)

    console.print(f"\n[bold]supervisor:[/bold]")
    _print_model(SupervisorSettings, indent=1)

    console.print(f"\n[bold]limits (dict):[/bold]")
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
                console_err.print(f"[green]   ✓ done[/green]")
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(inputs)} items...", total=len(inputs)
        )

        for item in inputs:
            result = runner.run(item, secrets=secrets_dict, output_format="json")
            results.append(result)
            progress.advance(task)

    # Count successes/failures
    successes = sum(1 for r in results if r["ok"])
    failures = len(results) - successes

    # Output
    output_content = "\n".join(json.dumps(r) for r in results)

    if output_file:
        with open(output_file, "w") as f:
            f.write(output_content + "\n")
        console.print(
            f"[green]✓[/green] Processed {len(results)} items "
            f"({successes} succeeded, {failures} failed)"
        )
        console.print(f"  Results written to [cyan]{output_file}[/cyan]")
    else:
        click.echo(output_content)


@cli.command()
def agents():
    """List all registered agent instances."""
    registry = AgentRegistry()
    instances = registry.list_all()

    if not instances:
        console.print("[dim]No active agent instances[/dim]")
        return

    # Build a table
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
def cleanup():
    """Clean up completed/failed agent instances from the registry."""
    registry = AgentRegistry()
    count = registry.cleanup_completed()

    if count == 0:
        console.print("[dim]No instances to clean up[/dim]")
    else:
        console.print(f"[green]✓[/green] Cleaned up {count} instance(s)")


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
        console.print(f"\n[bold]Metadata:[/bold]")
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
                console.print(f"\n[bold red]STDERR:[/bold red]")
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
            from pathlib import Path
            import re

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
