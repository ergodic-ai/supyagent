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

from supyagent.core.agent import Agent
from supyagent.core.config import ConfigManager, load_config
from supyagent.core.executor import ExecutionRunner
from supyagent.core.registry import AgentRegistry
from supyagent.core.session_manager import SessionManager
from supyagent.default_tools import install_default_tools, list_default_tools
from supyagent.models.agent_config import AgentNotFoundError, load_agent_config

console = Console()
console_err = Console(stderr=True)


@click.group()
@click.version_option(version="0.2.6", prog_name="supyagent")
def cli():
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
    pass


@cli.command()
@click.option(
    "--tools-dir",
    "-t",
    default="supypowers",
    help="Directory for tools (default: supypowers/)",
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
    - Default tools in supypowers/ (shell commands, file operations)
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
def new(name: str, agent_type: str):
    """
    Create a new agent from template.

    NAME is the agent name (will create agents/NAME.yaml)
    """
    agents_dir = Path("agents")
    agents_dir.mkdir(exist_ok=True)

    agent_path = agents_dir / f"{name}.yaml"

    if agent_path.exists():
        if not click.confirm(f"Agent '{name}' already exists. Overwrite?"):
            return

    # Create template based on type
    if agent_type == "interactive":
        template = f"""name: {name}
description: An interactive AI assistant
version: "1.0"
type: interactive

model:
  provider: anthropic/claude-3-5-sonnet-20241022
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
  provider: anthropic/claude-3-5-sonnet-20241022
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
        # Resume specific session
        session = session_mgr.load_session(agent_name, session_id)
        if not session:
            console.print(f"[red]Error:[/red] Session '{session_id}' not found")
            console.print("\nAvailable sessions:")
            for s in session_mgr.list_sessions(agent_name):
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
                        "  /tools             List available tools\n"
                        "  /creds [action]    Manage credentials (list|set|delete)\n"
                        "  /sessions          List all sessions\n"
                        "  /session <id>      Switch to another session\n"
                        "  /new               Start a new session\n"
                        "  /history [n]       Show last n messages (default: 10)\n"
                        "  /context           Show context window usage and status\n"
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
                        table.add_column("Updated", style="dim")
                        table.add_column("", style="green")

                        current_id = agent.session.meta.session_id
                        for s in sessions:
                            marker = "← current" if s.session_id == current_id else ""
                            title = s.title or "(untitled)"
                            updated = s.updated_at.strftime("%Y-%m-%d %H:%M")
                            table.add_row(s.session_id, title, updated, marker)

                        console.print(table)
                    continue

                elif cmd == "session":
                    if len(cmd_parts) < 2:
                        console.print("[yellow]Usage: /session <id>[/yellow]")
                        continue

                    target_id = cmd_parts[1]
                    new_sess = session_mgr.load_session(agent_name, target_id)
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
                    conversation_messages = [
                        m for m in agent.messages if m.get("role") != "system"
                    ]
                    status = agent.context_manager.get_trigger_status(
                        conversation_messages
                    )

                    console.print(f"\n[cyan]Context Status[/cyan]")

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
def sessions(agent_name: str):
    """List all sessions for an agent."""
    session_mgr = SessionManager()
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

    table = Table(title=f"Sessions for {agent_name}")
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Created", style="dim")
    table.add_column("Updated", style="dim")
    table.add_column("", style="green")

    for s in session_list:
        marker = "← current" if s.session_id == current_id else ""
        title = s.title or "(untitled)"
        created = s.created_at.strftime("%Y-%m-%d %H:%M")
        updated = s.updated_at.strftime("%Y-%m-%d %H:%M")
        table.add_row(s.session_id, title, created, updated, marker)

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


if __name__ == "__main__":
    cli()
