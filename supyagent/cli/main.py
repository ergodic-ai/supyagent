"""
CLI entry point for supyagent — cloud integrations for AI agents.
"""

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    from importlib.metadata import version as pkg_version

    _version = pkg_version("supyagent")
except Exception:
    _version = "0.6.2"

from supyagent.core.config import ConfigManager

console = Console()
console_err = Console(stderr=True)


# =============================================================================
# Root CLI Group
# =============================================================================


@click.group(invoke_without_command=True)
@click.version_option(version=_version, prog_name="supyagent")
@click.option("--debug", is_flag=True, hidden=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, debug: bool):
    """
    Supyagent — cloud integrations for AI agents.

    Connect third-party services and use them from the command line.

    \b
        supyagent connect          # Authenticate with the service
        supyagent status           # Show connection & available tools
        supyagent service tools    # List available tools
        supyagent service run ...  # Execute a tool
        supyagent inbox            # View incoming events
        supyagent skills generate  # Generate AI skill files
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    if debug:
        import logging

        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    if ctx.invoked_subcommand is None:
        from supyagent.core.service import get_service_client

        client = get_service_client()
        if client:
            client.close()
            ctx.invoke(status)
        else:
            console.print(
                Panel(
                    "[bold]Welcome to supyagent[/bold]\n\n"
                    "Connect to the supyagent service to unlock cloud integrations\n"
                    "(Gmail, Slack, GitHub, Calendar, and 50+ more tools).\n\n"
                    "Run [cyan]supyagent connect[/cyan] to get started.",
                    border_style="blue",
                )
            )


# =============================================================================
# Connection Commands
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
    console.print("[bold green]\u2713 Connected![/bold green]")
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
                tool_table.add_row(provider, ", ".join(services) if services else "-")
            console.print(tool_table)
            console.print(f"[bright_black]{len(tools)} tools available[/bright_black]")
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
        console.print(
            "[grey62]No integrations connected. Visit the dashboard to add services.[/grey62]"
        )
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
@click.option(
    "--status",
    "-s",
    type=click.Choice(["unread", "read", "archived"]),
    default=None,
    help="Filter by status",
)
@click.option(
    "--provider", "-p", default=None, help="Filter by provider (e.g. github, slack)"
)
@click.option(
    "--limit", "-n", default=20, type=int, help="Number of events (default: 20)"
)
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

        console.print(
            Panel(
                f"[bold]{event.get('provider', '?')}[/bold] / {event.get('event_type', '?')}\n"
                f"[grey62]{event.get('received_at', '')}[/grey62]\n\n"
                f"{event.get('summary', 'No summary')}\n\n"
                f"[grey62]Status: {event.get('status', '?')} | ID: {event.get('id', '?')}[/grey62]",
                title="Inbox Event",
                border_style="cyan",
            )
        )

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
        console.print(
            f"\n[bright_black]Showing {len(events)} of {total}. Use -n to show more.[/bright_black]"
        )


# =============================================================================
# Doctor Command
# =============================================================================


@cli.command()
def doctor():
    """
    Diagnose your supyagent cloud setup.

    Checks service connection, config encryption, and reachability.
    """
    from supyagent.core.service import (
        DEFAULT_SERVICE_URL,
        SERVICE_API_KEY,
        SERVICE_URL,
        ServiceClient,
    )

    all_ok = True

    # 1. Version
    console.print(f"[green]  ok[/green] supyagent v{_version}")

    # 2. Config encryption
    config_dir = Path.home() / ".supyagent" / "config"
    key_file = config_dir / ".key"
    if key_file.exists():
        console.print("[green]  ok[/green] Config encryption working")
    else:
        console.print("[dim]  --[/dim] Config encryption not yet initialized")

    # 3. Service connection
    config_mgr = ConfigManager()
    api_key = config_mgr.get(SERVICE_API_KEY)

    if not api_key:
        console.print("[yellow]  !![/yellow] Not connected to service")
        console.print("     Run: [cyan]supyagent connect[/cyan]")
        all_ok = False
    else:
        base_url = config_mgr.get(SERVICE_URL) or DEFAULT_SERVICE_URL
        console.print(f"[green]  ok[/green] Service API key configured ({base_url})")

        # 4. Health check
        client = ServiceClient(api_key=api_key, base_url=base_url)
        reachable = client.health_check()
        if reachable:
            console.print("[green]  ok[/green] Service is reachable")
            tools = client.discover_tools()
            if tools:
                console.print(f"[green]  ok[/green] {len(tools)} tools available")
            else:
                console.print(
                    "[yellow]  !![/yellow] No tools available (connect integrations on dashboard)"
                )
        else:
            console.print(f"[yellow]  !![/yellow] Service not reachable at {base_url}")
            all_ok = False
        client.close()

    # 5. Stored API keys
    stored_keys = config_mgr.list_keys()
    if stored_keys:
        console.print(f"[green]  ok[/green] {len(stored_keys)} config key(s) stored")
    else:
        console.print("[dim]  --[/dim] No API keys stored")

    # Summary
    if all_ok:
        console.print("\n[green]All checks passed.[/green]")
    else:
        console.print("\n[yellow]Some issues found. See above for fixes.[/yellow]")


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
        console.print(f"[green]\u2713[/green] Saved {key_name}")
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
        console.print(f"[green]\u2713[/green] Deleted {key_name}")
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
                            console.print(f"  [green]\u2713[/green] {name}")
                            imported += 1
        else:
            imported = config_mgr.set_from_file(file_path)

        if imported == 0:
            console.print("[yellow]No keys found in file[/yellow]")
        else:
            console.print(f"\n[green]\u2713[/green] Imported {imported} key(s)")

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

    console.print(f"[green]\u2713[/green] Exported {len(keys)} key(s) to {file_path}")


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
        console.print(
            "[dim]No tools available. Connect integrations on the dashboard.[/dim]"
        )
        return

    # Filter by provider if specified
    if provider:
        tools = [
            t
            for t in tools
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
            output.append(
                {
                    "name": f"{meta.get('provider', '?')}:{func.get('name', '?')}",
                    "description": func.get("description", ""),
                    "provider": meta.get("provider"),
                    "service": meta.get("service"),
                    "method": meta.get("method"),
                    "parameters": func.get("parameters", {}),
                }
            )
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
def service_inbox(status: str | None, provider: str | None, limit: int, as_json: bool):
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
            else (
                "[bright_black]read[/bright_black]"
                if st == "read"
                else "[bright_black]archived[/bright_black]"
            )
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
        console.print(
            f"  Event ID:  [bright_black]{event['provider_event_id']}[/bright_black]"
        )
    console.print()
    console.print("[bold]Payload:[/bold]")
    payload_str = json.dumps(event.get("payload", {}), indent=2)
    console.print(f"[grey62]{payload_str}[/grey62]")


@service_group.command("inbox:archive")
@click.argument("event_id", required=False, default=None)
@click.option("--all", "archive_all", is_flag=True, help="Archive all events")
@click.option(
    "--provider",
    "-p",
    default=None,
    help="When using --all, archive only this provider",
)
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
            console.print(
                f"[green]Archived {count} event{'s' if count != 1 else ''}.[/green]"
            )
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


@service_group.command("run", context_settings={"ignore_unknown_options": True})
@click.argument("tool_spec")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(),
    help="Read arguments from JSON file (use '-' for stdin)",
)
@click.option(
    "--timeout",
    "-t",
    type=float,
    default=None,
    help="Read timeout in seconds (default: 180)",
)
def service_run(
    tool_spec: str,
    extra_args: tuple[str, ...],
    input_file: str | None,
    timeout: float | None,
):
    """
    Execute a service tool directly.

    TOOL_SPEC is the tool name (e.g., gmail_send_message or gmail:send_message).
    Pass arguments as JSON or as --key value flags.

    \b
    Examples:
        supyagent service run image_generate --prompt "A red cube"
        supyagent service run gmail_list_messages --max_results 5
        supyagent service run gmail_list_messages '{"max_results": 5}'
        supyagent service run slack:send_message '{"channel": "#general", "text": "Hello"}'
        echo '{"query": "test"}' | supyagent service run gmail_search_messages --input -
    """
    from supyagent.core.service import get_service_client

    client = get_service_client(timeout=timeout)
    if not client:
        console.print("[yellow]Not connected to service.[/yellow]")
        console.print("Run [cyan]supyagent connect[/cyan] to authenticate.")
        sys.exit(1)

    # Parse arguments from --input, JSON string, or --key value flags
    args: dict[str, Any] = {}

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

    # Check if first extra arg is a JSON string (legacy positional usage)
    remaining = list(extra_args)
    if remaining and remaining[0].startswith("{"):
        try:
            args = {**args, **json.loads(remaining[0])}
            remaining = remaining[1:]
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON arguments: {e}")
            sys.exit(1)

    # Parse --key value pairs from remaining args
    i = 0
    while i < len(remaining):
        token = remaining[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                value = remaining[i + 1]
                # Auto-convert booleans and numbers
                if value.lower() == "true":
                    args[key] = True
                elif value.lower() == "false":
                    args[key] = False
                else:
                    try:
                        args[key] = int(value)
                    except ValueError:
                        try:
                            args[key] = float(value)
                        except ValueError:
                            args[key] = value
                i += 2
            else:
                # Bare flag with no value → treat as boolean true
                args[key] = True
                i += 1
        else:
            console.print(f"[red]Error:[/red] Unexpected argument: {token}")
            sys.exit(1)
            i += 1

    # Normalize tool spec: support both "gmail:send_message" and "gmail_send_message"
    tool_name = tool_spec.replace(":", "_")

    # Discover tools to find metadata
    console_err.print("[dim]Discovering tools...[/dim]")
    tools = client.discover_tools()

    if not tools:
        console.print(
            "[red]Error:[/red] No tools available. Connect integrations on the dashboard."
        )
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
            if name.endswith(f"_{tool_name}") or name.endswith(
                f"_{tool_spec.replace(':', '_')}"
            ):
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

    # Materialize binary content (PDFs, images) to temp files so that
    # external tools like Claude Code's Read can view them visually.
    if result.get("ok") and result.get("data"):
        from supyagent.utils.binary import cleanup_temp_dir, materialize_binary_content

        cleanup_temp_dir()
        result["data"] = materialize_binary_content(result["data"])

    # Output result as JSON
    click.echo(json.dumps(result, indent=2))

    if not result.get("ok"):
        sys.exit(1)


# =============================================================================
# Skills Commands
# =============================================================================


@cli.group("skills")
def skills_group():
    """Generate skill files for AI coding assistants."""
    pass


@skills_group.command("generate")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory. If omitted, detects AI tool folders interactively.",
)
@click.option(
    "--stdout",
    is_flag=True,
    help="Print to stdout instead of writing to files",
)
@click.option(
    "--all",
    "select_all",
    is_flag=True,
    help="Write skills to all detected AI tool folders without prompting.",
)
def skills_generate(output: str | None, stdout: bool, select_all: bool):
    """
    Generate skill files for AI coding assistants from connected integrations.

    Queries your connected service integrations and generates one skill
    file per integration. Automatically detects AI tool folders (.claude,
    .cursor, .agents, .copilot, .windsurf) and lets you choose which to
    populate.

    \b
    Examples:
        supyagent skills generate
        supyagent skills generate --all
        supyagent skills generate --stdout
        supyagent skills generate -o custom/path/
    """
    from supyagent.cli.skills import (
        generate_skill_files,
        generate_skill_md,
        resolve_output_dirs,
        write_skills_to_dir,
    )
    from supyagent.core.service import get_service_client

    client = get_service_client()
    if not client:
        console_err.print("[yellow]Not connected to service.[/yellow]")
        console_err.print("Run [cyan]supyagent connect[/cyan] to authenticate.")
        sys.exit(1)

    try:
        tools = client.discover_tools()
    finally:
        client.close()

    if not tools:
        console_err.print("[yellow]No tools available.[/yellow]")
        console_err.print(
            "Connect integrations on the dashboard, then run this command again."
        )
        sys.exit(1)

    if stdout:
        click.echo(generate_skill_md(tools))
        return

    skill_files = generate_skill_files(tools)
    output_dirs = resolve_output_dirs(output, select_all)

    for output_dir in output_dirs:
        write_skills_to_dir(output_dir, skill_files)

    console_err.print(
        f"[green]\u2713[/green] Generated [cyan]{len(skill_files)}[/cyan] "
        f"skills ({len(tools)} tools) in [cyan]{len(output_dirs)}[/cyan] location(s)"
    )
    for output_dir in output_dirs:
        console_err.print(f"  [dim]{output_dir}/[/dim]")
        for dir_name in sorted(skill_files.keys()):
            console_err.print(f"    {dir_name}/SKILL.md")
    console_err.print()
    console_err.print(
        "[dim]Your AI coding assistant will automatically use these skills when you ask "
        "about connected services.[/dim]"
    )


if __name__ == "__main__":
    cli()
