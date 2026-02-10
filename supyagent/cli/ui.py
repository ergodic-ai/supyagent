"""
UI components for the supyagent CLI.

Provides enhanced input (prompt_toolkit) and output (Rich Markdown rendering)
for the interactive chat and non-interactive run commands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

# ---------------------------------------------------------------------------
# Slash command definitions (command -> description)
# ---------------------------------------------------------------------------

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/image": "Send an image with optional message",
    "/tools": "List available tools",
    "/creds": "Manage credentials (list|set|delete)",
    "/sessions": "List all sessions",
    "/session": "Switch to another session",
    "/new": "Start a new session",
    "/delete": "Delete a session (default: current)",
    "/rename": "Set display title for current session",
    "/history": "Show last n messages (default: 10)",
    "/context": "Show context window usage",
    "/tokens": "Toggle token usage display",
    "/debug": "Toggle verbose debug mode",
    "/summarize": "Force context summarization",
    "/export": "Export conversation to markdown",
    "/model": "Show or change model",
    "/reload": "Reload tools",
    "/clear": "Clear screen",
    "/quit": "Exit the chat",
}

# Aliases that resolve to the same handler — included for completions
_SLASH_ALIASES: dict[str, str] = {
    "/h": "Show available commands",
    "/?": "Show available commands",
    "/exit": "Exit the chat",
    "/q": "Exit the chat",
}

# ---------------------------------------------------------------------------
# prompt_toolkit components (lazy imports to avoid hard crash if missing)
# ---------------------------------------------------------------------------


def _get_prompt_toolkit():
    """Import prompt_toolkit components. Returns None if unavailable."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import Completer, Completion, PathCompleter
        from prompt_toolkit.document import Document
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.lexers import Lexer
        from prompt_toolkit.styles import Style

        return {
            "PromptSession": PromptSession,
            "AutoSuggestFromHistory": AutoSuggestFromHistory,
            "Completer": Completer,
            "Completion": Completion,
            "PathCompleter": PathCompleter,
            "Document": Document,
            "FileHistory": FileHistory,
            "Lexer": Lexer,
            "Style": Style,
        }
    except ImportError:
        return None


class ChatCompleter:
    """
    Context-aware completer for the chat input.

    - Input starting with `/`: complete slash commands with descriptions
    - `/image`, `/export` arguments: complete file paths
    - `/creds` arguments: complete list|set|delete
    - `@` anywhere: complete file paths
    - Regular text: no completions (avoids noise)
    """

    def __init__(self):
        pt = _get_prompt_toolkit()
        if pt is None:
            raise ImportError("prompt_toolkit is required for ChatCompleter")
        self._Completion = pt["Completion"]
        self._PathCompleter = pt["PathCompleter"]
        self._Document = pt["Document"]
        self._path_completer = self._PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        _completion_cls = self._Completion
        _document_cls = self._Document

        # -- Slash command completion --
        if text.startswith("/"):
            if " " not in text:
                # Still typing the command name
                word = text
                all_commands = {**SLASH_COMMANDS, **_SLASH_ALIASES}
                for cmd, desc in all_commands.items():
                    if cmd.startswith(word):
                        yield _completion_cls(
                            cmd,
                            start_position=-len(word),
                            display_meta=desc,
                        )
                return

            # After the command — context-dependent sub-completions
            parts = text.split(None, 1)
            cmd_part = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            if cmd_part in ("/image", "/export"):
                sub_doc = _document_cls(rest, len(rest))
                yield from self._path_completer.get_completions(sub_doc, complete_event)
            elif cmd_part == "/creds":
                if " " not in rest:
                    for action in ("list", "set", "delete"):
                        if action.startswith(rest):
                            yield _completion_cls(action, start_position=-len(rest))
            elif cmd_part == "/debug":
                if " " not in rest:
                    for opt in ("on", "off"):
                        if opt.startswith(rest):
                            yield _completion_cls(opt, start_position=-len(rest))
            return

        # -- @-path completion --
        at_pos = text.rfind("@")
        if at_pos >= 0:
            path_fragment = text[at_pos + 1 :]
            # Only complete if we haven't moved past the path (no space after it)
            if " " not in path_fragment or path_fragment.endswith("/"):
                sub_doc = _document_cls(path_fragment, len(path_fragment))
                for c in self._path_completer.get_completions(sub_doc, complete_event):
                    yield c


class ChatLexer:
    """Syntax highlighting for chat input: /commands green, @paths blue."""

    def lex_document(self, document):
        def get_line(lineno):
            line = document.lines[lineno]
            result = []

            if line.startswith("/"):
                # Highlight command name green, rest as default
                parts = line.split(" ", 1)
                result.append(("class:slash-command", parts[0]))
                if len(parts) > 1:
                    result.append(("", " " + parts[1]))
                return result

            # Highlight @paths in blue
            i = 0
            while i < len(line):
                at_pos = line.find("@", i)
                if at_pos == -1:
                    result.append(("", line[i:]))
                    break
                if at_pos > i:
                    result.append(("", line[i:at_pos]))
                # Find end of path (next whitespace or end of line)
                end = at_pos + 1
                while end < len(line) and line[end] not in (" ", "\t"):
                    end += 1
                result.append(("class:at-path", line[at_pos:end]))
                i = end

            return result

        return get_line


# Style for the prompt, input highlighting, and completion menu
CHAT_STYLE_DICT = {
    "prompt": "bold #5599ff",
    "slash-command": "#00cc00 bold",
    "at-path": "#5599ff",
    "": "",
    # Completion menu
    "completion-menu.completion": "bg:#1a1a2e #cccccc",
    "completion-menu.completion.current": "bg:#16213e #ffffff",
    "completion-menu.meta.completion": "bg:#1a1a2e #666688",
    "completion-menu.meta.completion.current": "bg:#16213e #aaaacc",
    # Auto-suggest (ghost text)
    "auto-suggest": "#444466",
}


def create_chat_session(agent_name: str):
    """
    Create a prompt_toolkit PromptSession for the chat command.

    Returns None if not in a real terminal (e.g. under CliRunner in tests)
    or if prompt_toolkit is not available.
    """
    if not sys.stdin.isatty():
        return None

    pt = _get_prompt_toolkit()
    if pt is None:
        return None

    try:
        history_dir = Path(".supyagent") / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / f"{agent_name}.hist"

        return pt["PromptSession"](
            completer=ChatCompleter(),
            lexer=ChatLexer(),
            style=pt["Style"].from_dict(CHAT_STYLE_DICT),
            history=pt["FileHistory"](str(history_file)),
            auto_suggest=pt["AutoSuggestFromHistory"](),
            enable_history_search=True,
            complete_while_typing=False,
            multiline=False,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Streaming Markdown Renderer
# ---------------------------------------------------------------------------


class StreamingMarkdownRenderer:
    """
    Renders streaming LLM output as Rich Markdown.

    In terminal mode, uses Rich Live display to progressively render markdown
    as text chunks arrive. In non-terminal mode (piped), writes raw text.
    """

    def __init__(self, console: Console, use_live: bool | None = None):
        self._console = console
        self._buffer = ""
        self._live: Live | None = None
        self._use_live = console.is_terminal if use_live is None else use_live

    def feed(self, chunk: str):
        """Feed a text chunk from the LLM stream."""
        self._buffer += chunk
        if self._use_live:
            if self._live is None:
                self._live = Live(
                    Text(""),
                    console=self._console,
                    refresh_per_second=8,
                    transient=True,
                )
                self._live.start()
            try:
                self._live.update(Markdown(self._buffer))
            except Exception:
                self._live.update(Text(self._buffer))
        else:
            # Non-terminal: write raw text directly
            self._console.file.write(chunk)
            self._console.file.flush()

    def has_content(self) -> bool:
        """Check if any content has been accumulated."""
        return bool(self._buffer.strip())

    def flush_raw(self):
        """
        Print current buffer as plain text and clear it.

        Used for intermediate text segments before tool calls — we don't want
        to markdown-render partial responses that precede tool execution.
        """
        if self._live:
            self._live.stop()
            self._live = None
        if self._buffer:
            self._console.print(self._buffer, end="", highlight=False)
            self._buffer = ""

    def finish(self):
        """
        Stop live display and render the final text as Markdown.

        This produces the clean, formatted output with syntax-highlighted code
        blocks, proper headers, bold/italic, etc.
        """
        if self._live:
            self._live.stop()
            self._live = None
        if self._buffer.strip():
            self._console.print(Markdown(self._buffer))
            self._buffer = ""
        elif self._buffer:
            self._console.print()
            self._buffer = ""

    def get_text(self) -> str:
        """Get the accumulated raw text."""
        return self._buffer


# ---------------------------------------------------------------------------
# Tool call display helpers
# ---------------------------------------------------------------------------


def render_tool_start(console: Console, name: str, arguments: str | None):
    """Render a tool call start event."""
    console.print(f"\n  [cyan]{name}[/cyan]", highlight=False)
    if arguments:
        try:
            args = json.loads(arguments)
            if len(args) <= 2 and all(
                isinstance(v, (str, int, bool)) for v in args.values()
            ):
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                console.print(f"  [dim]  {args_str}[/dim]")
            else:
                args_str = json.dumps(args, indent=2)
                for line in args_str.split("\n"):
                    console.print(f"  [dim]  {line}[/dim]")
        except json.JSONDecodeError:
            pass


def render_tool_end(console: Console, result: dict):
    """Render a tool call end event."""
    if result.get("ok", False):
        console.print("  [green]  done[/green]")
    else:
        error = result.get("error", "failed")
        console.print(f"  [red]  error: {error}[/red]")


def render_reasoning(console: Console, content: str, is_first: bool):
    """Render reasoning/thinking content from the LLM."""
    if is_first:
        console.print("[magenta dim]  thinking... [/magenta dim]", end="")
    console.print(f"[magenta dim]{content}[/magenta dim]", end="")
