"""
Tests for CLI UI components (supyagent/cli/ui.py).
"""

from io import StringIO

import pytest
from prompt_toolkit.document import Document
from rich.console import Console

from supyagent.cli.ui import (
    SLASH_COMMANDS,
    ChatCompleter,
    ChatLexer,
    StreamingMarkdownRenderer,
    render_tool_end,
    render_tool_start,
)

# ---------------------------------------------------------------------------
# ChatCompleter tests
# ---------------------------------------------------------------------------


class TestChatCompleter:
    @pytest.fixture
    def completer(self):
        return ChatCompleter()

    def _complete(self, completer, text):
        """Helper to get completions for a text string."""
        doc = Document(text, len(text))
        return list(completer.get_completions(doc, None))

    def test_slash_command_completion(self, completer):
        completions = self._complete(completer, "/he")
        texts = [c.text for c in completions]
        assert "/help" in texts

    def test_slash_command_shows_description(self, completer):
        completions = self._complete(completer, "/he")
        help_completion = next(c for c in completions if c.text == "/help")
        assert help_completion.display_meta is not None

    def test_all_commands_complete_from_slash(self, completer):
        completions = self._complete(completer, "/")
        texts = [c.text for c in completions]
        # Should include all main commands and aliases
        for cmd in SLASH_COMMANDS:
            assert cmd in texts

    def test_no_completions_for_regular_text(self, completer):
        completions = self._complete(completer, "hello world")
        assert len(completions) == 0

    def test_no_completions_for_empty_input(self, completer):
        completions = self._complete(completer, "")
        assert len(completions) == 0

    def test_creds_subcompletions(self, completer):
        completions = self._complete(completer, "/creds l")
        texts = [c.text for c in completions]
        assert "list" in texts

    def test_creds_all_actions(self, completer):
        completions = self._complete(completer, "/creds ")
        texts = [c.text for c in completions]
        assert "list" in texts
        assert "set" in texts
        assert "delete" in texts

    def test_debug_subcompletions(self, completer):
        completions = self._complete(completer, "/debug ")
        texts = [c.text for c in completions]
        assert "on" in texts
        assert "off" in texts

    def test_at_path_triggers_completion(self, completer, tmp_path):
        # The @ should trigger file path completion
        # We can't easily test the actual file paths, but we can verify
        # the completer doesn't crash
        completions = self._complete(completer, "@/tmp")
        # Should get some completions for /tmp (if it exists)
        assert isinstance(completions, list)

    def test_at_path_mid_sentence(self, completer):
        completions = self._complete(completer, "look at this @/tmp")
        assert isinstance(completions, list)


# ---------------------------------------------------------------------------
# ChatLexer tests
# ---------------------------------------------------------------------------


class TestChatLexer:
    @pytest.fixture
    def lexer(self):
        return ChatLexer()

    def test_slash_command_highlighted(self, lexer):
        doc = Document("/help", len("/help"))
        fragments = lexer.lex_document(doc)(0)
        assert fragments[0] == ("class:slash-command", "/help")

    def test_slash_command_with_args(self, lexer):
        doc = Document("/model gpt-4", len("/model gpt-4"))
        fragments = lexer.lex_document(doc)(0)
        assert fragments[0] == ("class:slash-command", "/model")
        assert fragments[1] == ("", " gpt-4")

    def test_at_path_highlighted(self, lexer):
        doc = Document("check @/some/path here", len("check @/some/path here"))
        fragments = lexer.lex_document(doc)(0)
        # Should have: text, @path, text
        styles = [frag[0] for frag in fragments]
        assert "class:at-path" in styles

    def test_regular_text_no_style(self, lexer):
        doc = Document("hello world", len("hello world"))
        fragments = lexer.lex_document(doc)(0)
        assert all(frag[0] == "" for frag in fragments)

    def test_multiple_at_paths(self, lexer):
        doc = Document("@foo and @bar", len("@foo and @bar"))
        fragments = lexer.lex_document(doc)(0)
        at_path_count = sum(1 for frag in fragments if frag[0] == "class:at-path")
        assert at_path_count == 2


# ---------------------------------------------------------------------------
# StreamingMarkdownRenderer tests
# ---------------------------------------------------------------------------


class TestStreamingMarkdownRenderer:
    def _make_console(self):
        buf = StringIO()
        return Console(file=buf, force_terminal=True, width=80), buf

    def test_feed_accumulates_text(self):
        console, buf = self._make_console()
        renderer = StreamingMarkdownRenderer(console, use_live=False)
        renderer.feed("hello ")
        renderer.feed("world")
        assert renderer.get_text() == "hello world"

    def test_has_content(self):
        console, buf = self._make_console()
        renderer = StreamingMarkdownRenderer(console, use_live=False)
        assert not renderer.has_content()
        renderer.feed("hi")
        assert renderer.has_content()

    def test_finish_renders_markdown(self):
        console, buf = self._make_console()
        renderer = StreamingMarkdownRenderer(console, use_live=False)
        renderer.feed("# Hello\n\nWorld")
        renderer.finish()
        output = buf.getvalue()
        assert "Hello" in output
        assert "World" in output

    def test_flush_raw_clears_buffer(self):
        console, buf = self._make_console()
        renderer = StreamingMarkdownRenderer(console, use_live=False)
        renderer.feed("partial text")
        renderer.flush_raw()
        assert not renderer.has_content()
        assert renderer.get_text() == ""

    def test_flush_raw_prints_text(self):
        console, buf = self._make_console()
        renderer = StreamingMarkdownRenderer(console, use_live=False)
        renderer.feed("partial text")
        renderer.flush_raw()
        output = buf.getvalue()
        assert "partial text" in output

    def test_finish_clears_buffer(self):
        console, buf = self._make_console()
        renderer = StreamingMarkdownRenderer(console, use_live=False)
        renderer.feed("content")
        renderer.finish()
        assert not renderer.has_content()

    def test_non_terminal_writes_raw(self):
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=80)
        renderer = StreamingMarkdownRenderer(console)  # should detect non-terminal
        renderer.feed("raw text")
        # In non-terminal mode, text is written directly
        output = buf.getvalue()
        assert "raw text" in output

    def test_empty_finish_is_safe(self):
        console, buf = self._make_console()
        renderer = StreamingMarkdownRenderer(console, use_live=False)
        renderer.finish()  # Should not crash
        assert buf.getvalue() == "" or buf.getvalue().strip() == ""


# ---------------------------------------------------------------------------
# Tool rendering helpers tests
# ---------------------------------------------------------------------------


class TestToolRendering:
    def _make_console(self):
        buf = StringIO()
        return Console(file=buf, force_terminal=True, width=80), buf

    def test_render_tool_start_simple_args(self):
        console, buf = self._make_console()
        render_tool_start(console, "web_search", '{"query": "hello"}')
        output = buf.getvalue()
        assert "web_search" in output
        assert "hello" in output

    def test_render_tool_start_no_args(self):
        console, buf = self._make_console()
        render_tool_start(console, "my_tool", None)
        output = buf.getvalue()
        assert "my_tool" in output

    def test_render_tool_start_complex_args(self):
        console, buf = self._make_console()
        import json

        args = json.dumps({"a": 1, "b": 2, "c": 3, "d": [1, 2, 3]})
        render_tool_start(console, "complex_tool", args)
        output = buf.getvalue()
        assert "complex_tool" in output

    def test_render_tool_end_success(self):
        console, buf = self._make_console()
        render_tool_end(console, {"ok": True, "data": "result"})
        output = buf.getvalue()
        assert "done" in output

    def test_render_tool_end_failure(self):
        console, buf = self._make_console()
        render_tool_end(console, {"ok": False, "error": "something broke"})
        output = buf.getvalue()
        assert "something broke" in output
