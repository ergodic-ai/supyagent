"""Template loader for UI HTML files."""

from __future__ import annotations

import importlib.resources

from jinja2 import BaseLoader, Environment, TemplateNotFound


class _ResourceLoader(BaseLoader):
    """Load templates from this package using importlib.resources."""

    def get_source(self, environment, template):
        ref = importlib.resources.files(__package__).joinpath(template)
        try:
            source = ref.read_text(encoding="utf-8")
        except (FileNotFoundError, TypeError):
            raise TemplateNotFound(template)
        return source, str(ref), lambda: True


_env = Environment(loader=_ResourceLoader(), autoescape=False)


def render_template(name: str, **kwargs) -> str:
    """Render a Jinja2 template by name."""
    tmpl = _env.get_template(name)
    return tmpl.render(**kwargs)


def load_template(name: str) -> str:
    """Load raw template source (for non-Jinja use)."""
    ref = importlib.resources.files(__package__).joinpath(name)
    return ref.read_text(encoding="utf-8")
