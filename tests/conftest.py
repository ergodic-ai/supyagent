"""
Pytest fixtures for supyagent tests.
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_env():
    """Prevent environment variable pollution between tests.

    Some CLI commands call load_config() which loads real credentials
    from ~/.supyagent/config/ into os.environ. Without this fixture,
    those variables persist across tests.
    """
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
