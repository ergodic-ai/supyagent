"""FastAPI dependency injection for agents, sessions, and configuration."""

from __future__ import annotations

import threading
from collections import OrderedDict

from supyagent.core.agent import Agent
from supyagent.core.credentials import CredentialManager
from supyagent.core.session_manager import SessionManager
from supyagent.models.agent_config import load_agent_config
from supyagent.models.session import Session


class AgentPool:
    """
    Thread-safe LRU cache of Agent instances keyed by (agent_name, session_id).

    Agents are expensive to create (tool discovery, context initialization).
    This pool avoids re-creating them on every request while bounding memory.
    """

    def __init__(self, max_size: int = 50):
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], Agent] = OrderedDict()
        self._agent_locks: dict[tuple[str, str], threading.Lock] = {}
        self._max_size = max_size
        self._session_manager = SessionManager()
        self._credential_manager = CredentialManager()

    def get_or_create(
        self,
        agent_name: str,
        session_id: str | None = None,
    ) -> Agent:
        """Get a cached agent or create a new one."""
        effective_session_id = session_id or "__new__"
        key = (agent_name, effective_session_id)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        config = load_agent_config(agent_name)

        # Resolve session: load from disk, or create with the provided ID
        session: Session | None = None
        if session_id:
            session = self._session_manager.load_session(agent_name, session_id)
            if session is None:
                # Create a new session using the caller-provided ID so the
                # file name matches future load_session() lookups.
                session = self._session_manager.create_session(
                    agent_name, config.model.provider, session_id=session_id
                )

        # Create outside lock (agent init can be slow)
        agent = Agent(
            config,
            session=session,
            session_manager=self._session_manager,
            credential_manager=self._credential_manager,
        )

        with self._lock:
            self._cache[key] = agent
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._agent_locks.pop(evicted_key, None)

        return agent

    def get_lock(self, agent_name: str, session_id: str) -> threading.Lock:
        """Get a per-agent lock for serializing requests to the same session."""
        key = (agent_name, session_id)
        with self._lock:
            if key not in self._agent_locks:
                self._agent_locks[key] = threading.Lock()
            return self._agent_locks[key]

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager


# Module-level singleton
_pool: AgentPool | None = None
_pool_lock = threading.Lock()


def get_agent_pool() -> AgentPool:
    """Get or create the global agent pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = AgentPool()
    return _pool


def reset_agent_pool() -> None:
    """Reset the global pool (for testing)."""
    global _pool
    _pool = None
