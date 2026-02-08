"""
Agent Registry for tracking active agent instances.

Enables multi-agent orchestration by tracking parent-child relationships
and managing agent lifecycles.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from supyagent.core.agent import Agent


@dataclass
class AgentInstance:
    """
    Metadata for a running agent instance.

    Attributes:
        name: The agent type name (e.g., "planner", "researcher")
        instance_id: Unique identifier for this instance
        created_at: When the instance was created
        parent_id: Instance ID of the parent agent (if spawned by another)
        status: Current status (active, completed, failed)
    """

    name: str
    instance_id: str
    created_at: datetime
    parent_id: str | None = None
    status: str = "active"
    depth: int = 0  # Delegation depth for preventing infinite loops

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "instance_id": self.instance_id,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
            "status": self.status,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentInstance":
        """Deserialize from dictionary."""
        created_at = data["created_at"]
        if isinstance(created_at, str):
            # Handle both aware and naive datetime strings
            if created_at.endswith("Z"):
                created_at = created_at[:-1] + "+00:00"
            created_at = datetime.fromisoformat(created_at)

        return cls(
            name=data["name"],
            instance_id=data["instance_id"],
            created_at=created_at,
            parent_id=data.get("parent_id"),
            status=data.get("status", "active"),
            depth=data.get("depth", 0),
        )


class AgentRegistry:
    """
    Manages agent instances and their relationships.

    Enables agents to spawn and communicate with sub-agents while
    tracking the hierarchy for debugging and resource management.
    """

    # Maximum delegation depth to prevent infinite loops
    MAX_DEPTH = 5

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize the registry.

        Args:
            base_dir: Base directory for storing registry data.
                      Defaults to ~/.supyagent/
        """
        if base_dir is None:
            base_dir = Path.home() / ".supyagent"

        self.base_dir = Path(base_dir)
        self.registry_path = self.base_dir / "registry.json"

        self._instances: dict[str, AgentInstance] = {}
        self._agents: dict[str, "Agent"] = {}  # Live agent objects

        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                    for item in data.get("instances", []):
                        inst = AgentInstance.from_dict(item)
                        self._instances[inst.instance_id] = inst
            except (json.JSONDecodeError, KeyError):
                # Corrupted registry, start fresh
                self._instances = {}

    def _save(self) -> None:
        """Persist registry to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        with open(self.registry_path, "w") as f:
            json.dump(
                {
                    "instances": [
                        inst.to_dict()
                        for inst in self._instances.values()
                    ]
                },
                f,
                indent=2,
            )

    def register(
        self,
        agent: "Agent",
        parent_id: str | None = None,
    ) -> str:
        """
        Register an agent instance and return its ID.

        Args:
            agent: The agent to register
            parent_id: Instance ID of the parent agent (if any)

        Returns:
            The new instance ID

        Raises:
            ValueError: If max delegation depth would be exceeded
        """
        instance_id = str(uuid.uuid4())[:8]

        # Calculate depth
        depth = 0
        if parent_id:
            parent = self._instances.get(parent_id)
            if parent:
                depth = parent.depth + 1

            if depth > self.MAX_DEPTH:
                raise ValueError(
                    f"Maximum delegation depth ({self.MAX_DEPTH}) exceeded. "
                    "Cannot create more sub-agents."
                )

        instance = AgentInstance(
            name=agent.config.name,
            instance_id=instance_id,
            created_at=datetime.now(UTC),
            parent_id=parent_id,
            depth=depth,
        )

        self._instances[instance_id] = instance
        self._agents[instance_id] = agent
        self._save()

        return instance_id

    def get_agent(self, instance_id: str) -> "Agent | None":
        """Get a live agent by instance ID."""
        return self._agents.get(instance_id)

    def get_instance(self, instance_id: str) -> AgentInstance | None:
        """Get instance metadata by ID."""
        return self._instances.get(instance_id)

    def list_all(self) -> list[AgentInstance]:
        """List all registered instances."""
        return list(self._instances.values())

    def list_active(self) -> list[AgentInstance]:
        """List all active instances."""
        return [i for i in self._instances.values() if i.status == "active"]

    def list_children(self, parent_id: str) -> list[AgentInstance]:
        """List all agents spawned by a parent."""
        return [i for i in self._instances.values() if i.parent_id == parent_id]

    def mark_completed(self, instance_id: str) -> None:
        """Mark an agent as completed."""
        if instance_id in self._instances:
            self._instances[instance_id].status = "completed"
            self._save()

    def mark_failed(self, instance_id: str) -> None:
        """Mark an agent as failed."""
        if instance_id in self._instances:
            self._instances[instance_id].status = "failed"
            self._save()

    def cleanup(self, instance_id: str) -> None:
        """Remove an agent instance."""
        if instance_id in self._agents:
            del self._agents[instance_id]

        if instance_id in self._instances:
            del self._instances[instance_id]
            self._save()

    def cleanup_completed(self) -> int:
        """
        Remove all completed/failed instances.

        Returns:
            Number of instances removed
        """
        to_remove = [
            iid
            for iid, inst in self._instances.items()
            if inst.status in ("completed", "failed")
        ]

        for iid in to_remove:
            self.cleanup(iid)

        return len(to_remove)

    def prune_stale(self, max_age_hours: float = 24) -> int:
        """
        Remove stale entries: 'active' instances older than max_age_hours.

        These are typically leftover from crashed processes.

        Returns:
            Number of instances removed
        """
        cutoff = datetime.now(UTC) - __import__("datetime").timedelta(hours=max_age_hours)
        to_remove = [
            iid
            for iid, inst in self._instances.items()
            if inst.status == "active" and inst.created_at < cutoff
        ]

        for iid in to_remove:
            self.cleanup(iid)

        return len(to_remove)

    def stats(self) -> dict[str, int]:
        """Return a summary of registry state."""
        active = sum(1 for i in self._instances.values() if i.status == "active")
        completed = sum(1 for i in self._instances.values() if i.status == "completed")
        failed = sum(1 for i in self._instances.values() if i.status == "failed")
        return {
            "total": len(self._instances),
            "active": active,
            "completed": completed,
            "failed": failed,
        }

    def get_depth(self, instance_id: str) -> int:
        """Get the delegation depth of an instance."""
        inst = self._instances.get(instance_id)
        return inst.depth if inst else 0
