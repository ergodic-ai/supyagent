"""
Tests for AgentRegistry.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from supyagent.core.registry import AgentInstance, AgentRegistry


class TestAgentInstance:
    """Tests for AgentInstance dataclass."""

    def test_instance_creation(self):
        """Test creating an agent instance."""
        now = datetime.now(UTC)
        instance = AgentInstance(
            name="test-agent",
            instance_id="abc123",
            created_at=now,
        )

        assert instance.name == "test-agent"
        assert instance.instance_id == "abc123"
        assert instance.created_at == now
        assert instance.parent_id is None
        assert instance.status == "active"
        assert instance.depth == 0

    def test_instance_with_parent(self):
        """Test creating an instance with a parent."""
        instance = AgentInstance(
            name="sub-agent",
            instance_id="def456",
            created_at=datetime.now(UTC),
            parent_id="abc123",
            depth=1,
        )

        assert instance.parent_id == "abc123"
        assert instance.depth == 1

    def test_to_dict(self):
        """Test serializing to dictionary."""
        now = datetime.now(UTC)
        instance = AgentInstance(
            name="test",
            instance_id="id123",
            created_at=now,
            parent_id="parent",
            status="completed",
            depth=2,
        )

        d = instance.to_dict()
        assert d["name"] == "test"
        assert d["instance_id"] == "id123"
        assert d["parent_id"] == "parent"
        assert d["status"] == "completed"
        assert d["depth"] == 2

    def test_from_dict(self):
        """Test deserializing from dictionary."""
        data = {
            "name": "test",
            "instance_id": "id123",
            "created_at": "2024-01-15T10:30:00+00:00",
            "parent_id": "parent",
            "status": "active",
            "depth": 1,
        }

        instance = AgentInstance.from_dict(data)
        assert instance.name == "test"
        assert instance.instance_id == "id123"
        assert instance.parent_id == "parent"
        assert instance.depth == 1


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_register_agent(self, temp_dir):
        """Test registering an agent."""
        registry = AgentRegistry(base_dir=temp_dir)

        mock_agent = MagicMock()
        mock_agent.config.name = "test-agent"

        instance_id = registry.register(mock_agent)

        assert len(instance_id) == 8
        assert registry.get_instance(instance_id) is not None
        assert registry.get_agent(instance_id) == mock_agent

    def test_register_with_parent(self, temp_dir):
        """Test registering a sub-agent with parent."""
        registry = AgentRegistry(base_dir=temp_dir)

        parent_agent = MagicMock()
        parent_agent.config.name = "parent"
        parent_id = registry.register(parent_agent)

        child_agent = MagicMock()
        child_agent.config.name = "child"
        child_id = registry.register(child_agent, parent_id=parent_id)

        child_instance = registry.get_instance(child_id)
        assert child_instance.parent_id == parent_id
        assert child_instance.depth == 1

    def test_max_depth_exceeded(self, temp_dir):
        """Test that max delegation depth is enforced."""
        registry = AgentRegistry(base_dir=temp_dir)

        # Create a chain of agents up to max depth
        current_id = None
        for i in range(AgentRegistry.MAX_DEPTH + 1):
            agent = MagicMock()
            agent.config.name = f"agent-{i}"

            if i <= AgentRegistry.MAX_DEPTH:
                current_id = registry.register(agent, parent_id=current_id)
            else:
                # Should raise error at max depth
                with pytest.raises(ValueError, match="Maximum delegation depth"):
                    registry.register(agent, parent_id=current_id)

    def test_list_children(self, temp_dir):
        """Test listing child agents."""
        registry = AgentRegistry(base_dir=temp_dir)

        parent = MagicMock()
        parent.config.name = "parent"
        parent_id = registry.register(parent)

        child1 = MagicMock()
        child1.config.name = "child1"
        registry.register(child1, parent_id=parent_id)

        child2 = MagicMock()
        child2.config.name = "child2"
        registry.register(child2, parent_id=parent_id)

        children = registry.list_children(parent_id)
        assert len(children) == 2
        assert all(c.parent_id == parent_id for c in children)

    def test_mark_completed(self, temp_dir):
        """Test marking an agent as completed."""
        registry = AgentRegistry(base_dir=temp_dir)

        agent = MagicMock()
        agent.config.name = "test"
        instance_id = registry.register(agent)

        assert registry.get_instance(instance_id).status == "active"

        registry.mark_completed(instance_id)
        assert registry.get_instance(instance_id).status == "completed"

    def test_mark_failed(self, temp_dir):
        """Test marking an agent as failed."""
        registry = AgentRegistry(base_dir=temp_dir)

        agent = MagicMock()
        agent.config.name = "test"
        instance_id = registry.register(agent)

        registry.mark_failed(instance_id)
        assert registry.get_instance(instance_id).status == "failed"

    def test_cleanup(self, temp_dir):
        """Test cleaning up an agent."""
        registry = AgentRegistry(base_dir=temp_dir)

        agent = MagicMock()
        agent.config.name = "test"
        instance_id = registry.register(agent)

        assert registry.get_instance(instance_id) is not None

        registry.cleanup(instance_id)

        assert registry.get_instance(instance_id) is None
        assert registry.get_agent(instance_id) is None

    def test_cleanup_completed(self, temp_dir):
        """Test cleaning up all completed agents."""
        registry = AgentRegistry(base_dir=temp_dir)

        # Create some agents with different statuses
        for name, status in [
            ("active", "active"),
            ("completed1", "completed"),
            ("completed2", "completed"),
            ("failed", "failed"),
        ]:
            agent = MagicMock()
            agent.config.name = name
            iid = registry.register(agent)
            if status != "active":
                registry._instances[iid].status = status
                registry._save()

        assert len(registry.list_all()) == 4

        count = registry.cleanup_completed()
        assert count == 3  # completed1, completed2, failed

        remaining = registry.list_all()
        assert len(remaining) == 1
        assert remaining[0].name == "active"

    def test_persistence(self, temp_dir):
        """Test that registry persists to disk."""
        # Create and register
        registry1 = AgentRegistry(base_dir=temp_dir)
        agent = MagicMock()
        agent.config.name = "test"
        instance_id = registry1.register(agent)

        # Create new registry instance - should load from disk
        registry2 = AgentRegistry(base_dir=temp_dir)
        instance = registry2.get_instance(instance_id)

        assert instance is not None
        assert instance.name == "test"

    def test_list_active(self, temp_dir):
        """Test listing only active agents."""
        registry = AgentRegistry(base_dir=temp_dir)

        # Create agents
        for name in ["active1", "completed", "active2"]:
            agent = MagicMock()
            agent.config.name = name
            iid = registry.register(agent)
            if name == "completed":
                registry.mark_completed(iid)

        active = registry.list_active()
        assert len(active) == 2
        assert all(a.status == "active" for a in active)
