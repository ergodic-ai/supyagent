# Sprint 5: Multi-Agent Architecture

**Goal**: Enable agents to create and invoke other agents, with a central planning agent for orchestration.

**Duration**: ~4-5 days

**Depends on**: Sprint 4

---

## Deliverables

### 5.1 Agent Registry

Track active agent instances:

```python
# supyagent/core/registry.py
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

@dataclass
class AgentInstance:
    """A running agent instance."""
    name: str
    instance_id: str
    created_at: datetime
    parent_id: str | None = None  # If spawned by another agent
    status: str = "active"  # active, completed, failed

class AgentRegistry:
    """
    Manages agent instances and their relationships.
    Enables agents to spawn and communicate with sub-agents.
    """
    
    def __init__(self, base_dir: Path = Path(".supyagent")):
        self.base_dir = base_dir
        self.registry_path = base_dir / "registry.json"
        self._instances: dict[str, AgentInstance] = {}
        self._agents: dict[str, "Agent"] = {}  # Live agent objects
        self._load()
    
    def _load(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                for item in data.get("instances", []):
                    inst = AgentInstance(**item)
                    self._instances[inst.instance_id] = inst
    
    def _save(self):
        """Persist registry to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump({
                "instances": [
                    {
                        "name": i.name,
                        "instance_id": i.instance_id,
                        "created_at": i.created_at.isoformat(),
                        "parent_id": i.parent_id,
                        "status": i.status
                    }
                    for i in self._instances.values()
                ]
            }, f, indent=2)
    
    def register(self, agent: "Agent", parent_id: str = None) -> str:
        """Register an agent instance and return its ID."""
        import uuid
        instance_id = str(uuid.uuid4())[:8]
        
        instance = AgentInstance(
            name=agent.config.name,
            instance_id=instance_id,
            created_at=datetime.utcnow(),
            parent_id=parent_id
        )
        
        self._instances[instance_id] = instance
        self._agents[instance_id] = agent
        self._save()
        
        return instance_id
    
    def get_agent(self, instance_id: str) -> "Agent | None":
        """Get a live agent by instance ID."""
        return self._agents.get(instance_id)
    
    def get_instance(self, instance_id: str) -> AgentInstance | None:
        """Get instance metadata."""
        return self._instances.get(instance_id)
    
    def list_children(self, parent_id: str) -> list[AgentInstance]:
        """List all agents spawned by a parent."""
        return [i for i in self._instances.values() if i.parent_id == parent_id]
    
    def mark_completed(self, instance_id: str):
        """Mark an agent as completed."""
        if instance_id in self._instances:
            self._instances[instance_id].status = "completed"
            self._save()
    
    def cleanup(self, instance_id: str):
        """Remove an agent instance."""
        if instance_id in self._agents:
            del self._agents[instance_id]
        if instance_id in self._instances:
            del self._instances[instance_id]
            self._save()
```

### 5.2 Delegation Tools

Tools that allow agents to invoke other agents:

```python
# supyagent/core/delegation.py
from .registry import AgentRegistry
from .agent import Agent
from .executor import ExecutionRunner
from ..models.agent_config import AgentConfig, load_agent_config
import json

class DelegationManager:
    """
    Manages agent-to-agent delegation.
    """
    
    def __init__(self, registry: AgentRegistry, parent_agent: Agent):
        self.registry = registry
        self.parent = parent_agent
        self.parent_id = registry.register(parent_agent)
    
    def get_delegation_tools(self) -> list[dict]:
        """
        Generate tool schemas for each delegatable agent.
        """
        tools = []
        
        for delegate_name in self.parent.config.delegates:
            try:
                delegate_config = load_agent_config(delegate_name)
            except FileNotFoundError:
                continue
            
            tool = {
                "type": "function",
                "function": {
                    "name": f"delegate_to_{delegate_name}",
                    "description": f"Delegate a task to the {delegate_name} agent. {delegate_config.description}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task to delegate to this agent"
                            },
                            "context": {
                                "type": "string",
                                "description": "Optional context from the current conversation to pass along"
                            },
                            "wait_for_result": {
                                "type": "boolean",
                                "description": "Whether to wait for the result (true) or fire-and-forget (false)",
                                "default": True
                            }
                        },
                        "required": ["task"]
                    }
                }
            }
            tools.append(tool)
        
        # Also add a generic spawn tool for creating new agents
        tools.append({
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": "Create a new agent instance for a specific purpose",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_type": {
                            "type": "string",
                            "description": "The type of agent to spawn (must be in delegates list)"
                        },
                        "task": {
                            "type": "string",
                            "description": "The initial task for the agent"
                        }
                    },
                    "required": ["agent_type", "task"]
                }
            }
        })
        
        return tools
    
    def execute_delegation(self, tool_call) -> dict:
        """
        Execute a delegation tool call.
        """
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        if name == "spawn_agent":
            return self._spawn_agent(args["agent_type"], args["task"])
        
        if name.startswith("delegate_to_"):
            agent_name = name[len("delegate_to_"):]
            return self._delegate_task(
                agent_name,
                args["task"],
                args.get("context"),
                args.get("wait_for_result", True)
            )
        
        return {"ok": False, "error": f"Unknown delegation tool: {name}"}
    
    def _delegate_task(
        self, 
        agent_name: str, 
        task: str, 
        context: str = None,
        wait: bool = True
    ) -> dict:
        """
        Delegate a task to another agent.
        """
        try:
            config = load_agent_config(agent_name)
        except FileNotFoundError:
            return {"ok": False, "error": f"Agent '{agent_name}' not found"}
        
        # Build the task with context
        full_task = task
        if context:
            full_task = f"Context from parent task:\n{context}\n\nTask:\n{task}"
        
        if config.type == "execution":
            # Use execution runner for execution agents
            runner = ExecutionRunner(config)
            result = runner.run(full_task, output_format="json")
        else:
            # For interactive agents, create a new instance and run synchronously
            sub_agent = Agent(config)
            self.registry.register(sub_agent, parent_id=self.parent_id)
            
            response = sub_agent.send_message(full_task)
            result = {"ok": True, "data": response}
            
            self.registry.mark_completed(sub_agent.instance_id)
        
        return result
    
    def _spawn_agent(self, agent_type: str, task: str) -> dict:
        """
        Spawn a new agent instance.
        """
        if agent_type not in self.parent.config.delegates:
            return {"ok": False, "error": f"Cannot spawn '{agent_type}' - not in delegates list"}
        
        return self._delegate_task(agent_type, task, wait=True)
```

### 5.3 Planning Agent Template

A built-in orchestrator agent:

```yaml
# agents/planner.yaml
name: planner
description: Orchestrates complex tasks by breaking them down and delegating to specialist agents
type: interactive
version: "1.0"

model:
  provider: anthropic/claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 4096

system_prompt: |
  You are a Planning Agent - an intelligent orchestrator that breaks down complex tasks
  and delegates them to specialist agents.
  
  ## Your Capabilities
  
  You can delegate tasks to these specialist agents:
  - **researcher**: Finding, analyzing, and summarizing information
  - **coder**: Writing, reviewing, and debugging code
  - **writer**: Creating written content, documentation, emails
  
  ## Your Process
  
  When given a task:
  1. **Analyze** - Understand what needs to be done
  2. **Plan** - Break into subtasks and identify which agents are needed
  3. **Delegate** - Send subtasks to appropriate agents
  4. **Synthesize** - Combine results into a coherent response
  
  ## Guidelines
  
  - Always explain your plan before executing
  - Delegate specific, well-scoped tasks
  - Provide relevant context when delegating
  - If a delegation fails, try an alternative approach
  - Synthesize results, don't just concatenate them
  
  ## Example Workflow
  
  User: "Create a Python library for data validation"
  
  Your plan:
  1. Research existing validation libraries (delegate to researcher)
  2. Design the API based on research (you)
  3. Write the core implementation (delegate to coder)
  4. Write documentation (delegate to writer)
  5. Review and synthesize everything

tools:
  allow:
    - "*"  # Can use any tools directly if needed

delegates:
  - researcher
  - coder
  - writer

limits:
  max_tool_calls_per_turn: 20
  max_total_tool_calls: 100
```

### 5.4 Integrate Delegation into Agent

Update Agent class to support delegation:

```python
# supyagent/core/agent.py (updated)

class Agent:
    def __init__(self, config: AgentConfig, session: Session = None, registry: AgentRegistry = None):
        self.config = config
        self.llm = LLMClient(config.model.provider, config.model.temperature)
        self.session_manager = SessionManager()
        self.credential_mgr = CredentialManager()
        
        # Set up registry and delegation
        self.registry = registry or AgentRegistry()
        self.instance_id = None
        
        if config.delegates:
            self.delegation_mgr = DelegationManager(self.registry, self)
            self.instance_id = self.delegation_mgr.parent_id
        else:
            self.delegation_mgr = None
        
        # Load tools (including delegation tools)
        self.tools = self._load_tools()
        
        # ... rest of init
    
    def _load_tools(self) -> list[dict]:
        """Load all available tools including delegation tools."""
        tools = []
        
        # Supypowers tools
        sp_tools = discover_tools()
        tools.extend(tools_to_openai_schema(sp_tools, self.config.tools))
        
        # Credential request tool
        tools.append(REQUEST_CREDENTIAL_TOOL)
        
        # Delegation tools (if this agent can delegate)
        if self.delegation_mgr:
            tools.extend(self.delegation_mgr.get_delegation_tools())
        
        return tools
    
    def _execute_tool_call(self, tool_call) -> dict:
        """Execute a tool call, handling different tool types."""
        name = tool_call.function.name
        
        # Credential request
        if name == "request_credential":
            return self._handle_credential_request(tool_call)
        
        # Delegation tools
        if name.startswith("delegate_to_") or name == "spawn_agent":
            return self.delegation_mgr.execute_delegation(tool_call)
        
        # Regular supypowers tool
        script, func = name.split("__")
        args = json.loads(tool_call.function.arguments)
        return execute_tool(script, func, args, self._get_secrets())
    
    def _get_secrets(self) -> dict:
        """Get all available secrets for tool execution."""
        secrets = {}
        for cred_name in self.credential_mgr.list_credentials(self.config.name):
            value = self.credential_mgr.get(self.config.name, cred_name)
            if value:
                secrets[cred_name] = value
        return secrets
```

### 5.5 Context Passing

Enable rich context passing between agents:

```python
# supyagent/core/context.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DelegationContext:
    """
    Context passed from parent to child agent.
    """
    parent_agent: str
    parent_task: str
    conversation_summary: str | None = None
    relevant_facts: list[str] = field(default_factory=list)
    shared_data: dict[str, Any] = field(default_factory=dict)
    
    def to_prompt(self) -> str:
        """Convert context to a prompt prefix."""
        parts = [
            f"You are being called by the '{self.parent_agent}' agent.",
            f"Parent's current task: {self.parent_task}",
        ]
        
        if self.conversation_summary:
            parts.append(f"\nConversation context:\n{self.conversation_summary}")
        
        if self.relevant_facts:
            parts.append("\nRelevant information:")
            for fact in self.relevant_facts:
                parts.append(f"- {fact}")
        
        return "\n".join(parts)


def summarize_conversation(messages: list[dict], llm: LLMClient) -> str:
    """
    Use LLM to create a summary of the conversation for context passing.
    """
    # Extract just user and assistant messages
    conversation = []
    for msg in messages[-10:]:  # Last 10 messages
        if msg.get("role") in ("user", "assistant"):
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg.get("content", "")[:200]
            conversation.append(f"{role}: {content}")
    
    if not conversation:
        return None
    
    summary_prompt = f"""Summarize this conversation in 2-3 sentences, focusing on the main task and key decisions:

{chr(10).join(conversation)}

Summary:"""
    
    response = llm.chat([{"role": "user", "content": summary_prompt}])
    return response.choices[0].message.content
```

### 5.6 Multi-Agent CLI Commands

Commands for managing multi-agent workflows:

```python
# supyagent/cli/main.py (additions)

@cli.command()
def agents():
    """List all active agent instances."""
    registry = AgentRegistry()
    
    instances = list(registry._instances.values())
    if not instances:
        click.echo("No active agents")
        return
    
    for inst in instances:
        parent = f" (child of {inst.parent_id})" if inst.parent_id else ""
        click.echo(f"  {inst.instance_id}: {inst.name} [{inst.status}]{parent}")
        click.echo(f"    Created: {inst.created_at.strftime('%Y-%m-%d %H:%M')}")


@cli.command()
@click.argument("task")
@click.option("--planner", default="planner", help="Planning agent to use")
def plan(task: str, planner: str):
    """
    Run a task through the planning agent for orchestration.
    
    Example:
        supyagent plan "Build a web scraper for news articles"
    """
    config = load_agent_config(planner)
    if not config.delegates:
        click.echo(f"Agent '{planner}' has no delegates configured", err=True)
        return
    
    agent = Agent(config)
    
    click.echo(f"Planning agent: {planner}")
    click.echo(f"Available delegates: {', '.join(config.delegates)}")
    click.echo(f"Task: {task}\n")
    click.echo("-" * 40)
    
    response = agent.send_message(task)
    click.echo(response)
```

---

## Acceptance Criteria

1. **Delegation works**: Agent can successfully delegate to sub-agents
2. **Registry tracks**: All agent instances are tracked in registry
3. **Context passes**: Parent context is available to child agents
4. **Execution agents**: Can delegate to both interactive and execution agents
5. **Planning agent**: Built-in planner can orchestrate multi-step tasks
6. **Recursive delegation**: Sub-agents can delegate further (with depth limits)

---

## Test Scenarios

### Scenario 1: Simple Delegation
```
You> Research the latest developments in quantum computing

planner: I'll delegate this research task to the researcher agent.

[Delegating to researcher: "Find and summarize the latest developments in quantum computing from the past year"]

researcher: Based on my research, here are the key developments...

planner: Based on the research, here's a summary of quantum computing developments:
1. ...
2. ...
```

### Scenario 2: Multi-Agent Workflow
```
You> Create a Python script to analyze CSV files with documentation

planner: This task requires multiple specialists. Here's my plan:
1. Research existing CSV analysis patterns (researcher)
2. Write the Python script (coder)
3. Create documentation (writer)

Executing plan...

[Step 1: Delegating to researcher]
[Step 2: Delegating to coder]
[Step 3: Delegating to writer]

Here's your complete deliverable:

## Script
```python
...
```

## Documentation
...
```

### Scenario 3: Agent Listing
```bash
$ supyagent agents
  abc123: planner [active]
    Created: 2024-01-15 10:00
  def456: researcher [completed] (child of abc123)
    Created: 2024-01-15 10:01
  ghi789: coder [active] (child of abc123)
    Created: 2024-01-15 10:02
```

---

## Architecture Diagram

```
                    ┌─────────────┐
                    │    User     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Planner   │
                    │   Agent     │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │ Researcher  │ │   Coder     │ │   Writer    │
    │   Agent     │ │   Agent     │ │   Agent     │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │ supypowers  │ │ supypowers  │ │ supypowers  │
    │   tools     │ │   tools     │ │   tools     │
    └─────────────┘ └─────────────┘ └─────────────┘
```

---

## Notes

- Add depth limit to prevent infinite delegation loops
- Consider async execution for parallel sub-agent tasks
- Child agents should inherit relevant credentials from parent
- May need rate limiting for sub-agent LLM calls
- Consider adding "agent memory" for cross-session learning
