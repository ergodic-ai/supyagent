# Sprint 3: Interactive REPL

**Goal**: Build a full-featured interactive REPL with meta-commands and credential prompting.

**Duration**: ~3-4 days

**Depends on**: Sprint 2

---

## Deliverables

### 3.1 REPL Meta-Commands

Implement in-chat commands:

```python
# supyagent/cli/repl.py
from dataclasses import dataclass
from typing import Callable
import shlex

@dataclass
class Command:
    name: str
    aliases: list[str]
    description: str
    handler: Callable
    usage: str = ""

class REPLCommands:
    def __init__(self, agent, session_manager):
        self.agent = agent
        self.session_mgr = session_manager
        self.commands = self._register_commands()
    
    def _register_commands(self) -> dict[str, Command]:
        cmds = [
            Command("help", ["h", "?"], "Show available commands", self.cmd_help),
            Command("history", ["hist"], "Show conversation history", self.cmd_history, "[n]"),
            Command("tools", ["t"], "List available tools", self.cmd_tools),
            Command("sessions", ["sess"], "List all sessions", self.cmd_sessions),
            Command("switch", ["sw"], "Switch to another session", self.cmd_switch, "<session_id>"),
            Command("new", [], "Start a new session", self.cmd_new),
            Command("save", [], "Save session checkpoint", self.cmd_save, "[name]"),
            Command("export", [], "Export conversation to file", self.cmd_export, "[filename]"),
            Command("model", ["m"], "Show or change model", self.cmd_model, "[model]"),
            Command("clear", ["cls"], "Clear the screen", self.cmd_clear),
            Command("quit", ["exit", "q"], "Exit the chat", self.cmd_quit),
        ]
        
        registry = {}
        for cmd in cmds:
            registry[cmd.name] = cmd
            for alias in cmd.aliases:
                registry[alias] = cmd
        return registry
    
    def parse_and_execute(self, input_str: str) -> tuple[bool, str | None]:
        """
        Parse input and execute if it's a command.
        Returns (was_command, output_or_none)
        """
        if not input_str.startswith("/"):
            return False, None
        
        parts = shlex.split(input_str[1:])  # Remove leading /
        if not parts:
            return True, "Empty command"
        
        cmd_name = parts[0].lower()
        args = parts[1:]
        
        if cmd_name not in self.commands:
            return True, f"Unknown command: /{cmd_name}. Type /help for available commands."
        
        cmd = self.commands[cmd_name]
        try:
            result = cmd.handler(*args)
            return True, result
        except TypeError as e:
            return True, f"Usage: /{cmd.name} {cmd.usage}"
    
    def cmd_help(self) -> str:
        lines = ["Available commands:", ""]
        seen = set()
        for cmd in self.commands.values():
            if cmd.name in seen:
                continue
            seen.add(cmd.name)
            aliases = f" (/{', /'.join(cmd.aliases)})" if cmd.aliases else ""
            usage = f" {cmd.usage}" if cmd.usage else ""
            lines.append(f"  /{cmd.name}{usage}{aliases}")
            lines.append(f"      {cmd.description}")
        return "\n".join(lines)
    
    def cmd_history(self, n: str = "10") -> str:
        n = int(n)
        messages = self.agent.session.messages[-n:]
        lines = []
        for msg in messages:
            if msg.type == "user":
                lines.append(f"You: {msg.content}")
            elif msg.type == "assistant":
                lines.append(f"{self.agent.config.name}: {msg.content[:100]}...")
            elif msg.type == "tool_result":
                lines.append(f"[Tool: {msg.name}]")
        return "\n".join(lines) or "(no history)"
    
    def cmd_tools(self) -> str:
        lines = ["Available tools:", ""]
        for tool in self.agent.tools:
            lines.append(f"  {tool['function']['name']}")
            lines.append(f"      {tool['function'].get('description', 'No description')[:60]}")
        return "\n".join(lines) or "No tools available"
    
    def cmd_sessions(self) -> str:
        sessions = self.session_mgr.list_sessions(self.agent.config.name)
        if not sessions:
            return "No sessions found"
        
        current_id = self.agent.session.meta.session_id
        lines = ["Sessions:", ""]
        for s in sessions:
            marker = " <- current" if s.session_id == current_id else ""
            lines.append(f"  {s.session_id}: {s.title or '(untitled)'}{marker}")
        return "\n".join(lines)
    
    def cmd_switch(self, session_id: str) -> str:
        session = self.session_mgr.load_session(self.agent.config.name, session_id)
        if not session:
            return f"Session '{session_id}' not found"
        
        self.agent.session = session
        self.agent.messages = self.agent._reconstruct_messages(session)
        self.session_mgr._set_current(self.agent.config.name, session_id)
        return f"Switched to session {session_id}"
    
    def cmd_new(self) -> str:
        session = self.session_mgr.create_session(
            self.agent.config.name,
            self.agent.config.model.provider
        )
        self.agent.session = session
        self.agent.messages = [{"role": "system", "content": self.agent.config.system_prompt}]
        return f"Started new session {session.meta.session_id}"
    
    def cmd_save(self, name: str = None) -> str:
        # For now, sessions auto-save. This could create named checkpoints later.
        return f"Session {self.agent.session.meta.session_id} is auto-saved"
    
    def cmd_export(self, filename: str = None) -> str:
        if not filename:
            filename = f"{self.agent.config.name}_{self.agent.session.meta.session_id}.md"
        
        lines = [f"# Conversation with {self.agent.config.name}", ""]
        for msg in self.agent.session.messages:
            if msg.type == "user":
                lines.append(f"**You:** {msg.content}\n")
            elif msg.type == "assistant":
                lines.append(f"**{self.agent.config.name}:** {msg.content}\n")
        
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        return f"Exported to {filename}"
    
    def cmd_model(self, new_model: str = None) -> str:
        if new_model:
            self.agent.llm.model = new_model
            return f"Model changed to {new_model}"
        return f"Current model: {self.agent.llm.model}"
    
    def cmd_clear(self) -> str:
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        return ""
    
    def cmd_quit(self) -> str:
        raise SystemExit(0)
```

### 3.2 Credential Manager

Handle secure credential storage and prompting:

```python
# supyagent/core/credentials.py
from pathlib import Path
from cryptography.fernet import Fernet
import json
import os
import getpass

class CredentialManager:
    def __init__(self, base_dir: Path = Path(".supyagent/credentials")):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._fernet = self._get_fernet()
        self._cache: dict[str, dict[str, str]] = {}  # agent -> {name -> value}
    
    def _get_fernet(self) -> Fernet:
        """Get or create encryption key."""
        key_file = self.base_dir / ".key"
        if key_file.exists():
            key = key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            key_file.chmod(0o600)  # Owner read/write only
        return Fernet(key)
    
    def _cred_path(self, agent: str) -> Path:
        return self.base_dir / f"{agent}.enc"
    
    def _load_credentials(self, agent: str) -> dict[str, str]:
        """Load and decrypt credentials for an agent."""
        if agent in self._cache:
            return self._cache[agent]
        
        path = self._cred_path(agent)
        if not path.exists():
            self._cache[agent] = {}
            return {}
        
        encrypted = path.read_bytes()
        decrypted = self._fernet.decrypt(encrypted)
        creds = json.loads(decrypted)
        self._cache[agent] = creds
        return creds
    
    def _save_credentials(self, agent: str, creds: dict[str, str]):
        """Encrypt and save credentials."""
        encrypted = self._fernet.encrypt(json.dumps(creds).encode())
        self._cred_path(agent).write_bytes(encrypted)
        self._cache[agent] = creds
    
    def get(self, agent: str, name: str) -> str | None:
        """Get a credential value."""
        # First check environment
        if name in os.environ:
            return os.environ[name]
        # Then check stored credentials
        creds = self._load_credentials(agent)
        return creds.get(name)
    
    def set(self, agent: str, name: str, value: str, persist: bool = True):
        """Set a credential value."""
        if persist:
            creds = self._load_credentials(agent)
            creds[name] = value
            self._save_credentials(agent, creds)
        else:
            # Session-only: just set in environment
            os.environ[name] = value
    
    def has(self, agent: str, name: str) -> bool:
        """Check if a credential exists."""
        return self.get(agent, name) is not None
    
    def prompt_for_credential(
        self, 
        name: str, 
        description: str,
        persist_prompt: bool = True
    ) -> str | None:
        """
        Interactively prompt user for a credential.
        Returns the value or None if user skipped.
        """
        print()
        print(f"🔑 Credential requested: {name}")
        print(f"   Purpose: {description}")
        print()
        
        value = getpass.getpass(f"Enter value (or press Enter to skip): ")
        if not value:
            return None
        
        if persist_prompt:
            save = input("Save for future sessions? [Y/n]: ").strip().lower()
            return value, save != 'n'
        
        return value, False
    
    def list_credentials(self, agent: str) -> list[str]:
        """List stored credential names for an agent."""
        creds = self._load_credentials(agent)
        return list(creds.keys())
    
    def delete(self, agent: str, name: str):
        """Delete a stored credential."""
        creds = self._load_credentials(agent)
        if name in creds:
            del creds[name]
            self._save_credentials(agent, creds)
```

### 3.3 Credential Request Tool

Special tool that LLM can call to request credentials:

```python
# supyagent/core/tools.py (addition)

REQUEST_CREDENTIAL_TOOL = {
    "type": "function",
    "function": {
        "name": "request_credential",
        "description": "Request an API key, token, or other credential from the user. Use this when you need authentication to use a service.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The environment variable name (e.g., SLACK_API_TOKEN)"
                },
                "description": {
                    "type": "string", 
                    "description": "Explain to the user why this credential is needed"
                },
                "service": {
                    "type": "string",
                    "description": "The service this credential is for (e.g., 'Slack', 'GitHub')"
                }
            },
            "required": ["name", "description"]
        }
    }
}

def is_credential_request(tool_call) -> bool:
    """Check if a tool call is a credential request."""
    return tool_call.function.name == "request_credential"
```

### 3.4 Integrate into Agent

Handle credential requests in the agent loop:

```python
# supyagent/core/agent.py (updated send_message)

def send_message(self, content: str) -> str:
    """Send user message and get response, handling tool calls."""
    # ... (record user message)
    
    while True:
        response = self.llm.chat(self.messages, tools=self.tools)
        assistant_msg = response.choices[0].message
        
        # ... (record assistant message)
        
        if not assistant_msg.tool_calls:
            return assistant_msg.content
        
        for tool_call in assistant_msg.tool_calls:
            if is_credential_request(tool_call):
                result = self._handle_credential_request(tool_call)
            else:
                result = self._execute_tool_call(tool_call)
            
            # ... (record tool result)

def _handle_credential_request(self, tool_call) -> dict:
    """Handle a credential request from the LLM."""
    args = json.loads(tool_call.function.arguments)
    name = args["name"]
    description = args["description"]
    
    # Check if we already have it
    existing = self.credential_mgr.get(self.config.name, name)
    if existing:
        return {"ok": True, "message": f"Credential {name} is available"}
    
    # Prompt user
    result = self.credential_mgr.prompt_for_credential(name, description)
    if result is None:
        return {"ok": False, "error": "User declined to provide credential"}
    
    value, persist = result
    self.credential_mgr.set(self.config.name, name, value, persist=persist)
    
    # Record in session (without the actual value!)
    cred_msg = Message(
        type="credential_request",
        content=json.dumps({"name": name, "provided": True})
    )
    self.session_manager.append_message(self.session, cred_msg)
    
    return {"ok": True, "message": f"Credential {name} has been provided"}
```

### 3.5 Updated REPL Loop

Bring it all together:

```python
# supyagent/cli/chat.py
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from .repl import REPLCommands

console = Console()

def run_chat(agent: Agent, session_manager: SessionManager):
    """Main chat loop with REPL commands."""
    commands = REPLCommands(agent, session_manager)
    
    # Welcome message
    if agent.session.messages:
        console.print(f"[dim]Resuming session {agent.session.meta.session_id}[/dim]")
        console.print(f"[dim]({len(agent.session.messages)} messages in history)[/dim]\n")
    else:
        console.print(f"[dim]New session {agent.session.meta.session_id}[/dim]")
        console.print(f"[dim]Type /help for commands[/dim]\n")
    
    while True:
        try:
            user_input = Prompt.ask(f"[bold blue]You[/bold blue]")
            
            if not user_input.strip():
                continue
            
            # Check for commands
            is_cmd, cmd_output = commands.parse_and_execute(user_input)
            if is_cmd:
                if cmd_output:
                    console.print(cmd_output)
                continue
            
            # Send to agent
            with console.status("[bold green]Thinking..."):
                response = agent.send_message(user_input)
            
            # Display response
            console.print()
            console.print(f"[bold green]{agent.config.name}[/bold green]")
            console.print(Markdown(response))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/dim]")
        except SystemExit:
            console.print("[dim]Goodbye![/dim]")
            break
```

### 3.6 Credential REPL Commands

Add credential management commands:

```python
# Additional commands for REPLCommands

Command("creds", ["credentials"], "Manage credentials", self.cmd_creds, "[list|set|delete] [name]"),

def cmd_creds(self, action: str = "list", name: str = None) -> str:
    if action == "list":
        creds = self.agent.credential_mgr.list_credentials(self.agent.config.name)
        if not creds:
            return "No stored credentials"
        return "Stored credentials:\n" + "\n".join(f"  - {c}" for c in creds)
    
    elif action == "set" and name:
        result = self.agent.credential_mgr.prompt_for_credential(
            name, 
            "Manually setting credential"
        )
        if result:
            value, persist = result
            self.agent.credential_mgr.set(self.agent.config.name, name, value, persist)
            return f"Credential {name} saved"
        return "Cancelled"
    
    elif action == "delete" and name:
        self.agent.credential_mgr.delete(self.agent.config.name, name)
        return f"Credential {name} deleted"
    
    return "Usage: /creds [list|set|delete] [name]"
```

---

## Acceptance Criteria

1. **Meta-commands work**: All `/command` variations function correctly
2. **Credential prompting**: LLM can request and receive credentials
3. **Secure storage**: Credentials are encrypted on disk
4. **Password hiding**: Input is masked when entering credentials
5. **Persistence choice**: User can choose to save or not save credentials
6. **Rich output**: Using rich library for formatted output

---

## Test Scenarios

### Scenario 1: REPL Commands
```
You> /help
Available commands:
  /help (/h, /?)
      Show available commands
  /history [n] (/hist)
      Show conversation history
  ...

You> /tools
Available tools:
  web_search__search
      Search the web for information
  ...

You> /sessions
Sessions:
  abc123: What is Python... <- current
  def456: Help with Docker...
```

### Scenario 2: Credential Request
```
You> Search for Python tutorials on YouTube

🔑 Credential requested: YOUTUBE_API_KEY
   Purpose: To search YouTube via the YouTube Data API

Enter value (or press Enter to skip): ********
Save for future sessions? [Y/n]: y

researcher: I found several Python tutorials on YouTube...
```

### Scenario 3: Credential Reuse
```
# Second session, credential already stored
You> Find more YouTube videos about machine learning
researcher: Here are some ML tutorials I found...
# (No prompt - credential was already saved)
```

---

## Notes

- Credentials file should have restrictive permissions (600)
- Consider adding credential expiry in future
- The `request_credential` tool should be injected automatically, not defined in supypowers
- Session JSONL should NOT store actual credential values, only the fact that they were provided
