# Sprint 4: Execution Mode

**Goal**: Implement non-interactive "execution agents" for automation and pipelines.

**Duration**: ~2-3 days

**Depends on**: Sprint 3

---

## Deliverables

### 4.1 Execution Agent Type

Execution agents are stateless, non-interactive, and designed for automation:

```yaml
# agents/summarizer.yaml
name: summarizer
description: Summarizes text content
type: execution  # Key difference

model:
  provider: anthropic/claude-3-5-sonnet-20241022
  temperature: 0.3  # Lower for consistency
  max_tokens: 2048

system_prompt: |
  You are a summarization tool. Given input text, produce a concise summary.
  Be factual and preserve key information.
  Do not engage in conversation - just output the summary.

# Input/output schema for execution mode
input_schema:
  type: object
  properties:
    text:
      type: string
      description: The text to summarize
    max_length:
      type: integer
      description: Maximum words in summary
      default: 200
  required: [text]

output_schema:
  type: object
  properties:
    summary:
      type: string
    word_count:
      type: integer
    key_points:
      type: array
      items:
        type: string

tools:
  allow: []  # Execution agents often don't need tools
```

### 4.2 Execution Runner

Non-interactive execution logic:

```python
# supyagent/core/executor.py
from .agent import Agent
from .llm import LLMClient
from ..models.agent_config import AgentConfig
import json

class ExecutionRunner:
    """
    Runs execution-type agents in a non-interactive, input->output fashion.
    """
    
    def __init__(self, config: AgentConfig):
        if config.type != "execution":
            raise ValueError(f"Agent '{config.name}' is not an execution agent")
        
        self.config = config
        self.llm = LLMClient(config.model.provider, config.model.temperature)
        self.tools = self._load_tools() if config.tools.allow else []
    
    def run(
        self, 
        task: str | dict,
        secrets: dict[str, str] = None,
        output_format: str = "raw"
    ) -> dict:
        """
        Execute a task and return the result.
        
        Args:
            task: Either a string task description or a dict matching input_schema
            secrets: Pre-provided credentials (no prompting in execution mode)
            output_format: "raw" | "json" | "markdown"
        
        Returns:
            {"ok": True, "data": ...} or {"ok": False, "error": ...}
        """
        # Inject secrets into environment
        if secrets:
            for k, v in secrets.items():
                os.environ[k] = v
        
        try:
            # Build the prompt
            if isinstance(task, dict):
                user_content = self._format_structured_input(task)
            else:
                user_content = task
            
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # Run with tool loop (max iterations for safety)
            max_iterations = self.config.limits.get("max_tool_calls", 20) if hasattr(self.config, 'limits') else 20
            iterations = 0
            
            while iterations < max_iterations:
                response = self.llm.chat(messages, tools=self.tools or None)
                assistant_msg = response.choices[0].message
                messages.append(assistant_msg.model_dump())
                
                if not assistant_msg.tool_calls:
                    # Done - format and return
                    return self._format_output(assistant_msg.content, output_format)
                
                # Execute tools
                for tool_call in assistant_msg.tool_calls:
                    # No credential prompting in execution mode!
                    if tool_call.function.name == "request_credential":
                        return {
                            "ok": False, 
                            "error": f"Credential required but not provided: {tool_call.function.arguments}"
                        }
                    
                    result = self._execute_tool(tool_call, secrets)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                iterations += 1
            
            return {"ok": False, "error": "Max tool iterations exceeded"}
            
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def _format_structured_input(self, task: dict) -> str:
        """Format a structured input dict into a prompt."""
        # Validate against input_schema if defined
        if hasattr(self.config, 'input_schema'):
            # Could use jsonschema validation here
            pass
        
        return json.dumps(task, indent=2)
    
    def _format_output(self, content: str, format: str) -> dict:
        """Format the output according to requested format."""
        if format == "json":
            try:
                # Try to parse as JSON
                data = json.loads(content)
                return {"ok": True, "data": data}
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                import re
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        return {"ok": True, "data": data}
                    except:
                        pass
                return {"ok": True, "data": content}
        
        elif format == "markdown":
            return {"ok": True, "data": content, "format": "markdown"}
        
        else:  # raw
            return {"ok": True, "data": content}
    
    def _execute_tool(self, tool_call, secrets: dict = None) -> dict:
        """Execute a tool call."""
        name = tool_call.function.name
        script, func = name.split("__")
        args = json.loads(tool_call.function.arguments)
        return execute_tool(script, func, args, secrets)
```

### 4.3 CLI Run Command

Command-line interface for execution mode:

```python
# supyagent/cli/main.py (additions)

@cli.command()
@click.argument("agent_name")
@click.argument("task", required=False)
@click.option("--input", "-i", "input_file", type=click.Path(exists=True), help="Read task from file")
@click.option("--output", "-o", "output_format", type=click.Choice(["raw", "json", "markdown"]), default="raw")
@click.option("--secrets", "-s", multiple=True, help="Secrets as KEY=VALUE or path to .env file")
@click.option("--quiet", "-q", is_flag=True, help="Only output the result, no status messages")
def run(agent_name: str, task: str, input_file: str, output_format: str, secrets: tuple, quiet: bool):
    """
    Run an agent in execution mode (non-interactive).
    
    Examples:
        supyagent run summarizer "Summarize this text..."
        supyagent run summarizer --input document.txt --output json
        supyagent run api-caller '{"endpoint": "/users"}' --secrets API_KEY=xxx
        echo "text to process" | supyagent run summarizer --input -
    """
    config = load_agent_config(agent_name)
    
    # Execution agents only
    if config.type != "execution":
        if not quiet:
            click.echo(f"Warning: '{agent_name}' is an interactive agent. Consider using 'chat' instead.", err=True)
    
    # Parse secrets
    secrets_dict = parse_secrets(secrets)
    
    # Get task content
    if input_file:
        if input_file == "-":
            task_content = sys.stdin.read()
        else:
            with open(input_file) as f:
                task_content = f.read()
    elif task:
        # Try to parse as JSON, otherwise use as string
        try:
            task_content = json.loads(task)
        except json.JSONDecodeError:
            task_content = task
    else:
        # Read from stdin if no task provided
        if not sys.stdin.isatty():
            task_content = sys.stdin.read()
        else:
            click.echo("Error: No task provided. Use positional argument, --input, or pipe to stdin.", err=True)
            sys.exit(1)
    
    # Run
    runner = ExecutionRunner(config)
    
    if not quiet:
        click.echo(f"Running {agent_name}...", err=True)
    
    result = runner.run(task_content, secrets=secrets_dict, output_format=output_format)
    
    # Output
    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    elif result["ok"]:
        click.echo(result["data"])
    else:
        click.echo(f"Error: {result['error']}", err=True)
        sys.exit(1)


def parse_secrets(secrets: tuple) -> dict:
    """Parse secrets from KEY=VALUE pairs or .env files."""
    result = {}
    for secret in secrets:
        if "=" in secret:
            key, value = secret.split("=", 1)
            result[key] = value
        elif os.path.isfile(secret):
            # Parse as .env file
            with open(secret) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        result[key.strip()] = value.strip()
    return result
```

### 4.4 Batch Processing

Support for processing multiple inputs:

```python
# supyagent/cli/main.py (addition)

@cli.command()
@click.argument("agent_name")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", "output_file", type=click.Path(), help="Output file (default: stdout)")
@click.option("--format", "-f", "input_format", type=click.Choice(["jsonl", "csv"]), default="jsonl")
@click.option("--secrets", "-s", multiple=True)
@click.option("--parallel", "-p", type=int, default=1, help="Number of parallel workers")
def batch(agent_name: str, input_file: str, output_file: str, input_format: str, secrets: tuple, parallel: int):
    """
    Run an agent on multiple inputs from a file.
    
    Input formats:
        - jsonl: One JSON object per line
        - csv: CSV with headers, each row becomes a dict
    
    Example:
        supyagent batch summarizer inputs.jsonl --output results.jsonl
    """
    config = load_agent_config(agent_name)
    secrets_dict = parse_secrets(secrets)
    
    # Load inputs
    inputs = []
    if input_format == "jsonl":
        with open(input_file) as f:
            for line in f:
                inputs.append(json.loads(line))
    elif input_format == "csv":
        import csv
        with open(input_file) as f:
            reader = csv.DictReader(f)
            inputs = list(reader)
    
    # Process
    results = []
    runner = ExecutionRunner(config)
    
    with click.progressbar(inputs, label=f"Processing {len(inputs)} items") as items:
        for item in items:
            result = runner.run(item, secrets=secrets_dict, output_format="json")
            results.append(result)
    
    # Output
    output = "\n".join(json.dumps(r) for r in results)
    if output_file:
        with open(output_file, "w") as f:
            f.write(output)
        click.echo(f"Results written to {output_file}")
    else:
        click.echo(output)
```

### 4.5 Execution Agent Templates

Pre-built templates for common patterns:

```yaml
# agents/_templates/execution_basic.yaml
name: ${NAME}
description: ${DESCRIPTION}
type: execution

model:
  provider: anthropic/claude-3-5-sonnet-20241022
  temperature: 0.3
  max_tokens: 2048

system_prompt: |
  You are a tool that ${PURPOSE}.
  
  Instructions:
  - Process the input according to the task
  - Output only the result, no conversation
  - Be precise and factual
  
  ${ADDITIONAL_INSTRUCTIONS}

input_schema:
  type: object
  properties:
    input:
      type: string
      description: The input to process
  required: [input]

tools:
  allow: []
```

```yaml
# agents/_templates/execution_with_tools.yaml
name: ${NAME}
description: ${DESCRIPTION}
type: execution

model:
  provider: anthropic/claude-3-5-sonnet-20241022
  temperature: 0.5
  max_tokens: 4096

system_prompt: |
  You are a tool that ${PURPOSE}.
  
  You have access to tools to help accomplish the task.
  Use tools as needed, then provide the final result.

tools:
  allow:
    - ${TOOL_1}
    - ${TOOL_2}

limits:
  max_tool_calls: 10
```

---

## Acceptance Criteria

1. **Execution mode works**: `supyagent run` executes and returns results
2. **No interactivity**: Never prompts user for input during execution
3. **Secrets pre-provided**: Credentials must be passed via `--secrets`
4. **JSON output**: Can output structured JSON with `--output json`
5. **Stdin support**: Can pipe input from stdin
6. **Batch processing**: Can process multiple inputs from JSONL/CSV
7. **Exit codes**: Returns 0 on success, non-zero on failure

---

## Test Scenarios

### Scenario 1: Simple Execution
```bash
$ supyagent run summarizer "The quick brown fox jumps over the lazy dog. This is a simple sentence used for testing."

A test sentence about a fox jumping over a dog.
```

### Scenario 2: JSON Input/Output
```bash
$ supyagent run summarizer '{"text": "Long article...", "max_length": 50}' --output json

{
  "ok": true,
  "data": {
    "summary": "Article discusses...",
    "word_count": 45,
    "key_points": ["Point 1", "Point 2"]
  }
}
```

### Scenario 3: File Input
```bash
$ supyagent run summarizer --input article.txt --output markdown

## Summary

The article discusses...
```

### Scenario 4: Pipeline
```bash
$ cat document.txt | supyagent run summarizer | supyagent run translator '{"target": "spanish"}'
```

### Scenario 5: Batch Processing
```bash
$ cat inputs.jsonl
{"text": "First document..."}
{"text": "Second document..."}
{"text": "Third document..."}

$ supyagent batch summarizer inputs.jsonl --output results.jsonl
Processing 3 items [████████████████████████████████] 100%
Results written to results.jsonl
```

### Scenario 6: Missing Credentials (Failure)
```bash
$ supyagent run api-caller '{"endpoint": "/users"}'

{
  "ok": false,
  "error": "Credential required but not provided: {\"name\": \"API_KEY\"}"
}
$ echo $?
1
```

---

## Notes

- Execution agents should be deterministic when possible (lower temperature)
- Consider adding timeout for long-running executions
- Parallel batch processing can be added later (with rate limiting)
- Output validation against `output_schema` is optional but helpful for pipelines
- Consider adding `--dry-run` flag to show what would be executed
