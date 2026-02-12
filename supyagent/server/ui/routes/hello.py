"""Hello wizard UI routes."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

router = APIRouter()


# ── Request models ──────────────────────────────────────────────────


class ServiceStartBody(BaseModel):
    base_url: str | None = None


class ServicePollBody(BaseModel):
    base_url: str
    device_code: str
    interval: int = 5
    timeout: int = 10


class IntegrationConnectBody(BaseModel):
    provider: str


class ImportEnvBody(BaseModel):
    key_names: list[str] = []


class ProfileInstallBody(BaseModel):
    profile: str
    model: str | None = None


class GoalsBody(BaseModel):
    goals: str


class SettingsBody(BaseModel):
    execution_mode: str = "yolo"
    heartbeat_enabled: bool = False
    heartbeat_interval: str = "5m"


class FinishBody(BaseModel):
    start_chat: bool = False
    agent_name: str = "assistant"


# ── HTML page ───────────────────────────────────────────────────────


@router.get("/hello", response_class=HTMLResponse)
async def hello_page():
    """Serve the hello wizard page."""
    from supyagent.server.ui.templates import render_template

    html = render_template("hello.html")
    return HTMLResponse(html)


# ── State ───────────────────────────────────────────────────────────


@router.get("/api/hello/state")
async def hello_state():
    """Detect current setup state (mirrors _detect_state from cli/hello.py)."""
    from supyagent.cli.hello import MODEL_PROVIDERS
    from supyagent.core.config import get_config_manager
    from supyagent.core.model_registry import get_model_registry
    from supyagent.core.service import SERVICE_API_KEY
    from supyagent.core.workspace import is_workspace_initialized

    config_mgr = get_config_manager()
    registry = get_model_registry()

    agents_dir = Path("agents")
    powers_dir = Path("powers")

    has_agents_dir = agents_dir.exists()
    has_tools = powers_dir.exists() and any(
        f for f in powers_dir.glob("*.py") if f.name != "__init__.py"
    )
    agent_yamls = [f.stem for f in agents_dir.glob("*.yaml")] if has_agents_dir else []

    service_key = config_mgr.get(SERVICE_API_KEY)

    llm_keys = {}
    env_keys = {}
    for provider_name, provider_info in MODEL_PROVIDERS.items():
        key_name = provider_info["key_name"]
        val = config_mgr.get(key_name)
        if val:
            llm_keys[key_name] = provider_name
        elif os.environ.get(key_name):
            env_keys[key_name] = provider_name

    return {
        "has_agents_dir": has_agents_dir,
        "has_tools": has_tools,
        "agent_yamls": agent_yamls,
        "service_connected": service_key is not None,
        "llm_keys": llm_keys,
        "env_keys": env_keys,
        "is_workspace": is_workspace_initialized(),
        "is_setup": has_agents_dir and has_tools,
        "has_models": len(registry.list_models()) > 0,
        "has_goals": Path("GOALS.md").exists(),
    }


# ── Step 1: Service Connection ──────────────────────────────────────


@router.post("/api/hello/service/start")
async def service_start(body: ServiceStartBody):
    """Start device code flow."""
    from supyagent.core.service import DEFAULT_SERVICE_URL, request_device_code

    base_url = body.base_url or DEFAULT_SERVICE_URL
    try:
        data = request_device_code(base_url)
        return {"ok": True, "base_url": base_url, **data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/hello/service/poll")
async def service_poll(body: ServicePollBody):
    """
    Single poll attempt for device code approval.

    Unlike the CLI which polls in a blocking loop, we do one attempt at a time
    and let the browser JS drive the loop via setInterval.
    """
    import httpx

    url = body.base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=body.timeout) as client:
            response = await client.post(
                f"{url}/api/v1/auth/device/token",
                json={"device_code": body.device_code},
            )

        if response.status_code == 200:
            data = response.json()
            api_key = data["api_key"]
            from supyagent.core.service import store_service_credentials

            store_service_credentials(api_key, body.base_url)
            return {"ok": True, "status": "approved", "api_key": api_key}
        elif response.status_code == 428:
            return {"ok": True, "status": "pending"}
        elif response.status_code == 403:
            return {"ok": False, "status": "denied", "error": "Authorization denied"}
        elif response.status_code == 410:
            return {"ok": False, "status": "expired", "error": "Device code expired"}
        else:
            return {"ok": False, "status": "error", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"ok": False, "status": "error", "error": str(e)}


# ── Step 1b: Integrations ──────────────────────────────────────────


@router.get("/api/hello/integrations")
async def integrations_list():
    """List integration providers and their connection status."""
    from supyagent.cli.hello import INTEGRATION_PROVIDERS
    from supyagent.core.config import get_config_manager
    from supyagent.core.service import SERVICE_API_KEY, ServiceClient

    config_mgr = get_config_manager()
    api_key = config_mgr.get(SERVICE_API_KEY)
    connected_providers: set[str] = set()

    if api_key:
        try:
            from supyagent.core.service import DEFAULT_SERVICE_URL, SERVICE_URL

            base_url = config_mgr.get(SERVICE_URL) or DEFAULT_SERVICE_URL
            client = ServiceClient(api_key=api_key, base_url=base_url)
            current = client.list_integrations()
            client.close()
            connected_providers = {i["provider"] for i in current}
        except Exception:
            pass

    result = []
    for provider_id, name, desc in INTEGRATION_PROVIDERS:
        result.append({
            "id": provider_id,
            "name": name,
            "description": desc,
            "connected": provider_id in connected_providers,
        })
    return {"integrations": result, "service_connected": api_key is not None}


@router.post("/api/hello/integrations/connect")
async def integration_connect(body: IntegrationConnectBody):
    """Get the OAuth URL for connecting a provider."""
    from supyagent.core.config import get_config_manager
    from supyagent.core.service import DEFAULT_SERVICE_URL, SERVICE_URL

    config_mgr = get_config_manager()
    base_url = config_mgr.get(SERVICE_URL) or DEFAULT_SERVICE_URL
    connect_url = f"{base_url}/integrations?connect={body.provider}"
    return {"ok": True, "url": connect_url}


# ── Step 2: Models (reuses /api/models/* and /api/keys/* routes) ──


@router.get("/api/hello/providers")
async def hello_providers():
    """List LLM providers with models and key status."""
    from supyagent.cli.hello import MODEL_PROVIDERS
    from supyagent.core.config import get_config_manager

    config_mgr = get_config_manager()
    result = []
    for name, info in MODEL_PROVIDERS.items():
        has_key = config_mgr.get(info["key_name"]) is not None
        result.append({
            "name": name,
            "key_name": info["key_name"],
            "has_key": has_key,
            "models": [{"id": m[0], "description": m[1]} for m in info["models"]],
        })
    return {"providers": result}


@router.post("/api/hello/models/import-env")
async def import_env_keys(body: ImportEnvBody):
    """Import API keys from environment variables."""
    from supyagent.cli.hello import MODEL_PROVIDERS
    from supyagent.core.config import get_config_manager

    config_mgr = get_config_manager()
    imported = []

    if body.key_names:
        keys_to_import = body.key_names
    else:
        keys_to_import = []
        for _name, info in MODEL_PROVIDERS.items():
            kn = info["key_name"]
            if not config_mgr.get(kn) and os.environ.get(kn):
                keys_to_import.append(kn)

    for key_name in keys_to_import:
        val = os.environ.get(key_name)
        if val:
            config_mgr.set(key_name, val)
            imported.append(key_name)

    return {"ok": True, "imported": imported}


# ── Step 3: Profiles ───────────────────────────────────────────────


@router.get("/api/hello/profiles")
async def hello_profiles():
    """List workspace profiles and available agent roles."""
    from supyagent.default_agents import AGENT_ROLES, WORKSPACE_PROFILES

    profiles = []
    for name, agent_list in WORKSPACE_PROFILES.items():
        descriptions = {
            "coding": "Assistant + Coder + Planner",
            "automation": "Assistant + Writer",
            "full": "All agents (Assistant + Coder + Planner + Writer)",
        }
        profiles.append({
            "name": name,
            "description": descriptions.get(name, ", ".join(agent_list)),
            "agents": agent_list,
        })

    roles = []
    for role, meta in AGENT_ROLES.items():
        roles.append({
            "role": role,
            "description": meta["description"],
        })

    return {"profiles": profiles, "roles": roles}


@router.post("/api/hello/profile/install")
async def install_profile(body: ProfileInstallBody):
    """Install agents and tools for a workspace profile."""
    from supyagent.default_agents import WORKSPACE_PROFILES, install_workspace_agents
    from supyagent.default_tools import install_default_tools

    # Install tools
    tools_path = Path("powers")
    tools_count = 0
    if not (tools_path.exists() and any(
        f for f in tools_path.glob("*.py") if f.name != "__init__.py"
    )):
        tools_count = install_default_tools(tools_path)

    # Install agents
    agents_dir = Path("agents")
    agents_dir.mkdir(parents=True, exist_ok=True)

    profile = body.profile
    if profile not in WORKSPACE_PROFILES:
        profile = "coding"

    paths = install_workspace_agents(profile, agents_dir, model=body.model)
    installed = [p.stem for p in paths] if paths else list(WORKSPACE_PROFILES[profile])

    return {
        "ok": True,
        "profile": profile,
        "tools_installed": tools_count,
        "agents": installed,
    }


# ── Step 4: Goals ──────────────────────────────────────────────────


@router.post("/api/hello/goals")
async def save_goals(body: GoalsBody):
    """Create or update GOALS.md."""
    from supyagent.core.workspace import create_goals_file

    goals_text = body.goals.strip()
    if goals_text:
        lines = goals_text.split("\n")
        formatted = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("- "):
                line = f"- {line}"
            formatted.append(line)
        goals_text = "\n".join(formatted)

    create_goals_file(goals_text)
    return {"ok": True}


# ── Step 5: Settings ──────────────────────────────────────────────


@router.post("/api/hello/settings")
async def save_settings(body: SettingsBody):
    """Save workspace settings (execution mode, heartbeat)."""
    from supyagent.core.workspace import (
        ExecutionConfig,
        HeartbeatConfig,
        WorkspaceConfig,
        save_workspace,
    )

    save_workspace(
        WorkspaceConfig(
            name=Path.cwd().name,
            profile="coding",
            execution=ExecutionConfig(mode=body.execution_mode),
            heartbeat=HeartbeatConfig(
                enabled=body.heartbeat_enabled,
                interval=body.heartbeat_interval,
            ),
        )
    )
    return {"ok": True}


# ── Finish ─────────────────────────────────────────────────────────


@router.post("/api/hello/finish")
async def finish_wizard(request: Request, body: FinishBody):
    """Finalize the wizard and signal done."""
    callback = request.app.state.done_callback
    if callback:
        callback({
            "action": "hello_done",
            "start_chat": body.start_chat,
            "agent_name": body.agent_name,
        })
    return {"ok": True}
