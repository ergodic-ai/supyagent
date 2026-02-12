"""Model and API key management UI routes."""

from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

router = APIRouter()


# ── Request models ──────────────────────────────────────────────────

class ModelBody(BaseModel):
    model: str
    key_name: str | None = None
    key_value: str | None = None


class RoleBody(BaseModel):
    role: str
    model: str


class UnassignRoleBody(BaseModel):
    role: str


class KeySetBody(BaseModel):
    key_name: str
    value: str


class VerifyBody(BaseModel):
    model: str


class KeyDeleteBody(BaseModel):
    key_name: str


# ── HTML page ───────────────────────────────────────────────────────

@router.get("/models", response_class=HTMLResponse)
async def models_page():
    """Serve the models management page."""
    from supyagent.server.ui.templates import render_template

    html = render_template("models.html")
    return HTMLResponse(html)


# ── State endpoint ──────────────────────────────────────────────────

@router.get("/api/models/state")
async def models_state():
    """Full model state: registered models, default, roles, key status."""
    from supyagent.core.config import KNOWN_LLM_KEYS, get_config_manager
    from supyagent.core.model_registry import STANDARD_ROLES, get_model_registry

    registry = get_model_registry()
    config_mgr = get_config_manager()
    config_mgr.load_into_environment()
    data = registry.summary()

    models_info = []
    for model_str in data["registered"]:
        provider_name = registry.detect_provider_name(model_str) or "Unknown"
        has_key = registry.check_api_key(model_str)
        key_name = registry.detect_provider_key(model_str)
        missing_keys = _get_missing_keys(model_str)
        roles_for = [r for r, m in data["roles"].items() if m == model_str]
        models_info.append({
            "model": model_str,
            "provider": provider_name,
            "key_name": key_name,
            "has_key": has_key,
            "missing_keys": missing_keys,
            "is_default": model_str == data["default"],
            "roles": roles_for,
        })

    # Provider key status
    providers = []
    for key_name, description in KNOWN_LLM_KEYS.items():
        has_stored = key_name in config_mgr.list_keys()
        has_env = key_name in os.environ
        providers.append({
            "key_name": key_name,
            "description": description,
            "configured": has_stored or has_env,
            "source": "env" if has_env else ("stored" if has_stored else None),
        })

    return {
        "default": data["default"],
        "roles": data["roles"],
        "standard_roles": list(STANDARD_ROLES),
        "models": models_info,
        "providers": providers,
    }


# ── Verify endpoint ────────────────────────────────────────────────

def _get_missing_keys(model: str) -> list[str]:
    """Use litellm.validate_environment to detect missing env vars for a model.

    Falls back to empty list if litellm is unavailable or the model is unknown.
    """
    try:
        import litellm
        result = litellm.validate_environment(model)
        if result and result.get("missing_keys"):
            return [k for k in result["missing_keys"] if k]
        return []
    except Exception:
        return []


@router.post("/api/models/verify")
async def models_verify(body: VerifyBody):
    """Verify a model by checking env vars and pinging with a minimal completion."""
    from supyagent.core.config import get_config_manager

    # Load stored keys into environment so LiteLLM can see them
    config_mgr = get_config_manager()
    config_mgr.load_into_environment()

    missing = _get_missing_keys(body.model)
    if missing:
        return {
            "ok": False,
            "status": "missing_keys",
            "missing_keys": missing,
            "error": None,
        }

    # All keys present — attempt a real ping
    try:
        import litellm
        litellm.completion(
            model=body.model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return {
            "ok": True,
            "status": "verified",
            "missing_keys": [],
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "missing_keys": [],
            "error": str(exc),
        }


# ── Model CRUD ──────────────────────────────────────────────────────

@router.post("/api/models/add")
async def models_add(body: ModelBody):
    from supyagent.core.config import get_config_manager
    from supyagent.core.model_registry import get_model_registry

    registry = get_model_registry()
    config_mgr = get_config_manager()

    if registry.is_registered(body.model):
        return {"ok": True, "message": "Already registered"}

    # If the caller provided a custom key_name + value, store it
    if body.key_name and body.key_value:
        config_mgr.set(body.key_name, body.key_value)

    registry.add(body.model)
    if not registry.get_default():
        registry.set_default(body.model)

    # Load stored keys so litellm can detect them
    config_mgr.load_into_environment()

    # Return provider detection info so the UI can prompt if needed
    detected_key = registry.detect_provider_key(body.model)
    has_key = registry.check_api_key(body.model)
    missing_keys = _get_missing_keys(body.model)
    return {
        "ok": True,
        "detected_key": detected_key,
        "has_key": has_key,
        "missing_keys": missing_keys,
    }


@router.post("/api/models/remove")
async def models_remove(body: ModelBody):
    from supyagent.core.model_registry import get_model_registry

    registry = get_model_registry()
    removed = registry.remove(body.model)
    return {"ok": removed, "message": "Removed" if removed else "Not found"}


@router.post("/api/models/default")
async def models_default(body: ModelBody):
    from supyagent.core.model_registry import get_model_registry

    registry = get_model_registry()
    registry.set_default(body.model)
    return {"ok": True}


@router.post("/api/models/assign-role")
async def models_assign_role(body: RoleBody):
    from supyagent.core.model_registry import get_model_registry

    registry = get_model_registry()
    registry.assign_role(body.role, body.model)
    return {"ok": True}


@router.post("/api/models/unassign-role")
async def models_unassign_role(body: UnassignRoleBody):
    from supyagent.core.model_registry import get_model_registry

    registry = get_model_registry()
    removed = registry.unassign_role(body.role)
    return {"ok": removed}


# ── Key management ──────────────────────────────────────────────────

@router.get("/api/keys/status")
async def keys_status():
    from supyagent.core.config import KNOWN_LLM_KEYS, get_config_manager

    config_mgr = get_config_manager()
    result = []
    for key_name, description in KNOWN_LLM_KEYS.items():
        has_stored = key_name in config_mgr.list_keys()
        has_env = key_name in os.environ
        result.append({
            "key_name": key_name,
            "description": description,
            "configured": has_stored or has_env,
            "source": "env" if has_env else ("stored" if has_stored else None),
        })
    return {"keys": result}


@router.post("/api/keys/set")
async def keys_set(body: KeySetBody):
    from supyagent.core.config import get_config_manager

    config_mgr = get_config_manager()
    config_mgr.set(body.key_name, body.value)
    return {"ok": True}


@router.post("/api/keys/delete")
async def keys_delete(body: KeyDeleteBody):
    from supyagent.core.config import get_config_manager

    config_mgr = get_config_manager()
    deleted = config_mgr.delete(body.key_name)
    return {"ok": deleted}
