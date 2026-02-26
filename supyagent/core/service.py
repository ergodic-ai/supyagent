"""
Service client for supyagent_service integration.

Handles device auth flow, tool discovery, and tool execution
via the supyagent_service REST API.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx

from supyagent.core.config import get_config_manager

logger = logging.getLogger(__name__)

# Config keys stored in ~/.supyagent/config/
SERVICE_API_KEY = "SUPYAGENT_SERVICE_API_KEY"
SERVICE_URL = "SUPYAGENT_SERVICE_URL"
DEFAULT_SERVICE_URL = "https://app.supyagent.com"
DEFAULT_READ_TIMEOUT = 180.0  # 3 minutes — accommodates slow endpoints (image/video gen)


class ServiceClient:
    """
    Client for interacting with the supyagent_service API.

    Discovers and executes service tools (Gmail, Slack, GitHub, etc.)
    that the user has connected on the dashboard.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        config_mgr = get_config_manager()
        self.api_key = api_key or config_mgr.get(SERVICE_API_KEY)
        self.base_url = (
            base_url or config_mgr.get(SERVICE_URL) or DEFAULT_SERVICE_URL
        ).rstrip("/")
        read_timeout = timeout if timeout is not None else DEFAULT_READ_TIMEOUT
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(30.0, read=read_timeout),
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def is_connected(self) -> bool:
        return self.api_key is not None

    def discover_tools(self) -> list[dict[str, Any]]:
        """
        Discover available tools from the service.

        Returns tools already in OpenAI function-calling format with metadata.
        """
        if not self.api_key:
            return []

        try:
            response = self._client.get("/api/v1/tools")
            response.raise_for_status()
            data = response.json()
            # Response shape: {"ok": true, "data": {"tools": [...]}}
            inner = data.get("data", data)
            return inner.get("tools", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.warning(
                    "Service API key is invalid or expired. "
                    "Run 'supyagent connect' to reconnect."
                )
            else:
                logger.warning(f"Service tool discovery failed: {e}")
            return []
        except httpx.RequestError as e:
            logger.warning(f"Could not reach service at {self.base_url}: {e}")
            return []
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid response from service: {e}")
            return []

    def list_integrations(self) -> list[dict[str, Any]]:
        """
        List user's connected integrations from the service.

        Returns list of dicts with provider, status, connected_at, services.
        """
        if not self.api_key:
            return []

        try:
            response = self._client.get("/api/v1/integrations")
            response.raise_for_status()
            data = response.json()
            # Response shape: {"ok": true, "data": {"integrations": [...]}}
            inner = data.get("data", data)
            return inner.get("integrations", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.warning(
                    "Service API key is invalid or expired. "
                    "Run 'supyagent connect' to reconnect."
                )
            else:
                logger.warning(f"Service integration listing failed: {e}")
            return []
        except httpx.RequestError as e:
            logger.warning(f"Could not reach service at {self.base_url}: {e}")
            return []
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid response from service: {e}")
            return []

    @staticmethod
    def _resolve_path(path: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Substitute {param} placeholders in a path with values from arguments.

        Returns the resolved path and a copy of arguments with consumed keys removed.
        E.g. path="/api/v1/inbox/{id}", arguments={"id": "abc"} -> ("/api/v1/inbox/abc", {})
        """
        import re

        remaining = dict(arguments)
        def replacer(match: re.Match) -> str:
            key = match.group(1)
            if key in remaining:
                val = str(remaining.pop(key))
                return val
            return match.group(0)  # leave as-is if not found

        resolved = re.sub(r"\{(\w+)\}", replacer, path)
        return resolved, remaining

    def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a service tool by calling the appropriate API endpoint.

        Args:
            name: Tool name (e.g., 'gmail_list_messages')
            arguments: Tool arguments dict
            metadata: Tool metadata containing method and path

        Returns:
            Result dict following the {"ok": bool, "data": ...} convention.
        """
        if not self.api_key:
            return {
                "ok": False,
                "error": "Not connected to service. Run 'supyagent connect'.",
            }

        method = metadata.get("method", "GET").upper()
        path = metadata.get("path", "")

        if not path:
            return {"ok": False, "error": f"No API path for tool '{name}'"}

        # Substitute path parameters (e.g. /inbox/{id} -> /inbox/abc123)
        path, remaining_args = self._resolve_path(path, arguments)

        try:
            if method == "GET":
                response = self._client.get(path, params=remaining_args)
            elif method == "POST":
                defaults = metadata.get("bodyDefaults", {})
                body = {**defaults, **remaining_args}
                response = self._client.post(path, json=body)
            elif method == "PUT":
                defaults = metadata.get("bodyDefaults", {})
                body = {**defaults, **remaining_args}
                response = self._client.put(path, json=body)
            elif method == "PATCH":
                response = self._client.patch(path, json=remaining_args)
            elif method == "DELETE":
                response = self._client.delete(path, params=remaining_args)
            else:
                return {"ok": False, "error": f"Unsupported HTTP method: {method}"}

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if content_type.startswith("application/json") or not content_type:
                data = response.json()
                return {"ok": True, "data": data}
            else:
                return self._save_binary_response(response, content_type)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", str(e))
            except Exception:
                error_msg = str(e)

            if status == 401:
                return {
                    "ok": False,
                    "error": "Service API key expired. Run 'supyagent connect' to reconnect.",
                }
            elif status == 403:
                return {
                    "ok": False,
                    "error": f"Permission denied for {name}: {error_msg}",
                }
            elif status == 429:
                return {"ok": False, "error": f"Rate limit exceeded: {error_msg}"}
            else:
                return {"ok": False, "error": f"Service error ({status}): {error_msg}"}

        except httpx.RequestError as e:
            return {"ok": False, "error": f"Could not reach service: {e}"}

    @staticmethod
    def _save_binary_response(
        response: httpx.Response, content_type: str
    ) -> dict[str, Any]:
        """Save a binary HTTP response to ~/.supyagent/tmp/ and return the path."""
        import hashlib
        import mimetypes

        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".bin"
        digest = hashlib.sha256(response.content).hexdigest()[:12]
        tmp_dir = Path.home() / ".supyagent" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        file_path = tmp_dir / f"{digest}{ext}"
        file_path.write_bytes(response.content)
        return {
            "ok": True,
            "data": {
                "filePath": str(file_path),
                "contentType": content_type,
                "size": len(response.content),
            },
        }

    # ------------------------------------------------------------------
    # Inbox
    # ------------------------------------------------------------------

    def inbox_list(
        self,
        status: str | None = None,
        provider: str | None = None,
        event_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List inbox events."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if provider:
            params["provider"] = provider
        if event_type:
            params["event_type"] = event_type

        try:
            response = self._client.get("/api/v1/inbox", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"events": [], "total": 0, "error": str(e)}
        except httpx.RequestError as e:
            return {"events": [], "total": 0, "error": str(e)}

    def inbox_get(self, event_id: str) -> dict[str, Any] | None:
        """Get a single inbox event (auto-marks as read)."""
        try:
            response = self._client.get(f"/api/v1/inbox/{event_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError:
            return None
        except httpx.RequestError:
            return None

    def inbox_archive(self, event_id: str) -> bool:
        """Archive a single inbox event."""
        try:
            response = self._client.post(f"/api/v1/inbox/{event_id}/archive")
            response.raise_for_status()
            return True
        except (httpx.HTTPStatusError, httpx.RequestError):
            return False

    def inbox_archive_all(
        self, provider: str | None = None, before: str | None = None
    ) -> int:
        """Archive all inbox events. Returns count of archived events."""
        body: dict[str, str] = {}
        if provider:
            body["provider"] = provider
        if before:
            body["before"] = before

        try:
            response = self._client.post("/api/v1/inbox/archive-all", json=body)
            response.raise_for_status()
            data = response.json()
            return data.get("archived", 0)
        except (httpx.HTTPStatusError, httpx.RequestError):
            return 0

    def fetch_docs(self) -> str | None:
        """Fetch combined API documentation markdown from the service."""
        if not self.api_key:
            return None

        try:
            response = self._client.get("/api/v1/skills")
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.warning(
                    "Service API key is invalid or expired. "
                    "Run 'supyagent connect' to reconnect."
                )
            else:
                logger.warning(f"Failed to fetch docs: {e}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Could not reach service at {self.base_url}: {e}")
            return None

    def health_check(self) -> bool:
        """Check if the service is reachable."""
        try:
            response = self._client.get("/api/v1/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()


# ---------------------------------------------------------------------------
# Device Authorization Flow helpers
# ---------------------------------------------------------------------------


def request_device_code(base_url: str | None = None) -> dict[str, Any]:
    """
    Request a device code from the service (step 1 of device auth).

    Returns:
        Dict with device_code, user_code, verification_uri, expires_in, interval.
    """
    url = (base_url or DEFAULT_SERVICE_URL).rstrip("/")

    with httpx.Client(timeout=15.0) as client:
        response = client.post(f"{url}/api/v1/auth/device/code")
        response.raise_for_status()
        return response.json()


def poll_for_token(
    base_url: str | None = None,
    device_code: str = "",
    interval: int = 5,
    expires_in: int = 900,
) -> str:
    """
    Poll the service until the user approves or denies the device code.

    Args:
        base_url: Service URL
        device_code: The device_code from request_device_code()
        interval: Polling interval in seconds
        expires_in: Max seconds to wait

    Returns:
        The API key string on approval.

    Raises:
        TimeoutError: Code expired before approval.
        PermissionError: User denied the request.
        RuntimeError: Unexpected error.
    """
    url = (base_url or DEFAULT_SERVICE_URL).rstrip("/")
    deadline = time.time() + expires_in

    with httpx.Client(timeout=15.0) as client:
        while time.time() < deadline:
            time.sleep(interval)

            response = client.post(
                f"{url}/api/v1/auth/device/token",
                json={"device_code": device_code},
            )

            if response.status_code == 200:
                data = response.json()
                return data["api_key"]
            elif response.status_code == 428:
                # authorization_pending — keep polling
                continue
            elif response.status_code == 403:
                raise PermissionError(
                    "Device authorization was denied by the user."
                )
            elif response.status_code == 410:
                raise TimeoutError(
                    "Device code has expired. Please try again."
                )
            else:
                try:
                    data = response.json()
                    msg = data.get("error", response.status_code)
                except Exception:
                    msg = response.status_code
                raise RuntimeError(f"Unexpected error: {msg}")

    raise TimeoutError("Device code has expired. Please try again.")


# ---------------------------------------------------------------------------
# Credential management helpers
# ---------------------------------------------------------------------------


def store_service_credentials(
    api_key: str, base_url: str | None = None
) -> None:
    """Store service API key and URL in global config."""
    config_mgr = get_config_manager()
    config_mgr.set(SERVICE_API_KEY, api_key)
    if base_url and base_url != DEFAULT_SERVICE_URL:
        config_mgr.set(SERVICE_URL, base_url)


def clear_service_credentials() -> bool:
    """Remove stored service credentials. Returns True if key was removed."""
    config_mgr = get_config_manager()
    removed = config_mgr.delete(SERVICE_API_KEY)
    config_mgr.delete(SERVICE_URL)
    return removed


def get_service_client(timeout: float | None = None) -> ServiceClient | None:
    """
    Get a ServiceClient if service credentials exist.

    Returns None if not connected.
    """
    config_mgr = get_config_manager()
    api_key = config_mgr.get(SERVICE_API_KEY)
    if not api_key:
        return None
    return ServiceClient(api_key=api_key, timeout=timeout)
