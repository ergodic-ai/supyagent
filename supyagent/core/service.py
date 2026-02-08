"""
Service client for supyagent_service integration.

Handles device auth flow, tool discovery, and tool execution
via the supyagent_service REST API.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from supyagent.core.config import get_config_manager

logger = logging.getLogger(__name__)

# Config keys stored in ~/.supyagent/config/
SERVICE_API_KEY = "SUPYAGENT_SERVICE_API_KEY"
SERVICE_URL = "SUPYAGENT_SERVICE_URL"
DEFAULT_SERVICE_URL = "https://app.supyagent.com"


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
    ):
        config_mgr = get_config_manager()
        self.api_key = api_key or config_mgr.get(SERVICE_API_KEY)
        self.base_url = (
            base_url or config_mgr.get(SERVICE_URL) or DEFAULT_SERVICE_URL
        ).rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=30.0,
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
            return data.get("tools", [])
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
            return data.get("integrations", [])
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

        try:
            if method == "GET":
                response = self._client.get(path, params=arguments)
            elif method == "POST":
                response = self._client.post(path, json=arguments)
            elif method == "PATCH":
                response = self._client.patch(path, json=arguments)
            elif method == "DELETE":
                response = self._client.delete(path, params=arguments)
            else:
                return {"ok": False, "error": f"Unsupported HTTP method: {method}"}

            response.raise_for_status()
            data = response.json()
            return {"ok": True, "data": data}

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

    def health_check(self) -> bool:
        """Check if the service is reachable."""
        try:
            response = self._client.get("/api/health")
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


def get_service_client() -> ServiceClient | None:
    """
    Get a ServiceClient if service credentials exist.

    Returns None if not connected.
    """
    config_mgr = get_config_manager()
    api_key = config_mgr.get(SERVICE_API_KEY)
    if not api_key:
        return None
    return ServiceClient(api_key=api_key)
