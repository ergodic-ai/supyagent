# /// script
# dependencies = ["pydantic", "httpx", "markdownify", "beautifulsoup4"]
# ///
"""
Web fetching tools.

Fetch URLs and make HTTP requests. Converts HTML pages to clean markdown
for easy consumption by agents. No browser needed — uses HTTP directly.
"""

import json
from typing import Any, Dict, Optional

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pydantic import BaseModel, Field

# Default headers to look like a normal browser
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Tags to remove from HTML before converting to markdown
_REMOVE_TAGS = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "noscript",
    "iframe",
    "svg",
]


def _html_to_markdown(html: str, include_links: bool = True) -> str:
    """Convert HTML to clean markdown text."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in _REMOVE_TAGS:
        for el in soup.find_all(tag):
            el.decompose()

    # Remove images if links not wanted
    if not include_links:
        for img in soup.find_all("img"):
            img.decompose()

    # Convert to markdown
    text = md(str(soup))

    # Clean up excessive whitespace
    lines = text.splitlines()
    cleaned = []
    prev_blank = False
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            if not prev_blank:
                cleaned.append("")
                prev_blank = True
        else:
            cleaned.append(stripped)
            prev_blank = False

    return "\n".join(cleaned).strip()


# =============================================================================
# Fetch URL → Markdown
# =============================================================================


class FetchUrlInput(BaseModel):
    """Input for fetch_url function."""

    url: str = Field(description="The URL to fetch")
    include_links: bool = Field(
        default=True, description="Include hyperlinks in the markdown output"
    )
    max_length: int = Field(
        default=50000,
        description="Maximum characters to return (truncates if longer)",
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="Additional HTTP headers to send"
    )


class FetchUrlOutput(BaseModel):
    """Output for fetch_url function."""

    ok: bool
    content: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None  # Final URL after redirects
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    truncated: bool = False
    error: Optional[str] = None


def fetch_url(input: FetchUrlInput) -> FetchUrlOutput:
    """
    Fetch a URL and return its content as clean markdown.

    Strips navigation, scripts, styles, and other noise. Great for reading
    documentation, articles, and web pages. Follows redirects automatically.

    For JavaScript-rendered pages (SPAs), use browse() from the browser tool instead.

    Examples:
        >>> fetch_url({"url": "https://docs.python.org/3/library/pathlib.html"})
        >>> fetch_url({"url": "https://example.com", "include_links": False})
    """
    try:
        headers = {**_DEFAULT_HEADERS}
        if input.headers:
            headers.update(input.headers)

        with httpx.Client(
            follow_redirects=True,
            timeout=input.timeout,
            headers=headers,
        ) as client:
            response = client.get(input.url)

        content_type = response.headers.get("content-type", "")
        final_url = str(response.url)

        # Handle different content types
        if "application/json" in content_type:
            try:
                data = response.json()
                content = json.dumps(data, indent=2)
            except Exception:
                content = response.text
        elif "text/plain" in content_type:
            content = response.text
        elif "text/html" in content_type or "application/xhtml" in content_type:
            # Extract title
            soup = BeautifulSoup(response.text, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else None

            content = _html_to_markdown(response.text, include_links=input.include_links)

            truncated = False
            if len(content) > input.max_length:
                content = content[: input.max_length] + "\n\n... [truncated]"
                truncated = True

            return FetchUrlOutput(
                ok=True,
                content=content,
                title=title,
                url=final_url,
                status_code=response.status_code,
                content_type=content_type,
                truncated=truncated,
            )
        else:
            # Binary or unknown — just return info
            content = f"[Binary content: {content_type}, {len(response.content)} bytes]"

        truncated = False
        if len(content) > input.max_length:
            content = content[: input.max_length] + "\n\n... [truncated]"
            truncated = True

        return FetchUrlOutput(
            ok=True,
            content=content,
            url=final_url,
            status_code=response.status_code,
            content_type=content_type,
            truncated=truncated,
        )

    except httpx.TimeoutException:
        return FetchUrlOutput(
            ok=False, error=f"Request timed out after {input.timeout}s"
        )
    except httpx.ConnectError as e:
        return FetchUrlOutput(ok=False, error=f"Connection failed: {e}")
    except Exception as e:
        return FetchUrlOutput(ok=False, error=str(e))


# =============================================================================
# HTTP Request (full control)
# =============================================================================


class HttpRequestInput(BaseModel):
    """Input for http_request function."""

    method: str = Field(
        default="GET",
        description="HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS",
    )
    url: str = Field(description="The URL to request")
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="HTTP headers"
    )
    body: Optional[str] = Field(
        default=None,
        description="Request body (string). For JSON, pass a JSON string.",
    )
    json_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Request body as JSON object (sets Content-Type automatically)",
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")


class HttpRequestOutput(BaseModel):
    """Output for http_request function."""

    ok: bool
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    body: Optional[str] = None
    url: Optional[str] = None  # Final URL after redirects
    error: Optional[str] = None


def http_request(input: HttpRequestInput) -> HttpRequestOutput:
    """
    Make an HTTP request with full control over method, headers, and body.

    Use this for API calls, webhooks, and any HTTP interaction beyond
    simple page fetching. Returns raw response body and headers.

    Examples:
        >>> http_request({"url": "https://api.github.com/repos/python/cpython"})
        >>> http_request({"method": "POST", "url": "https://httpbin.org/post", "json_body": {"key": "value"}})
        >>> http_request({"method": "DELETE", "url": "https://api.example.com/items/1", "headers": {"Authorization": "Bearer token"}})
    """
    try:
        method = input.method.upper()
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
            return HttpRequestOutput(ok=False, error=f"Invalid HTTP method: {method}")

        headers = input.headers or {}

        with httpx.Client(
            follow_redirects=input.follow_redirects,
            timeout=input.timeout,
        ) as client:
            kwargs: Dict[str, Any] = {"headers": headers}

            if input.json_body is not None:
                kwargs["json"] = input.json_body
            elif input.body is not None:
                kwargs["content"] = input.body

            response = client.request(method, input.url, **kwargs)

        # Limit body size
        body = response.text
        if len(body) > 100000:
            body = body[:100000] + "\n\n... [truncated at 100KB]"

        resp_headers = dict(response.headers)

        return HttpRequestOutput(
            ok=True,
            status_code=response.status_code,
            headers=resp_headers,
            body=body,
            url=str(response.url),
        )

    except httpx.TimeoutException:
        return HttpRequestOutput(
            ok=False, error=f"Request timed out after {input.timeout}s"
        )
    except httpx.ConnectError as e:
        return HttpRequestOutput(ok=False, error=f"Connection failed: {e}")
    except Exception as e:
        return HttpRequestOutput(ok=False, error=str(e))
