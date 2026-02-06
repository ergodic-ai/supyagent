# /// script
# dependencies = ["pydantic", "litellm", "httpx"]
# ///
"""
Image generation tools.

Generate images using DALL-E, Stable Diffusion, Flux, Gemini Imagen,
and other models via LiteLLM's unified API. Supports all major providers.
"""

import base64
import hashlib
import os
from pathlib import Path
from typing import Optional

import httpx
import litellm
from pydantic import BaseModel, Field

# Suppress litellm's verbose stdout messages that corrupt JSON output
litellm.suppress_debug_info = True


# ── generate_image ──────────────────────────────────────────────────────────


class GenerateImageInput(BaseModel):
    """Input for image generation."""

    prompt: str = Field(..., description="Text description of the image to generate")
    model: str = Field(
        default="dall-e-3",
        description=(
            "LiteLLM model identifier. Examples: "
            "'dall-e-3', 'stability/sd3-large', "
            "'fal_ai/fal-ai/flux-pro/v1.1', "
            "'gemini/imagen-3.0-fast-generate-001'"
        ),
    )
    size: str = Field(
        default="1024x1024",
        description="Image dimensions (e.g. '1024x1024', '1024x1792', '1792x1024')",
    )
    quality: str = Field(
        default="standard",
        description="Quality level: 'standard' or 'hd' (DALL-E 3 only)",
    )
    n: int = Field(default=1, description="Number of images to generate (1-4)")
    output_dir: str = Field(
        default="generated_images",
        description="Directory to save generated images",
    )


class GenerateImageOutput(BaseModel):
    """Output for image generation."""

    ok: bool
    paths: list[str] = Field(default_factory=list, description="Absolute paths to saved images")
    images: list[str] = Field(default_factory=list, description="Image paths for multimodal detection")
    model: str | None = None
    error: str | None = None


def generate_image(input: GenerateImageInput) -> GenerateImageOutput:
    """
    Generate images using AI models (DALL-E, Stable Diffusion, Flux, Gemini, etc.).

    Uses LiteLLM to call any supported image generation provider. Any valid LiteLLM
    model identifier works, not just the ones in the curated list. The generated
    images are saved to disk and returned as file paths.

    API keys are read from environment variables:
    - OPENAI_API_KEY for DALL-E models
    - STABILITY_API_KEY for Stable Diffusion models
    - FAL_API_KEY for Flux models
    - GOOGLE_API_KEY for Gemini Imagen models
    - OPENROUTER_API_KEY for any model via OpenRouter

    Examples:
        >>> generate_image({"prompt": "A sunset over mountains", "model": "dall-e-3"})
        >>> generate_image({"prompt": "A cat in space", "model": "openrouter/google/gemini-2.5-flash-image"})
    """
    try:
        # Create output directory
        out_dir = Path(input.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Allow providers that don't support response_format to still work
        litellm.drop_params = True

        # Call litellm image generation
        response = litellm.image_generation(
            prompt=input.prompt,
            model=input.model,
            n=input.n,
            size=input.size,
            quality=input.quality,
            response_format="b64_json",
        )

        saved_paths: list[str] = []

        for img in response.data:
            # Try b64_json first, fall back to URL download
            if img.b64_json:
                raw = base64.b64decode(img.b64_json)
            elif img.url:
                resp = httpx.get(img.url, timeout=60, follow_redirects=True)
                resp.raise_for_status()
                raw = resp.content
            else:
                continue

            # Save with content-hash filename for deduplication
            content_hash = hashlib.sha256(raw).hexdigest()[:12]
            filename = f"{content_hash}.png"
            filepath = out_dir / filename
            filepath.write_bytes(raw)
            saved_paths.append(str(filepath.absolute()))

        return GenerateImageOutput(
            ok=True,
            paths=saved_paths,
            images=saved_paths,
            model=input.model,
        )

    except litellm.AuthenticationError as e:
        # Extract which key is needed from the model prefix
        key_hint = _api_key_hint(input.model)
        return GenerateImageOutput(
            ok=False,
            model=input.model,
            error=f"Authentication failed. Make sure {key_hint} is set. Details: {e}",
        )
    except Exception as e:
        return GenerateImageOutput(
            ok=False,
            model=input.model,
            error=str(e),
        )


def _api_key_hint(model: str) -> str:
    """Return the likely API key env var name for a model."""
    m = model.lower()
    if "dall-e" in m or m.startswith("openai"):
        return "OPENAI_API_KEY"
    if "stability" in m:
        return "STABILITY_API_KEY"
    if "fal" in m:
        return "FAL_API_KEY"
    if "gemini" in m or "imagen" in m:
        return "GOOGLE_API_KEY"
    if "openrouter" in m:
        return "OPENROUTER_API_KEY"
    if "bedrock" in m:
        return "AWS credentials"
    if "azure" in m:
        return "AZURE_API_KEY"
    return "the appropriate API key"


# ── list_image_models ───────────────────────────────────────────────────────


class ListImageModelsInput(BaseModel):
    """Input for listing available image generation models."""

    provider: Optional[str] = Field(
        default=None,
        description="Filter by provider name (e.g. 'openai', 'stability', 'fal', 'google')",
    )


class ListImageModelsOutput(BaseModel):
    """Output for listing image generation models."""

    ok: bool
    models: list[dict] = Field(default_factory=list)
    error: str | None = None


# Curated list of popular image generation models
_IMAGE_MODELS = [
    {
        "name": "dall-e-3",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "sizes": ["1024x1024", "1024x1792", "1792x1024"],
        "description": "OpenAI DALL-E 3 — high quality, follows prompts closely",
    },
    {
        "name": "dall-e-2",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "sizes": ["256x256", "512x512", "1024x1024"],
        "description": "OpenAI DALL-E 2 — faster and cheaper, supports image edits",
    },
    {
        "name": "stability/sd3-large",
        "provider": "stability",
        "api_key_env": "STABILITY_API_KEY",
        "sizes": ["1024x1024"],
        "description": "Stable Diffusion 3 Large — high quality open model",
    },
    {
        "name": "stability/sd3-large-turbo",
        "provider": "stability",
        "api_key_env": "STABILITY_API_KEY",
        "sizes": ["1024x1024"],
        "description": "Stable Diffusion 3 Large Turbo — fast inference",
    },
    {
        "name": "fal_ai/fal-ai/flux-pro/v1.1",
        "provider": "fal",
        "api_key_env": "FAL_API_KEY",
        "sizes": ["1024x1024", "1024x768", "768x1024"],
        "description": "Flux Pro v1.1 — professional quality, excellent detail",
    },
    {
        "name": "fal_ai/fal-ai/flux-schnell",
        "provider": "fal",
        "api_key_env": "FAL_API_KEY",
        "sizes": ["1024x1024", "1024x768", "768x1024"],
        "description": "Flux Schnell — fast variant, good quality",
    },
    {
        "name": "gemini/imagen-3.0-fast-generate-001",
        "provider": "google",
        "api_key_env": "GOOGLE_API_KEY",
        "sizes": ["1024x1024"],
        "description": "Google Imagen 3 Fast — fast generation, good quality",
    },
    {
        "name": "gemini/imagen-3.0-capability-001",
        "provider": "google",
        "api_key_env": "GOOGLE_API_KEY",
        "sizes": ["1024x1024"],
        "description": "Google Imagen 3 — highest quality Google image model",
    },
    {
        "name": "openrouter/google/gemini-2.5-flash-image",
        "provider": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "sizes": ["1024x1024"],
        "description": "Gemini 2.5 Flash with image generation via OpenRouter",
    },
]


def list_image_models(input: ListImageModelsInput) -> ListImageModelsOutput:
    """
    List available image generation models and their requirements.

    Returns a curated list of popular models with provider info and required API keys.
    Use this to help choose which model to use for image generation.
    Note: generate_image accepts ANY valid LiteLLM model identifier, not just those in this list.

    Examples:
        >>> list_image_models({})
        >>> list_image_models({"provider": "openai"})
    """
    models = _IMAGE_MODELS
    if input.provider:
        provider = input.provider.lower()
        models = [m for m in models if m["provider"] == provider]

    return ListImageModelsOutput(ok=True, models=models)
