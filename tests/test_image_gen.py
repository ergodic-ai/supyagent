"""Unit tests for supypowers/image_gen.py image generation tool."""

import base64
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the tool module directly (it's a supypowers script, not a package)
_IMAGE_GEN_PATH = Path(__file__).parent.parent / "supypowers" / "image_gen.py"


@pytest.fixture
def image_gen():
    """Import image_gen module from supypowers directory."""
    spec = importlib.util.spec_from_file_location("image_gen", _IMAGE_GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Minimal valid PNG for test fixtures
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
TINY_PNG_B64 = base64.b64encode(TINY_PNG).decode()


def _mock_image_response(b64_json=None, url=None, n=1):
    """Create a mock litellm ImageResponse."""
    resp = MagicMock()
    resp.data = []
    for _ in range(n):
        img = MagicMock()
        img.b64_json = b64_json
        img.url = url
        resp.data.append(img)
    return resp


# =============================================================================
# generate_image
# =============================================================================


class TestGenerateImage:
    def test_success_b64(self, image_gen, tmp_path):
        """Successful generation with b64_json response."""
        mock_resp = _mock_image_response(b64_json=TINY_PNG_B64)

        with patch.object(image_gen.litellm, "image_generation", return_value=mock_resp):
            inp = image_gen.GenerateImageInput(
                prompt="a red square",
                model="dall-e-3",
                output_dir=str(tmp_path),
            )
            result = image_gen.generate_image(inp)

        assert result.ok is True
        assert result.model == "dall-e-3"
        assert len(result.paths) == 1
        assert len(result.images) == 1
        assert result.paths == result.images
        # File exists and has correct content
        assert Path(result.paths[0]).exists()
        assert Path(result.paths[0]).read_bytes() == TINY_PNG

    def test_success_url_fallback(self, image_gen, tmp_path):
        """Falls back to URL download when b64_json is None."""
        mock_resp = _mock_image_response(url="https://example.com/image.png")
        mock_http_resp = MagicMock()
        mock_http_resp.content = TINY_PNG
        mock_http_resp.raise_for_status = MagicMock()

        with (
            patch.object(image_gen.litellm, "image_generation", return_value=mock_resp),
            patch("httpx.get", return_value=mock_http_resp) as mock_get,
        ):
            inp = image_gen.GenerateImageInput(
                prompt="a blue circle",
                output_dir=str(tmp_path),
            )
            result = image_gen.generate_image(inp)

        assert result.ok is True
        assert len(result.paths) == 1
        mock_get.assert_called_once()

    def test_multiple_images(self, image_gen, tmp_path):
        """Generates n images and saves them all."""
        # Use different content for each to get different hashes
        png2 = TINY_PNG + b"\x00"
        b64_2 = base64.b64encode(png2).decode()

        mock_resp = MagicMock()
        img1 = MagicMock()
        img1.b64_json = TINY_PNG_B64
        img1.url = None
        img2 = MagicMock()
        img2.b64_json = b64_2
        img2.url = None
        mock_resp.data = [img1, img2]

        with patch.object(image_gen.litellm, "image_generation", return_value=mock_resp):
            inp = image_gen.GenerateImageInput(
                prompt="two variations",
                n=2,
                output_dir=str(tmp_path),
            )
            result = image_gen.generate_image(inp)

        assert result.ok is True
        assert len(result.paths) == 2
        assert len(result.images) == 2
        assert all(Path(p).exists() for p in result.paths)

    def test_creates_output_dir(self, image_gen, tmp_path):
        """Output directory is created if it doesn't exist."""
        out_dir = tmp_path / "new_dir" / "nested"
        assert not out_dir.exists()

        mock_resp = _mock_image_response(b64_json=TINY_PNG_B64)
        with patch.object(image_gen.litellm, "image_generation", return_value=mock_resp):
            inp = image_gen.GenerateImageInput(
                prompt="test",
                output_dir=str(out_dir),
            )
            result = image_gen.generate_image(inp)

        assert result.ok is True
        assert out_dir.exists()

    def test_api_error(self, image_gen, tmp_path):
        """API error returns ok=False with error message."""
        with patch.object(
            image_gen.litellm, "image_generation",
            side_effect=Exception("Rate limit exceeded"),
        ):
            inp = image_gen.GenerateImageInput(
                prompt="test",
                output_dir=str(tmp_path),
            )
            result = image_gen.generate_image(inp)

        assert result.ok is False
        assert "Rate limit exceeded" in result.error
        assert result.model == "dall-e-3"

    def test_auth_error(self, image_gen, tmp_path):
        """Auth error includes API key hint."""
        import litellm

        with patch.object(
            image_gen.litellm, "image_generation",
            side_effect=litellm.AuthenticationError(
                message="Invalid API key", llm_provider="openai", model="dall-e-3"
            ),
        ):
            inp = image_gen.GenerateImageInput(
                prompt="test",
                model="dall-e-3",
                output_dir=str(tmp_path),
            )
            result = image_gen.generate_image(inp)

        assert result.ok is False
        assert "OPENAI_API_KEY" in result.error

    def test_deduplication(self, image_gen, tmp_path):
        """Same image content produces same filename (content-hash)."""
        mock_resp = _mock_image_response(b64_json=TINY_PNG_B64)

        with patch.object(image_gen.litellm, "image_generation", return_value=mock_resp):
            inp = image_gen.GenerateImageInput(prompt="test", output_dir=str(tmp_path))
            r1 = image_gen.generate_image(inp)
            r2 = image_gen.generate_image(inp)

        assert r1.paths == r2.paths  # Same hash = same filename


# =============================================================================
# Image detection integration
# =============================================================================


class TestImageDetectionIntegration:
    def test_images_field_detected(self, image_gen, tmp_path):
        """detect_images_in_tool_result finds images from the 'images' field in nested data."""
        from supyagent.utils.media import detect_images_in_tool_result

        # Create actual image files
        img_path = tmp_path / "test.png"
        img_path.write_bytes(TINY_PNG)

        # Simulate the supypowers wrapper format: {"ok": true, "data": {...}}
        result = {
            "ok": True,
            "data": {
                "ok": True,
                "paths": [str(img_path)],
                "images": [str(img_path)],
                "model": "dall-e-3",
            },
            "tool_name": "image_gen__generate_image",
        }

        detected = detect_images_in_tool_result(result)
        assert detected == [str(img_path)]

    def test_top_level_images_detected(self, tmp_path):
        """detect_images_in_tool_result also finds _images at top level."""
        from supyagent.utils.media import detect_images_in_tool_result

        img_path = tmp_path / "test.png"
        img_path.write_bytes(TINY_PNG)

        result = {"ok": True, "_images": [str(img_path)]}
        detected = detect_images_in_tool_result(result)
        assert detected == [str(img_path)]

    def test_no_duplicates(self, tmp_path):
        """Same path in both _images and nested images should not duplicate."""
        from supyagent.utils.media import detect_images_in_tool_result

        img_path = tmp_path / "test.png"
        img_path.write_bytes(TINY_PNG)
        path_str = str(img_path)

        result = {
            "ok": True,
            "_images": [path_str],
            "data": {"ok": True, "images": [path_str]},
        }
        detected = detect_images_in_tool_result(result)
        assert len(detected) == 1


# =============================================================================
# list_image_models
# =============================================================================


class TestListImageModels:
    def test_all_models(self, image_gen):
        """Returns all models when no filter."""
        inp = image_gen.ListImageModelsInput()
        result = image_gen.list_image_models(inp)
        assert result.ok is True
        assert len(result.models) > 0
        # Check structure
        first = result.models[0]
        assert "name" in first
        assert "provider" in first
        assert "api_key_env" in first
        assert "description" in first

    def test_filter_by_provider(self, image_gen):
        """Filters models by provider name."""
        inp = image_gen.ListImageModelsInput(provider="openai")
        result = image_gen.list_image_models(inp)
        assert result.ok is True
        assert all(m["provider"] == "openai" for m in result.models)
        assert len(result.models) >= 2  # dall-e-2 and dall-e-3

    def test_filter_no_match(self, image_gen):
        """Unknown provider returns empty list."""
        inp = image_gen.ListImageModelsInput(provider="nonexistent")
        result = image_gen.list_image_models(inp)
        assert result.ok is True
        assert result.models == []


# =============================================================================
# _api_key_hint
# =============================================================================


class TestApiKeyHint:
    def test_openai(self, image_gen):
        assert image_gen._api_key_hint("dall-e-3") == "OPENAI_API_KEY"

    def test_stability(self, image_gen):
        assert image_gen._api_key_hint("stability/sd3-large") == "STABILITY_API_KEY"

    def test_fal(self, image_gen):
        assert image_gen._api_key_hint("fal_ai/fal-ai/flux-pro/v1.1") == "FAL_API_KEY"

    def test_google(self, image_gen):
        assert image_gen._api_key_hint("gemini/imagen-3.0-fast") == "GOOGLE_API_KEY"

    def test_unknown(self, image_gen):
        assert "appropriate" in image_gen._api_key_hint("some-unknown-model")
