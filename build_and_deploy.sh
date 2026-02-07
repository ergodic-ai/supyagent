#!/usr/bin/env bash
set -euo pipefail

# Build and upload `supyagent` to PyPI using uv + twine.
#
# Requirements:
# - `uv` on PATH
# - a `.env` file in the repo root with:
#     TWINE_USERNAME=__token__
#     TWINE_PASSWORD=pypi-...
#
# Usage:
#   ./build_and_deploy.sh          # upload to real PyPI
#   ./build_and_deploy.sh --test   # upload to TestPyPI

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH" >&2
  exit 127
fi

ENV_FILE="${ENV_FILE:-.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Create it with TWINE_USERNAME/TWINE_PASSWORD." >&2
  exit 2
fi

TARGET="${1:-}"
if [[ -n "${TARGET}" && "${TARGET}" != "--test" ]]; then
  echo "Unknown argument: ${TARGET}" >&2
  echo "Usage: $0 [--test]" >&2
  exit 2
fi

echo "Cleaning dist/ and build/ ..."
rm -rf dist build

echo "Building sdist+wheel ..."
uv run --no-project --with build --with twine python -m build

echo "Verifying sdist does not contain .env ..."
python3 - <<'PY'
import tarfile
from pathlib import Path

dist = Path("dist")
sdists = sorted(dist.glob("*.tar.gz"))
if not sdists:
    raise SystemExit("No sdist found in dist/")

sdist = sdists[-1]
with tarfile.open(sdist, "r:gz") as tf:
    for member in tf.getmembers():
        name = member.name
        if name.endswith("/.env") or name == ".env":
            raise SystemExit(f"Refusing to upload: .env found in sdist ({sdist})")
print("OK: .env not found in sdist")
PY

echo "Uploading ..."
if [[ "${TARGET}" == "--test" ]]; then
  uv run --no-project --env-file "${ENV_FILE}" --with twine \
    python -m twine upload --repository testpypi --non-interactive dist/*
else
  uv run --no-project --env-file "${ENV_FILE}" --with twine \
    python -m twine upload --non-interactive dist/*
fi

echo "Done."

