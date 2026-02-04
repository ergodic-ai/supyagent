#!/bin/bash
#
# Release script for supyagent
# Usage: ./scripts/release.sh [version]
#
# Examples:
#   ./scripts/release.sh 0.2.0    # Release version 0.2.0
#   ./scripts/release.sh patch    # Bump patch version (0.1.0 -> 0.1.1)
#   ./scripts/release.sh minor    # Bump minor version (0.1.0 -> 0.2.0)
#   ./scripts/release.sh major    # Bump major version (0.1.0 -> 1.0.0)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo -e "${BLUE}📦 Supyagent Release Script${NC}"
echo "================================"
echo ""

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo -e "Current version: ${YELLOW}${CURRENT_VERSION}${NC}"

# Determine new version
if [ -z "$1" ]; then
    echo -e "${RED}Error: Version argument required${NC}"
    echo ""
    echo "Usage: $0 [version|patch|minor|major]"
    echo ""
    echo "Examples:"
    echo "  $0 0.2.0   # Set specific version"
    echo "  $0 patch   # Bump patch (0.1.0 -> 0.1.1)"
    echo "  $0 minor   # Bump minor (0.1.0 -> 0.2.0)"
    echo "  $0 major   # Bump major (0.1.0 -> 1.0.0)"
    exit 1
fi

# Parse version
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

case "$1" in
    patch)
        NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"
        ;;
    minor)
        NEW_VERSION="$MAJOR.$((MINOR + 1)).0"
        ;;
    major)
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    *)
        NEW_VERSION="$1"
        ;;
esac

echo -e "New version: ${GREEN}${NEW_VERSION}${NC}"
echo ""

# Confirm
read -p "Proceed with release? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}Step 1: Update version in pyproject.toml${NC}"
sed -i.bak "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
rm -f pyproject.toml.bak

# Also update version in CLI if it's hardcoded there
if grep -q "version=\"${CURRENT_VERSION}\"" supyagent/cli/main.py 2>/dev/null; then
    sed -i.bak "s/version=\"${CURRENT_VERSION}\"/version=\"${NEW_VERSION}\"/" supyagent/cli/main.py
    rm -f supyagent/cli/main.py.bak
fi

echo -e "${GREEN}✓${NC} Version updated to ${NEW_VERSION}"

echo ""
echo -e "${BLUE}Step 2: Run tests${NC}"
source .venv/bin/activate 2>/dev/null || true
python -m pytest tests/ -q
echo -e "${GREEN}✓${NC} All tests passed"

echo ""
echo -e "${BLUE}Step 3: Clean old builds${NC}"
rm -rf dist/ build/ *.egg-info
echo -e "${GREEN}✓${NC} Cleaned build directories"

echo ""
echo -e "${BLUE}Step 4: Build package${NC}"
python -m build
echo -e "${GREEN}✓${NC} Built successfully"

echo ""
echo -e "${BLUE}Step 5: Verify package${NC}"
twine check dist/*
echo -e "${GREEN}✓${NC} Package verified"

echo ""
echo -e "${BLUE}Step 6: Upload to PyPI${NC}"
# Load credentials from .env if present
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi
twine upload dist/*
echo -e "${GREEN}✓${NC} Uploaded to PyPI"

echo ""
echo -e "${BLUE}Step 7: Clear local caches${NC}"

# Clear pip cache
echo "  Clearing pip cache..."
pip cache purge 2>/dev/null || true

# Clear uv cache for supyagent
echo "  Clearing uv cache..."
uv cache clean supyagent 2>/dev/null || true

# Remove any local installs
echo "  Removing local editable install..."
pip uninstall -y supyagent 2>/dev/null || true
uv pip uninstall supyagent 2>/dev/null || true

echo -e "${GREEN}✓${NC} Caches cleared"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 Released supyagent ${NEW_VERSION}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "View on PyPI: https://pypi.org/project/supyagent/${NEW_VERSION}/"
echo ""
echo "To install the new version:"
echo "  pip install supyagent==${NEW_VERSION}"
echo "  # or"
echo "  uv pip install supyagent==${NEW_VERSION}"
echo ""
echo "To reinstall locally for development:"
echo "  uv pip install -e ."
