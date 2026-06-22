#!/usr/bin/env bash
# prepare-python.sh
# Downloads a standalone Python and installs all dependencies for macOS distribution.
#
# Dependencies are read from uv.lock (via `uv export`) — pyproject.toml is the
# single source of truth. No hardcoded dependency lists.
#
# Uses python-build-standalone (https://github.com/astral-sh/python-build-standalone)
# which provides relocatable Python builds for macOS.
#
# Prerequisites:
# - uv must be installed (https://docs.astral.sh/uv/)
# - curl must be available
# - git must be available (for git-based Python packages)

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-$(cat "$(dirname "$0")/../backend/.python-version" | tr -d '[:space:]')}"
OUTPUT_DIR="python-embed"
ARCH="${ARCH:-$(uname -m)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
OUTPUT_PATH="$PROJECT_DIR/$OUTPUT_DIR"
TEMP_DIR="$(mktemp -d)"
RELEASE_JSON="$TEMP_DIR/release.json"

case "$ARCH" in
    arm64|aarch64) PBS_ARCH="aarch64" ;;
    *)
        echo "ERROR: Unsupported architecture: $ARCH. MLX builds require Apple Silicon (arm64)."
        exit 1
        ;;
esac

fetch_latest_release_json() {
    local out="$1"
    local api_url="https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"

    if command -v gh &>/dev/null; then
        if [ -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]; then
            echo "  Fetching release metadata via gh api..."
            gh api "$api_url" > "$out"
            return 0
        fi
        if gh auth status &>/dev/null; then
            echo "  Fetching release metadata via gh api (authenticated)..."
            gh api "$api_url" > "$out"
            return 0
        fi
    fi

    echo "  Fetching release metadata via curl..."
    local curl_args=(
        -fsSL
        -H "Accept: application/vnd.github+json"
        -H "User-Agent: ltx-desktop-macos-build"
    )
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        curl_args+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    fi
    curl "${curl_args[@]}" -o "$out" "$api_url"
}

download_release_asset() {
    local url="$1"
    local out="$2"
    local curl_args=(
        -L
        --fail
        --progress-bar
        -H "User-Agent: ltx-desktop-macos-build"
    )
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        curl_args+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    fi
    curl "${curl_args[@]}" -o "$out" "$url"
}

echo "Resolving latest python-build-standalone release..."
fetch_latest_release_json "$RELEASE_JSON"

PBS_TAG="$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$RELEASE_JSON" | head -1)"
ASSET_NAME="$(grep -o "cpython-${PYTHON_VERSION}[^\"]*${PBS_ARCH}-apple-darwin-install_only_stripped.tar.gz" "$RELEASE_JSON" \
    | sed 's/\\u002B/+/g' \
    | sort -V \
    | tail -1)"

if [ -z "$ASSET_NAME" ]; then
    ASSET_NAME="$(grep -o "cpython-3\.12[^\"]*${PBS_ARCH}-apple-darwin-install_only_stripped.tar.gz" "$RELEASE_JSON" \
        | sed 's/\\u002B/+/g' \
        | sort -V \
        | tail -1)"
fi

if [ -z "$ASSET_NAME" ] || [ -z "$PBS_TAG" ]; then
    echo "ERROR: Could not resolve python-build-standalone asset for Python $PYTHON_VERSION on $PBS_ARCH."
    exit 1
fi

PYTHON_VERSION="$(echo "$ASSET_NAME" | sed -n 's/cpython-\([0-9.]*\)+.*/\1/p')"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${ASSET_NAME}"

echo "========================================"
echo " LTX Desktop - Python Environment Setup"
echo " Platform: macOS ($ARCH)"
echo " Python: $PYTHON_VERSION"
echo " PBS tag: $PBS_TAG"
echo "========================================"

echo ""
echo "Step 1: Verifying prerequisites..."

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it: https://docs.astral.sh/uv/"
    exit 1
fi
echo "  uv: $(command -v uv)"

if ! command -v curl &>/dev/null; then
    echo "ERROR: curl not found."
    exit 1
fi
echo "  curl: $(command -v curl)"

if ! command -v git &>/dev/null; then
    echo "ERROR: git not found (needed for git-based Python packages)."
    exit 1
fi
echo "  git: $(command -v git)"

echo ""
echo "Step 2: Generating requirements.txt from uv.lock..."

REQUIREMENTS_FILE="$BACKEND_DIR/requirements-dist.txt"

uv export --frozen --no-hashes --no-editable --no-emit-project \
    --no-header --no-annotate \
    --project "$BACKEND_DIR" \
    > "$REQUIREMENTS_FILE"

DEP_COUNT=$(grep -c '^\S' "$REQUIREMENTS_FILE" || true)
echo "  Exported $DEP_COUNT dependencies from uv.lock"

echo ""
echo "Step 3: Preparing directories..."

if [ -d "$OUTPUT_PATH" ]; then
    echo "  Removing existing $OUTPUT_DIR directory..."
    rm -rf "$OUTPUT_PATH"
fi

mkdir -p "$OUTPUT_PATH"

echo ""
echo "Step 4: Downloading Python $PYTHON_VERSION standalone ($PBS_ARCH)..."
echo "  URL: $PBS_URL"

PYTHON_TAR="$TEMP_DIR/python-standalone.tar.gz"
download_release_asset "$PBS_URL" "$PYTHON_TAR"
echo "  Downloaded Python standalone package"

echo "  Extracting..."
tar -xzf "$PYTHON_TAR" -C "$TEMP_DIR"
mv "$TEMP_DIR/python/"* "$OUTPUT_PATH/"
echo "  Extracted to $OUTPUT_PATH"

PYTHON_EXE="$OUTPUT_PATH/bin/python3"
if [ ! -f "$PYTHON_EXE" ]; then
    echo "ERROR: Python binary not found at $PYTHON_EXE"
    exit 1
fi

echo "  Python binary: $PYTHON_EXE"
"$PYTHON_EXE" --version

echo ""
echo "Step 5: Setting up pip..."

if ! "$PYTHON_EXE" -m pip --version &>/dev/null; then
    echo "  Installing pip..."
    curl -sL https://bootstrap.pypa.io/get-pip.py -o "$TEMP_DIR/get-pip.py"
    "$PYTHON_EXE" "$TEMP_DIR/get-pip.py" --no-warn-script-location
fi
echo "  pip: $("$PYTHON_EXE" -m pip --version)"

echo ""
echo "Step 6: Installing dependencies from requirements.txt..."
echo "  (This may take a while — MLX libraries are large)"

"$PYTHON_EXE" -m pip install -r "$REQUIREMENTS_FILE" \
    --no-warn-script-location --quiet

echo "  All dependencies installed"

echo ""
echo "Step 7: Cleaning up..."

find "$OUTPUT_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pyc" -delete 2>/dev/null || true

rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/pip 2>/dev/null || true
rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/pip-*.dist-info 2>/dev/null || true
rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/setuptools 2>/dev/null || true
rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/setuptools-*.dist-info 2>/dev/null || true

find "$OUTPUT_PATH/lib" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "$OUTPUT_PATH/lib" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

rm -rf "$OUTPUT_PATH/include" "$OUTPUT_PATH/share" 2>/dev/null || true
find "$OUTPUT_PATH/lib" -type d -name "include" -exec rm -rf {} + 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pyi" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pxd" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pyx" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.hpp" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.cpp" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.h" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.cuh" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.cu" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.cmake" -delete 2>/dev/null || true

rm -rf "$TEMP_DIR"
rm -f "$REQUIREMENTS_FILE"

echo "  Cleanup complete"

echo ""
echo "Step 8: Verifying installation..."

"$PYTHON_EXE" -c "
import sys
print(f'  Python: {sys.version}')
try:
    import mlx.core as mx
    print(f'  MLX: OK (device={mx.default_device()})')
except ImportError as e:
    print(f'  MLX import FAILED: {e}')
    sys.exit(1)
try:
    import fastapi
    print(f'  FastAPI: {fastapi.__version__}')
except ImportError as e:
    print(f'  FastAPI import FAILED: {e}')
    sys.exit(1)
try:
    import ltx_pipelines_mlx
    print('  ltx-pipelines-mlx: OK')
except ImportError as e:
    print(f'  ltx-pipelines-mlx: FAILED - {e}')
    sys.exit(1)
try:
    import ltx_core_mlx
    print('  ltx-core-mlx: OK')
except ImportError as e:
    print(f'  ltx-core-mlx: FAILED - {e}')
    sys.exit(1)
"

SIZE_BYTES=$(du -sk "$OUTPUT_PATH" | awk '{print $1 * 1024}')
SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1073741824" | bc)

echo ""
echo "========================================"
echo " Python environment ready!"
echo " Location: $OUTPUT_PATH"
echo " Size: ${SIZE_GB} GB"
echo "========================================"
