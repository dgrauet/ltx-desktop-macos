#!/usr/bin/env bash
# bundle-app.sh
# Copy embedded Python, backend source, and ffmpeg into a built .app bundle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

APP_PATH="${1:?Usage: $0 <path-to-app.bundle>}"

if [ ! -d "$APP_PATH/Contents" ]; then
    echo "ERROR: Invalid app bundle: $APP_PATH"
    exit 1
fi

if [ ! -d "$PROJECT_DIR/python-embed" ]; then
    echo "ERROR: python-embed not found. Run scripts/prepare-python.sh first."
    exit 1
fi

RESOURCES="$APP_PATH/Contents/Resources"
mkdir -p "$RESOURCES"

echo "Bundling Python runtime..."
rm -rf "$RESOURCES/python"
cp -R "$PROJECT_DIR/python-embed" "$RESOURCES/python"

echo "Bundling backend source..."
rm -rf "$RESOURCES/backend"
mkdir -p "$RESOURCES/backend"
rsync -a \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache' \
    --exclude 'requirements-dist.txt' \
    "$PROJECT_DIR/backend/" "$RESOURCES/backend/"

echo "Bundling ffmpeg..."
if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    echo "ERROR: ffmpeg/ffprobe not found in PATH. Install with: brew install ffmpeg"
    exit 1
fi

BIN_DIR="$RESOURCES/bin"
LIB_DIR="$RESOURCES/lib"
rm -rf "$BIN_DIR" "$LIB_DIR"
mkdir -p "$BIN_DIR" "$LIB_DIR"

copy_with_deps() {
    local binary="$1"
    local dest_name="$2"
    local dest="$BIN_DIR/$dest_name"

    cp "$binary" "$dest"
    chmod +x "$dest"

    local deps
    deps=$(otool -L "$dest" | tail -n +2 | awk '{print $1}' || true)
    while IFS= read -r dep; do
        [ -z "$dep" ] && continue
        case "$dep" in
            /usr/lib/*|/System/*|@*) continue ;;
        esac
        if [ ! -f "$dep" ]; then
            continue
        fi
        local lib_name
        lib_name=$(basename "$dep")
        if [ ! -f "$LIB_DIR/$lib_name" ]; then
            cp "$dep" "$LIB_DIR/$lib_name"
            install_name_tool -id "@rpath/$lib_name" "$LIB_DIR/$lib_name" 2>/dev/null || true
        fi
        install_name_tool -change "$dep" "@rpath/$lib_name" "$dest" 2>/dev/null || true
    done <<< "$deps"

    install_name_tool -add_rpath "@executable_path/../lib" "$dest" 2>/dev/null || true
}

copy_with_deps "$(command -v ffmpeg)" "ffmpeg"
copy_with_deps "$(command -v ffprobe)" "ffprobe"

echo "Bundle complete: $APP_PATH"
du -sh "$RESOURCES/python" "$RESOURCES/backend" "$RESOURCES/bin" "$RESOURCES/lib" 2>/dev/null || true
