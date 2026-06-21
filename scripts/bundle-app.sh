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
BUNDLE_TMP="$(mktemp -d)"
PROCESSED_LIST="$BUNDLE_TMP/processed.txt"
: > "$PROCESSED_LIST"

cleanup() {
    rm -rf "$BUNDLE_TMP"
}
trap cleanup EXIT

rm -rf "$BIN_DIR" "$LIB_DIR"
mkdir -p "$BIN_DIR" "$LIB_DIR"

resolve_path() {
    python3 - "$1" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
}

is_system_dep() {
    case "$1" in
        /usr/lib/*|/System/*|@*) return 0 ;;
    esac
    return 1
}

should_bundle_dep() {
    case "$1" in
        /opt/homebrew/*|/usr/local/*) return 0 ;;
    esac
    return 1
}

is_processed() {
    grep -Fxq "$1" "$PROCESSED_LIST" 2>/dev/null
}

mark_processed() {
    printf '%s\n' "$1" >> "$PROCESSED_LIST"
}

rewrite_dep() {
    local target="$1"
    local old_path="$2"
    local lib_name="$3"

    install_name_tool -change "$old_path" "@rpath/$lib_name" "$target" 2>/dev/null || true
}

# Walk the full Mach-O dependency closure: copy every Homebrew dylib transitively
# and rewrite install names so bundled ffmpeg/ffprobe run on machines without brew.
bundle_mach_o_closure() {
    local queue=()
    local item dep dep_resolved lib_name dest

    for item in "$@"; do
        queue+=("$item")
    done

    while [ "${#queue[@]}" -gt 0 ]; do
        local current="${queue[0]}"
        queue=("${queue[@]:1}")

        [ -f "$current" ] || continue
        if is_processed "$current"; then
            continue
        fi
        mark_processed "$current"

        if [[ "$current" == "$BIN_DIR"/* ]]; then
            install_name_tool -add_rpath "@executable_path/../lib" "$current" 2>/dev/null || true
        elif [[ "$current" == "$LIB_DIR"/* ]]; then
            install_name_tool -add_rpath "@loader_path" "$current" 2>/dev/null || true
        fi

        while IFS= read -r dep; do
            [ -z "$dep" ] && continue
            if is_system_dep "$dep"; then
                continue
            fi
            if ! should_bundle_dep "$dep"; then
                continue
            fi
            if [ ! -e "$dep" ]; then
                echo "WARNING: Missing dependency for $(basename "$current"): $dep"
                continue
            fi

            dep_resolved="$(resolve_path "$dep")"
            lib_name="$(basename "$dep_resolved")"
            dest="$LIB_DIR/$lib_name"

            if [ ! -f "$dest" ]; then
                cp "$dep_resolved" "$dest"
                install_name_tool -id "@rpath/$lib_name" "$dest" 2>/dev/null || true
                queue+=("$dest")
            fi

            rewrite_dep "$current" "$dep" "$lib_name"
            if [ "$dep_resolved" != "$dep" ]; then
                rewrite_dep "$current" "$dep_resolved" "$lib_name"
            fi
        done < <(otool -L "$current" | tail -n +2 | awk '{print $1}')
    done
}

verify_bundle_is_self_contained() {
    local file bad
    local failed=0

    for file in "$BIN_DIR"/* "$LIB_DIR"/*; do
        [ -f "$file" ] || continue
        bad="$(otool -L "$file" | tail -n +2 | awk '{print $1}' | grep -E '^/opt/homebrew|^/usr/local' || true)"
        if [ -n "$bad" ]; then
            echo "ERROR: $(basename "$file") still references external libraries:"
            echo "$bad"
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "ERROR: ffmpeg bundle is not self-contained."
        exit 1
    fi
}

copy_and_bundle_binary() {
    local source="$1"
    local dest_name="$2"
    local dest="$BIN_DIR/$dest_name"

    cp "$source" "$dest"
    chmod +x "$dest"
    bundle_mach_o_closure "$dest"
}

copy_and_bundle_binary "$(command -v ffmpeg)" "ffmpeg"
copy_and_bundle_binary "$(command -v ffprobe)" "ffprobe"

echo "Verifying ffmpeg dependency closure..."
verify_bundle_is_self_contained

echo "Bundled $(find "$LIB_DIR" -type f | wc -l | tr -d ' ') dylibs for ffmpeg/ffprobe"

echo "Bundle complete: $APP_PATH"
du -sh "$RESOURCES/python" "$RESOURCES/backend" "$RESOURCES/bin" "$RESOURCES/lib" 2>/dev/null || true
