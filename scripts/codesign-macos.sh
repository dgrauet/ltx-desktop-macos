#!/bin/bash
# Code signing script for LTX Desktop macOS application.
# Adapted from: https://github.com/audiohacking/AceForge/blob/main/build/macos/codesign.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

APP_PATH="${1:?Usage: $0 <path-to-app.bundle>}"
SIGNING_IDENTITY="${MACOS_SIGNING_IDENTITY:--}"
ENTITLEMENTS_PATH="$PROJECT_DIR/build/macos/entitlements.plist"

echo "=================================================="
echo "LTX Desktop macOS Code Signing"
echo "=================================================="
echo "App path: $APP_PATH"
echo "Signing identity: $SIGNING_IDENTITY"
echo "Entitlements: $ENTITLEMENTS_PATH"
echo ""

if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    exit 1
fi

if [ ! -f "$ENTITLEMENTS_PATH" ]; then
    echo "Error: Entitlements file not found at $ENTITLEMENTS_PATH"
    exit 1
fi

sign_binary() {
    local target="$1"
    echo "Signing: $target"

    local cmd=(
        xcrun codesign
        --sign "$SIGNING_IDENTITY"
        --force
        --options runtime
        --entitlements "$ENTITLEMENTS_PATH"
        --deep
    )

    if [ "$SIGNING_IDENTITY" != "-" ]; then
        cmd+=(--timestamp)
    fi

    cmd+=("$target")

    if "${cmd[@]}"; then
        echo "✓ Successfully signed: $target"
        return 0
    else
        echo "✗ Failed to sign: $target"
        return 1
    fi
}

echo "Step 1: Signing frameworks and libraries..."
find "$APP_PATH/Contents" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 | while IFS= read -r -d '' lib; do
    sign_binary "$lib" || true
done

if [ -d "$APP_PATH/Contents/Frameworks" ]; then
    find "$APP_PATH/Contents/Frameworks" -type f -perm -111 -print0 | while IFS= read -r -d '' binary; do
        sign_binary "$binary" || true
    done
fi

echo ""
echo "Step 2: Signing bundled binaries..."
if [ -d "$APP_PATH/Contents/Resources/bin" ]; then
    for exe in "$APP_PATH/Contents/Resources/bin"/*; do
        if [ -f "$exe" ] && [ -x "$exe" ]; then
            sign_binary "$exe" || true
        fi
    done
fi

echo ""
echo "Step 3: Signing main executables..."
for exe in "$APP_PATH/Contents/MacOS"/*; do
    if [ -f "$exe" ] && [ -x "$exe" ]; then
        sign_binary "$exe"
    fi
done

echo ""
echo "Step 4: Signing the app bundle..."
if sign_binary "$APP_PATH"; then
    echo ""
    echo "=================================================="
    echo "✓ Code signing completed successfully!"
    echo "=================================================="
    echo ""
    echo "Verification:"
    xcrun codesign --verify --deep --strict --verbose=2 "$APP_PATH" 2>&1 || true
    echo ""
    echo "Signature info:"
    xcrun codesign -dv --verbose=4 "$APP_PATH" 2>&1 || true
    exit 0
else
    echo ""
    echo "=================================================="
    echo "✗ Code signing failed!"
    echo "=================================================="
    exit 1
fi
