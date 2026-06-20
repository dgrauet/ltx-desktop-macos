#!/usr/bin/env bash
# build-release.sh
# Build, bundle, sign, and package LTX Desktop for macOS (Apple Silicon / MLX).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DERIVED_DATA="$PROJECT_DIR/build/DerivedData"
BUILD_PRODUCTS="$DERIVED_DATA/Build/Products/Release"

cd "$PROJECT_DIR"

if [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: MLX builds require Apple Silicon (arm64)."
    exit 1
fi

if [ ! -d "$PROJECT_DIR/python-embed" ]; then
    echo "python-embed not found — running prepare-python.sh..."
    bash "$SCRIPT_DIR/prepare-python.sh"
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg via Homebrew..."
    brew install ffmpeg
fi

echo "Building LTXDesktop (Release, arm64)..."
xcodebuild \
    -project app/LTXDesktop.xcodeproj \
    -scheme LTXDesktop \
    -configuration Release \
    -destination 'platform=macOS,arch=arm64' \
    -derivedDataPath "$DERIVED_DATA" \
    CODE_SIGN_IDENTITY="-" \
    CODE_SIGN_ENTITLEMENTS="$PROJECT_DIR/build/macos/entitlements.plist" \
    ENABLE_HARDENED_RUNTIME=YES \
    build

APP_PATH="$BUILD_PRODUCTS/LTXDesktop.app"
if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: Built app not found at $APP_PATH"
    exit 1
fi

echo "Bundling runtime resources..."
bash "$SCRIPT_DIR/bundle-app.sh" "$APP_PATH"

echo "Code signing app bundle..."
bash "$SCRIPT_DIR/codesign-macos.sh" "$APP_PATH"

echo "Creating DMG..."
bash "$SCRIPT_DIR/create-dmg.sh" "$APP_PATH"

echo ""
echo "Release build complete."
echo "  App: $APP_PATH"
echo "  DMG: $PROJECT_DIR/release/LTXDesktop-"*.dmg
