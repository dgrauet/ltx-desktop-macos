#!/usr/bin/env bash
# create-dmg.sh
# Package a signed .app bundle into a compressed DMG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

APP_PATH="${1:?Usage: $0 <path-to-app.bundle>}"
VERSION="${2:-$(defaults read "$APP_PATH/Contents/Info" CFBundleShortVersionString 2>/dev/null || echo "0.1.0")}"

APP_NAME="$(basename "$APP_PATH" .app)"
RELEASE_DIR="$PROJECT_DIR/release"
STAGING_DIR="$RELEASE_DIR/dmg-staging"
DMG_PATH="$RELEASE_DIR/${APP_NAME}-${VERSION}.dmg"

rm -rf "$STAGING_DIR" "$DMG_PATH"
mkdir -p "$STAGING_DIR" "$RELEASE_DIR"

cp -R "$APP_PATH" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$STAGING_DIR" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

rm -rf "$STAGING_DIR"

echo "DMG created: $DMG_PATH"
ls -lh "$DMG_PATH"
