#!/usr/bin/env bash
# create-dmg.sh
# Package a signed .app bundle into a compressed DMG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

APP_PATH="${1:?Usage: $0 <path-to-app.bundle> [version]}"

# Version priority: explicit arg 2 (e.g. the release tag — authoritative) >
# the built app's CFBundleShortVersionString (read with PlistBuddy, which is
# reliable for an arbitrary plist path, unlike `defaults read`) > hard error.
# Never silently default to a fixed version: that previously mislabelled the
# DMG (a failed read fell back to "0.1.0" regardless of the real version).
VERSION="${2:-}"
if [[ -z "$VERSION" ]]; then
    VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
        "$APP_PATH/Contents/Info.plist" 2>/dev/null || true)"
fi
if [[ -z "$VERSION" ]]; then
    echo "ERROR: could not determine app version. Pass it as arg 2, or ensure" \
        "CFBundleShortVersionString is set in $APP_PATH/Contents/Info.plist" >&2
    exit 1
fi

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
