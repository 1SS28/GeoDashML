#!/usr/bin/env bash
# Pin the Steam macOS build of Geometry Dash at 2.2081, the version Geode 5.11.2 targets.
#
# Before running: in the Steam console (open steam://open/console) run
#   download_depot 322170 322172 8780295549609362522
# and wait for it to finish. Then run this script with the game closed.
#
# What it does:
#   1. verifies the downloaded binary's UUIDs against Geode's own 2.2081 table,
#   2. moves the currently installed app to ~/Library/Application Support/GeoDashML/backups,
#   3. copies the 2.2081 app into the Steam install folder,
#   4. drops the spike mod into the app's geode/mods folder if it has been built,
#   5. locks appmanifest_322170.acf so Steam cannot update the game.
#
# To undo: chflags nouchg "$STEAM/steamapps/appmanifest_322170.acf", then in Steam
# right-click the game, Properties, Installed Files, Verify integrity of game files.
set -euo pipefail

STEAM="$HOME/Library/Application Support/Steam"
COMMON="$STEAM/steamapps/common/Geometry Dash"
DEPOT="$STEAM/Steam.AppBundle/Steam/Contents/MacOS/steamapps/content/app_322170/depot_322172"
MANIFEST="$STEAM/steamapps/appmanifest_322170.acf"
BACKUPS="$HOME/Library/Application Support/GeoDashML/backups"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MOD="$REPO/mods/spike/build/sanjey.gdml_spike.geode"

# From the Geode SDK: loader/src/platform/mac/LoaderImpl.mm, uuidToVersionMap()
WANT_ARM64="ABCA9EC8-DDEE-317A-BC32-3F41A842311D"
WANT_X86="3FBF9F72-5A83-3918-A82A-37BD3C019D7A"

if pgrep -f "Geometry Dash.app/Contents/MacOS/Geometry Dash" >/dev/null; then
    echo "Geometry Dash is running. Quit it first." >&2
    exit 1
fi
if [ ! -d "$DEPOT/Geometry Dash.app" ]; then
    echo "Depot not found at: $DEPOT" >&2
    echo "Run the download_depot command in the Steam console first." >&2
    exit 1
fi

uuids="$(dwarfdump --uuid "$DEPOT/Geometry Dash.app/Contents/MacOS/Geometry Dash")"
if ! grep -q "$WANT_ARM64" <<<"$uuids" || ! grep -q "$WANT_X86" <<<"$uuids"; then
    echo "Downloaded binary is NOT 2.2081. Refusing to install it:" >&2
    echo "$uuids" >&2
    exit 1
fi
echo "Depot binary verified as 2.2081."

installed="$(dwarfdump --uuid "$COMMON/Geometry Dash.app/Contents/MacOS/Geometry Dash" 2>/dev/null || true)"
if grep -q "$WANT_ARM64" <<<"$installed"; then
    echo "Installed game is already 2.2081."
else
    mkdir -p "$BACKUPS"
    ver="$(defaults read "$COMMON/Geometry Dash.app/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null || echo unknown)"
    dest="$BACKUPS/Geometry Dash $ver.app"
    if [ -e "$dest" ]; then
        dest="$BACKUPS/Geometry Dash $ver $(date +%Y%m%d-%H%M%S).app"
    fi
    mv "$COMMON/Geometry Dash.app" "$dest"
    echo "Previous app ($ver) moved to: $dest"
    ditto "$DEPOT/Geometry Dash.app" "$COMMON/Geometry Dash.app"
    echo "2.2081 copied into place."
fi

mkdir -p "$COMMON/Geometry Dash.app/Contents/geode/mods"
if [ -f "$MOD" ]; then
    cp "$MOD" "$COMMON/Geometry Dash.app/Contents/geode/mods/"
    echo "Spike mod copied into the mods folder."
fi

chflags uchg "$MANIFEST"
echo "Locked $MANIFEST so Steam cannot update the game."
echo
echo "Installed binary now:"
dwarfdump --uuid "$COMMON/Geometry Dash.app/Contents/MacOS/Geometry Dash"
echo
echo "Done. Next: run the Geode installer pkg, then launch the game."
