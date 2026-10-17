#!/usr/bin/env python3
"""Keep the pinned 2.2081 install launchable after RobTop pushes a new Steam build.

The game is pinned by locking appmanifest_322170.acf (scripts/pin-gd-2.2081.sh). Steam launches a
game only when the manifest's buildid and depot manifest match the current public branch; when
a new build appears it flags "Update Required", fails to write the locked manifest, and refuses to
launch (AppError_11, "Disk write failure" in content_log.txt). This script reads the current public
buildid and depot manifest from Steam's own appinfo.vdf cache and writes them into the locked
manifest, so Steam believes the install is current and launches the pinned binary unchanged.

    python3 scripts/refresh-steam-manifest.py            # apply
    python3 scripts/refresh-steam-manifest.py --dry-run  # show what would change

Run it with Steam running (so appinfo.vdf is fresh) and the game closed. Nothing in the game
folder is touched; only the manifest text changes, and it stays locked afterwards.
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
from pathlib import Path

APP_ID = 322170
DEPOT_ID = 322172
STEAM = Path.home() / "Library/Application Support/Steam"
APPINFO = STEAM / "appcache/appinfo.vdf"
MANIFEST = STEAM / f"steamapps/appmanifest_{APP_ID}.acf"


def read_appinfo(path: Path, app_id: int) -> dict:
    """Parses the binary appinfo.vdf (v29 string-table format) and returns one app's tree."""
    data = path.read_bytes()
    magic, _universe, table_off = struct.unpack_from("<IIq", data, 0)
    if magic != 0x07564429:
        raise SystemExit(f"unexpected appinfo.vdf magic {magic:#x}; the format changed, update this script")
    count = struct.unpack_from("<I", data, table_off)[0]
    strings, pos = [], table_off + 4
    for _ in range(count):
        end = data.index(b"\x00", pos)
        strings.append(data[pos:end].decode("utf-8", "replace"))
        pos = end + 1

    def parse_map(pos: int) -> tuple[dict, int]:
        out: dict = {}
        while True:
            t = data[pos]
            pos += 1
            if t == 0x08:
                return out, pos
            key = strings[struct.unpack_from("<i", data, pos)[0]]
            pos += 4
            if t == 0x00:
                out[key], pos = parse_map(pos)
            elif t == 0x01:
                end = data.index(b"\x00", pos)
                out[key] = data[pos:end].decode("utf-8", "replace")
                pos = end + 1
            elif t == 0x02:
                out[key] = struct.unpack_from("<i", data, pos)[0]
                pos += 4
            elif t == 0x07:
                out[key] = struct.unpack_from("<q", data, pos)[0]
                pos += 8
            else:
                raise SystemExit(f"unknown appinfo node type {t} at {pos}")

    pos = 16
    while pos < table_off:
        appid, size = struct.unpack_from("<II", data, pos)
        if appid == 0:
            break
        if appid == app_id:
            node, _ = parse_map(pos + 8 + 4 + 4 + 8 + 20 + 4 + 20)
            return node.get("appinfo", node)
        pos += 8 + size
    raise SystemExit(f"app {app_id} not in {path}")


def set_field(text: str, key: str, value: str) -> str:
    new, n = re.subn(rf'("{key}"\s+)"[^"]*"', rf'\g<1>"{value}"', text, count=1)
    if n != 1:
        raise SystemExit(f"field {key} not found in manifest")
    return new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    app = read_appinfo(APPINFO, APP_ID)
    depots = app["depots"]
    buildid = str(depots["branches"]["public"]["buildid"])
    gid = str(depots[str(DEPOT_ID)]["manifests"]["public"]["gid"])

    text = MANIFEST.read_text()
    have_build = re.search(r'"buildid"\s+"(\d+)"', text).group(1)
    have_gid = re.search(r'"manifest"\s+"(\d+)"', text).group(1)
    print(f"manifest: buildid {have_build}, depot manifest {have_gid}")
    print(f"steam public branch: buildid {buildid}, depot manifest {gid}")
    if (have_build, have_gid) == (buildid, gid):
        print("already current; nothing to do")
        return 0

    new = set_field(text, "buildid", buildid)
    new = set_field(new, "TargetBuildID", buildid)
    new = set_field(new, "manifest", gid)
    if args.dry_run:
        print("dry run; would rewrite", MANIFEST)
        return 0

    subprocess.run(["chflags", "nouchg", str(MANIFEST)], check=True)
    try:
        MANIFEST.write_text(new)
    finally:
        subprocess.run(["chflags", "uchg", str(MANIFEST)], check=True)
    print(f"rewrote {MANIFEST} and locked it again; launch with: open steam://rungameid/{APP_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
