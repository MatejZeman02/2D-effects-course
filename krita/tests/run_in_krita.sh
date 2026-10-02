#!/usr/bin/env bash
# Runs krita_smoke.py in a headless Krita and prints its log.
#
#   krita/tests/run_in_krita.sh <unpacked Krita AppImage> <NumPy folder for Krita's Python> [plugin folder]
#
# The plugin folder defaults to krita/pga_filter. Point it at a pga_filter folder
# unpacked from a zip in krita/dist to test what students install, with an
# empty NumPy folder so only the plugin's own NumPy can be found.
#
# Unpack the AppImage once with: krita-6.0.3-x86_64.AppImage --appimage-extract
# Krita's settings and data go to a folder beside the unpacked AppImage, so the
# Krita you paint with is never touched.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
appdir="$(cd "$1" && pwd)"
vendor="$(cd "$2" && pwd)"
plugin="$(cd "${3:-$root/krita/pga_filter}" && pwd)"
state="$(dirname "$appdir")/krita-smoke-state"
mkdir -p "$state/data/kritarunner/pykrita" "$state/config"
ln -sfn "$plugin" "$state/data/kritarunner/pykrita/pga_filter"
ln -sfn "$here/krita_smoke.py" "$state/data/kritarunner/pykrita/krita_smoke.py"
rm -f "$here/smoke.log"
APPDIR="$appdir" LD_LIBRARY_PATH="$appdir/usr/lib" \
XDG_DATA_HOME="$state/data" XDG_CONFIG_HOME="$state/config" \
    timeout 300 "$appdir/usr/bin/kritarunner" -s krita_smoke "$vendor" "$root" >/dev/null 2>&1 || true
cat "$here/smoke.log"
