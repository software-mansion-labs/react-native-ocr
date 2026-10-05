#!/usr/bin/env bash
# Copies the CRAFT heatmap fixtures the example app's EasyOCR screen saved
# ("Save detector heatmap") from the booted iOS simulator or the connected
# Android device into cpp/tests/fixtures/craft, where CraftParityTest reads them.
#
#   scripts/pull-craft-fixtures.sh ios        # from the booted simulator
#   scripts/pull-craft-fixtures.sh android    # from the first adb device
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/cpp/tests/fixtures/craft"
APP_ID="com.swmansion.rnocr.example"
NAMES=(sample vertical)

mkdir -p "$DEST"
case "${1:-ios}" in
  ios)
    DOCS="$(xcrun simctl get_app_container booted "$APP_ID" data)/Documents"
    for name in "${NAMES[@]}"; do
      if [[ -f "$DOCS/$name.heat" ]]; then
        cp "$DOCS/$name.heat" "$DOCS/$name.json" "$DEST/"
        echo "pulled $name"
      fi
    done
    ;;
  android)
    for name in "${NAMES[@]}"; do
      for ext in heat json; do
        if adb shell run-as "$APP_ID" test -f "files/$name.$ext"; then
          adb shell run-as "$APP_ID" cat "files/$name.$ext" > "$DEST/$name.$ext"
        fi
      done
      [[ -f "$DEST/$name.heat" ]] && echo "pulled $name"
    done
    ;;
  *)
    echo "usage: $0 [ios|android]" >&2
    exit 2
    ;;
esac
ls -l "$DEST"
