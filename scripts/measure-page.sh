#!/usr/bin/env bash
# Times a page of the example app on a device without tapping: opens a screen
# by deep link with `runs=N`, waits for the N runs, prints the warm median and
# the files the run used. The screen writes every run to Documents/last-run.json
# (see "Scripted runs" in CONTRIBUTING.md); a Release build needs no Metro.
#
#   scripts/measure-page.sh <device> '<screen>?<query>' [runs]
#
#   <device>   an iPhone UDID (xcrun devicectl, the app under APP_ID must be
#              installed), `booted` for the iOS simulator, or an adb serial
#   <screen>   easyocr or document, with the screen's own query parameters;
#              modelUrl / helpersUrl / layoutUrl take a URL or a bare file name
#              pushed into the app's documents directory
#   [runs]     runs after load, 5 by default; the first one is dropped as warm-up
#
#   APP_ID=com.swmansion.rnocr.example.dev scripts/measure-page.sh 00008150-… \
#     'document?sample=rotated&backend=COREML&helpersBackend=COREML&modelUrl=pp_ocrv6_coreml_fp16.pte'
#   scripts/measure-page.sh RFCX806ACPY 'easyocr?lang=ENGLISH&backend=XNNPACK&sample=sample' 7
set -euo pipefail

DEVICE="${1:?device: iPhone UDID, 'booted' or an adb serial}"
LINK="${2:?screen?query}"
RUNS="${3:-5}"
APP_ID="${APP_ID:-com.swmansion.rnocr.example}"
URL="rnocr-example://${LINK}&runs=${RUNS}"
OUT="$(mktemp -t last-run).json"
PREV="$(mktemp -t last-run-prev).json"

case "$DEVICE" in
  booted)   KIND=sim ;;
  *-*)      KIND=ios ;;
  *)        KIND=android ;;
esac

pull() { # last-run.json -> $1, or fail
  case "$KIND" in
    ios) xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer \
           --domain-identifier "$APP_ID" --source Documents/last-run.json --destination "$1" >/dev/null 2>&1 ;;
    sim) cp "$(xcrun simctl get_app_container booted "$APP_ID" data)/Documents/last-run.json" "$1" 2>/dev/null ;;
    android) adb -s "$DEVICE" shell run-as "$APP_ID" cat files/last-run.json > "$1" 2>/dev/null && [[ -s "$1" ]] ;;
  esac
}

pull "$PREV" || : > "$PREV"
case "$KIND" in
  ios)
    # A cold start swallows the payload URL, so launch first and route second.
    xcrun devicectl device process launch --terminate-existing --device "$DEVICE" "$APP_ID" >/dev/null 2>&1
    sleep 4
    xcrun devicectl device process launch --device "$DEVICE" --payload-url "$URL" "$APP_ID" >/dev/null 2>&1 ;;
  sim)
    xcrun simctl terminate booted "$APP_ID" >/dev/null 2>&1 || true
    xcrun simctl launch booted "$APP_ID" >/dev/null 2>&1
    sleep 4
    xcrun simctl openurl booted "$URL" ;;
  android)
    adb -s "$DEVICE" shell am force-stop "$APP_ID"
    adb -s "$DEVICE" shell am start -W -a android.intent.action.VIEW -d "'$URL'" "$APP_ID" >/dev/null ;;
esac

for _ in $(seq 1 120); do
  sleep 4
  pull "$OUT" || continue
  if python3 - "$OUT" "$PREV" "$RUNS" "${LINK%%\?*}" <<'PY'; then break; fi
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
same = open(sys.argv[1], "rb").read() == open(sys.argv[2], "rb").read()
ok = not same and d.get("screen") == sys.argv[4] and len(d.get("msAll", [])) >= int(sys.argv[3])
sys.exit(0 if ok else 1)
PY
done

python3 - "$OUT" <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))
g = d.get("grade") or {}
ms = d.get("msAll", [])
if not ms:
    sys.exit("no runs recorded; is the app installed, the screen name right, the device unlocked?")
warm = ms[1:] or ms
grade = "%s/%s" % (g.get("found"), g.get("total")) if "total" in g else ""
if "charAccuracy" in g:
    grade += " %.1f%% chars" % (100 * g["charAccuracy"])
print("%s %s %s %s %s: warm median %.0f ms, runs %s" % (
    d.get("screen"), d.get("backend"), d.get("helpersBackend", ""), d.get("sample"), grade,
    statistics.median(warm), ms))
for key, path in (d.get("models") or {}).items():
    if path:
        print("  %-8s %s" % (key, path.rsplit("/", 1)[-1]))
PY
rm -f "$OUT" "$PREV"
