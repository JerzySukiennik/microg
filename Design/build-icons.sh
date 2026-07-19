#!/bin/bash
# Build every raster asset from the SVG sources.
#
# No dependencies: qlmanage is macOS QuickLook (it renders SVG natively) and
# sips is the built-in image tool. Render once at 1024 and downscale from
# there — rasterising each size separately gives slightly different stroke
# rounding, which reads as a wobble when the icon animates between sizes.
set -euo pipefail
cd "$(dirname "$0")"

OUT=assets
ICONSET="$OUT/MicroG.iconset"
rm -rf "$OUT" && mkdir -p "$ICONSET"

render() {  # svg -> png at 1024
  local svg=$1 dest=$2 tmp
  tmp=$(mktemp -d)
  qlmanage -t -s 1024 -o "$tmp" "$svg" >/dev/null 2>&1
  mv "$tmp/$(basename "$svg").png" "$dest"
  rm -rf "$tmp"
}

render microg-icon.svg "$OUT/icon-1024.png"
render microg-mark.svg "$OUT/mark-1024.png"
render microg-mark-inverse.svg "$OUT/mark-inverse-1024.png"

# macOS iconset: each logical size plus its @2x retina counterpart.
for pair in "16 32" "32 64" "128 256" "256 512" "512 1024"; do
  set -- $pair
  sips -Z "$1" "$OUT/icon-1024.png" --out "$ICONSET/icon_${1}x${1}.png" >/dev/null
  sips -Z "$2" "$OUT/icon-1024.png" --out "$ICONSET/icon_${1}x${1}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$OUT/MicroG.icns"

# Standalone PNGs for Electron's window icon, the README and the favicon.
for s in 512 256 128 64 32; do
  sips -Z "$s" "$OUT/icon-1024.png" --out "$OUT/icon-${s}.png" >/dev/null
done
sips -Z 32 "$OUT/icon-1024.png" --out "$OUT/favicon-32.png" >/dev/null

echo "built:"
ls -1sh "$OUT" | sed 's/^/  /'
echo "  $(ls -1 "$ICONSET" | wc -l | tr -d ' ') files in MicroG.iconset"
