#!/usr/bin/env bash
# Download the official BigStitcher example archive from its OSF project.
# The 3D raw and aligned examples are about 287 MB compressed; they are not
# committed to this repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/data/bigstitcher"
ARCHIVE="$DEST/BigStitcher_Examples.zip"
mkdir -p "$DEST"

if [[ ! -s "$ARCHIVE" ]]; then
  curl -L --fail --retry 3 -o "$ARCHIVE" "https://osf.io/download/tfhqc/"
fi

unzip -q -o "$ARCHIVE" -d "$DEST"
unzip -q -o "$DEST/BigStitcher_Examples/Grid_3d.zip" -d "$DEST/raw3d"
unzip -q -o "$DEST/BigStitcher_Examples/Grid_3d_h5_aligned.zip" -d "$DEST/aligned3d"
echo "Ready: $DEST/aligned3d/grid-3d-stitched-h5/dataset.xml"
