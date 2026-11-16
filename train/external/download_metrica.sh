#!/bin/bash
# Metrica Sports sample-data (github.com/metrica-sports/sample-data).
set -u
OUT=${MTAH_ROOT:-.}/external/metrica/raw
mkdir -p "$OUT"; cd "$OUT"
[ -d sample-data ] || git clone --depth 1 https://github.com/metrica-sports/sample-data.git
cd sample-data && git rev-parse HEAD && du -sh . && find . -type f -not -path './.git/*' | xargs ls -la
echo METRICA_DOWNLOAD_DONE
