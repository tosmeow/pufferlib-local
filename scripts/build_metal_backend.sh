#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT/build/metal/libpuffer_metal.dylib}"

mkdir -p "$(dirname "$OUTPUT")"
/usr/bin/clang++ \
    -std=c++17 \
    -O2 \
    -fPIC \
    -fobjc-arc \
    -dynamiclib \
    -Wl,-install_name,@rpath/libpuffer_metal.dylib \
    -I"$ROOT/src/metal" \
    "$ROOT/src/metal/metal_backend.mm" \
    -framework Foundation \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -o "$OUTPUT"

echo "$OUTPUT"
