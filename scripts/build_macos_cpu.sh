#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${1:-breakout}"
shift || true

: "${PYTHON:=.venv/bin/python}"
: "${CC:=/opt/homebrew/opt/llvm/bin/clang -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}"
: "${CXX:=/opt/homebrew/opt/llvm/bin/clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}"

export PYTHON CC CXX
bash build.sh "$ENV_NAME" --cpu "$@"
