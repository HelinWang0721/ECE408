#!/usr/bin/env bash
# build.sh <file.cu> <out-binary> [extra g++ flags...]
# Compile a .cu file for the CPU emulator (no GPU / no nvcc needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$1"; out="$2"; shift 2
tmp="$(mktemp --suffix=.cpp)"
python3 "$HERE/cudasim.py" "$src" "$tmp"
g++ -std=c++20 -O1 -g -w -include "$HERE/cuda_sim.h" -I "$HERE/../include" -I "$(dirname "$src")" "$@" -x c++ "$tmp" -o "$out"
rm -f "$tmp"
