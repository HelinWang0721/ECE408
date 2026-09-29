#!/usr/bin/env bash
# run_mps.sh [sim|gpu] [MP1 MP2 ...]
#   sim : build with the CPU emulator (tools/cudasim), no GPU required
#   gpu : build with nvcc and run on the real GPU (set NVCC_ARCH, default sm_70..native)
# Every MP's datasets are run exactly like its run_datasets script and a
# PASS/FAIL table is printed at the end.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-sim}"; shift || true
MPS=("$@"); [ ${#MPS[@]} -eq 0 ] && MPS=(MP1 MP2 MP3 MP4 MP5 MP6 MP7 MP8)
OUT="${BUILD_DIR:-$ROOT/build}/$MODE"; mkdir -p "$OUT"
declare -A RES
for mp in "${MPS[@]}"; do
  bin="$OUT/$mp"
  if [ "$MODE" = sim ]; then
    "$ROOT/tools/cudasim/build.sh" "$ROOT/$mp/template.cu" "$bin" || { RES[$mp]="BUILD-FAIL"; continue; }
  else
    nvcc -std=c++17 -O2 -arch="${NVCC_ARCH:-native}" -I "$ROOT/tools/include" "$ROOT/$mp/template.cu" -o "$bin" || { RES[$mp]="BUILD-FAIL"; continue; }
  fi
  pass=0; fail=0
  # reuse the dataset command lines from the MP's own run_datasets script
  while read -r i; do
    cmd="$(grep -E '^\s*\./template' "$ROOT/$mp/run_datasets" | sed "s#\${i}#$i#g; s#^\s*\./template#$bin#")"
    log="$OUT/$mp.data$i.log"
    (cd "$ROOT/$mp" && timeout 600 bash -c "$cmd") >"$log" 2>&1
    if grep -q "WB_RESULT PASS" "$log" && ! grep -q "\[cudasim\] ERROR" "$log"; then pass=$((pass+1)); else fail=$((fail+1)); echo "  $mp dataset $i FAILED -> $log"; fi
  done < <(grep -oE 'for i in [0-9 ]+' "$ROOT/$mp/run_datasets" | sed 's/for i in //' | tr ' ' '\n' | grep -v '^$')
  RES[$mp]="pass=$pass fail=$fail"
done
echo; echo "===== $MODE summary ====="
for mp in "${MPS[@]}"; do printf "%-5s %s\n" "$mp" "${RES[$mp]}"; done
