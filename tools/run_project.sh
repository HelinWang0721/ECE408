#!/usr/bin/env bash
# run_project.sh [sim|gpu] [base op1 ... new-forward]
# Builds Project/test/test_ops.cu once per implementation and runs the
# correctness suite (and, in gpu mode, the batch-5000 benchmark).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-sim}"; shift || true
OPS=("$@"); [ ${#OPS[@]} -eq 0 ] && OPS=(base op1 op2 op3 op4 op5 new-forward)
OUT="${BUILD_DIR:-$ROOT/build}/$MODE"; mkdir -p "$OUT"
declare -A RES
for op in "${OPS[@]}"; do
  f="../$op.cu"; [ "$op" = new-forward ] && f="../custom/new-forward.cu"
  tol=1e-3; [ "$op" = op3 ] && tol=2e-2
  bin="$OUT/test_$op"
  if [ "$MODE" = sim ]; then
    # the op file is #included by the driver, so translate its <<<>>> launches first
    python3 "$ROOT/tools/cudasim/cudasim.py" "$ROOT/Project/test/$f" "$OUT/$op.sim.cu"
    "$ROOT/tools/cudasim/build.sh" "$ROOT/Project/test/test_ops.cu" "$bin" -I "$ROOT/Project/custom" "-DOP_FILE=\"$OUT/$op.sim.cu\"" -DTOL=$tol || { RES[$op]=BUILD-FAIL; continue; }
    ( cd "$ROOT/Project/test" && "$bin" 2 ) > "$OUT/$op.log" 2>&1
  else
    nvcc -std=c++17 -O3 -arch="${NVCC_ARCH:-native}" -I "$ROOT/Project/custom" "-DOP_FILE=\"$f\"" -DTOL=$tol "$ROOT/Project/test/test_ops.cu" -o "$bin" || { RES[$op]=BUILD-FAIL; continue; }
    ( "$bin" && "$bin" bench 5000 ) > "$OUT/$op.log" 2>&1
  fi
  if grep -q "\[cudasim\] ERROR" "$OUT/$op.log"; then RES[$op]="FAIL (emulator error)";
  elif grep -q "RESULT FAIL" "$OUT/$op.log" || ! grep -q "RESULT PASS" "$OUT/$op.log"; then RES[$op]=FAIL; else RES[$op]=PASS; fi
  grep -q WARNING "$OUT/$op.log" && RES[$op]="${RES[$op]} (+warnings)"
done
echo "===== Project ($MODE) ====="
for op in "${OPS[@]}"; do printf "%-12s %s   log: %s\n" "$op" "${RES[$op]}" "$OUT/$op.log"; done
