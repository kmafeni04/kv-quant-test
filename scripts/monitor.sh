#!/usr/bin/env bash
# One progress snapshot for a running stamp, appended to results/STAMP/monitor.log.
#
#   ./scripts/monitor.sh [STAMP]
#
# Safe to run at any time, including after the run has finished or been cut.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-$(cat "$ROOT/.cache/last-bench-stamp" 2>/dev/null || date -u +%Y-%m-%d)}"
OUT="$ROOT/results/$STAMP"
LOG="$OUT/monitor.log"

[[ -d "$OUT" ]] || { echo "no such stamp: $OUT" >&2; exit 1; }

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') stamp=$STAMP ==="
  if pgrep -f "kvtest.run --stamp $STAMP" >/dev/null; then
    echo "runner: alive (pid $(pgrep -f "kvtest.run --stamp $STAMP" | head -1))"
  else
    echo "runner: not running"
  fi
  python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
expected = {"W1": 1, "W2": 12, "W3": 12, "W4": 6, "W5": 6, "W6": 6}
full = sum(expected.values())
total_done, total_expected, rows = 0, 0, []
for name in sorted(os.listdir(os.path.join(out, "arms")) if os.path.isdir(os.path.join(out, "arms")) else []):
    if not name.endswith(".json"):
        continue
    a = json.load(open(os.path.join(out, "arms", name)))
    done = a.get("done") or []
    n = sum(1 for k in done if k.split(":")[0] in expected)
    exp = full
    total_done += n
    total_expected += exp
    state = "finished" if a.get("finished") else ("cut: " + str(a.get("cut")) if a.get("cut") else "running")
    rows.append(f"  {name[:-5]:28} {n:3}/{exp:<3} {state}")
print("\n".join(rows))
print(f"  {'TOTAL':28} {total_done:3}/{total_expected:<3} instances")
PY
  echo "run.out tail:"
  tail -3 "$OUT/run.out" 2>/dev/null | sed 's/^/  /'
  echo -n "gpu: "
  nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "unavailable"
  echo "ram free: $(free -g | awk '/^Mem:/ {print $7 " GiB"}')"
  echo
} >> "$LOG" 2>&1
tail -"${2:-60}" "$LOG"
