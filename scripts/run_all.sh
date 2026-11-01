#!/usr/bin/env bash
# Full pipeline: preflight -> models -> selftest -> fit -> matrix -> report, with a trap that
# restarts whatever was holding the GPU when the script started.
#
#   ./scripts/run_all.sh                 # full study
#   ./scripts/run_all.sh --smoke         # 10-minute end-to-end proof on one model
#   DEADLINE_HOURS=22 ./scripts/run_all.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAMP="${STAMP:-$(date -u +%Y-%m-%dT%H:%M)}"
DEADLINE_HOURS="${DEADLINE_HOURS:-22}"
MODE="${1:-}"

STRATA_CMD_FILE="$ROOT/results/$STAMP/strata-cmd.txt"

restore_gpu_holder() {
  [[ -s "$STRATA_CMD_FILE" ]] || return 0
  local cwd args
  cwd="$(sed -n 1p "$STRATA_CMD_FILE")"
  args="$(sed -n 2p "$STRATA_CMD_FILE")"
  [[ -n "$args" ]] || return 0
  if pgrep -f "strata --serve" >/dev/null; then
    echo "[restore] strata --serve already running"
    return 0
  fi
  echo "[restore] relaunching the GPU holder from $cwd"
  ( cd "$cwd" && nohup setsid $args >/tmp/strata-restore.log 2>&1 & )
  sleep 10
  if pgrep -f "strata --serve" >/dev/null; then
    echo "[restore] strata is back (pid $(pgrep -f 'strata --serve' | head -1))"
  else
    echo "[restore] WARNING: strata did not come back - see /tmp/strata-restore.log"
  fi
}
trap restore_gpu_holder EXIT

if [[ "$MODE" == "--smoke" ]]; then
  python3 -m kvtest.selftest || exit 1
  python3 -m kvtest.run --smoke --resume || exit 1
  python3 -m kvtest.report results/smoke || exit 1
  echo "smoke OK: results/smoke/report.html"
  exit 0
fi

bash scripts/preflight.sh "$STAMP" || exit 1
bash scripts/download_models.sh || exit 1
python3 -m kvtest.selftest || exit 1

# stop whatever holds the GPU (captured by preflight), waiting for the memory to come back
HOLDER=$(python3 - "$ROOT/results/$STAMP/gpu-holders.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except OSError:
    raise SystemExit(0)
for p in d.get("processes", []):
    if "strata" in p["args"] and "--serve" in p["args"]:
        print(p["pid"])
        break
PY
)
if [[ -n "${HOLDER:-}" ]]; then
  echo "[gpu] stopping pid $HOLDER"
  kill -TERM "$HOLDER" 2>/dev/null
  for i in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [[ "$used" -lt 1024 ]] && break
    sleep 1
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [[ "$used" -ge 1024 ]]; then
    echo "[gpu] still ${used} MiB after 120 s, escalating"
    kill -KILL "$HOLDER" 2>/dev/null
    sleep 5
  fi
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee "$ROOT/results/$STAMP/vram-free-mib.txt"

python3 -m kvtest.fit --stamp "$STAMP" --ladder 102400 || exit 1
python3 -m kvtest.run --stamp "$STAMP" --resume --deadline-hours "$DEADLINE_HOURS" \
  > "$ROOT/results/$STAMP/run.out" 2>&1
python3 -m kvtest.report "results/$STAMP"
echo "done: results/$STAMP/REPORT.md and report.html"
