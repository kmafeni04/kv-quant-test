#!/usr/bin/env bash
# Preflight: assert the machine can run the matrix, capture whatever is holding the GPU, and
# record the free-VRAM number into the run stamp. Exits non-zero rather than starting a run that
# cannot finish.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-$(date -u +%Y-%m-%dT%H:%M)}"
OUT="$ROOT/results/$STAMP"
mkdir -p "$OUT/logs"
fail=0
say() { printf '%-42s %s\n' "$1" "$2"; }

# GPU answers at all
if ! command -v nvidia-smi >/dev/null; then
  say "nvidia-smi" "MISSING"; fail=1
else
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  say "GPU total / used MiB" "$total / $used"
  printf '%s\n' "$used" > "$OUT/free-vram-at-start-mib.txt"
  if [[ "$used" -gt 2048 ]]; then
    say "GPU is busy" "something holds ${used} MiB - capture and stop it first"; fail=1
  fi
fi

# no llama-server of ours already running
if pgrep -af 'llama-server.*--port 18400' >/dev/null; then
  say "port 18400" "already served by a llama-server"; fail=1
else
  say "port 18400" "free"
fi

# disk
free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
say "free on \$HOME (GB)" "$free_gb"
[[ "$free_gb" -ge 60 ]] || { say "disk" "need >= 60 GB"; fail=1; }

# no half-written model files left behind
parts=$(find "$HOME/.local/share/llm-models" -maxdepth 3 -name '*.part' -size +0 2>/dev/null | wc -l)
say "incomplete .part files" "$parts"
[[ "$parts" -eq 0 ]] || say "note" "a download is running or was interrupted (resume with -C -)"

# RAM
avail_gb=$(free -g | awk '/^Mem:/{print $7}')
say "available RAM (GB)" "$avail_gb"
[[ "$avail_gb" -ge 24 ]] || { say "RAM" "need >= 24 GB available"; fail=1; }

# llama.cpp build
B=/home/kome/Projects/code/clones/llama.cpp
if [[ -x "$B/build/bin/llama-server" ]]; then
  say "llama-server" "$("$B/build/bin/llama-server" --version 2>&1 | head -2 | tr '\n' ' ')"
  say "llama.cpp commit" "$(git -C "$B" rev-parse --short HEAD)"
  if [[ -n "$(git -C "$B" status --porcelain)" ]]; then
    say "llama.cpp tree" "DIRTY - numbers will not match models.lock.json"; fail=1
  fi
else
  say "llama-server" "missing (run scripts/build_llama.sh)"; fail=1
fi

# python deps: stdlib only, by design
python3 - <<'PY'
import sys
say = lambda a, b: print(f"{a:<42} {b}")
say("python", sys.version.split()[0])
bad = [m for m in ("numpy", "pytest") if __import__("importlib.util", fromlist=["x"]).find_spec(m)]
say("third-party deps", "none" if not bad else f"ignored: {bad}")
PY

# capture anything else on the GPU (the Strata service is the expected holder)
python3 - "$OUT" <<'PY'
import json, subprocess, sys
out = sys.argv[1]
apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                       "--format=csv,noheader"], capture_output=True, text=True).stdout
rec = {"compute_apps": apps.strip().splitlines()}
details = []
for line in rec["compute_apps"]:
    pid = line.split(",")[0].strip()
    if not pid.isdigit():
        continue
    try:
        args = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode().strip()
        cwd = __import__("os").readlink(f"/proc/{pid}/cwd")
    except OSError:
        args, cwd = "", ""
    details.append({"pid": int(pid), "args": args, "cwd": cwd})
rec["processes"] = details
json.dump(rec, open(f"{out}/gpu-holders.json", "w"), indent=2)
strata = [d for d in details if "strata" in d["args"] and "--serve" in d["args"]]
if strata:
    open(f"{out}/strata-cmd.txt", "w").write(
        strata[0]["cwd"] + "\n" + strata[0]["args"] + "\n")
    print(f"{'GPU holder':<42} strata --serve pid {strata[0]['pid']} (captured to strata-cmd.txt)")
PY

echo
if [[ "$fail" -ne 0 ]]; then
  echo "preflight FAILED - do not start the matrix"
  exit 1
fi
echo "preflight OK - stamp $STAMP"
