#!/usr/bin/env bash
# Fetch the five study GGUFs into ~/.local/share/llm-models/, verify sha256, and record the
# digests in models.lock.json. Idempotent and resumable: a complete, verified file is skipped;
# a partial download continues with `curl -C -`.
#
#   ./scripts/download_models.sh            # fetch + verify + lock
#   ./scripts/download_models.sh --verify   # verify only, no network
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_ROOT="${MODELS_ROOT:-$HOME/.local/share/llm-models}"
H="${HF_ENDPOINT:-https://huggingface.co}"
LOCK="$ROOT/models.lock.json"
VERIFY_ONLY=0
[[ "${1:-}" == "--verify" ]] && VERIFY_ONLY=1

# repo_dir|repo|filename|bytes. An empty repo means the file was already on this host and its
# upstream provenance was not re-verified; its sha256 in models.lock.json is still authoritative.
SPECS=(
  "Qwen3.6-35B-A3B||Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf|22853663008"
  "Qwen3.8-27B-GSQ-RCO||Qwen3.8-27B-GSQ-RCO-IQ3_XXS-MTP.gguf|10442827840"
  "Qwen3.8-Flash-Next-GSQ-RCO||Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf|47039860096"
  "gemma-4-26B-A4B-it-GGUF|ggml-org/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-Q4_0.gguf|14618145824"
  "Ornith-1.5-35B-A3B-GGUF|ornith-ai/Ornith-1.5-35B-A3B-GGUF|Ornith-1.5-35B-Q4_K_M.gguf|21713463040"
)

lock_sha() { # $1 = path -> sha256 from lock, or empty
  [[ -f "$LOCK" ]] || return 0
  python3 - "$LOCK" "$1" <<'PY'
import json, sys
lock, path = json.load(open(sys.argv[1])), sys.argv[2]
for m in lock.get("models", {}).values():
    if m.get("path") == path:
        print(m.get("sha256") or "")
        break
PY
}

needs_download=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r dir repo file bytes <<<"$spec"
  dest="$MODELS_ROOT/$dir/$file"
  [[ "$dir" == gemma-4-26B-A4B-it-GGUF || "$dir" == Ornith-1.5-35B-A3B-GGUF ]] && needs_download=1
  if [[ -f "$dest" ]] && [[ "$(stat -c %s "$dest")" == "$bytes" ]]; then
    echo "present: $file"
  else
    echo "missing/incomplete: $dest"
    [[ $VERIFY_ONLY == 1 ]] && { echo "FAIL: $dest not complete" >&2; exit 1; }
  fi
done

if [[ $VERIFY_ONLY == 0 ]]; then
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r dir repo file bytes <<<"$spec"
    dest="$MODELS_ROOT/$dir/$file"
    mkdir -p "$MODELS_ROOT/$dir"
    if [[ -f "$dest" ]] && [[ "$(stat -c %s "$dest")" == "$bytes" ]]; then
      continue
    fi
    echo "== downloading $file ($bytes B)"
    curl -L --fail --retry 5 --retry-delay 5 -C - -o "$dest.part" \
      "$H/$repo/resolve/main/$file"
    mv -f "$dest.part" "$dest"
    [[ "$(stat -c %s "$dest")" == "$bytes" ]] || { echo "FAIL size $file" >&2; exit 1; }
  done
fi

echo "== sha256"
python3 - "$ROOT" "$MODELS_ROOT" "$LOCK" "${SPECS[@]}" <<'PY'
import hashlib, json, os, subprocess, sys

root, models_root, lock_path = sys.argv[1], sys.argv[2], sys.argv[3]
specs = [s.split("|") for s in sys.argv[4:]]

lock = {"models": {}}
if os.path.exists(lock_path):
    lock = json.load(open(lock_path))
lock.setdefault("models", {})

KEYS = {"Qwen3.6-35B-A3B": "qwen36-35b", "Qwen3.8-27B-GSQ-RCO": "qwen38-27b",
        "Qwen3.8-Flash-Next-GSQ-RCO": "qwen38-flash",
        "gemma-4-26B-A4B-it-GGUF": "gemma4-26b-a4b", "Ornith-1.5-35B-A3B-GGUF": "ornith-35b"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


bad = 0
for repo_dir, repo, file, bytes_ in specs:
    path = os.path.join(models_root, repo_dir, file)
    size = os.path.getsize(path)
    assert size == int(bytes_), f"{file}: {size} != {bytes_}"
    key = KEYS[repo_dir]
    prev = lock["models"].get(key) or {}
    digest = sha256(path)
    marker = path + ".sha256.ok"
    if prev.get("sha256") and prev["sha256"] != digest:
        print(f"MISMATCH {file}\n  lock {prev['sha256']}\n  disk {digest}", file=sys.stderr)
        bad += 1
        continue
    lock["models"][key] = {"path": path, "hf_repo": repo, "filename": file,
                           "bytes": size, "sha256": digest}
    with open(marker, "w") as f:
        f.write(digest + "\n")
    print(f"ok {key:14} {digest}")

if bad:
    sys.exit(1)

lock["document"] = lock.get("document") or {}
llama = subprocess.run(["git", "-C", os.environ.get("LLAMA_DIR",
                        "/home/kome/Projects/code/clones/llama.cpp"),
                        "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
if llama:
    lock["llama_cpp"] = lock.get("llama_cpp") or {}
    lock["llama_cpp"]["commit"] = llama
with open(lock_path, "w") as f:
    json.dump(lock, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"wrote {lock_path}")
PY
echo "download_models: OK"
