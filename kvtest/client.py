"""HTTP client for llama-server: health, one chat completion, tokenization.

Everything the harness needs from a response is captured here once: content, reasoning_content,
tool_calls, usage, llama.cpp `timings`. A request failure becomes a dict with `error`, never an
exception that kills an arm.
"""
import json
import re
import time
import urllib.error
import urllib.request


def _req(url, data=None, timeout=60, method=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def health(base, timeout=2):
    try:
        return _req(f"{base}/health", timeout=timeout)
    except Exception:
        return None


def wait_ready(base, timeout=600.0, proc=None, interval=2.0):
    """Wait for {"status":"ok"}. Returns (ok, seconds_waited, last_payload)."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False, time.time() - t0, {"status": "process_exited", "rc": proc.returncode}
        try:
            last = health(base, timeout=3)
        except Exception as e:                                    # noqa: BLE001
            last = {"status": f"error: {type(e).__name__}"}
        if isinstance(last, dict) and last.get("status") == "ok":
            return True, time.time() - t0, last
        time.sleep(interval)
    return False, time.time() - t0, last


def _timings(d):
    t = d.get("timings") or {}
    out = {}
    for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
              "predicted_per_second", "cache_n"):
        if k in t:
            out[k] = t[k]
    return out


def chat(base, messages, max_tokens=1024, timeout=1800, tools=None, **sampling):
    """One non-streaming chat completion. Sampling kwargs are sent per request."""
    payload = {"messages": messages, "max_tokens": max_tokens, "stream": False,
               "cache_prompt": True}
    payload.update({k: v for k, v in sampling.items() if v is not None})
    if tools:
        payload["tools"] = tools
    t0 = time.time()
    try:
        d = _req(f"{base}/v1/chat/completions", payload, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:400]
        except Exception:                                          # noqa: BLE001
            pass
        return {"error": f"HTTP {e.code}: {detail}", "wall_s": round(time.time() - t0, 2),
                "content": "", "reasoning_content": "", "tool_calls": [], "usage": {},
                "timings": {}}
    except Exception as e:                                         # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:400], "wall_s": round(time.time() - t0, 2),
                "content": "", "reasoning_content": "", "tool_calls": [], "usage": {},
                "timings": {}}
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = d.get("usage") or {}
    tim = _timings(d)
    return {
        "error": None,
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "usage": {"prompt_tokens": usage.get("prompt_tokens"),
                  "completion_tokens": usage.get("completion_tokens"),
                  "total_tokens": usage.get("total_tokens")},
        "timings": tim,
        "prefill_tps": round(tim.get("prompt_per_second") or 0, 1),
        "decode_tps": round(tim.get("predicted_per_second") or 0, 1),
        "wall_s": round(time.time() - t0, 2),
    }


def props(base):
    return _req(f"{base}/props", timeout=30)


def token_count(tokenize_bin, model_path, text, timeout=600):
    """Exact token count for a GGUF tokenizer: llama-tokenize -m M --ids --no-bos -f FILE."""
    import os
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        p = subprocess.run([tokenize_bin, "-m", model_path, "--ids", "--no-bos", "-f", tmp],
                           capture_output=True, text=True, timeout=timeout)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError(f"llama-tokenize produced no ids: {p.stderr[-400:]}")
        # --ids prints one JSON array of token ids per input file
        ids = json.loads(lines[-1]) if lines[-1].lstrip().startswith("[") else \
            [int(x) for x in re.findall(r"-?\d+", lines[-1])]
        if not ids:
            raise RuntimeError(f"llama-tokenize produced no ids: {p.stderr[-400:]}")
        return len(ids), ids
    finally:
        os.unlink(tmp)
