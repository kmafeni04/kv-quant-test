"""Multi-turn tool runner with per-step loop detection and a compaction wall.

One runner drives W4 (ledger), W5 (search) and the model-driven compaction variant. Per step it
records the call, its arguments, whether it did anything (state hash before/after), the token
count observed at that request, and any loop. The runner halts on: no tool call (`completed`), a
step guard, a loop, the harness context wall, or the 64k response ceiling.
"""
import json
import re
from collections import Counter

from . import config

LOOP_IDENTICAL = 2
LOOP_REPEAT_X3 = 3
OSCILLATION_CYCLES = 3
NO_PROGRESS_WINDOW = 6


def signature(name, args):
    return name + ":" + json.dumps(args or {}, sort_keys=True, separators=(",", ":"))


def loop_scan(signatures, state_hashes):
    """First loop condition over the transcript so far -> (flagged, reason, step, span)."""
    n = len(signatures)
    if n >= 2 and signatures[-1] == signatures[-2]:
        return True, "identical_consecutive_call", n - 1, signatures[-1]
    if n >= 2 * OSCILLATION_CYCLES:
        tail = signatures[-2 * OSCILLATION_CYCLES:]
        a, b = tail[0], tail[1]
        if a != b and all(tail[i] == (a if i % 2 == 0 else b) for i in range(len(tail))):
            return True, "oscillation", n - 1, f"{a} <-> {b}"
    counts = Counter(signatures)
    sig, count = counts.most_common(1)[0]
    if count >= LOOP_REPEAT_X3:
        return True, f"identical_call_x{count}", n - 1, sig
    if n >= NO_PROGRESS_WINDOW:
        window = state_hashes[-NO_PROGRESS_WINDOW:]
        if len(set(window)) == 1:
            return True, f"no_progress_{NO_PROGRESS_WINDOW}", n - 1, window[0]
    return False, "", None, None


def _normalize_calls(calls):
    out = []
    for c in calls or []:
        fn = c.get("function", c) if isinstance(c, dict) else {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:                                      # noqa: BLE001
                args = {"_raw": args}
        out.append({"name": fn.get("name", ""), "args": args or {}, "id": c.get("id")})
    return out


def _context_full(r, ctx):
    if r.get("error") and re.search(r"context|n_tokens|too long|exceed", r["error"], re.I):
        return True
    usage = r.get("usage") or {}
    used = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    return bool(ctx) and used >= ctx


def compact_tool_spec():
    return {"type": "function", "function": {
        "name": "compact",
        "description": (f"Replace the conversation with a handover brief of at most "
                        f"{config.BRIEF_WORD_CAP} words. Everything outside the brief is dropped."),
        "parameters": {"type": "object", "properties": {
            "brief": {"type": "string", "description": "the handover brief, verbatim"}},
            "required": ["brief"]}}}


def compact_messages(messages, brief, step):
    """Drop the transcript and re-insert the brief verbatim; returns the compaction event."""
    before_chars = sum(len(m.get("content") or "") for m in messages)
    messages[:] = [{"role": "user",
                    "content": (f"Handover brief, verbatim:\n\n{brief}\n\n"
                                "Continue the task using only this brief and the tools.")}]
    words = len(re.findall(r"\S+", brief or ""))
    return {"step": step, "brief_words": words, "within_cap": words <= config.BRIEF_WORD_CAP,
            "before_chars": before_chars, "after_chars": len(messages[0]["content"]),
            "brief": brief}


def run_tool_loop(chat, messages, tools, dispatch, guard, max_tokens=None, ctx=None,
                  ctx_wall=None):
    """Drive one episode. `dispatch(name, args, messages)` -> {ok, payload, diff, end}.

    Returns content, reasoning, transcript, compaction events, halt reason and token position.
    """
    max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
    transcript, compactions, errors = [], [], []
    signatures, state_hashes = [], []
    content, reasoning = "", ""
    prompt_total = 0
    out_tokens = 0
    last_used = 0
    halted_by, halt_pos, loop = "completed", None, None
    for step in range(guard):
        if ctx_wall and ctx and last_used >= ctx_wall * ctx and not compactions:
            halted_by, halt_pos = "context_full", last_used
            break
        r = chat(messages, max_tokens=max_tokens, tools=tools)
        if r.get("error"):
            errors.append(r["error"])
        usage = r.get("usage") or {}
        prompt_total = usage.get("prompt_tokens") or prompt_total
        out_tokens += usage.get("completion_tokens") or 0
        used = prompt_total + (usage.get("completion_tokens") or 0)
        last_used = used
        content += r.get("content") or ""
        reasoning += r.get("reasoning_content") or ""
        if r.get("error"):
            halted_by = "context_full" if _context_full(r, ctx) else "error"
            halt_pos = used or prompt_total
            break
        if usage.get("completion_tokens") and usage["completion_tokens"] >= max_tokens:
            halted_by, halt_pos = "ceiling", used
            break
        calls = _normalize_calls(r.get("tool_calls"))
        if not calls:
            halted_by = "completed"
            break
        messages.append({"role": "assistant", "content": r.get("content") or "",
                         "tool_calls": [{"id": c["id"] or f"call_{step}_{i}",
                                         "type": "function",
                                         "function": {"name": c["name"],
                                                      "arguments": json.dumps(c["args"])}}
                                        for i, c in enumerate(calls)]})
        end = False
        for i, c in enumerate(calls):
            cid = c["id"] or f"call_{step}_{i}"
            before = dispatch.state_hash()
            try:
                res = dispatch(c["name"], c["args"] or {}, messages) or {}
            except Exception as e:                                 # noqa: BLE001
                res = {"ok": False, "payload": f"tool error: {type(e).__name__}: {e}"[:300]}
            after = dispatch.state_hash()
            if c["name"] == "compact" and res.get("ok"):
                compactions.append(res.get("compaction") or {})
            compacted = c["name"] == "compact" and res.get("ok")
            if not compacted:
                messages.append({"role": "tool", "tool_call_id": cid,
                                 "content": str(res.get("payload", ""))[:20000]})
            signatures.append(signature(c["name"], c["args"]))
            state_hashes.append(after)
            flagged, why, at, span = loop_scan(signatures, state_hashes)
            transcript.append(dict(step=step, tool=c["name"], args=c["args"], ok=bool(res.get("ok")),
                                   diff=res.get("diff"), state=after,
                                   prompt_tokens=prompt_total,
                                   out_tokens=usage.get("completion_tokens"),
                                   token_pos=used, loop=why if flagged else None,
                                   span=span if flagged else None))
            end = end or bool(res.get("end")) or compacted
            if flagged:
                loop = dict(reason=why, step=len(transcript) - 1, token_pos=used, span=span)
                halted_by, halt_pos = "loop", used
                end = True
                break
        if end:
            break
    else:
        halted_by, halt_pos = "guard", last_used
    return dict(content=content, reasoning=reasoning, transcript=transcript,
                compactions=compactions, errors=errors, halted_by=halted_by,
                halt_token_pos=halt_pos, prompt_tokens=prompt_total,
                out_tokens=out_tokens, loop=loop, tools_used=len(transcript))


class ToolBox:
    """Minimal container: tool specs + a dispatch callable + a state hash."""

    def __init__(self, tools, handler, state_hash):
        self.tools = tools
        self._handler = handler
        self._state_hash = state_hash

    def __call__(self, name, args, messages):
        return self._handler(name, args, messages)

    def state_hash(self):
        return self._state_hash()
