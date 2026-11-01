"""Tier R (retrieval / tracing / aggregation) and tier L (loop probes).

Every graded prompt ends with the same strict instruction, so grading is a regex and not a judge.
Partial credit is one point per correct sub-value, which keeps `agg_sum` (2 points) and
`niah_multi` (3 points) from being worth the same as a single exact match.
"""
import json
import re
from collections import Counter

from . import config

ANSWER_RE = re.compile(r"^ANSWER:\s*(.+?)\s*$", re.M)


def extract_answer(text):
    """The value of the LAST `ANSWER: ...` line; models sometimes write one inside reasoning."""
    hits = ANSWER_RE.findall(text or "")
    return hits[-1].strip() if hits else None


def _norm(s):
    return re.sub(r"\s+", "", (s or "")).upper()


def _parts(value, n):
    """Split an answer into exactly n comma-separated pieces (missing -> None)."""
    vals = [_norm(v) for v in re.split(r"[,;/|]", value or "") if _norm(v)]
    return (vals + [None] * n)[:n]


def _grade_parts(value, expected):
    got = _parts(value, len(expected))
    pts = sum(1 for g, e in zip(got, [_norm(e) for e in expected]) if g == e)
    ok = pts == len(expected)
    return pts, len(expected), ok, f"{got} vs {expected}"


def _grade_int(value, expected):
    nums = re.findall(r"-?\d+", (value or "").replace(",", ""))
    ok = bool(nums) and int(nums[-1]) == int(expected)
    return (1 if ok else 0), 1, ok, f"{nums[-1] if nums else None} vs {expected}"


# ---------------------------------------------------------------- tier R
def r_tasks(facts):
    """Six graded retrieval/tracing/aggregation tasks with exact-match graders."""
    shards = facts["shards"]
    ans = _answers(facts)
    picked = [shards[4], shards[17], shards[31]]
    deep = max(shards, key=lambda s: s["frac"])
    chain = next(c for c in facts["chains"] if c["head"] == "EMP-0042")
    final = next(r for r in facts["records"] if r["emp"] == chain["hops"][-1])
    n_red, sum_red = ans["agg_sum"]
    proj, proj_n = ans["max_project"]
    out = []

    out.append(dict(
        id="niah_multi", tier="R",
        prompt=("From the document, give the codes of the shards named "
                f"{', '.join(s['name'] for s in picked)} — in exactly this order, comma-separated, "
                "each code in its XY-#### form."),
        expected=[s["code"] for s in picked],
        kind="parts"))

    out.append(dict(
        id="niah_deep", tier="R",
        prompt=(f"The shard record that sits deepest in the document (near its end) — report its "
                f"code, in its XY-#### form."),
        expected=[deep["code"]], kind="parts"))

    out.append(dict(
        id="vt_trace", tier="R",
        prompt=("Starting at the employee EMP-0042, follow the `manager` field 4 hops down the "
                "chain (EMP-0042 -> hop1 -> hop2 -> hop3 -> hop4) and report hop4's badge and "
                "project as `B-####,PRJ-##`."),
        expected=[final["badge"], final["project"]], kind="parts"))

    out.append(dict(
        id="agg_sum", tier="R",
        prompt=("Over the LEDGER rows with `status=OPEN` and `tag=RED` only, report the count of "
                "such rows and the sum of their `amount`, as `<count>,<sum>`."),
        expected=[str(n_red), str(sum_red)], kind="parts"))

    out.append(dict(
        id="count_proj", tier="R",
        prompt=f"How many RECORD rows have project={proj}? Report the integer alone.",
        expected=[str(proj_n)], kind="int"))

    out.append(dict(
        id="fwe_count", tier="R",
        prompt=("How many times does the exact token `ZQ-7731` occur in the document? Count only "
                "whole occurrences of that exact token — not ZQ-77310, ZQ-7731X or ZQ-7732. "
                "Report the integer alone."),
        expected=[str(facts["fwe_count"])], kind="int"))

    for t in out:
        t["prompt"] = t["prompt"] + "\n\n" + config.ANSWER_INSTRUCTION
    return out


def grade_r(task, text):
    value = extract_answer(text)
    if value is None:
        return {"ok": False, "points": 0, "of": len(task["expected"]),
                "why": "no ANSWER line", "answer": None}
    if task["kind"] == "int":
        pts, of, ok, why = _grade_int(value, task["expected"][0])
    else:
        pts, of, ok, why = _grade_parts(value, task["expected"])
    return {"ok": ok, "points": pts, "of": of, "why": why, "answer": value}


def _answers(facts):
    reds = [lg for lg in facts["ledgers"] if lg["status"] == "OPEN" and lg["tag"] == "RED"]
    counts = Counter(r["project"] for r in facts["records"])
    proj, n = max(sorted(counts.items()), key=lambda kv: kv[1])
    return dict(agg_sum=(len(reds), sum(lg["amount"] for lg in reds)),
                max_project=(proj, n))


# ---------------------------------------------------------------- tier L
def l_tasks(facts):
    """Four probes that make idle repetition the thing being measured."""
    ledgers = facts["ledgers"]
    start = 40
    twelve = ledgers[start:start + 12]
    out = []

    out.append(dict(
        id="gen_manual", tier="L",
        prompt=("Write an operator's manual with one short paragraph of 60-120 words for each of "
                "RULE 1 through RULE 12 as stated in the document. Number each paragraph with its "
                "rule. Do not add commentary."),
        expected={"paragraphs": 12},
        check=_check_gen_manual))

    out.append(dict(
        id="copy_ledgers", tier="L",
        prompt=(f"Reproduce the 12 consecutive LEDGER rows of the document that begin at "
                f"{twelve[0]['id']}, verbatim, one row per line, starting from {twelve[0]['id']} "
                "in document order. No numbering, no commentary."),
        expected={"rows": [f"LEDGER {r['id']} status={r['status']} tag={r['tag']} "
                           f"amount={r['amount']}" for r in twelve]},
        check=_check_copy_ledgers))

    out.append(dict(
        id="enumerate_shards", tier="L",
        prompt=("List all 40 shards of the document, one per line, in document order, in the exact "
                "form `<name>=<code>`. No numbering, no headers, no commentary."),
        expected={"rows": [f"{s['name']}={s['code']}" for s in facts["shards"]]},
        check=_check_enumerate_shards))

    out.append(dict(
        id="tool_episode", tier="L",
        prompt=("Use the tools. For each of these shards — orion, lyra, capella, wasat — call "
                "read_shard with its name to retrieve its code, then call submit_report once with "
                "every shard you successfully read, as a list of `<name>=<code>` strings. Do not "
                "invent codes: only report what read_shard returned."),
        expected={"names": ["orion", "lyra", "capella", "wasat"]},
        check=None))                      # graded by the transcript, see run_tool_episode
    return out


def _paragraphs(text):
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    good = []
    for b in blocks:
        n = len(re.findall(r"\S+", b))
        if n >= 25:
            good.append(b)
    return good


def _check_gen_manual(text, expected, facts):
    paras = _paragraphs(text)
    numbered = len({m for m in re.findall(r"(?m)^\s*(?:#+\s*)?(?:\*\*)?RULE\s*(\d{1,2})", text or "")})
    ok = len(paras) >= expected["paragraphs"]
    return dict(ok=ok, score=min(len(paras), 12) / 12.0,
                why=f"paragraphs={len(paras)} numbered_rules={numbered}")


def _ledger_lines(text):
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip().upper().startswith("LEDGER ")]


def _check_copy_ledgers(text, expected, facts):
    rows = _ledger_lines(text)
    want = expected["rows"]
    hit = sum(1 for a, b in zip(rows, want) if a == b)
    ok = len(rows) == len(want) and hit == len(want)
    return dict(ok=ok, score=hit / len(want),
                why=f"lines={len(rows)} exact={hit}/{len(want)}")


def _check_enumerate_shards(text, expected, facts):
    want = expected["rows"]
    rows = [ln.strip().strip("`-* ") for ln in (text or "").splitlines() if "=" in ln]
    rows = [_norm(r) for r in rows]
    want_n = [_norm(r) for r in want]
    hit = sum(1 for a, b in zip(rows, want_n) if a == b)
    present = sum(1 for w in want_n if w in rows)
    ok = hit == len(want_n)
    return dict(ok=ok, score=present / len(want_n),
                why=f"lines={len(rows)} in_order={hit}/{len(want_n)} present={present}")


# ---------------------------------------------------------------- tool episode
TOOLS = [
    {"type": "function", "function": {
        "name": "read_shard",
        "description": "Read one shard record from the document by its shard name.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "shard name, e.g. orion"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "submit_report",
        "description": "Submit the final report and end the episode.",
        "parameters": {"type": "object", "properties": {
            "shards": {"type": "array", "items": {"type": "string"},
                       "description": "entries of the form <name>=<code>"}},
            "required": ["shards"]}}},
]


def normalize_calls(calls):
    out = []
    for c in calls or []:
        fn = c.get("function", c) if isinstance(c, dict) else {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:                                      # noqa: BLE001
                args = {}
        out.append({"name": fn.get("name", ""), "args": args or {}, "id": c.get("id")})
    return out


def tool_loop(transcript):
    """transcript: [{"tool": name, "args": dict, "ok": bool}, ...] in call order."""
    sigs = [(c["tool"], json.dumps(c.get("args") or {}, sort_keys=True)) for c in transcript]
    for a, b in zip(sigs, sigs[1:]):
        if a == b:
            return True, "identical_consecutive_call"
    dup = [s for s, n in Counter(sigs).items() if n >= 3]
    if dup:
        return True, f"identical_call_x3:{dup[0][0]}:{dup[0][1][:60]}"
    if len(transcript) >= 4 and not any(c["ok"] for c in transcript):
        return True, "no_progress_after_4_turns"
    return False, ""


def run_tool_episode(chat, doc, task, facts, budget, max_turns=6, path=None):
    """Drive the tool conversation. Returns the content/reasoning, the transcript, the last
    submitted report and whether it carried the right entries."""
    by_name = {s["name"]: s for s in facts["shards"]}
    messages = [{"role": "user", "content": doc + "\n\n" + task["prompt"]}]
    transcript, content, reasoning, cache_n = [], "", "", None
    submitted, errors, submitted_report = None, [], False
    for turn in range(max_turns):
        r = chat(messages, max_tokens=budget, tools=TOOLS, **(path or {}))
        errors.append(r.get("error"))
        content += r.get("content") or ""
        reasoning += r.get("reasoning_content") or ""
        if r.get("timings", {}).get("cache_n") is not None:
            cache_n = r["timings"]["cache_n"]
        calls = normalize_calls(r.get("tool_calls"))
        if not calls:
            break
        messages.append({"role": "assistant", "content": r.get("content") or "",
                         "tool_calls": [{"id": c["id"] or f"call_{turn}_{i}",
                                         "type": "function",
                                         "function": {"name": c["name"],
                                                      "arguments": json.dumps(c["args"])}}
                                        for i, c in enumerate(calls)]})
        for i, c in enumerate(calls):
            cid = c["id"] or f"call_{turn}_{i}"
            ok, payload = False, ""
            if c["name"] == "read_shard":
                name = str((c["args"] or {}).get("name", "")).strip().lower()
                s = by_name.get(name)
                if s:
                    ok = True
                    payload = f"SHARD {s['name']} code={s['code']} custodian={s['custodian']}"
                else:
                    payload = "NOT FOUND"
            elif c["name"] == "submit_report":
                entries = (c["args"] or {}).get("shards") or []
                if isinstance(entries, str):
                    entries = [e for e in re.split(r"[,\n]", entries) if e.strip()]
                submitted = [str(e).strip() for e in entries]
                want = {_norm(f"{s['name']}={s['code']}") for s in facts["shards"]
                        if s["name"] in set(task["expected"]["names"])}
                ok = want.issubset({_norm(e) for e in submitted})
                payload = "report accepted; episode over" if ok else \
                    "report rejected: missing or wrong shard entries"
            else:
                payload = f"unknown tool {c['name']}"
            transcript.append({"tool": c["name"], "args": c["args"], "ok": ok})
            messages.append({"role": "tool", "tool_call_id": cid, "content": payload})
            if c["name"] == "submit_report":
                # the episode ends with the report; continued calls are a real loop signal
                submitted_report = True
        if submitted_report:
            break
    return dict(content=content, reasoning=reasoning, transcript=transcript,
                messages=messages, submitted=submitted, cache_n=cache_n, errors=errors)


def grade_tool_episode(transcript, submitted):
    """Loop verdict from the transcript; work done = a report that carries the right entries."""
    flagged, why = tool_loop(transcript)
    return dict(flagged=flagged, reason=why, submitted=submitted, calls=len(transcript))
