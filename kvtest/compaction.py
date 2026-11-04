"""W3: compaction.

Phase A is forced and deterministic: the model reads the full document and writes a handover
brief of at most 1,500 words. Phase B discards the document and the phase-A transcript, sends the
brief verbatim, and asks questions whose answers lived deep in the discarded document.

Three channels stay separate: brief retention out of 20 (the cache-damage channel and the primary
endpoint), phase-B keys correct out of 5 (short-context reasoning), and degeneration over both
responses (long-horizon behaviour). The 20 are `facts["w3_keys"]`, a spread across the document;
W2 asks about all 64 shards.
"""
import re

from . import config, ledger, loops, tasks, toolrun

PHASE_B_QUESTIONS = 5


def mutation_rows(facts, tenant):
    ids = []
    for op in ledger.target_ops(facts, tenant):
        for key in ("row", "to"):
            if op.get(key) and op[key] not in ids:
                ids.append(op[key])
    return ids


def phase_a_prompt(facts, tenant):
    names = ", ".join(f"{s['name']}={s['code']}" for s in facts["w3_keys"])
    rows = ", ".join(mutation_rows(facts, tenant))
    invs = "; ".join(facts["invariants"])
    return (
        f"Read the document and write a handover brief of at most {config.BRIEF_WORD_CAP} words "
        "for a worker who will never see the document. The brief must carry, exactly and in "
        "full:\n"
        f"1. these twenty shard identifiers: {names}\n"
        f"2. every ledger row touched by tenant {tenant}'s mutations: {rows}\n"
        f"3. these five invariants: {invs}\n\n"
        "Write the brief only. No preamble, no commentary, no apology.")


def _assoc_status(text, key):
    """present if name and live code appear together; mutated if the wrong code does."""
    name, code, decoy = key["name"], key["code"], key.get("decoy")
    lower = (text or "").lower()
    for m in re.finditer(re.escape(name.lower()), lower):
        if code.lower() in lower[m.start():m.end() + 240]:
            return "present"
    for m in re.finditer(re.escape(name.lower()), lower):
        if decoy and decoy.lower() in lower[m.start():m.end() + 240]:
            return "mutated"
    return "absent"


def grade_brief(brief, facts):
    keys = [dict(name=s["name"], code=s["code"], decoy=s["decoy"], depth=s["frac"])
            for s in facts["w3_keys"]]
    for k in keys:
        k["status"] = _assoc_status(brief, k)
    words = len(re.findall(r"\S+", brief or ""))
    rows = facts.get("_brief_rows") or []
    invs = facts["invariants"]
    return dict(
        retention=sum(1 for k in keys if k["status"] == "present"),
        of=len(keys),
        mutated=sum(1 for k in keys if k["status"] == "mutated"),
        absent=sum(1 for k in keys if k["status"] == "absent"),
        keys=keys, words=words, within_cap=words <= config.BRIEF_WORD_CAP,
        ledger_rows_present=sum(1 for r in rows if r.lower() in (brief or "").lower()),
        of_ledger_rows=len(rows),
        invariants_present=sum(1 for inv in invs
                               if inv.split(":", 1)[-1].strip().lower()[:60]
                               in (brief or "").lower()),
        of_invariants=len(invs))


def phase_b_questions(facts):
    """Five questions answerable from the brief alone; answers lived deep in the document."""
    deep = sorted(facts["w3_keys"], key=lambda s: -s["frac"])
    row_facts = {lg["id"]: lg for lg in facts["ledgers"]}
    deep_rows = [row_facts[r] for r in mutation_rows(facts, facts["tenants"][0]) if r in row_facts]
    deep_rows.sort(key=lambda lg: -lg["frac"])
    qs = []
    for shard in deep[:3]:
        qs.append(dict(id=f"b_code_{len(qs) + 1}", expected=shard["code"],
                       prompt=(f"What is the live code of shard {shard['name']}? Answer with the "
                               "code alone.")))
    if deep_rows:
        qs.append(dict(id="b_amount", expected=str(deep_rows[0]["amount"]),
                       prompt=f"What is the amount of ledger row {deep_rows[0]['id']}? "
                              "Answer with the integer alone."))
        qs.append(dict(id="b_tag", expected=deep_rows[-1]["tag"],
                       prompt=f"What is the tag of ledger row {deep_rows[-1]['id']}? "
                              "Answer with the tag alone."))
    while len(qs) < PHASE_B_QUESTIONS:
        shard = deep[len(qs) % len(deep)]
        qs.append(dict(id=f"b_code_{len(qs) + 1}", expected=shard["code"],
                       prompt=f"What is the live code of shard {shard['name']}? "
                              "Answer with the code alone."))
    for q in qs:
        q["prompt"] += "\n\n" + config.ANSWER_INSTRUCTION
    return qs[:PHASE_B_QUESTIONS]


def _norm(v):
    return re.sub(r"\s+", "", (v or "")).upper()


def grade_phase_b(facts, answers):
    qs = facts["_phase_b"]
    hits = [_norm(a) == _norm(q["expected"]) for q, a in zip(qs, answers)]
    return dict(keys_correct=sum(hits), of=len(qs), hits=hits,
                expected=[q["expected"] for q in qs], answers=answers)


def loop_channels(content, reasoning, out_tokens):
    out = {}
    for part, text in (("content", content), ("reasoning_content", reasoning)):
        m = loops.metrics(text or "")
        flagged, why = loops.verdict(m, graded=False, got_answer=True,
                                     gen_tokens=out_tokens or 0)
        out[part] = dict(metrics=m, flagged=flagged, reason=why,
                         token_pos=m.get("flag_token_est") if flagged else None)
    return out


def instance_facts(facts, tenant):
    out = dict(facts)
    out["_brief_rows"] = mutation_rows(facts, tenant)
    out["_phase_b"] = phase_b_questions(facts)
    return out


def _all_answers(content, reasoning, n):
    """The last n ANSWER lines in order, from content then reasoning, padded at the end."""
    found = tasks.ANSWER_RE.findall(content or "")
    if len(found) < n:
        found += tasks.ANSWER_RE.findall(reasoning or "")
    vals = [v.strip() for v in found[-n:]]
    return vals + [None] * (n - len(vals))


def run(chat, facts, doc, tenant, instance, ctx=None):
    f = instance_facts(facts, tenant)
    rec_a = chat([{"role": "user", "content": doc + "\n\n" + phase_a_prompt(f, tenant)}],
                 max_tokens=config.MAX_OUTPUT_TOKENS)
    brief = rec_a.get("content") or ""
    used_a = ((rec_a.get("usage") or {}).get("prompt_tokens") or 0) + \
             ((rec_a.get("usage") or {}).get("completion_tokens") or 0)
    loop_a = loop_channels(brief, rec_a.get("reasoning_content") or "",
                           (rec_a.get("usage") or {}).get("completion_tokens"))
    halted_a = "error" if rec_a.get("error") else (
        "loop" if (rec_a.get("loop_halt") or any(v["flagged"] for v in loop_a.values())) else
        "context_full" if used_a and ctx and used_a >= ctx else
        "ceiling" if ((rec_a.get("usage") or {}).get("completion_tokens") or 0)
        >= config.MAX_OUTPUT_TOKENS else "completed")
    halt_pos_a = (rec_a.get("loop_halt") or {}).get("token_pos") or used_a
    flagged_a = [v for v in loop_a.values() if v["flagged"]]
    halt_reason_a = ((f"{rec_a['loop_halt'].get('stream')}: {rec_a['loop_halt'].get('reason')}"
                      if rec_a.get("loop_halt") else None)
                     or (flagged_a[0]["reason"] if flagged_a else None))

    msgs_b = [{"role": "user",
               "content": ("Handover brief, verbatim:\n\n" + brief + "\n\n" +
                           "\n\n".join(q["prompt"] for q in f["_phase_b"]))}]
    rec_b = chat(msgs_b, max_tokens=config.MAX_OUTPUT_TOKENS)
    answers = _all_answers(rec_b.get("content") or "", rec_b.get("reasoning_content") or "",
                           len(f["_phase_b"]))
    loop_b = loop_channels(rec_b.get("content") or "", rec_b.get("reasoning_content") or "",
                           (rec_b.get("usage") or {}).get("completion_tokens"))
    brief_grade = grade_brief(brief, f)
    return dict(
        workflow="W3", instance=instance, tenant=tenant,
        phase_a=dict(content=brief, reasoning=rec_a.get("reasoning_content") or "",
                     halted_by=halted_a, halt_token_pos=halt_pos_a,
                     halt_reason=halt_reason_a,
                     loop_halt=rec_a.get("loop_halt"), error=rec_a.get("error"),
                     prompt_tokens=(rec_a.get("usage") or {}).get("prompt_tokens"),
                     out_tokens=(rec_a.get("usage") or {}).get("completion_tokens"),
                     thinking_tokens=rec_a.get("thinking_tokens"),
                     thinking_share=rec_a.get("thinking_share"),
                     first_visible_s=rec_a.get("first_visible_s"),
                     wall_s=rec_a.get("wall_s"), loops=loop_a,
                     flagged=any(v["flagged"] for v in loop_a.values())),
        phase_b=dict(content=rec_b.get("content") or "",
                     reasoning=rec_b.get("reasoning_content") or "",
                     halted_by=("loop" if (rec_b.get("loop_halt") or any(
                         v["flagged"] for v in loop_b.values())) else "completed"),
                     halt_reason=(f"{rec_b['loop_halt'].get('stream')}: "
                                  f"{rec_b['loop_halt'].get('reason')}"
                                  if rec_b.get("loop_halt") else
                                  next((v["reason"] for v in loop_b.values() if v["flagged"]),
                                       None)),
                     error=rec_b.get("error"),
                     prompt_tokens=(rec_b.get("usage") or {}).get("prompt_tokens"),
                     out_tokens=(rec_b.get("usage") or {}).get("completion_tokens"),
                     thinking_tokens=rec_b.get("thinking_tokens"),
                     thinking_share=rec_b.get("thinking_share"),
                     first_visible_s=rec_b.get("first_visible_s"),
                     wall_s=rec_b.get("wall_s"), loops=loop_b,
                     flagged=any(v["flagged"] for v in loop_b.values())),
        brief_grade=brief_grade,
        phase_b_grade=grade_phase_b(f, answers),
        out_tokens=(((rec_a.get("usage") or {}).get("completion_tokens") or 0) +
                    ((rec_b.get("usage") or {}).get("completion_tokens") or 0)),
        halted_by=halted_a,
        primary=brief_grade["retention"])


TOOLS = [toolrun.compact_tool_spec()]


def run_model_driven(chat, facts, doc, tenant, instance, ctx, guard=4):
    """Secondary variant: the model decides when to compact, with a 90% context wall."""
    f = instance_facts(facts, tenant)
    messages = [{"role": "user",
                 "content": doc + "\n\n" + phase_a_prompt(f, tenant) +
                 "\n\nWhen the brief is ready, call `compact` with it, then answer from the "
                 "brief alone."}]

    class _Box:
        def __init__(self):
            self.events = []

        def __call__(self, name, args, msgs):
            if name != "compact":
                return dict(ok=False, payload=f"unknown tool {name}", diff=None)
            brief = str(args.get("brief") or "")
            ev = toolrun.compact_messages(msgs, brief, len(self.events))
            self.events.append(ev)
            return dict(ok=True, payload="context replaced by the brief", compaction=ev,
                        diff={"brief_words": ev["brief_words"]})

        def state_hash(self):
            return f"compact:{len(self.events)}"

    box = _Box()
    res = toolrun.run_tool_loop(chat, messages, TOOLS, box, guard=guard, ctx=ctx,
                                ctx_wall=config.COMPACT_WALL_FRACTION)
    brief = box.events[-1]["brief"] if box.events else ""
    steps = {t["step"]: t for t in res["transcript"]}
    for ev in box.events:
        ev["token_pos"] = (steps.get(ev["step"]) or {}).get("token_pos")
    grade_ = grade_brief(brief, f)
    return dict(workflow="W3", variant="model_driven", instance=instance, tenant=tenant,
                phase_a=dict(content=res["content"], reasoning=res["reasoning"],
                             halted_by=res["halted_by"], loops={},
                             prompt_tokens=res["prompt_tokens"], out_tokens=res["out_tokens"],
                             wall_s=None, flagged=bool(res["loop"])),
                phase_b=None, compactions=box.events, brief_grade=grade_,
                out_tokens=res["out_tokens"],
                primary=grade_["retention"], halted_by=res["halted_by"],
                loop=res["loop"], transcript=res["transcript"])
