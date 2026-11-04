"""Self-test: the harness, the graders and the report, against a mock llama-server.

No GPU, no model files, no network. The mock serves both non-streaming and streaming chat
completions so the request path (greedy payload, usage, timings, tool-free replies) is exercised
exactly as a run would exercise it.
"""
import http.server
import json
import os
import re
import socketserver
import tempfile
import threading
import time

from . import (client, compaction, config, ledger, longctx, loops, report, run, search,
               stats, tasks, toolrun)

CHECKS = []
CANNED = {"answer": "ACK", "reasoning": ""}
REQUESTS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


# ---------------------------------------------------------------- mock server
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path == "/props":
            self._send({"default_generation_settings": {}})
        else:
            self._send({"ok": True})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        REQUESTS.append({"path": self.path, "payload": req})
        text, reasoning = _canned(req)
        if req.get("stream"):
            chunks = [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            ]
            for part in [text[i:i + 8] for i in range(0, len(text), 8)]:
                chunks.append({"choices": [{"index": 0, "delta": {"content": part}}]})
            chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            chunks.append({"choices": [], "usage": {"prompt_tokens": 120,
                                                    "completion_tokens": len(text) // 4 + 1,
                                                    "total_tokens": 140},
                           "timings": {"prompt_n": 120, "prompt_ms": 120.0,
                                       "prompt_per_second": 1000.0, "predicted_n": 10,
                                       "predicted_ms": 50.0, "predicted_per_second": 200.0,
                                       "cache_n": 100}})
            payload = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
            payload += b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._send({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text,
                                                 "reasoning_content": reasoning},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140},
            "timings": {"prompt_n": 120, "prompt_ms": 120.0, "prompt_per_second": 1000.0,
                        "predicted_n": 20, "predicted_ms": 100.0, "predicted_per_second": 200.0,
                        "cache_n": 100},
        })


def _canned(req):
    last = ""
    for m in req.get("messages") or []:
        if m.get("role") == "user":
            last = m.get("content") or ""
    if "operator's manual" in last:
        return CANNED.get("w6") or "ACK", ""
    if re.search(r"\d+ shards", last):
        return CANNED.get("w2") or "ACK", ""
    if "handover brief" in last:
        return CANNED.get("brief") or "ACK", ""
    return CANNED["answer"], CANNED["reasoning"]


class MockServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def start_mock():
    srv = MockServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ---------------------------------------------------------------- fake arm server
class FakeVram:
    peak = 0


class FakeServer:
    def __init__(self, model_key, kv, ctx, knobs=None, port=None, log_path=None):
        self.base = BASE
        self.ctx = ctx
        self.start_s = 0.1
        self.vram = FakeVram()

    def start(self):
        return True

    def stop(self):
        pass

    def log_tail(self, n=1000):
        return ""

    def sampler_resolved(self):
        return {"temperature": 0.0, "top_k": 20.0, "top_p": 0.95, "greedy": True,
                "raw": "\ttop_k = 20, top_p = 0.950, ..., temp = 0.000"}


# ---------------------------------------------------------------- tests
def test_client():
    r = client.chat(BASE, [{"role": "user", "content": "hello"}], max_tokens=8,
                    **config.SAMPLING)
    check("client: non-streaming reply", r.get("error") is None and r["content"] == "ACK")
    payload = REQUESTS[-1]["payload"]
    check("client: greedy payload has temperature 0.0", payload.get("temperature") == 0.0)
    check("client: no sampler keys beyond temperature",
          not ({"top_k", "top_p", "min_p", "repeat_penalty", "seed"} & set(payload)),
          str({k: payload[k] for k in ("top_k", "seed") if k in payload}))
    s = client.chat_stream(BASE, [{"role": "user", "content": "hello"}], max_tokens=8,
                           **config.SAMPLING)
    check("client: streaming content assembled", s["content"] == "ACK", s["content"])
    check("client: streaming measures first visible token",
          s["first_visible_s"] is not None and s["thinking_tokens"] == 0)
    check("client: streaming usage and timings", (s["usage"]["prompt_tokens"] == 120
                                                  and s["timings"]["cache_n"] == 100))


def test_document():
    doc, facts = longctx.build_document(filler_chars=6000)
    codes = [s["code"] for s in facts["shards"]]
    check("document: 64 shards with unique live codes", len(codes) == 64 and len(set(codes)) == 64)
    check("document: every shard is a W2 key, in document order",
          len(facts["w2_keys"]) == 64
          and [s["name"] for s in facts["w2_keys"]] == [s["name"] for s in facts["shards"]])
    check("document: 20 W3 brief identifiers, spread across the document",
          len(facts["w3_keys"]) == 20
          and facts["w3_keys"][0] is facts["shards"][0]
          and facts["w3_keys"][-1] is facts["shards"][-1])
    decoys = [s["decoy"] for s in facts["shards"] if s["decoy"]]
    check("document: code decoys differ from every live code",
          decoys and not (set(decoys) & set(codes)))
    check("document: every code decoy is a near miss (same prefix)",
          all(d[:2] == s["code"][:2] for s in facts["shards"] if s["decoy"]
              for d in [s["decoy"]]))
    check("document: 60 search clusters in 15 four-hop chains",
          len(facts["clusters"]) == 60
          and all(facts["clusters"][i * 4 + 3]["next"] == "NONE" for i in range(15)))
    check("document: 400 ops, 20 per tenant for six tenants",
          len(facts["ops"]) == 400
          and all(sum(1 for o in facts["ops"]
                      if o["tenant"] == t and not o.get("foreign")) == 20
                  for t in facts["tenants"]))
    check("document: unique shard codes occur once each",
          all(longctx.exact_occurrences(doc, s["code"]) == 1 for s in facts["shards"]))
    check("document: five invariants", len(facts["invariants"]) == 5)


def test_w2_grader():
    _, facts = longctx.build_document(filler_chars=4000)
    keys = facts["w2_keys"]
    codes = [s["code"] for s in keys]
    good = ", ".join(codes)
    grade = tasks.grade_w2(facts, "listing...\nANSWER: " + good)
    n = len(keys)
    check(f"W2: {n} correct codes score {n}", grade["score"] == n and grade["source"] == "content")
    near = list(codes)
    near[0] = keys[0]["decoy"] or "ZZ-0000"
    grade = tasks.grade_w2(facts, "ANSWER: " + ", ".join(near))
    check("W2: near-miss decoy is `mutated`", grade["keys"][0]["status"] == "mutated")
    check("W2: one mutated key costs one key", grade["score"] == n - 1)
    swap = list(codes)
    swap[1], swap[2] = swap[2], swap[1]
    grade = tasks.grade_w2(facts, "ANSWER: " + ", ".join(swap))
    check("W2: displaced codes are `misordered`",
          grade["keys"][1]["status"] == "misordered"
          and grade["keys"][2]["status"] == "misordered")
    short = ", ".join(codes[:-1])
    grade = tasks.grade_w2(facts, "ANSWER: " + short)
    check("W2: missing position is `absent` and costs one key",
          grade["keys"][-1]["status"] == "absent" and grade["score"] == n - 1)
    grade = tasks.grade_w2(facts, "", "ANSWER: " + good)
    check("W2: ANSWER is parsed from reasoning when content has none",
          grade["score"] == n and grade["source"] == "reasoning")
    grade = tasks.grade_w2(facts, "no answer here")
    check("W2: no ANSWER line scores 0 and marks every key absent",
          grade["score"] == 0 and all(k["status"] == "absent" for k in grade["keys"]))
    newline = "\n".join(codes)
    check("W2: newline-separated values accepted",
          tasks.grade_w2(facts, "ANSWER: " + newline)["score"] == n)


def test_w6_grader():
    titles = [t for t, _ in tasks.SECTIONS]
    text = []
    for i, (title, target) in enumerate(tasks.SECTIONS):
        text.append(f"## {i + 1}. {title}\n" + "some words " * 40 +
                    f"\nSee Section {target}.")
    grade = tasks.grade_w6("\n\n".join(text))
    check("W6: six sections present", grade["sections_present"] == 6)
    check("W6: six cross-references resolve", grade["refs_resolved"] == 6 and grade["ok"])
    bad = "\n\n".join(text[:4])
    check("W6: missing sections are counted, not inferred", tasks.grade_w6(bad)["sections_present"] == 4)


def test_loop_scan():
    sigs = ["a", "a"]
    check("loop: identical consecutive call", toolrun.loop_scan(sigs, ["1", "2"])[1]
          == "identical_consecutive_call")
    sigs = ["a", "b", "a", "b", "a", "b"]
    check("loop: A<->B oscillation", toolrun.loop_scan(sigs, list("123456"))[1] == "oscillation")
    sigs = ["x", "y", "z", "w", "v", "u", "t"]
    flagged, why, _, _ = toolrun.loop_scan(sigs, ["1"] * 7)
    check("loop: no-progress window", flagged and why == "no_progress_6")
    sigs = ["a", "b", "c"]
    check("loop: clean transcript is not flagged", not toolrun.loop_scan(sigs, "123")[0])
    m = loops.metrics("same line\n" * 8)
    check("loop: degenerate text fires the detector", loops.verdict(m)[0], m["span"][:20]
          if m["span"] else "")
    structured = ("Task: list the facts. Constraint 1: no preamble. Constraint 2: be exact.\n" +
                  "\n".join(f"  * item {i}: shard {i} has code AK-{1000 + i} and custodian EMP-{i:04d}"
                            for i in range(12)))
    check("loop: structured reasoning is not flagged by the zlib ratio",
          not loops.verdict(loops.metrics(structured))[0])
    semantic = " ".join(
        f"Wait, let me re-check shard {name} again. Its code is {code}, and the retired one is not "
        f"the live code." for name, code in [("orion", "AK-3652"), ("lyra", "CZ-1593"),
                                            ("cygnus", "EG-8452"), ("rigel", "JP-4425")] * 12)
    check("loop: semantic re-check loop fires on low 4-gram diversity",
          loops.verdict(loops.metrics(semantic))[0])
    enum = "\n".join(f"L-{i:04d} (TEN-01)" for i in range(60))
    m = loops.metrics(enum)
    check("loop: monotone enumeration fires as a template run",
          loops.verdict(m)[0] and "template_run" in loops.verdict(m)[1])
    codes = "\n".join(f"AK-{1000 + i}" for i in range(64))
    check("loop: a legitimate sixty-four-item answer list is not flagged",
          not loops.verdict(loops.metrics(codes))[0])
    sentences = ["Custody requires a signature before any move.",
                 "Settlement follows the order of record.",
                 "Containers are shared only when a shard is closed.",
                 "Void rows stay in the ledger for audit.",
                 "A leaving custodian hands every badge upward.",
                 "Even-numbered projects need a second signature.",
                 "Deep shards are re-sealed when read.",
                 "Badge numbers are never reused."]
    clean = loops.metrics(" ".join(sentences * 3))
    check("loop: ordinary prose does not fire", not loops.verdict(clean)[0])


def test_compaction_grader():
    _, facts = longctx.build_document(filler_chars=4000)
    brief_ok = "\n".join(f"SHARD {s['name']} code={s['code']}" for s in facts["w3_keys"])
    g = compaction.grade_brief(brief_ok, facts)
    check("W3: retention counts 20 identifiers", g["retention"] == 20 and g["absent"] == 0)
    decoy = facts["w3_keys"][0]["decoy"]
    mutated_brief = f"SHARD {facts['w3_keys'][0]['name']} code={decoy}"
    g = compaction.grade_brief(mutated_brief, facts)
    check("W3: wrong code for a name is `mutated`",
          g["keys"][0]["status"] == "mutated" and g["mutated"] == 1)
    f = compaction.instance_facts(facts, facts["tenants"][0])
    qs = f["_phase_b"]
    answers = [q["expected"] for q in qs]
    check("W3: phase-B questions grade from exact answers",
          compaction.grade_phase_b(f, answers)["keys_correct"] == 5)
    empty = compaction.grade_phase_b(f, [None] * 5)
    check("W3: missing phase-B answers score 0", empty["keys_correct"] == 0)
    # forced two-phase run against a scripted chat that records what each phase saw
    replies = [{"content": brief_ok, "reasoning_content": "",
                "usage": {"prompt_tokens": 90000, "completion_tokens": 600}},
               {"content": "\n".join(f"ANSWER: {q['expected']}" for q in qs),
                "reasoning_content": "", "usage": {"prompt_tokens": 900, "completion_tokens": 40}}]
    seen = []

    def chat(messages, max_tokens=None):
        seen.append(messages[0]["content"])
        return replies.pop(0)

    rec = compaction.run(chat, facts, "THE-DOCUMENT", facts["tenants"][0], 0, ctx=102400)
    check("W3: forced pair records retention and phase-B keys",
          rec["primary"] == 20 and rec["phase_b_grade"]["keys_correct"] == 5)
    check("W3: phase B sees the brief and not the document",
          "THE-DOCUMENT" not in seen[1] and facts["w3_keys"][0]["code"] in seen[1])


def test_ledger_workflow():
    _, facts = longctx.build_document(filler_chars=4000)
    tenant = facts["tenants"][0]
    with tempfile.TemporaryDirectory() as tmp:
        env, path = ledger.build_instance(facts, "DOC", tenant, tmp, 0)
        check("W4: the op log carries the mutation block once",
              open(path).read().count("BEGIN MUTATION LOG") == 1)
        ops = ledger.target_ops(facts, tenant)
        check("W4: exactly 20 target mutations", len(ops) == 20)
        ok = env("read_ops", {"start": 0, "count": 3}, [])
        check("W4: read_ops returns a window with its line range",
              ok["ok"] and ok["payload"].startswith("lines 1-"))
        for op in ops:
            args = {"op": op["id"], "row": op["row"]}
            name = {"append": "apply_append", "retract": "apply_retract",
                    "transfer": "apply_transfer"}[op["kind"]]
            if op["kind"] == "append":
                args.update(tag=op["tag"], amount=op["amount"])
            if op["kind"] == "transfer":
                args.update(to=op["to"], amount=op["amount"])
            res = env(name, args, [])
            if not res["ok"]:
                check(f"W4: {op['id']} {op['kind']} applies", False, res["payload"])
                break
        else:
            check("W4: all 20 mutation records apply cleanly", True)
        grade = ledger.grade(env)
        check("W4: all 20 applied exactly once in log order",
              grade["state_ok"] and grade["order_ok"] and not grade["double_applied"],
              f"{grade['final_hash'][:8]} vs {grade['expected_hash'][:8]}")
        check("W4: all five invariants hold", all(grade["invariants"].values()))
        before = env.state_hash()
        extra = next(op for op in facts["ops"] if op["tenant"] != tenant and not op.get("foreign"))
        env("apply_retract", {"op": extra["id"], "row": extra["row"]}, [])
        check("W4: applying a foreign tenant's mutation fails",
              env.state_hash() == before and extra["id"] in env.foreign)
        first = next(op for op in ops if op["kind"] != "retract")
        name = {"append": "apply_append", "retract": "apply_retract",
                "transfer": "apply_transfer"}[first["kind"]]
        args = {"op": first["id"], "row": first["row"]}
        if first["kind"] == "append":
            args.update(tag=first["tag"], amount=first["amount"])
        if first["kind"] == "transfer":
            args.update(to=first["to"], amount=first["amount"])
        env(name, args, [])
        check("W4: re-applying a mutation changes the hash", first["id"] in env.double
              and not ledger.grade(env)["state_ok"])


def test_search_workflow():
    _, facts = longctx.build_document(filler_chars=4000)
    q = facts["multihop"][0]
    hits = search.search(facts, q["head"])
    check("W5: search finds the head cluster first", hits and hits[0]["id"] == q["head"])
    env = search.SearchEnv(facts)
    for question in facts["multihop"]:
        term = question["head"]
        for _ in range(2):
            res = env("search", {"term": term}, [])
            hit = {c["id"]: c for c in facts["clusters"]}[res["diff"]["hits"][0]]
            term = hit["next"]
        clusters = {c["id"]: c for c in facts["clusters"]}
        got = clusters[term]["value"]
        check(f"W5: {question['id']} is reachable in three searches", got == question["expected"])
    grade = search.grade([q["expected"] for q in facts["multihop"]],
                         [q["expected"] for q in facts["multihop"]], 9)
    check("W5: three correct keys and the floor met", grade["keys_correct"] == 3
          and grade["floor_met"])


def test_kld_parser():
    text = """
Mean PPL(Q)                   :   12.345678 ±   0.012345
Mean PPL(base)                :   11.111111 ±   0.010000
Mean PPL(Q)/PPL(base)         :    1.111111 ±    0.001111
Mean ln(PPL(Q)/PPL(base))     :    0.105361 ±    0.001000
Mean    KLD:   0.000660 ±  0.000012
Mean    Δp:   1.234 ±   0.010 %
RMS Δp    :   2.500 ±   0.020 %
Same top p:  99.100 ±   0.100 %
Final estimate: PPL = 12.3456 +/- 0.0123
"""
    from . import kld
    parsed = kld.parse_kld(text)
    check("W1: KLD metrics parsed", parsed["mean_kld"] == 0.00066
          and parsed["same_top"] == 0.991)
    check("W1: PPL ratio parsed", abs(parsed["ppl_ratio"] - 1.111111) < 1e-9)
    ppl, unc = kld.parse_ppl(text)
    check("W1: PPL final estimate parsed", ppl == 12.3456 and unc == 0.0123)


def test_stats():
    p, lo, hi = stats.wilson(8, 10)
    check("stats: Wilson interval brackets the proportion", lo < p < hi and abs(p - 0.8) < 1e-9)
    diffs = [-1.0] * 12
    paired = stats.paired_bootstrap(diffs, resamples=200, seed=1)
    check("stats: all-negative differences read worse", paired["worse"]
          and paired["hi"] < 0)
    paired = stats.paired_bootstrap([0.0] * 12, resamples=200, seed=1)
    check("stats: zero differences are a tie", not paired["worse"] and not paired["better"])
    check("stats: smallest p at n=12", abs(stats.smallest_p(12) - 2 / 4096) < 1e-12)


def test_runner_end_to_end(tmp, mock_base):
    global BASE
    old = run.srv.Server
    run.srv.Server = FakeServer
    try:
        doc, facts = longctx.build_document(filler_chars=3000)
        CANNED["w2"] = "ANSWER: " + ", ".join(s["code"] for s in facts["w2_keys"])
        CANNED["w6"] = "\n\n".join(
            f"## {i + 1}. {t}\n" + " ".join(
                f"The {t.lower()} clause {j} records that shard {facts['shards'][i * 6 + j]['name']} "
                f"keeps custodian {facts['shards'][i * 6 + j]['custodian']} and settles under the "
                f"invariant {facts['invariants'][j][:60]}. It also names ledger row "
                f"{facts['ledgers'][i * 7 + j]['id']} as the record of reference."
                for j in range(4)) +
            f"\nSee Section {target}." for i, (t, target) in enumerate(tasks.SECTIONS))
        out = os.path.join(tmp, "run")
        runner = run.Runner(stamp="selftest", doc=doc, facts=facts, fit={},
                            models=["gemma4-26b-a4b"], kvs=["q4_0"],
                            workflows=["W2", "W6"], instances={"W2": 2, "W6": 1},
                            ctx_override=8192, deadline_hours=0, resume=False, out_dir=out)
        raw = runner.run()
        check("runner: two scored probes and one long-form instance",
              raw["samples"] == 3, str(raw["samples"]))
        arm = raw["arms"]["gemma4-26b-a4b__q4_0"]
        check("runner: arm finished with a sampler record",
              arm["finished"] and arm["sampler_resolved"]["greedy"])
        w2 = arm["workflows"]["W2"]["instances"]
        n_keys = len(facts["w2_keys"])
        check("runner: W2 scored every key twice",
              [i["score"] for i in w2] == [n_keys, n_keys])
        check("runner: W6 sections and refs graded",
              arm["workflows"]["W6"]["instances"][0]["grade"]["ok"])
        # resume must not repeat finished instances
        again = run.Runner(stamp="selftest", doc=doc, facts=facts, fit={},
                           models=["gemma4-26b-a4b"], kvs=["q4_0"],
                           workflows=["W2", "W6"], instances={"W2": 2, "W6": 1},
                           ctx_override=8192, deadline_hours=0, resume=True, out_dir=out)
        raw2 = again.run()
        check("runner: resume is idempotent", raw2["samples"] == 3, str(raw2["samples"]))
        md = report.build(out)
        check("report: W2 and W6 sections written",
              f"W2: {len(facts['w2_keys'])}-key retrieval" in md and "W6" in md)
        check("report: HTML is a single self-contained page",
              os.path.exists(os.path.join(out, "report.html"))
              and "<svg" in open(os.path.join(out, "report.html")).read())
        check("report: figures written",
              os.path.exists(os.path.join(out, "figures", "w2-depth.svg")))
        check("report: CSV has one row per arm and a header",
              len(open(os.path.join(out, "matrix.csv")).read().strip().splitlines()) == 2)
        check("runner: greedy temperature is the only sampler field sent",
              all("temperature" in r["payload"] and r["payload"]["temperature"] == 0.0
                  for r in REQUESTS if r["path"].endswith("completions")))
    finally:
        run.srv.Server = old


def test_report_all_workflows(tmp):
    """Report paths for W1/W3/W4/W5, including a paired W2 comparison, from a synthetic raw."""
    from . import stats, tasks as t

    def arm(model, kv, w2_scores):
        keys = [dict(name=f"shard{i}", depth=i / 19, expected=f"AK-{1000 + i}",
                     got=f"AK-{1000 + i}", status="hit") for i in range(20)]
        return dict(model=model, kv=kv, ctx=102400, finished="2026-01-01T00:00:00Z",
                    wall_s=100.0, sampler_resolved={"temperature": 0.0, "greedy": True},
                    done=[], cut=None,
                    workflows={
                        "W1": dict(kld=dict(mean_kld=0.001 if kv != "f16" else 0.0,
                                            mean_kld_unc=0.0001, same_top=0.99,
                                            log_ppl_ratio=0.01, ppl=800.0),
                                   deep_ppl=180.0 if kv != "f16" else 177.0, deep_ctx=65536),
                        "W2": dict(instances=[dict(
                            instance=i, score=s, halted_by="completed",
                            grade=dict(answer="x", score=s, keys=keys)) for i, s in
                            enumerate(w2_scores)]),
                        "W3": dict(instances=[dict(
                            instance=0, score=18, halted_by="completed",
                            phase_a=dict(halted_by="completed", content="brief"),
                            brief_grade=dict(mutated=1, absent=1, within_cap=True),
                            phase_b_grade=dict(keys_correct=4))]),
                        "W4": dict(instances=[dict(instance=0, halted_by="completed", grade=dict(
                            state_ok=True, invariants_ok=5, reported_invariants_match=5,
                            double_applied=[], foreign_applied=[], reads=6, applied=20))]),
                        "W5": dict(instances=[dict(instance=0, halted_by="completed", grade=dict(
                            keys_correct=3, calls=12, floor_met=True))]),
                        "W6": dict(instances=[dict(instance=0, halted_by="loop", grade=dict(
                            sections_present=5, refs_resolved=4, ok=False))]),
                    })

    out = os.path.join(tmp, "report-all")
    os.makedirs(out, exist_ok=True)
    raw = {
        "stamp": "fixture", "manifest": {"instances": {"W2": 2, "W3": 1, "W4": 1, "W5": 1,
                                                       "W6": 1},
                                          "sampling": config.SAMPLING, "rungs": {},
                                          "llama_cpp": {}, "models": {}},
        "document": {"chars": 100, "sha256": "deadbeef"},
        "cells": [dict(model="m", kv=kv, workflows=["W1", "W2", "W3", "W4", "W5", "W6"])
                  for kv in ("f16", "q4_0")],
        "cut_cells": [], "unrun_cells": [],
        "arms": {"m__f16": arm("m", "f16", [20, 19]),
                 "m__q4_0": arm("m", "q4_0", [18, 17])},
    }
    with open(os.path.join(out, "raw.json"), "w") as f:
        json.dump(raw, f)
    md = report.build(out)
    for heading in ("W3: compaction", "W4: long-horizon ledger", "W5: agentic search",
                    "W6: unbounded long-form"):
        check(f"report: {heading} section written", heading in md)
    check("report: paired W2 difference reads worse", "worse than f16" in md)
    check("report: W4 state-hash and invariant counts rendered", "1/1" in md)
    check("report: figure files exist",
          os.path.exists(os.path.join(out, "figures", "w1-kld.svg"))
          and os.path.exists(os.path.join(out, "figures", "w2-depth.svg")))
    check("report: CSV has two arms",
          len(open(os.path.join(out, "matrix.csv")).read().strip().splitlines()) == 3)

    raw_fail = json.loads(json.dumps(raw))
    for name in raw_fail["arms"]:
        raw_fail["arms"][name]["workflows"]["W1"] = {"error": "llama-perplexity missing",
                                                    "kld": {}}
    md_fail = report.write_markdown(out, raw_fail, report.aggregate(raw_fail),
                                    os.path.join(out, "REPORT-fail.md"))
    check("report: a failed W1 pass shows its recorded error, not 'never ran'",
          "recorded no KLD metrics" in md_fail
          and "llama-perplexity missing" in md_fail)

    raw_mixed = json.loads(json.dumps(raw))
    raw_mixed["arms"]["m__q4_0"]["workflows"]["W1"] = {"error": "llama-perplexity missing",
                                                       "kld": {}}
    md_mixed = report.write_markdown(out, raw_mixed, report.aggregate(raw_mixed),
                                     os.path.join(out, "REPORT-mixed.md"))
    check("report: a mixed W1 stamp lists only the failed arm",
          "m__q4_0` has no KLD row" in md_mixed
          and "m__f16` has no KLD row" not in md_mixed
          and "no f16 reference" not in md_mixed)


# ---------------------------------------------------------------- main
def main():
    global BASE
    mock, BASE = start_mock()
    print(f"mock llama-server on {BASE}")
    tmp = tempfile.mkdtemp(prefix="kvtest-selftest-")
    t0 = time.time()
    try:
        test_client()
        test_document()
        test_w2_grader()
        test_w6_grader()
        test_loop_scan()
        test_compaction_grader()
        test_ledger_workflow()
        test_search_workflow()
        test_kld_parser()
        test_stats()
        test_runner_end_to_end(tmp, BASE)
        test_report_all_workflows(tmp)
    finally:
        mock.shutdown()
    bad = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS)} checks, {len(CHECKS) - len(bad)} passed, {len(bad)} failed "
          f"in {time.time() - t0:.1f}s")
    if bad:
        for name, _, detail in bad:
            print(f"FAILED: {name} {detail}")
        raise SystemExit(1)
    print("selftest OK")


if __name__ == "__main__":
    main()
