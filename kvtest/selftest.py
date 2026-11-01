"""Selftest: everything the harness claims, checked without a GPU, a model or the network.

`python3 -m kvtest.selftest` must print `selftest OK`. It stands up a stdlib HTTP mock that speaks
`/health`, `/v1/models` and `/v1/chat/completions`, drives one full arm through `run.Runner` against
it, and validates the graders, the loop instrument, the code-tier runners and the report generator.
"""
import http.server
import importlib.util
import json
import os
import shutil
import socketserver
import sys
import tempfile
import threading

from . import client, codegraders, config, loops, longctx, report, tasks

OK = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    OK.append(name)
    print(f"  ok  {name}")


# ---------------------------------------------------------------- mock llama-server
LOOP_TEXT = ("the shard is sealed and the ledger is closed " * 40)
CLEAN_TEXT = ("The operator signs the manifest. A RED row is never settled against an OPEN shard. "
              "Each custodian hands over the badge on retirement.")
TOOL_SEQ = [
    {"name": "read_shard", "args": {"name": "orion"}},
    {"name": "read_shard", "args": {"name": "orion"}},
    {"name": "submit_report", "args": {"shards": ["orion=AK-1000"]}},
]


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                                     # silence
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "ok"})
        if self.path == "/v1/models":
            return self._json({"data": [{"id": "mock"}]})
        if self.path == "/props":
            return self._json({"model_path": "mock"})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages") or []
        text = "\n".join(m.get("content") or "" for m in msgs)
        if self.path == "/boom":
            return self._json({"error": "server exploded"}, 500)
        if req.get("tools"):
            content, out_tokens, tc = _tool_reply(msgs, req)
            return self._json(self._completion(content, out_tokens, tool_calls=tc))
        if "[[loop]]" in text:
            return self._json(self._completion(LOOP_TEXT, 900))
        if "[[trunc]]" in text:
            return self._json(self._completion("", 2048, reasoning="thinking " * 300,
                                              finish="length"))
        if "[[clean]]" in text:
            return self._json(self._completion(CLEAN_TEXT, 60))
        return self._json(self._completion(_canned(text), 400))

    def _completion(self, content, out_tokens, reasoning="", finish="stop", tool_calls=None):
        usage = {"prompt_tokens": 96000, "completion_tokens": out_tokens,
                 "total_tokens": 96000 + out_tokens}
        msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": usage,
                "timings": {"prompt_n": 96000, "prompt_ms": 240000.0, "prompt_per_second": 400.0,
                            "predicted_n": out_tokens, "predicted_ms": out_tokens * 50.0,
                            "predicted_per_second": 20.0, "cache_n": 95500}}


_TOOL_STATE = {"turn": 0}


def _tool_reply(msgs, req):
    """First turn: two identical calls (a loop). Then a submit_report."""
    n_tool_msgs = sum(1 for m in msgs if m.get("role") == "tool")
    call = TOOL_SEQ[2] if n_tool_msgs >= 2 else TOOL_SEQ[0]
    tc = [{"id": f"c{n_tool_msgs}", "type": "function",
           "function": {"name": call["name"], "arguments": json.dumps(call["args"])}}]
    return "", 40, tc


def _canned(text):
    if "RULE 1 through RULE 12" in text:
        return "\n\n".join(f"RULE {i}: " + "operators check the manifest and sign it. " * 6
                           for i in range(1, 13))
    if "Reproduce the 12 consecutive LEDGER" in text:
        return "\n".join(f"LEDGER L-{i:04d} status=OPEN tag=RED amount={i}" for i in range(41, 53))
    if "List all 40 shards" in text:
        return "\n".join(f"shard{i}=ZZ-{i:04d}" for i in range(40))
    if "### FILE:" in text:
        return ("### FILE: intervalkit/core.py\n```python\n"
                "from .intervalkit import nothing\n```\n")
    if "ANSWER:" in text:
        return "reasoning about the document\nANSWER: ZZ-0000"
    return CLEAN_TEXT


class MockServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_mock(port=0):
    srv = MockServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ---------------------------------------------------------------- runner against the mock
class _FakeVram:
    peak = 1234


class FakeArmServer:
    """Stands in for server.Server: same surface, backed by the mock HTTP server."""

    def __init__(self, model_key, kv, ctx, knobs=None, port=None, log_path=None):
        self.model_key, self.kv, self.ctx, self.knobs = model_key, kv, ctx, knobs or {}
        self.log_path = log_path or "/tmp/fake.log"
        self._s, self.base = start_mock(0)
        self.start_s, self.vram = 0.1, _FakeVram()
        self.proc = None

    def start(self, timeout=10):
        return True

    def stop(self, hard_after=0):
        self._s.shutdown()
        self._s.server_close()

    def log_tail(self, n=4000):
        return ""

    def oom_reason(self):
        return None


def test_client(mock_base):
    r = client.chat(mock_base, [{"role": "user", "content": "hello [[clean]]"}], max_tokens=64)
    check("client parses content", "operator signs the manifest" in r["content"], r["content"][:60])
    check("client parses usage", r["usage"]["prompt_tokens"] == 96000, r["usage"])
    check("client parses timings", r["decode_tps"] == 20.0 and r["timings"]["cache_n"] == 95500)
    r = client.chat(mock_base, [{"role": "user", "content": "x [[trunc]]"}], max_tokens=64)
    check("client captures reasoning", len(r["reasoning_content"]) > 100)
    check("client captures finish_reason", r["finish_reason"] == "length")
    r = client.chat(mock_base, [{"role": "user", "content": "x"}], max_tokens=8)
    r = client.chat("http://127.0.0.1:1", [{"role": "user", "content": "x"}], max_tokens=8,
                    timeout=2)
    check("client turns connection failure into error", bool(r["error"]), r)
    tools = [{"type": "function", "function": {"name": "read_shard", "parameters": {}}}]
    r = client.chat(mock_base, [{"role": "user", "content": "use tools"}], max_tokens=8,
                    tools=tools)
    calls = tasks.normalize_calls(r["tool_calls"])
    check("tool_calls normalised to name+args", calls[0]["name"] == "read_shard"
          and calls[0]["args"] == {"name": "orion"}, calls)


def test_graders(facts):
    rts = tasks.r_tasks(facts)
    multi = next(t for t in rts if t["id"] == "niah_multi")
    right = ",".join(multi["expected"])
    g = tasks.grade_r(multi, f"some reasoning\nANSWER: {right}\n")
    check("R grader accepts correct answer", g["ok"] and g["points"] == 3, g)
    g = tasks.grade_r(multi, "ANSWER: " + right.split(",")[0] + ",ZZ-0000,ZZ-0001")
    check("R grader gives partial credit", g["points"] == 1 and not g["ok"], g)
    g = tasks.grade_r(multi, "no answer line here")
    check("R grader rejects missing ANSWER", not g["ok"] and g["why"] == "no ANSWER line", g)
    agg = next(t for t in rts if t["id"] == "agg_sum")
    g = tasks.grade_r(agg, "thinking\nANSWER: 999,999")
    check("R grader rejects wrong values", not g["ok"] and g["points"] == 0, g)
    g = tasks.grade_r(multi, f"ANSWER: {' '.join(right.split(','))}")
    check("R grader rejects a non-comma form", not g["ok"], g)

    lts = {t["id"]: t for t in tasks.l_tasks(facts)}
    man = _canned("RULE 1 through RULE 12")
    check("L gen_manual counts 12 paragraphs",
          lts["gen_manual"]["check"](man, lts["gen_manual"]["expected"], facts)["ok"])
    rows = "\n".join(lts["copy_ledgers"]["expected"]["rows"])
    check("L copy_ledgers accepts the exact rows",
          lts["copy_ledgers"]["check"](rows, lts["copy_ledgers"]["expected"], facts)["ok"])
    short = "\n".join(lts["copy_ledgers"]["expected"]["rows"][:5])
    check("L copy_ledgers rejects a short copy",
          not lts["copy_ledgers"]["check"](short, lts["copy_ledgers"]["expected"], facts)["ok"])
    enumer = "\n".join(lts["enumerate_shards"]["expected"]["rows"])
    check("L enumerate_shards accepts the exact list",
          lts["enumerate_shards"]["check"](enumer, lts["enumerate_shards"]["expected"], facts)["ok"])


def test_loops():
    m_loop = loops.metrics(LOOP_TEXT)
    flagged, reason = loops.verdict(m_loop)
    check("loops flags repeated text", flagged, reason)
    m_clean = loops.metrics(CLEAN_TEXT)
    flagged, reason = loops.verdict(m_clean)
    check("loops does not flag clean text", not flagged, reason)
    check("loops flags no-answer after long generation",
          loops.verdict(loops.metrics(CLEAN_TEXT + " x"), graded=True, got_answer=False,
                        gen_tokens=2048)[0])
    rep = [{"tool": "read_shard", "args": {"name": "orion"}, "ok": True}] * 2
    check("tool_loop flags identical consecutive calls", tasks.tool_loop(rep)[0])
    many = [{"tool": "read_shard", "args": {"name": "orion"}, "ok": False}] * 3
    check("tool_loop flags three identical calls", tasks.tool_loop(many)[0])
    stuck = [{"tool": "read_shard", "args": {"name": f"s{i}"}, "ok": False} for i in range(4)]
    check("tool_loop flags no progress after 4 turns", tasks.tool_loop(stuck)[0])
    varied = [{"tool": "read_shard", "args": {"name": f"s{i}"}, "ok": True} for i in range(4)]
    check("tool_loop stays quiet when work happens", not tasks.tool_loop(varied)[0])


def _gen_codegen_module():
    path = os.path.join(config.FIXTURES, "codegen", "gen-codegen.py")
    spec = importlib.util.spec_from_file_location("_gen_codegen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_code_tier(tmp):
    gen = _gen_codegen_module()
    refs = {"statistics": "REF_STATS", "fixedpoint": "REF_FIXED", "semver": "REF_SEMVER",
            "csvlite": "REF_CSV", "intervals": "REF_INTERVALS"}
    semver = getattr(gen, refs["semver"])
    cand = os.path.join(tmp, "semver.py")
    with open(cand, "w") as f:
        f.write(semver)
    res = codegraders.run_codegen(cand, os.path.join(codegraders.CODEGEN_DIR, "semver",
                                                    "checks.json"))
    check("reference semver scores 261/261", res["passed"] == res["of"] == 261,
          f"{res['passed']}/{res['of']} {res.get('load_error')}")

    broken = semver.replace("def compare(", "def disabled_compare(", 1)
    with open(cand, "w") as f:
        f.write(broken)
    res = codegraders.run_codegen(cand, os.path.join(codegraders.CODEGEN_DIR, "semver",
                                                    "checks.json"))
    check("broken semver scores below full", res["passed"] < 261, f"{res['passed']}/{res['of']}")

    text = ("blah\n```python\nthis is not python(\n```\n"
            "```python\n" + semver + "```\n")
    check("extract_code prefers the block that compiles",
          codegraders.extract_code(text).strip().startswith("import")
          or "def " in codegraders.extract_code(text))
    files = codegraders.extract_files(
        "### FILE: intervalkit/core.py\n```python\nclass Interval:\n    pass\n```\n"
        "### FILE: intervalkit/merge.py\nimport os\n    def f():\n        return 1\n")
    check("extract_files parses both header shapes",
          set(files) == {"intervalkit/core.py", "intervalkit/merge.py"}
          and files["intervalkit/core.py"].startswith("class Interval"), files.keys())

    work = os.path.join(tmp, "repo-fixed")
    shutil.copytree(codegraders.REPO_DIR, work)
    _repair_repo(work)
    res = codegraders.run_repo(work, os.path.join(codegraders.REPO_DIR, "checks.json"))
    check("repaired repository scores 54/54", res["passed"] == res["of"] == 54,
          f"{res['passed']}/{res['of']} {res.get('load_error')}")
    work2 = os.path.join(tmp, "repo-buggy")
    shutil.copytree(codegraders.REPO_DIR, work2)
    res = codegraders.run_repo(work2, os.path.join(codegraders.REPO_DIR, "checks.json"))
    check("buggy repository scores below full", res["passed"] < res["of"],
          f"{res['passed']}/{res['of']}")


def _repair_repo(path):
    """The known-good repair of the shipped fixture: abuts must require a covered meeting point."""
    core = os.path.join(path, "intervalkit", "core.py")
    s = open(core).read()
    s = s.replace("""        if self.hi == other.lo:
            return True
        if other.hi == self.lo:
            return True""", """        if self.hi == other.lo:
            return bool(self.hi_closed or other.lo_closed)
        if other.hi == self.lo:
            return bool(other.hi_closed or self.lo_closed)""")
    with open(core, "w") as f:
        f.write(s)


def test_runner_end_to_end(tmp):
    from . import run as runmod
    doc, facts = longctx.build_document(filler_chars=4000)
    out_dir = os.path.join(tmp, "results", "selftest")
    os.makedirs(out_dir, exist_ok=True)
    real = runmod.srv.Server
    runmod.srv.Server = FakeArmServer
    try:
        runner = runmod.Runner(stamp="selftest", ctx=8192, knobs_map={}, doc=doc, facts=facts,
                               reps=1, code_reps=1, models=["qwen38-27b"], kvs=["q4_0"],
                               deadline_hours=1, out_dir=out_dir)
        raw = runner.run()
    finally:
        runmod.srv.Server = real
    samples = [json.loads(ln) for ln in open(os.path.join(out_dir, "samples.jsonl"))
               if ln.strip()]
    check("runner produced 6 R + 6 C + 4 L samples", len(samples) == 16, len(samples))
    check("every sample carries loop metrics and a grade",
          all("loops" in s and ("grade" in s or "detail" in s) for s in samples))
    check("every sample is attributed to an arm",
          all(s["model"] and s["kv"] for s in samples),
          [s["task"] for s in samples if not s["model"]])
    check("arm checkpoint finished",
          json.load(open(os.path.join(out_dir, "arms", "qwen38-27b__q4_0.json")))["finished"])
    tool = next(s for s in samples if s["task"] == "tool_episode")
    check("tool episode flagged as a loop", tool["flagged"], tool.get("flag_reason"))
    check("loop sample flagged in the sample stream", any(s["flagged"] for s in samples))
    check("run recorded the hardware it ran on",
          os.path.exists(os.path.join(out_dir, "hardware.json")), os.listdir(out_dir))
    hw = json.load(open(os.path.join(out_dir, "hardware.json")))
    from . import hostinfo
    check("hardware capture names the GPU, the CPU and the build flags",
          bool((hw.get("gpu") or {}).get("name")) and bool((hw.get("cpu") or {}).get("model"))
          and (hw.get("llama_cpp") or {}).get("flags"), hostinfo.one_line(hw))
    check("raw.json has one arm and a manifest",
          len(raw["arms"]) == 1 and "llama_cpp" in raw["manifest"])
    return out_dir


def test_report(tmp, runner_out):
    out = os.path.join(tmp, "results", "synthetic")
    os.makedirs(out, exist_ok=True)
    models = ["qwen36-35b", "qwen38-27b", "qwen38-flash", "gemma4-26b-a4b", "ornith-35b"]
    samples = []
    for mi, model in enumerate(models):
        for ki, kv in enumerate(config.KV_ARMS):
            for t in range(3):
                for rep in range(2):
                    flagged = (kv == "q4_0" and t == 0 and rep == 0)
                    samples.append({
                        "stamp": "synthetic", "model": model, "kv": kv, "tier": "R",
                        "task": ["niah_multi", "agg_sum", "fwe_count"][t], "rep": rep,
                        "seed": 1001 + rep, "prompt_tokens": 95000 + mi * 100,
                        "prefill_tps": 400, "decode_tps": 20 - mi, "cache_n": 94000,
                        "out_tokens": 900, "error": None, "flagged": flagged,
                        "flag_reason": "loop" if flagged else "",
                        "grade": {"points": 2 - (ki + rep) % 2, "of": 2, "ok": True,
                                  "answer": f"A{ki}{rep}" if kv != "f16" else "A0",
                                  "why": ""},
                        "loops": {"content": {"flagged": flagged, "reason": "x",
                                              "metrics": {"word_run_1": 3, "chars": 900,
                                                          "zip_ratio": 0.3}},
                                  "reasoning_content": {"flagged": False, "reason": "",
                                                        "metrics": {"word_run_1": 2}}},
                    })
            samples.append({"stamp": "synthetic", "model": model, "kv": kv, "tier": "C",
                            "task": "cg_semver", "rep": 0, "seed": 1001, "error": None,
                            "flagged": False, "out_tokens": 400,
                            "grade": {"passed": 261 - mi * ki, "of": 261, "resolved": ki == 0,
                                      "lines": 70, "fails": []}, "loops": {}})
            samples.append({"stamp": "synthetic", "model": model, "kv": kv, "tier": "L",
                            "task": "gen_manual", "rep": 0, "seed": 1001, "error": None,
                            "flagged": kv == "q4_0", "flag_reason": "",
                            "grade": {"ok": True, "score": 1.0, "why": ""},
                            "loops": {"content": {"flagged": kv == "q4_0", "reason": "line_run=6",
                                                  "metrics": {"line_run": 6, "chars": 1200,
                                                              "zip_ratio": 0.5}}}})
    with open(os.path.join(out, "samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    with open(os.path.join(out, "raw.json"), "w") as f:
        json.dump({"manifest": config.build_manifest(), "stamp": "synthetic", "context": 98304,
                   "arms": {f"{m}__{kv}": {"model": m, "kv": kv, "ctx": 98304, "knobs": {}}
                            for m in models for kv in config.KV_ARMS},
                   "samples": len(samples), "document": {"sha256": "0" * 64},
                   "deadline_reached": False, "unrun_arms": []}, f)
    out_dir, raw = report.build(out)
    page = open(os.path.join(out, "report.html")).read()
    check("report html has the three tabs",
          all(f'id="t-{t}"' in page for t in ("matrix", "tasks", "loops")))
    check("report html has 3 figures of 15 cells", page.count('rx="4"') == 45,
          page.count('rx="4"'))
    figs = os.path.join(out, "figures")
    check("report exported standalone figures",
          sorted(os.listdir(figs)) == [f"fig{i}.svg" for i in range(1, 6)] + ["figures.json"],
          os.listdir(figs) if os.path.isdir(figs) else "missing")
    before = open(os.path.join(figs, "fig1.svg")).read()
    report.build(out)
    check("figure export is byte-identical on rebuild",
          before == open(os.path.join(figs, "fig1.svg")).read())
    check("paired test is exact and symmetric",
          report.perm_p([1, 1, 1, 1, 1, 1])[1] == 2 / 64
          and report.perm_p([0] * 6)[1] == 1.0
          and report.perm_p([1, -1])[1] == 1.0
          and report.perm_p([-1, -1, -1, -1, -1, -1])[1] == 2 / 64)
    check("report html embeds the raw data", "const RAW =" in page)
    check("report markdown is non-empty and has the tables",
          os.path.getsize(os.path.join(out, "REPORT.md")) > 1500
          and "## Loop table" in open(os.path.join(out, "REPORT.md")).read())
    check("matrix.csv has one row per arm",
          len(open(os.path.join(out, "matrix.csv")).read().strip().splitlines()) == 16)
    check("report is idempotent (rebuild from raw+samples)", report.build(out)[1]["stamp"]
          == "synthetic")
    v = out_dir["verdict"]
    check("verdict ranks every arm and names one overall winner",
          v.get("overall") and len(v["flat"]) == 15 and len(v["per_model"]) == 5,
          (v.get("overall"), len(v.get("flat") or [])))
    check("verdict ranking follows capability then loops then decode",
          [t[2] for t in v["flat"]] == sorted([t[2] for t in v["flat"]], reverse=True),
          [t[2] for t in v["flat"]])
    # and it must also work on the real end-to-end arm the mock produced
    report.build(runner_out)
    md = open(os.path.join(runner_out, "REPORT.md")).read()
    check("report prints the hardware and build section",
          "## Hardware and build this ran on" in md and "| GPU |" in md,
          [ln for ln in md.splitlines() if "GPU" in ln][:2])
    check("report falls back to hardware.json when raw.json predates capture",
          "hardware.json" in md or "captured at matrix start" in md,
          [ln for ln in md.splitlines() if "Recorded:" in ln][:1])


def main():
    print("selftest: no GPU, no models, no network")
    mocksrv, base = start_mock(0)
    tmp = tempfile.mkdtemp(prefix="kvselftest-")
    try:
        print("- client / http")
        test_client(base)
        doc, facts = longctx.build_document(filler_chars=4000)
        print("- document + graders")
        check("document is deterministic",
              longctx.sha256(longctx.build_document(filler_chars=4000)[0])
              == longctx.sha256(doc))
        check("shard codes are unique", len({s['code'] for s in facts['shards']}) == 40)
        check("every ledger id is unique", len({l['id'] for l in facts['ledgers']}) == 120)
        check("ZQ-7731 decoys are not counted", facts["fwe_count"] == 20, facts["fwe_count"])
        test_graders(facts)
        print("- loops")
        test_loops()
        print("- code tier (subprocess runners)")
        test_code_tier(tmp)
        print("- orchestrator against the mock server")
        runner_out = test_runner_end_to_end(tmp)
        print("- report")
        test_report(tmp, runner_out)
    finally:
        mocksrv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nselftest OK ({len(OK)} checks)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nselftest FAILED: {e}")
        sys.exit(1)
