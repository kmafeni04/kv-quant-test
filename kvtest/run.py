"""Orchestrator: 5 models x 3 KV types, resumable per arm, deadline-governed.

Ordering is f16 first within each model (it is the reference), models in plan priority order, so a
deadline cut always costs whole arms at the tail rather than half of a KV ladder.

Every request appends one line to samples.jsonl and rewrites the arm checkpoint atomically, so a
crash, a closed terminal or a deadline cut loses at most one request.
"""
import argparse
import json
import os
import statistics
import time

from . import (client, codegraders, config, fit as fitmod, hostinfo, longctx, loops,
               server as srv, tasks)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


class Runner:
    def __init__(self, stamp, ctx, knobs_map, doc, facts, reps=config.REPS,
                 code_reps=1, models=None, kvs=None, deadline_hours=22.0, resume=False,
                 code_temp=0.0, out_dir=None):
        self.stamp = stamp
        self.out_dir = out_dir or os.path.join(config.ROOT, "results", stamp)
        self.arms_dir = os.path.join(self.out_dir, "arms")
        self.logs_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.arms_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.ctx = ctx
        self.knobs_map = knobs_map or {}
        self.doc = doc
        self.facts = facts
        self.reps = reps
        self.code_reps = code_reps
        self.code_temp = code_temp
        self.models = models or [k for k, _ in sorted(config.MODELS.items(),
                                                     key=lambda kv: kv[1]["priority"])]
        self.kvs = kvs or config.KV_ARMS
        self.deadline = time.time() + deadline_hours * 3600 if deadline_hours else None
        self.resume = resume
        self.samples_path = os.path.join(self.out_dir, "samples.jsonl")
        self.fit = {}
        fit_path = os.path.join(self.out_dir, "fit.json")
        if os.path.exists(fit_path):
            self.fit = json.load(open(fit_path))
        self.arm_durations = []
        self.r_tasks = tasks.r_tasks(facts)
        self.l_tasks = tasks.l_tasks(facts)
        self.c_tasks = codegraders.c_tasks()
        self.doc_sha = longctx.sha256(doc)
        self.run_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.run_finished = None
        self.hardware = hostinfo.snapshot()
        _atomic_write(os.path.join(self.out_dir, "hardware.json"), self.hardware)
        log(f"host: {hostinfo.one_line(self.hardware)}")

    # ------------------------------------------------------------ request plumbing
    def _sample(self, base, prompt_or_messages, budget, sampling, tier, task_id, rep, seed,
                graded=False, timeout=3600, tools=None):
        messages = prompt_or_messages if isinstance(prompt_or_messages, list) else \
            [{"role": "user", "content": self.doc + "\n\n" + prompt_or_messages}]
        sampling = dict(sampling)
        rb = config.REASONING_BUDGET_TOKENS.get(tier)
        if rb is not None:
            sampling["reasoning_budget_tokens"] = rb
        r = client.chat(base, messages, max_tokens=budget, timeout=timeout,
                        tools=tools, seed=seed, **sampling)
        rec = {
            "stamp": self.stamp, "model": self.cur_model, "kv": self.cur_kv,
            "tier": tier, "task": task_id,
            "rep": rep, "seed": seed, "budget": budget, "graded": graded,
            "error": r.get("error"),
            "content": r.get("content") or "", "reasoning_content": r.get("reasoning_content") or "",
            "tool_calls": r.get("tool_calls") or [],
            "finish_reason": r.get("finish_reason"),
            "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
            "out_tokens": (r.get("usage") or {}).get("completion_tokens"),
            "prefill_tps": r.get("prefill_tps"), "decode_tps": r.get("decode_tps"),
            "cache_n": (r.get("timings") or {}).get("cache_n"),
            "prompt_ms": (r.get("timings") or {}).get("prompt_ms"),
            "wall_s": r.get("wall_s"),
        }
        return rec, r

    def _loop_fields(self, rec, graded, got_answer):
        out = {}
        for part in ("reasoning_content", "content"):
            m = loops.metrics(rec.get(part) or "")
            flagged, reason = loops.verdict(m, graded=graded, got_answer=got_answer,
                                            gen_tokens=rec.get("out_tokens") or 0)
            out[part] = {"metrics": m, "flagged": flagged, "reason": reason}
        return out

    # ------------------------------------------------------------ one arm
    def run_arm(self, model, kv):
        name = f"{model}__{kv}"
        arm_path = os.path.join(self.arms_dir, name + ".json")
        knobs = self.knobs_map.get(model) or config.MODELS[model]["knobs"]
        state = {"model": model, "kv": kv, "ctx": self.ctx, "knobs": knobs,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "finished": None, "done": [], "flags": {}, "errors": [],
                 "document_sha256": self.doc_sha}
        if self.resume and os.path.exists(arm_path):
            prev = json.load(open(arm_path))
            if prev.get("finished"):
                log(f"arm {name}: already finished, skipping")
                return prev
            if prev.get("document_sha256") == self.doc_sha:
                state.update(done=prev.get("done") or [], flags=prev.get("flags") or {},
                             errors=prev.get("errors") or [], started=prev.get("started"))
                log(f"arm {name}: resuming with {len(state['done'])} requests already done")

        self.cur_model, self.cur_kv = model, kv
        s = srv.Server(model, kv, self.ctx, knobs=knobs, port=config.PORT,
                       log_path=os.path.join(self.logs_dir, name + ".log"))
        t0 = time.time()
        if not s.start():
            state.update(finished=None, failed="server did not start",
                         log_tail=s.log_tail(2000))
            _atomic_write(arm_path, state)
            log(f"arm {name}: SERVER DID NOT START")
            return state
        state["load_s"] = s.start_s
        log(f"arm {name}: up in {s.start_s}s, knobs={knobs}, ctx={self.ctx}")

        def progress(key):
            state["done"].append(key)
            _atomic_write(arm_path, state)
            if self.deadline and time.time() > self.deadline:
                raise TimeoutError("deadline reached mid-arm")

        def finish():
            state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            state["wall_s"] = round(time.time() - t0, 1)
            state["vram_peak_mib"] = s.vram.peak
            _atomic_write(arm_path, state)

        done = set(state["done"])
        first_request = not done
        cache_expect = None
        try:
            # ---- tier R: retrieval / tracing / aggregation, temp 0.7, repetition suppression off
            for task in self.r_tasks:
                for rep in range(self.reps):
                    key = f"R:{task['id']}:{rep}"
                    if key in done:
                        continue
                    rec, r = self._sample(s.base, task["prompt"], config.BUDGETS["R"],
                                          config.SAMPLING_LOOP, "R", task["id"], rep,
                                          config.SEEDS[rep % len(config.SEEDS)], graded=True)
                    if first_request and not r.get("error"):
                        state["flags"]["reasoning_supported"] = bool(r.get("reasoning_content"))
                        first_request = False
                    grade = tasks.grade_r(task, rec["content"]) if not rec["error"] else \
                        {"ok": False, "points": 0, "of": len(task["expected"]), "why": rec["error"],
                         "answer": None}
                    rec["grade"] = grade
                    rec["budget_raised"] = False
                    if (not rec["error"] and rec.get("finish_reason") == "length"
                            and grade.get("answer") is None):
                        rec2, _ = self._sample(s.base, task["prompt"],
                                              config.BUDGETS["R"] * config.BUDGET_RETRY_MULT,
                                              config.SAMPLING_LOOP, "R", task["id"], rep,
                                              config.SEEDS[rep % len(config.SEEDS)], graded=True)
                        if not rec2["error"]:
                            rec2["grade"] = tasks.grade_r(task, rec2["content"])
                            rec2["budget_raised"] = True
                            rec = rec2
                    rec.update(model=model, kv=kv, cache_expect=cache_expect)
                    rec["loops"] = self._loop_fields(rec, True, rec["grade"].get("answer") is not None)
                    rec["flagged"] = any(v["flagged"] for v in rec["loops"].values())
                    rec["flag_reason"] = "; ".join(f"{k}:{v['reason']}" for k, v in rec["loops"].items()
                                                   if v["flagged"])
                    cache_expect = rec.get("prompt_tokens")
                    if rec.get("cache_n") and cache_expect and rec["cache_n"] < 0.5 * cache_expect \
                            and len(state["done"]) > 0:
                        state["flags"]["cache_miss"] = state["flags"].get("cache_miss", 0) + 1
                    _append(self.samples_path, rec)
                    progress(key)
                    log(f"  {name:26} R {task['id']:14} r{rep} "
                        f"{'ERR' if rec['error'] else f'{rec['grade']['points']}/{rec['grade']['of']}'}"
                        f" out={rec['out_tokens']} dec={rec['decode_tps']} "
                        f"cache={rec.get('cache_n')}")

            # ---- tier C: greedy code tier
            for task in self.c_tasks:
                for rep in range(self.code_reps):
                    key = f"C:{task['id']}:{rep}"
                    if key in done:
                        continue
                    rec, r = self._sample(s.base, task["prompt"], config.BUDGETS["C"],
                                          config.SAMPLING_GRADED, "C", task["id"], rep,
                                          config.SEEDS[rep % len(config.SEEDS)], graded=True,
                                          timeout=7200)
                    work = codegraders.workdir(self.out_dir, f"{name}-{task['id']}-r{rep}")
                    grade = codegraders.grade_c(task, rec["content"], work) if not rec["error"] \
                        else {"passed": 0, "of": task["of"], "score": 0.0, "resolved": False,
                              "load_error": rec["error"], "fails": [], "lines": 0}
                    rec["grade"] = grade
                    rec["budget_raised"] = False
                    if (not rec["error"] and rec.get("finish_reason") == "length"
                            and grade.get("lines", 0) < 5):
                        rec2, _ = self._sample(s.base, task["prompt"],
                                              config.BUDGETS["C"] * config.BUDGET_RETRY_MULT,
                                              config.SAMPLING_GRADED, "C", task["id"], rep,
                                              config.SEEDS[rep % len(config.SEEDS)], graded=True,
                                              timeout=7200)
                        if not rec2["error"]:
                            rec2["grade"] = codegraders.grade_c(task, rec2["content"], work)
                            rec2["budget_raised"] = True
                            rec = rec2
                    rec.update(model=model, kv=kv)
                    # "answered" means code was emitted, not that every hidden check passed
                    rec["loops"] = self._loop_fields(rec, True, (grade.get("lines") or 0) > 0)
                    rec["flagged"] = any(v["flagged"] for v in rec["loops"].values())
                    rec["flag_reason"] = "; ".join(f"{k}:{v['reason']}" for k, v in rec["loops"].items()
                                                   if v["flagged"])
                    rec["detail"] = grade
                    _append(self.samples_path, rec)
                    progress(key)
                    log(f"  {name:26} C {task['id']:14} r{rep} "
                        f"{'ERR' if rec['error'] else f'{grade['passed']}/{grade['of']}'}"
                        f" lines={grade.get('lines')} out={rec['out_tokens']}")

            # ---- tier L: loop probes, repetition suppression off
            for task in self.l_tasks:
                for rep in range(self.reps):
                    key = f"L:{task['id']}:{rep}"
                    if key in done:
                        continue
                    seed = config.SEEDS[rep % len(config.SEEDS)]
                    if task["id"] == "tool_episode":
                        ep = tasks.run_tool_episode(
                            lambda msgs, **kw: client.chat(s.base, msgs, timeout=3600,
                                                           seed=seed, **kw),
                            self.doc, task, self.facts, config.BUDGETS["tool_turn"],
                            max_turns=6,
                            path={**config.SAMPLING_LOOP,
                                  "reasoning_budget_tokens":
                                      config.REASONING_BUDGET_TOKENS["tool_turn"]})
                        rec = {"stamp": self.stamp, "model": model, "kv": kv, "tier": "L",
                               "task": task["id"], "rep": rep, "seed": seed,
                               "budget": config.BUDGETS["tool_turn"], "graded": False,
                               "error": next((e for e in ep["errors"] if e), None),
                               "content": ep["content"], "reasoning_content": ep["reasoning"],
                               # the real calls live in transcript/detail; mirror them here so a
                               # reader grepping tool_calls sees the episode instead of []
                               "tool_calls": [{"name": c["tool"], "arguments": c["args"]}
                                              for c in ep["transcript"]],
                               "finish_reason": None, "prompt_tokens": None,
                               "out_tokens": None, "prefill_tps": None, "decode_tps": None,
                               "cache_n": ep["cache_n"], "wall_s": None, "budget_raised": False}
                        grade = tasks.grade_tool_episode(ep["transcript"], ep["submitted"])
                        if not ep["transcript"]:
                            state["flags"]["tool_unsupported"] = True
                        rec["detail"] = grade
                        rec["loops"] = self._loop_fields(rec, False, True)
                        if grade["flagged"]:
                            rec["loops"]["transcript"] = {
                                "metrics": {}, "flagged": True,
                                "reason": grade["reason"],
                                "span": json.dumps(ep["transcript"])[:600]}
                        rec["flagged"] = any(v["flagged"] for v in rec["loops"].values())
                        rec["flag_reason"] = "; ".join(f"{k}:{v['reason']}"
                                                       for k, v in rec["loops"].items() if v["flagged"])
                        rec["transcript"] = ep["transcript"]
                    else:
                        rec, _ = self._sample(s.base, task["prompt"], config.BUDGETS["L"],
                                              config.SAMPLING_LOOP, "L", task["id"], rep, seed)
                        check = task["check"](rec["content"], task["expected"], self.facts)
                        rec["grade"] = check
                        rec["budget_raised"] = False
                        rec["loops"] = self._loop_fields(rec, False, True)
                        rec["flagged"] = any(v["flagged"] for v in rec["loops"].values())
                        rec["flag_reason"] = "; ".join(f"{k}:{v['reason']}"
                                                       for k, v in rec["loops"].items()
                                                       if v["flagged"])
                    _append(self.samples_path, rec)
                    progress(key)
                    log(f"  {name:26} L {task['id']:14} r{rep} "
                        f"{'ERR' if rec['error'] else 'loop' if rec['flagged'] else 'clean'}"
                        f" {rec['flag_reason'][:60]}")
        except TimeoutError as e:
            state["cut"] = str(e)
            log(f"arm {name}: cut by deadline")
        except KeyboardInterrupt:
            state["cut"] = "interrupted"
            finish()
            s.stop()
            raise
        except Exception as e:                                     # noqa: BLE001
            state["cut"] = f"{type(e).__name__}: {e}"[:300]
            log(f"arm {name}: ABORTED {state['cut']}")
        finally:
            s.stop()
        finish()
        self.arm_durations.append(time.time() - t0)
        log(f"arm {name}: done in {round(time.time() - t0, 1)}s, vram peak {s.vram.peak} MiB")
        return state

    # ------------------------------------------------------------ whole matrix
    def run(self):
        planned = [(m, kv) for m in self.models for kv in self.kvs]
        unrun = []
        for i, (model, kv) in enumerate(planned):
            if self.deadline and i > 0:
                est = statistics.median(self.arm_durations or [self._estimate(model)])
                est = max(est, self._estimate(model))
                if time.time() + est > self.deadline:
                    log(f"deadline governor: est {est / 3600:.2f}h for next arm does not fit; "
                        f"stopping with {len(planned) - i} arms unrun")
                    unrun = planned[i:]
                    break
            if not os.path.exists(config.MODELS[model]["path"]):
                log(f"arm {model}/{kv}: model file missing, skipped")
                continue
            try:
                self.run_arm(model, kv)
            except KeyboardInterrupt:
                raise
            except Exception as e:                                 # noqa: BLE001
                log(f"arm {model}/{kv}: crashed ({type(e).__name__}: {e}); continuing")
        self.run_finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # an arm that never reached a finished checkpoint is incomplete whether it was skipped by the
        # deadline, a crash, or a server that did not start: the report must say so instead of
        # silently shipping a partial matrix as "complete"
        incomplete = [f"{m}__{kv}" for m, kv in planned
                      if not self._arm_finished(m, kv)]
        raw = self.build_raw(unrun=unrun, incomplete=incomplete)
        _atomic_write(os.path.join(self.out_dir, "raw.json"), raw)
        if incomplete:
            log(f"raw.json written: INCOMPLETE - no finished checkpoint for "
                f"{', '.join(incomplete)}")
        else:
            log(f"raw.json written: complete")
        return raw

    def _arm_finished(self, model, kv):
        path = os.path.join(self.arms_dir, f"{model}__{kv}.json")
        if not os.path.exists(path):
            return False
        try:
            return bool(json.load(open(path)).get("finished"))
        except json.JSONDecodeError:
            return False

    def _estimate(self, model):
        """Seconds for one arm from the fit probe's prefill/decode rates."""
        entry = (self.fit.get("models") or {}).get(model) or {}
        prefill = entry.get("prefill_tps") or 200
        decode = entry.get("decode_tps") or 15
        prompt_tokens = entry.get("prompt_tokens") or (self.ctx - config.GEN_HEADROOM)
        gen = (len(self.r_tasks) * self.reps * config.BUDGETS["R"]
               + len(self.c_tasks) * self.code_reps * config.BUDGETS["C"]
               + len(self.l_tasks) * self.reps * config.BUDGETS["L"]
               + self.reps * 3 * config.BUDGETS["tool_turn"])
        return prompt_tokens / prefill + gen / decode + 120

    def build_raw(self, unrun=None, incomplete=None):
        samples = []
        if os.path.exists(self.samples_path):
            with open(self.samples_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        samples = [s for s in samples if s.get("stamp") == self.stamp]
        arms = {}
        for fn in sorted(os.listdir(self.arms_dir)):
            if not fn.endswith(".json"):
                continue
            a = json.load(open(os.path.join(self.arms_dir, fn)))
            if a.get("model") not in self.models or a.get("kv") not in self.kvs:
                continue
            arms[f"{a['model']}__{a['kv']}"] = a
        manifest = config.build_manifest()
        manifest["document_sha256"] = self.doc_sha
        manifest["context"] = self.ctx
        manifest["knobs"] = self.knobs_map
        manifest["host_runtime"] = self.hardware
        return {
            "manifest": manifest,
            "stamp": self.stamp,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "context": self.ctx,
            "document": {"sha256": self.doc_sha, "chars": len(self.doc),
                         "filler_chars": (self.fit.get("document") or {}).get("filler_chars"),
                         "tokens_by_model": (self.fit.get("document") or {}).get("tokens_by_model")},
            "fit": {k: self.fit.get(k) for k in ("common_ctx", "unfit", "geometry",
                                                 "geometry_warnings", "document")},
            "arms": arms,
            "samples": len(samples),
            "deadline_reached": bool(unrun),
            "incomplete_arms": list(incomplete or []),
            "unrun_arms": [{"model": m, "kv": kv} for m, kv in (unrun or [])],
            "timing": {
                "run_started": getattr(self, "run_started", None),
                "run_finished": getattr(self, "run_finished", None),
                "arms_wall_s": int(round(sum((a.get("wall_s") or 0)
                                             for a in arms.values()))),
                "fit_wall_s": (self.fit or {}).get("wall_s_total") or (self.fit or {}).get("wall_s"),
            },
            "tasks": {"R": [t["id"] for t in self.r_tasks],
                      "C": [t["id"] for t in self.c_tasks],
                      "L": [t["id"] for t in self.l_tasks]},
            }


def load_fit(out_dir, ctx_override=None):
    path = os.path.join(out_dir, "fit.json")
    if not os.path.exists(path):
        return {}, ctx_override
    fit = json.load(open(path))
    return fit, ctx_override or fit.get("common_ctx")


def main(argv=None):
    ap = argparse.ArgumentParser(description="run the KV-quant x long-context matrix")
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--deadline-hours", type=float, default=22.0)
    ap.add_argument("--models", default=None, help="comma-separated model keys")
    ap.add_argument("--kvs", default=None, help="comma-separated KV types")
    ap.add_argument("--reps", type=int, default=config.REPS)
    ap.add_argument("--code-reps", type=int, default=1)
    ap.add_argument("--code-temp", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true",
                    help="1 model x q4_0, -c 8192, 1 rep, tiny report")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ctx", type=int, default=None, help="override the fitted context")
    a = ap.parse_args(argv)

    stamp = a.stamp or time.strftime("%Y-%m-%dT%H:%M")
    out_dir = os.path.join(config.ROOT, "results", "smoke" if a.smoke else stamp)
    os.makedirs(out_dir, exist_ok=True)
    fit, ctx = load_fit(out_dir, a.ctx or (16384 if a.smoke else None))

    models = a.models.split(",") if a.models else None
    kvs = a.kvs.split(",") if a.kvs else None
    if a.smoke:
        models = models or ["qwen38-27b"]
        kvs = kvs or ["q4_0"]
        ctx = ctx or 16384
    if ctx is None:
        raise SystemExit("no fit.json and no --ctx: run `python3 -m kvtest.fit` first")

    knobs_map = {}
    for m in models or config.MODELS:
        entry = (fit.get("models") or {}).get(m) or {}
        knobs_map[m] = entry.get("knobs") or config.MODELS[m]["knobs"]

    doc_fit = (fit.get("document") or {})
    filler = doc_fit.get("filler_chars")
    include_code = True
    if a.smoke:
        # A tiny but structurally complete document: every planted record is there, the package
        # listing is not, so the code tier scores zero but the pipeline is exercised end to end.
        filler, include_code = 8000, False
    doc, facts = longctx.build_document(filler_chars=filler, include_code=include_code)
    if not a.smoke and doc_fit.get("sha256") and longctx.sha256(doc) != doc_fit["sha256"]:
        log("WARNING: rebuilt document does not match fit.json sha256 (filler_chars drift)")

    runner = Runner(stamp=("smoke" if a.smoke else stamp), ctx=ctx, knobs_map=knobs_map,
                    doc=doc, facts=facts, reps=a.reps, code_reps=a.code_reps,
                    models=models, kvs=kvs, deadline_hours=a.deadline_hours,
                    resume=a.resume, code_temp=a.code_temp, out_dir=out_dir)
    runner.fit = fit
    raw = runner.run()
    return raw


if __name__ == "__main__":
    main()
