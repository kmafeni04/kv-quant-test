"""Orchestrator: one rung per model, f16 + q4_0 arms, six workflows, fixed tranches.

A cell is (model, kv, workflow list). Tranches are fixed in advance, so a deadline cut lands on the
least informative remaining cell. Every instance checkpoints into the arm file, so a cut night
resumes instead of restarting.
"""
import argparse
import json
import os
import statistics
import time

from . import (client, compaction, config, kld, ledger, longctx, loops,
               search, server as srv, stats, tasks)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def plan_cells(tranches, models=None, kvs=None, workflows=None):
    """Tranche order -> [(model, kv, workflows)]. T3+T4 for one model merge into one arm."""
    cells, order = {}, []
    for t in config.TRANCHES:
        if tranches and t["id"] not in tranches:
            continue
        if models and t["model"] not in models:
            continue
        for kv in (kvs or config.KV_ARMS):
            key = (t["model"], kv)
            if key not in cells:
                cells[key] = []
                order.append(key)
            for wf in t["workflows"]:
                if (workflows and wf not in workflows) or wf in cells[key]:
                    continue
                cells[key].append(wf)
    return [(m, kv, cells[(m, kv)]) for m, kv in order if cells[(m, kv)]]


class Runner:
    def __init__(self, stamp, doc, facts, fit, models, kvs, workflows, instances,
                 tranches=None, ctx_override=None, deadline_hours=24.0, resume=False,
                 out_dir=None, compaction_variant="forced", deep_ppl_ctx=None, w1_deep=True):
        self.stamp = stamp
        self.out_dir = out_dir or os.path.join(config.ROOT, "results", stamp)
        self.arms_dir = os.path.join(self.out_dir, "arms")
        self.logs_dir = os.path.join(self.out_dir, "logs")
        self.scratch = os.path.join(self.out_dir, "scratch")
        for d in (self.arms_dir, self.logs_dir, self.scratch):
            os.makedirs(d, exist_ok=True)
        self.doc, self.facts, self.fit = doc, facts, fit
        self.doc_sha = longctx.sha256(doc)
        self.ctx_override = ctx_override
        self.models, self.kvs = models, kvs
        self.workflows, self.instances = workflows, instances
        self.compaction_variant = compaction_variant
        self.deep_ppl_ctx = deep_ppl_ctx
        self.w1_deep = w1_deep
        self.deadline = time.time() + deadline_hours * 3600 if deadline_hours else None
        self.resume = resume
        self.cells = plan_cells(tranches, models, kvs, workflows)
        self.samples_path = os.path.join(self.out_dir, "samples.jsonl")
        self.arm_durations = []
        self.run_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.run_finished = None
        self.cut_cells = []
        self.unrun_cells = []
        self.w2_stop = None
        self.hardware = None

    # ------------------------------------------------------------ helpers
    def rung(self, model):
        if self.ctx_override:
            return self.ctx_override
        return ((self.fit.get("models") or {}).get(model) or {}).get("ctx")

    def knobs(self, model):
        return ((self.fit.get("models") or {}).get(model) or {}).get("knobs") \
            or config.MODELS[model]["knobs"]

    def chat(self, base, messages, max_tokens=None, timeout=7200, tools=None):
        return client.chat(base, messages, max_tokens=max_tokens or config.MAX_OUTPUT_TOKENS,
                           timeout=timeout, tools=tools, **config.SAMPLING)

    def tool_chat(self, base):
        def _chat(messages, max_tokens=None, tools=None):
            return self.chat(base, messages, max_tokens=max_tokens, tools=tools)
        return _chat

    def chat_stream(self, base, messages, max_tokens=None, timeout=7200, loop_guard=None):
        return client.chat_stream(base, messages,
                                  max_tokens=max_tokens or config.MAX_OUTPUT_TOKENS,
                                  timeout=timeout,
                                  loop_guard=loop_guard or self._loop_guard,
                                  **config.SAMPLING)

    def _loop_guard(self, reasoning, content, tokens):
        for part, text in (("reasoning_content", reasoning), ("content", content)):
            m = loops.metrics(text)
            flagged, why = loops.verdict(m)
            if flagged:
                return {"reason": why, "stream": part, "span": m.get("span"),
                        "token_pos": tokens}
        return None

    def _halted_by(self, r, ctx, max_tokens):
        if r.get("error"):
            return "context_full" if "context" in r["error"].lower() else "error"
        used = ((r.get("usage") or {}).get("prompt_tokens") or 0) + \
               ((r.get("usage") or {}).get("completion_tokens") or 0)
        if ctx and used >= ctx:
            return "context_full"
        if (r.get("usage") or {}).get("completion_tokens") and \
                (r.get("usage") or {}).get("completion_tokens") >= max_tokens:
            return "ceiling"
        return "completed"

    def _loops(self, content, reasoning, out_tokens):
        out = {}
        for part, text in (("content", content or ""), ("reasoning_content", reasoning or "")):
            m = loops.metrics(text)
            flagged, why = loops.verdict(m, graded=False, got_answer=True,
                                         gen_tokens=out_tokens or 0)
            out[part] = dict(metrics=m, flagged=flagged, reason=why,
                             token_pos=m.get("flag_token_est") if flagged else None)
        return out

    def _single(self, base, prompt, workflow, instance, ctx):
        r = self.chat_stream(base, [{"role": "user", "content": self.doc + "\n\n" + prompt}])
        content = r.get("content") or ""
        reasoning = r.get("reasoning_content") or ""
        usage = r.get("usage") or {}
        rec = dict(workflow=workflow, instance=instance, error=r.get("error"),
                   content=content, reasoning_content=reasoning,
                   finish_reason=r.get("finish_reason"),
                   prompt_tokens=usage.get("prompt_tokens"),
                   out_tokens=usage.get("completion_tokens"),
                   prefill_tps=r.get("prefill_tps"), decode_tps=r.get("decode_tps"),
                   cache_n=(r.get("timings") or {}).get("cache_n"),
                   thinking_tokens=r.get("thinking_tokens"),
                   thinking_share=r.get("thinking_share"),
                   first_visible_s=r.get("first_visible_s"),
                   window_full=self._halted_by(r, ctx, config.MAX_OUTPUT_TOKENS) == "context_full",
                   loop_halt=r.get("loop_halt"),
                   wall_s=r.get("wall_s"))
        rec["loops"] = self._loops(content, reasoning, usage.get("completion_tokens"))
        detected = [v for v in rec["loops"].values() if v["flagged"]]
        if r.get("loop_halt"):
            rec["halted_by"] = "loop"
            rec["halt_token_pos"] = r["loop_halt"]["token_pos"]
            rec["halt_reason"] = (f"{r['loop_halt'].get('stream')}: "
                                  f"{r['loop_halt'].get('reason')}")
        elif detected:
            rec["halted_by"] = "loop"
            rec["halt_token_pos"] = detected[0].get("token_pos")
            rec["halt_reason"] = detected[0].get("reason")
        else:
            rec["halted_by"] = self._halted_by(r, ctx, config.MAX_OUTPUT_TOKENS)
            rec["halt_token_pos"] = None
        if workflow == "W2":
            rec["grade"] = tasks.grade_w2(self.facts, content, reasoning)
        elif workflow == "W6":
            rec["grade"] = tasks.grade_w6(content)
        return rec

    def _request_summary(self, rec):
        return {k: rec.get(k) for k in ("workflow", "instance", "error", "finish_reason",
                                        "prompt_tokens", "out_tokens", "prefill_tps",
                                        "decode_tps", "cache_n", "wall_s", "halted_by",
                                        "halt_token_pos", "halt_reason", "window_full",
                                        "thinking_share")}

    # ------------------------------------------------------------ one arm
    def run_arm(self, model, kv, workflows):
        name = f"{model}__{kv}"
        path = os.path.join(self.arms_dir, name + ".json")
        ctx = self.rung(model)
        state = dict(model=model, kv=kv, ctx=ctx, knobs=self.knobs(model),
                     document_sha256=self.doc_sha,
                     started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     finished=None, done=[], workflows={}, errors=[], cut=None,
                     sampler_resolved=None, kld_baseline=None,
                     declared_sampling=(self.fit.get("declared_sampling") or {}).get(model))
        if self.resume and os.path.exists(path):
            prev = json.load(open(path))
            if (prev.get("finished") and set(workflows) <= set(prev.get("workflows") or {})
                    and self._arm_done(prev, workflows)):
                log(f"arm {name}: already finished, skipping")
                return prev
            if prev.get("document_sha256") == self.doc_sha:
                state.update(done=prev.get("done") or [], workflows=prev.get("workflows") or {},
                             errors=prev.get("errors") or [], started=prev.get("started"),
                             sampler_resolved=prev.get("sampler_resolved"),
                             kld_baseline=prev.get("kld_baseline"))
                log(f"arm {name}: resuming with {len(state['done'])} instances done")

        def progress(key):
            state["done"].append(key)
            _write(path, state)

        def save():
            _write(path, state)

        t0 = time.time()
        s = None
        try:
            if "W1" in workflows:
                self._run_w1(model, kv, ctx, state)
                save()
            tcp_workflows = [w for w in workflows if w != "W1"]
            if tcp_workflows:
                s = srv.Server(model, kv, ctx, knobs=self.knobs(model), port=config.PORT,
                               log_path=os.path.join(self.logs_dir, name + ".log"))
                if not s.start():
                    state.update(failed="server did not start", log_tail=s.log_tail(2000))
                    save()
                    log(f"arm {name}: SERVER DID NOT START")
                    return state
                state["load_s"] = s.start_s
                log(f"arm {name}: up in {s.start_s}s, knobs={self.knobs(model)}, ctx={ctx}")
                for wf in tcp_workflows:
                    if self.deadline and time.time() + self._estimate_wf(model, wf) \
                            > self.deadline:
                        state["cut"] = f"deadline before {wf}"
                        self.cut_cells.append(dict(model=model, kv=kv, workflows=[wf],
                                                   reason="deadline before workflow"))
                        log(f"arm {name}: deadline before {wf}; remaining workflows not started")
                        break
                    self._run_workflow(s.base, model, kv, ctx, wf, state, progress, save)
                    state["sampler_resolved"] = s.sampler_resolved()
                    if wf == "W2" and kv == "f16":
                        self._audit_w2(state, model)
                    save()
        except TimeoutError as e:
            state["cut"] = str(e)
            self.cut_cells.append({"model": model, "kv": kv, "workflows": workflows,
                                   "reason": str(e)})
            log(f"arm {name}: cut by deadline")
        except KeyboardInterrupt:
            state["cut"] = "interrupted"
            if s:
                s.stop()
            raise
        except Exception as e:                                     # noqa: BLE001
            state["cut"] = f"{type(e).__name__}: {e}"[:300]
            state["errors"].append(state["cut"])
            log(f"arm {name}: ABORTED {state['cut']}")
        finally:
            if s:
                s.stop()
        state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["wall_s"] = round(time.time() - t0, 1)
        state["vram_peak_mib"] = s.vram.peak if s else None
        save()
        self.arm_durations.append(time.time() - t0)
        log(f"arm {name}: done in {state['wall_s']}s")
        return state

    def _arm_done(self, prev, workflows):
        done = set(prev.get("done") or [])
        for wf in workflows:
            n = 1 if wf == "W1" else self.instances.get(
                wf, config.WORKFLOWS[wf]["instances"])
            if any(f"{wf}:{i}" not in done for i in range(n)):
                return False
        return True

    # ------------------------------------------------------------ workflows
    def _run_w1(self, model, kv, ctx, state):
        if "W1:0" in state["done"]:
            return
        deep_ctx = self.deep_ppl_ctx or config.DEEP_PPL_CTX or ctx
        s4, sd = kld.prepare_slices(self.doc, model, ctx, self.out_dir,
                                    deep_ctx=deep_ctx, deep=self.w1_deep)
        rec = kld.run_arm(model, kv, ctx, s4, sd, self.logs_dir,
                          deep_ctx=deep_ctx, knobs=self.knobs(model),
                          deep=self.w1_deep)
        state["workflows"]["W1"] = rec
        state["kld_baseline"] = rec.get("baseline")
        state["done"].append("W1:0")
        _append(self.samples_path, dict(stamp=self.stamp, model=model, kv=kv, ctx=ctx,
                                        workflow="W1", instance=0,
                                        grade=rec.get("kld"), detail=rec,
                                        halted_by="completed", error=rec.get("error")))
        log(f"  {model}__{kv} W1 KLD={rec.get('kld', {}).get('mean_kld')} "
            f"deep_ppl={rec.get('deep_ppl')}")

    def _run_workflow(self, base, model, kv, ctx, wf, state, progress, save):
        n = self.instances.get(wf, config.WORKFLOWS[wf]["instances"])
        for i in range(n):
            key = f"{wf}:{i}"
            if key in state["done"]:
                continue
            t0 = time.time()
            if wf == "W2":
                rec = self._single(base, tasks.w2_prompt(self.facts), "W2", i, ctx)
                rec["score"] = rec["grade"]["score"]
            elif wf == "W3":
                tenant = self.facts["tenants"][i % len(self.facts["tenants"])]
                if self.compaction_variant == "model":
                    rec = compaction.run_model_driven(
                        lambda messages, max_tokens=None:
                        self.chat_stream(base, messages, max_tokens),
                        self.facts, self.doc, tenant, i, ctx)
                else:
                    rec = compaction.run(
                        lambda messages, max_tokens=None:
                        self.chat_stream(base, messages, max_tokens),
                        self.facts, self.doc, tenant, i, ctx)
                rec["score"] = rec["primary"]
            elif wf == "W4":
                tenant = self.facts["tenants"][i % len(self.facts["tenants"])]
                odir = os.path.join(self.scratch, f"{model}__{kv}-W4-i{i}")
                rec = ledger.run(self.tool_chat(base), self.facts, self.doc,
                                 tenant, odir, i,
                                 guard=config.WORKFLOWS["W4"]["guard"], ctx=ctx)
                rec["score"] = 1 if rec["grade"]["state_ok"] else 0
            elif wf == "W5":
                rec = search.run(self.tool_chat(base), self.facts, i,
                                 guard=config.WORKFLOWS["W5"]["guard"], ctx=ctx)
                rec["score"] = rec["grade"]["keys_correct"]
            elif wf == "W6":
                rec = self._single(base, tasks.w6_prompt(self.facts), "W6", i, ctx)
                rec["score"] = rec["grade"]["score"]
            else:
                continue
            rec["wall_s"] = round(time.time() - t0, 1)
            transcript = rec.get("transcript")
            rec = {k: v for k, v in rec.items() if k not in ("transcript",)}
            state["workflows"].setdefault(wf, {"instances": []})["instances"].append(rec)
            if transcript:
                state["workflows"][wf].setdefault("transcripts", {})[str(i)] = transcript
            _append(self.samples_path, dict(stamp=self.stamp, model=model, kv=kv, ctx=ctx,
                                            **self._request_summary(rec), score=rec.get("score"),
                                            grade=rec.get("grade"), detail={
                                                k: rec.get(k) for k in
                                                ("brief_grade", "phase_b_grade", "compactions",
                                                 "phase_a", "phase_b", "calls", "reads",
                                                 "lines_read", "tools_used")}))
            progress(key)
            log(f"  {model}__{kv} {wf} i{i} score={rec.get('score')} "
                f"halt={rec.get('halted_by')} wall={rec['wall_s']}s")

    def _audit_w2(self, state, model):
        """Stop rule 1: an f16 arm that loses a key means the task, not the cache, is wrong."""
        inst = (state.get("workflows") or {}).get("W2", {}).get("instances") or []
        scored = [i for i in inst if (i.get("grade") or {}).get("answer") is not None]
        misses = [(i.get("instance"), k.get("name"), k.get("status"))
                  for i in scored for k in (i.get("grade") or {}).get("keys") or []
                  if k.get("status") != "hit"]
        state.setdefault("flags", {})
        if misses:
            state["flags"]["w2_f16_miss"] = misses
            self.w2_stop = (f"{model} f16 lost {len(misses)} key(s) on W2 "
                            f"(first: instance {misses[0][0]} {misses[0][1]} {misses[0][2]})")
            log(f"STOP RULE: {self.w2_stop}; fix the document or the prompt before other arms")
        elif inst and not scored:
            state["flags"]["w2_f16_unscored"] = len(inst)
            log(f"WARNING: {model} f16 produced no scored W2 instance ({len(inst)} halted); "
                "this is a halt counter, not a lost key, but the probe has no f16 data yet")

    # ------------------------------------------------------------ matrix
    def _estimate_wf(self, model, wf):
        entry = (self.fit.get("models") or {}).get(model) or {}
        prefill = entry.get("prefill_tps") or 200
        decode = entry.get("decode_tps") or 15
        doc_tokens = ((self.fit.get("document") or {}).get("tokens_by_model") or {}).get(model) \
            or 94000
        gen = {"W1": 0, "W2": 300, "W3": 6000, "W4": 18000, "W5": 8000, "W6": 12000}
        if wf == "W1":
            return 900
        n = self.instances.get(wf, config.WORKFLOWS[wf]["instances"])
        prefix = 0 if wf == "W4" else doc_tokens
        return n * (prefix / prefill + 45 + gen.get(wf, 5000) / decode)

    def run(self):
        for idx, (model, kv, workflows) in enumerate(self.cells):
            if self.w2_stop:
                log(f"W2 stop rule holds: {self.w2_stop}; not starting the remaining cells")
                self.unrun_cells = [dict(model=m, kv=k, workflows=w, reason="W2 stop rule")
                                    for (m, k, w) in self.cells[idx:]]
                break
            if self.deadline:
                # do not pay a server load for a cell whose first workflow cannot fit
                est = max([statistics.median(self.arm_durations or [0])]
                          + [120 + self._estimate_wf(model, workflows[0])])
                if time.time() + est > self.deadline:
                    log(f"deadline governor: next cell {model}/{kv} ({','.join(workflows)}) "
                        f"needs ~{est / 3600:.2f}h and does not fit; not started")
                    self.unrun_cells = [dict(model=m, kv=k, workflows=w)
                                        for (m, k, w) in self.cells[idx:]]
                    self.cut_cells.append(dict(model=model, kv=kv, workflows=workflows,
                                               reason="deadline governor"))
                    break
            if not self.rung(model):
                log(f"cell {model}/{kv}: no fitted rung, skipped")
                self.unrun_cells.append(dict(model=model, kv=kv, workflows=workflows,
                                             reason="unfit"))
                continue
            if not os.path.exists(config.MODELS[model]["path"]):
                log(f"cell {model}/{kv}: model file missing, skipped")
                self.unrun_cells.append(dict(model=model, kv=kv, workflows=workflows,
                                             reason="model file missing"))
                continue
            try:
                self.run_arm(model, kv, workflows)
            except KeyboardInterrupt:
                raise
            except Exception as e:                                 # noqa: BLE001
                log(f"cell {model}/{kv}: crashed ({type(e).__name__}: {e}); continuing")
        self.run_finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        raw = self.build_raw()
        _write(os.path.join(self.out_dir, "raw.json"), raw)
        log("raw.json written")
        return raw

    # ------------------------------------------------------------ raw.json
    def build_raw(self):
        samples = []
        if os.path.exists(self.samples_path):
            with open(self.samples_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if s.get("stamp") == self.stamp:
                        samples.append(s)
        # re-running an instance (a resumed stamp, a deliberate retry) replaces its line
        dedup = {}
        for s in samples:
            dedup[(s.get("model"), s.get("kv"), s.get("workflow"), s.get("instance"))] = s
        samples = list(dedup.values())
        arms = {}
        for fn in sorted(os.listdir(self.arms_dir)):
            if not fn.endswith(".json"):
                continue
            a = json.load(open(os.path.join(self.arms_dir, fn)))
            arms[f"{a['model']}__{a['kv']}"] = a
        manifest = config.build_manifest()
        manifest["document_sha256"] = self.doc_sha
        manifest["rungs"] = {m: self.rung(m) for m in {c[0] for c in self.cells}}
        manifest["sampler_resolved"] = {k: (v.get("sampler_resolved") or {})
                                        for k, v in arms.items()}
        manifest["declared_sampling"] = self.fit.get("declared_sampling") or {}
        manifest["kld_baselines"] = {k: v.get("kld_baseline") for k, v in arms.items()}
        manifest["w1_slices"] = {k: dict(
            slice4k=(v.get("workflows", {}).get("W1") or {}).get("slice4k"),
            deep=(v.get("workflows", {}).get("W1") or {}).get("deep_slice"),
            deep_ctx=(v.get("workflows", {}).get("W1") or {}).get("deep_ctx"))
            for k, v in arms.items()}
        manifest["instances"] = {wf: self.instances.get(wf, config.WORKFLOWS[wf]["instances"])
                                 for wf in self.workflows}
        manifest["compaction_variant"] = self.compaction_variant
        manifest["host_runtime"] = self.hardware or _host()
        complete = all(not a.get("cut") and a.get("finished") for a in arms.values())
        return dict(
            manifest=manifest, stamp=self.stamp, context={m: self.rung(m) for m in
                                                          {c[0] for c in self.cells}},
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            document={"sha256": self.doc_sha, "chars": len(self.doc),
                      "filler_chars": (self.fit.get("document") or {}).get("filler_chars"),
                      "tokens_by_model": (self.fit.get("document") or {}).get("tokens_by_model")},
            fit={k: self.fit.get(k) for k in ("ladder", "context_floor", "unfit", "document",
                                              "declared_sampling", "geometry_warnings")},
            arms=arms, samples=len(samples),
            cells=[dict(model=m, kv=k, workflows=w) for m, k, w in self.cells],
            cut_cells=self.cut_cells, unrun_cells=self.unrun_cells,
            complete=complete,
            timing=dict(run_started=self.run_started, run_finished=self.run_finished,
                        arms_wall_s=int(round(sum((a.get("wall_s") or 0)
                                                  for a in arms.values())))),
        )


def _host():
    try:
        from . import hostinfo
        return hostinfo.snapshot()
    except Exception:                                              # noqa: BLE001
        return {}


def load_document(fit, smoke=False, filler_override=None):
    doc_fit = fit.get("document") or {}
    filler = filler_override if filler_override is not None else doc_fit.get("filler_chars")
    if smoke and filler_override is None:
        filler = 12000
    doc, facts = longctx.build_document(filler_chars=filler)
    if not smoke and doc_fit.get("sha256") and longctx.sha256(doc) != doc_fit["sha256"]:
        log("WARNING: rebuilt document does not match fit.json sha256")
    return doc, facts


def main(argv=None):
    ap = argparse.ArgumentParser(description="run the full-context KV-quant study")
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--models", default=None, help="comma-separated model keys")
    ap.add_argument("--kvs", default=None, help="comma-separated KV types")
    ap.add_argument("--workflows", default=None, help="comma-separated W ids")
    ap.add_argument("--instances", type=int, default=None, help="override instances per workflow")
    ap.add_argument("--tranches", default=None, help="comma-separated tranche ids")
    ap.add_argument("--deadline-hours", type=float, default=24.0)
    ap.add_argument("--ctx", type=int, default=None, help="override every model rung")
    ap.add_argument("--deep-ppl-ctx", type=int, default=None)
    ap.add_argument("--no-deep-ppl", action="store_true",
                    help="skip W1's depth pass (the pilot only needs the KLD pass)")
    ap.add_argument("--compaction", choices=["forced", "model"], default="forced")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny document, one instance, no W1; for the harness self-test")
    ap.add_argument("--pilot", action="store_true",
                    help="micro-pilot: one model, f16, one 20-key sample, one compaction pair")
    a = ap.parse_args(argv)

    stamp = a.stamp or time.strftime("%Y-%m-%dT%H:%M")
    out_dir = os.path.join(config.ROOT, "results", "smoke" if a.smoke else stamp)
    os.makedirs(out_dir, exist_ok=True)
    fit_path = os.path.join(out_dir, "fit.json")
    fit = json.load(open(fit_path)) if os.path.exists(fit_path) else {}
    if a.smoke and fit:
        fit = dict(fit, document={})      # the smoke document is not the fitted one

    models = a.models.split(",") if a.models else config.STUDY_MODELS
    kvs = a.kvs.split(",") if a.kvs else config.KV_ARMS
    workflows = a.workflows.split(",") if a.workflows else None
    tranches = a.tranches.split(",") if a.tranches else None
    instances = {}
    if a.instances:
        instances = {wf: a.instances for wf in config.WORKFLOW_IDS}
    ctx = a.ctx
    if a.smoke:
        models = models or ["gemma4-26b-a4b"]
        kvs = kvs or ["q4_0"]
        workflows = workflows or ["W2", "W3", "W4", "W5", "W6"]
        instances = {wf: 1 for wf in workflows}
        ctx = ctx or 32768
        tranches = None
    if a.pilot:
        models, kvs = ["gemma4-26b-a4b"], ["f16"]
        workflows = workflows or ["W1", "W2", "W3"]
        instances = {wf: 1 for wf in workflows}
        tranches = None
    if ctx is None and not fit:
        raise SystemExit("no fit.json and no --ctx: run `python3 -m kvtest.fit` first")
    doc, facts = load_document(fit, smoke=a.smoke,
                               filler_override=None if not a.smoke else 12000)

    runner = Runner(stamp=("smoke" if a.smoke else stamp), doc=doc, facts=facts, fit=fit,
                    models=models, kvs=kvs, workflows=workflows or config.WORKFLOW_IDS,
                    instances=instances, tranches=tranches, ctx_override=ctx,
                    deadline_hours=a.deadline_hours, resume=a.resume, out_dir=out_dir,
                    compaction_variant=a.compaction, deep_ppl_ctx=a.deep_ppl_ctx,
                    w1_deep=not (a.no_deep_ppl or a.pilot))
    runner.hardware = _host()
    _write(os.path.join(out_dir, "hardware.json"), runner.hardware)
    raw = runner.run()
    return raw


if __name__ == "__main__":
    main()
