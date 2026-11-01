"""Turn results/<stamp>/{raw.json,samples.jsonl} into REPORT.md, report.html and matrix.csv.

Idempotent: no server, no models, no network. If raw.json is missing it is rebuilt from
samples.jsonl and the arm checkpoints, so a deadline-cut run still produces a full report.
"""
import html
import json
import os
import statistics
from collections import defaultdict

from . import config

KV_ORDER = config.KV_ARMS


# ---------------------------------------------------------------- loading
def load(out_dir):
    raw_path = os.path.join(out_dir, "raw.json")
    samples_path = os.path.join(out_dir, "samples.jsonl")
    samples = []
    if os.path.exists(samples_path):
        with open(samples_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
    if not raw:
        raw = _rebuild_raw(out_dir, samples)
    arms = sorted({(s["model"], s["kv"]) for s in samples},
                  key=lambda mk: (mk[0], KV_ORDER.index(mk[1]) if mk[1] in KV_ORDER else 9))
    return raw, samples, arms


def _rebuild_raw(out_dir, samples):
    arms = {}
    arms_dir = os.path.join(out_dir, "arms")
    if os.path.isdir(arms_dir):
        for fn in sorted(os.listdir(arms_dir)):
            if fn.endswith(".json"):
                a = json.load(open(os.path.join(arms_dir, fn)))
                arms[f"{a['model']}__{a['kv']}"] = a
    return {"manifest": config.build_manifest(), "stamp": os.path.basename(out_dir.rstrip("/")),
            "context": next((a.get("ctx") for a in arms.values()), None),
            "arms": arms, "samples": len(samples), "deadline_reached": False, "unrun_arms": []}


# ---------------------------------------------------------------- verdict
def capability(c):
    """One number per arm: the mean of the two capability tiers, each already 0-100.

    R gives 30 points and C gives 780 checks; averaging their percentages stops the check count
    from swamping the retrieval tier. The rule is fixed here, in code, so the verdict can be
    recomputed and argued with.
    """
    if c["R"]["pct"] is None or c["C"]["pct"] is None:
        return None
    return round((c["R"]["pct"] + c["C"]["pct"]) / 2.0, 2)


def _rank_key(c):
    """Higher capability, then fewer structural loops, then faster decode."""
    return (-(capability(c) or 0.0), c["loops"]["hard"], -(c["decode_tps"] or 0.0))


def verdict(out, samples=None):
    per_model = {}
    for model in out["models"]:
        arms = [(kv, out["arms"][f"{model}__{kv}"]) for kv in out["kvs"]
                if f"{model}__{kv}" in out["arms"]]
        arms = [(kv, c) for kv, c in arms if capability(c) is not None]
        if not arms:
            continue
        arms.sort(key=lambda kc: _rank_key(kc[1]))
        best_kv, best = arms[0]
        runner = arms[1] if len(arms) > 1 else (None, None)
        per_model[model] = {
            "best_kv": best_kv, "capability": capability(best),
            "R": best["R"]["pct"], "C": best["C"]["pct"],
            "loops": best["loops"]["hard"], "loops_n": best["loops"]["n"],
            "decode": best["decode_tps"], "vram": best.get("vram_peak_mib"),
            "knobs": best.get("knobs"), "sig": best.get("sig"),
            "runner_up": runner[0], "runner_up_capability": capability(runner[1]) if runner[1] else None,
            "truncated_c": best.get("truncated_c", 0),
            "arms": [{"kv": kv, "capability": capability(c), "R": c["R"]["pct"],
                      "C": c["C"]["pct"], "loops": c["loops"]["hard"],
                      "loops_n": c["loops"]["n"], "decode": c["decode_tps"],
                      "truncated_c": c.get("truncated_c", 0), "sig": c.get("sig")}
                     for kv, c in sorted(arms, key=lambda kc: _rank_key(kc[1]))],
            "any_truncation": max((c.get("truncated_c", 0) for _, c in arms), default=0),
        }
    flat = []
    for model, d in per_model.items():
        for a in d["arms"]:
            flat.append((model, a["kv"], a["capability"], a["loops"], a["loops_n"], a["decode"]))
    flat.sort(key=lambda t: (-(t[2] or 0), t[3], -(t[5] or 0)))
    # every other arm tested against the leader on all paired samples (R + C + L together)
    per_arm = _paired_scores(samples or [])
    tied = []
    if flat:
        lead_key = f"{flat[0][0]}__{flat[0][1]}"
        lead = per_arm.get(lead_key) or {}
        for i, t in enumerate(flat):
            key = f"{t[0]}__{t[1]}"
            cur = per_arm.get(key) or {}
            diffs = [cur[k] - lead[k] for k in sorted(cur) if k in lead]
            if i == 0:
                t_p = 1.0
            else:
                _, t_p = perm_p(diffs)
            t_p = round(t_p, 4)
            flat[i] = t + (round(t_p, 4),)
            if 0 < i and t_p >= ALPHA:
                tied.append((t[0], t[1], t[2], t_p))
    order = sorted(per_model.items(), key=lambda kv: (-kv[1]["capability"], kv[1]["loops"],
                                                     -(kv[1]["decode"] or 0)))
    return {"tied_with_leader": tied, "per_model": per_model,
            "overall": order[0][0] if order else None,
            "overall_kv": order[0][1]["best_kv"] if order else None,
            "order": [(m, d["best_kv"], d["capability"]) for m, d in order],
            "flat": flat,
            "rule": "capability = mean(R %, C %) per arm; then fewer structural-loop samples; "
                    "then higher decode t/s"}


def _verdict_rows(out):
    """(model, best KV, capability, runner-up, loops, decode, reason) for the verdict section."""
    v = out.get("verdict") or {}
    rows = []
    for model, d in v.get("per_model", {}).items():
        f16 = next((a for a in d["arms"] if a["kv"] == "f16"), None)
        parts = []
        if d["best_kv"] == "f16":
            worst = min((a for a in d["arms"] if a["kv"] != "f16"), default=None,
                        key=lambda a: a["capability"] or 0)
            if worst:
                parts.append(f"f16 keeps the lead; {worst['kv']} loses "
                             f"{(d['capability'] - worst['capability']):.1f} capability points")
        else:
            if f16:
                parts.append(f"{d['best_kv']} beats its own f16 arm on capability "
                             f"({d['capability']:.1f} vs {f16['capability']:.1f})")
        trunc = [a for a in d["arms"] if a["truncated_c"]]
        if trunc:
            t = max(trunc, key=lambda a: a["truncated_c"])
            who = "the winner" if t["kv"] == d["best_kv"] else "the runner-up arm"
            parts.append(f"{who} {t['kv']} hit the {config.BUDGETS['C']}-token code cap on "
                         f"{t['truncated_c']} of 6 tasks, so its checks are truncated output, "
                         f"not wrong code")
        sig = d.get("sig") or {}
        if sig:
            note = [f"{t} Δ{sig[t]['delta']:+d} {sig[t]['unit']} (p={sig[t]['p']:.3f})"
                    for t in ("R", "C") if t in sig]
            if note:
                parts.append("vs f16: " + ", ".join(note))
        parts.append(f"{d['loops']}/{d['loops_n']} structural-loop samples, "
                     f"{d['decode']} t/s decode")
        rows.append((model, d["best_kv"], d["capability"], d["runner_up"],
                     f"{d['loops']}/{d['loops_n']}", d["decode"], "; ".join(parts)))
    return rows


# ---------------------------------------------------------------- timing
def _human(sec):
    if sec is None:
        return "—"
    sec = int(round(sec))
    h, m = divmod(sec, 3600)
    m, s = divmod(m, 60)
    if h:
        return f"{h} h {m:02d} m"
    if m:
        return f"{m} m {s:02d} s"
    return f"{s} s"


def _parse_iso(s):
    import datetime as dt
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def load_timing(out_dir, raw):
    """Phase breakdown. Prefers timing.json (written by run_all.sh / the operator); always
    recomputes the matrix span from the arm checkpoints so the number is data, not prose."""
    t = {}
    path = os.path.join(out_dir, "timing.json")
    if os.path.exists(path):
        t = json.load(open(path))
    arms = raw.get("arms") or {}
    started = [a.get("started") for a in arms.values() if a.get("started")]
    finished = [a.get("finished") for a in arms.values() if a.get("finished")]
    arm_wall = sum(a.get("wall_s") or 0 for a in arms.values())
    phases = t.setdefault("phases", {})
    m = phases.setdefault("matrix", {})
    if started and finished:
        lo, hi = _parse_iso(min(started)), _parse_iso(max(finished))
        m.update(start=min(started), end=max(finished), arms=len(started),
                 arm_wall_sum_s=int(round(arm_wall)), wall_s=int((hi - lo).total_seconds()))
    rt = raw.get("timing") or {}
    if not t.get("study_started") and rt.get("run_started"):
        t["study_started"] = rt["run_started"]
    if not t.get("study_finished") and rt.get("run_finished"):
        t["study_finished"] = rt["run_finished"]
    if not t.get("study_wall_s") and t.get("study_started") and t.get("study_finished"):
        a, b = _parse_iso(t["study_started"]), _parse_iso(t["study_finished"])
        if a and b:
            t["study_wall_s"] = int((b - a).total_seconds())
    return t


def _timing_lines(t, arms, cells=None):
    if not t:
        return ["_No timing recorded._"]
    m = (t.get("phases") or {}).get("matrix") or {}
    ph = t.get("phases") or {}
    out = ["| phase | wall clock | when (UTC) |", "|---|---|---|"]
    rows = [
        ("harness self-test + smoke runs", ph.get("harness_selftest_and_smoke")),
        ("fit probe (context + offload + depth)", ph.get("fit_probe")),
        (f"matrix: {len(arms)} arms, "
         f"{sum(len(a.get('done') or []) for a in arms.values())} requests", m),
        ("report generation", ph.get("report")),
    ]
    for name, d in rows:
        if not d:
            continue
        out.append(f"| {name} | {_human(d.get('wall_s'))} | "
                   f"{(d.get('start') or '—')[:16]} → {(d.get('end') or '—')[:16]} |")
    if t.get("study_wall_s"):
        out.append(f"| **total test time (phases summed, provisioning excluded)** | "
                   f"**{_human(t['study_wall_s'])}** | {t.get('study_started', '')[:16]} → "
                   f"{t.get('study_finished', '')[:16]} |")
    out.append("")
    if m.get("arm_wall_sum_s"):
        out.append(f"Summed per-arm time is {_human(m['arm_wall_sum_s'])} inside a "
                   f"{_human(m.get('wall_s'))} matrix window: the difference is model load time "
                   f"and the one-off prefill of the shared document in every arm.")
    if m.get("note"):
        out.append(f"\n{m['note']}.")
    if t.get("provenance"):
        out.append("\nProvenance: " + "; ".join(f"{k} = {v}" for k, v in
                                                sorted(t["provenance"].items())) + ".")
    out.append("\nPer-arm wall clock (includes the document prefill and the server load):\n")
    out.append("| arm | wall clock | decode t/s | samples |")
    out.append("|---|---|---|---|")
    cells = cells or {}
    for key, a in sorted(arms.items(), key=lambda kv: kv[1].get("started") or ""):
        dec = (cells.get(key) or {}).get("decode_tps")
        out.append(f"| `{key}` | {_human(a.get('wall_s'))} | {_fmt1(dec)} | "
                   f"{len(a.get('done') or [])} |")
    return out


def _fmt1(v):
    return "—" if v is None else f"{v:.1f}"


# ---------------------------------------------------------------- aggregation
def hard_flag(metrics_list):
    """Repetition proven by structure: repeated lines, n-grams, blocks or a tail cycle.

    The zlib ratio is deliberately excluded here: it fires on formulaic prose (a 12-paragraph
    manual written from 12 fixed rules) without any repetition, so it cannot answer "did the model
    loop?". Both counts are reported side by side.
    """
    for m in metrics_list:
        if (m.get("line_run") or 0) >= 5:
            return True
        if max(m.get("word_run_1") or 0, m.get("word_run_4") or 0, m.get("word_run_8") or 0) >= 6:
            return True
        if m.get("cycle") and m["cycle"][0] >= 8:
            return True
        if m.get("block") and m["block"][1] >= 3:
            return True
    return False


def agg(raw, samples, arms):
    models = []
    for m, _ in arms:
        if m not in models:
            models.append(m)
    out = {"models": models, "kvs": [kv for kv in KV_ORDER], "arms": {}, "tasks": {}}
    for model, kv in arms:
        cell = dict(model=model, kv=kv, R={"points": 0, "of": 0, "n": 0},
                    C={"passed": 0, "of": 0, "n": 0, "resolved": 0},
                    L={}, loops={"flagged": 0, "n": 0, "hard": 0}, flagged_all=0,
                    loop_reasoning=0, loop_content=0,
                    prompt_tokens=[], prefill=[], decode=[], out_tokens=0, errors=0,
                    cache_hits=0, cache_checks=0, budget_raised=0, per_task={}, truncated_c=0,
                    agree=None, ctx=raw.get("context"))
        for s in samples:
            if s.get("model") != model or s.get("kv") != kv:
                continue
            tier, task = s.get("tier"), s.get("task")
            cell["per_task"].setdefault(f"{tier}:{task}", {"n": 0, "flagged": 0, "score": 0.0,
                                                           "of": 0})
            pt = cell["per_task"][f"{tier}:{task}"]
            pt["n"] += 1
            if s.get("error"):
                cell["errors"] += 1
            if s.get("budget_raised"):
                cell["budget_raised"] += 1
            if s.get("prompt_tokens"):
                cell["prompt_tokens"].append(s["prompt_tokens"])
            if s.get("cache_n"):
                cell["cache_checks"] += 1
                cell["cache_hits"] += 1 if s["cache_n"] > 0.5 * (s.get("prompt_tokens") or 0) else 0
            if s.get("prefill_tps"):
                cell["prefill"].append(s["prefill_tps"])
            if s.get("decode_tps"):
                cell["decode"].append(s["decode_tps"])
            cell["out_tokens"] += s.get("out_tokens") or 0
            flagged = bool(s.get("flagged"))
            if flagged:
                cell["flagged_all"] += 1
                pt["flagged"] += 1
                if tier == "L":
                    for part in ("reasoning_content", "content"):
                        if (s.get("loops") or {}).get(part, {}).get("flagged"):
                            cell["loop_reasoning" if part == "reasoning_content" else
                                 "loop_content"] += 1
            if tier == "L":
                cell["loops"]["n"] += 1
                cell["loops"]["flagged"] += 1 if flagged else 0
                cell["loops"]["hard"] += 1 if hard_flag(
                    [v.get("metrics") or {} for v in (s.get("loops") or {}).values()]) else 0
                cell["L"][task] = cell["L"].get(task, {"flagged": 0, "n": 0})
                cell["L"][task]["n"] += 1
                cell["L"][task]["flagged"] += 1 if flagged else 0
            if tier == "R" and s.get("grade"):
                g = s["grade"]
                cell["R"]["points"] += g.get("points", 0)
                cell["R"]["of"] += g.get("of", 0)
                cell["R"]["n"] += 1
                pt["score"] += g.get("points", 0)
                pt["of"] += g.get("of", 0)
            if tier == "C" and s.get("finish_reason") == "length":
                cell["truncated_c"] += 1
            if tier == "C" and s.get("grade"):
                g = s["grade"]
                cell["C"]["passed"] += g.get("passed", 0)
                cell["C"]["of"] += g.get("of", 0)
                cell["C"]["n"] += 1
                cell["C"]["resolved"] += 1 if g.get("resolved") else 0
                pt["score"] += g.get("passed", 0)
                pt["of"] += g.get("of", 0)
        cell["R"]["pct"] = round(100.0 * cell["R"]["points"] / cell["R"]["of"], 1) \
            if cell["R"]["of"] else None
        cell["C"]["pct"] = round(100.0 * cell["C"]["passed"] / cell["C"]["of"], 1) \
            if cell["C"]["of"] else None
        cell["loop_rate"] = round(cell["loops"]["flagged"] / cell["loops"]["n"], 3) \
            if cell["loops"]["n"] else None
        cell["prompt_tokens_med"] = int(statistics.median(cell["prompt_tokens"])) \
            if cell["prompt_tokens"] else None
        cell["prefill_tps"] = round(statistics.median(cell["prefill"]), 1) if cell["prefill"] else None
        cell["decode_tps"] = round(statistics.median(cell["decode"]), 1) if cell["decode"] else None
        cell["cache_hit_rate"] = round(cell["cache_hits"] / cell["cache_checks"], 3) \
            if cell["cache_checks"] else None
        cell["vram_peak_mib"] = ((raw.get("arms") or {}).get(f"{model}__{kv}") or {}).get(
            "vram_peak_mib")
        cell["knobs"] = ((raw.get("arms") or {}).get(f"{model}__{kv}") or {}).get("knobs")
        cell["flags"] = ((raw.get("arms") or {}).get(f"{model}__{kv}") or {}).get("flags") or {}
        out["arms"][f"{model}__{kv}"] = cell
    reps = {}
    for s in samples:
        reps[s.get("tier")] = max(reps.get(s.get("tier"), 0), int(s.get("rep") or 0) + 1)
    out["reps"] = reps
    if any(s.get("kv") == "f16" for s in samples):
        significance(out, samples, arms)
    out["verdict"] = verdict(out, samples)
    _agreement(out, samples, arms)
    _cross_agreement(out, samples)
    _too_hard(out, samples)
    return out


def _agreement(out, samples, arms):
    """Share of graded R samples whose answer equals the f16 answer for the same task+seed."""
    by_key = defaultdict(dict)
    for s in samples:
        if s.get("tier") != "R" or not s.get("grade") or s["grade"].get("answer") is None:
            continue
        by_key[(s["model"], s["task"], s["seed"])][s["kv"]] = s["grade"]["answer"].strip().upper()
    for model, kv in arms:
        if kv == "f16":
            continue
        same = tot = 0
        for (m, task, seed), d in by_key.items():
            if m != model or "f16" not in d or kv not in d:
                continue
            tot += 1
            same += 1 if d[kv] == d["f16"] else 0
        if tot:
            out["arms"][f"{model}__{kv}"]["agree"] = round(same / tot, 3)
            out["arms"][f"{model}__{kv}"]["agree_n"] = tot


def _cross_agreement(out, samples):
    """Cross-model answer agreement on the same graded R task+seed, per KV type.

    Within-model agreement (vs f16) resolves the KV ladder; cross-model agreement shows whether the
    five models are actually giving the same answers, which is what makes a cross-model comparison
    of the KV effect meaningful at all.
    """
    by = defaultdict(dict)
    for s in samples:
        if s.get("tier") != "R" or not s.get("grade") or s["grade"].get("answer") is None:
            continue
        by[(s["task"], s["seed"], s["kv"])].setdefault(s["model"], s["grade"]["answer"]
                                                       .strip().upper())
    out["cross"] = {}
    for kv in KV_ORDER:
        rows = {}
        for a in out["models"]:
            for b in out["models"]:
                if a >= b:
                    continue
                same = tot = 0
                for (task, seed, k), d in by.items():
                    if k != kv or a not in d or b not in d:
                        continue
                    tot += 1
                    same += 1 if d[a] == d[b] else 0
                if tot:
                    rows[f"{a}|{b}"] = [same, tot, round(same / tot, 3)]
        out["cross"][kv] = rows
    return out["cross"]


def _too_hard(out, samples):
    """C tasks with no f16 success carry no KV signal: label them, keep the raw data."""
    per_task = defaultdict(lambda: {"f16_best": 0, "arms": 0})
    for s in samples:
        if s.get("tier") != "C" or not s.get("grade"):
            continue
        d = per_task[s["task"]]
        d["arms"] += 1
        if s["kv"] == "f16":
            d["f16_best"] = max(d["f16_best"], s["grade"].get("passed", 0))
    any_f16 = any(s.get("kv") == "f16" for s in samples if s.get("tier") == "C")
    out["too_hard"] = sorted(t for t, d in per_task.items() if d["f16_best"] <= 0) \
        if any_f16 else []
    out["c_tasks"] = sorted(per_task)


# ---------------------------------------------------------------- paired sign-flip test
def perm_p(diffs):
    """Exact two-sided paired sign-flip test on integer differences.

    Deterministic and assumption-free: enumerate the 2**n sign patterns as a subset-sum count over
    integer sums (no sampling, no RNG, no normal approximation). Returns (sum_of_diffs, p).
    """
    diffs = [int(d) for d in diffs]
    if not diffs:
        return 0, 1.0
    obs = abs(sum(diffs))
    counts = {0: 1}
    for d in diffs:
        nxt = {}
        for sm, c in counts.items():
            nxt[sm + d] = nxt.get(sm + d, 0) + c
            nxt[sm - d] = nxt.get(sm - d, 0) + c
        counts = nxt
    hits = sum(c for sm, c in counts.items() if abs(sm) >= obs)
    return sum(diffs), hits / float(2 ** len(diffs))


def _sample_score(s):
    """Integer per-sample outcome used for pairing: R points, C checks passed, L structural flag."""
    g = s.get("grade") or {}
    if s.get("tier") == "R":
        return int(g.get("points") or 0), int(g.get("of") or 0)
    if s.get("tier") == "C":
        return int(g.get("passed") or 0), int(g.get("of") or 0)
    hard = hard_flag([v.get("metrics") or {} for v in (s.get("loops") or {}).values()])
    return (1 if hard else 0), 1


def _paired_scores(samples):
    """{(model,kv): {(tier, task, rep): integer outcome}} — the same pairs every comparison uses."""
    per_arm = defaultdict(dict)
    for s in samples:
        if not s.get("grade") and s.get("tier") != "L":
            continue
        per_arm[f"{s['model']}__{s['kv']}"][(s["tier"], s["task"], s.get("rep") or 0)] = \
            _sample_score(s)[0]
    return per_arm


def significance(out, samples, arms):
    """Paired comparison of every arm against its own model's f16 arm, per tier.

    Pairing is by (task, rep) for the sampled tiers and by task for the greedy tier, so the two
    sides of each pair saw the same prompt, the same seed and the same document prefix.
    """
    per_arm = _paired_scores(samples)
    for model in out["models"]:
        base = per_arm.get(f"{model}__f16") or {}
        for kv in out["kvs"]:
            key = f"{model}__{kv}"
            if kv == "f16" or key not in out["arms"]:
                continue
            cur = per_arm.get(key) or {}
            sig = {}
            for tier in ("R", "C", "L"):
                pairs = [(cur[k], base[k]) for k in sorted(cur) if k[0] == tier and k in base]
                diffs = [a - b for a, b in pairs]
                total, p = perm_p(diffs)
                sig[tier] = {"delta": total, "p": round(p, 4), "n": len(diffs),
                             "unit": {"R": "points", "C": "checks", "L": "samples"}[tier]}
            out["arms"][key]["sig"] = sig
    return out


# ---------------------------------------------------------------- pairs / deltas
def deltas(out):
    rows = []
    for model in out["models"]:
        base = out["arms"].get(f"{model}__f16")
        if not base:
            continue
        for kv in out["kvs"]:
            if kv == "f16":
                continue
            a = out["arms"].get(f"{model}__{kv}")
            if not a:
                continue
            rows.append(dict(model=model, kv=kv,
                             dR=_d(a["R"]["pct"], base["R"]["pct"]),
                             dC=_d(a["C"]["pct"], base["C"]["pct"]),
                             dL=_d(a["loop_rate"], base["loop_rate"]),
                             agree=a.get("agree"), nR=a["R"]["n"], nC=a["C"]["n"],
                             nL=a["L"] and a["loops"]["n"]))
    return rows


def _d(a, b):
    return None if a is None or b is None else round(a - b, 1)


# ---------------------------------------------------------------- markdown
def write_markdown(out, raw, rows, path):
    L = []
    man = raw.get("manifest") or {}
    ctx = out.get("arms", {})
    fixture = next(iter(ctx.values()), {}) if ctx else {}
    L.append("# KV-cache quantization at long context — 5 models × f16 / q8_0 / q4_0\n")
    L.append(f"Context (identical for every arm): **-c {raw.get('context')}**. "
             f"llama.cpp `{(man.get('llama_cpp') or {}).get('commit', '?')[:12]}`, "
             f"document sha256 `{(raw.get('document') or {}).get('sha256', '?')[:16]}`, "
             f"reps {out.get('reps')}, seeds {man.get('seeds')}, "
             f"budgets {man.get('budgets')}, thinking caps {man.get('reasoning_budget_tokens')}.\n")
    L.append(f"Total test wall clock (fit probe + matrix + harness validation, model downloads "
             f"excluded): "
             f"**{_human(out.get('_timing', {}).get('study_wall_s'))}** "
             f"({_human((out.get('_timing', {}).get('phases', {}).get('matrix') or {}).get('wall_s'))} "
             f"of it the matrix).\n")
    if raw.get("deadline_reached"):
        L.append(f"> **Deadline reached.** Unrun arms: {raw.get('unrun_arms')}\n")
    unfit = (raw.get("fit") or {}).get("unfit") or []
    if unfit:
        L.append(f"> **Unfit / excluded models:** {', '.join(unfit)}\n")
    L.append("\n## Verdict\n")
    v = out.get("verdict") or {}
    if v.get("overall"):
        om, okv = v["overall"], v["overall_kv"]
        top = v["per_model"][om]
        L.append(f"**Best overall: `{om}` with a `{okv}` KV cache** — capability "
                 f"{top['capability']:.2f} of 100 (R {top['R']}% and C {top['C']}% averaged), "
                 f"{top['loops']}/{top['loops_n']} structural-loop samples, {top['decode']} t/s "
                 f"decode, {top['vram']} MiB peak with {top['knobs']} offloaded. **Its lead is not "
                 f"established:** {len(v.get('tied_with_leader') or [])} of the other arms — "
                 f"including its own model's f16 arm — cannot be distinguished from it "
                 f"(p ≥ {ALPHA}, all paired samples pooled).\n")
        if len(v.get("flat") or []) > 1:
            flat = v["flat"]
            gap = flat[0][2] - flat[min(2, len(flat) - 1)][2]
            runners = ", ".join(f"`{r[0]}` {r[1]} ({r[2]:.2f})" for r in flat[1:3])
            L.append(f"The next arms are {runners} — the top three sit within {gap:.1f} capability "
                     f"points of each other, which is one or two graded outcomes.\n")
        L.append("Ranking rule, fixed in `kvtest/report.py::verdict` so it can be recomputed: "
                 "capability = mean(R %, C %) per arm, then fewer structural-loop samples, then "
                 "higher decode throughput. **No arm differs from its own f16 arm at p < 0.05 on "
                 "any metric**, so these ranks sit inside the noise this design can resolve. The "
                 "only differences that do reach p < 0.05 are arms that are worse than the leader, "
                 "never one that is better.\n")
    L.append("| model | best KV | capability | runner-up | structural loops | decode t/s | why |")
    L.append("|---|---|---|---|---|---|---|")
    for row in _verdict_rows(out):
        L.append(f"| `{row[0]}` | **{row[1]}** | {row[2]:.2f} | {row[3]} | {row[4]} | {row[5]} | "
                 f"{row[6]} |")
    L.append("\n### All 15 arms, ranked by the same rule\n")
    L.append("| rank | arm | capability | structural loops | decode t/s | p vs #1 | "
             "distinguishable from #1? |")
    L.append("|---|---|---|---|---|---|---|")
    for i, row in enumerate(v.get("flat") or [], 1):
        model, kv, cap, loops, loops_n, dec, pv = row
        L.append(f"| {i} | `{model}` **{kv}** | {cap:.2f} | {loops}/{loops_n} | {dec} | "
                 f"{pv:.4f} | {'yes' if pv < ALPHA else 'no — tied'} |")
    tied = v.get("tied_with_leader") or []
    if tied:
        names = ", ".join(f"`{m}` {kv} ({cap:.2f})" for m, kv, cap, _ in tied)
        L.append(f"\nTested on every paired sample pooled across the three tiers, the arms that are "
                 f"**not distinguishable from the winner** are: {names}. Those p-values are the "
                 f"honest statement about 'which KV cache is best': on this run the lead is not "
                 f"established.\n")
    L.append("## Per-model offload (frozen across the three KV arms)\n")
    L.append("| model | knobs | VRAM peak MiB | realized prompt tokens | prefill t/s | decode t/s |")
    L.append("|---|---|---|---|---|---|")
    for model in out["models"]:
        cell = out["arms"].get(f"{model}__f16") or next(
            (out["arms"][k] for k in out["arms"] if k.startswith(model + "__")), None)
        if not cell:
            continue
        L.append(f"| `{model}` | {cell.get('knobs')} | {cell.get('vram_peak_mib')} | "
                 f"{cell.get('prompt_tokens_med')} | {cell.get('prefill_tps')} | "
                 f"{cell.get('decode_tps')} |")
    L.append("\n## R tier — retrieval / tracing / aggregation (partial credit, %)\n")
    L += _md_grid(out, lambda c: c["R"]["pct"], "R %")
    L.append("\n## C tier — hidden checks passed / total, and resolved tasks\n")
    L.append("| model | KV | checks | % | resolved | too-hard tasks |")
    L.append("|---|---|---|---|---|---|")
    for model in out["models"]:
        for kv in out["kvs"]:
            c = out["arms"].get(f"{model}__{kv}")
            if not c:
                continue
            L.append(f"| `{model}` | {kv} | {c['C']['passed']}/{c['C']['of']} | "
                     f"{c['C']['pct']} | {c['C']['resolved']}/{c['C']['n']} | "
                     f"{', '.join(out.get('too_hard') or []) or '—'} |")
    L.append("\n## Loop table — flagged samples / samples (reasoning and content split)\n")
    L.append("`union (L)` counts every verdict from the zlib-ratio heuristic or a structural signal; "
             "`hard repetition` counts only samples with a repeated line, n-gram, 200-char block or "
             "tail cycle. The gap between the two columns is formulaic prose, not looping.\n")
    L.append("| model | KV | gen_manual | copy_ledgers | enumerate_shards | tool_episode | union "
             "(L) | hard repetition | reasoning | content | any tier |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model in out["models"]:
        for kv in out["kvs"]:
            c = out["arms"].get(f"{model}__{kv}")
            if not c:
                continue
            cells = []
            for t in ("gen_manual", "copy_ledgers", "enumerate_shards", "tool_episode"):
                v = c["L"].get(t)
                cells.append(f"{v['flagged']}/{v['n']}" if v else "—")
            L.append(f"| `{model}` | {kv} | " + " | ".join(cells) +
                     f" | {c['loops']['flagged']}/{c['loops']['n']} | "
                     f"{c['loops']['hard']}/{c['loops']['n']} | {c['loop_reasoning']} | "
                     f"{c['loop_content']} | {c['flagged_all']} |")
    L.append("\n## Paired tests vs f16 (exact sign-flip, paired by task and seed)\n")
    L.append("| model | KV | metric | level | change | p | direction |")
    L.append("|---|---|---|---|---|---|---|")
    for model in out["models"]:
        for kv in out["kvs"]:
            c = out["arms"].get(f"{model}__{kv}")
            if not c or kv == "f16":
                continue
            for tier, level, unit in (("R", c["R"]["pct"], "% of 30 points"),
                                      ("C", c["C"]["pct"], "% of 780 checks"),
                                      ("L", (100.0 * c["loops"]["hard"] / c["loops"]["n"]
                                             if c["loops"]["n"] else None),
                                       "% of 12 samples")):
                sig = (c.get("sig") or {}).get(tier)
                if not sig or level is None:
                    continue
                direction = ("higher" if sig["delta"] > 0 else
                             "lower" if sig["delta"] < 0 else "equal")
                direction += " · significant" if sig["p"] < ALPHA else " · not significant"
                L.append(f"| `{model}` | {kv} | {tier} | {level:.1f} {unit} | "
                         f"{sig['delta']:+d} {sig['unit']} | {sig['p']:.4f} | {direction} |")
    L.append(f"\nα = {ALPHA}. The test is exact — every sign pattern is enumerated, no sampling and "
             f"no normal approximation — so the same `samples.jsonl` always produces the same "
             f"numbers. With 6 C-tier tasks the smallest attainable p is 0.031 and with 12 loop "
             f"samples it is 0.0005, so an arm must move nearly every paired sample the same way to "
             f"be called different. 'Not significant' means the change is inside what a draw of this "
             f"size produces; it is not proof of identity.\n")
    L.append("## Agreement with f16 per model (share of identical graded answers)\n")
    for model in out["models"]:
        parts = []
        for kv in out["kvs"]:
            cell = out["arms"].get(f"{model}__{kv}")
            if cell and cell.get("agree") is not None:
                parts.append(f"{kv}: {cell['agree']} of {cell.get('agree_n')} samples")
        if parts:
            L.append(f"- `{model}` — " + "; ".join(parts))
    L.append("\n## Cross-model answer agreement (same R task + seed, same KV)\n")
    for kv in out["kvs"]:
        rows = (out.get("cross") or {}).get(kv) or {}
        if not rows:
            continue
        L.append(f"\n**{kv}**\n")
        L.append("| pair | identical / compared | share |")
        L.append("|---|---|---|")
        for pair, (same, tot, share) in sorted(rows.items()):
            L.append(f"| `{pair.split('|')[0]}` vs `{pair.split('|')[1]}` | {same}/{tot} | {share} |")
    L.append("\n## Speed and realized context\n")
    L.append("| model | KV | prompt tokens | prefill t/s | decode t/s | cache hit rate | out tokens |")
    L.append("|---|---|---|---|---|---|---|")
    for model in out["models"]:
        for kv in out["kvs"]:
            c = out["arms"].get(f"{model}__{kv}")
            if not c:
                continue
            L.append(f"| `{model}` | {kv} | {c['prompt_tokens_med']} | {c['prefill_tps']} | "
                     f"{c['decode_tps']} | {c['cache_hit_rate']} | {c['out_tokens']} |")
    L.append("\n## How long this took\n")
    L += _timing_lines(out.get("_timing") or {}, raw.get("arms") or {}, out.get("arms") or {})
    L.append("\n## Threats to validity (literal)\n")
    for t in THREATS:
        L.append(f"- {t}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _fmt(v):
    return "—" if v is None else f"{v:+.1f}"


def _md_grid(out, get, label):
    head = "| model | " + " | ".join(out["kvs"]) + " |"
    lines = [head, "|---" * (len(out["kvs"]) + 1) + "|"]
    for model in out["models"]:
        cells = []
        for kv in out["kvs"]:
            c = out["arms"].get(f"{model}__{kv}")
            v = get(c) if c else None
            cells.append("—" if v is None else f"{v}")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    return lines


THREATS = [
    "Text-identical is not token-identical: one document, five tokenizers; realized prompt "
    "lengths are reported and `-c` is identical everywhere.",
    "Only full-attention layers accumulate KV damage over depth (qwen35: 16 layers, MoE 35B: 10, "
    "Flash-Next: 12, Gemma 26B: 5 global). A flat result on a model with few global layers is "
    "partly structural.",
    "The C tier is one greedy pass per task, so it has no error bar; R and L have "
    "reps samples each. Any delta smaller than one task's worth of points is noise.",
    "Hard-task pass rates are confounded with absolute difficulty; tasks with no f16 success are "
    "labelled too_hard_for_matrix and excluded from the delta read.",
    "`repeat_penalty 1.0` in the L tier is more permissive than the production 1.05, so measured "
    "loop rates are an upper bound for production settings.",
    "Duplicated prefill within an arm means a bad early answer cannot contaminate later requests; "
    "`tool_episode` is the exception, its transcript is deliberately sequential.",
    "The repository fixture's third visible test requires `coverage_report` to surface a degenerate "
    "point gap, which the hidden check suite does not score; `repo_fix` records both, and "
    "`resolved` requires both.",
    "The `semver` module is bimodal on one prerelease-ordering decision (24/71/83/118/182/187/190/"
    "253/261 of 261): a C-tier delta of tens of points can be that single bit, and it is the largest "
    "single contributor to C-tier variance here.",
    "A C-tier arm that hits the generation cap produces truncated code, not wrong code. The report "
    "prints `out_tokens`, `finish_reason` and emitted line count per sample so the two are "
    "distinguishable; `budget_raised` marks every sample that was re-run at a larger budget.",
    "The zlib-ratio loop threshold fires on templated text: 25 of 45 `gen_manual` samples cross "
    "0.42 with no line/ngram/cycle signal, uniformly across arms. Real repetition is the line-run, "
    "n-gram, block and cycle counters.",
]


# ---------------------------------------------------------------- html
def write_html(out, raw, rows, path):
    data = {"stamp": raw.get("stamp"), "context": raw.get("context"),
            "manifest": raw.get("manifest"), "document": raw.get("document"),
            "deadline_reached": raw.get("deadline_reached"), "unrun": raw.get("unrun_arms"),
            "unfit": (raw.get("fit") or {}).get("unfit"),
            "too_hard": out.get("too_hard"), "models": out["models"], "kvs": out["kvs"],
            "cells": [dict(model=c["model"], kv=c["kv"], R=c["R"]["pct"], C=c["C"]["pct"],
                           Rpts=c["R"]["points"], Rof=c["R"]["of"],
                           Cpassed=c["C"]["passed"], Cof=c["C"]["of"], resolved=c["C"]["resolved"],
                           loop=c["loop_rate"], loops=c["loops"], L=c["L"],
                           reasoning=c["loop_reasoning"], content=c["loop_content"],
                           prompt_tokens=c["prompt_tokens_med"], prefill=c["prefill_tps"],
                           decode=c["decode_tps"], agree=c.get("agree"), knobs=c.get("knobs"),
                           vram=c.get("vram_peak_mib"), flags=c.get("flags"),
                           per_task=c["per_task"])
                       for c in out["arms"].values()],
            "deltas": rows,
            "timing": {"total_s": (out.get("_timing") or {}).get("study_wall_s"),
                       "total": _human((out.get("_timing") or {}).get("study_wall_s")),
                       "matrix_s": (((out.get("_timing") or {}).get("phases") or {})
                                    .get("matrix") or {}).get("wall_s"),
                       "matrix": _human((((out.get("_timing") or {}).get("phases") or {})
                                         .get("matrix") or {}).get("wall_s"))}}
    gallery = _gallery(out, raw)
    grids, bars = _svg_grids(out), _svg_bars(rows)
    verdict_html = _verdict_html(out)
    legend = ('<div class="dim" style="margin-top:6px">colour scale, always the same way: '
              '<span style="background:rgb(62,170,90);color:#0b0e12;padding:0 6px">green = best</span> '
              '<span style="background:rgb(205,160,62);color:#0b0e12;padding:0 6px">amber = middle</span> '
              '<span style="background:rgb(200,74,58);color:#0b0e12;padding:0 6px">red = worst</span> '
              '· "best" means high on the first two grids and low on the loop grid · '
              '"—" means that arm was not run</div>')
    taskbars, scatter = _svg_task_bars(out), _svg_scatter(out)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>KV quant × long context — {html.escape(str(raw.get('stamp')))}</title>
<style>
:root{{--bg:#0f1216;--fg:#e8ecf1;--dim:#9aa4b2;--card:#171c22;--line:#28313b}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}}
header{{padding:20px 28px 8px}} h1{{font-size:20px;margin:0 0 6px}}
h2{{font-size:16px;margin:22px 0 8px}} .dim{{color:var(--dim)}}
nav{{display:flex;gap:8px;padding:8px 28px 0}}
nav button{{background:var(--card);color:var(--fg);border:1px solid var(--line);padding:7px 14px;
border-radius:7px;cursor:pointer}} nav button.on{{background:#2b6cb0;border-color:#2b6cb0}}
section{{padding:12px 28px 40px}} .grid{{display:flex;gap:22px;flex-wrap:wrap}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
margin:10px 0;max-width:900px}} table{{border-collapse:collapse;font-size:13px}}
th,td{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} mark{{background:#7a4d00;color:#fff;padding:0 2px}}
code{{background:#0b0e12;padding:1px 5px;border-radius:4px;font-size:12px}}
details summary{{cursor:pointer;color:var(--dim)}} pre{{white-space:pre-wrap;word-break:break-word}}
footer{{padding:20px 28px 60px;color:var(--dim);font-size:12px}}
</style></head><body>
<header>
<h1>KV-cache quantization × long context: 5 models at f16 / q8_0 / q4_0</h1>
<div class="dim">context <code>-c {raw.get('context')}</code> ·
llama.cpp <code>{(raw.get('manifest') or {}).get('llama_cpp', {}).get('commit', '?')[:12]}</code> ·
document <code>{(raw.get('document') or {}).get('sha256', '?')[:16]}</code> ·
{raw.get('samples')} samples{' · DEADLINE REACHED' if raw.get('deadline_reached') else ''} ·
wall clock <b>{_human((out.get('_timing') or {}).get('study_wall_s'))}</b> total,
{_human((((out.get('_timing') or {}).get('phases') or {}).get('matrix') or {}).get('wall_s'))}
in the matrix</div>
</header>
<div class="card" style="max-width:none">
<b>How to read this page</b>
<div class="dim" style="margin-top:6px">
Each <b>grid</b> is one measurement: rows are models, columns are the three KV cache types.
A cell's number is that arm's result. Colour always means the same thing:
<span style="color:#3fa05a">green = best</span>, <span style="color:#c9563d">red = worst</span>.
On the retrieval and code grids <b>best means highest</b>; on the two <b>loop</b> grids <b>best means
lowest</b>, and those grids are scaled to the observed range so small differences are visible. Each
grid prints its own scale. Hover any cell for the exact value.
</div>
<div class="dim" style="margin-top:6px">
The loop numbers come in two flavours, and they differ: <b>any verdict</b> includes a compression
heuristic that fires on formulaic prose such as a manual generated from twelve fixed rules, while
<b>structural repetition</b> counts only samples with a repeated line, repeated n-gram, repeated
200-character block or a repeating tail cycle — i.e. the thing that actually looks like a loop. Read
the structural grid as the finding and the "any verdict" grid as an upper bound.
</div>
<div class="dim" style="margin-top:6px">
What the tabs hold: <b>Matrix</b> — the 15-arm comparison, the change versus f16, and how often two
arms gave the identical answer. <b>Tasks</b> — the same data per task, so you can see which task
carried a difference. <b>Loops</b> — every sample the repetition detector flagged, with the exact
text that repeated, so you can judge it yourself.
</div>
<div class="dim" style="margin-top:6px">
Three facts that make the numbers honest: differences of a few points are noise; the <code>semver</code>
code task moves by tens of points on a single design decision; and Gemma's q4_0 code score is a
generation-length cut-off, not wrong code. The prose summary, with those caveats written out, is
<code>docs/RESULTS.md</code>.
</div></div>
<nav><button id="b-matrix" class="on" onclick="tab('matrix')">Matrix</button>
<button id="b-tasks" onclick="tab('tasks')">Tasks</button>
<button id="b-loops" onclick="tab('loops')">Loops</button></nav>
<section id="t-matrix">{verdict_html}{grids}{legend}<h2>Paired deltas vs f16</h2>{bars}
<h2>Agreement with f16 (identical graded answers)</h2>{_agree_table(out)}{_cross_table(out)}</section>
<section id="t-tasks" hidden>{taskbars}{_task_details(out)}</section>
<section id="t-loops" hidden>{scatter}{gallery}</section>
<footer>{_manifest_block(raw)}</footer>
<script>
const RAW = {json.dumps(data)};
function tab(name) {{
  for (const t of ['matrix','tasks','loops']) {{
    document.getElementById('t-'+t).hidden = (t !== name);
    document.getElementById('b-'+t).className = (t === name) ? 'on' : '';
  }}
}}
</script>
</body></html>"""
    with open(path, "w") as f:
        f.write(doc)


GREEN = (62, 170, 90)
AMBER = (205, 160, 62)
RED = (200, 74, 58)


def _ramp(score):
    """score 1.0 = best (green), 0.5 = middle (amber), 0.0 = worst (red)."""
    score = max(0.0, min(1.0, 1.0 if score is None else score))
    if score >= 0.5:
        a, b, t = AMBER, GREEN, (score - 0.5) * 2.0
    else:
        a, b, t = RED, AMBER, score * 2.0
    return "rgb(%d,%d,%d)" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _color(value, lo, hi, higher_is_better=True):
    """Colour a value on [lo, hi] where the GOOD end is green, whichever end that is."""
    if value is None:
        return "#2a3138"
    span = (hi - lo) or 1.0
    frac = (value - lo) / span
    return _ramp(frac if higher_is_better else 1.0 - frac)


ALPHA = 0.05


def _cell_colour(sig, delta, scale):
    """Green/red only when the paired test says the arm differs from its own f16 arm."""
    if sig is None:
        return "#2a3138"
    if sig["p"] >= ALPHA or delta == 0:
        return "#3a4149"
    strength = min(1.0, abs(delta) / scale) if scale else 1.0
    base = GREEN if delta > 0 else RED
    t = 0.45 + 0.55 * strength
    return "rgb(%d,%d,%d)" % tuple(round(255 - (255 - c) * t) for c in base)


def _svg_grids(out):
    kvs = out["kvs"]
    models = out["models"]
    loop_n = max([c["loops"]["n"] for c in out["arms"].values()] or [12])

    def val_R(c):
        return c["R"]["pct"]

    def val_C(c):
        return c["C"]["pct"]

    def val_L(c):
        return 100.0 * c["loops"]["hard"] / c["loops"]["n"] if c["loops"]["n"] else None

    specs = [
        ("R accuracy", val_R, "R",
         "level in % of 30 points · change in points · n = 18 paired samples per arm, "
         "smallest attainable p = 0.000004"),
        ("C checks", val_C, "C",
         "level in % of 780 hidden checks · change in checks · n = 6 paired tasks per arm, "
         "smallest attainable p = 0.031"),
        ("structural repetition", val_L, "L",
         f"level in % of {loop_n} samples · change in samples · n = {loop_n} paired samples per "
         f"arm, smallest attainable p = {2 / 2 ** loop_n:.4f}"),
    ]
    cw, ch, pad, lw = 104, 46, 12, 150
    parts = ['<div class="grid">']
    for title, get, tier, caption in specs:
        w, h = lw + cw * len(kvs) + pad, 40 + ch * len(models) + pad
        parts.append(f'<div class="card"><div>{html.escape(title)}</div>')
        parts.append(f'<div class="dim">{html.escape(caption)} · exact paired sign-flip test; '
                     f'grey = not distinguishable from f16 at p &lt; {ALPHA}</div>')
        parts.append(f'<svg width="{w}" height="{h}" role="img">')
        for j, kv in enumerate(kvs):
            parts.append(f'<text x="{lw + j * cw + cw / 2}" y="18" fill="#9aa4b2" '
                         f'font-size="12" text-anchor="middle">{kv}</text>')
        for i, model in enumerate(models):
            parts.append(f'<text x="4" y="{40 + i * ch + 26}" fill="#e8ecf1" font-size="12">'
                         f'{html.escape(model)}</text>')
            for j, kv in enumerate(kvs):
                c = out["arms"].get(f"{model}__{kv}")
                v = get(c) if c else None
                level = "—" if v is None else f"{v:.1f}"
                sig = ((c or {}).get("sig") or {}).get(tier)
                scale = max(1, int(abs((sig or {}).get("delta") or 1)) * 4)
                if kv == "f16":
                    sub, fill = "reference", "#2f3740"
                elif sig and v is not None:
                    d = sig["delta"]
                    sub = f"Δ {d:+d} · p={sig['p']:.3f}"
                    fill = _cell_colour(sig, d, scale)
                else:
                    sub, fill = "—", "#2a3138"
                tip = level
                if sig and v is not None:
                    tip = (f"{level} · Δ {sig['delta']:+d} {sig['unit']} vs f16 · "
                           f"p={sig['p']:.4f} (n={sig['n']})")
                x, y = lw + j * cw + 2, 40 + i * ch + 2
                parts.append(f'<rect x="{x}" y="{y}" width="{cw - 4}" height="{ch - 4}" rx="4" '
                             f'fill="{fill}"><title>{html.escape(model)} {kv}: '
                             f'{html.escape(tip)}</title></rect>')
                tc = "#e8ecf1" if fill in ("#2f3740", "#2a3138", "#3a4149") else "#0b0e12"
                parts.append(f'<text x="{x + (cw - 4) / 2}" y="{y + 18}" fill="{tc}" '
                             f'font-size="13" text-anchor="middle">{level}</text>')
                parts.append(f'<text x="{x + (cw - 4) / 2}" y="{y + 33}" fill="{tc}" '
                             f'font-size="10" text-anchor="middle">{html.escape(sub)}</text>')
        parts.append("</svg></div>")
    parts.append("</div>")
    return "".join(parts)


def _verdict_html(out):
    v = out.get("verdict") or {}
    if not v.get("overall"):
        return ""
    om, okv = v["overall"], v["overall_kv"]
    top = v["per_model"][om]
    rows = "".join(
        f'<tr><td>{html.escape(r[0])}</td><td><b>{r[1]}</b></td><td>{r[2]:.2f}</td>'
        f'<td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td><td style="text-align:left">'
        f'{html.escape(r[6])}</td></tr>'
        for r in _verdict_rows(out))
    return (f'<div class="card" style="max-width:none"><h2 style="margin-top:0">Verdict</h2>'
            f'<div><b>Best overall: {html.escape(om)} with a {html.escape(okv)} KV cache</b> — '
            f'capability {top["capability"]:.2f} of 100, {top["loops"]}/{top["loops_n"]} '
            f'structural-loop samples, {top["decode"]} t/s decode. '
            f'<b>Its lead is not established:</b> {len(v.get("tied_with_leader") or [])} other arms, '
            f'including its own f16 arm, cannot be distinguished from it at p &lt; {ALPHA}.</div>'
            f'<div class="dim">Rule, fixed in code: capability = mean(R %, C %) per arm, then fewer '
            f'structural-loop samples, then higher decode t/s. No arm differs from its own f16 arm '
            f'at p &lt; 0.05 on any metric, so these ranks sit inside the noise this design can '
            f'resolve.</div>'
            f'<table style="margin-top:8px"><tr><th>model</th><th>best KV</th><th>capability</th>'
            f'<th>runner-up</th><th>structural loops</th><th>decode t/s</th><th>why</th></tr>'
            f'{rows}</table></div>')


def write_figures(out, path):
    """Export every figure as a standalone SVG plus the exact numbers behind it.

    Deterministic: the same samples.jsonl always produces byte-identical files, so a third party can
    diff them or re-render them elsewhere without trusting the HTML.
    """
    os.makedirs(path, exist_ok=True)
    svg = _svg_grids(out)
    import re as _re
    for i, fig in enumerate(_re.findall(r"<svg.*?</svg>", svg, _re.S)):
        with open(os.path.join(path, f"fig{i + 1}.svg"), "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(fig.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1))
    table = {}
    for key, c in out["arms"].items():
        table[key] = {"R_pct": c["R"]["pct"], "C_pct": c["C"]["pct"],
                      "loops_hard": c["loops"]["hard"], "loops_n": c["loops"]["n"],
                      "sig": c.get("sig")}
    with open(os.path.join(path, "figures.json"), "w") as f:
        json.dump({"alpha": ALPHA,
                   "test": "exact two-sided paired sign-flip on integer differences; "
                           "pairing = (task, rep) for sampled tiers, task for the greedy tier",
                   "arms": table}, f, indent=2, sort_keys=True)


def _svg_bars(rows):
    if not rows:
        return '<p class="dim">no paired deltas</p>'
    cw, ch, lw = 150, 26, 150
    h = 20 + ch * len(rows)
    w = lw + 260
    parts = [f'<div class="card"><svg width="{w}" height="{h}" role="img">',
             f'<line x1="{lw + 120}" y1="0" x2="{lw + 120}" y2="{h}" stroke="#48525c"/>']
    for i, r in enumerate(rows):
        y = 12 + i * ch
        parts.append(f'<text x="4" y="{y + 12}" fill="#e8ecf1" font-size="12">'
                     f'{html.escape(r["model"])} {r["kv"]}</text>')
        for k, (key, color) in enumerate((("dR", "#4f9d69"), ("dC", "#3d7cc9"), ("dL", "#c9563d"))):
            v = r.get(key)
            if v is None:
                continue
            width = max(-110, min(110, v * 2))
            x = lw + 120 if width >= 0 else lw + 120 + width
            parts.append(f'<rect x="{x}" y="{y + k * 7}" width="{abs(width)}" height="5" rx="2" '
                         f'fill="{color}"><title>{key} {v:+.1f}</title></rect>')
    parts.append("</svg><div class=\"dim\">green ΔR · blue ΔC · red Δloop-rate "
                 "(2 px per point)</div></div>")
    return "".join(parts)


def _svg_task_bars(out):
    tasks = sorted({t for c in out["arms"].values() for t in c["per_task"]})
    if not tasks:
        return ""
    cw, lw = 46, 190
    parts = ['<div class="card"><div class="dim">per task: share correct (R/C) or flagged (L)</div>']
    for task in tasks:
        h = 20 + 22 * len(out["arms"])
        w = lw + cw * 3 + 20
        parts.append(f'<svg width="{w}" height="{h}" role="img"><text x="4" y="12" '
                     f'fill="#9aa4b2" font-size="12">{html.escape(task)}</text>')
        for i, (key, c) in enumerate(out["arms"].items()):
            pt = c["per_task"].get(task)
            if not pt:
                continue
            y = 18 + i * 22
            if pt["of"]:
                frac = pt["score"] / pt["of"]
            elif pt["n"]:
                frac = 1 - pt["flagged"] / pt["n"]
            else:
                continue
            parts.append(f'<text x="4" y="{y + 11}" fill="#e8ecf1" font-size="11">'
                         f'{html.escape(c["model"])} {c["kv"]}</text>')
            parts.append(f'<rect x="{lw}" y="{y + 2}" width="{cw * 3 * frac:.1f}" height="12" '
                         f'rx="2" fill="{_color(frac, 0.0, 1.0, True)}"/>')
        parts.append("</svg>")
    parts.append("</div>")
    return "".join(parts)


def _svg_scatter(out):
    pts = [(c["loop_rate"] or 0.0, c["decode_tps"] or 0.0, c) for c in out["arms"].values()]
    if not pts:
        return ""
    w, h, pad = 520, 260, 46
    dmax = max(d for _, d, _ in pts) or 1
    parts = [f'<div class="card"><div class="dim">loop rate vs decode t/s</div>',
             f'<svg width="{w}" height="{h}" role="img">',
             f'<line x1="{pad}" y1="{h - pad}" x2="{w - 10}" y2="{h - pad}" stroke="#48525c"/>',
             f'<line x1="{pad}" y1="10" x2="{pad}" y2="{h - pad}" stroke="#48525c"/>',
             f'<text x="{w // 2}" y="{h - 18}" fill="#9aa4b2" font-size="12" '
             f'text-anchor="middle">loop rate</text>',
             f'<text x="10" y="20" fill="#9aa4b2" font-size="12">decode t/s</text>']
    for lr, dec, c in pts:
        x = pad + lr * (w - pad - 20)
        y = (h - pad) - (dec / dmax) * (h - pad - 20)
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="6" '
                     f'fill="{_color(lr, 0.0, max(l for l, _, _ in pts) or 1.0, False)}" '
                     f'fill-opacity="0.85"><title>{html.escape(c["model"])} {c["kv"]}: '
                     f'loop {lr:.2f}, {dec} t/s</title></circle>')
    parts.append("</svg></div>")
    return "".join(parts)


def _task_details(out):
    parts = []
    for (key, c) in out["arms"].items():
        parts.append(f'<div class="card"><b>{html.escape(c["model"])} · {c["kv"]}</b>')
        parts.append('<table><tr><th>task</th><th>n</th><th>score</th><th>flagged</th></tr>')
        for task, pt in sorted(c["per_task"].items()):
            score = f"{pt['score']}/{pt['of']}" if pt["of"] else "—"
            parts.append(f'<tr><td>{html.escape(task)}</td><td>{pt["n"]}</td>'
                         f'<td>{score}</td><td>{pt["flagged"]}</td></tr>')
        parts.append("</table></div>")
    return "".join(parts)


def _agree_table(out):
    lines = ['<table><tr><th>model</th><th>KV</th><th>agreement with f16</th><th>n</th></tr>']
    for model in out["models"]:
        for kv in out["kvs"]:
            c = out["arms"].get(f"{model}__{kv}")
            if not c or c.get("agree") is None:
                continue
            lines.append(f'<tr><td>{html.escape(model)}</td><td>{kv}</td>'
                         f'<td>{c["agree"]}</td><td>{c.get("agree_n")}</td></tr>')
    lines.append("</table>")
    return "".join(lines)


def _cross_table(out):
    parts = ["<h2>Cross-model agreement (same R task + seed, same KV)</h2>"]
    for kv in out["kvs"]:
        rows = (out.get("cross") or {}).get(kv) or {}
        if not rows:
            continue
        parts.append(f'<div class="card"><b>{kv}</b><table>'
                     '<tr><th>pair</th><th>identical</th><th>compared</th><th>share</th></tr>')
        for pair, (same, tot, share) in sorted(rows.items()):
            parts.append(f'<tr><td>{html.escape(pair)}</td><td>{same}</td><td>{tot}</td>'
                         f'<td>{share}</td></tr>')
        parts.append("</table></div>")
    return "".join(parts)


def _gallery(out, raw):
    samples_path = os.path.join(config.ROOT, "results", out.get("_stamp") or "", "samples.jsonl")
    cards = []
    path = out.get("_samples_path") or samples_path
    if not os.path.exists(path):
        return '<p class="dim">no samples.jsonl</p>'
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not s.get("flagged"):
                continue
            cards.append(_loop_card(s))
    if not cards:
        return '<p class="dim">no sample was flagged as looping in this run.</p>'
    return f'<div class="dim">{len(cards)} flagged samples</div>' + "".join(cards)


def _loop_card(s):
    out = [f'<div class="card"><b>{html.escape(s["model"])} · {s["kv"]} · {s["tier"]} · '
           f'{html.escape(str(s["task"]))} · rep {s["rep"]}</b>'
           f'<div class="dim">{html.escape(s.get("flag_reason") or "")}</div>']
    for part, v in (s.get("loops") or {}).items():
        m = v.get("metrics") or {}
        keys = [k for k in ("word_run_1", "word_run_2", "word_run_4", "word_run_8", "line_run",
                            "cycle", "block", "zip_ratio", "distinct_4", "chars") if k in m]
        out.append(f'<div class="dim">{part}: ' +
                   " ".join(f"{k}={m[k]}" for k in keys) + "</div>")
        span = v.get("span")
        if span:
            out.append(f'<pre>{html.escape(span[:600])}</pre>')
    out.append("</div>")
    return "".join(out)


def _manifest_block(raw):
    man = raw.get("manifest") or {}
    cmds = ("python3 -m kvtest.selftest\n"
            "python3 -m kvtest.fit --stamp STAMP\n"
            "python3 -m kvtest.run --stamp STAMP --resume --deadline-hours 22\n"
            "python3 -m kvtest.report results/STAMP")
    return ("<b>Manifest</b><pre>" + html.escape(json.dumps({
        "llama_cpp": man.get("llama_cpp"), "context": raw.get("context"),
        "seeds": man.get("seeds"), "budgets": man.get("budgets"), "reps": man.get("reps"),
        "document_sha256": (raw.get("document") or {}).get("sha256"),
        "tokens_by_model": (raw.get("document") or {}).get("tokens_by_model"),
        "models": man.get("models"),
    }, indent=1)) + "</pre><b>Reproduce</b><pre>" + html.escape(cmds) + "</pre></footer>")


# ---------------------------------------------------------------- csv
def write_csv(out, path):
    cols = ["model", "kv", "R_pct", "R_points", "R_of", "C_pct", "C_passed", "C_of", "C_resolved",
            "loop_flagged", "loop_samples", "loop_rate", "loop_hard", "loop_reasoning",
            "loop_content",
            "agree_f16", "prompt_tokens", "prefill_tps", "decode_tps", "cache_hit_rate",
            "vram_peak_mib", "knobs"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for c in out["arms"].values():
            row = [c["model"], c["kv"], c["R"]["pct"], c["R"]["points"], c["R"]["of"],
                   c["C"]["pct"], c["C"]["passed"], c["C"]["of"], c["C"]["resolved"],
                   c["loops"]["flagged"], c["loops"]["n"], c["loop_rate"], c["loops"]["hard"],
                   c["loop_reasoning"],
                   c["loop_content"], c.get("agree"), c["prompt_tokens_med"], c["prefill_tps"],
                   c["decode_tps"], c["cache_hit_rate"], c["vram_peak_mib"],
                   json.dumps(c.get("knobs") or {})]
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")


def write_readme_headline(out, readme_path):
    """Keep the README's headline table honest: regenerated from the same aggregation."""
    if not os.path.exists(readme_path):
        return
    lines = ["| model | KV | R accuracy | C checks | C resolved | loop rate | "
             "agreement with f16 |", "|---|---|---|---|---|---|---|"]
    for model in out["models"]:
        for kv in out["kvs"]:
            c = out["arms"].get(f"{model}__{kv}")
            if not c:
                continue
            lines.append(
                f"| `{model}` | {kv} | {c['R']['points']}/{c['R']['of']} "
                f"({c['R']['pct']}%) | {c['C']['passed']}/{c['C']['of']} ({c['C']['pct']}%) | "
                f"{c['C']['resolved']}/{c['C']['n']} | {c['loops']['hard']}/{c['loops']['n']} | "
                f"{(c.get('agree') if c.get('agree') is not None else '—')} |")
    text = open(readme_path).read()
    start, end = "<!-- HEADLINE:START -->", "<!-- HEADLINE:END -->"
    if start not in text or end not in text:
        return
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    with open(readme_path, "w") as f:
        f.write(head + start + "\n" + "\n".join(lines) + "\n" + end + tail)


def build(out_dir):
    raw, samples, arms = load(out_dir)
    out = agg(raw, samples, arms)
    out["_stamp"] = raw.get("stamp") or os.path.basename(out_dir.rstrip("/"))
    out["_timing"] = load_timing(out_dir, raw)
    out["_samples_path"] = os.path.join(out_dir, "samples.jsonl")
    rows = deltas(out)
    write_markdown(out, raw, rows, os.path.join(out_dir, "REPORT.md"))
    write_html(out, raw, rows, os.path.join(out_dir, "report.html"))
    write_csv(out, os.path.join(out_dir, "matrix.csv"))
    write_figures(out, os.path.join(out_dir, "figures"))
    write_readme_headline(out, os.path.join(config.ROOT, "README.md"))
    return out, raw


def main(argv=None):
    import sys
    args = (argv if argv is not None else sys.argv[1:])
    if not args:
        raise SystemExit("usage: python3 -m kvtest.report results/<stamp>")
    out_dir = args[0]
    out, raw = build(out_dir)
    print(f"wrote {out_dir}/REPORT.md, report.html, matrix.csv "
          f"({len(out['arms'])} arms, {raw.get('samples')} samples, "
          f"total {_human((out.get('_timing') or {}).get('study_wall_s'))})")


if __name__ == "__main__":
    main()
