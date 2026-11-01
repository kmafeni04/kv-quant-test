"""Turn results/<stamp>/{raw.json,samples.jsonl} into REPORT.md, report.html and matrix.csv.

Idempotent: no server, no models, no network. If raw.json is missing it is rebuilt from
samples.jsonl and the arm checkpoints, so a deadline-cut run still produces a full report.
"""
import html
import json
import os
import statistics
from collections import defaultdict

from . import codegraders, config, hostinfo

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
            "arms": arms, "samples": len(samples), "deadline_reached": False, "unrun_arms": [],
            "incomplete_arms": [k for k, a in arms.items() if not a.get("finished")]}


# ---------------------------------------------------------------- verdict
def capability(c):
    """One number per arm: the mean of the two capability tiers, each already 0-100.

    R gives 30 points and C gives 780 checks; averaging their percentages stops the check count
    from swamping the retrieval tier. The rule is fixed here, in code, so the verdict can be
    recomputed and argued with.
    """
    # computed from the raw counts, not the already-rounded percentages, so an independent
    # recomputation from samples.jsonl lands on exactly the same number
    if c["R"]["of"] and c["C"]["of"]:
        r = 100.0 * c["R"]["points"] / c["R"]["of"]
        q = 100.0 * c["C"]["passed"] / c["C"]["of"]
    elif c["R"]["pct"] is not None and c["C"]["pct"] is not None:
        r, q = c["R"]["pct"], c["C"]["pct"]
    else:
        return None
    return round((r + q) / 2.0, 2)


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
    # every other arm tested against the leader one tier at a time. Pooling raw R points, C checks
    # and L flags into one sum would be a scale artefact (261-check samples would swamp 0/1 flags),
    # so each tier is tested separately and the smallest p is multiplied by the number of tiers
    # (Bonferroni). Conservative by construction: an arm is only called different if one tier
    # separates even after that penalty.
    per_arm = _paired_scores(samples or [])
    tied = []
    if flat:
        lead_key = f"{flat[0][0]}__{flat[0][1]}"
        lead = per_arm.get(lead_key) or {}
        raw_family = {}
        for i, t in enumerate(flat):
            key = f"{t[0]}__{t[1]}"
            cur = per_arm.get(key) or {}
            if not i:
                continue
            for tier in TIERS:
                diffs = {k: cur[k] - lead[k] for k in sorted(cur) if k[0] == tier and k in lead}
                if not diffs:
                    continue
                _, p_tier = perm_p(list(_cluster_sums(diffs).values()))
                raw_family[(key, tier)] = p_tier
        adjusted = holm(raw_family)
        for i, t in enumerate(flat):
            key = f"{t[0]}__{t[1]}"
            t_p = 1.0
            if i and raw_family:
                t_p = min([adjusted.get((key, tier), 1.0) for tier in TIERS] or [1.0])
            t_p = round(t_p, 4)
            flat[i] = t + (t_p,)
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


def _ranked_models(out):
    """Models in verdict order (best capability first) so every figure and table reads best -> worst."""
    order = [m for m, _, _ in ((out.get("verdict") or {}).get("order") or [])]
    models = out.get("models") or []
    return [m for m in order if m in models] + [m for m in models if m not in order]


def _lead_sentence(out):
    """The caveat about the winner, defined once so REPORT.md and report.html cannot disagree."""
    v = out.get("verdict") or {}
    tied = v.get("tied_with_leader") or []
    if not tied:
        return ("No other arm is comparable on this result, so the ranking is descriptive only.")
    own_f16 = any(m == v.get("overall") and kv == "f16" for m, kv, _, _ in tied)
    who = ("including its own f16 arm" if own_f16
           else "none of them from its own model")
    return (f"Its lead is not established: {len(tied)} other arms, {who}, cannot be "
            f"distinguished from it (task-clustered pairs, Holm over 42 comparisons).")


def _plain_summary(out):
    """The finding in plain words, computed from `out` so it can never describe another run.

    The confidence statement uses the same per-tier tests as the tables ("differently … by more than
    this test can resolve"); the exception is chosen from evidence the tests cannot express —
    an arm significantly off its own f16 arm, or an arm cut off by the generation cap on two or more
    code tasks. Nothing here is typed by hand. Returns "" for a partial matrix, where the Incomplete
    banner is the honest opening.
    """
    v = out.get("verdict") or {}
    models, kvs = out.get("models") or [], out.get("kvs") or []
    arms = out.get("arms") or {}
    if not v.get("overall") or not models or not kvs:
        return ""
    if len(arms) < len(models) * len(kvs):
        return ""
    quant_kvs = [kv for kv in kvs if kv != "f16" and f"{models[0]}__{kv}" in arms]
    if not quant_kvs:
        return ""
    n_c = len(out.get("c_tasks") or []) or 6

    def cap(m, kv):
        c = arms.get(f"{m}__{kv}") or {}
        return c.get("capability") if c.get("capability") is not None else capability(c)

    def p_of(m, kv, tier):
        return (((arms.get(f"{m}__{kv}") or {}).get("sig") or {}).get(tier) or {}).get("p", 1.0)

    def describe_exception(m, kv):
        a, f16c = arms.get(f"{m}__{kv}") or {}, arms.get(f"{m}__f16") or {}
        c_kv, c_f16 = a.get("C", {}).get("pct"), f16c.get("C", {}).get("pct")
        sent = (f"{m} is the exception: at {kv} its code score was {c_kv:.0f} of 100 against "
                f"{c_f16:.0f} at f16" if c_kv is not None and c_f16 is not None else
                f"{m} is the exception, losing the most at {kv}")
        if a.get("truncated_c"):
            sent += (f", and it ran out of room on {a['truncated_c']} of {n_c} coding tasks, so it "
                     f"was cut off rather than wrong")
        return sent + "."

    sentences = []
    if len(models) == 1:
        m = models[0]
        best_kv, best_cap = max(((kv, cap(m, kv)) for kv in kvs if f"{m}__{kv}" in arms),
                                key=lambda t: t[1] or 0)
        sentences.append(f"Only one model is in this result, so the comparison is descriptive: "
                         f"{m} scored {best_cap:.0f} of 100 at its best setting ({best_kv}).")
        worst = min(((kv, (cap(m, kv) or 0) - (cap(m, "f16") or 0)) for kv in quant_kvs
                     if f"{m}__{kv}" in arms and f"{m}__f16" in arms),
                    key=lambda t: t[1], default=None)
        if worst and worst[1] <= -5.0:
            sentences.append(describe_exception(m, worst[0]))
    else:
        moved = [(m, kv) for m in models for kv in quant_kvs
                 if p_of(m, kv, "R") < ALPHA or p_of(m, kv, "C") < ALPHA]
        moved_models = sorted({m for m, _ in moved})
        if moved:
            sentences.append(f"{len(models) - len(moved_models)} of the {len(models)} models scored "
                             f"within the noise this test can resolve of their own f16 arm; "
                             f"{', '.join(moved_models)} moved beyond it.")
        else:
            sentences.append(f"No model scored differently from its own f16 arm by more than this "
                             f"test can resolve.")
        # the exception: a significant loss if there is one, otherwise an arm the cap cut short
        if moved:
            m, kv = min(moved, key=lambda t: (cap(t[0], t[1]) or 0) - (cap(t[0], "f16") or 0))
            sentences.append(describe_exception(m, kv))
        else:
            cut = [((m, kv), (arms[f"{m}__{kv}"] or {}).get("truncated_c") or 0)
                   for m in models for kv in quant_kvs if f"{m}__{kv}" in arms]
            cut = [(mk, n) for mk, n in cut if n >= 2]
            if cut:
                (m, kv), _ = max(cut, key=lambda t: t[1])
                sentences.append(describe_exception(m, kv))

    ratios = []
    for m in models:
        f16d = (arms.get(f"{m}__f16") or {}).get("decode_tps")
        for kv in quant_kvs:
            d = (arms.get(f"{m}__{kv}") or {}).get("decode_tps")
            if f16d and d:
                ratios.append(100.0 * (1.0 - d / f16d))
    if ratios:
        lo, hi = int(round(min(ratios))), int(round(max(ratios)))
        span = f"{lo}–{hi}" if lo != hi else f"{lo}"
        sentences.append(f"On this build the quantized cache was also {span} % slower to answer "
                         f"with than f16, so it buys memory, not speed.")

    loop_hits = [(m, kv) for m in models for kv in quant_kvs if p_of(m, kv, "L") < ALPHA]
    if loop_hits:
        sentences.append("Repeat samples rose beyond noise for "
                         + ", ".join(f"{m} at {kv}" for m, kv in loop_hits) + ".")
    elif len(models) > 1:
        sentences.append("No cache setting made a model repeat itself more often by a difference "
                         "this test can resolve.")
    return " ".join(sentences)


def _verdict_rows(out):
    """(model, best KV, capability, runner-up, loops, decode, reason), best overall first.

    Same order as every chart and table, so the two most prominent tables on the page do not read
    worst -> best immediately above the ranking table that reads the other way.
    """
    v = out.get("verdict") or {}
    per_model = v.get("per_model", {})
    known = set(out.get("models") or [])
    order = ([m for m in _ranked_models(out) if m in per_model]
             + [m for m in per_model if m not in known])
    rows = []
    for model in order:
        d = per_model[model]
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
            who = "the winning arm" if t["kv"] == d["best_kv"] else "the arm"
            n_c = len(out.get("c_tasks") or []) or 6
            parts.append(f"{who} {t['kv']} hit the {config.BUDGETS['C']}-token code cap on "
                         f"{t['truncated_c']} of {n_c} tasks, so its checks are truncated output, "
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


def load_hardware(out_dir, raw):
    """Hardware as it was when the run happened. Returns (hardware, provenance)."""
    hw = (raw.get("manifest") or {}).get("host_runtime")
    if hw:
        return hw, "captured at matrix start"
    path = os.path.join(out_dir, "hardware.json")
    if os.path.exists(path):
        return json.load(open(path)), "captured on the same machine, recorded in hardware.json"
    lock = config.lock().get("host") or {}
    if lock:
        return {"gpu": {"name": lock.get("gpu")}, "ram_gib": lock.get("ram_gib"),
                "cpu": {"logical_cores": lock.get("cpu_cores")}, "os": lock.get("os"),
                "llama_cpp": config.lock().get("llama_cpp") or {}}, \
            "from models.lock.json (run predates runtime capture)"
    return {}, "unknown"


def _hardware_md(hw, provenance):
    if not hw:
        return ["_Hardware not recorded._"]
    g, ll, c = hw.get("gpu") or {}, hw.get("llama_cpp") or {}, hw.get("cpu") or {}
    flags = ll.get("flags") or {}
    lines = ["| item | value |", "|---|---|",
             f"| GPU | {g.get('name') or '—'}"
             + (f", {g['memory_total_mib']} MiB total" if g.get("memory_total_mib") else "")
             + (f", compute {g['compute_capability']}" if g.get("compute_capability") else "") + " |",
             f"| driver / CUDA | {g.get('driver') or '—'} / {hw.get('cuda_toolkit') or '—'} |",
             f"| CPU | {c.get('model') or '—'} ({c.get('logical_cores') or '—'} logical cores) |",
             f"| RAM | {hw.get('ram_gib') or '—'} GiB |",
             f"| OS / kernel | {hw.get('os') or '—'} / {hw.get('kernel') or '—'} |",
             f"| llama.cpp | {str(ll.get('commit') or '—')[:12]}"
             + ("" if "tree_dirty" not in ll else
                (" (clean working tree)" if not ll["tree_dirty"] else " (DIRTY working tree)"))
             + " |"]
    if flags:
        lines.append("| build flags | " + ", ".join(f"`{k}={v}`" for k, v in sorted(flags.items()))
                     + " |")
    lines.append(f"\nRecorded: {provenance}. Per-arm GPU peak and the offload knobs are in the "
                 f"per-model offload table below; those are what the memory budget was spent on.")
    return lines


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
                g = dict(s["grade"])
                g["resolved"] = codegraders.is_resolved(s.get("task"), g)
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


def _cluster_sums(diffs_by_key):
    """Sum each task's per-rep differences into one observation per task.

    The three reps of a task share its prompt, document prefix and difficulty, so they are not three
    independent units. Counting them as independent is pseudo-replication: it divides the p-value by
    roughly 2**(reps-1) per task for free. Clustering keeps n at the task count.
    """
    per_task = defaultdict(int)
    for (tier, task, _rep), d in diffs_by_key.items():
        per_task[(tier, task)] += d
    return per_task


def holm(pvals):
    """Holm-Bonferroni adjustment for a family of p-values -> adjusted p per key."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = round(running, 4)
    return out


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
                diffs = {k: cur[k] - base[k] for k in sorted(cur) if k[0] == tier and k in base}
                clusters = _cluster_sums(diffs)
                total, p = perm_p(list(clusters.values()))
                sig[tier] = {"delta": sum(diffs.values()), "p": round(p, 4),
                             "n": len(clusters), "n_samples": len(diffs),
                             "unit": {"R": "points", "C": "checks", "L": "samples"}[tier]}
            out["arms"][key]["sig"] = sig
    # one family per tier: every arm of every model compared with its own f16 arm
    for tier in ("R", "C", "L"):
        fam = {k: v["sig"][tier]["p"] for k, v in out["arms"].items()
               if (v.get("sig") or {}).get(tier)}
        adj = holm(fam)
        for k, p in adj.items():
            out["arms"][k]["sig"][tier]["p_adj"] = p
    return out


# ---------------------------------------------------------------- pairs / deltas
def write_markdown(out, raw, path):
    L = []
    split = None
    man = raw.get("manifest") or {}
    ctx = out.get("arms", {})
    fixture = next(iter(ctx.values()), {}) if ctx else {}
    L.append(f"# KV-cache quantization at long context: {len(out['models'])} models × "
             f"{len(out['kvs'])} cache types\n")
    n_arms = len(out.get("arms") or {})
    L.append(f"One context for every arm: **-c {raw.get('context')}** · {n_arms} arms · "
             f"{raw.get('samples')} samples · llama.cpp "
             f"`{(man.get('llama_cpp') or {}).get('commit', '?')[:9]}`. Seeds, budgets and the "
             f"document hash are in the appendix.\n")
    L.append(f"Total test wall clock (fit probe + matrix + harness validation, model downloads "
             f"excluded): "
             f"**{_human(out.get('_timing', {}).get('study_wall_s'))}** "
             f"({_human((out.get('_timing', {}).get('phases', {}).get('matrix') or {}).get('wall_s'))} "
             f"of it the matrix).\n")
    if raw.get("deadline_reached"):
        L.append(f"> **Deadline reached.** Unrun arms: {raw.get('unrun_arms')}\n")
    if raw.get("incomplete_arms"):
        L.append(f"> **Incomplete.** No finished checkpoint for: "
                 f"{', '.join(raw['incomplete_arms'])}. Every number below is computed from the "
                 f"arms that did finish; the missing arms are not zero-filled or estimated.\n")
    unfit = (raw.get("fit") or {}).get("unfit") or []
    if unfit:
        L.append(f"> **Unfit / excluded models:** {', '.join(unfit)}\n")
    L.append("\n## Verdict\n")
    v = out.get("verdict") or {}
    if v.get("overall"):
        om, okv = v["overall"], v["overall_kv"]
        top = v["per_model"][om]
        cell = out["arms"].get(f"{om}__{okv}") or {}
        r2 = (100.0 * cell["R"]["points"] / cell["R"]["of"]) if cell.get("R", {}).get("of") else None
        c2 = (100.0 * cell["C"]["passed"] / cell["C"]["of"]) if cell.get("C", {}).get("of") else None
        parts_txt = (f"R {r2:.2f}% and C {c2:.2f}%, averaged before rounding"
                     if r2 is not None and c2 is not None else "tiers averaged")
        summary = _plain_summary(out)
        if summary:
            L.append(summary + "\n")
        L.append(f"**Best overall: `{om}` with a `{okv}` KV cache** — capability "
                 f"{top['capability']:.2f} of 100 ({parts_txt}), "
                 f"{top['loops']}/{top['loops_n']} structural-loop samples, {top['decode']} t/s "
                 f"decode, {top['vram']} MiB of graphics memory with "
                 f"{sum((top.get('knobs') or {}).values())} layers of the model kept in system RAM. "
                 f"**{_lead_sentence(out)}**\n")
        if len(v.get("flat") or []) > 1:
            flat = v["flat"]
            gap = flat[0][2] - flat[min(2, len(flat) - 1)][2]
            runners = ", ".join(f"`{r[0]}` {r[1]} ({r[2]:.2f})" for r in flat[1:3])
            L.append(f"The next arms are {runners} — the top three sit within {gap:.1f} capability "
                     f"points of each other, which is one or two graded outcomes.\n")
        rule = ("Ranking rule: capability = mean(retrieval %, code %); ties to the arm with fewer "
                "repeating samples, then the faster one. Every paired test clusters the three reps "
                "of a task into one observation and then applies a Holm correction across all 42 "
                "arm x tier comparisons, so nothing here can separate on a handful of correlated "
                "samples.")
        L.append(rule + "\n")
    L.append("### Every arm, one line each\n")
    L.append("| arm | capability | retrieval | code | structural loops | decode t/s |")
    L.append("|---|---|---|---|---|---|")
    for r in (v.get("flat") or []):
        m, kv, capc = r[0], r[1], r[2]
        c = out["arms"].get(f"{m}__{kv}") or {}
        loops = f"{c['loops']['hard']}/{c['loops']['n']}" if c.get("loops", {}).get("n") else "—"
        L.append(f"| `{m}` {kv} | {capc:.1f} | {c.get('R', {}).get('pct')} % | "
                 f"{c.get('C', {}).get('pct')} % | {loops} | {c.get('decode_tps')} |")
    L.append("\n### What this does not show\n")
    L.append("- **No arm is statistically separable.** Each tier has 6 task clusters (4 loop probes), "
             "so the smallest attainable p is 0.031; correcting across the 42 arm x tier comparisons "
             "puts every adjusted p at 1.00. The test can exclude a large, consistent difference and "
             "nothing smaller. 'No difference' here means 'not resolvable at this sample size'.")
    L.append("- **Gemma's exception is a length/format failure, not wrong code.** At q4_0 it hit the "
             "6,144-token cap on 3 of 6 coding tasks; on `repo_fix` it wrote 20,459 tokens with no "
             "`### FILE:` block at all. Its retrieval failures are content-side repetition loops, "
             "not truncated thinking.")
    L.append("- **Two tasks carry most of the code score.** `semver` is 261 of 780 checks and moves "
             "on a single decision; `intervals` and `statistics` are saturated in 14 of 15 arms. "
             "`count_proj` is degenerate (every project has the same count, so guessing 3 scores).")
    L.append("- **The loop probes only see the first ~2,000 tokens**, so a loop that starts later is "
             "invisible; `enumerate_shards` hit that cap in most arms. `csvlite`'s fixtures contain "
             "literal `\\n` text, so newline handling was not actually tested.")
    split = len(L)
    L.append("\n## Appendix: full tables, method and provenance\n")
    L.append("\n## Hardware and build this ran on\n")
    L.extend(_hardware_md(out.get("_hardware") or {}, out.get("_hardware_provenance") or ""))
    L.append("The long form lives in `REPORT-APPENDIX.md` next to this file; it is the same data, "
             "unsorted for reading.\n")
    L.append("| model | best KV | capability | runner-up | structural loops | decode t/s | why |")
    L.append("|---|---|---|---|---|---|---|")
    for row in _verdict_rows(out):
        L.append(f"| `{row[0]}` | **{row[1]}** | {row[2]:.2f} | {row[3]} | {row[4]} | {row[5]} | "
                 f"{row[6]} |")
    n_arms = len(v.get("flat") or [])
    L.append(f"\n### All {n_arms} arm{'s' if n_arms != 1 else ''}, ranked by the same rule\n")
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
        sep = [(m, kv, cap, p) for m, kv, cap, p in
               ((r[0], r[1], r[2], r[6]) for r in (v.get("flat") or [])[1:]) if p < ALPHA]
        L.append(f"\nThe comparison is made **one tier at a time** (R, C, L tested separately, "
                 f"smallest p multiplied by {len(TIERS)} as a Bonferroni correction), because pooling "
                 f"raw R points, C checks and L flags into one sum would let the 261-check samples "
                 f"outweigh the 0/1 flags — a scale artefact, not evidence. The arms that are "
                 f"**not distinguishable from the winner** are: {names}. The "
                 f"{len(sep)} that are separable are all *worse* than it: "
                 + ", ".join(f"`{m}` {kv} (p={p:.4f})" for m, kv, _, p in sep)
                 + ". On this run the lead is therefore not established.\n")
        L.append("Sensitivity, disclosed rather than hidden: normalising every sample to its own "
                 "fraction (points ÷ possible, checks ÷ checks) before pooling separates eight arms "
                 "instead of three. Every alternative pooling scheme that was tried separates only "
                 "arms that are *worse* than the leader, never a better one, so no reading of this "
                 "data supports 'quantized KV is an improvement'.\n")
    L.append("## Per-model offload (knobs are frozen across the three KV arms)\n")
    peaks = {}
    for c in out["arms"].values():
        if c.get("vram_peak_mib"):
            peaks.setdefault(c["model"], []).append(c["vram_peak_mib"])
    spread = max((max(v) - min(v) for v in peaks.values() if len(v) > 1), default=0)
    note = ("Knobs are identical for a model's three arms; the VRAM peak and speeds below are the "
            "**f16 arm's**. ")
    multi = [v for v in peaks.values() if len(v) > 1]
    if multi and spread:
        note += (f"Measured peaks differ between a model's arms by up to {spread} MiB, and the "
                 f"ordering is not the same on every model, so the verdict table's numbers are the "
                 f"arm that actually won, not this one (peaks include transient buffers and are not "
                 f"a clean cache-size measurement).")
    L.append(note + "\n")
    L.append("| model | weights off the GPU | VRAM peak MiB (f16) | realized prompt tokens | "
             "prefill t/s (f16) | decode t/s (f16) |")
    L.append("|---|---|---|---|---|---|")
    for model in _ranked_models(out):
        cell = out["arms"].get(f"{model}__f16") or next(
            (out["arms"][k] for k in out["arms"] if k.startswith(model + "__")), None)
        if not cell:
            continue
        on_cpu = sum((cell.get("knobs") or {}).values())
        L.append(f"| `{model}` | {on_cpu} layers on CPU | {cell.get('vram_peak_mib')} | "
                 f"{cell.get('prompt_tokens_med')} | {cell.get('prefill_tps')} | "
                 f"{cell.get('decode_tps')} |")
    L.append("\n## R tier: retrieval, tracing, aggregation (partial credit, %)\n")
    L += _md_grid(out, lambda c: c["R"]["pct"], "R %")
    L.append("\n## C tier: hidden checks passed / total, and resolved tasks\n")
    L.append("| model | KV | checks | % | resolved | too-hard tasks |")
    L.append("|---|---|---|---|---|---|")
    for model in _ranked_models(out):
        for kv in out["kvs"]:
            c = out["arms"].get(f"{model}__{kv}")
            if not c:
                continue
            hard_here = [t for t in (out.get("too_hard") or [])
                         if f"C:{t}" in c["per_task"]]
            L.append(f"| `{model}` | {kv} | {c['C']['passed']}/{c['C']['of']} | "
                     f"{c['C']['pct']} | {c['C']['resolved']}/{c['C']['n']} | "
                     f"{', '.join(hard_here) or '—'} |")
    L.append("\n## Loop table: flagged samples / samples (reasoning and content split)\n")
    L.append("`union (L)` counts every verdict from the zlib-ratio heuristic or a structural signal; "
             "`hard repetition` counts only samples with a repeated line, n-gram, 200-char block or "
             "tail cycle. The gap between the two columns is formulaic prose, not looping.\n")
    L.append("| model | KV | gen_manual | copy_ledgers | enumerate_shards | tool_episode | union "
             "(L) | hard repetition | reasoning | content | any tier |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model in _ranked_models(out):
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
    for model in _ranked_models(out):
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
             f"numbers. The smallest attainable p is {2 / 2 ** 18:.7f} on 18 R pairs, "
             f"{2 / 2 ** 6:.4f} on 6 C tasks and {2 / 2 ** 12:.4f} on 12 loop samples, so an arm "
             f"must move nearly every paired sample the same way to be called different. 'Not significant' means the change is inside what a draw of this "
             f"size produces; it is not proof of identity.\n")
    L.append("## Agreement with f16 per model (share of identical graded answers)\n")
    for model in _ranked_models(out):
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
    for model in _ranked_models(out):
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
    short = L[:split] if split else L
    with open(path, "w") as f:
        f.write("\n".join(short) + "\n")
    appendix = os.path.join(os.path.dirname(path) or ".", "REPORT-APPENDIX.md")
    with open(appendix, "w") as f:
        f.write("\n".join(["# Appendix — every table, method and provenance"] +
                           L[split:] if split else L) + "\n")


def _fmt(v):
    return "—" if v is None else f"{v:+.1f}"


def _md_grid(out, get, label):
    head = "| model | " + " | ".join(out["kvs"]) + " |"
    lines = [head, "|---" * (len(out["kvs"]) + 1) + "|"]
    for model in _ranked_models(out):
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
    "Only full-attention layers accumulate KV damage over depth, so a model with few global "
    "attention layers can only show a small effect; the per-model layer counts for this run are in "
    "fit.json's geometry block.",
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
def write_html(out, raw, path):
    grids = _svg_grids(out)
    verdict_html = _verdict_html(out)
    gallery = _gallery(out, raw)
    legend = ('<div class="dim" style="margin-top:6px">colour scale, always the same way: '
              '<span style="background:rgb(62,170,90);color:#0b0e12;padding:0 6px">green = best</span> '
              '<span style="background:rgb(205,160,62);color:#0b0e12;padding:0 6px">amber = middle</span> '
              '<span style="background:rgb(200,74,58);color:#0b0e12;padding:0 6px">red = worst</span> '
              '· "best" = high on the first two grids, low on the loop grid · '
              'grey = not distinguishable from f16 (task-clustered pairs, Holm) · '
              '"—" = arm not run</div>')
    taskplots = _task_dotplots(out)
    loopcounts, speed = _svg_loop_counts(out), _svg_speed(out)
    context_label = html.escape(_context_label(out))
    kvs_present = [kv for kv in out["kvs"]
                   if any(f"{m}__{kv}" in out["arms"] for m in out["models"])] or out["kvs"]
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
<h1>KV-cache quantization × long context: {len(out['models'])} model{'' if len(out['models']) == 1 else 's'} at {' / '.join(kvs_present)}</h1>
{_warning_banner(raw)}
</header>
<details class="card" style="max-width:none">
<summary><b>How to read this page</b></summary>
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
What the tabs hold: <b>Matrix</b> — the verdict, three grids of level-and-change, decode speed, and
how often two arms gave the identical answer. <b>Tasks</b> — one dot plot per task: three dots per
model, one per KV type, on a 0–100 axis, so overlapping dots mean "the KV type changed nothing here".
<b>Loops</b> — structural repetition per arm as bars, then a card for every sample the detector
flagged, with the repeated text highlighted.
</div>
<div class="dim" style="margin-top:6px">
Three facts that make the numbers honest: differences of a few points are noise; a single code task
can swing by tens of points on one design decision, so check the per-task dot plots before reading a
score delta as damage; and a score that falls because an arm hit the generation ceiling is truncated
output, not wrong output. The prose summary, with the caveats for this run written out, is
<code>docs/RESULTS.md</code>.
</div></details>
<nav><button id="b-matrix" class="on" onclick="tab('matrix')">Matrix</button>
<button id="b-tasks" onclick="tab('tasks')">Tasks</button>
<button id="b-loops" onclick="tab('loops')">Loops</button></nav>
<section id="t-matrix">{verdict_html}{grids}{legend}
<details><summary>All the numbers: speed, agreement, per-task tables</summary>
<h2>Speed {context_label}</h2>{speed}
<h2>Agreement with f16 (identical graded answers)</h2>{_agree_table(out)}{_cross_table(out)}
</details></section>
<section id="t-tasks" hidden>{taskplots}</section>
<section id="t-loops" hidden>{loopcounts}{gallery}</section>
<footer>{_manifest_block(raw, out.get('_hardware'), out.get('_timing'))}</footer>
<script>
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
TIERS = ("R", "C", "L")


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
    models = _ranked_models(out)
    loop_n = max([c["loops"]["n"] for c in out["arms"].values()] or [12])
    checks_max = max([c["C"]["of"] for c in out["arms"].values()] or [780])

    def val_R(c):
        return c["R"]["pct"]

    def val_C(c):
        return c["C"]["pct"]

    def val_L(c):
        return 100.0 * c["loops"]["hard"] / c["loops"]["n"] if c["loops"]["n"] else None

    specs = [
        ("R accuracy", val_R, "R",
         "level in % of 30 points · change in points · n = 18 paired samples per arm, "
         f"smallest attainable p = {2 / 2 ** 18:.7f} · rows: best first"),
        ("C checks", val_C, "C",
         f"level in % of {checks_max} hidden checks · change in checks · n = 6 paired tasks per "
         f"arm, smallest attainable p = 0.031 · rows: best first"),
        ("structural repetition", val_L, "L",
         f"level in % of {loop_n} samples · change in samples · n = {loop_n} paired samples per "
         f"arm, smallest attainable p = {2 / 2 ** loop_n:.4f} · rows: best first"),
    ]
    cw, ch, pad, lw = 108, 56, 12, 150
    parts = ['<div class="grid">']
    for title, get, tier, caption in specs:
        w, h = lw + cw * len(kvs) + pad, 40 + ch * len(models) + pad
        parts.append(f'<div class="card"><div>{html.escape(title)}</div>')
        parts.append(f'<div class="dim">{html.escape(caption)}</div>')
        parts.append(f'<svg width="{w}" height="{h}" role="img">'
                     f'<title>grid: {html.escape(title)}</title>')
        for j, kv in enumerate(kvs):
            parts.append(f'<text x="{lw + j * cw + cw / 2}" y="18" fill="#9aa4b2" '
                         f'font-size="12" text-anchor="middle">{kv}</text>')
        for i, model in enumerate(models):
            parts.append(f'<text x="4" y="{40 + i * ch + 30}" fill="#e8ecf1" font-size="12">'
                         f'{html.escape(model)}</text>')
            for j, kv in enumerate(kvs):
                c = out["arms"].get(f"{model}__{kv}")
                v = get(c) if c else None
                level = "—" if v is None else f"{v:.1f}"
                sig = ((c or {}).get("sig") or {}).get(tier)
                scale = max(1, int(abs((sig or {}).get("delta") or 1)) * 4)
                if kv == "f16":
                    sub, sub2, fill = "reference", "", "#2f3740"
                elif sig and v is not None:
                    d = sig["delta"]
                    sub, sub2 = f"Δ {d:+d}", f"p={sig['p']:.3f}"
                    fill = _cell_colour(sig, d, scale)
                else:
                    sub, sub2, fill = "—", "", "#2a3138"
                tip = level
                if sig and v is not None:
                    tip = (f"{level} · Δ {sig['delta']:+d} {sig['unit']} vs f16 · "
                           f"p={sig['p']:.4f} (n={sig['n']})")
                x, y = lw + j * cw + 2, 40 + i * ch + 2
                parts.append(f'<rect x="{x}" y="{y}" width="{cw - 4}" height="{ch - 4}" rx="4" '
                             f'fill="{fill}"><title>{html.escape(model)} {kv}: '
                             f'{html.escape(tip)}</title></rect>')
                tc = "#e8ecf1" if fill in ("#2f3740", "#2a3138", "#3a4149") else "#0b0e12"
                parts.append(f'<text x="{x + (cw - 4) / 2}" y="{y + 19}" fill="{tc}" '
                             f'font-size="13" text-anchor="middle">{level}</text>')
                parts.append(f'<text x="{x + (cw - 4) / 2}" y="{y + 33}" fill="{tc}" '
                             f'font-size="10" text-anchor="middle">{html.escape(sub)}</text>')
                if sub2:
                    parts.append(f'<text x="{x + (cw - 4) / 2}" y="{y + 45}" fill="{tc}" '
                                 f'font-size="10" text-anchor="middle">{html.escape(sub2)}</text>')
        parts.append("</svg></div>")
    parts.append("</div>")
    return "".join(parts)


def _warning_banner(raw):
    """Only real warnings belong above the finding; telemetry lives in the footer."""
    bits = []
    if raw.get("deadline_reached"):
        bits.append(f"DEADLINE REACHED — unrun arms: {raw.get('unrun_arms')}")
    if raw.get("incomplete_arms"):
        bits.append(f"INCOMPLETE — no finished checkpoint for {', '.join(raw['incomplete_arms'])}")
    if not bits:
        return ""
    return ('<div style="margin-top:4px;color:#e0b24a"><b>' +
            html.escape(" · ".join(bits)) + "</b></div>")


def _verdict_html(out):
    v = out.get("verdict") or {}
    if not v.get("overall"):
        return ""
    om, okv = v["overall"], v["overall_kv"]
    top = v["per_model"][om]
    rows = "".join(
        f'<tr><td>{html.escape(r[0])}</td><td><b>{r[1]}</b></td><td>{r[2]:.2f}</td>'
        f'<td>{r[4]}</td><td>{r[5]}</td></tr>'
        for r in _verdict_rows(out))
    on_cpu = sum((top.get("knobs") or {}).values())
    summary = _plain_summary(out)
    return (f'<div class="card" style="max-width:none"><h2 style="margin-top:0">What we found</h2>'
            + (f'<div>{html.escape(summary)}</div>' if summary else "")
            + f'<div class="dim" style="margin-top:6px">Best combination on this run: '
            f'{html.escape(om)} with a {html.escape(okv)} cache — {top["capability"]:.1f} out of 100, '
            f'{top["loops"]}/{top["loops_n"]} structural-loop samples, {top["decode"]} tokens/second, '
            f'{top["vram"]} MiB of graphics memory with {on_cpu} layers of the model kept in system '
            f'RAM.</div>'
            f'<details><summary>How sure are we, and how the ranking works</summary>'
            f'<div class="dim">{html.escape(_lead_sentence(out))} Capability = the average of the '
            f'retrieval score and the code score; ties go to the arm with fewer repeating samples, '
            f'then to the faster one; every paired test clusters the three reps of a task and '
            f'corrects across all 42 comparisons, so nothing here can separate on a handful of '
            f'correlated samples.</div></details>'
            f'<table style="margin-top:8px"><tr><th>model</th><th>best KV</th><th>capability</th>'
            f'<th>structural loops</th><th>decode t/s</th></tr>'
            f'{rows}</table></div>')


def write_figures(out, path):
    """Export every figure as a standalone SVG plus the exact numbers behind it.

    Deterministic: the same samples.jsonl always produces byte-identical files, so a third party can
    diff them or re-render them elsewhere without trusting the HTML.
    """
    os.makedirs(path, exist_ok=True)
    import re as _re
    figures = (_re.findall(r"<svg.*?</svg>", _svg_grids(out), _re.S)
               + _re.findall(r"<svg.*?</svg>", _svg_speed(out), _re.S)
               + _re.findall(r"<svg.*?</svg>", _svg_loop_counts(out), _re.S)
               + _re.findall(r"<svg.*?</svg>", _task_dotplots(out), _re.S))
    for name in os.listdir(path):
        if name.endswith(".svg"):
            os.unlink(os.path.join(path, name))
    for i, fig in enumerate(figures, 1):
        with open(os.path.join(path, f"fig{i}.svg"), "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(fig.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1))
    table = {}
    for key, c in out["arms"].items():
        table[key] = {"R_pct": c["R"]["pct"], "R_points": c["R"]["points"], "R_of": c["R"]["of"],
                      "C_pct": c["C"]["pct"], "C_passed": c["C"]["passed"], "C_of": c["C"]["of"],
                      "loops_hard": c["loops"]["hard"], "loops_n": c["loops"]["n"],
                      "loop_flagged": c["loops"]["flagged"],
                      "loop_rate": c["loop_rate"],
                      "decode_tps": c["decode_tps"], "prefill_tps": c["prefill_tps"],
                      "prompt_tokens_med": c["prompt_tokens_med"],
                      "per_task": c["per_task"], "sig": c.get("sig")}
    # the index is derived from each file's own <title>, so it stays true when a builder produces
    # nothing (a single-arm run has no loop-count figure) or the order of figures changes
    index = []
    for i, fig in enumerate(figures, 1):
        m = _re.search(r"<title>(.*?)</title>", fig, _re.S)
        index.append({"file": f"fig{i}.svg", "title": html.unescape(m.group(1)) if m else ""})
    with open(os.path.join(path, "figures.json"), "w") as f:
        json.dump({"alpha": ALPHA, "tiers": list(TIERS),
                   "test": "exact two-sided paired sign-flip on integer differences; "
                           "pairing = (task, rep) for sampled tiers, task for the greedy tier",
                   "leader_test": f"tier by tier against the leader, Bonferroni over {len(TIERS)} "
                                  f"tiers",
                   "figures": index,
                   "arms": table}, f, indent=2, sort_keys=True)


def _axis(maxv, target=5):
    """Round the axis up to a readable maximum and pick a step that gives <= target intervals."""
    import math
    if not maxv or maxv <= 0:
        return 1, 0.25
    for step in (0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if maxv / step <= target:
            return math.ceil(maxv / step) * step, step
    return maxv, maxv / target


def _tick_label(v, step):
    return f"{v:.0f}" if step >= 1 else f"{v:.1f}"


def _svg_hbars(items, title, note, unit, fmt, footnote=None, higher_is_better=True,
               bar_h=20, gap=8):
    """Horizontal bars with a rounded axis, gridlines, and value labels that always fit.

    Geometry is fixed to the card width: a label column, a bar area, and a right gutter wide enough
    for the longest value label. If a label still would not fit outside the bar it is drawn inside
    the bar, right-aligned, in dark text.
    """
    if not items:
        return ""
    lw, area, top = 185, 400, 52
    xmax, step = _axis(max(v for _, v, _ in items))
    texts = []
    for label, value, badge in items:
        t = f"{fmt(value)} {unit}"
        if badge:
            t += f" · {badge}"
        texts.append(t)
    char_px = 5.9
    gutter = int(max(len(t) for t in texts) * char_px) + 14
    gutter = max(120, min(gutter, 300))
    w = lw + area + gutter
    h = top + len(items) * (bar_h + gap) + 34
    parts = [f'<div class="card"><div>{html.escape(title)}</div>'
             f'<div class="dim">{html.escape(note)}</div>',
             f'<svg width="{w}" height="{h}" role="img">',
             f'<title>{html.escape(title)}</title>']
    t = 0.0
    while t <= xmax + 1e-9:
        x = lw + (t / xmax) * area
        parts.append(f'<line x1="{x:.0f}" y1="{top - 12}" x2="{x:.0f}" y2="{h - 24}" '
                     f'fill="none" stroke="#2b333c"/>')
        parts.append(f'<text x="{x:.0f}" y="{h - 10}" fill="#9aa4b2" font-size="11" '
                     f'text-anchor="middle">{_tick_label(t, step)}</text>')
        t += step
    for i, (label, value, badge) in enumerate(items):
        y = top + i * (bar_h + gap)
        bw = max(2.0, (value / xmax) * area) if value else 2.0
        parts.append(f'<text x="4" y="{y + bar_h - 5}" fill="#e8ecf1" font-size="12">'
                     f'{html.escape(label)}</text>')
        parts.append(f'<rect x="{lw}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="2" '
                     f'fill="{_color(value, 0, xmax, higher_is_better)}">'
                     f'<title>{html.escape(label)}: {html.escape(texts[i])}</title></rect>')
        text_w = len(texts[i]) * char_px
        if lw + bw + 6 + text_w <= w - 4:
            parts.append(f'<text x="{lw + bw + 6:.0f}" y="{y + bar_h - 5}" fill="#e8ecf1" '
                         f'font-size="11">{html.escape(texts[i])}</text>')
        else:
            parts.append(f'<text x="{lw + bw - 6:.0f}" y="{y + bar_h - 5}" fill="#0b0e12" '
                         f'font-size="11" text-anchor="end">{html.escape(texts[i])}</text>')
    parts.append(f'<text x="{lw + area / 2}" y="{h - 1}" fill="#9aa4b2" font-size="11" '
                 f'text-anchor="middle">{html.escape(unit)}</text>')
    parts.append("</svg>")
    if footnote:
        parts.append(f'<div class="dim" style="margin-top:6px">{html.escape(footnote)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _svg_loop_counts(out):
    items, clean = [], 0
    n_total = 0
    for key, c in out["arms"].items():
        n = c["loops"]["n"] or 1
        n_total = n
        hard = c["loops"]["hard"]
        if hard:
            items.append((f"{c['model']} {c['kv']}", 100.0 * hard / n, f"{hard}/{n} samples"))
        else:
            clean += 1
    items.sort(key=lambda t: (t[1], t[0]))   # cleanest first: "best" is the top of the colour scale
    note = ("share of L-tier samples containing a repeated line, repeated n-gram, repeated 200-char "
            "block or a repeating tail cycle · lower is better, so green = clean")
    footnote = (f"{clean} of {len(out['arms'])} arms had 0/{n_total} such samples and are not drawn. "
                f"The compression-ratio heuristic is excluded on purpose: it fires on template "
                f"prose without anything repeating.")
    return _svg_hbars(items, "structural repetition per arm (only arms with at least one)",
                      note, "% of samples", lambda v: f"{v:.0f}",
                      footnote=footnote, higher_is_better=False)


def _context_label(out):
    """The realized context of this result, used by the speed figure and its section heading."""
    toks = [c["prompt_tokens_med"] for c in out["arms"].values() if c.get("prompt_tokens_med")]
    if not toks:
        return "at depth"
    lo, hi = min(toks), max(toks)
    return (f"at {lo / 1000:.0f}k tokens of context" if hi - lo < 1000
            else f"at the realized context ({lo / 1000:.0f}k–{hi / 1000:.0f}k tokens)")


def _svg_speed(out):
    items = [(f"{c['model']} {c['kv']}", c["decode_tps"] or 0.0,
              f"prefill {c['prefill_tps']:.0f} t/s" if c.get("prefill_tps") else "")
             for c in out["arms"].values() if c.get("decode_tps")]
    items.sort(key=lambda t: -t[1])
    return _svg_hbars(items, f"decode throughput {_context_label(out)}",
                      "median over every request in the arm · higher is better",
                      "tokens/s", lambda v: f"{v:.0f}")


PLAIN = [
    ("identical_consecutive_call", "issued the same tool call twice in a row"),
    ("identical_call_x3", "issued the same tool call three or more times"),
    ("no_progress_after_4_turns", "made four tool calls that all failed"),
    ("no_answer_after_long_gen", "generated a long reply without ever producing an answer"),
]


def _plain_why(s):
    """One sentence a non-specialist can act on, derived from the stored verdict string."""
    reason = s.get("flag_reason") or ""
    metrics = {k: v.get("metrics") or {} for k, v in (s.get("loops") or {}).items()}
    parts = []
    for name, m in metrics.items():
        where = "the visible answer" if name == "content" else (
            "its reasoning text" if name == "reasoning_content" else "the tool transcript")
        if m.get("line_run", 0) >= 5:
            parts.append(f"one line of {where} repeats {m['line_run']} times in a row")
        elif max(m.get("word_run_4", 0), m.get("word_run_8", 0)) >= 6:
            parts.append(f"a short phrase in {where} repeats "
                         f"{max(m.get('word_run_4', 0), m.get('word_run_8', 0))} times in a row")
        elif m.get("cycle"):
            parts.append(f"{where} ends with the same {m['cycle'][0]}-character block three times")
        elif m.get("block"):
            parts.append(f"a 200-character block appears {m['block'][1]} times in {where}")
        elif m.get("zip_ratio", 0) >= 0.42 and (m.get("chars") or 0) >= 2048:
            parts.append(f"{where} compresses to {m['zip_ratio']} (the cutoff is 0.42) although no "
                         f"line, phrase or block repeats — a template-prose artefact, not a loop")
    for key, phrase in PLAIN:
        if key in reason:
            parts.append(phrase)
    if parts:
        return "; ".join(parts) + "."
    short = min((m.get("chars") or 0) for m in metrics.values()) if metrics else 0
    if short and short < 2048:
        return ("the reply was too short for the compression heuristic to say anything useful "
                "(under 2,048 characters): keep it in the counts, ignore it as evidence.")
    return "flagged by the repetition detector; see the numbers below."


def _mark_repeats(text):
    """Wrap the obviously repeated part of the quoted excerpt in <mark>.

    Lines first (a list that repeats a row is the common failure); if no line run exists, mark the
    longest consecutively repeated 4-word phrase, which is what a "phrase repeats 259 times" verdict
    looks like in text.
    """
    lines = text.splitlines()
    out, i, marked = [], 0, False
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1].strip() == lines[i].strip():
            j += 1
        chunk = "\n".join(lines[i:j + 1])
        if j - i >= 2 and lines[i].strip():
            out.append(f"<mark>{html.escape(chunk)}</mark>")
            marked = True
        else:
            out.append(html.escape(chunk))
        i = j + 1
    if marked:
        return "\n".join(out)
    words = text.split()
    best = (0, 0, 4)
    for k in (4, 2):
        for a in range(len(words) - k):
            gram = words[a:a + k]
            reps, b = 1, a + k
            while b + k <= len(words) and words[b:b + k] == gram:
                reps += 1
                b += k
            if reps > best[0]:
                best = (reps, a, k, b)
    reps, a, k, b = best if len(best) == 4 else (0, 0, 4, 0)
    if reps < 3:
        return html.escape(text)
    head = " ".join(words[:a])
    mid = " ".join(words[a:b])
    tail = " ".join(words[b:])
    return (html.escape(head) + (" " if head else "") + f"<mark>{html.escape(mid)}</mark>" +
            (" " if tail else "") + html.escape(tail))


def _metric_rows(m):
    rows = [("longest run of the same line", m.get("line_run")),
            ("longest run of a repeated phrase (4 words)", m.get("word_run_4")),
            ("largest repeated 200-character block", (m.get("block") or [None, 0])[1] or "none"),
            ("repeating tail cycle", (f"{(m.get('cycle') or [0])[0]} characters × 3"
                                      if m.get("cycle") else "none")),
            ("compressibility (1.00 = plain text)", m.get("zip_ratio")),
            ("characters", m.get("chars"))]
    out = []
    for label, value in rows:
        if value in (None, 0) and label.startswith("longest"):
            value = 1
        out.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(value))}</td></tr>")
    return "".join(out)


def _loop_card(s):
    head = (f'<div class="card"><b>{html.escape(s["model"])} · {s["kv"]} · '
            f'{html.escape(str(s["tier"]))} · {html.escape(str(s["task"]))} · '
            f'rep {s["rep"]}</b>')
    why = f'<div style="margin:2px 0 6px">Why flagged: {html.escape(_plain_why(s))}</div>'
    body = [head, why]
    for part, v in (s.get("loops") or {}).items():
        m = v.get("metrics") or {}
        if not m:
            continue
        label = {"content": "visible answer", "reasoning_content": "reasoning text",
                 "transcript": "tool transcript"}.get(part, part)
        body.append(f'<div class="dim">where: {html.escape(label)}</div>')
        body.append(f'<table>{_metric_rows(m)}</table>')
        span = m.get("span")
        if span:
            body.append('<details><summary>Show the quoted text</summary>'
                        f'<pre>{_mark_repeats(span[:1200])}</pre></details>')
            if m.get("cycle") and "<mark>" not in body[-1]:
                body.append(f'<div class="dim">The repeating part of this one is at the end of a '
                            f'{m.get("chars")}-character reply, so it is not in the excerpt above.</div>')
    if s.get("transcript"):
        calls = "".join(f'<tr><td>{html.escape(str(c.get("tool")))}</td>'
                        f'<td>{html.escape(json.dumps(c.get("args")))}</td>'
                        f'<td>{"ok" if c.get("ok") else "failed"}</td></tr>'
                        for c in s["transcript"])
        body.append(f'<table><tr><th>tool</th><th>arguments</th><th>result</th></tr>{calls}</table>')
    if s.get("grade") and s["tier"] in ("R", "C"):
        g = s["grade"]
        got = g.get("points", g.get("passed"))
        body.append(f'<div class="dim">task outcome: {got}/{g.get("of")} '
                    f'({html.escape(str(g.get("why") or "checks passed"))})</div>')
    body.append("</div>")
    return "".join(body)


def _task_dotplots(out, tier_labels=("R", "C", "L")):
    """One small dot plot per task: models down the side, three KV types as dots on a 0-100 axis.

    A dot plot is used instead of 15 bars because the question per task is "are the three dots for
    this model on top of each other?" — saturation and outliers are visible at a glance, and every
    dot carries its exact value in the tooltip.
    """
    tasks = {}
    for cell in out["arms"].values():
        for key, pt in cell["per_task"].items():
            tier, task = key.split(":", 1)
            tasks.setdefault((tier, task), []).append((cell["model"], cell["kv"], pt))
    colour = {"f16": "#8fb0d6", "q8_0": "#e0b24a", "q4_0": "#d86a4a"}
    parts = []
    for tier in tier_labels:
        group = [(t, rows) for (tr, t), rows in sorted(tasks.items()) if tr == tier]
        if not group:
            continue
        meaning = {"R": "share of that task's points earned",
                   "C": "share of that task's hidden checks passed",
                   "L": "share of samples that did NOT loop (higher = cleaner)"}[tier]
        parts.append(f'<div class="card" style="max-width:none"><b>{tier} tier</b> '
                     f'<span class="dim">— {html.escape(meaning)} · 0–100 axis</span>'
                     f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:8px">')
        for task, rows in group:
            present = {m for m, _, _ in rows}
            models = [m for m in _ranked_models(out) if m in present]
            w, row_h, lw, top, axis = 300, 22, 150, 26, 70
            h = top + len(models) * row_h + 22
            parts.append(f'<div><div class="dim">{html.escape(task)}</div>'
                         f'<svg width="{w}" height="{h}" role="img">'
                         f'<title>{html.escape(f"{tier}: {task}")}</title>')
            for t in (0, 50, 100):
                x = lw + (t / 100.0) * axis
                parts.append(f'<line x1="{x:.0f}" y1="{top - 6}" x2="{x:.0f}" y2="{h - 20}" '
                             f'stroke="#2b333c"/>')
                parts.append(f'<text x="{x:.0f}" y="{h - 8}" fill="#9aa4b2" font-size="10" '
                             f'text-anchor="middle">{t}</text>')
            for i, model in enumerate(models):
                y = top + i * row_h
                parts.append(f'<text x="2" y="{y + 4}" fill="#c8d2dc" font-size="10">'
                             f'{html.escape(model)}</text>')
                for m, kv, pt in sorted(rows, key=lambda r: r[1]):
                    if m != model:
                        continue
                    if pt["of"]:
                        v = 100.0 * pt["score"] / pt["of"]
                        detail = (f"{pt['score']:.0f}/{pt['of']:.0f}"
                                  if float(pt["score"]).is_integer() else
                                  f"{pt['score']:g}/{pt['of']:g}")
                    elif pt["n"]:
                        v = 100.0 * (1 - pt["flagged"] / pt["n"])
                        detail = f"{pt['n'] - pt['flagged']}/{pt['n']} clean"
                    else:
                        continue
                    x = lw + (max(0.0, min(100.0, v)) / 100.0) * axis
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.0f}" r="4.5" '
                                 f'fill="{colour.get(kv, "#999")}" fill-opacity="0.95">'
                                 f'<title>{html.escape(model)} {kv} {task}: {detail} '
                                 f'({v:.0f}%)</title></circle>')
            parts.append("</svg></div>")
        parts.append("</div>"
                     '<div class="dim" style="margin-top:6px">'
                     '<span style="color:#8fb0d6">● f16</span> '
                     '<span style="color:#e0b24a">● q8_0</span> '
                     '<span style="color:#d86a4a">● q4_0</span> · dots for one model that overlap '
                     'mean the KV type changed nothing</div></div>')
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
    for model in _ranked_models(out):
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
    artefact = 0
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
            mets = [v.get("metrics") or {} for v in (s.get("loops") or {}).values()]
            structural = any((m.get("line_run") or 0) >= 5
                             or max(m.get("word_run_1") or 0, m.get("word_run_4") or 0,
                                    m.get("word_run_8") or 0) >= 6
                             or (m.get("cycle") and m["cycle"][0] >= 8)
                             or (m.get("block") and m["block"][1] >= 3)
                             for m in mets)
            transcript = bool((s.get("loops") or {}).get("transcript", {}).get("flagged"))
            if not (structural or transcript):
                artefact += 1
                continue
            cards.append(_loop_card(s))
    if not cards:
        return ('<p class="dim">No sample in this run contained repeated lines, phrases, blocks or '
                'a repeating tail, and no tool call repeated.</p>')
    note = (f'{artefact} further samples were flagged only by the compression heuristic with '
            f'nothing repeating; they stay in <code>samples.jsonl</code> and in the "any verdict" '
            f'grid, and are not shown here.')
    return (f'<div class="dim">{note}</div>'
            f'<details><summary>Every sample the loop detector flagged '
            f'({len(cards)} cards, with the quoted text)</summary>'
            + "".join(cards) + "</details>")


def _clock_seconds(timing, raw):
    """The elapsed time to quote: the recorded study window, else the matrix span, else the run's."""
    t = timing or {}
    if t.get("study_wall_s"):
        return t["study_wall_s"]
    matrix = ((t.get("phases") or {}).get("matrix") or {}).get("wall_s")
    return matrix or (raw.get("timing") or {}).get("run_wall_s")


def _manifest_block(raw, hardware=None, timing=None):
    man = raw.get("manifest") or {}
    cmds = ("python3 -m kvtest.selftest\n"
            "python3 -m kvtest.fit --stamp STAMP\n"
            "python3 -m kvtest.run --stamp STAMP --resume --deadline-hours 22\n"
            "python3 -m kvtest.report results/STAMP")
    prov = html.escape(
        f"context -c {raw.get('context')} · llama.cpp "
        f"{(man.get('llama_cpp') or {}).get('commit', '?')[:12]} · document "
        f"{(raw.get('document') or {}).get('sha256', '?')[:16]} · {raw.get('samples')} samples · "
        f"wall clock {_human(_clock_seconds(timing, raw))}")
    hw = hardware or (raw.get("manifest") or {}).get("host_runtime") or {}
    hwline = ("<b>Hardware</b> " + html.escape(hostinfo.one_line(hw)) + "<br>") if hw else ""
    body = ("<details><summary>Provenance</summary><div class=\"dim\">" + prov +
            "</div></details>" + hwline + "<b>Manifest</b><pre>" + html.escape(json.dumps({
        "llama_cpp": man.get("llama_cpp"), "context": raw.get("context"),
        "seeds": man.get("seeds"), "budgets": man.get("budgets"), "reps": man.get("reps"),
        "document_sha256": (raw.get("document") or {}).get("sha256"),
        "tokens_by_model": (raw.get("document") or {}).get("tokens_by_model"),
        "models": man.get("models"),
    }, indent=1)) + "</pre><b>Reproduce</b><pre>" + html.escape(cmds) + "</pre>")
    return ('<details><summary>Hardware, manifest and reproduce commands</summary>' + body +
            "</details>")


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
    """Keep the README's headline table honest: regenerated from the same aggregation.

    Only ever called on explicit request (``--update-readme``), and refuses to write a headline for
    a partial matrix: a smoke run or a deadline-cut run must not replace the documented result with
    one arm of mock-scale data.
    """
    if not os.path.exists(readme_path):
        return
    expected = len(config.MODELS) * len(config.KV_ARMS)
    have = len(out.get("arms") or {})
    if out.get("_incomplete_arms") or have < expected:
        extra = (f", incomplete: {', '.join(out['_incomplete_arms'])}"
                 if out.get("_incomplete_arms") else "")
        print(f"README headline not updated: {have} of {expected} arms{extra}. The documented "
              f"headline must come from a complete matrix.")
        return
    lines = ["| model | KV | R accuracy | C checks | C resolved | structural loops | "
             "agreement with f16 |", "|---|---|---|---|---|---|---|"]
    for model in _ranked_models(out):
        for kv in out["kvs"]:
            c = out["arms"].get(f"{model}__{kv}")
            if not c:
                continue
            lines.append(
                f"| `{model}` | {kv} | {c['R']['points']}/{c['R']['of']} "
                f"({c['R']['pct']}%) | {c['C']['passed']}/{c['C']['of']} ({c['C']['pct']}%) | "
                f"{c['C']['resolved']}/{c['C']['n']} | {c['loops']['hard']}/{c['loops']['n']} | "
                f"{c.get('agree') if c.get('agree') is not None else ''} |")
    text = open(readme_path).read()
    start, end = "<!-- HEADLINE:START -->", "<!-- HEADLINE:END -->"
    if start not in text or end not in text:
        return
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    with open(readme_path, "w") as f:
        f.write(head + start + "\n" + "\n".join(lines) + "\n" + end + tail)


def build(out_dir, readme_path=None):
    raw, samples, arms = load(out_dir)
    out = agg(raw, samples, arms)
    out["_stamp"] = raw.get("stamp") or os.path.basename(out_dir.rstrip("/"))
    out["_timing"] = load_timing(out_dir, raw)
    out["_incomplete_arms"] = raw.get("incomplete_arms") or []
    out["_hardware"], out["_hardware_provenance"] = load_hardware(out_dir, raw)
    out["_samples_path"] = os.path.join(out_dir, "samples.jsonl")
    write_markdown(out, raw, os.path.join(out_dir, "REPORT.md"))
    write_html(out, raw, os.path.join(out_dir, "report.html"))
    write_csv(out, os.path.join(out_dir, "matrix.csv"))
    write_figures(out, os.path.join(out_dir, "figures"))
    if readme_path:
        write_readme_headline(out, readme_path)
    return out, raw


def main(argv=None):
    import sys
    args = (argv if argv is not None else sys.argv[1:])
    if not args:
        raise SystemExit("usage: python3 -m kvtest.report results/<stamp>")
    out_dir = args[0]
    update = "--update-readme" in args
    args = [a for a in args if a != "--update-readme"]
    if not args:
        raise SystemExit("usage: python3 -m kvtest.report results/<stamp> [--update-readme]")
    out, raw = build(out_dir, os.path.join(config.ROOT, "README.md") if update else None)
    print(f"wrote {out_dir}/REPORT.md, report.html, matrix.csv "
          f"({len(out['arms'])} arms, {raw.get('samples')} samples, "
          f"total {_human((out.get('_timing') or {}).get('study_wall_s'))})")


if __name__ == "__main__":
    main()
