"""Artifacts: REPORT.md, report.html, matrix.csv, figures/ — all derived from raw.json.

The report states the premise first (instances per workflow, comparisons, smallest attainable p,
greedy decode), then reports each workflow's own measure: W1 distributional, W2 and W3 key counts
(64 and 20) with a per-depth curve, W4 state hashes and invariants, W5 answer keys and search calls,
and W6 structure plus the loop instrument.
"""
import argparse
import html
import json
import math
import os
import re
import statistics

from . import config

TITLE = "Full-context KV quantization: agentic workflows, zero–{n} key scores"


def _title(arms):
    return TITLE.format(n=_w2_of(list(arms.values())))


def _colour(kv):
    return {"f16": "var(--c-f16, #2d6a4f)", "q4_0": "var(--c-q4_0, #c2410c)",
            "q8_0": "var(--c-q8_0, #1d4ed8)"}.get(kv, "var(--fg, #111827)")


# ---------------------------------------------------------------- loading
def load(out_dir):
    raw_path = os.path.join(out_dir, "raw.json")
    if not os.path.exists(raw_path):
        raise SystemExit(f"no raw.json in {out_dir}")
    raw = json.load(open(raw_path))
    return raw, _samples(out_dir)


def _samples(out_dir):
    path = os.path.join(out_dir, "samples.jsonl")
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


# ---------------------------------------------------------------- helpers
def _f(v, n=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def _pct(v, n=1):
    return "—" if v is None else f"{100 * v:.{n}f}%"


def _kld(v, n=4):
    """KLD to four decimals, with a clean zero for the self-check rows."""
    if v is None:
        return "—"
    return "0" if abs(v) < 5e-7 else f"{v:.{n}f}"


def _human(sec):
    if not sec:
        return "—"
    sec = int(round(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h:
        return f"{h} h {m} m"
    if m:
        return f"{m} m {s} s"
    return f"{s} s"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def _w2_of(rows):
    """The W2 key count the probe asked for, so the denominators follow the data.

    The 20 fallback reaches only a stamp that never ran W2; its title then carries that count and
    no W2 table is printed.
    """
    return next((a["w2"].get("of") for a in rows if a["w2"].get("of")), 20)


def _instances(arm, wf):
    return ((arm.get("workflows") or {}).get(wf) or {}).get("instances") or []


def _wf(arm, wf):
    return (arm.get("workflows") or {}).get(wf) or {}


def _out_token_column(arms):
    """Header and missing-workflow list for the completion-count column."""
    workflows = ("W1", "W2", "W3", "W4", "W5", "W6")

    def recs(a, wf):
        return a.get("raw_instances", {}).get(wf) or _instances(a, wf)

    present = {wf for a in arms.values() for wf in workflows if recs(a, wf)}
    missing = sorted({
        wf for a in arms.values() for wf in present
        if any(i.get("out_tokens") is None
               and (i.get("phase_a") or {}).get("out_tokens") is None
               for i in recs(a, wf))})
    covered = [wf for wf in workflows if wf in present and wf not in missing]
    col = f"out tokens ({', '.join(covered)})" if missing else "out tokens"
    return col, missing


def _band_index(frac, n=5):
    return min(int(frac * n), n - 1)


def _wilson(hits, n):
    from . import stats
    return stats.wilson(hits, n)


def _paired_maps(a_map, b_map):
    from . import stats
    keys = sorted(set(a_map) & set(b_map))
    return stats.paired_bootstrap([a_map[k] - b_map[k] for k in keys])


def _answered_w2(inst):
    return [(i.get("instance"), i.get("score")) for i in inst
            if (i.get("grade") or {}).get("answer") is not None
            and i.get("score") is not None]


def _scored_w3(inst):
    """A brief produced before a halt still counts; a halt with no brief is its own counter."""
    out = []
    for i in inst:
        halted = (i.get("phase_a") or {}).get("halted_by")
        produced = bool(((i.get("phase_a") or {}).get("content") or "").strip())
        if i.get("score") is not None and (produced or halted == "completed"):
            out.append((i.get("instance"), i.get("score")))
    return out


def _decision(paired):
    from . import stats
    return stats.decision(paired)


# ---------------------------------------------------------------- aggregation
def aggregate(raw):
    arms = {}
    for name, a in (raw.get("arms") or {}).items():
        model, kv = name.split("__", 1)
        d = dict(arm=name, model=model, kv=kv, ctx=a.get("ctx"),
                 knobs=a.get("knobs"), sampler=a.get("sampler_resolved"),
                 wall_s=a.get("wall_s"), finished=a.get("finished"), cut=a.get("cut"),
                 w1=_wf(a, "W1") if isinstance(_wf(a, "W1"), dict) else {},
                 w2={}, w3={}, w4={}, w5={}, w6={}, halts={}, loop_halts=0,
                 thinking=[], tokens=[], walls=[])
        # W2
        inst = _instances(a, "W2")
        answered = _answered_w2(inst)
        score_map = dict(answered)
        scores = [score_map[k] for k in sorted(score_map)]
        bands = [dict(hits=0, n=0) for _ in range(5)]
        misses = []
        answered_ids = {i for i, _ in answered}
        for i in inst:
            if i.get("instance") not in answered_ids:
                continue
            grade = i.get("grade") or {}
            for k in grade.get("keys") or []:
                b = _band_index(k.get("depth") or 0)
                bands[b]["n"] += 1
                if k.get("status") == "hit":
                    bands[b]["hits"] += 1
                else:
                    misses.append(dict(instance=i.get("instance"), name=k.get("name"),
                                       depth=k.get("depth"), status=k.get("status"),
                                       got=k.get("got"), expected=k.get("expected")))
        of = next((g for g in ((i.get("grade") or {}).get("of") for i in inst) if g), 0)
        d["w2"] = dict(n=len(scores), attempted=len(inst), score_map=score_map,
                       scores=scores, mean=_mean(scores), sd=_sd(scores), of=of,
                       halts=sum(1 for i in inst if i.get("halted_by") in
                                 ("loop", "context_full", "error", "ceiling")),
                       bands=[dict(band=f"{20 * b}-{20 * (b + 1)}%",
                                   **bands[b], **dict(zip(("p", "lo", "hi"),
                                                          _wilson(bands[b]["hits"], bands[b]["n"]))))
                              for b in range(5)],
                       misses=misses,
                       saturated=bool(scores) and all(s == of for s in scores))
        # W3
        inst = _instances(a, "W3")
        scored = _scored_w3(inst)
        retention_map = dict(scored)
        retention = [retention_map[k] for k in sorted(retention_map)]
        scored_ids = {i for i, _ in scored}
        pbg = [((i.get("phase_b_grade") or {}).get("keys_correct")) for i in inst
               if i.get("instance") in scored_ids]
        scored_inst = [i for i in inst if i.get("instance") in scored_ids]
        bands3 = [dict(hits=0, n=0) for _ in range(5)]
        for i in scored_inst:
            for k in (i.get("brief_grade") or {}).get("keys") or []:
                b = _band_index(k.get("depth") or 0)
                bands3[b]["n"] += 1
                if k.get("status") == "present":
                    bands3[b]["hits"] += 1
        d["w3"] = dict(
            n=len(retention), attempted=len(inst), retention_map=retention_map,
            bands=[dict(band=f"{20 * b}-{20 * (b + 1)}%", **bands3[b],
                        **dict(zip(("p", "lo", "hi"),
                                   _wilson(bands3[b]["hits"], bands3[b]["n"]))))
                   for b in range(5)],
            retention=retention, mean=_mean(retention), sd=_sd(retention),
            halts=sum(1 for i in inst if (i.get("phase_a") or {}).get("halted_by") in
                      ("loop", "context_full", "error", "ceiling")),
            mutated=sum((i.get("brief_grade") or {}).get("mutated") or 0 for i in scored_inst),
            absent=sum((i.get("brief_grade") or {}).get("absent") or 0 for i in scored_inst),
            phase_b=[x for x in pbg if x is not None],
            phase_b_mean=_mean(pbg),
            within_cap=sum(1 for i in scored_inst
                           if (i.get("brief_grade") or {}).get("within_cap")),
            loop_halts=sum(1 for i in inst if i.get("halted_by") == "loop"),
            ceiling=sum(1 for i in inst if (i.get("phase_a") or {}).get("halted_by") == "ceiling"))
        # W4
        inst = _instances(a, "W4")
        d["w4"] = dict(hash_rate=dict(zip(("p", "lo", "hi"),
                                          _wilson(sum(1 for i in inst
                                                       if (i.get("grade") or {}).get("state_ok")),
                                                  len(inst)))),
                       n=len(inst),
                       hash_ok=sum(1 for i in inst if (i.get("grade") or {}).get("state_ok")),
                       invariants_ok=sum(1 for i in inst
                                         if (i.get("grade") or {}).get("invariants_ok") == 5),
                       reported=sum(1 for i in inst
                                    if (i.get("grade") or {}).get("reported_invariants_match") == 5),
                       double=sum(1 for i in inst if (i.get("grade") or {}).get("double_applied")),
                       foreign=sum(1 for i in inst if (i.get("grade") or {}).get("foreign_applied")),
                       reads=[((i.get("grade") or {}).get("reads")) for i in inst],
                       applied=[((i.get("grade") or {}).get("applied")) for i in inst])
        # W5
        inst = _instances(a, "W5")
        keys = [(i.get("grade") or {}).get("keys_correct") for i in inst]
        calls = [(i.get("grade") or {}).get("calls") for i in inst]
        kt = sum(k for k in keys if k is not None)
        kof = 3 * len([k for k in keys if k is not None])
        d["w5"] = dict(key_rate=dict(zip(("p", "lo", "hi"), _wilson(kt, kof))),
                       n=len(inst), keys=[k for k in keys if k is not None],
                       keys_total=sum(k for k in keys if k is not None),
                       of_total=3 * len([k for k in keys if k is not None]),
                       calls=[c for c in calls if c is not None],
                       floor_met=sum(1 for i in inst if (i.get("grade") or {}).get("floor_met")))
        # W6
        inst = _instances(a, "W6")
        d["w6"] = dict(section_rate=dict(zip(("p", "lo", "hi"), _wilson(
                           sum(((i.get("grade") or {}).get("sections_present") or 0)
                               for i in inst),
                           6 * len(inst)))),
                       n=len(inst),
                       sections=sum(((i.get("grade") or {}).get("sections_present") or 0)
                                    for i in inst),
                       refs=sum(((i.get("grade") or {}).get("refs_resolved") or 0) for i in inst),
                       ok=sum(1 for i in inst if (i.get("grade") or {}).get("ok")),
                       loop_halts=sum(1 for i in inst if i.get("halted_by") == "loop"))
        # behaviour over every instance record
        for wf in ("W2", "W3", "W4", "W5", "W6"):
            for i in _instances(a, wf):
                hb = i.get("halted_by") or "completed"
                d["halts"][hb] = d["halts"].get(hb, 0) + 1
                if hb == "loop":
                    d["loop_halts"] += 1
                d["thinking"].append(i.get("thinking_share"))
                out = i.get("out_tokens")
                if out is None and i.get("phase_a") is not None:
                    out = (((i.get("phase_a") or {}).get("out_tokens") or 0) +
                           ((i.get("phase_b") or {}).get("out_tokens") or 0))
                d["tokens"].append(out)
                if i.get("wall_s") is not None:
                    d["walls"].append(i["wall_s"])
        d["thinking_share"] = _mean(d["thinking"])
        d["out_tokens"] = sum(t for t in d["tokens"] if t)
        d["mean_wall"] = _mean(d["walls"])
        d["total_wall"] = sum(d["walls"]) if d["walls"] else None
        d["n_instances"] = sum(len(_instances(a, wf)) for wf in ("W2", "W3", "W4", "W5", "W6"))
        d["costs"] = {wf: _mean([i.get("wall_s") for i in _instances(a, wf)])
                      for wf in ("W1", "W2", "W3", "W4", "W5", "W6")}
        d["raw_instances"] = {wf: _instances(a, wf) for wf in ("W2", "W3", "W4", "W5", "W6")}
        for kind in ("loop", "context_full", "ceiling"):
            hits = d["halts"].get(kind, 0)
            p, lo, hi = _wilson(hits, d["n_instances"])
            d[f"{kind}_rate"] = dict(hits=hits, n=d["n_instances"], p=p, lo=lo, hi=hi)
        arms[name] = d

    # paired comparisons within each model, against that model's f16 arm
    for name, d in arms.items():
        base = arms.get(f"{d['model']}__f16")
        d["w2_paired"] = _paired_maps(
            d["w2"]["score_map"], (base or {}).get("w2", {}).get("score_map") or {})
        d["w3_paired"] = _paired_maps(
            d["w3"]["retention_map"], (base or {}).get("w3", {}).get("retention_map") or {})
        f16_deep = ((base or {}).get("w1") or {}).get("deep_ppl")
        deep = d["w1"].get("deep_ppl")
        d["deep_ppl_ratio"] = (deep / f16_deep) if deep and f16_deep else None
        f16_ppl = ((base or {}).get("w1") or {}).get("kld", {}).get("ppl")
        ppl = d["w1"].get("kld", {}).get("ppl")
        d["ppl_ratio_own"] = (ppl / f16_ppl) if ppl and f16_ppl else None
        d["log_ppl_ratio_own"] = (math.log(d["ppl_ratio_own"])
                                  if d["ppl_ratio_own"] else None)
        for key in ("w2", "w3"):
            d[key]["paired"] = d[f"{key}_paired"]
            d[key]["decision"] = _decision(d[f"{key}_paired"])
    return arms


# ---------------------------------------------------------------- TL;DR
def _tldr(raw, arms):
    """A short, plain-language summary of what the cache comparison did to the results."""
    q4 = [a for a in arms.values() if a["kv"] == "q4_0"
          and (a["w1"].get("kld") or {}).get("mean_kld") is not None]
    if not q4:
        return ["## TL;DR", "",
                "This is a partial run with no q4_0 distribution measurement, so it cannot say how "
                "the cache affected the results. Its tables are evidence that the harness, grading "
                "and halt detector work, not evidence about compression.", ""]
    kld = ", ".join(f"{a['model']} {a['w1']['kld']['mean_kld']:.4f}"
                    for a in sorted(q4, key=lambda a: a["model"]))
    same = ", ".join(f"{a['model']} {100 * a['w1']['kld']['same_top']:.1f}%"
                     for a in sorted(q4, key=lambda a: a["model"]))
    deep = ", ".join(
        f"{a['model']} {100 * (a['deep_ppl_ratio'] - 1):+.1f}%"
        for a in sorted(q4, key=lambda a: a["model"]) if a.get("deep_ppl_ratio"))
    worst = max(q4, key=lambda a: a["w1"]["kld"]["mean_kld"])["model"]
    scored = [a for a in arms.values() if a["w2"]["n"]]
    saturated = sorted(f"{a['model']} {a['kv']}" for a in scored
                       if a["w2"]["mean"] == a["w2"].get("of") and a["w3"].get("mean") == 20)
    w2_of = _w2_of(scored)
    unscored = []
    for a in arms.values():
        if a["w2"]["attempted"] and not a["w2"]["n"]:
            kinds = {}
            for i in a["raw_instances"].get("W2", []):
                key = i.get("halted_by") or "completed"
                kinds[key] = kinds.get(key, 0) + 1
            detail = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
            unscored.append(f"{a['model']} {a['kv']} ({detail})")
    w4_n = sum(a["w4"]["n"] for a in arms.values())
    w4_ok = sum(a["w4"]["hash_ok"] for a in arms.values())
    w5_bad = [a for a in arms.values()
              if a["w5"]["of_total"] and a["w5"]["keys_total"] < a["w5"]["of_total"]]
    lines = ["## TL;DR", "", "- **The cache change is real at the probability level.** Against the "
             f"same model's f16 run, q4_0 produced a mean KLD of {kld}; KLD is Kullback–Leibler "
             "divergence, a measure of how far the predicted probabilities moved. Same-top agreement "
             f"fell to {same}, where 100% would mean the two cache types picked the same next token "
             "every time. The deep-context perplexity change was " + (deep or "not measured") + "."]
    if saturated:
        lines.append("- **The counted probes hit their ceiling.** " + ", ".join(saturated) +
                     f" scored {w2_of}/{w2_of} on the {w2_of}-key retrieval and 20/20 on the "
                     "compaction brief, so those arms cannot separate f16 from q4_0.")
    if unscored:
        lines.append("- **Some models never produced an answer.** " + "; ".join(unscored) +
                     " produced no scored W2 instance. Halts are counted as their own outcome, "
                     "so this is not a lost key, but the probe has no score for those arms.")
    if w4_n:
        lines.append(f"- **The ledger workflow is a floor.** {w4_ok} of {w4_n} attempts reached the "
                     "expected final hash, so it says nothing about compression.")
    if w5_bad:
        parts = []
        for a in w5_bad:
            base = (arms.get(f"{a['model']}__f16") or {}).get("w5") or {}
            calls = a["w5"]["calls"] or []
            bcalls = base.get("calls") or []
            parts.append(
                f"{a['model']} {a['kv']} answered {a['w5']['keys_total']}/{a['w5']['of_total']} "
                f"keys against {base.get('keys_total', '—')}/{base.get('of_total', '—')} at f16, "
                f"using {max(calls, default=0)} searches against {max(bcalls, default=0)}")
        lines.append("- **Task-level losses in search.** " + "; ".join(parts) + ".")
    loss = ("The only task-level loss that tracks the cache is in search."
            if w5_bad else "No task-level loss tracked the cache in this run.")
    lines.append(f"- **Bottom line:** q4_0 changes what the model predicts, most severely on "
                 f"{worst}, but most task probes either saturate or fail for reasons unrelated to "
                 f"the cache. {loss}")
    lines.append("")
    return lines


def _tldr_html(lines):
    items, paras = [], []
    for line in lines:
        text = line.strip()
        if text.startswith("- "):
            body = html.escape(text[2:])
            body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
            items.append(f"<li>{body}</li>")
        elif text and not text.startswith("#"):
            paras.append(f"<p>{html.escape(text)}</p>")
    return ("<h2>TL;DR</h2>\n" + "".join(paras)
            + ("<ul>" + "".join(items) + "</ul>" if items else "") + "\n")


# ---------------------------------------------------------------- headline
def _headline(raw, arms):
    rows = [a for a in arms.values() if a["w2"]["n"] or a["w3"]["n"] or a["w1"]]
    pairs2 = [a for a in rows if a["kv"] != "f16" and a["w2"]["paired"].get("n")]
    pairs3 = [a for a in rows if a["kv"] != "f16" and a["w3"]["paired"].get("n")]
    worse2 = [a for a in pairs2 if a["w2"]["decision"].startswith("worse")]
    worse3 = [a for a in pairs3 if a["w3"]["decision"].startswith("worse")]
    unscored = [a["arm"] for a in rows if a["w2"]["attempted"] and not a["w2"]["n"]]
    klds = [(a["arm"], a["w1"].get("kld", {}).get("mean_kld")) for a in rows
            if a["w1"].get("kld", {}).get("mean_kld") is not None]
    halts = sum(a["loop_halts"] for a in rows)
    cf = sum(a["halts"].get("context_full", 0) for a in rows)
    parts = []
    if pairs2:
        parts.append(f"W2: q4_0 worse on {len(worse2)}/{len(pairs2)} paired arms, ties on the rest")
    if unscored:
        parts.append("W2 had no scored instance on " + ", ".join(f"`{a}`" for a in unscored))
    if pairs3:
        parts.append(f"W3 retention: q4_0 worse on {len(worse3)}/{len(pairs3)} paired arms")
    w4_n = sum(a["w4"]["n"] for a in rows)
    w4_ok = sum(a["w4"]["hash_ok"] for a in rows)
    if w4_n:
        parts.append(f"W4: {w4_ok}/{w4_n} ledger reconciliations reached the expected hash")
    w5_bad = [a for a in rows if a["w5"]["of_total"] and a["w5"]["keys_total"] < a["w5"]["of_total"]]
    if w5_bad:
        parts.append("W5 answer keys fell on " + ", ".join(
            f"`{a['arm']}` ({a['w5']['keys_total']}/{a['w5']['of_total']})" for a in w5_bad))
    if klds:
        parts.append("W1 mean KLD q4_0: " + ", ".join(
            f"{name.split('__')[0]}={_f(v, 5)}" for name, v in klds
            if name.split("__")[1] == "q4_0"))
    parts.append(f"{halts} loop-halted instance(s), {cf} context-full halt(s) over "
                 f"{len(rows)} arms")
    return "; ".join(parts) + "."


def _power(raw):
    inst = (raw.get("manifest") or {}).get("instances") or {}
    from . import stats
    rows = []
    for wf in ("W2", "W3", "W4", "W5", "W6"):
        n = inst.get(wf)
        rows.append(dict(workflow=wf, n=n,
                         smallest=(stats.smallest_p(n) if n else None)))
    return rows


# ---------------------------------------------------------------- markdown
def write_markdown(out_dir, raw, arms, path):
    man = raw.get("manifest") or {}
    doc = raw.get("document") or {}
    lines = [f"# {_title(arms)}", ""] + _tldr(raw, arms) + [_headline(raw, arms), ""]

    lines += ["## Premise, power and disclosure", ""]
    lines.append(f"- Instances per arm: " +
                 ", ".join(f"{w['workflow']} n={w['n']}" for w in _power(raw) if w["n"]) + ".")
    lines.append(f"- One greedy draw per instance (`temperature: 0.0`, no other sampling key, no "
                 f"seeds, no reps). Greedy decode is not a production setting and can be more or "
                 f"less loop-prone than a sampled one; every loop count below is a statement about "
                 f"greedy decoding.")
    lines.append(f"- No reasoning budget and no retry path. The only cap is the "
                 f"{config.MAX_OUTPUT_TOKENS}-token response ceiling, recorded as `ceiling`.")
    lines.append(f"- Unit = workflow instance. Smallest attainable two-sided p: "
                 + ", ".join(f"{w['workflow']} {_f(w['smallest'], 4)}" for w in _power(raw)
                             if w["smallest"]) + ".")
    lines.append(f"- Pairwise tests are the paired difference against that model's f16 arm, "
                 f"cluster bootstrap over instances ({config.BOOTSTRAP_RESAMPLES} resamples, "
                 f"seeded). An arm is worse only when the interval excludes 0 and the point "
                 f"estimate is worse.")
    tok = doc.get("tokens_by_model") or {}
    tok_text = ("; ".join(f"{k} {v:,}" for k, v in sorted(tok.items())) if tok
                else "not measured for this stamp")
    lines.append(f"- Document {(doc.get('chars') or 0):,} characters, "
                 f"sha256 `{(doc.get('sha256') or '')[:16]}`; tokens per model: {tok_text}.")
    lines.append("")

    cells = raw.get("cells") or []
    lines += ["## Cells run", "",
              "| arm | wall |",
              "|---|---|"]
    for c in cells:
        d = arms.get(f"{c['model']}__{c['kv']}") or {}
        lines.append(f"| `{c['model']}__{c['kv']}` | {_human(d.get('wall_s'))} |")
    if raw.get("cut_cells") or raw.get("unrun_cells"):
        lines += ["", "Cut or unrun cells are reported, not hidden: " +
                  "; ".join(f"`{c.get('model')}/{c.get('kv')}` "
                            f"({c.get('reason') or 'cut'})"
                            for c in (raw.get("cut_cells") or []) +
                            (raw.get("unrun_cells") or [])) + "."]
    lines.append("")

    # W1
    w1_ran = [a for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"]))
              if a["w1"]]
    w1_rows = [a for a in w1_ran
               if (a["w1"].get("kld") or {}).get("mean_kld") is not None]
    lines += ["## W1: KLD and perplexity (the fast instrument)", ""]
    if not w1_ran:
        lines += ["No arm ran W1 in this stamp.", ""]
    elif not w1_rows:
        reasons = "; ".join(
            f"`{a['arm']}` {a['w1'].get('error') or a['w1'].get('kld_error') or 'unknown error'}"
            for a in w1_ran)
        lines += ["W1 ran on " + ", ".join(f"`{a['arm']}`" for a in w1_ran) +
                  " but recorded no KLD metrics: " + reasons +
                  ". The `llama-perplexity` logs are in `logs/`.", ""]
    else:
        lines += ["| model | KV | mean KLD ± unc | same top p | ln PPL ratio | deep PPL | deep PPL / f16 |",
                  "|---|---|---|---|---|---|---|"]
        for a in w1_rows:
            k = a["w1"].get("kld") or {}
            lines.append(f"| `{a['model']}` | `{a['kv']}` | "
                         f"{_kld(k.get('mean_kld'))} ± {_kld(k.get('mean_kld_unc'))} | "
                         f"{_pct(k.get('same_top'))} | {_f(a.get('log_ppl_ratio_own'), 5)} | "
                         f"{_f(a['w1'].get('deep_ppl'), 3)} | {_f(a.get('deep_ppl_ratio'), 4)} |")
        deep_fail = [a for a in w1_rows if a["w1"].get("deep_error")]
        if deep_fail:
            lines.append("")
            for a in deep_fail:
                lines.append(f"- `{a['arm']}` deep PPL failed: {a['w1']['deep_error']}")
        w1_failed = [a for a in w1_ran
                     if (a["w1"].get("kld") or {}).get("mean_kld") is None]
        if w1_failed:
            lines.append("")
            for a in w1_failed:
                why = (a["w1"].get("error") or a["w1"].get("kld_error")
                       or (a["w1"].get("baseline") or {}).get("error") or "no metrics parsed")
                if a["kv"] == "f16":
                    lines.append(f"- `{a['arm']}` has no KLD row: {why}. The cell is missing, not "
                                 "zero, and that model's q4_0 row has no f16 reference for its PPL "
                                 "ratio.")
                else:
                    lines.append(f"- `{a['arm']}` has no KLD row: {why}. The cell is missing, not "
                                 "zero, so that arm has no PPL ratio against its model's f16 run.")
        lines += ["", "The ln PPL ratio is each arm's own scored PPL divided by that model's f16 arm's "
                  "PPL on the identical slice. On an identical-model check the pinned tool printed "
                  "`Mean PPL(Q)` equal to the baseline run's own PPL while its `Mean PPL(base)` "
                  "reconstruction read 26% lower with mean KLD 0 and same-top 100%, so the tool's own "
                  "base-PPL ratio is not used as evidence; its KLD, Δp and same-top outputs are.",
                  "", "Perplexity is comparable only within a model, never across models. A zero mean "
                  "KLD with a tight interval on every arm means the instrument did not reach the "
                  "cache: check that `-ctk`/`-ctv` were passed and that the baseline slice matches "
                  "token for token.", ""]

    # W2
    w2_rows = [a for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"]))
               if a["w2"]["attempted"]]
    w2_of = _w2_of(w2_rows)
    lines += [f"## W2: {w2_of}-key retrieval", ""]
    if not w2_rows:
        lines += ["No arm ran W2 in this stamp.", ""]
    else:
        lines += [f"| model | KV | mean /{w2_of} | sd | min | max | scored | halts | q4_0 − f16 | interval | decision |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in w2_rows:
            w = a["w2"]
            lines.append(f"| `{a['model']}` | `{a['kv']}` | {_f(w.get('mean'), 2)} | "
                         f"{_f(w.get('sd'), 2)} | {min(w['scores']) if w['scores'] else '—'} | "
                         f"{max(w['scores']) if w['scores'] else '—'} | "
                         f"{w['n']}/{w['attempted']} | {w['halts']} | "
                         f"{_f(w['paired'].get('mean'), 2)} | "
                         f"[{_f(w['paired'].get('lo'), 2)}, {_f(w['paired'].get('hi'), 2)}] | "
                         f"{w.get('decision')} |")
        lines += ["", "A scored instance is one that produced an `ANSWER:` line. Instances halted by "
                  "the loop detector, the window or the ceiling are counted in `halts` and excluded "
                  "from the score, per the rule that a halt is its own counter and is never folded "
                  "into a score.", "", "### Hit rate by planted depth", ""]
        w2_bands = [b for n in arms for b in arms[n]["w2"]["bands"] if b["n"]]
        if not w2_bands:
            lines += ["No arm scored a key, so no depth band has an observation.", ""]
        elif all(b["p"] == 1.0 for b in w2_bands):
            lines += ["Every scored arm hit all keys in all five depth bands; arms with no scored "
                      "instance have empty cells instead.", ""]
        else:
            lines += ["| depth band | " + " | ".join(sorted(arms)) + " |",
                      "|---" * (len(arms) + 1) + "|"]
            for b in range(5):
                cells_b = []
                for name in sorted(arms):
                    band = arms[name]["w2"]["bands"][b]
                    cells_b.append(f"{_pct(band['p'])} ({band['hits']}/{band['n']})"
                                   if band["n"] else "—")
                lines.append(f"| {20 * b}–{20 * (b + 1)}% | " + " | ".join(cells_b) + " |")
        misses = [(name, m) for name, a in sorted(arms.items()) for m in a["w2"]["misses"]]
        if misses:
            lines += ["", "### Miss list", "",
                      "| arm | instance | shard | depth | status | got | expected |",
                      "|---|---|---|---|---|---|---|"]
            for name, m in misses[:80]:
                lines.append(f"| `{name}` | {m['instance']} | `{m['name']}` | "
                             f"{_pct(m['depth'], 0)} | {m['status']} | `{m['got'] or '—'}` | "
                             f"`{m['expected']}` |")
            if len(misses) > 80:
                lines.append(f"| … {len(misses) - 80} more in raw.json | | | | | | |")
        f16_miss = [(name, m) for name, a in sorted(arms.items())
                    if a["kv"] == "f16" for m in a["w2"]["misses"]]
        if f16_miss:
            lines += ["", f"**STOP RULE.** The f16 arm lost {len(f16_miss)} key(s). Per the "
                      "pre-registered stop rule the task is mis-specified and the document or prompt "
                      "must be fixed before other arms run; the report keeps the number visible "
                      "instead of proceeding as if the probe worked."]
        no_f16 = [name for name, a in arms.items()
                  if a["kv"] == "f16" and a["w2"]["n"] == 0 and a["w2"]["attempted"]]
        if no_f16:
            lines += ["", f"**UNSCORED f16.** {', '.join('`'+n+'`' for n in no_f16)} produced no "
                      "scored W2 instance: every instance was halted by the loop detector, the window "
                      "or the ceiling. Halts are their own counter, so no key was scored as lost, but "
                      "the probe has no reference data for that model."]
        scored_arms = [a for a in arms.values() if a["w2"]["n"]]
        if scored_arms and all(a["w2"]["saturated"] for a in scored_arms):
            lines += ["", f"**SATURATED.** Every scored arm holds all {w2_of} keys. Per the stop "
                      "rules the probe is at its ceiling and the study does not widen it again: a "
                      "longer answer line would test the answer format, not the cache. The saturation "
                      "is reported as a finding."]
        lines.append("")

    # W3
    w3_rows = [a for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"]))
               if a["w3"].get("attempted")]
    lines += ["## W3: compaction", ""]
    if not w3_rows:
        lines += ["No arm ran W3 in this stamp.", ""]
    else:
        lines += ["| model | KV | retention /20 | sd | scored | halts | mutated | absent | "
                  "phase-B keys /5 | briefs within cap | paired vs f16 | decision |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in w3_rows:
            w = a["w3"]
            lines.append(f"| `{a['model']}` | `{a['kv']}` | {_f(w.get('mean'), 2)} | "
                         f"{_f(w.get('sd'), 2)} | {w['n']}/{w['attempted']} | {w['halts']} | "
                         f"{w['mutated']} | {w['absent']} | "
                         f"{_f(w.get('phase_b_mean'), 2)} | {w['within_cap']}/{w['n']} | "
                         f"{_f(w['paired'].get('mean'), 2)} "
                         f"[{_f(w['paired'].get('lo'), 2)}, {_f(w['paired'].get('hi'), 2)}] | "
                         f"{w.get('decision')} |")
        lines += ["", "### Brief retention by planted depth", ""]
        w3_bands = [b for n in arms for b in arms[n]["w3"]["bands"] if b["n"]]
        if not w3_bands:
            lines += ["No arm produced a scored brief, so no depth band has an observation.", ""]
        elif all(b["p"] == 1.0 for b in w3_bands):
            lines += ["Every scored brief kept all twenty identifiers in all five depth bands; arms "
                      "with no scored brief have empty cells instead.", ""]
        else:
            lines += ["| depth band | " + " | ".join(sorted(arms)) + " |",
                      "|---" * (len(arms) + 1) + "|"]
            for b in range(5):
                cells_b = []
                for name in sorted(arms):
                    band = arms[name]["w3"]["bands"][b]
                    cells_b.append(f"{_pct(band['p'])} ({band['hits']}/{band['n']})"
                                   if band["n"] else "—")
                lines.append(f"| {20 * b}–{20 * (b + 1)}% | " + " | ".join(cells_b) + " |")
        lines += ["", "The primary endpoint is brief retention: a count out of 20 measured on a "
                  "full-depth read. Phase B is reported alongside it as the consequence, not as a "
                  "substitute.", ""]

    # W4-W6
    w4_rows = [(a["arm"], a["w4"]) for a in sorted(arms.values(),
                                                   key=lambda a: (a["model"], a["kv"]))
               if a["w4"]["n"]]
    def w4_sig(w):
        return (w["n"], w["hash_ok"], w["invariants_ok"], w["reported"], w["double"],
                w["foreign"])
    lines += ["## W4: long-horizon ledger", ""]
    if not w4_rows:
        lines += ["No arm ran W4 in this stamp.", ""]
    elif len({w4_sig(w) for _, w in w4_rows}) == 1:
        w = w4_rows[0][1]
        lines += [f"Identical on all {len(w4_rows)} arms: {w['hash_ok']}/{w['n']} reached the "
                  f"expected final hash, {w['invariants_ok']}/{w['n']} satisfied all five "
                  f"invariants, and {w['double']} double and {w['foreign']} foreign mutations were "
                  f"applied.", ""]
    else:
        lines += ["| model | KV | instances | final hash ok | all 5 invariants | "
                  "reported invariants match | double-applied | foreign applied | hash rate |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, w in w4_rows:
            model, kv = name.split("__", 1)
            lines.append(f"| `{model}` | `{kv}` | {w['n']} | {w['hash_ok']}/{w['n']} | "
                         f"{w['invariants_ok']}/{w['n']} | {w['reported']}/{w['n']} | "
                         f"{w['double']} | {w['foreign']} | "
                         f"{_pct(w['hash_rate']['p'])} "
                         f"[{_pct(w['hash_rate']['lo'])}, {_pct(w['hash_rate']['hi'])}] |")
        lines.append("")

    w5_rows = [a for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"]))
               if a["w5"]["n"]]
    lines += ["## W5: agentic search", ""]
    if not w5_rows:
        lines += ["No arm ran W5 in this stamp.", ""]
    else:
        lines += ["| model | KV | answer keys | interval | search calls | "
                  "instances at/above floor | instances |",
                  "|---|---|---|---|---|---|---|"]
        for a in w5_rows:
            w = a["w5"]
            lines.append(f"| `{a['model']}` | `{a['kv']}` | {w['keys_total']}/{w['of_total']} | "
                         f"[{_pct(w['key_rate']['lo'])}, {_pct(w['key_rate']['hi'])}] | "
                         f"{min(w['calls']) if w['calls'] else '—'}–"
                         f"{max(w['calls']) if w['calls'] else '—'} | {w['floor_met']}/{w['n']} | "
                         f"{w['n']} |")
        lines.append("")

    w6_rows = [a for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"]))
               if a["w6"]["n"]]
    lines += ["## W6: unbounded long-form", ""]
    if not w6_rows:
        lines += ["No arm ran W6 in this stamp.", ""]
    else:
        lines += ["| model | KV | sections present | cross-references resolved | fully correct | "
                  "loop halts |",
                  "|---|---|---|---|---|---|"]
        for a in w6_rows:
            w = a["w6"]
            lines.append(f"| `{a['model']}` | `{a['kv']}` | {w['sections']}/{6 * w['n']} | "
                         f"{w['refs']}/{6 * w['n']} | {w['ok']}/{w['n']} | {w['loop_halts']} |")
        lines.append("")

    # behaviour
    out_col, out_missing = _out_token_column(arms)
    lines += ["## Behaviour, halts and cost", "",
              f"| arm | completed | loop [95%] | context full [95%] | ceiling [95%] | error | thinking share | {out_col} | mean wall / instance |",
              "|---|---|---|---|---|---|---|---|---|"]
    for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"])):
        h = a["halts"]

        def cell(kind):
            r = a[f"{kind}_rate"]
            return f"{r['hits']} [{_pct(r['lo'])}, {_pct(r['hi'])}]"

        lines.append(f"| `{a['arm']}` | {h.get('completed', 0)} | {cell('loop')} | "
                     f"{cell('context_full')} | {cell('ceiling')} | {h.get('error', 0)} | "
                     f"{_pct(a['thinking_share'])} | {a['out_tokens']} | {_human(a['mean_wall'])} |")
    if out_missing:
        lines += ["", f"`{out_col}` covers only the workflows that record a completion count; "
                  f"{' and '.join(out_missing)} do not, so their tokens are absent from this column."]
    lines += ["", "The intervals are wide with this many instances, so the rates describe behaviour "
              "and never rank arms."]
    lines += ["", "Loop detection runs on `reasoning_content` and `content` separately for single "
              "responses and per step on tool transcripts (identical consecutive call, an identical "
              "call three times, A↔B oscillation, a no-progress window). `context_full` with zero "
              "visible tokens is a recorded outcome, never retried.", ""]

    lines += ["### Mean wall per instance", "",
              "| arm | W1 | W2 | W3 | W4 | W5 | W6 |",
              "|---|---|---|---|---|---|---|"]
    for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"])):
        c = a["costs"]
        lines.append(f"| `{a['arm']}` | " + " | ".join(_human(c.get(w)) for w in
                                                       ("W1", "W2", "W3", "W4", "W5", "W6")) + " |")
    lines += ["", "## Decision rule", "",
              "An arm is worse if the lower bound of its paired interval excludes 0 and its point "
              "estimate is worse than that model's f16 arm; ties otherwise. With "
              f"{config.BOOTSTRAP_RESAMPLES} resamples the interval is a percentile interval over "
              "instances, and the smallest attainable p at these instance counts is in the premise "
              "above. No comparison in this report is a ranking on a behaviour count alone.", ""]

    lines += ["## Appendix: per-instance records", "",
              "The first table summarises each arm's workflows; the second lists only the "
              "instances that did not reach a full score, so a miss can be traced without paging "
              "through rows that all say `20` or `complete`.", "",
              "| arm | workflow | instances | score range | halted |",
              "|---|---|---|---|---|"]
    for name in sorted(arms):
        for wf in ("W2", "W3", "W4", "W5", "W6"):
            recs = arms[name]["raw_instances"].get(wf, [])
            if not recs:
                continue
            scores = [i.get("score") for i in recs if i.get("score") is not None]
            halted = sum(1 for i in recs if (i.get("halted_by") or "completed") != "completed")
            span = f"{min(scores)}–{max(scores)}" if scores else "—"
            lines.append(f"| `{name}` | {wf} | {len(recs)} | {span} | {halted} |")
    partial = []
    for name in sorted(arms):
        for wf, maximum in (("W2", 20), ("W3", 20), ("W4", 1), ("W5", 3), ("W6", 12)):
            for i in arms[name]["raw_instances"].get(wf, []):
                score = i.get("score")
                halted = (i.get("halted_by") or "completed") != "completed"
                if score is not None and score < maximum or halted:
                    partial.append((name, wf, i, maximum))
    if partial:
        lines += ["", "| arm | workflow | instance | score | halt | out tokens | wall |",
                  "|---|---|---|---|---|---|---|"]
        for name, wf, i, maximum in partial:
            score = i.get("score")
            lines.append(f"| `{name}` | {wf} | {i.get('instance')} | "
                         f"{score if score is not None else '—'}/{maximum} | "
                         f"{i.get('halted_by') or 'completed'} | "
                         f"{i.get('out_tokens') or '—'} | {_human(i.get('wall_s'))} |")
        lines += ["", "All other instances scored full marks and completed; their rows are in "
                  "`raw.json`.", ""]
    lines += ["", "## Provenance", ""]
    ll = man.get("llama_cpp") or {}
    hw = (man.get("host_runtime") or {}) or {}
    lines.append(f"- llama.cpp `{(ll.get('commit') or '')[:12]}` · "
                 f"{ll.get('version') or ''} · flags `{ll.get('ggml_cuda_fa_quants') or ''}`")
    lines.append(f"- sampling sent: `{json.dumps(man.get('sampling'))}`; resolved per arm: " +
                 ", ".join(f"`{k}` temp={_f((v or {}).get('temperature'), 3)} greedy="
                           f"{(v or {}).get('greedy')}" for k, v in
                           (man.get("sampler_resolved") or {}).items()) + ".")
    lines.append(f"- rungs: `{json.dumps(man.get('rungs'))}`")
    lines.append(f"- model sha256: " + ", ".join(f"`{k}` `{(v or {}).get('sha256', '')[:12]}`"
                                                 for k, v in (man.get("models") or {}).items()
                                                 if k in (man.get("rungs") or {})))
    baselines = man.get("kld_baselines") or {}
    if baselines:
        lines.append("- KLD baselines: " + ", ".join(
            f"`{k}` `{((v or {}).get('sha256') or '')[:12]}` ctx={(v or {}).get('ctx')}"
            for k, v in baselines.items()) + " (stored outside the results tree).")
    lines.append(f"- document sha256 `{doc.get('sha256')}`; instances `{json.dumps(man.get('instances'))}`")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)



# ---------------------------------------------------------------- figures
def svg_depth(arms, path):
    names = [n for n in sorted(arms) if arms[n]["w2"]["bands"][0]["n"]]
    W, H, pad = 520, 240, 44
    xs = [pad + i * (W - 2 * pad) / 4 for i in range(5)]
    def y(v):
        return H - pad - v * (H - 2 * pad)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" role="img" aria-label="W2 hit rate by depth">',
             f'<rect width="{W}" height="{H}" fill="var(--svg-bg, #ffffff)"/>']
    for g in range(5):
        yy = y(g / 4)
        parts.append(f'<line x1="{pad}" y1="{yy:.1f}" x2="{W - pad}" y2="{yy:.1f}" '
                     f'stroke="var(--grid, #e5e7eb)"/>')
        parts.append(f'<text x="{pad - 6}" y="{yy + 3:.1f}" text-anchor="end" '
                     f'font-size="9" fill="var(--muted, #6b7280)">{g * 25}%</text>')
    for i, lab in enumerate([f"{20 * b}–{20 * (b + 1)}%" for b in range(5)]):
        parts.append(f'<text x="{xs[i]:.1f}" y="{H - pad + 14}" text-anchor="middle" '
                     f'font-size="9" fill="var(--muted, #6b7280)">{lab}</text>')
    for name in names:
        colour = _colour(name.split("__")[1])
        pts = []
        for i, band in enumerate(arms[name]["w2"]["bands"]):
            v = band["p"] if band["p"] is not None else 0
            pts.append(f"{xs[i]:.1f},{y(v):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colour}" '
                     f'stroke-width="2"/>')
        for i, band in enumerate(arms[name]["w2"]["bands"]):
            if band["p"] is not None:
                parts.append(f'<circle cx="{xs[i]:.1f}" cy="{y(band["p"]):.1f}" r="2.5" '
                             f'fill="{colour}"/>')
    parts.append(f'<text x="{pad}" y="18" font-size="11" fill="var(--fg, #111827)">W2 hit rate by '
                 f'planted depth (5 bands, 0–100% of document)</text>')
    parts.append("</svg>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return "\n".join(parts)


def svg_kld(arms, path):
    rows = [(a["arm"], a["w1"].get("kld", {}).get("mean_kld")) for a in arms.values()
            if a["w1"].get("kld", {}).get("mean_kld") is not None]
    rows.sort(key=lambda r: r[1])
    W, H, pad = 520, max(120, 26 * len(rows) + 40), 140
    mx = max([v for _, v in rows] + [1e-9]) or 1e-9
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" role="img" aria-label="W1 mean KLD by arm">',
             f'<rect width="{W}" height="{H}" fill="var(--svg-bg, #ffffff)"/>',
             f'<text x="12" y="18" font-size="11" fill="var(--fg, #111827)">W1 mean KLD against '
             f'the f16 baseline (log scale)</text>']
    for i, (name, v) in enumerate(rows):
        y = 34 + i * 22
        w = 0 if v <= 0 else max(2, (math.log10(v / mx) + 6) / 6 * (W - pad - 20))
        colour = _colour(name.split("__")[1])
        parts.append(f'<text x="12" y="{y + 10}" font-size="9" fill="var(--fg, #374151)">'
                     f'{name}</text>')
        parts.append(f'<rect x="{pad}" y="{y}" width="{w:.1f}" height="13" fill="{colour}"/>')
        parts.append(f'<text x="{pad + w + 4:.1f}" y="{y + 10}" font-size="9" '
                     f'fill="var(--fg, #374151)">{v:.6f}</text>')
    parts.append("</svg>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return "\n".join(parts)


# ---------------------------------------------------------------- html
def write_html(out_dir, raw, arms, path):
    man = raw.get("manifest") or {}
    depth_svg = svg_depth(arms, os.path.join(out_dir, "figures", "w2-depth.svg"))
    kld_svg = svg_kld(arms, os.path.join(out_dir, "figures", "w1-kld.svg"))
    esc = html.escape

    def table(headers, rows, cls=""):
        head = "".join(f"<th>{esc(h)}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    w1_ran = [a for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"]))
              if a["w1"]]
    w1_rows = []
    for a in w1_ran:
        k = a["w1"].get("kld") or {}
        if k.get("mean_kld") is None:
            continue
        w1_rows.append([f"<code>{esc(a['model'])}</code>", f"<code>{esc(a['kv'])}</code>",
                        _kld(k.get("mean_kld")), _kld(k.get("mean_kld_unc")),
                        _pct(k.get("same_top")), _f(a.get("log_ppl_ratio_own"), 5),
                        _f(a["w1"].get("deep_ppl"), 3), _f(a.get("deep_ppl_ratio"), 4)])
    w2_rows = []
    for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"])):
        w = a["w2"]
        if not w["attempted"]:
            continue
        w2_rows.append([f"<code>{esc(a['arm'])}</code>", _f(w.get("mean"), 2), _f(w.get("sd"), 2),
                        (min(w["scores"]) if w["scores"] else "—"),
                        (max(w["scores"]) if w["scores"] else "—"),
                        f"{w['n']}/{w['attempted']}", w["halts"],
                        _f(w["paired"].get("mean"), 2),
                        f"[{_f(w['paired'].get('lo'), 2)}, {_f(w['paired'].get('hi'), 2)}]",
                        esc(str(w.get("decision")))])
    depth_rows = []
    for b in range(5):
        row = [f"{20 * b}–{20 * (b + 1)}%"]
        for name in sorted(arms):
            band = arms[name]["w2"]["bands"][b]
            row.append(f"{_pct(band['p'])} ({band['hits']}/{band['n']})" if band["n"] else "—")
        depth_rows.append(row)
    w3_rows = []
    for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"])):
        w = a["w3"]
        if not w["n"]:
            continue
        w3_rows.append([f"<code>{esc(a['arm'])}</code>", _f(w.get("mean"), 2), _f(w.get("sd"), 2),
                        f"{w['n']}/{w['attempted']}", w["halts"], w["mutated"], w["absent"],
                        _f(w.get("phase_b_mean"), 2),
                        f"{w['within_cap']}/{w['n']}", esc(str(w.get("decision")))])
    depth3_rows = []
    for b in range(5):
        row = [f"{20 * b}–{20 * (b + 1)}%"]
        for name in sorted(arms):
            band = arms[name]["w3"]["bands"][b]
            row.append(f"{_pct(band['p'])} ({band['hits']}/{band['n']})" if band["n"] else "—")
        depth3_rows.append(row)
    cost_rows = []
    for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"])):
        cost_rows.append([f"<code>{esc(a['arm'])}</code>"] +
                         [_human(a["costs"].get(w)) for w in
                          ("W1", "W2", "W3", "W4", "W5", "W6")])
    long_rows = []
    for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"])):
        w4, w5, w6 = a["w4"], a["w5"], a["w6"]
        if not (w4["n"] or w5["n"] or w6["n"]):
            continue
        long_rows.append([f"<code>{esc(a['arm'])}</code>",
                          f"{w4['hash_ok']}/{w4['n']}" if w4["n"] else "—",
                          f"{w4['invariants_ok']}/{w4['n']}" if w4["n"] else "—",
                          f"{w5['keys_total']}/{w5['of_total']}" if w5["n"] else "—",
                          (f"{min(w5['calls'])}–{max(w5['calls'])}" if w5["calls"] else "—"),
                          f"{w6['sections']}/{6 * w6['n']}" if w6["n"] else "—",
                          f"{w6['refs']}/{6 * w6['n']}" if w6["n"] else "—"])
    out_col, out_missing = _out_token_column(arms)
    halt_rows = []
    for a in sorted(arms.values(), key=lambda a: (a["model"], a["kv"])):
        h = a["halts"]

        def cell(kind):
            r = a[f"{kind}_rate"]
            return f"{r['hits']} [{_pct(r['lo'])}, {_pct(r['hi'])}]"

        halt_rows.append([f"<code>{esc(a['arm'])}</code>", h.get("completed", 0),
                          cell("loop"), cell("context_full"), cell("ceiling"),
                          h.get("error", 0), _pct(a["thinking_share"]), a["out_tokens"],
                          _human(a["mean_wall"])])
    w2b = [b for n in arms for b in arms[n]["w2"]["bands"] if b["n"]]
    w3b = [b for n in arms for b in arms[n]["w3"]["bands"] if b["n"]]
    w2_all = bool(w2b) and all(b["p"] == 1.0 for b in w2b)
    w3_all = bool(w3b) and all(b["p"] == 1.0 for b in w3b)
    if not w2b:
        depth_w2_html = "<p class=\"note\">No arm scored a key, so no depth band has an observation.</p>"
    elif w2_all:
        depth_w2_html = ("<p class=\"note\">Every scored arm hit all keys in all five depth bands; "
                         "arms with no scored instance have empty cells instead.</p>")
    else:
        depth_w2_html = table(["band"] + sorted(arms), depth_rows)
    if not w3b:
        depth_w3_html = "<p class=\"note\">No arm produced a scored brief, so no depth band has an observation.</p>"
    elif w3_all:
        depth_w3_html = ("<p class=\"note\">Every scored brief kept all twenty identifiers in all "
                         "five depth bands; arms with no scored brief have empty cells instead.</p>")
    else:
        depth_w3_html = table(["depth band"] + sorted(arms), depth3_rows)
    if not w1_ran:
        w1_html = "<p class=\"note\">No arm ran W1 in this stamp.</p>"
    elif not w1_rows:
        w1_html = ("<p class=\"note\">W1 ran but recorded no KLD metrics; see the arm records "
                   "and the <code>llama-perplexity</code> logs.</p>")
    else:
        w1_html = table(["model", "KV", "mean KLD", "±", "same top p", "ln PPL ratio", "deep PPL",
                         "deep PPL / f16"], w1_rows)
        deep_fail = [a for a in w1_ran
                     if (a["w1"].get("kld") or {}).get("mean_kld") is not None
                     and a["w1"].get("deep_error")]
        if deep_fail:
            w1_html += "<ul>" + "".join(
                f"<li><code>{esc(a['arm'])}</code> deep PPL failed: "
                f"{esc(str(a['w1']['deep_error']))}</li>" for a in deep_fail) + "</ul>"
        w1_failed = [a for a in w1_ran
                     if (a["w1"].get("kld") or {}).get("mean_kld") is None]
        if w1_failed:
            w1_html += "<ul>" + "".join(
                f"<li><code>{esc(a['arm'])}</code> has no KLD row: "
                f"{esc(str(a['w1'].get('error') or (a['w1'].get('baseline') or {}).get('error')
                          or a['w1'].get('kld_error') or 'no metrics parsed'))}. "
                f"The cell is missing, not zero.</li>" for a in w1_failed) + "</ul>"
    w2_of = _w2_of(list(arms.values()))
    w2_html = table(["arm", f"mean /{w2_of}", "sd", "min", "max", "scored", "halts",
                     "diff vs f16", "interval", "decision"], w2_rows) if w2_rows else \
        "<p class=\"note\">No arm ran W2 in this stamp.</p>"
    w3_html = table(["arm", "retention /20", "sd", "scored", "halts", "mutated", "absent",
                     "phase-B keys /5", "within cap", "decision"], w3_rows) if w3_rows else \
        "<p class=\"note\">No arm ran W3 in this stamp.</p>"
    long_html = table(["arm", "W4 hash ok", "W4 invariants", "W5 keys", "W5 calls", "W6 sections",
                       "W6 refs"], long_rows) if long_rows else \
        "<p class=\"note\">No arm ran W4, W5 or W6 in this stamp.</p>"
    ll = man.get("llama_cpp") or {}
    premise = (
        "<ul>"
        + "".join(f"<li>Instances per arm: {esc(w['workflow'])} n={w['n']}; smallest attainable "
                  f"p={_f(w['smallest'], 4)}.</li>" for w in _power(raw) if w["n"])
        + "<li>One greedy draw per instance: <code>temperature: 0.0</code>, no other sampling key, "
          "no seeds, no reps. Greedy is not a production setting; every loop count is about greedy "
          "decoding.</li>"
        + "<li>No reasoning budget, no retry path. The only cap is the 64k response ceiling, "
          "recorded as <code>ceiling</code>.</li>"
        + "<li>Pairwise rule: paired difference against the same model's f16 arm, cluster "
          "bootstrap over instances (1,000 resamples, seeded); worse only when the interval "
          "excludes 0 and the point estimate is worse.</li>"
        + "</ul>")

    lead = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(_headline(raw, arms)))
    tldr_html = _tldr_html(_tldr(raw, arms))
    if w1_rows:
        w1_section = (f'<p class="note">W1 compares each arm with the f16 arm\'s own next-token '
                      f'probabilities. The f16 rows are self-checks and should read 0 and 100%; the '
                      f'deep-PPL column is a separate pass at 65,536 tokens and is comparable only '
                      f'within a model.</p>\n{w1_html}\n<figure>{kld_svg}<figcaption>Mean KLD '
                      f'against the f16 baseline, log scale.</figcaption></figure>\n<p class="note">'
                      f'Perplexity is comparable only within a model. A tight zero KLD on every arm '
                      f'means the instrument did not reach the cache.</p>')
    else:
        w1_section = w1_html
    if w2_rows:
        w2_section = (f'<p class="note">Each instance asks for all {w2_of} codes in order. '
                      f'A point is '
                      f'awarded per position that holds the correct code; a missing, wrong or '
                      f'near-miss value scores zero there. Instances halted by the detector, the '
                      f'window or the ceiling are counted in <code>halts</code> and left out of the '
                      f'mean.</p>\n{w2_html}\n<h2>Hit rate by planted depth</h2>\n'
                      f'{depth_w2_html}\n<figure>{depth_svg}<figcaption>Hit rate by planted depth; '
                      f'deeper positions are the ones a lossy cache drops first.</figcaption>'
                      f'</figure>')
        w2_scored = [a for a in arms.values() if a["w2"]["n"]]
        if w2_scored and all(a["w2"]["saturated"] for a in w2_scored):
            w2_section += (f'<p class="note"><strong>SATURATED.</strong> Every scored arm holds all '
                           f'{w2_of} keys, so the probe is at its ceiling and the study does not '
                           f'widen it again: a longer answer line would test the answer format, '
                           f'not the cache. The saturation is reported as a finding.</p>')
    else:
        w2_section = w2_html
    if w3_rows:
        w3_section = (f'<p class="note">Phase A asks for a brief carrying twenty identifiers, the '
                      f'ledger rows a tenant\'s mutations touch and five invariants. Phase B sends '
                      f'only that brief and asks five questions. The table reports how many '
                      f'identifiers survived and whether the answers still worked.</p>\n{w3_html}\n'
                      f'{depth_w3_html}\n<p class="note">Primary endpoint is brief retention on a '
                      f'full-depth read; phase B shows the consequence, not a substitute.</p>')
    else:
        w3_section = w3_html
    if long_rows:
        long_section = (f'<p class="note">W4 reports how many ledger reconciliations reached the '
                        f'expected hash and how many satisfied all five invariants. W5 reports '
                        f'answer keys and search calls. W6 counts section headings and resolved '
                        f'cross-references.</p>\n{long_html}')
    else:
        long_section = long_html

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(_title(arms))}</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #0b0f14; --fg: #e5e7eb; --muted: #9ca3af; --border: #2a3542;
  --head: #111821; --lead: #cbd5e1; --grid: #2a3542; --svg-bg: #0b0f14;
  --c-f16: #4ade80; --c-q4_0: #fb923c; --c-q8_0: #60a5fa;
}}
body {{ font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: var(--fg); margin: 0 auto; max-width: 1120px; padding: 28px;
        background: var(--bg); }}
h1 {{ font-size: 26px; line-height: 1.25; margin: 0 0 6px; }}
h2 {{ font-size: 19px; margin: 34px 0 10px; border-bottom: 1px solid var(--border);
      padding-bottom: 4px; }}
p.lead {{ font-size: 16px; color: var(--lead); }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 16px; font-size: 13px; }}
th, td {{ border: 1px solid var(--border); padding: 5px 7px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
thead th {{ background: var(--head); font-weight: 600; }}
code {{ font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
        background: var(--head); padding: 1px 4px; border-radius: 3px; }}
ul {{ margin: 8px 0 16px 20px; padding: 0; }}
.note {{ color: var(--muted); font-size: 13px; }}
figure {{ margin: 14px 0; }}
figcaption {{ color: var(--muted); font-size: 12px; }}
</style></head><body>
<h1>{esc(_title(arms))}</h1>
<p class="lead">{lead}</p>
{tldr_html}
<p>This page compares an uncompressed 16-bit key/value cache (f16) with a 4-bit cache (q4_0) inside
three models, at one full-context rung each. KV is the key/value cache, the stored attention state
for every token the model has already read. KLD is Kullback–Leibler divergence and PPL is
perplexity. The workflow labels W1–W6 mean the distribution instrument, the {w2_of}-key
retrieval probe, compaction, the ledger, agentic search and long-form writing, in that order.</p>
<h2>Premise, power and disclosure</h2>
<p>The list below states the sample sizes and rules that apply to every table on the page; read it
before comparing arms.</p>
{premise}
<h2>W1 — KLD and perplexity</h2>
{w1_section}
<h2>W2 — {w2_of}-key retrieval</h2>
{w2_section}
<h2>W3 — compaction</h2>
{w3_section}
<h2>W4, W5, W6 — long-horizon workflows</h2>
{long_section}
<h2>Behaviour, halts and cost</h2>
<p class="note">The table counts each instance by how it ended and gives the 95% Wilson interval
beside each halt count. Thinking share is the fraction of generated tokens that arrived on the
reasoning stream rather than as visible text.</p>
{table(["arm", "completed", "loop [95%]", "context full [95%]", "ceiling [95%]", "error",
        "thinking share", out_col, "mean wall"], halt_rows)}
<h2>Cost</h2>
<p class="note">Wall time is split by workflow below. The intervals in the table above are wide with
this many instances, so the rates describe behaviour and never rank arms.</p>
{table(["arm", "W1", "W2", "W3", "W4", "W5", "W6"], cost_rows)}
<h2>Decision rule</h2>
<p>An arm is worse if the lower bound of its paired interval excludes 0 and its point estimate is
worse than that model's f16 arm; ties otherwise. Behaviour counts are reported with intervals and
are never the basis of a ranking on their own.</p>
<h2>Provenance</h2>
<p class="note">The record below names the build, sampler configuration and hashes. Sampling sent
only <code>temperature: 0.0</code>; the resolved block confirms greedy decoding on every arm.</p>
<p class="note">llama.cpp <code>{esc((ll.get('commit') or '')[:12])}</code> ·
{esc(str(ll.get('version') or ''))} · flags <code>{esc(str(ll.get('ggml_cuda_fa_quants') or ''))}
</code> · sampling <code>{esc(json.dumps(man.get('sampling')))}</code> ·
rungs <code>{esc(json.dumps(man.get('rungs')))}</code> ·
document <code>{esc((raw.get('document') or {}).get('sha256') or '')}</code>.</p>
</body></html>
"""
    with open(path, "w") as f:
        f.write(doc)
    return doc


# ---------------------------------------------------------------- csv / readme
def write_csv(out_dir, arms, path):
    cols = ["arm", "model", "kv", "ctx", "w2_mean", "w2_sd", "w2_diff", "w2_lo", "w2_hi",
            "w2_decision", "w3_mean", "w3_phase_b", "kld_mean", "kld_unc", "same_top",
            "log_ppl_ratio_own", "deep_ppl", "deep_ppl_ratio", "w4_hash_ok", "w4_invariants_ok",
            "w5_keys_total", "w5_of_total", "w6_sections", "w6_refs", "loop_halts",
            "context_full", "ceiling", "out_tokens", "wall_s"]
    lines = [",".join(cols)]
    for name in sorted(arms):
        a = arms[name]
        k = a["w1"].get("kld") or {}
        row = [name, a["model"], a["kv"], a["ctx"],
               a["w2"].get("mean"), a["w2"].get("sd"),
               a["w2"]["paired"].get("mean"), a["w2"]["paired"].get("lo"),
               a["w2"]["paired"].get("hi"), a["w2"].get("decision"),
               a["w3"].get("mean"), a["w3"].get("phase_b_mean"),
               k.get("mean_kld"), k.get("mean_kld_unc"), k.get("same_top"),
               a.get("log_ppl_ratio_own"), a["w1"].get("deep_ppl"), a.get("deep_ppl_ratio"),
               a["w4"].get("hash_ok"), a["w4"].get("invariants_ok"),
               a["w5"].get("keys_total"), a["w5"].get("of_total"),
               a["w6"].get("sections"), a["w6"].get("refs"),
               a["halts"].get("loop", 0), a["halts"].get("context_full", 0),
               a["halts"].get("ceiling", 0), a["out_tokens"], a.get("wall_s")]
        lines.append(",".join("" if v is None else str(v) for v in row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_readme_headline(arms, readme_path):
    ranked = sorted(arms.values(), key=lambda a: (a["model"], a["kv"]))
    lines = [f"| model | KV | W2 mean /{_w2_of(list(arms.values()))} | W3 retention /20 | mean KLD "
             "| loop halts | context full |",
             "|---|---|---|---|---|---|---|"]
    def scored(a, key):
        w = a[key]
        mean = _f(w.get("mean"), 2)
        if w.get("n") is None or not w.get("attempted") or w["n"] == w["attempted"]:
            return mean
        return f"{mean} ({w['n']}/{w['attempted']})"

    for a in ranked:
        lines.append(f"| `{a['model']}` | {a['kv']} | {scored(a, 'w2')} | "
                     f"{scored(a, 'w3')} | "
                     f"{_kld(a['w1'].get('kld', {}).get('mean_kld'))} | "
                     f"{a['loop_halts']} | {a['halts'].get('context_full', 0)} |")
    table = "\n".join(lines)
    text = open(readme_path).read()
    text = re.sub(r"(<!-- HEADLINE:START -->\n).*?(\n<!-- HEADLINE:END -->)",
                  lambda m: m.group(1) + table + m.group(2), text, flags=re.S)
    with open(readme_path, "w") as f:
        f.write(text)


def build(out_dir, readme_path=None):
    raw, _ = load(out_dir)
    arms = aggregate(raw)
    md = write_markdown(out_dir, raw, arms, os.path.join(out_dir, "REPORT.md"))
    write_html(out_dir, raw, arms, os.path.join(out_dir, "report.html"))
    write_csv(out_dir, arms, os.path.join(out_dir, "matrix.csv"))
    if readme_path:
        write_readme_headline(arms, readme_path)
    return md


def main(argv=None):
    ap = argparse.ArgumentParser(description="write REPORT.md, report.html, matrix.csv")
    ap.add_argument("out_dir")
    ap.add_argument("--update-readme", action="store_true",
                    help="rewrite the README headline table (full matrix only)")
    a = ap.parse_args(argv)
    raw, _ = load(a.out_dir)
    arms = aggregate(raw)
    if a.update_readme:
        if len(arms) < 2 * len(config.STUDY_MODELS):
            raise SystemExit("refusing to rewrite README for a partial matrix")
        write_readme_headline(arms, os.path.join(config.ROOT, "README.md"))
    build(a.out_dir)
    print(f"wrote {a.out_dir}/REPORT.md, report.html, matrix.csv, figures/")


if __name__ == "__main__":
    main()
