"""Fit probe: resolve each model's one full-context rung, its offload knobs, and the document.

D2 replaces the shared ladder with one rung per model: the largest ladder entry that loads with a
>= 91k-token prompt. Every arm of that model runs at that rung; a model that cannot reach the floor
is reported as unfit rather than quietly compared at a shallower depth.

Passes:
1. walk the ladder top-down per model with a cheap ~32k probe, bumping the offload knob on an
   allocation failure;
2. calibrate the shared document so its LARGEST active tokenization is 92-95k tokens;
3. prefill the real document at each model's own rung, which proves the depth and measures rates.
"""
import json
import os
import time

from . import client, config, ggufmeta, longctx, server as srv


def log(msg):
    print(msg, flush=True)


def doc_tokens(text, models=None):
    out = {}
    for key in (models or config.STUDY_MODELS):
        path = config.MODELS[key]["path"]
        if not os.path.exists(path):
            out[key] = {"error": "missing model file"}
            continue
        try:
            n, _ = client.token_count(config.TOKENIZE_BIN, path, text)
            out[key] = n
        except Exception as e:                                     # noqa: BLE001
            out[key] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def calibrate_document(models, lo=config.DOC_TARGET_MIN, hi=config.DOC_TARGET_MAX,
                       target=config.DOC_TARGET_TOKENS, max_iter=6, start=None):
    """Find the filler length whose largest tokenization lands in [lo, hi]."""
    filler = start or config.DOC_FILLER_TARGET
    text, facts = longctx.build_document(filler_chars=filler)
    hist = []
    for _ in range(max_iter):
        counts = doc_tokens(text, models)
        nums = [v for v in counts.values() if isinstance(v, int)]
        if not nums:
            hist.append({"filler_chars": filler, "counts": counts, "note": "no tokenizer"})
            break
        mx = max(nums)
        hist.append({"filler_chars": filler, "chars": len(text), "max_tokens": mx,
                     "min_tokens": min(nums), "counts": counts})
        if lo <= mx <= hi:
            return filler, text, facts, hist
        goal = target if mx > hi else (lo + hi) / 2
        filler = max(2000, int(filler * goal / mx * (0.997 if mx > hi else 1.01)))
        text, facts = longctx.build_document(filler_chars=filler)
    return filler, text, facts, hist


def tiny_probe(base, timeout=600):
    return client.chat(base, [{"role": "user", "content": "Reply with one short sentence."}],
                       max_tokens=64, timeout=timeout, **config.SAMPLING)


def depth_probe(base, doc, max_tokens=32, timeout=7200, prompt_suffix=None):
    suffix = prompt_suffix if prompt_suffix is not None else "\n\nReply with the single word: ACK."
    return client.chat(base, [{"role": "user", "content": doc + suffix}],
                       max_tokens=max_tokens, timeout=timeout, **config.SAMPLING)


def probe_model(key, ladder, probe_doc, port=config.PORT, history=None):
    """First rung (largest) that loads and prefills the probe; bump the knob within a rung."""
    m = config.MODELS[key]
    knob_name = "ncmoe" if "ncmoe" in m["knobs"] else "ncffn"
    attempts = []
    for rung in ladder:
        if rung < config.CONTEXT_FLOOR:
            break
        knob = dict(m["knobs"])
        while True:
            t0 = time.time()
            log_path = os.path.join(history, f"fit__{key}__{rung}__{knob.get(knob_name, 0)}.log") \
                if history else None
            s = srv.Server(key, "f16", rung, knobs=knob, port=port, log_path=log_path)
            ok = s.start()
            entry = {"model": key, "ctx": rung, "knobs": dict(knob), "load_s": s.start_s}
            if not ok:
                entry.update(ok=False, reason=s.oom_reason() or "did not become healthy",
                             log_tail=s.log_tail(1200))
                attempts.append(entry)
                s.stop()
                log(f"[fit] {key:16} ctx={rung:6} {knob_name}={knob.get(knob_name, 0):3} "
                    f"FAILED to start")
            else:
                r = tiny_probe(s.base)
                if r.get("error"):
                    entry.update(ok=False, reason=f"tiny probe: {r['error'][:160]}")
                    attempts.append(entry)
                    s.stop()
                    log(f"[fit] {key:16} ctx={rung:6} up but request failed")
                else:
                    d = depth_probe(s.base, probe_doc, max_tokens=16)
                    entry["probe_tokens"] = (d.get("usage") or {}).get("prompt_tokens")
                    if d.get("error"):
                        entry.update(ok=False, reason=f"depth probe: {d['error'][:160]}")
                        attempts.append(entry)
                        s.stop()
                        log(f"[fit] {key:16} ctx={rung:6} {entry['probe_tokens']}-token "
                            "prefill failed")
                    else:
                        entry.update(ok=True, probe_prefill_tps=d.get("prefill_tps"),
                                     probe_decode_tps=d.get("decode_tps"),
                                     wall_s=round(time.time() - t0, 1),
                                     vram_peak_mib=s.vram.peak)
                        attempts.append(entry)
                        s.stop()
                        log(f"[fit] {key:16} ctx={rung:6} OK  prefill "
                            f"{entry['probe_prefill_tps']} t/s  vram {entry['vram_peak_mib']} MiB")
                        return entry, attempts
            bump = m["knob_bump"](knob.get(knob_name, 0))
            if bump <= m["knob_max"]:
                knob = {**knob, knob_name: bump}
                continue
            break
    return None, attempts


def depth_verify(key, ctx, knobs, doc, port=config.PORT, log_dir=None, retries=3):
    m = config.MODELS[key]
    knob_name = "ncmoe" if "ncmoe" in m["knobs"] else "ncffn"
    knob = dict(knobs)
    attempts = []
    for _ in range(retries):
        s = srv.Server(key, "f16", ctx, knobs=knob, port=port,
                       log_path=os.path.join(log_dir, f"fit__depth__{key}.log") if log_dir else None)
        if not s.start():
            attempts.append({"knobs": knob, "ok": False, "reason": s.oom_reason()})
            s.stop()
        else:
            big = depth_probe(s.base, doc, max_tokens=32)
            entry = {"knobs": dict(knob), "ok": not big.get("error"),
                     "prompt_tokens": (big.get("usage") or {}).get("prompt_tokens"),
                     "prefill_tps": big.get("prefill_tps"), "decode_tps": big.get("decode_tps"),
                     "ttft_s": round((big.get("timings") or {}).get("prompt_ms", 0) / 1000.0, 1),
                     "vram_peak_mib": s.vram.peak, "error": big.get("error")}
            attempts.append(entry)
            s.stop()
            if entry["ok"]:
                log(f"[fit] {key:16} depth {entry['prompt_tokens']} tok  prefill "
                    f"{entry['prefill_tps']} t/s  decode {entry['decode_tps']} t/s")
                return entry, attempts
            log(f"[fit] {key:16} depth prefill FAILED: {(entry['error'] or '')[:120]}")
        bump = m["knob_bump"](knob.get(knob_name, 0))
        if bump > m["knob_max"]:
            break
        knob = {**knob, knob_name: bump}
    return None, attempts


def geometry_and_sampling(models):
    geo, sampling, warnings = {}, {}, []
    for key in models:
        path = config.MODELS[key]["path"]
        if not os.path.exists(path):
            geo[key] = {"error": "missing model file"}
            continue
        g, warn = ggufmeta.verify_geometry(key, path, None)
        geo[key] = g
        sampling[key] = ggufmeta.declared_sampling(path)
        if warn:
            warnings.append(warn)
        log(f"[fit] geometry {key:16} arch={g['arch']:10} layers={g['n_layers']} "
            f"full_attn={g['full_attn_layers']} kv_heads={g['n_kv_heads']} "
            f"B/token f16/q4_0={g['kv_bytes_per_token']['f16']}/"
            f"{g['kv_bytes_per_token']['q4_0']}")
        log(f"[fit] declared sampling {key:16} {sampling[key]}")
    return geo, sampling, warnings


def run(stamp=None, models=None, out_dir=None, resume=False):
    stamp = stamp or time.strftime("%Y-%m-%dT%H:%M")
    out_dir = out_dir or os.path.join(config.ROOT, "results", stamp)
    logs = os.path.join(out_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    models = models or config.STUDY_MODELS
    t0 = time.time()
    fit = {"stamp": stamp, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "ladder": config.CONTEXT_LADDER, "context_floor": config.CONTEXT_FLOOR,
           "models": {}, "unfit": [], "attempts": []}
    log(f"[fit] free VRAM {srv.gpu_mib()} MiB")
    fit["geometry"], fit["declared_sampling"], fit["geometry_warnings"] = \
        geometry_and_sampling(list(config.MODELS))

    prev = {}
    prev_path = os.path.join(out_dir, "fit.json")
    if resume and os.path.exists(prev_path):
        prev = json.load(open(prev_path))
        fit["models"] = dict(prev.get("models") or {})
        fit["attempts"] = list(prev.get("attempts") or [])
        fit["unfit"] = list(prev.get("unfit") or [])

    present = [k for k in config.STUDY_MODELS if os.path.exists(config.MODELS[k]["path"])]
    pf, probe_doc, _, probe_hist = calibrate_document(present, lo=28000, hi=36000,
                                                      target=32000, start=120000)
    fit["probe_document"] = {"filler_chars": pf, "tokens": probe_hist[-1]["counts"]
                             if probe_hist else {}}
    log(f"[fit] probe document {len(probe_doc)} chars")

    for key in models:
        if not os.path.exists(config.MODELS[key]["path"]):
            fit["unfit"].append(key)
            log(f"[fit] {key}: model file missing, excluded")
            continue
        old = (prev.get("models") or {}).get(key)
        if old and old.get("ok"):
            fit["models"][key] = old
            log(f"[fit] {key}: reusing rung={old['ctx']} knobs={old['knobs']}")
            continue
        best, attempts = probe_model(key, config.CONTEXT_LADDER, probe_doc, history=logs)
        fit["attempts"].extend(attempts)
        if best is None:
            fit["unfit"].append(key)
            log(f"[fit] {key}: UNFIT at every rung down to {config.CONTEXT_FLOOR}, excluded")
            continue
        fit["models"][key] = best

    if not fit["models"]:
        fit["wall_s"] = round(time.time() - t0, 1)
        _write(prev_path, fit)
        return fit

    filler, doc, facts, hist = calibrate_document(present)
    fit["document"] = {"filler_chars": filler, "chars": len(doc), "sha256": longctx.sha256(doc),
                       "tokens_by_model": hist[-1]["counts"] if hist else {},
                       "calibration": hist,
                       "target_range": [config.DOC_TARGET_MIN, config.DOC_TARGET_MAX]}
    log(f"[fit] document {len(doc)} chars sha256={fit['document']['sha256'][:16]} "
        f"tokens={fit['document']['tokens_by_model']}")
    with open(os.path.join(out_dir, "document.txt"), "w") as f:
        f.write(doc)
    _record_document_hash(fit["document"])

    for key in [k for k in fit["models"] if k in models]:
        entry = fit["models"][key]
        res, attempts = depth_verify(key, entry["ctx"], entry["knobs"], doc, log_dir=logs)
        entry["depth_attempts"] = attempts
        entry["reverified"] = bool(res)
        if res:
            entry["knobs"] = res["knobs"]
            entry.update(prompt_tokens=res["prompt_tokens"], prefill_tps=res["prefill_tps"],
                         decode_tps=res["decode_tps"], ttft_s=res["ttft_s"],
                         vram_peak_mib=res["vram_peak_mib"])
        else:
            fit.setdefault("reverify_failed", []).append(key)
            log(f"[fit] {key}: could not prefill the document at {entry['ctx']}")

    fit["wall_s"] = round(time.time() - t0, 1)
    _write(prev_path, fit)
    log(f"[fit] wrote {out_dir}/fit.json in {fit['wall_s']}s")
    return fit


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _record_document_hash(doc):
    path = os.path.join(config.ROOT, "models.lock.json")
    lock = json.load(open(path)) if os.path.exists(path) else {}
    lock["document"] = {"sha256": doc["sha256"], "chars": doc["chars"],
                        "filler_chars": doc["filler_chars"],
                        "tokens_by_model": doc["tokens_by_model"]}
    with open(path, "w") as f:
        json.dump(lock, f, indent=2, sort_keys=True)
        f.write("\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="resolve each model's full-context rung")
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--models", default=None, help="comma-separated model keys")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    models = a.models.split(",") if a.models else None
    fit = run(a.stamp, models=models, resume=a.resume)
    print(json.dumps({"rung": {k: v.get("ctx") for k, v in fit["models"].items()},
                      "unfit": fit["unfit"], "wall_s": fit.get("wall_s")}, indent=2))


if __name__ == "__main__":
    main()
