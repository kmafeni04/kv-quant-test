"""Fit probe: resolve the ONE shared context, per-model offload knobs, and the document size.

Three passes, because the document length depends on the context and the context depends on what
fits:

1. `probe_model` walks the context ladder top-down with a cheap probe (a tiny completion, then a
   ~16k-token prefill at depth) and bumps the model's offload knob on an allocation failure.
   The f16 arm is the only one probed: it is the most expensive KV cache and q8_0/q4_0 are strictly
   smaller, and every arm is health-checked again when the run starts it.
2. `common_ctx = min(highest passing rung per model)`, and the document is calibrated so the
   LARGEST of the five tokenizations fits inside `common_ctx - GEN_HEADROOM`.
3. `depth_verify` prefills the real document at `common_ctx` on every model, which is what proves
   the depth works and produces the prefill/decode rates the deadline estimate uses.

The knobs that worked are frozen per model so all three KV arms of a model are identical.
"""
import json
import os
import time

from . import client, config, ggufmeta, longctx, server as srv


def log(msg):
    print(msg, flush=True)


def doc_tokens(text, models=None):
    out = {}
    for key in (models or list(config.MODELS)):
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


def calibrate_document(target, models=None, max_iter=5, start=None):
    """Find the filler length whose LARGEST tokenization is <= target.

    The text is canonical: identical for every model. Only its length is chosen here, so the
    comparison across arms and models is one text, not five variants.
    """
    filler = start or config.DOC_FILLER_TARGET
    hist = []
    text, facts = longctx.build_document(filler_chars=filler)
    for _ in range(max_iter):
        counts = doc_tokens(text, models)
        nums = [v for v in counts.values() if isinstance(v, int)]
        if not nums:
            hist.append({"filler_chars": filler, "counts": counts, "note": "no tokenizer"})
            break
        mx = max(nums)
        hist.append({"filler_chars": filler, "chars": len(text), "max_tokens": mx,
                     "min_tokens": min(nums), "counts": counts})
        if mx <= target:
            return filler, text, facts, hist
        filler = max(2000, int(filler * target / mx * 0.97))
        text, facts = longctx.build_document(filler_chars=filler)
    return filler, text, facts, hist


def tiny_probe(base, timeout=600):
    return client.chat(base, [{"role": "user", "content": "Reply with one short sentence."}],
                       max_tokens=64, temperature=0.0, timeout=timeout)


def depth_probe(base, doc, max_tokens=32, timeout=7200, prompt_suffix=None):
    """Prefill the document, then generate a few tokens at that depth."""
    suffix = prompt_suffix if prompt_suffix is not None else "\n\nReply with the single word: ACK."
    return client.chat(base, [{"role": "user", "content": doc + suffix}],
                       max_tokens=max_tokens, temperature=0.0, timeout=timeout)


def probe_model(key, ladder, probe_doc, port=config.PORT, history=None):
    """Try rungs top-down, bumping the offload knob when allocation at the rung fails."""
    m = config.MODELS[key]
    knob_name = "ncmoe" if "ncmoe" in m["knobs"] else "ncffn"
    attempts = []
    for rung in ladder:
        knob = dict(m["knobs"])
        while True:
            t0 = time.time()
            log_path = os.path.join(history, f"fit__{key}__{rung}__{knob.get(knob_name, 0)}.log") \
                if history else None
            s = srv.Server(key, "f16", rung, knobs=knob, port=port, log_path=log_path)
            ok = s.start()
            entry = {"model": key, "ctx": rung, "knobs": dict(knob), "load_s": s.start_s}
            if not ok:
                reason = s.oom_reason() or "did not become healthy"
                entry.update(ok=False, reason=reason, log_tail=s.log_tail(1500))
                attempts.append(entry)
                s.stop()
                log(f"[fit] {key:14} ctx={rung:6} {knob_name}={knob.get(knob_name, 0):3} "
                    f"FAILED to start ({reason})")
                break
            r = tiny_probe(s.base)
            entry["tiny_ok"] = not r.get("error")
            entry["tiny_error"] = r.get("error")
            if r.get("error"):
                attempts.append(entry)
                s.stop()
                log(f"[fit] {key:14} ctx={rung:6} {knob_name}={knob.get(knob_name, 0):3} "
                    f"up but request failed: {r['error'][:140]}")
                break
            d = depth_probe(s.base, probe_doc, max_tokens=16)
            entry["probe_tokens"] = (d.get("usage") or {}).get("prompt_tokens")
            if d.get("error"):
                entry.update(ok=False, reason=f"depth probe failed: {d['error'][:160]}")
                attempts.append(entry)
                s.stop()
                log(f"[fit] {key:14} ctx={rung:6} {knob_name}={knob.get(knob_name, 0):3} "
                    f"{entry['probe_tokens']}-token prefill failed")
                break
            entry.update(ok=True, probe_prefill_tps=d.get("prefill_tps"),
                         probe_decode_tps=d.get("decode_tps"),
                         wall_s=round(time.time() - t0, 1), vram_peak_mib=s.vram.peak)
            attempts.append(entry)
            s.stop()
            log(f"[fit] {key:14} ctx={rung:6} {knob_name}={knob.get(knob_name, 0):3} OK   "
                f"probe prefill {entry['probe_prefill_tps']} t/s  "
                f"vram {entry['vram_peak_mib']} MiB")
            return entry, attempts
        bump = m["knob_bump"](knob.get(knob_name, 0))
        if bump <= m["knob_max"]:
            knob = {**knob, knob_name: bump}
            continue
        break
    return None, attempts


def depth_verify(key, ctx, knobs, doc, port=config.PORT, log_dir=None, retries=3):
    """Prefill the real document at the shared context; bump the knob if allocation fails."""
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
                     "vram_peak_mib": s.vram.peak,
                     "error": big.get("error"), "log_tail": None if not big.get("error")
                     else s.log_tail(1200)}
            attempts.append(entry)
            s.stop()
            if entry["ok"]:
                log(f"[fit] {key:14} depth {entry['prompt_tokens']} tok  "
                    f"prefill {entry['prefill_tps']} t/s  decode {entry['decode_tps']} t/s  "
                    f"vram {entry['vram_peak_mib']} MiB")
                return entry, attempts
            log(f"[fit] {key:14} depth prefill FAILED: {(entry['error'] or '')[:120]}")
        bump = m["knob_bump"](knob.get(knob_name, 0))
        if bump > m["knob_max"]:
            break
        knob = {**knob, knob_name: bump}
    return None, attempts


def verify_geometry_all():
    planned = {"qwen36-35b": 20480, "qwen38-27b": 65536, "qwen38-flash": 24576,
               "gemma4-26b-a4b": 20480, "ornith-35b": 20480}
    out, warnings = {}, []
    for key, m in config.MODELS.items():
        if not os.path.exists(m["path"]):
            out[key] = {"error": "missing model file"}
            continue
        g, warn = ggufmeta.verify_geometry(key, m["path"], planned.get(key))
        out[key] = g
        if warn:
            warnings.append(warn)
            log(f"[fit] WARN {warn}")
        log(f"[fit] geometry {key:14} arch={g['arch']:10} layers={g['n_layers']} "
            f"full_attn={g['full_attn_layers']} kv_heads={g['n_kv_heads']} "
            f"B/token f16/q8_0/q4_0 = {g['kv_bytes_per_token']['f16']}/"
            f"{g['kv_bytes_per_token']['q8_0']}/{g['kv_bytes_per_token']['q4_0']}")
    return out, warnings


def run(stamp, ladder=None, models=None, out_dir=None, resume=False):
    stamp = stamp or time.strftime("%Y-%m-%dT%H:%M")
    out_dir = out_dir or os.path.join(config.ROOT, "results", stamp)
    logs = os.path.join(out_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    ladder = ladder or config.CONTEXT_LADDER
    models = models or [k for k, _ in sorted(config.MODELS.items(),
                                             key=lambda kv: kv[1]["priority"])]

    t0 = time.time()
    fit = {"stamp": stamp, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "ladder": ladder, "models": {}, "unfit": [], "attempts": [],
           "free_vram_mib_at_start": srv.gpu_mib()}
    log(f"[fit] free VRAM {fit['free_vram_mib_at_start']} MiB")

    fit["geometry"], warns = verify_geometry_all()
    fit["geometry_warnings"] = warns

    prev = {}
    prev_path = os.path.join(out_dir, "fit.json")
    if resume and os.path.exists(prev_path):
        prev = json.load(open(prev_path))
        log(f"[fit] resuming: {len(prev.get('models') or {})} models already probed")

    # ---- pass 1: context + knobs, with a small probe document (cheap at every rung)
    present = [k for k in models if os.path.exists(config.MODELS[k]["path"])]
    probe_target = min(32000, min(ladder) - config.GEN_HEADROOM - 512)
    pf, probe_doc, _, probe_hist = calibrate_document(probe_target, models=present,
                                                     start=120000)
    fit["probe_document"] = {"filler_chars": pf, "tokens": probe_hist[-1]["counts"]
                             if probe_hist else {}}
    log(f"[fit] probe document {len(probe_doc)} chars "
        f"~{probe_hist[-1].get('max_tokens') if probe_hist else '?'} tokens")

    for key in models:
        if not os.path.exists(config.MODELS[key]["path"]):
            fit["unfit"].append(key)
            log(f"[fit] {key}: model file missing, excluded")
            continue
        old = (prev.get("models") or {}).get(key)
        if old and old.get("ok"):
            fit["models"][key] = old
            log(f"[fit] {key}: reusing ctx={old['ctx']} knobs={old['knobs']} from previous probe")
            continue
        best, attempts = probe_model(key, ladder, probe_doc, history=logs)
        fit["attempts"].extend(attempts)
        if best is None:
            fit["unfit"].append(key)
            log(f"[fit] {key}: UNFIT at every rung down to {ladder[-1]}, excluded")
            continue
        fit["models"][key] = best

    fit["common_ctx"] = min((fit["models"][k]["ctx"] for k in fit["models"]), default=None)
    log(f"[fit] common_ctx = {fit['common_ctx']}")
    if fit["common_ctx"] is None:
        fit["wall_s"] = round(time.time() - t0, 1)
        with open(prev_path, "w") as f:
            json.dump(fit, f, indent=2, sort_keys=True)
            f.write("\n")
        return fit

    # ---- pass 2: calibrate the REAL document against the resolved context
    target = min(config.DOC_TARGET_TOKENS, fit["common_ctx"] - config.GEN_HEADROOM - 512)
    filler, doc, facts, hist = calibrate_document(target, models=present)
    fit["document"] = {"filler_chars": filler, "chars": len(doc), "sha256": longctx.sha256(doc),
                       "tokens_by_model": hist[-1]["counts"] if hist else {},
                       "calibration": hist, "target_tokens": target,
                       "fwe_count": facts["fwe_count"]}
    log(f"[fit] document {len(doc)} chars sha256={fit['document']['sha256'][:16]} "
        f"tokens={fit['document']['tokens_by_model']}")
    with open(os.path.join(out_dir, "document.txt"), "w") as f:
        f.write(doc)
    _record_document_hash(fit["document"])

    # ---- pass 3: prove the real depth on every model and measure speed there
    for key in list(fit["models"]):
        entry = fit["models"][key]
        res, attempts = depth_verify(key, fit["common_ctx"], entry["knobs"], doc,
                                    log_dir=logs)
        entry["depth_attempts"] = attempts
        entry["reverified"] = bool(res)
        if res:
            entry["knobs"] = res["knobs"]
            entry.update(prompt_tokens=res["prompt_tokens"], prefill_tps=res["prefill_tps"],
                         decode_tps=res["decode_tps"], ttft_s=res["ttft_s"],
                         vram_peak_mib=res["vram_peak_mib"], ctx=fit["common_ctx"])
        else:
            fit.setdefault("reverify_failed", []).append(key)
            log(f"[fit] {key}: could not prefill the document at {fit['common_ctx']}")

    fit["wall_s"] = round(time.time() - t0, 1)
    with open(prev_path, "w") as f:
        json.dump(fit, f, indent=2, sort_keys=True)
        f.write("\n")
    log(f"[fit] wrote {out_dir}/fit.json in {fit['wall_s']}s")
    return fit


def _record_document_hash(doc):
    path = os.path.join(config.ROOT, "models.lock.json")
    lock = {}
    if os.path.exists(path):
        lock = json.load(open(path))
    lock["document"] = {"sha256": doc["sha256"], "chars": doc["chars"],
                        "filler_chars": doc["filler_chars"],
                        "tokens_by_model": doc["tokens_by_model"]}
    with open(path, "w") as f:
        json.dump(lock, f, indent=2, sort_keys=True)
        f.write("\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="resolve the shared context and per-model offload")
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--ladder", type=int, default=config.CONTEXT_LADDER[0],
                    help="top of the context ladder")
    ap.add_argument("--models", default=None, help="comma-separated model keys")
    ap.add_argument("--resume", action="store_true",
                    help="reuse models already probed in results/<stamp>/fit.json")
    a = ap.parse_args(argv)
    ladder = [c for c in config.CONTEXT_LADDER if c <= a.ladder]
    models = a.models.split(",") if a.models else None
    fit = run(a.stamp, ladder=ladder, models=models, resume=a.resume)
    print(json.dumps({k: fit.get(k) for k in ("common_ctx", "unfit", "wall_s")}, indent=2))


if __name__ == "__main__":
    main()
