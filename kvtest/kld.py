"""W1: KLD and perplexity through `llama-perplexity`.

Three passes per arm (D4):
  1. f16 baseline logits, once per model, over a fixed scored slice;
  2. every arm, `--kl-divergence` against that baseline on the same slice;
  3. every arm, perplexity at depth with no KLD file.

The tool scores the last half of each context window and its logits path wants at least 2x the
context in tokens, so slices are built to satisfy that guard and their sha256 and token counts are
recorded. The baseline file is kept outside the results tree.
"""
import json
import os
import re
import subprocess
import time

from . import config, longctx, server as srv

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _slice_targets(ctx):
    """(tokens in the file, tokens actually scored) for one perplexity window."""
    return 2 * ctx + 2048, ctx // 2 - 1


def build_slice(doc, model_path, target_tokens, path, tokens_per_char=None, tokenize=None):
    """Write a text prefix long enough to tokenize to >= target_tokens under this model."""
    from . import client
    tokenize = tokenize or client.token_count
    if tokens_per_char is None:
        n, _ = tokenize(config.TOKENIZE_BIN, model_path, doc)
        tokens_per_char = n / max(len(doc), 1)
    chars = int(target_tokens / max(tokens_per_char, 1e-6) * 1.06) + 100
    text = (doc * (chars // max(len(doc), 1) + 1))[:chars]
    tokens = 0
    for _ in range(4):
        tokens, _ = tokenize(config.TOKENIZE_BIN, model_path, text)
        if tokens >= target_tokens:
            break
        chars = int(chars * (target_tokens / max(tokens, 1)) * 1.04) + 200
        text = (doc * (chars // max(len(doc), 1) + 1))[:chars]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return dict(path=path, sha256=longctx.sha256(text), chars=len(text), tokens=tokens,
                target_tokens=target_tokens, offset=0)


def slice_meta(path):
    if not os.path.exists(path + ".json"):
        return None
    try:
        return json.load(open(path + ".json"))
    except json.JSONDecodeError:
        return None


def _run(cmd, log_path, timeout=14400):
    # A wedged loader can hold the device for minutes after its process was signalled.
    srv.wait_vram_free(timeout=600)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "wb") as f:
        f.write(("# " + " ".join(cmd) + "\n").encode())
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 60000))
        tail = f.read().decode("utf-8", "replace")
    return p.returncode, tail, round(time.time() - t0, 1)


def _pairs(pattern, text):
    m = re.search(pattern, text)
    if not m:
        return None, None
    return float(m.group(1)), (float(m.group(2)) if m.lastindex and m.lastindex >= 2 else None)


def parse_kld(text):
    out = {}
    names = {
        "mean_kld": r"Mean\s+KLD:\s*(" + NUM + r")\s*±\s*(" + NUM + r")",
        "mean_dp": r"Mean\s+Δp:\s*(" + NUM + r")\s*±\s*(" + NUM + r")",
        "rms_dp": r"RMS\s+Δp\s*:\s*(" + NUM + r")\s*±\s*(" + NUM + r")",
        "same_top": r"Same top p:\s*(" + NUM + r")\s*±\s*(" + NUM + r")",
        "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*(" + NUM + r")\s*±\s*(" + NUM + r")",
        "log_ppl_ratio": r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*(" + NUM + r")\s*±\s*("
                         + NUM + r")",
        "ppl": r"Mean PPL\(Q\)\s*:\s*(" + NUM + r")\s*±\s*(" + NUM + r")",
        "ppl_base": r"Mean PPL\(base\)\s*:\s*(" + NUM + r")\s*±\s*(" + NUM + r")",
    }
    for key, pat in names.items():
        val, unc = _pairs(pat, text)
        out[key] = val
        out[key + "_unc"] = unc
    if out["same_top"] is not None:
        out["same_top"] /= 100.0
        if out["same_top_unc"] is not None:
            out["same_top_unc"] /= 100.0
    return out


def parse_ppl(text):
    m = re.search(r"Final estimate: PPL = (" + NUM + r")\s*\+/-\s*(" + NUM + r")", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"\[(\d+)\](" + NUM + r")", text)
    if m:
        return float(m.group(2)), None
    return None, None


def base_cmd(model, kv, knobs=None):
    m = config.MODELS[model]
    cmd = [config.PERPLEXITY_BIN, "-m", m["path"], "-ctk", kv, "-ctv", kv,
           "-fa", "on", "-ngl", "999"]
    cmd += list(m.get("extra") or [])
    knobs = dict(m["knobs"] if knobs is None else knobs)
    if knobs.get("ncmoe"):
        cmd += ["--n-cpu-moe", str(knobs["ncmoe"])]
    if knobs.get("ncffn"):
        cmd += ["--n-cpu-ffn", str(knobs["ncffn"])]
    return cmd


def baseline_path(model):
    doc_sha = (config.lock().get("document") or {}).get("sha256") or "unknown"
    return os.path.join(config.KLD_CACHE, doc_sha[:16], f"{model}__kld{config.KLD_CTX}.kld")


def _file_sha256(path, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_baseline(model, slice_meta_, log_dir, knobs=None, timeout=14400):
    """f16 logits baseline, once per model, kept out of the results tree."""
    path = baseline_path(model)
    meta_path = path + ".json"
    if os.path.exists(path) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta.get("slice_sha256") == slice_meta_["sha256"] and meta.get("ctx") == config.KLD_CTX:
            if _file_sha256(path) == meta.get("sha256"):
                return dict(meta, cached=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = base_cmd(model, "f16", knobs) + [
        "-c", str(config.KLD_CTX), "--chunks", "1", "-f", slice_meta_["path"],
        "--save-all-logits", path]
    rc, tail, wall = _run(cmd, os.path.join(log_dir, f"w1__{model}__f16__baseline.log"),
                          timeout=timeout)
    if rc != 0 or not os.path.exists(path):
        return dict(error=f"baseline rc={rc}", tail=tail[-1200:])
    meta = dict(model=model, ctx=config.KLD_CTX, slice_sha256=slice_meta_["sha256"],
                slice_tokens=slice_meta_["tokens"], path=path,
                bytes=os.path.getsize(path), sha256=_file_sha256(path),
                wall_s=wall, cached=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta


def run_arm(model, kv, ctx, slice4k, deep_slice, log_dir, timeout=14400, deep_ctx=None,
            knobs=None, deep=True):
    """Passes 2 and 3 for one arm. Returns the arm's W1 record."""
    deep_ctx = deep_ctx or ctx
    rec = dict(model=model, kv=kv, kld_ctx=config.KLD_CTX, deep_ctx=deep_ctx,
               scored_tokens=config.KLD_CTX // 2 - 1, deep_scored_tokens=deep_ctx // 2 - 1,
               slice4k={k: slice4k[k] for k in ("sha256", "tokens", "target_tokens", "chars")},
               deep_slice=({k: deep_slice[k] for k in ("sha256", "tokens", "target_tokens",
                                                       "chars")}
                           if deep_slice else None))
    base = ensure_baseline(model, slice4k, log_dir, knobs=knobs, timeout=timeout)
    rec["baseline"] = {k: base.get(k) for k in ("path", "sha256", "bytes", "ctx",
                                                 "slice_sha256", "slice_tokens", "cached",
                                                 "error")}
    if base.get("error"):
        rec["error"] = base["error"]
        rec["tail"] = base.get("tail")
        return rec
    cmd = base_cmd(model, kv, knobs) + [
        "-c", str(config.KLD_CTX), "--chunks", "1", "-f", slice4k["path"],
        "--kl-divergence", "--kl-divergence-base", base["path"]]
    rc, tail, wall = _run(cmd, os.path.join(log_dir, f"w1__{model}__{kv}__kld.log"),
                          timeout=timeout)
    rec["kld"] = parse_kld(tail)
    rec["kld_wall_s"] = wall
    rec["kld_rc"] = rc
    if rc != 0 or rec["kld"].get("mean_kld") is None:
        rec["kld_error"] = f"rc={rc}"
        rec["kld_tail"] = tail[-1500:]
    if not deep:
        rec["deep_skipped"] = True
        rec["wall_s"] = round(wall, 1)
        return rec
    cmd2 = base_cmd(model, kv, knobs) + [
        "-c", str(deep_ctx), "--chunks", "1", "-f", deep_slice["path"]]
    rc2, tail2, wall2 = _run(cmd2, os.path.join(log_dir, f"w1__{model}__{kv}__ppl.log"),
                             timeout=timeout)
    ppl, ppl_unc = parse_ppl(tail2)
    rec["deep_ppl"] = ppl
    rec["deep_ppl_unc"] = ppl_unc
    rec["deep_wall_s"] = wall2
    rec["deep_rc"] = rc2
    if rc2 != 0 or ppl is None:
        rec["deep_error"] = f"rc={rc2}"
        rec["deep_tail"] = tail2[-1500:]
    rec["wall_s"] = round(wall + wall2, 1)
    return rec


def prepare_slices(doc, model, ctx, out_dir, deep_ctx=None, deep=True):
    deep_ctx = deep_ctx or ctx
    t4, _ = _slice_targets(config.KLD_CTX)
    from . import client
    n, _ = client.token_count(config.TOKENIZE_BIN, config.MODELS[model]["path"], doc)
    tpc = n / max(len(doc), 1)
    s4 = build_slice(doc, config.MODELS[model]["path"], t4,
                     os.path.join(out_dir, "kld", f"{model}__4k.txt"), tokens_per_char=tpc)
    with open(s4["path"] + ".json", "w") as f:
        json.dump(s4, f, indent=2, sort_keys=True)
    if not deep:
        return s4, None
    td, _ = _slice_targets(deep_ctx)
    sd = build_slice(doc, config.MODELS[model]["path"], td,
                     os.path.join(out_dir, "kld", f"{model}__deep.txt"), tokens_per_char=tpc)
    with open(sd["path"] + ".json", "w") as f:
        json.dump(sd, f, indent=2, sort_keys=True)
    return s4, sd
