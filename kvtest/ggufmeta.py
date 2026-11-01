"""Minimal pure-stdlib GGUF metadata + tensor-size reader.

Needed because numpy is absent on this host and gguf-py imports it. Covers f32/f16/bf16 and the
k-/i-quant block types, which is enough to reconstruct KV geometry and total tensor bytes.
"""
import struct

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 9: None, 10: 8, 11: 8, 12: 8}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# ggml type -> (block_elems, bytes_per_block)
TYPES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
         9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
         14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98),
         19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
         24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 2), 29: (1, 4),
         30: (1, 2), 34: (256, 54), 35: (256, 66)}


def _s(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def _v(f, t):
    if t == 8:
        return _s(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            return [_s(f) for _ in range(n)]
        return list(struct.unpack(f"<{n}{FMT[et]}", f.read(n * SIZES[et])))
    return struct.unpack("<" + FMT[t], f.read(SIZES[t]))[0]


def read(path):
    """-> (version, n_tensors, kv_dict, [(name, nbytes), ...])"""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, = struct.unpack("<I", f.read(4))
        nt, nkv = struct.unpack("<QQ", f.read(16))
        kv = {}
        for _ in range(nkv):
            k = _s(f)
            t, = struct.unpack("<I", f.read(4))
            kv[k] = _v(f, t)
        tens = []
        for _ in range(nt):
            name = _s(f)
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            tt, = struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))
            n = 1
            for d in dims:
                n *= d
            bs, bb = TYPES.get(tt, (1, 0))
            tens.append((name, n if bb == 0 else (n // bs) * bb))
        return ver, nt, kv, tens


def geometry(path):
    """KV bytes/token for the three arms, derived from GGUF metadata.

    Full-attention layers only: layers under a sliding window cap their cache at the window, so
    they do not grow with context. `full_attention_interval` (qwen35*), or the sliding window
    pattern (gemma4), decides which layers those are.
    """
    ver, nt, kv, tens = read(path)
    arch = kv.get("general.architecture", "")

    def m(name, default=None):
        if f"{arch}.{name}" in kv:
            return kv[f"{arch}.{name}"]
        return kv.get(name, default)

    n_layers = m("block_count")
    n_kv = m("attention.head_count_kv", 0)
    n_head = m("attention.head_count", 1) or 1
    n_kv_layers = n_kv if isinstance(n_kv, list) else [n_kv] * n_layers
    key_len = m("attention.key_length") or m("embedding_length") // max(n_head, 1)
    val_len = m("attention.value_length") or key_len
    interval = m("full_attention_interval")
    swe = m("attention.sliding_window")
    pattern = m("attention.sliding_window_pattern")
    if interval:
        full = [i for i in range(n_layers) if (i + 1) % interval == 0]
    elif pattern:
        full = [i for i, v in enumerate(pattern) if not v]
    else:
        full = list(range(n_layers))
    per_token_f16 = sum(n_kv_layers[i] * (key_len + val_len) * 2 for i in full)
    return {
        "file": path, "gguf_version": ver, "n_tensors": nt, "n_layers": n_layers,
        "n_kv_heads": n_kv, "n_heads": n_head, "key_length": key_len,
        "value_length": val_len, "full_attn_layers": len(full), "sliding_window": swe,
        "full_attention_interval": interval,
        "kv_bytes_per_token": {"f16": per_token_f16,
                               "q8_0": per_token_f16 * 34 // 64,
                               "q4_0": per_token_f16 * 18 // 64},
        "context_length": m("context_length"),
        "tensor_bytes": sum(n for _, n in tens),
        "rope_freq_base": m("rope.freq_base"), "arch": arch,
        "model_name": kv.get("general.name"),
    }


def verify_geometry(key, path, expected_f16_bytes, tolerance=0.05):
    """Compare measured KV bytes/token with the plan's table. Warns, does not raise."""
    g = geometry(path)
    got = g["kv_bytes_per_token"]["f16"]
    warn = None
    if expected_f16_bytes and abs(got - expected_f16_bytes) > tolerance * expected_f16_bytes:
        warn = (f"{key}: measured KV f16 {got} B/token vs planned {expected_f16_bytes} B/token "
                f"({100.0 * (got - expected_f16_bytes) / expected_f16_bytes:+.1f}%); "
                "using the measured value")
    return g, warn


if __name__ == "__main__":
    import json
    import sys
    for p in sys.argv[1:]:
        print(json.dumps(geometry(p), indent=2))
