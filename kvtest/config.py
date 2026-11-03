"""Single source of truth for the full-context KV-quant study.

One rung per model, greedy decode, no reasoning budget, six workflows, fixed tranches. Everything
that changes what is measured lives here.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("KV_LLAMA_BIN", "/home/kome/Projects/code/clones/llama.cpp/build/bin")
LLAMA_DIR = os.environ.get("LLAMA_DIR", "/home/kome/Projects/code/clones/llama.cpp")
MODELS_ROOT = os.environ.get("KV_MODELS_ROOT", "/home/kome/.local/share/llm-models")
FIXTURES = os.path.join(ROOT, "fixtures")

SERVER_BIN = os.path.join(BASE, "llama-server")
TOKENIZE_BIN = os.path.join(BASE, "llama-tokenize")
PERPLEXITY_BIN = os.path.join(BASE, "llama-perplexity")

# D2: the largest ladder entry that loads with a >= 91k-token prompt. The ladder is only used to
# report which rungs a model can reach; every arm of a model runs at that model's resolved rung.
CONTEXT_LADDER = [102400, 98304, 90112, 81920, 73728, 65536, 57344, 49152]
CONTEXT_FLOOR = 90112
KV_ARMS = ["f16", "q4_0"]
PORT = 18400

# D5: temperature 0 and nothing else. No top_k, top_p, min_p, repeat penalty, seeds or reps.
SAMPLING = {"temperature": 0.0}
# D6: the only cap left, and only as a runaway guard.
MAX_OUTPUT_TOKENS = 65536

MODELS = {
    "gemma4-26b-a4b": dict(
        path=f"{MODELS_ROOT}/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_0.gguf",
        knobs={"ncmoe": 8}, knob_flag="--n-cpu-moe",
        knob_bump=lambda n: n + 6, knob_max=30,
        extra=["-ub", "512"], priority=1, tranche="T1"),
    "ornith-35b": dict(
        path=f"{MODELS_ROOT}/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf",
        knobs={"ncmoe": 10}, knob_flag="--n-cpu-moe",
        knob_bump=lambda n: n + 6, knob_max=40,
        extra=["-ub", "1024"], priority=2, tranche="T2"),
    "qwen36-35b": dict(
        path=f"{MODELS_ROOT}/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        knobs={"ncmoe": 26}, knob_flag="--n-cpu-moe",
        knob_bump=lambda n: n + 4, knob_max=40,
        extra=["-ub", "1024"], priority=3, tranche="T3"),
    # Dropped by D7, kept so they can be run later on the two cheap probes.
    "qwen38-27b": dict(
        path=f"{MODELS_ROOT}/Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-MTP.gguf",
        knobs={"ncffn": 8}, knob_flag="--n-cpu-ffn",
        knob_bump=lambda n: n + 6, knob_max=44,
        extra=["-ub", "512"], priority=4, tranche=None),
    "qwen38-flash": dict(
        path=(f"{MODELS_ROOT}/Qwen3.8-Flash-Next-GSQ-RCO/"
              "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
        knobs={"ncmoe": 48}, knob_flag="--n-cpu-moe",
        knob_bump=lambda n: n + 4, knob_max=64,
        extra=["--moe-cache-mib", "2900", "-lm", "mmap", "-lzm", "on", "-ub", "1024"],
        priority=5, tranche=None),
}
STUDY_MODELS = [k for k, _ in sorted(MODELS.items(), key=lambda kv: kv[1]["priority"])
                if MODELS[k].get("tranche")]

# -fa on is mandatory: a quantized V cache refuses to load without flash attention. No reasoning
# budget, no reasoning flags, no sampler flags: the model file's own defaults are never overridden.
# -v makes llama-server print the resolved sampler chain and params, which is captured per arm.
SERVER_COMMON = ["-ngl", "999", "-fa", "on", "-np", "1", "-b", "2048",
                 "--jinja", "--no-warmup", "--metrics", "--cache-reuse", "0",
                 "--ctx-checkpoints", "8", "--checkpoint-min-step", "8192", "-v"]

DOC_SEED = 7
# D2: the document alone is 92-95k tokens in the LARGEST of the three active tokenizers.
DOC_FILLER_TARGET = 400000
DOC_TARGET_TOKENS = 94000
DOC_TARGET_MIN = 92000
DOC_TARGET_MAX = 95000

ANSWER_INSTRUCTION = (
    "Answer with a single final line of exactly this form, and nothing after it:\n"
    "ANSWER: <value>"
)

# ---------------------------------------------------------------- workflows
# Instances are per arm and set by information per hour (D1/D3 and the workflow table).
WORKFLOWS = {
    "W1": dict(name="KLD and perplexity", instances=3, guard=None, tier="instrument"),
    "W2": dict(name="20-key retrieval", instances=12, guard="1 request", tier="counted"),
    "W3": dict(name="Compaction", instances=12, guard="2 requests", tier="counted"),
    "W4": dict(name="Long-horizon ledger", instances=6, guard=120, tier="long"),
    "W5": dict(name="Agentic search", instances=6, guard=60, tier="long"),
    "W6": dict(name="Unbounded long-form", instances=6, guard=None, tier="long"),
}
WORKFLOW_IDS = ["W1", "W2", "W3", "W4", "W5", "W6"]

# Fixed in advance: a deadline cut lands on the least informative remaining cell.
TRANCHES = [
    dict(id="T1", model="gemma4-26b-a4b", workflows=["W1", "W2", "W3", "W4", "W5", "W6"],
         hours=5.2, decision="whole result on the model with the only prior large effect"),
    dict(id="T2", model="ornith-35b", workflows=["W1", "W2", "W3", "W4", "W5", "W6"],
         hours=5.0, decision="clean second architecture, and the long workflows finish"),
    dict(id="T3", model="qwen36-35b", workflows=["W1", "W2", "W3"],
         hours=2.0, decision="counted probes and instrument on a third architecture"),
    dict(id="T4", model="qwen36-35b", workflows=["W4", "W5", "W6"],
         hours=4.0, decision="long-horizon workflows on the third model, only on slack"),
]

# ---------------------------------------------------------------- compaction (D3)
BRIEF_WORD_CAP = 1500
COMPACT_WALL_FRACTION = 0.90      # model-driven variant: compact before 90% of the window

# ---------------------------------------------------------------- KLD/PPL (D4)
# The f16 logits baseline is tokens x vocabulary x 2 bytes. The 4k scored slice keeps it ~1.2 GB,
# and the file is written outside the results tree.
KLD_CACHE = os.environ.get("KV_KLD_CACHE", os.path.join(ROOT, ".cache", "kv-quant-test"))
KLD_SLICE_TOKENS = 4000           # scored context; the file carries >= 2x that (tool guard)
KLD_CTX = 4000
DEEP_PPL_CTX = 65536              # tool needs ctx/2 x vocab x 4 B of host logits; 102400 aborts

# ---------------------------------------------------------------- statistics
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261101


def lock():
    p = os.path.join(ROOT, "models.lock.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def model_sha(key):
    return ((lock().get("models") or {}).get(key) or {}).get("sha256")


def build_manifest():
    """Provenance block embedded in raw.json and both reports."""
    lk = lock()
    return {
        "llama_cpp": lk.get("llama_cpp", {}),
        "models": {k: {"path": v.get("path"), "sha256": v.get("sha256"), "bytes": v.get("bytes")}
                   for k, v in (lk.get("models") or {}).items()},
        "document_sha256": (lk.get("document") or {}).get("sha256"),
        "kv_arms": KV_ARMS,
        "server_common": SERVER_COMMON,
        "sampling": SAMPLING,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_budget": None,
        "workflows": WORKFLOWS,
        "tranches": TRANCHES,
        "kld_slice_tokens": KLD_SLICE_TOKENS,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "host": (lock().get("host") or {}),
    }
