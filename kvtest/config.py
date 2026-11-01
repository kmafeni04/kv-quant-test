"""Single source of truth for the KV-quant x long-context study.

Every number that changes what is measured lives here: model paths, offload knobs, the context
ladder, sampling, budgets, seeds, the shared document parameters.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "/home/kome/Projects/code/clones/llama.cpp/build/bin"
LLAMA_DIR = "/home/kome/Projects/code/clones/llama.cpp"
MODELS_ROOT = os.environ.get("KV_MODELS_ROOT", "/home/kome/.local/share/llm-models")
FIXTURES = os.path.join(ROOT, "fixtures")

SERVER_BIN = os.path.join(BASE, "llama-server")
TOKENIZE_BIN = os.path.join(BASE, "llama-tokenize")

CONTEXT_LADDER = [102400, 98304, 90112, 81920, 73728, 65536, 57344, 49152]
KV_ARMS = ["f16", "q8_0", "q4_0"]
GEN_HEADROOM = 8192          # a -c value must exceed prompt + this
PORT = 18400

# Per-arm offload knob. `knob_bump` is applied when the server cannot allocate at the current
# rung; `knob_max` is where the rung is abandoned. Fixed per model across the three KV arms,
# because moving layers on and off the GPU between arms is not the variable under test.
MODELS = {
    "qwen36-35b": dict(
        path=f"{MODELS_ROOT}/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        knobs={"ncmoe": 26}, knob_flag="--n-cpu-moe",
        knob_bump=lambda n: n + 4, knob_max=40,
        extra=["-ub", "1024"], priority=1),
    "qwen38-27b": dict(
        path=f"{MODELS_ROOT}/Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-MTP.gguf",
        knobs={"ncffn": 8}, knob_flag="--n-cpu-ffn",
        knob_bump=lambda n: n + 6, knob_max=44,
        extra=["-ub", "512"], priority=2),
    "qwen38-flash": dict(
        path=(f"{MODELS_ROOT}/Qwen3.8-Flash-Next-GSQ-RCO/"
              "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
        knobs={"ncmoe": 48}, knob_flag="--n-cpu-moe",
        knob_bump=lambda n: n + 4, knob_max=64,
        extra=["--moe-cache-mib", "2900", "-lm", "mmap", "-lzm", "on", "-ub", "1024"],
        priority=3),
    "gemma4-26b-a4b": dict(
        path=f"{MODELS_ROOT}/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_0.gguf",
        knobs={"ncmoe": 8}, knob_flag="--n-cpu-moe",
        knob_bump=lambda n: n + 6, knob_max=30,
        extra=["-ub", "512"], priority=4),
    "ornith-35b": dict(
        path=f"{MODELS_ROOT}/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf",
        knobs={"ncmoe": 10}, knob_flag="--n-cpu-moe",
        knob_bump=lambda n: n + 6, knob_max=40,
        extra=["-ub", "1024"], priority=5),
}

# -fa on is mandatory: a quantized V cache refuses to load without flash attention. The build's
# GGML_CUDA_FA_QUANTS covers exactly f16/f16, q8_0/q8_0, q4_0/q4_0.
SERVER_COMMON = ["-ngl", "999", "-fa", "on", "-np", "1", "-b", "2048",
                 "--jinja", "--no-warmup", "--metrics",
                 "--cache-reuse", "0", "--ctx-checkpoints", "8", "--checkpoint-min-step", "8192",
                 "--reasoning", "on", "--reasoning-effort", "medium",
                 "--reasoning-budget", "8192", "--reasoning-format", "deepseek"]

SAMPLING_GRADED = dict(temperature=0.0, top_k=20, top_p=0.95, min_p=0.0,
                       repeat_penalty=1.05, repeat_last_n=64, dry_multiplier=0.0)
# Repetition suppression off: a repeat penalty is exactly the thing that masks looping.
SAMPLING_LOOP = dict(temperature=0.7, top_k=20, top_p=0.95, min_p=0.0,
                     repeat_penalty=1.0, repeat_last_n=0, dry_multiplier=0.0)
# L was 1,536 in the design; the measured 40-shard listing ran out of room at exactly 1,536 tokens
# (88 lines emitted, 39/40 shards present, finish_reason=length), so 2,048 is used for every arm so
# that a complete answer can fit. A truncation artefact must not read as "did no work".
BUDGETS = {"R": 2048, "C": 6144, "L": 2048, "tool_turn": 512}
# Thinking is capped per REQUEST (`reasoning_budget_tokens`) so the answer always has room: with an
# uncapped medium-effort template the code tier spent all 6,144 tokens thinking and emitted zero
# characters of code (measured twice on the dense 27B, and the earlier 32k study hit the same wall
# at 32,489 completion tokens with 0 lines). The cap is identical for every arm, and the sampler
# forces the end-of-thinking tag when it is reached, so the model still answers.
REASONING_BUDGET_TOKENS = {"R": 1024, "C": 3072, "L": 256, "tool_turn": 256}
# A thinking model can spend the whole budget before it answers. When a sample ends at the cap
# with no answer, run.py retries that one sample at this multiple and logs `budget_raised`; the
# default path stays at the budgets above so the run stays inside its deadline.
BUDGET_RETRY_MULT = 4
REPS = 3
SEEDS = [1001, 1002, 1003]

DOC_SEED = 7
DOC_TARGET_TOKENS = 96000     # <= this for the LARGEST of the five tokenizations
DOC_FILLER_TARGET = 430000    # characters of filler before task blocks; recalibrated in fit

ANSWER_INSTRUCTION = (
    "Answer with a single final line of exactly this form, and nothing after it:\n"
    "ANSWER: <value>"
)


def lock():
    p = os.path.join(ROOT, "models.lock.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def build_manifest():
    """Provenance block embedded in raw.json and both reports."""
    lk = lock()
    return {
        "llama_cpp": lk.get("llama_cpp", {}),
        "models": {k: {"path": v.get("path"), "sha256": v.get("sha256"), "bytes": v.get("bytes")}
                   for k, v in (lk.get("models") or {}).items()},
        "document_sha256": (lk.get("document") or {}).get("sha256"),
        "context_ladder": CONTEXT_LADDER,
        "kv_arms": KV_ARMS,
        "server_common": SERVER_COMMON,
        "sampling_graded": SAMPLING_GRADED,
        "sampling_loop": SAMPLING_LOOP,
        "budgets": BUDGETS,
        "reasoning_budget_tokens": REASONING_BUDGET_TOKENS,
        "reps": REPS,
        "seeds": SEEDS,
        "host": "RTX 3080 Laptop 16384 MiB / 64 GiB RAM",
    }
