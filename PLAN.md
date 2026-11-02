# KV-cache quant × long-context: 5 models at f16 / q8_0 / q4_0 through llama.cpp

Goal: at one identical long context, for each model, measure (a) whether hard, automatically
verifiable tasks get worse as the KV cache, meaning the stored attention state for every token already
read, is quantized from f16 to q8_0 to q4_0, and (b) whether the model starts **looping**, which here
means repeating the same words, commands, tool calls or actions without doing work. Deliver a runnable
harness, the results, and a single-file HTML report, and make the whole thing reproducible by a third
party.

Everything below was verified on this machine unless marked **UNVERIFIED** (those are listed in
§12 and are the points where a smaller model must not assume).

---

## 0. Environment facts (verified, use verbatim in the manifest)

| fact | value |
|---|---|
| project dir (new, empty) | `/home/kome/Projects/kv-quant-test` |
| host | Arch Linux, kernel 7.2.11, 16 cores, 62 GiB RAM, **240 GB free on /home** |
| GPU | RTX 3080 Laptop, **16,023 MiB**, sm_86; driver 615.78.08; **CUDA 13.4**, meaning version 13.4 of NVIDIA's GPU compute toolkit |
| llama.cpp | `/home/kome/Projects/code/clones/llama.cpp`, commit `2a667032dc788c3a071d538e0b572e45a3f12731`, version `0.7.0-dev (build 11698)`, clean tree |
| binaries | `.../llama.cpp/build/bin/{llama-server,llama-tokenize,llama-perplexity,llama-bench}` |
| build flags that matter | `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, **`GGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`**, `GGML_CUDA_FA=ON`, `GGML_CUDA_FA_ALL_QUANTS=OFF` |
| python | 3.14.8, **stdlib only** (numpy/pytest are NOT installed — do not add dependencies) |
| GPU/RAM currently held by | `/home/kome/Projects/code/clones/Strata/engine/strata --serve --pack /home/kome/.local/share/strata/packs/iq3_xxs --native /home/kome/.local/share/llm-models/Qwen3.8-Flash-Next-GSQ-RCO/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf …-00002-of-00002.gguf --expert-profile …/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp /home/kome/.local/share/strata/mtp/rt --max-context 262144 --kv fp16 --kv-resident 32768 --vision --vram-reserve-mib 3072` — pid **2556064**, 43.5 GB RSS |

The `GGML_CUDA_FA_QUANTS` line is load-bearing: the three KV arms in this study are exactly the
combinations the CUDA flash-attention kernels were compiled for. A rebuild with different settings
changes both memory and numerics, so it is pinned in the lock file.

Prior art to read before writing code (do not skip):
`/home/kome/.local/share/llm-models/logs/kv-quant-27b-2026-10-10.md` (the earlier 27B study at
32k — capability read flat, only KL separated the quants) and
`/home/kome/.local/share/llm-models/bench/` (the previous harness: `bench.py`, `codegen-run.py`,
`repo-run.py`, `fixtures/codegen/*`, `fixtures/repo/*`). This plan reuses that harness's fixtures
and grader contract and deliberately *does not* modify it.

---

## 1. Resolved decisions (do not re-litigate; each has a reason)

1. **Arms = 5 models × 3 symmetric KV types = 15.** f16/f16, q8_0/q8_0, q4_0/q4_0. Symmetric only
   (asymmetric K/V is not a production setting here). `-fa on` is mandatory: quantized V cache
   refuses to load without flash attention (`src/llama-context.cpp` logs
   "quantized V cache requires flash_attn to be enabled").
2. **Speculative decoding / MTP is OFF for every arm.** Speculative decoding verifies against the
   target model and cannot change the distribution, so it is irrelevant to the question, it costs
   VRAM that the f16 arm needs, and the Gemma/Ornith MTP drafters would be extra downloads.
   Do not pass `--spec-type`.
3. **Gemma 4 31B is dropped** — consequence of your rule "dense models: pick the quant that fits
   VRAM". Numbers: 10 global attention layers × 4 KV heads × 512 dim × K+V × 2 B = **80 KiB/token
   while the cache is at f16, which becomes 7.8 GiB at 100k tokens**, plus 838 MiB of sliding-window
   caches, plus 18.0 GB of Q4_0
   weights against a 15.3 GiB budget. No published quant of that model (smallest: unsloth Q3_K_S,
   13.2 GB) fits at 100k with an f16 KV cache. It comes back only if the common context drops to
   ~65k; record that in `docs/RESULTS.md` as a conscious exclusion.
4. **Weights per model**: the dense 27B keeps IQ3_XXS, which is already on disk, and because the model
   is dense its full weight set has to fit; the mixture-of-experts models may be larger, because their
   expert weights spill to RAM through `--n-cpu-moe`.
5. **Common context = one number for every arm**, resolved empirically by the fit probe (§5) from
   the ladder `102400, 98304, 90112, 81920, 73728, 65536, 57344, 49152`. Target is 102,400; the
   binding constraint is the dense 27B at f16 (prior measurement: 81,920 max with *no* offload;
   the probe tries `-ncffn` to reach 102,400 first). Every arm uses the same `-c`. What "context"
   means in the report: the `-c` value (identical everywhere) plus the realized
   `usage.prompt_tokens` (differs per tokenizer; recorded, not equalized).
6. **Reasoning on, `--reasoning-effort medium --reasoning-budget 8192`** for the graded tiers.
   If a model's template ignores it, the arm logs `reasoning_unsupported` and runs thinking-off;
   reported as a caveat, never silently.
7. **Repetition suppression is off inside the loop probes** (`repeat_penalty 1.0`, `dry_multiplier 0`)
   because a repetition penalty is exactly the thing that masks looping. The graded tiers use the
   production value `1.05`. Both are per-request body fields — verified the server accepts
   `repeat_penalty`, `dry_multiplier`, `seed`, `top_k`, `min_p` per request — so one server per arm
   serves both.
8. **Reps**: `--reps 3` on the sampled tiers (R and L, temp 0.7, seeds 1001/1002/1003);
   `--code-reps 1` greedy on the code tier, because temperature 0 on a fixed build and fixed
   `-ub` is bit-reproducible and repeating it would only burn hours. `--code-reps 3 --code-temp 0.7`
   exists for anyone who wants pass@k. This is written up in `docs/METHOD.md` so the choice is
   auditable, not hidden.
9. **Budget**: expected 15–18 h for the full matrix, hard governor `--deadline-hours 22`, matrix
   resumable per arm, Ornith last. Wall clock over 12 h is accepted.
10. **Never touch `/home/kome/.local/share/llm-models/models.ini`.** It is the production preset
    (27B at 128k, q4_0 KV, MTP) and this experiment must not disturb it. All flags live in
    `kvtest/config.py`.
11. **No network at run time.** Model downloads happen once, up front, with sha256 verification.
    After that the harness runs offline: the report is an HTML file opened from `file://` with every
    asset inlined, so it needs no CDN, meaning no content-delivery network to fetch scripts or fonts.

---

## 2. Model matrix

| key | file | arch | weights | full-attn layers | f16 KV @102,400 | q8_0 | q4_0 | offload knob |
|---|---|---|---|---|---|---|---|---|
| `qwen36-35b` | `Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | qwen35moe | 22.85 GB / 21.3 GiB | 10 (of 41) | 1.95 GiB | 1.04 | 0.55 | `-ncmoe` |
| `qwen38-27b` | `Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-MTP.gguf` | qwen35 (hybrid SSM) | 10.44 GB / 10.36 GiB | 16 (of 65) | **6.25 GiB** | 3.30 | 1.76 | `-ncffn` |
| `qwen38-flash` | `Qwen3.8-Flash-Next-GSQ-RCO/…-IQ3_XXS-00001-of-00002.gguf` | qwen4exp | 75.8 GB / 70.6 GiB | 12 (of 48) | 2.34 GiB | 1.24 | 0.66 | `-ncmoe 48 --moe-cache-mib 2900 -lm mmap -lzm on` |
| `gemma4-26b-a4b` | `gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_0.gguf` | gemma4 MoE | 14.6 GB / 13.6 GiB | 5 full + 25 swa | 1.95 + 0.21 | 1.04 | 0.55 | `-ncmoe` |
| `ornith-35b` | `Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf` | qwen35moe | 21.7 GB / 20.2 GiB | 10 (of 41) | 1.95 GiB | 1.04 | 0.55 | `-ncmoe` |

KV bytes/token = `full_attn_layers × kv_heads × head_dim × 2 (K,V) × bytes_per_elem`; q8_0 is
34/64 of f16 and q4_0 is 18/64 (this reproduces the earlier measured 64/34/18 KiB/token on the
27B). Recompute the Ornith row from its GGUF at download time (`kvtest/config.py` has a
`verify_geometry()` helper that parses GGUF metadata with the pure-stdlib reader given in §5.3 —
it must agree with the table within 5%, otherwise print a warning and use the measured value).

**Downloads (exact URLs; no weights for vision, none needed):**

```
H=https://huggingface.co
$H/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q4_0.gguf     # 14,618,145,824 B
$H/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-Q4_K_M.gguf      # 21,713,463,040 B
```

Land them in `/home/kome/.local/share/llm-models/<repo-dir>/` alongside the existing models.
`scripts/download_models.sh` works through each file in three steps. It skips the download when a file
of the right size already exists; otherwise it fetches with `curl -L --retry 5 -C -`. It then compares
`sha256sum` against `models.lock.json` and writes `<file>.sha256.ok` on a match. Total transfer is
about 36 GB, which fits comfortably in the 240 GB free on this host.

---

## 3. Repository layout (create exactly this)

```
/home/kome/Projects/kv-quant-test/
├── README.md                     # what/why, 60-second quickstart, headline results table
├── models.lock.json              # pinned models + sha256 + llama.cpp commit + build flags
├── kvtest/
│   ├── __init__.py
│   ├── config.py                 # MODELS, KV_ARMS, ladder, budgets, seeds, prompts
│   ├── ggufmeta.py               # stdlib GGUF metadata+tensor-size reader (§5.3)
│   ├── server.py                 # llama-server lifecycle, health, log capture, VRAM
│   ├── client.py                 # chat completion: content, reasoning_content, usage, timings
│   ├── longctx.py                # deterministic long document (§6)
│   ├── tasks.py                  # R and L tiers: prompts + graders (§7)
│   ├── codegraders.py            # C tier: codegen hidden checks + repo fix (§8)
│   ├── loops.py                  # loop/degeneration metrics (§9)
│   ├── fit.py                    # context + offload probe (§5)
│   ├── run.py                    # orchestrator: resumable, deadline, JSONL (§10)
│   ├── report.py                 # raw -> REPORT.md + report.html (§11)
│   └── selftest.py               # mock-server end-to-end, no GPU (§12)
├── fixtures/
│   ├── codegen/{csvlite,fixedpoint,intervals,semver,statistics}/{README.md,checks.json}
│   ├── repo/repofixture/{README.md,test_visible.py,checks.json,intervalkit/*.py}
│   ├── codegen-run.py            # copied verbatim
│   └── repo-run.py               # copied verbatim
├── scripts/
│   ├── build_llama.sh            # pinned clone+build for third parties
│   ├── download_models.sh
│   ├── preflight.sh              # GPU/RAM/disk checks + Strata capture/stop
│   └── run_all.sh                # orchestration entry point + Strata restart trap
├── docs/
│   ├── METHOD.md                 # hypotheses, design, threats to validity
│   ├── REPRODUCE.md              # third-party reproduction, hardware classes, expected output
│   └── RESULTS.md                # written findings (filled after the run)
└── results/<UTC-stamp>/
    ├── fit.json  raw.json  samples.jsonl  matrix.csv
    ├── REPORT.md  report.html
    ├── arms/<model>__<kv>.json   # per-arm checkpoint (resume unit)
    └── logs/<model>__<kv>.log    # llama-server stderr
```

Copy the fixtures with `cp -a /home/kome/.local/share/llm-models/bench/fixtures/{codegen,repo} fixtures/`
and the two runners from `bench/`. They are the user's own files; keep them verbatim so results
stay comparable to the earlier 32k numbers.

---

## 4. `kvtest/config.py` — the single source of truth

```python
BASE = "/home/kome/Projects/code/clones/llama.cpp/build/bin"
MODELS_ROOT = "/home/kome/.local/share/llm-models"

CONTEXT_LADDER = [102400, 98304, 90112, 81920, 73728, 65536, 57344, 49152]
KV_ARMS = ["f16", "q8_0", "q4_0"]          # run in this order; f16 is the reference
GEN_HEADROOM = 8192                         # -c must exceed prompt + this

MODELS = {
  "qwen36-35b": dict(
      path=f"{MODELS_ROOT}/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
      knobs={"ncmoe": 26}, knob_bump=lambda n: n + 4, knob_max=40, knob_flag="--n-cpu-moe",
      extra=["-ub", "1024"], priority=1),
  "qwen38-27b": dict(..., knobs={"ncffn": 8}, knob_flag="--n-cpu-ffn",
                     knob_bump=lambda n: n + 6, knob_max=44, extra=["-ub", "512"], priority=2),
  "qwen38-flash": dict(..., knobs={"ncmoe": 48}, extra=["--moe-cache-mib", "2900",
                     "-lm", "mmap", "-lzm", "on", "-ub", "1024"], priority=3),
  "gemma4-26b-a4b": dict(..., knobs={"ncmoe": 8}, knob_flag="--n-cpu-moe",
                     knob_bump=lambda n: n + 6, knob_max=30, extra=["-ub", "512"], priority=4),
  "ornith-35b": dict(..., knobs={"ncmoe": 10}, knob_flag="--n-cpu-moe",
                     knob_bump=lambda n: n + 6, knob_max=40, extra=["-ub", "1024"], priority=5),
}

SERVER_COMMON = ["-ngl", "999", "-fa", "on", "-np", "1", "-b", "2048",
                 "--jinja", "--no-warmup", "--metrics",
                 "--cache-reuse", "0", "--ctx-checkpoints", "8", "--checkpoint-min-step", "8192",
                 "--reasoning", "on", "--reasoning-effort", "medium",
                 "--reasoning-budget", "8192", "--reasoning-format", "deepseek"]

SAMPLING_GRADED = dict(temperature=0.0, top_k=20, top_p=0.95, min_p=0.0,
                       repeat_penalty=1.05, repeat_last_n=64, dry_multiplier=0.0)
SAMPLING_LOOP   = dict(temperature=0.7, top_k=20, top_p=0.95, min_p=0.0,
                       repeat_penalty=1.0,  repeat_last_n=0,  dry_multiplier=0.0)
BUDGETS = {"R": 2048, "C": 6144, "L": 1536, "tool_turn": 512}
REPS = 3
SEEDS = [1001, 1002, 1003]
```

`fit.json` may override the `knobs` after the probe; if it does, write the resolved values back
into `results/<stamp>/fit.json` and use those for every KV arm of that model (fixed knobs per
model is what keeps the three KV arms comparable — changing offload per arm would move attention
layers on and off the GPU, which is not the variable under test).

---

## 5. Fit probe — resolve the one shared context (`kvtest/fit.py`)

Algorithm, in order:

1. For each model, **only the f16 arm** is probed (it is the most expensive KV; q8/q4 are strictly
   smaller, and each arm is still health-checked when the run starts it).
2. Start at `CONTEXT_LADDER[0]` = 102400 and the model's starting knobs.
3. Call `start_server(...)` and wait for `GET /health` to return `{"status":"ok"}`, with a 600 s
   timeout. Then send one 64-token completion; success means HTTP 200 and `usage.prompt_tokens > 0`.
4. On failure, read the server stderr. If it contains `out of memory`, `failed to allocate` or
   `CUDA error`, or the process exited, bump the knob with `knob_bump` and retry, up to `knob_max`.
   When the knob is exhausted, drop one rung, reset the knob to its start value, and repeat.
5. If a model fails at 49152, mark it `unfit` and **exclude it from the matrix** (the common
   context is defined over the arms that run; log loudly).
6. `common_ctx = min(highest passing rung per model)`; re-verify every model at `common_ctx`.
7. Record per model: rung attempted, knob used, peak VRAM (sample `nvidia-smi --query-gpu=memory.used`
   every 1 s while the server is up), time to first token at that depth.

Expected outcomes to sanity-check against: the 35B, Flash-Next, Gemma-26B and Ornith pass at 102400
with a handful of expert layers offloaded. The **27B at f16 needs about 2.6 GiB moved off the GPU**,
because 6.25 GiB of KV, 10.36 GiB of weights, 0.58 GiB of recurrent state and roughly 0.7 GiB of compute
add up to 17.9 GiB against a budget of about 15.3 GiB. That puts `--n-cpu-ffn` around 24 to 32, so
prefill will be slower than the production preset. If even `-ncffn 44` fails, the ladder drops the whole
matrix to 98304, then 90112, then **81920**, which was the f16 ceiling in the earlier study without
offload. Whatever rung it lands on, every arm uses that value.

Also probe the *decode rate* while you are there (the tiny request plus a 64-token generation) and
write it to `fit.json`; the orchestrator uses it to estimate whether the next arm fits under the
deadline.

### 5.3 `kvtest/ggufmeta.py` (pure stdlib, ~60 lines, already written and tested in this session)

Needed because numpy is absent and `gguf-py` imports it. Copy this working implementation:

```python
import struct
SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,8:None,9:None,10:8,11:8,12:8}
FMT   = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
# ggml type -> (block_elems, bytes_per_block); covers f32/f16/bf16 and all k-quants/i-quants
TYPES = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),
         10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),
         16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),21:(256,110),
         22:(256,82),23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,2),29:(1,4),
         30:(1,2),34:(256,54),35:(256,66)}

def _s(f):
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n).decode('utf-8','replace')

def _v(f, t):
    if t == 8: return _s(f)
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]; n = struct.unpack('<Q', f.read(8))[0]
        if et == 8: return [_s(f) for _ in range(n)]
        return list(struct.unpack(f'<{n}{FMT[et]}', f.read(n*SIZES[et])))
    return struct.unpack('<'+FMT[t], f.read(SIZES[t]))[0]

def read(path):
    """-> (version, n_tensors, kv_dict, [(name, nbytes), ...])"""
    with open(path,'rb') as f:
        assert f.read(4) == b'GGUF'
        ver, = struct.unpack('<I', f.read(4))
        nt, nkv = struct.unpack('<QQ', f.read(16))
        kv = {}
        for _ in range(nkv):
            k = _s(f); t, = struct.unpack('<I', f.read(4)); kv[k] = _v(f, t)
        tens = []
        for _ in range(nt):
            name = _s(f); nd, = struct.unpack('<I', f.read(4))
            dims = struct.unpack(f'<{nd}Q', f.read(8*nd))
            tt, = struct.unpack('<I', f.read(4)); struct.unpack('<Q', f.read(8))
            n = 1
            for d in dims: n *= d
            bs, bb = TYPES.get(tt, (1, 0))
            tens.append((name, n if bb == 0 else (n//bs)*bb))
        return ver, nt, kv, tens
```

Use it to `verify_geometry()`: read `block_count`, `attention.head_count_kv`,
`attention.key_length`, `full_attention_interval` (or `attention.sliding_window_pattern` for
gemma4) and compare the resulting KV bytes/token with the table in §2.

---

## 6. `kvtest/longctx.py` — the one document every task appends to

The whole point of the design: **one 100k-token document per arm, all 12 tasks ask about it, and
llama-server reuses the KV prefix for every subsequent request** (verified: `server-context.cpp`
computes `get_common_prefix(...)` and rewinds `n_past` when `cache_prompt` is set and the task is
not stateless). So the 100k prefill is paid **once per arm**, not once per task, and reps are
nearly free. Watch `timings.cache_n` in every response: on request #2..N it must be ≈ the document
length, else log `cache_miss` (the run is still correct, just slower).

`build_document(seed=7, target_tokens=96_000)` returns a `str`:

- Filler: deterministic word soup from a 400-word vocabulary, RNG seeded — ~400k chars.
- Planted blocks at fixed *fractions* of the final text (so they land at known depths):
  - 40 × `SHARD <name> code=<XX-####> custodian=<emp>` at 0.02…0.98. Names from a fixed list
    (`orion, vega, lyra, draco, cygnus, …`), codes unique.
  - 120 × `LEDGER <L-####> status=<OPEN|CLOSED|VOID> tag=<RED|BLUE|GOLD> amount=<int>` at 0.01…0.99.
  - 60 × `RECORD <EMP-####> name=<slug> manager=<EMP-####|NONE> badge=<B-####> project=<PRJ-##>`
    forming real 4-hop chains.
  - 12 numbered `RULE <n>: <sentence>` at 0.50…0.60.
  - The **code section at 0.85**: for the `repo_fix` task, the full `intervalkit` package dump
    (`### FILE: intervalkit/core.py` style, straight from `fixtures/repo/repofixture/`), plus the
    visible failing test file.
  - Decoy near-misses next to every planted item (`ZQ-7732` beside `ZQ-7731`, `PRJ-13` beside
    `PRJ-12`, `B-1042` beside `B-1043`) so retrieval cannot be done by fuzzy surface matching.
- Trailer: `\n\n--- END OF DOCUMENT ---\n` then the task text.

Sizing: tokenizers differ, so calibrate once per model with
`llama-tokenize -m <gguf> --ids --no-bos -f doc.txt | wc -l` and scale the filler until the count
is **≤ 96,000** for that model. Use the *canonical text* (identical for all models) and pick its
length so that the **largest** of the five token counts is ≤ 96,000; then every model realizes
92k–96k tokens inside a `-c 102400` window (8192 of generation headroom). Record the realized
count per model in `raw.json`. Do **not** emit per-model document variants — text identity across
arms is what makes the comparison valid; the tokenizer spread is a documented threat (§13).

Determinism: the document depends only on `seed` and on the code-section files. Store its sha256 in
`raw.json` so a third party can prove they ran the same text.

---

## 7. Tiers R and L (`kvtest/tasks.py`)

Every graded prompt ends with a strict instruction so grading is a regex, not a judge:

```
Answer with a single final line of exactly this form, and nothing after it:
ANSWER: <value>
```

`extract_answer(text)` = the value of the **last** `^ANSWER:\s*(.+)$` line (models sometimes write
one inside reasoning). When no such line is present the grader records `ok=False` with
`why="no ANSWER line"`.

**R tier — retrieval / tracing / aggregation (RULER-derived), graded, 3 reps at temp 0.7:**

| id | question | answer form | grader |
|---|---|---|---|
| `niah_multi` | three named shards' codes, in the order asked | `XX-####,XX-####,XX-####` | exact set+order after uppercasing/trimming |
| `niah_deep` | the shard code whose line sits at 97% depth | `XX-####` | exact |
| `vt_trace` | follow `manager` 4 hops from `EMP-0042`, report final manager's badge and project | `B-####,PRJ-##` | exact |
| `agg_sum` | count and sum of `amount` over LEDGER rows with `status=OPEN` and `tag=RED` | `n,sum` | exact ints |
| `count_proj` | how many RECORD rows have `project=PRJ-k` | int | exact |
| `fwe_count` | how many times the exact token `ZQ-7731` appears (decoys present) | int | exact |

Each answer is a short string, which makes grading cheap and leaves no judge variance to argue about.
Partial credit is 1 point per correct sub-value, so `agg_sum` is worth 2 points and `niah_multi` 3.

**L tier — loop probes, 3 reps at temp 0.7, repetition suppression OFF:**

| id | prompt | cap | what proves "did no work" |
|---|---|---|---|
| `gen_manual` | "Write an operator's manual, one short paragraph (60–120 words) for each of RULE 1..12." | 1536 | loop metrics + `paragraphs ≥ 12` |
| `copy_ledgers` | "Reproduce the 12 consecutive LEDGER lines starting at L-0xxx verbatim, one per line." | 1536 | loop metrics + `lines == 12` |
| `enumerate_shards` | "List all 40 shards as `<name>=<code>`, one per line, in document order." | 1536 | loop metrics + correctness vs planted order |
| `tool_episode` | multi-turn: `read_shard(name)` then `submit_report(shards=[...])`; the harness returns content only for planted names and `NOT FOUND` otherwise | 512/turn, ≤6 turns | identical tool call twice in a row, or ≥3 identical total, or 0 successful reads after ≥4 turns |

`tool_episode` is the direct implementation of your definition ("repeating the same tool calls /
actions without doing work"). It needs the tools to be exposed via the OpenAI `tools` field with
`--jinja`; if a model emits no tool call at all in turn 1, record `tool_unsupported` and exclude
that arm from the tool-loop denominator (do not count it as a pass). Tool-call parsing: reuse the
`normalize_calls` logic from `bench.py` (`tools/server` returns OpenAI-style `tool_calls`).

---

## 8. Tier C — the hard, verifiable, SWE-flavoured tier (`kvtest/codegraders.py`)

Reuse the proven fixtures and runners rather than inventing tasks. All grading is "run the hidden
checks", i.e. the SWE-bench contract (must-pass set), with dense partial credit.

Six tasks, greedy (temp 0), one request each:

| id | task | output shape | grader | checks |
|---|---|---|---|---|
| `cg_csvlite` | implement the module described by the embedded spec | one ```python block | `fixtures/codegen-run.py cand.py checks.json` | 29 |
| `cg_fixedpoint` | same | same | same | 45 |
| `cg_intervals` | same | same | same | 162 |
| `cg_semver` | same | same | same | 261 |
| `cg_statistics` | same | same | same | 229 |
| `repo_fix` | the intervalkit package dumped at 85% depth has a genuine defect; emit the corrected files | `### FILE: intervalkit/xxx.py` blocks | `fixtures/repo-run.py cand_dir checks.json` + `test_visible.py` must pass | 162 |

Port these three helpers from `bench.py` **exactly** (they encode hard-won failure modes):

- `_extract_code(text)` — pick the largest fenced block that `ast.parse`s; fall back to a lax
  fence regex. (The naive first-block regex produced a 0/261 that was an extraction artifact.)
- `_extract_files(text)` — parse `### FILE: <path>` headers; a fenced body is used as-is, an
  unfenced body is `textwrap.dedent`ed (dedenting the fenced case would flatten real indentation).
- `_run_codegen/_run_repo` — run the runner in a **subprocess** with a 180 s timeout and parse the
  last stdout line as JSON; a candidate that hangs, recurses or `sys.exit`s must not kill the
  harness.

`score = passed_checks / total_checks`, plus a boolean `all_or_nothing` (all checks pass) so the
report can show both SWE-bench-style resolution and partial credit. Two extra process metrics per
solution: output tokens and number of lines emitted (a model that "passes" by dumping 900 lines is
visible).

**Calibration requirement (important):** hard tasks can be too hard, and then "no model passes at
q4_0" is not evidence of KV damage. After the run, `report.py` computes, for every C task, whether
**at least one f16 arm scored > 0**. Tasks with no f16 success are labelled `too_hard_for_matrix`
in the report and are excluded from the KV-delta comparison (they stay in the raw data). This keeps
"the task is hard" from being read as "q4_0 broke it".

---

## 9. `kvtest/loops.py` — the loop/degeneration instrument

Per user definition: repeating the same words, commands, tool calls or actions without doing work.
Run the metrics on `reasoning_content` and `content` **separately** (thinking loops are the common
failure and would be hidden by concatenation), and on the tool transcript.

```python
import re, zlib
from collections import Counter

def words(t):  return re.findall(r"\S+", t)

def ngram_max_run(toks, k):
    """Longest run of one k-gram repeated CONSECUTIVELY -> (reps, start_index, kgram)."""
    best = (1, 0, None)
    n = len(toks)
    for i in range(n - k):
        gram = tuple(toks[i:i+k])
        reps, j = 1, i + k
        while j + k <= n and tuple(toks[j:j+k]) == gram:
            reps += 1; j += k
        if reps > best[0]:
            best = (reps, i, gram)
    return best

def line_max_run(t):
    best, run, prev = 0, 0, None
    for raw in t.splitlines():
        line = raw.strip()
        if not line:  continue
        run = run + 1 if line == prev else 1
        prev = line
        best = max(best, run)
    return best

def repeated_blocks(t, size=200, step=50, min_reps=3):
    """Any >=size-char window occurring >=min_reps times anywhere -> (text, count)."""
    seen = Counter()
    for i in range(0, max(len(t) - size, 0), step):
        seen[t[i:i+size]] += 1
    if not seen: return None
    blk, n = seen.most_common(1)[0]
    return (blk, n) if n >= min_reps else None

def suffix_cycle(t, max_period=300, min_reps=3):
    """Largest p such that the last m*p chars are m identical blocks -> (p, m)."""
    for p in range(min(max_period, len(t)//min_reps), 7, -1):
        tail = t[-3*p:]
        if len(tail) == 3*p and tail[:p] == tail[p:2*p] == tail[2*p:]:
            return (p, 3)
    return None

def metrics(t):
    toks = words(t or "")
    per_k = {k: ngram_max_run(toks, k) for k in (1, 2, 4, 8)}
    lines = line_max_run(t or "")
    blk   = repeated_blocks(t or "")
    cyc   = suffix_cycle(t or "")
    zr    = (len(zlib.compress((t or "").encode(), 6)) / max(len((t or "").encode()), 1))
    grams = Counter(tuple(toks[i:i+4]) for i in range(max(len(toks)-3, 0)))
    dist  = sum(grams.values()) and len(grams) / sum(grams.values())
    return dict(word_run_1=per_k[1][0], word_run_2=per_k[2][0],
                word_run_4=per_k[4][0], word_run_8=per_k[8][0],
                line_run=lines, block=blk, cycle=cyc, zip_ratio=round(zr, 3),
                distinct_4=round(dist or 1.0, 3), chars=len(t or ""),
                span=_span(t, per_k, lines))

def verdict(m, graded=False, got_answer=True, gen_tokens=0):
    reasons = []
    if m["line_run"] >= 5:                                   reasons.append(f"line_run={m['line_run']}")
    if m["word_run_8"] >= 6 or m["word_run_4"] >= 6 or m["word_run_1"] >= 8:
        reasons.append(f"ngram_run={max(m['word_run_1'], m['word_run_4'], m['word_run_8'])}")
    if m["cycle"] and m["cycle"][0] >= 8:                    reasons.append(f"cycle={m['cycle']}")
    if m["block"] and m["block"][1] >= 3:                    reasons.append(f"block_x{m['block'][1]}")
    if m["chars"] >= 1024 and m["zip_ratio"] >= 0.42:         reasons.append(f"zip={m['zip_ratio']}")
    if graded and gen_tokens >= 1024 and not got_answer:       reasons.append("no_answer_after_long_gen")
    return (bool(reasons), "; ".join(reasons))
```

Tool-loop detection (separate, in `tasks.py`):

```python
def tool_loop(transcript):
    """transcript: [{"tool": name, "args": dict, "ok": bool}, ...] in call order."""
    sigs = [(c["tool"], json.dumps(c["args"], sort_keys=True)) for c in transcript]
    for a, b in zip(sigs, sigs[1:]):
        if a == b:  return True, "identical_consecutive_call"
    dup = [s for s, n in Counter(sigs).items() if n >= 3]
    if dup:         return True, f"identical_call_x3:{dup[0][:80]}"
    if len(transcript) >= 4 and not any(c["ok"] for c in transcript):
        return True, "no_progress_after_4_turns"
    return False, ""
```

Store for every flagged sample a `span` = the first 600 chars of the offending region, so the HTML
gallery can `<mark>` it. Thresholds are constants at the top of `loops.py` with a comment
explaining each: a line repeated 5×, a 4-gram repeated 6× consecutively, a 3-cycle in the tail, a
200-char block 3×, or zlib ratio ≥ 0.42 on ≥1 KiB, is degenerate text by any reasonable reading —
and the *raw numbers are always reported beside the verdict*, so a reader can apply their own
threshold.

---

## 10. `kvtest/run.py` — orchestrator

```
python3 -m kvtest.run --stamp 2026-10-29T22 --deadline-hours 22 [--models a,b] [--kvs f16,q8_0]
                      [--reps 3] [--code-reps 1] [--smoke] [--resume]
```

Per arm, in this order (f16 first *within* each model because it is the reference):

1. `Server.start(model, kv, ctx)` runs this command, where `<BIN>` is the directory holding the
   llama.cpp build (the `BASE` constant in `config.py`):
   ```
   <BIN>/llama-server -m <weights> --alias <model>-<kv> --port 18400
     -c <common_ctx> -ctk <kv> -ctv <kv> <SERVER_COMMON> <model extra> <knob flags>
   ```
   Wait for `/health`, log stdout+stderr to `results/<stamp>/logs/<model>__<kv>.log`, start the
   VRAM sampler.
2. Warm the document into the slot with the first R task (this is the one 100k prefill).
3. **R tier**: 6 tasks × `--reps` reps, temp 0.7, seeds 1001.. — `SAMPLING_LOOP`.
4. **C tier**: 6 tasks × `--code-reps`, temp 0.0 — `SAMPLING_GRADED`.
5. **L tier**: 4 probes × `--reps`, temp 0.7, repetition suppression off; `tool_episode` runs
   its up-to-6 turns per rep.
6. Write `arms/<model>__<kv>.json` atomically (`os.replace`) after **every request**, and append
   one line per request to `samples.jsonl` — a crash or a deadline cut loses nothing.
7. `Server.stop()` (SIGTERM, wait, SIGKILL); check the port is free before the next arm.

Request body (all of these were verified accepted per request):

Here `DOC` is the shared document text and `task_text` is the task instruction appended to it:

```python
{"messages": [{"role": "user", "content": DOC + "\n\n" + task_text}],
 "max_tokens": budget, "stream": False, "cache_prompt": True,
 "seed": seed, "temperature": t, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
 "repeat_penalty": rp, "repeat_last_n": rln, "dry_multiplier": 0.0}
```

Read back `choices[0].message.content`, `.reasoning_content`, `.tool_calls` and `usage.*`, plus the
`timings` block (`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `cache_n`). Those fields also
decide `reasoning_supported`: it is `bool(reasoning_content)` on the first request, and if it is false
the harness records that once per arm and carries on.

Deadline governor: before each arm, compute `est` as the median duration of the arms already
completed. If `now + est` would pass the deadline, stop, write `raw.json` from what exists, and set
`"deadline_reached": true`
and the list of unrun arms. Priority order guarantees Gemma/Ornith are the casualties, never
half of a KV ladder.

Error handling: any request exception (timeout 1800 s, HTTP error) becomes a sample with
`error` set — never an exception that kills the arm; a model that returns 0 tokens is a sample,
not a crash.

---

## 11. `kvtest/report.py` — REPORT.md + single-file report.html (inline SVG, no CDN)

Outputs into `results/<stamp>/`: `raw.json` (merged, with provenance), `samples.jsonl`,
`matrix.csv` (one row per model×kv: R accuracy, C check %, C resolved count, loop rate as
`flagged/samples`, prefill t/s, decode t/s, realized prompt tokens, offload knobs, VRAM peak).

`REPORT.md` sections:
1. Header: context `-c`, ladder result, llama.cpp commit, seeds, budgets, per-model offload.
2. **R tier matrix**: models × KV, accuracy % (partial credit), with the f16 column as reference.
3. **C tier matrix**: models × KV, checks passed / total and resolved (all-or-nothing) count;
   `too_hard_for_matrix` tasks listed separately.
4. **Loop table**: models × KV, `flagged/samples` for each probe + the union, split
   reasoning-vs-content.
5. **Paired deltas vs f16** per model: `q8_0 − f16`, `q4_0 − f16` for every metric, with the
   sample counts, so noise is visible (a 1-point delta on 6 tasks is noise; say so).
6. **Agreement matrix**: for graded R tasks, the share of samples where q4_0 gives the *same* answer
   as f16 for the same task/seed. This is the low-level signal that resolves differences even when
   nothing fails outright (it is an output-level analogue of the KL instrument that resolved the
   earlier 32k study).
7. **Speed at depth** and **realized prompt tokens** per model (tokenizer spread).
8. Threats to validity (from §13), literal and unhedged.

`report.html` — one file, ~self-contained, opens from `file://`, no network:

- Data embedded as `const RAW = {...};` from `raw.json`.
- Three tabs (vanilla JS, `<button>` + `hidden` attribute): **Matrix**, **Tasks**, **Loops**.
- **Matrix tab**: three heatmap grids drawn as inline `<svg>` by Python, side by side for R accuracy,
  C checks as a percentage, and loop rate. Rows are the five models with the f16, q8_0 and q4_0 arms as
  columns. The colour scale is computed in Python and runs green to amber to red as degradation
  increases; every cell also prints its number and carries a `<title>` tooltip. Under each grid sit the
  paired delta bars, one `<rect>` per cell with the zero line at f16.
- **Tasks tab**: per task, a grouped bar chart of all 15 arms + a `<details>` per arm listing the
  failed check ids and the emitted code size.
- **Loops tab**: a gallery — one card per flagged sample: model, KV, tier, probe, rep, the verdict
  reason, the metrics table, and the offending span with the repeated part wrapped in
  `<mark>`. Plus a scatter (inline SVG circles) of `loop-rate vs decode t/s` and
  `loop-rate vs realized context`.
- **Footer**: the manifest block (commit, build flags, model sha256, seeds, budgets) and the exact
  commands to reproduce, so the artifact is self-documenting.

`report.py` must be idempotent: `python3 -m kvtest.report results/<stamp>` regenerates both
artifacts from `raw.json`/`samples.jsonl` with no server and no models.

---

## 12. Reproducibility packaging (your last requirement)

1. **`models.lock.json`** — for each of the 5 GGUFs: HF repo, filename, byte size, sha256 (computed
   locally after download; for the three pre-existing files compute it too — the files are big but
   this is a one-off), plus `llama_cpp: {commit: 2a667032d…, version: 0.7.0-dev build 11698,
   cuda: 13.4, cuda_arch: 86, ggml_cuda_fa_quants: "q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16",
   build_type: Release}`, host class (`16 GiB VRAM, 64 GiB RAM`), and the document sha256.
2. **`scripts/build_llama.sh`** — clone at the pinned commit, then
   `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DGGML_CUDA_FA_ALL_QUANTS=OFF -DCMAKE_BUILD_TYPE=Release` + `cmake --build build -j --target llama-server llama-tokenize`.
   A third party on different hardware changes only `CMAKE_CUDA_ARCHITECTURES`.
3. **`scripts/download_models.sh`** — idempotent, resumable (`curl -C -`), sha256-verified.
4. **`kvtest/selftest.py`** — the important one. A stdlib `http.server` mock that speaks
   `/health`, `/v1/chat/completions` and `/v1/models`, returning canned responses including: a
   correct answer, a wrong answer, a **looping** response, a truncated-thinking response, and a
   tool-call response. Assertions: every R grader accepts the correct and rejects the wrong
   answer; `loops.verdict` flags the looping sample and not the clean one; `tool_loop` flags the
   repeated call; the C-tier runner scores a known-good module from `fixtures/codegen/semver/`
   reference implementation at 261/261 and a deliberately broken one below; `report.py` produces
   `report.html` containing 3 tabs and 15 cells per grid. Prints `selftest OK`. **No GPU, no
   models, no network** — so anyone can validate the harness in seconds before downloading 36 GB.
5. **`--smoke` profile**: 1 model at q4_0, `-c 8192`, `--reps 1`, all three tiers, taking about ten
   minutes and producing a complete but tiny report. It is documented as the first command in
   `REPRODUCE.md`.
6. **`docs/REPRODUCE.md`** — hardware classes and what auto-adapts (offload knobs and the context
   ladder are *measured*, not hardcoded, so a 24 GB card lands on 102400 with no changes and a
   12 GB card lands lower); the exact sequence of selftest, then smoke, then the full matrix;
   expected wall clock;
   and a "what will differ on your machine" section (tokenizer spread, achieved context rung,
   decode rates — never the *scoring* rules).
7. **Copyleft/licensing note** — the codegen/repofixture task fixtures came from the user's own
   `~/.local/share/llm-models/bench/fixtures`; state that in `docs/METHOD.md` (they are not
   third-party SWE-bench data). If real SWE-bench instances are ever added, the dataset licence and
   the per-instance docker images are the blocker — recorded as "considered and rejected" with the
   reason (a SWE-bench-Verified image per instance is 1–2 GB and cannot be graded offline in the
   time budget; the intervalkit/codegen tiers give the same contract — hidden tests, FAIL_TO_PASS
   — at zero provisioning cost).

---

## 13. Threats to validity (write these into `docs/METHOD.md` and the report)

1. **Text-identical ≠ token-identical.** One document text, five tokenizers: realized prompt
   lengths will differ by up to ~6%. `-c` is identical everywhere; the realized counts are
   reported. Cross-model comparison is descriptive; the KV ladder within a model is the experiment.
2. **Only full-attention layers accumulate KV damage over depth.** The qwen35 family quantizes
   16/10/10 layers, Flash-Next 12, Gemma 26B only 5 global layers (its other 25 are capped at a
   1024-token window). A "q4_0 is fine" result on the 27B is *expected* and partly structural.
3. **Greedy ≠ stochastic.** The C tier is one greedy pass, so it has no error bar; the R and L
   tiers have 3 samples each. Any delta smaller than one task's worth of points is noise and the
   report says so in the same sentence as the number.
4. **Hard-task pass rates are confounded with absolute difficulty.** Handled by the
   `too_hard_for_matrix` calibration and by always showing the f16 column.
5. **`repeat_penalty 1.0` in the L tier is more permissive than the production 1.05**, so measured
   loop rates are an upper bound for production settings. Both values are printed.
6. **Reps at temperature 0 are bit-identical** on a fixed build, `-ub` and seed — which is why the
   code tier is not repeated at temp 0, and why `--code-reps` uses temp 0.7 if anyone asks for it.
7. **The prefill is shared across tasks within an arm.** If a model produces a bad answer early it
   does not contaminate later requests (each request's KV is rewound to the common document
   prefix), except in `tool_episode`, where the transcript is deliberately sequential.

---

## 14. Execution order (what the implementer actually does, in sequence)

1. `scripts/preflight.sh` — assert: `nvidia-smi` answers, no `llama-server` running, ≥60 GB free
   disk, no `.part` model files. Capture the Strata command line into
   `results/<stamp>/strata-cmd.txt`, taking the command line from `ps -o args= -p 2556064` and the
   working directory from `readlink /proc/2556064/cwd`.
2. `scripts/download_models.sh` — the two new GGUFs, sha256-verified. Write `models.lock.json`.
3. `cp -a` the fixtures and runners, then run `python3 -m kvtest.selftest`, which **must print
   `selftest OK`** before anything touches the GPU.
4. `kill -TERM 2556064`, wait for VRAM < 1024 MiB (poll `nvidia-smi --query-gpu=memory.used`),
   escalate to `-KILL` after 120 s. Record the free-VRAM number in the stamp.
5. Run `python3 -m kvtest.fit --ladder 102400 --stamp <stamp>`, which writes `fit.json` with the
   resolved common context and the per-model knobs. **Read it**: if the resolved context is below 81920,
   say so in the report header, and if a model is `unfit` it is out of the matrix.
6. `python3 -m kvtest.run --stamp <stamp> --resume --deadline-hours 22` — the long one. Run it
   under `nohup … &` with output to `results/<stamp>/run.out` so a closed terminal does not kill
   it; it is resumable with `--resume`.
7. Run `python3 -m kvtest.report results/<stamp>`, which writes `REPORT.md`, `report.html` and
   `matrix.csv`.
8. **Restart Strata** (a `trap` in `run_all.sh` also does it on failure): relaunch the captured
   command line from the captured cwd with `nohup setsid`, then verify a new pid exists, the
   process is `strata --serve`, and its listening port (`ss -ltnp | grep strata`) answers. Only
   kill the exact pid captured in step 1, and only after confirming its command line contains
   `strata --serve`.
9. Fill `docs/RESULTS.md` from `REPORT.md` in prose: which arms degraded, by how much, which looped,
   and the one-line answer to the original question.

Time budget: fit ≈ 45–70 min (5 models; the 27B is the slow one), matrix ≈ 15–18 h at 71k
generated tokens per arm (≈30k code tier + ≈41k R/L with 3 reps) at 15–25 t/s, loads and prefills
included. Governor at 22 h, Ornith last, `--only`/`--resume` for everything in between.

---

## 15. UNVERIFIED points (state them, do not paper over them)

1. **Network** in this planning sandbox is absent; the two downloads were verified by HF API
   (`ggml-org/gemma-4-26B-A4B-it-GGUF`, `ornith-ai/Ornith-1.5-35B-A3B-GGUF`) but the files were
   not fetched here. If the implementer's shell also has no network, stop and report — do not
   fabricate sizes.
2. **Gemma 4 tool-calling** through `--jinja`: its chat template includes a tool section, but
   `docs/autoparser.md` only certifies specific Gemma builds. If `tool_episode` yields no calls,
   the arm is marked `tool_unsupported` for that probe only.
3. **`--reasoning-effort medium` support per family**: `--reasoning on --reasoning-effort medium
   --reasoning-budget 8192` was used successfully on this build for the Qwen 27B (see the
   2026-10-10 log); Gemma 4 and Ornith are unconfirmed. The `reasoning_supported` flag records it.
4. **The 27B at 102,400 with f16 KV** has never been run; prior measurement was 81,920 at
   zero offload, so the ladder may well settle at 81920. That is a valid outcome, not a failure.
5. **Ornith GGUF geometry** (KV heads/layers) was not read here; `verify_geometry()` reads it at
   download time.
6. **Flash-Next flags** `-lm mmap -lzm on` come from the previous harness, where they were
   required to keep the n-gram shard on disk. If the current build rejects them, use
   `--load-mode mmap --lazy-mode on` equivalents and note it.

---

## 16. Self-check before declaring the work done

- Make sure to compare the answers of each model between itself at different kv quants as well as between the other models
- `python3 -m kvtest.selftest` prints `selftest OK`, and it does so without a GPU.
- `python3 -m kvtest.run --smoke --stamp smoke` leaves `results/smoke/report.html` in place, with 3
  tabs and a 15-cell grid, and leaves `REPORT.md` non-empty.
- `python3 -c "import json;d=json.load(open('results/<stamp>/raw.json'));assert d['manifest']['llama_cpp']['commit'].startswith('2a667032d')"`
- `grep -c '"loop"' results/<stamp>/samples.jsonl` > 0 and at least one sample has
  `"loop": true` (the mock exercising it proves the detector wired in, the real run decides).
- `nvidia-smi` shows the Strata process's VRAM back after step 8, and `ss -ltnp` shows it
  listening.
- Every number in `REPORT.md` traceable to a row in `samples.jsonl`; no metric invented in prose.
