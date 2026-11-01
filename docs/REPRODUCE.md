# Reproduce

## What you need

| | study host | minimum |
|---|---|---|
| GPU | RTX 3080 Laptop, 16,384 MiB, sm_86 | any CUDA GPU with ≥12 GB; nothing is hardcoded to this card |
| RAM | 62 GiB | 32 GiB (MoE expert weights spill to RAM) |
| disk | 240 GB free | ~100 GB free (36 GB of new weights + results) |
| python | 3.14, **stdlib only** | 3.11+; no numpy, no pytest, no network at run time |

Nothing is network-dependent after `scripts/download_models.sh`: the report is a single HTML file
that opens from `file://` with inline SVG and embedded JSON.

## Sequence

```bash
# 0. build llama.cpp at the pinned commit with the flags this study depends on
./scripts/build_llama.sh

# 1. prove the harness works - no GPU, no models, no network, ~30 s
python3 -m kvtest.selftest              # must print "selftest OK"

# 2. fetch and verify the five GGUFs (~36 GB, writes models.lock.json)
./scripts/download_models.sh

# 3. ~10 minute end-to-end proof: one model, q4_0, -c 8192, one rep
./scripts/preflight.sh smoke
python3 -m kvtest.run --smoke --resume
python3 -m kvtest.report results/smoke

# 4. resolve the one shared context (45-70 min; the dense 27B at f16 is the slow one)
python3 -m kvtest.fit --stamp STAMP --ladder 102400

# 5. the matrix (15-18 h expected, 22 h hard governor, resumable per arm)
nohup python3 -m kvtest.run --stamp STAMP --resume --deadline-hours 22 \
  > results/STAMP/run.out 2>&1 &

# 6. artifacts (idempotent, no GPU, no models)
python3 -m kvtest.report results/STAMP
#    add --update-readme only for a full matrix: it rewrites the headline table in README.md,
#    and refuses to do so for a result with fewer than 5 arms.
```

`./scripts/run_all.sh` runs all of it and restores whatever was holding the GPU when it started.

## What adapts, and what does not

**Adapts (measured, not hardcoded).** The context rung and the per-model offload knobs come from
`kvtest.fit`: it starts at 102,400, and on an allocation failure it bumps `--n-cpu-moe` /
`--n-cpu-ffn`; when a knob is exhausted it drops a rung. A 24 GB card lands on 102,400 with no
changes; a 12 GB card lands lower. The document length is calibrated so the *largest* of the five
tokenizations stays under 96,000 tokens.

**Does not adapt (by design, and that is the point).** The scoring rules, the task text, the seeds,
the budgets, the sampling parameters and the KV arms. Those are what make two runs comparable.

## What will differ on your machine

- realized prompt tokens per model (tokenizer spread, up to ~6 %),
- which context rung is achieved and how many layers are offloaded per model,
- prefill and decode throughput, hence total wall clock,
- possibly whether a model is excluded as `unfit` if it cannot fit at 49,152 with the maximum
  offload on a small card.

Never the scoring rules.

## Reading the output

`results/STAMP/`:

| file | content |
|---|---|
| `fit.json` | resolved context, per-model knobs, VRAM peaks, prefill/decode at depth, document sha256 |
| `samples.jsonl` | one JSON line per request: prompt tokens, `cache_n`, timings, full text, grade, loop metrics |
| `arms/<model>__<kv>.json` | per-arm checkpoint (the resume unit) — written after every request |
| `raw.json` | merged provenance + arms + counts (rebuildable from the two above) |
| `matrix.csv` | one row per model × KV: R %, C %/resolved, loop rate, speeds, VRAM, knobs |
| `REPORT.md`, `report.html` | the write-up and the three-tab browser artifact |
| `figures/` | **every** SVG in `report.html` (the three grids, the speed chart, the loop-count chart and all per-task dot plots), plus `figures.json` carrying the numbers behind them — per-arm R/C levels, loop counts, decode/prefill rates, realized prompt tokens and the full `per_task` breakdown, so a saved figure can be checked without the HTML |
| `hardware.json` | GPU, VRAM, driver, CUDA, CPU, RAM, OS/kernel, llama.cpp commit and build flags, captured by `kvtest/hostinfo.py` when the run starts; the same block is printed under "Hardware and build this ran on" in `REPORT.md` and is embedded in `raw.json` as `manifest.host_runtime` |
| `logs/<model>__<kv>.log` | full llama-server stderr including the exact command line |

Every number in `REPORT.md` is derived from `samples.jsonl`; no metric is written by hand.

## Sanity checks

```bash
python3 -c "import json;d=json.load(open('results/STAMP/raw.json'));\
print(d['manifest']['llama_cpp']['commit'], d['context'])"
grep -c '"flagged": true' results/STAMP/samples.jsonl   # samples the loop detector fired on
grep -c '"loops"' results/STAMP/samples.jsonl           # every sample carries loop metrics
python3 -m kvtest.report results/STAMP    # regenerates both artifacts, must not change content
```
