# Reproduce

## What you need

The study host is a single laptop with one consumer GPU, and nothing below is hardcoded to it. The
table gives what that machine had in the study column, and the minimum that works in the last column.
Where the minimum is lower, the fit probe picks a smaller context and more layers go to system RAM,
which is slower but gives the same experiment.

| | study host | minimum |
|---|---|---|
| GPU | RTX 3080 Laptop, 16,384 MiB of VRAM (video memory), `sm_86` | any CUDA-capable GPU, meaning an NVIDIA card the compiler can target, with at least 12 GB; the fit probe adapts |
| RAM | 62 GiB | 32 GiB, because the expert weights of a mixture-of-experts model spill into system RAM |
| disk | 240 GB free | about 100 GB free: 36 GB of new model weights plus results |
| python | 3.14, **stdlib only** | 3.11 or newer; no numpy, no pytest, and no network access while a run is in progress |

Nothing needs the network after `scripts/download_models.sh` finishes. The report is a single HTML file
that opens from `file://`, with its charts drawn as inline SVG and its numbers embedded as JSON, so it
can be handed to someone else as one file.

## Sequence

The six steps below run in order, and each one prints what it did. Steps 1 and 2 need no GPU and no
downloaded weights, so it is worth running them before spending an hour on model downloads.

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
`kvtest.fit`. It starts at 102,400 tokens. When the server fails to allocate, it raises `--n-cpu-moe`
or `--n-cpu-ffn` to move more layers into system RAM, and when a knob runs out of room it drops a rung
and tries again. A 24 GB card therefore lands on 102,400 with no changes and a 12 GB card lands lower.
The document length is calibrated so that the largest of the five tokenizations stays under 96,000
tokens.

**Does not adapt (by design, and that is the point).** The scoring rules, the task text, the seeds, the
budgets, the sampling parameters and the cache types. Those are what make two runs comparable.

## What will differ on your machine

Run on different hardware, these are the quantities that move. None of them changes a scoring rule,
which is why a second run can still be compared with this one.

- realized prompt tokens per model, from tokenizer spread, by up to about 6 %,
- which context rung is achieved, and how many layers are offloaded per model,
- prefill and decode throughput, and therefore total wall clock,
- possibly whether a model is excluded as `unfit`, if it cannot fit at 49,152 tokens with the maximum
  offload on a small card.

## Reading the output

`results/STAMP/` holds the following files. The first two are the data; everything else is derived from
them, and `kvtest.report` can regenerate all of it.

| file | content |
|---|---|
| `fit.json` | resolved context, per-model knobs, VRAM peaks, prefill and decode rates at depth, document sha256 |
| `samples.jsonl` | one JSON line per request: prompt tokens, `cache_n` (tokens served from the reused prefix), timings, full text, grade, loop metrics |
| `arms/<model>__<kv>.json` | per-arm checkpoint, written after every request, and the unit of resume |
| `raw.json` | merged provenance, arms and counts, rebuildable from the two files above |
| `matrix.csv` | one row per model and cache type: retrieval %, code % and resolved count, loop rate, speeds, VRAM, knobs |
| `REPORT.md`, `report.html` | the write-up and the three-tab browser artifact |
| `figures/` | **every** SVG in `report.html`, meaning the three grids, the speed chart, the loop-count chart and all per-task dot plots, plus `figures.json` with the numbers behind them: per-arm levels, loop counts, decode and prefill rates, realized prompt tokens and the full `per_task` breakdown, so a saved figure can be checked without opening the HTML |
| `hardware.json` | GPU, VRAM, driver, CUDA version, CPU, RAM, OS and kernel, llama.cpp commit and build flags, captured by `kvtest/hostinfo.py` when the run starts; the same block appears under "Hardware and build this ran on" in `REPORT.md` and inside `raw.json` as `manifest.host_runtime` |
| `logs/<model>__<kv>.log` | full llama-server stderr, including the exact command line it was started with |

Every number in `REPORT.md` is derived from `samples.jsonl`; no metric is written by hand.

## Sanity checks

These four commands confirm that an artifact set is consistent: the manifest names the commit you
expect, the loop detector actually fired on some samples, every sample carries loop metrics, and
regenerating changes nothing.

```bash
python3 -c "import json;d=json.load(open('results/STAMP/raw.json'));\
print(d['manifest']['llama_cpp']['commit'], d['context'])"
grep -c '"flagged": true' results/STAMP/samples.jsonl   # samples the loop detector fired on
grep -c '"loops"' results/STAMP/samples.jsonl           # every sample carries loop metrics
python3 -m kvtest.report results/STAMP    # regenerates both artifacts, must not change content
```
