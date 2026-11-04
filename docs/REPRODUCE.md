# Reproduce

## What you need

The study host is a single laptop with one consumer GPU; nothing below is hardcoded to it. The table
gives the study host and the minimum that works.

| | study host | minimum |
|---|---|---|
| GPU | NVIDIA RTX 3080 Laptop (the GPU model), 16,384 MiB, `sm_86` (its compute capability) | any CUDA-capable NVIDIA GPU with ≥ 12 GB; the fit probe raises `--n-cpu-moe`/`--n-cpu-ffn` until the model loads |
| RAM | 62 GiB | 32 GiB for the models; the W1 deep pass allocates host logits for half the window (32 GiB at the default 65,536) |
| disk | 240 GB free | ~100 GB for weights, the KLD (Kullback–Leibler divergence) baseline cache (~1.2 GB per model, outside the results tree) and results |
| python | 3.14, **stdlib only** | 3.11+; no numpy, no pytest, no network during a run |

`llama.cpp` must be built with CUDA flash attention and `GGML_CUDA_FA_QUANTS` covering
`f16-f16;q4_0-q4_0` (the script pins the commit and the flags). `llama-server`, `llama-tokenize` and
`llama-perplexity` are all used; CUDA is NVIDIA's GPU compute toolkit, and GGUF is the model-file
format llama.cpp loads.

## Sequence

The commands below run in order. Steps 0–2 need no GPU and no model weights, and the micro-pilot in
step 4 is the gate before the overnight matrix: if it cannot prefill the full document and grade one
answer, the matrix will fail the same way for twelve hours.

```bash
# 0. build llama.cpp at the pinned commit (CUDA, flash attention, FA quants)
./scripts/build_llama.sh

# 1. prove the harness works: mock server, no GPU, no models, ~3 s
python3 -m kvtest.selftest              # must print "selftest OK"

# 2. fetch and verify the GGUFs (writes models.lock.json)
./scripts/download_models.sh

# 3. resolve one rung per model and the 92–95k document (GPU; ~30 min for three models)
python3 -m kvtest.fit --stamp STAMP

# 4. micro-pilot, ≤ 30 min, Gemma f16 only: one full-context W2 sample, one W3 pair, one W1 pass
python3 -m kvtest.run --stamp STAMP --pilot

# 5. the matrix, tranche order fixed, 24 h hard governor, resumable per instance
nohup python3 -m kvtest.run --stamp STAMP --resume --deadline-hours 24 \
  > results/STAMP/run.out 2>&1 &

# 6. artifacts (idempotent; no GPU, no models)
python3 -m kvtest.report results/STAMP
#    add --update-readme only for a complete matrix
```

One caveat on regeneration: the shipped `REPORT.md` files under `results/pilot`, `results/smoke` and
`results/2026-11-02` were edited by hand after generation. `python3 -m kvtest.report` regenerates the
tables from `raw.json` and replaces that narrative; the command guarantees the tables, not the prose.

`./scripts/run_all.sh` runs all of it, including the GPU-holder capture/restore. `./scripts/run_all.sh
--pilot` stops after the micro-pilot.

## What the pilot established on this host

The pilot is not a scoring run. Its artifacts are in `results/pilot/` (Gemma f16 at the full rung)
and `results/smoke/` (Ornith at 32k, a positive control for the 20-key probe).

The fit probe loaded all three models at 102,400 tokens. Measured prefill on the 94k-token document
was 935 tokens/s for Gemma, 900 for Ornith and 750 for Qwen3.6; decode was 47.1, 47.0 and 35.1
tokens/s. The document calibrated to 94,170 tokens in the largest tokenizer and 84,113 in Gemma's.
The server's resolved sampler block shows `temp = 0.000`; the only other sampler values present are
the model file's own defaults (Gemma declares `top_k 64`, `min_p 0.05` and `repeat_penalty 1.000`),
so greedy decoding was in force and no value was forced onto the chain.

On the distribution instrument, the f16 self-check read mean KLD (Kullback–Leibler divergence)
`-0.000000` and same-top agreement `100.0%`. The depth pass completed at `-c 65536` with PPL
(perplexity) 177.16 in 82.9 seconds. The planned `-c 102400` aborts with `std::bad_alloc` after
allocating 50 GiB of host logits for Gemma's 262,144-token vocabulary, which is why the default
depth is 65,536 and every W1 record names the depth it scored.

Gemma f16 at 84k, decoding greedily, degenerated in reasoning on both W2 and W3. In W2 it re-checked
the same shards in turn (44k characters, 4-gram diversity 0.169). In W3 it enumerated invented ledger
ids until one line template had repeated more than sixty times. The detector cut both responses early
and recorded `halted_by=loop` with the firing stream and token position, so no fake score was
produced. The block is reported, not silently retried. On the positive control, Ornith f16 and q4_0
each scored 20/20 on one W2 instance at 32k (49 s and 46 s), and W5 scored 3/3 on the smoke document.
The probe is answerable, and the grading, cache reuse, checkpoint and report paths work end to end.

## Tranches and the governor

The four tranches are fixed in `kvtest/config.py` and in the design document: T1 Gemma all six
workflows, T2 Ornith all six, T3 Qwen3.6 W1–W3, T4 Qwen3.6 W4–W6. `--tranches T1,T2` limits the run;
the default is all four in order. Before each cell the governor estimates its cost from the fit
probe's prefill and decode rates; if the estimate does not fit before `--deadline-hours`, the cell is
not started, and `cut_cells`/`unrun_cells` in `raw.json` and in the report name it. Every instance
checkpoints into `arms/<model>__<kv>.json`, so `--resume` continues exactly where the night stopped.

## What adapts, and what does not

The fit probe measures three things and writes them into `fit.json`: the context rung each model can
hold, the offload knob each model needs to fit, and the document length that lands between 92,000 and
95,000 tokens in the largest of the three tokenizers. The KLD deep pass can also be lowered with
`--deep-ppl-ctx` on a host without enough memory for the default depth.

Everything else is fixed by design, because those are the values that make two runs comparable: the
task text, the sixty-four-key selection, the tenant mutation sets, the five invariants, the brief's
word cap, the tool-call guards, `temperature: 0.0`, the 64k response ceiling, the cache types and the
statistical rule.

## Output files

The table lists what a completed run leaves in its stamp directory. The first two files are data; the
tables and figures after them are derived and can be regenerated with `kvtest.report`, while the
shipped narrative was edited by hand.

| file | content |
|---|---|
| `fit.json` | per-model rung, knobs, VRAM peak, prefill/decode at depth, declared sampling, document sha256 and token counts |
| `document.txt` | the canonical document, byte-identical to what every request prefixed |
| `samples.jsonl` | one line per request or instance: tokens, timings, grade, halts, loop metrics |
| `arms/<model>__<kv>.json` | per-arm checkpoint, written after every instance; the unit of resume |
| `raw.json` | merged provenance, arms, cells, cut/unrun cells; rebuildable from the two files above |
| `matrix.csv` | one row per arm: W2 mean and paired interval, W3 retention, KLD, hashes, halts, cost |
| `REPORT.md`, `report.html` | the write-up and the single-file HTML artifact |
| `figures/` | `w1-kld.svg` and `w2-depth.svg`, the same figures embedded in the HTML |
| `hardware.json` | GPU, driver, CUDA, CPU, RAM, OS, kernel, llama.cpp commit and build flags |
| `logs/` | full llama-server stderr and each W1 `llama-perplexity` log, including the resolved `sampler params:` block |
| `.cache/kv-quant-test/<document>/` | the f16 logits baselines, kept out of the results tree |

## Sanity checks

The four commands below check that an artifact set is internally consistent: the manifest names the
sampling and the rungs you expect, the loop detector fired on some samples, every arm produced a
resolved sampler block, and a report regenerated from `raw.json` reproduces its tables. On a stamp you
generated yourself, regeneration changes nothing. The three shipped stamps are hand-edited, so their
diff is expected.

```bash
python3 -c "import json;d=json.load(open('results/STAMP/raw.json'));\
print(d['manifest']['sampling'], d['manifest']['rungs'])"
grep -c '"halted_by": "loop"' results/STAMP/samples.jsonl
grep -c 'sampler params' results/STAMP/logs/*.log          # one resolved block per arm
STAMP=<fresh-stamp> python3 -m kvtest.report "results/$STAMP"   # fresh stamp: no diff
```

If a W1 arm reports mean KLD 0 with a tight interval, check that `-ctk`/`-ctv` reached
`llama-perplexity` and that the baseline slice sha256 matches the arm's slice sha256 in
`manifest.w1_slices`; the instrument then did not reach the cache and the workflows should not be
trusted as a comparison until it does.
