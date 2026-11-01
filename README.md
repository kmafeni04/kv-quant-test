# kv-quant-test — does KV-cache quantization hurt long-context work?

Five models, served by llama.cpp at one identical long context (~96k tokens), each run three times
with a different KV-cache type: **f16, q8_0, q4_0**. 15 arms, ~5,400 graded requests, one report.

Two things are measured per arm:

- **Does the work get worse?** Six retrieval/tracing tasks with regex-graded answers, and six code
  tasks graded by running hidden test suites (the SWE-bench contract: FAIL_TO_PASS, partial
  credit, all-or-nothing "resolved").
- **Does the model start looping?** Four probes whose output is measured for degenerate repetition
  (repeated n-grams, repeated lines, repeated 200-char blocks, tail cycles, compression ratio),
  including a multi-turn tool episode that catches the same tool call being issued over and over.

Every task appends to one canonical ~96k-token document, so the expensive prefill is paid once per
arm and the three KV arms see byte-identical input.

## Quickstart

```bash
python3 -m kvtest.selftest          # ~30 s, no GPU, no models, no network -> "selftest OK"
python3 -m kvtest.run --smoke       # ~10 min on one model at q4_0, -c 8192
python3 -m kvtest.report results/smoke
```

Full study: `./scripts/run_all.sh` — see [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Headline

**Best overall: `qwen38-flash` @ q4_0** (capability 84.94/100, 0/12 structural-loop samples, 12.3 t/s
decode) — but its lead is **not established**: 11 other arms, including its own f16 arm, cannot be
distinguished from it (p ≥ 0.05, each tier tested on its own pairs, Bonferroni over the three
tiers). Per model the best KV type is
q4_0 for Flash and the 35B, q8_0 for the dense 27B, f16 for Gemma and Ornith; **no arm differs from
its own f16 arm at p < 0.05 on any metric**. Rule and reasons: `REPORT.md` §Verdict.

Filled from `results/<stamp>/REPORT.md` after the run:

<!-- HEADLINE:START -->
| model | KV | R accuracy | C checks | C resolved | structural loops | agreement with f16 |
|---|---|---|---|---|---|---|
| `qwen38-flash` | f16 | 23/30 (76.7%) | 702/780 (90.0%) | 4/6 | 0/12 | — |
| `qwen38-flash` | q8_0 | 24/30 (80.0%) | 630/780 (80.8%) | 4/6 | 0/12 | 0.611 |
| `qwen38-flash` | q4_0 | 24/30 (80.0%) | 701/780 (89.9%) | 3/6 | 0/12 | 0.667 |
| `qwen38-27b` | f16 | 21/30 (70.0%) | 629/780 (80.6%) | 4/6 | 0/12 | — |
| `qwen38-27b` | q8_0 | 23/30 (76.7%) | 702/780 (90.0%) | 4/6 | 0/12 | 0.722 |
| `qwen38-27b` | q4_0 | 21/30 (70.0%) | 634/780 (81.3%) | 4/6 | 0/12 | 0.722 |
| `qwen36-35b` | f16 | 18/30 (60.0%) | 765/780 (98.1%) | 4/6 | 0/12 | — |
| `qwen36-35b` | q8_0 | 19/30 (63.3%) | 694/780 (89.0%) | 3/6 | 1/12 | 0.722 |
| `qwen36-35b` | q4_0 | 20/30 (66.7%) | 763/780 (97.8%) | 3/6 | 0/12 | 0.556 |
| `ornith-35b` | f16 | 19/30 (63.3%) | 548/780 (70.3%) | 4/6 | 0/12 | — |
| `ornith-35b` | q8_0 | 16/30 (53.3%) | 570/780 (73.1%) | 3/6 | 1/12 | 0.5 |
| `ornith-35b` | q4_0 | 18/30 (60.0%) | 475/780 (60.9%) | 3/6 | 0/12 | 0.706 |
| `gemma4-26b-a4b` | f16 | 11/30 (36.7%) | 694/780 (89.0%) | 4/6 | 3/12 | — |
| `gemma4-26b-a4b` | q8_0 | 7/30 (23.3%) | 701/780 (89.9%) | 3/6 | 2/12 | 0.2 |
| `gemma4-26b-a4b` | q4_0 | 8/30 (26.7%) | 318/780 (40.8%) | 3/6 | 3/12 | 0.2 |
<!-- HEADLINE:END -->

## What is here

```
kvtest/        the harness (config, server lifecycle, client, document, tasks, graders, loops,
               fit probe, orchestrator, report, selftest)
fixtures/      the user's own codegen + intervalkit fixtures, verbatim
scripts/       preflight, build, download, run_all
docs/          METHOD.md (design + threats), REPRODUCE.md, RESULTS.md (findings)
results/       <UTC-stamp>/{fit.json,samples.jsonl,arms/,logs/,REPORT.md,report.html}
models.lock.json   pinned GGUF sha256 + llama.cpp commit + build flags + document hash
```

## Reading a result

The experiment is the KV ladder *within* a model. Cross-model numbers are descriptive: the same
text tokenizes differently on five tokenizers, and only full-attention layers accumulate KV damage
with depth (16 on the dense 27B, 5 on Gemma-26B). `REPORT.md` prints the raw counts next to every
percentage, labels tasks no f16 arm could solve, and reports the share of answers that are
byte-identical to the f16 answer — the signal that still resolves arms when nothing fails outright.
