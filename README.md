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

Filled from `results/<stamp>/REPORT.md` after the run:

<!-- HEADLINE:START -->
| model | KV | R accuracy | C checks | C resolved | loop rate | agreement with f16 |
|---|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 7/10 (70.0%) | 721/780 (92.4%) | 3/6 | 0/4 | — |
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
