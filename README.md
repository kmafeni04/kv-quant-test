# kv-quant-test: does quantizing the KV cache hurt long-context work?

A language model working on a long document keeps its attention state in the KV cache, which stands
for key/value cache: a stored copy of every earlier token's keys and values, so the model does not have
to re-read the document for each new word. That cache can be compressed. This study asks what
compression costs.

Five models are served by llama.cpp at one identical long context of about 96k tokens, and each is run
three times with a different cache type: **f16** (uncompressed), **q8_0** (8-bit) and **q4_0** (4-bit).
That gives 15 arms and 540 requests, reported once.

Two things are measured per arm:

- **Does the work get worse?** Six retrieval and tracing tasks with answers graded by regex, plus six
  code tasks graded by running hidden test suites under the SWE-bench contract: FAIL_TO_PASS tests,
  partial credit, and an all-or-nothing "resolved" count. SWE-bench is the standard software-engineering
  benchmark, and its contract means a task passes only when the failing tests it targets start passing.
- **Does the model start looping?** Four probes whose output is measured for degenerate repetition
  (repeated n-grams, repeated lines, repeated 200-character blocks, tail cycles, compression ratio).
  One of the four is a multi-turn tool episode, which catches the same tool call being issued over and
  over, a failure that never appears in a single response.

Every task appends to one canonical ~96k-token document, so the expensive prefill is paid once per arm
and the three cache types in an arm group see byte-identical input.

## Quickstart

Three commands, none of which needs a GPU, a downloaded model or network access. The first checks the
harness against fixtures, the second runs one model end to end on a small context, and the third turns
the result into the written report and its HTML page.

```bash
python3 -m kvtest.selftest          # ~30 s -> "selftest OK"
python3 -m kvtest.run --smoke       # ~10 min on one model at q4_0, -c 8192
python3 -m kvtest.report results/smoke
```

The full study is `./scripts/run_all.sh`, and the individual steps are in
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Headline

Highest score: `qwen38-flash` with a q4_0 cache, at 84.94 of 100 capability, 0 of 12 samples showing a
structural loop, 12.3 tokens per second of decode. **Its lead is not established.** Every tier has six
task clusters, and once the reps of a task are clustered and a correction is applied across the 42
arm-by-tier comparisons, all 14 other arms, including its own f16 arm, are indistinguishable from it.

Per model the top arm is q4_0 for Flash and the 35B, q8_0 for the dense 27B, and f16 for Gemma and
Ornith, but none of those gaps is resolvable at this sample size. The ranking rule and the reasoning
behind it are in `REPORT.md`, section Verdict.

The table below is filled in from `results/<stamp>/REPORT.md` after a run. One row per arm: retrieval
accuracy is points out of 30, code is hidden checks passed out of 780 with resolved tasks out of 6,
structural loops counts samples showing real repetition, and agreement is the share of graded answers
identical to the same model's f16 answer.

<!-- HEADLINE:START -->
| model | KV | R accuracy | C checks | C resolved | structural loops | agreement with f16 |
|---|---|---|---|---|---|---|
| `qwen38-flash` | f16 | 23/30 (76.7%) | 702/780 (90.0%) | 4/6 | 0/12 |  |
| `qwen38-flash` | q8_0 | 24/30 (80.0%) | 630/780 (80.8%) | 4/6 | 0/12 | 0.611 |
| `qwen38-flash` | q4_0 | 24/30 (80.0%) | 701/780 (89.9%) | 3/6 | 0/12 | 0.667 |
| `qwen38-27b` | f16 | 21/30 (70.0%) | 629/780 (80.6%) | 3/6 | 0/12 |  |
| `qwen38-27b` | q8_0 | 23/30 (76.7%) | 702/780 (90.0%) | 4/6 | 0/12 | 0.722 |
| `qwen38-27b` | q4_0 | 21/30 (70.0%) | 634/780 (81.3%) | 3/6 | 0/12 | 0.722 |
| `qwen36-35b` | f16 | 18/30 (60.0%) | 765/780 (98.1%) | 4/6 | 0/12 |  |
| `qwen36-35b` | q8_0 | 19/30 (63.3%) | 694/780 (89.0%) | 3/6 | 1/12 | 0.722 |
| `qwen36-35b` | q4_0 | 20/30 (66.7%) | 763/780 (97.8%) | 3/6 | 0/12 | 0.556 |
| `ornith-35b` | f16 | 19/30 (63.3%) | 548/780 (70.3%) | 4/6 | 0/12 |  |
| `ornith-35b` | q8_0 | 16/30 (53.3%) | 570/780 (73.1%) | 2/6 | 1/12 | 0.5 |
| `ornith-35b` | q4_0 | 18/30 (60.0%) | 475/780 (60.9%) | 3/6 | 0/12 | 0.706 |
| `gemma4-26b-a4b` | f16 | 11/30 (36.7%) | 694/780 (89.0%) | 4/6 | 3/12 |  |
| `gemma4-26b-a4b` | q8_0 | 7/30 (23.3%) | 701/780 (89.9%) | 2/6 | 2/12 | 0.2 |
| `gemma4-26b-a4b` | q4_0 | 8/30 (26.7%) | 318/780 (40.8%) | 3/6 | 3/12 | 0.2 |
<!-- HEADLINE:END -->

## What is here

The tree below lists what each directory holds, so you can tell the harness from the fixtures from the
outputs. `models.lock.json` is the provenance record: it pins the model files by sha256 together with
the llama.cpp commit, the build flags and the document hash, which is what makes two runs comparable.

```
kvtest/        the harness (config, server lifecycle, client, document, tasks, graders, loops,
               fit probe, orchestrator, report, selftest)
fixtures/      the user's own codegen + intervalkit fixtures, verbatim
scripts/       preflight, build, download, run_all
docs/          METHOD.md (design + threats), REPRODUCE.md, RESULTS.md (findings),
               AGENTIC-BENCH-PLAN.md (design for the next study: agentic workflows at 100k)
results/       <UTC-stamp>/{fit.json,samples.jsonl,arms/,logs/,REPORT.md,report.html}
models.lock.json   pinned GGUF sha256 + llama.cpp commit + build flags + document hash
```

## Reading a result

The experiment is the cache ladder within a model, not a comparison between models. Cross-model numbers
are descriptive only: the same text tokenizes differently on five tokenizers, and only full-attention
layers accumulate cache damage with depth, of which there are 16 on the dense 27B and 5 on
Gemma-26B. `REPORT.md` prints raw counts next to every percentage and names the tasks no f16 arm could
solve, and the appendix reports the share of answers that are byte-identical to the f16 answer.
