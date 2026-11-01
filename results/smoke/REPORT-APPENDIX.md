# Appendix — every table, method and provenance

## Appendix: full tables, method and provenance


## Hardware and build this ran on

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU 16384 MiB sm_86 |
| driver / CUDA | — / — |
| CPU | — (16 logical cores) |
| RAM | 62.5 GiB |
| OS / kernel | Arch Linux, kernel 7.2.11-arch1-1 / — |
| llama.cpp | 2a667032dc78 |

Recorded: from models.lock.json (run predates runtime capture). Per-arm GPU peak and the offload knobs are in the per-model offload table below; those are what the memory budget was spent on.
The long form lives in `REPORT-APPENDIX.md` next to this file; it is the same data, unsorted for reading.

| model | best KV | capability | runner-up | structural loops | decode t/s | why |
|---|---|---|---|---|---|---|
| `qwen36-35b` | **q4_0** | 81.22 | None | 0/4 | 45.6 | 0/4 structural-loop samples, 45.6 t/s decode |

### All 1 arm, ranked by the same rule

| rank | arm | capability | structural loops | decode t/s | p vs #1 | distinguishable from #1? |
|---|---|---|---|---|---|---|
| 1 | `qwen36-35b` **q4_0** | 81.22 | 0/4 | 45.6 | 1.0000 | no — tied |
## Per-model offload (knobs are frozen across the three KV arms)

Knobs are identical for a model's three arms; the VRAM peak and speeds below are the **f16 arm's**. 

| model | weights off the GPU | VRAM peak MiB (f16) | realized prompt tokens | prefill t/s (f16) | decode t/s (f16) |
|---|---|---|---|---|---|
| `qwen36-35b` | 26 layers on CPU | 9541 | 10679 | 642.6 | 45.6 |

## R tier: retrieval, tracing, aggregation (partial credit, %)

| model | f16 | q8_0 | q4_0 |
|---|---|---|---|
| `qwen36-35b` | — | — | 70.0 |

## C tier: hidden checks passed / total, and resolved tasks

| model | KV | checks | % | resolved | too-hard tasks |
|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 721/780 | 92.4 | 3/6 | — |

## Loop table: flagged samples / samples (reasoning and content split)

`union (L)` counts every verdict from the zlib-ratio heuristic or a structural signal; `hard repetition` counts only samples with a repeated line, n-gram, 200-char block or tail cycle. The gap between the two columns is formulaic prose, not looping.

| model | KV | gen_manual | copy_ledgers | enumerate_shards | tool_episode | union (L) | hard repetition | reasoning | content | any tier |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 0/1 | 0/1 | 0/1 | 0/1 | 0/4 | 0/4 | 0 | 0 | 1 |

## Paired tests vs f16 (exact sign-flip, paired by task and seed)

| model | KV | metric | level | change | p | direction |
|---|---|---|---|---|---|---|

α = 0.05. The test is exact — every sign pattern is enumerated, no sampling and no normal approximation — so the same `samples.jsonl` always produces the same numbers. The smallest attainable p is 0.0000076 on 18 R pairs, 0.0312 on 6 C tasks and 0.0005 on 12 loop samples, so an arm must move nearly every paired sample the same way to be called different. 'Not significant' means the change is inside what a draw of this size produces; it is not proof of identity.

## Agreement with f16 per model (share of identical graded answers)


## Cross-model answer agreement (same R task + seed, same KV)


## Speed and realized context

| model | KV | prompt tokens | prefill t/s | decode t/s | cache hit rate | out tokens |
|---|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 10679 | 642.6 | 45.6 | 1.0 | 31815 |

## How long this took

| phase | wall clock | when (UTC) |
|---|---|---|
| matrix: 1 arms, 16 requests | 13 m 12 s | 2026-11-01T02:01 → 2026-11-01T02:14 |

Summed per-arm time is 13 m 11 s inside a 13 m 12 s matrix window: the difference is model load time and the one-off prefill of the shared document in every arm.

Per-arm wall clock (includes the document prefill and the server load):

| arm | wall clock | decode t/s | samples |
|---|---|---|---|
| `qwen36-35b__q4_0` | 13 m 11 s | 45.6 | 16 |

## Threats to validity (literal)

- Text-identical is not token-identical: one document, five tokenizers; realized prompt lengths are reported and `-c` is identical everywhere.
- Only full-attention layers accumulate KV damage over depth, so a model with few global attention layers can only show a small effect; the per-model layer counts for this run are in fit.json's geometry block.
- The C tier is one greedy pass per task, so it has no error bar; R and L have reps samples each. Any delta smaller than one task's worth of points is noise.
- Hard-task pass rates are confounded with absolute difficulty; tasks with no f16 success are labelled too_hard_for_matrix and excluded from the delta read.
- `repeat_penalty 1.0` in the L tier is more permissive than the production 1.05, so measured loop rates are an upper bound for production settings.
- Duplicated prefill within an arm means a bad early answer cannot contaminate later requests; `tool_episode` is the exception, its transcript is deliberately sequential.
- The repository fixture's third visible test requires `coverage_report` to surface a degenerate point gap, which the hidden check suite does not score; `repo_fix` records both, and `resolved` requires both.
- The `semver` module is bimodal on one prerelease-ordering decision (24/71/83/118/182/187/190/253/261 of 261): a C-tier delta of tens of points can be that single bit, and it is the largest single contributor to C-tier variance here.
- A C-tier arm that hits the generation cap produces truncated code, not wrong code. The report prints `out_tokens`, `finish_reason` and emitted line count per sample so the two are distinguishable; `budget_raised` marks every sample that was re-run at a larger budget.
- The zlib-ratio loop threshold fires on templated text: 25 of 45 `gen_manual` samples cross 0.42 with no line/ngram/cycle signal, uniformly across arms. Real repetition is the line-run, n-gram, block and cycle counters.
