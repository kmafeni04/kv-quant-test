# KV-cache quantization at long context — 5 models × f16 / q8_0 / q4_0

Context (identical for every arm): **-c 16384**. llama.cpp `2a667032dc78`, document sha256 `2b85355d504953ef`, reps {'R': 1, 'C': 1, 'L': 1}, seeds [1001, 1002, 1003], budgets {'C': 6144, 'L': 2048, 'R': 2048, 'tool_turn': 512}, thinking caps {'C': 3072, 'L': 256, 'R': 1024, 'tool_turn': 256}.

Total test wall clock (fit probe + matrix + harness validation, model downloads excluded): **—** (13 m 12 s of it the matrix).

## Per-model offload (frozen across the three KV arms)

| model | knobs | VRAM peak MiB | realized prompt tokens | prefill t/s | decode t/s |
|---|---|---|---|---|---|
| `qwen36-35b` | {'ncmoe': 26} | 9541 | 10679 | 642.6 | 45.6 |

## R tier — retrieval / tracing / aggregation (partial credit, %)

| model | f16 | q8_0 | q4_0 |
|---|---|---|---|
| `qwen36-35b` | — | — | 70.0 |

## C tier — hidden checks passed / total, and resolved tasks

| model | KV | checks | % | resolved | too-hard tasks |
|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 721/780 | 92.4 | 3/6 | — |

## Loop table — flagged samples / samples (reasoning and content split)

`union (L)` counts every verdict from the zlib-ratio heuristic or a structural signal; `hard repetition` counts only samples with a repeated line, n-gram, 200-char block or tail cycle. The gap between the two columns is formulaic prose, not looping.

| model | KV | gen_manual | copy_ledgers | enumerate_shards | tool_episode | union (L) | hard repetition | reasoning | content | any tier |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 0/1 | 0/1 | 0/1 | 0/1 | 0/4 | 0/4 | 0 | 0 | 1 |

## Paired deltas vs f16 (positive = worse looping, positive R/C = better)

| model | KV | ΔR pp | ΔC pp | Δloop-rate | answer agreement | n(R/C/L) |
|---|---|---|---|---|---|---|

A 1-point delta on 6 R tasks is one sub-value of one sample: noise. The agreement column is the low-level signal that still resolves arms when nothing fails outright.

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
- Only full-attention layers accumulate KV damage over depth (qwen35: 16 layers, MoE 35B: 10, Flash-Next: 12, Gemma 26B: 5 global). A flat result on a model with few global layers is partly structural.
- The C tier is one greedy pass per task, so it has no error bar; R and L have reps samples each. Any delta smaller than one task's worth of points is noise.
- Hard-task pass rates are confounded with absolute difficulty; tasks with no f16 success are labelled too_hard_for_matrix and excluded from the delta read.
- `repeat_penalty 1.0` in the L tier is more permissive than the production 1.05, so measured loop rates are an upper bound for production settings.
- Duplicated prefill within an arm means a bad early answer cannot contaminate later requests; `tool_episode` is the exception, its transcript is deliberately sequential.
- The repository fixture's third visible test requires `coverage_report` to surface a degenerate point gap, which the hidden check suite does not score; `repo_fix` records both, and `resolved` requires both.
- The `semver` module is bimodal on one prerelease-ordering decision (24/71/83/118/182/187/190/253/261 of 261): a C-tier delta of tens of points can be that single bit, and it is the largest single contributor to C-tier variance here.
- A C-tier arm that hits the generation cap produces truncated code, not wrong code. The report prints `out_tokens`, `finish_reason` and emitted line count per sample so the two are distinguishable; `budget_raised` marks every sample that was re-run at a larger budget.
- The zlib-ratio loop threshold fires on templated text: 25 of 45 `gen_manual` samples cross 0.42 with no line/ngram/cycle signal, uniformly across arms. Real repetition is the line-run, n-gram, block and cycle counters.
