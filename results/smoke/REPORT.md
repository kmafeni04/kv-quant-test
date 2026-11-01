# KV-cache quantization at long context — 5 models × f16 / q8_0 / q4_0

Context (identical for every arm): **-c 16384**. llama.cpp `2a667032dc78`, document sha256 `2b85355d504953ef`, reps {'R': 1, 'C': 1, 'L': 1}, seeds [1001, 1002, 1003], budgets {'C': 6144, 'L': 2048, 'R': 2048, 'tool_turn': 512}, thinking caps {'C': 3072, 'L': 256, 'R': 1024, 'tool_turn': 256}.

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

| model | KV | gen_manual | copy_ledgers | enumerate_shards | tool_episode | union (L) | reasoning | content | any tier |
|---|---|---|---|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 0/1 | 0/1 | 0/1 | 0/1 | 0/4 | 0 | 0 | 1 |

## Paired deltas vs f16 (positive = worse looping, positive R/C = better)

| model | KV | ΔR pp | ΔC pp | Δloop-rate | answer agreement | n(R/C/L) |
|---|---|---|---|---|---|---|

A 1-point delta on 6 R tasks is one sub-value of one sample: noise. The agreement column is the low-level signal that still resolves arms when nothing fails outright.

## Agreement with f16 per model (share of identical graded answers)


## Speed and realized context

| model | KV | prompt tokens | prefill t/s | decode t/s | cache hit rate | out tokens |
|---|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 10679 | 642.6 | 45.6 | 1.0 | 31815 |

## Threats to validity (literal)

- Text-identical is not token-identical: one document, five tokenizers; realized prompt lengths are reported and `-c` is identical everywhere.
- Only full-attention layers accumulate KV damage over depth (qwen35: 16 layers, MoE 35B: 10, Flash-Next: 12, Gemma 26B: 5 global). A flat result on a model with few global layers is partly structural.
- The C tier is one greedy pass per task, so it has no error bar; R and L have reps samples each. Any delta smaller than one task's worth of points is noise.
- Hard-task pass rates are confounded with absolute difficulty; tasks with no f16 success are labelled too_hard_for_matrix and excluded from the delta read.
- `repeat_penalty 1.0` in the L tier is more permissive than the production 1.05, so measured loop rates are an upper bound for production settings.
- Duplicated prefill within an arm means a bad early answer cannot contaminate later requests; `tool_episode` is the exception, its transcript is deliberately sequential.
- The repository fixture's third visible test requires `coverage_report` to surface a degenerate point gap, which the hidden check suite does not score; `repo_fix` records both, and `resolved` requires both.
