# KV-cache quantization at long context — 5 models × f16 / q8_0 / q4_0

Context (identical for every arm): **-c 102400**. llama.cpp `2a667032dc78`, document sha256 `944bb825ead422d7`, reps {'R': 3, 'C': 1, 'L': 3}, seeds [1001, 1002, 1003], budgets {'C': 6144, 'L': 2048, 'R': 2048, 'tool_turn': 512}, thinking caps {'C': 3072, 'L': 256, 'R': 1024, 'tool_turn': 256}.

Total test wall clock (fit probe + matrix + harness validation, model downloads excluded): **15 h 27 m** (14 h 12 m of it the matrix).

## Per-model offload (frozen across the three KV arms)

| model | knobs | VRAM peak MiB | realized prompt tokens | prefill t/s | decode t/s |
|---|---|---|---|---|---|
| `gemma4-26b-a4b` | {'ncmoe': 8} | 13757 | 81845 | 34.5 | 46.9 |
| `ornith-35b` | {'ncmoe': 16} | 15468 | 91357 | 67.4 | 48.8 |
| `qwen36-35b` | {'ncmoe': 26} | 11619 | 91357 | 57.2 | 38.7 |
| `qwen38-27b` | {'ncffn': 8} | 15899 | 91357 | 17.9 | 12.4 |
| `qwen38-flash` | {'ncmoe': 48} | 11295 | 91357 | 21.5 | 16.6 |

## R tier — retrieval / tracing / aggregation (partial credit, %)

| model | f16 | q8_0 | q4_0 |
|---|---|---|---|
| `gemma4-26b-a4b` | 36.7 | 23.3 | 26.7 |
| `ornith-35b` | 63.3 | 53.3 | 60.0 |
| `qwen36-35b` | 60.0 | 63.3 | 66.7 |
| `qwen38-27b` | 70.0 | 76.7 | 70.0 |
| `qwen38-flash` | 76.7 | 80.0 | 80.0 |

## C tier — hidden checks passed / total, and resolved tasks

| model | KV | checks | % | resolved | too-hard tasks |
|---|---|---|---|---|---|
| `gemma4-26b-a4b` | f16 | 694/780 | 89.0 | 4/6 | — |
| `gemma4-26b-a4b` | q8_0 | 701/780 | 89.9 | 3/6 | — |
| `gemma4-26b-a4b` | q4_0 | 318/780 | 40.8 | 3/6 | — |
| `ornith-35b` | f16 | 548/780 | 70.3 | 4/6 | — |
| `ornith-35b` | q8_0 | 570/780 | 73.1 | 3/6 | — |
| `ornith-35b` | q4_0 | 475/780 | 60.9 | 3/6 | — |
| `qwen36-35b` | f16 | 765/780 | 98.1 | 4/6 | — |
| `qwen36-35b` | q8_0 | 694/780 | 89.0 | 3/6 | — |
| `qwen36-35b` | q4_0 | 763/780 | 97.8 | 3/6 | — |
| `qwen38-27b` | f16 | 629/780 | 80.6 | 4/6 | — |
| `qwen38-27b` | q8_0 | 702/780 | 90.0 | 4/6 | — |
| `qwen38-27b` | q4_0 | 634/780 | 81.3 | 4/6 | — |
| `qwen38-flash` | f16 | 702/780 | 90.0 | 4/6 | — |
| `qwen38-flash` | q8_0 | 630/780 | 80.8 | 4/6 | — |
| `qwen38-flash` | q4_0 | 701/780 | 89.9 | 3/6 | — |

## Loop table — flagged samples / samples (reasoning and content split)

`union (L)` counts every verdict from the zlib-ratio heuristic or a structural signal; `hard repetition` counts only samples with a repeated line, n-gram, 200-char block or tail cycle. The gap between the two columns is formulaic prose, not looping.

| model | KV | gen_manual | copy_ledgers | enumerate_shards | tool_episode | union (L) | hard repetition | reasoning | content | any tier |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | f16 | 1/3 | 0/3 | 2/3 | 1/3 | 4/12 | 3/12 | 1 | 3 | 14 |
| `gemma4-26b-a4b` | q8_0 | 0/3 | 0/3 | 2/3 | 0/3 | 2/12 | 2/12 | 0 | 2 | 13 |
| `gemma4-26b-a4b` | q4_0 | 0/3 | 0/3 | 3/3 | 0/3 | 3/12 | 3/12 | 0 | 3 | 16 |
| `ornith-35b` | f16 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 4 |
| `ornith-35b` | q8_0 | 3/3 | 0/3 | 1/3 | 0/3 | 4/12 | 1/12 | 0 | 4 | 4 |
| `ornith-35b` | q4_0 | 3/3 | 0/3 | 0/3 | 0/3 | 3/12 | 0/12 | 0 | 3 | 5 |
| `qwen36-35b` | f16 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 3 |
| `qwen36-35b` | q8_0 | 2/3 | 0/3 | 1/3 | 0/3 | 3/12 | 1/12 | 0 | 3 | 4 |
| `qwen36-35b` | q4_0 | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 | 0/12 | 0 | 0 | 2 |
| `qwen38-27b` | f16 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 4 |
| `qwen38-27b` | q8_0 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 2 |
| `qwen38-27b` | q4_0 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 3 |
| `qwen38-flash` | f16 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 2 |
| `qwen38-flash` | q8_0 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 2 |
| `qwen38-flash` | q4_0 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 2 |

## Paired deltas vs f16 (positive = worse looping, positive R/C = better)

| model | KV | ΔR pp | ΔC pp | Δloop-rate | answer agreement | n(R/C/L) |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | q8_0 | -13.4 | +0.9 | -0.2 | 0.2 | 18/6/12 |
| `gemma4-26b-a4b` | q4_0 | -10.0 | -48.2 | -0.1 | 0.2 | 18/6/12 |
| `ornith-35b` | q8_0 | -10.0 | +2.8 | +0.2 | 0.5 | 18/6/12 |
| `ornith-35b` | q4_0 | -3.3 | -9.4 | +0.1 | 0.706 | 18/6/12 |
| `qwen36-35b` | q8_0 | +3.3 | -9.1 | +0.1 | 0.722 | 18/6/12 |
| `qwen36-35b` | q4_0 | +6.7 | -0.3 | -0.2 | 0.556 | 18/6/12 |
| `qwen38-27b` | q8_0 | +6.7 | +9.4 | +0.0 | 0.722 | 18/6/12 |
| `qwen38-27b` | q4_0 | +0.0 | +0.7 | +0.0 | 0.722 | 18/6/12 |
| `qwen38-flash` | q8_0 | +3.3 | -9.2 | +0.0 | 0.611 | 18/6/12 |
| `qwen38-flash` | q4_0 | +3.3 | -0.1 | +0.0 | 0.667 | 18/6/12 |

A 1-point delta on 6 R tasks is one sub-value of one sample: noise. The agreement column is the low-level signal that still resolves arms when nothing fails outright.

## Agreement with f16 per model (share of identical graded answers)

- `gemma4-26b-a4b` — q8_0: 0.2 of 5 samples; q4_0: 0.2 of 5 samples
- `ornith-35b` — q8_0: 0.5 of 18 samples; q4_0: 0.706 of 17 samples
- `qwen36-35b` — q8_0: 0.722 of 18 samples; q4_0: 0.556 of 18 samples
- `qwen38-27b` — q8_0: 0.722 of 18 samples; q4_0: 0.722 of 18 samples
- `qwen38-flash` — q8_0: 0.611 of 18 samples; q4_0: 0.667 of 18 samples

## Cross-model answer agreement (same R task + seed, same KV)


**f16**

| pair | identical / compared | share |
|---|---|---|
| `gemma4-26b-a4b` vs `ornith-35b` | 4/9 | 0.444 |
| `gemma4-26b-a4b` vs `qwen36-35b` | 3/9 | 0.333 |
| `gemma4-26b-a4b` vs `qwen38-27b` | 4/9 | 0.444 |
| `gemma4-26b-a4b` vs `qwen38-flash` | 4/9 | 0.444 |
| `ornith-35b` vs `qwen36-35b` | 13/18 | 0.722 |
| `ornith-35b` vs `qwen38-27b` | 10/18 | 0.556 |
| `ornith-35b` vs `qwen38-flash` | 10/18 | 0.556 |
| `qwen36-35b` vs `qwen38-27b` | 9/18 | 0.5 |
| `qwen36-35b` vs `qwen38-flash` | 9/18 | 0.5 |
| `qwen38-27b` vs `qwen38-flash` | 12/18 | 0.667 |

**q8_0**

| pair | identical / compared | share |
|---|---|---|
| `gemma4-26b-a4b` vs `ornith-35b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen36-35b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen38-27b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen38-flash` | 1/7 | 0.143 |
| `ornith-35b` vs `qwen36-35b` | 8/18 | 0.444 |
| `ornith-35b` vs `qwen38-27b` | 8/18 | 0.444 |
| `ornith-35b` vs `qwen38-flash` | 9/18 | 0.5 |
| `qwen36-35b` vs `qwen38-27b` | 10/18 | 0.556 |
| `qwen36-35b` vs `qwen38-flash` | 9/18 | 0.5 |
| `qwen38-27b` vs `qwen38-flash` | 11/18 | 0.611 |

**q4_0**

| pair | identical / compared | share |
|---|---|---|
| `gemma4-26b-a4b` vs `ornith-35b` | 2/7 | 0.286 |
| `gemma4-26b-a4b` vs `qwen36-35b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen38-27b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen38-flash` | 1/7 | 0.143 |
| `ornith-35b` vs `qwen36-35b` | 10/17 | 0.588 |
| `ornith-35b` vs `qwen38-27b` | 9/17 | 0.529 |
| `ornith-35b` vs `qwen38-flash` | 9/17 | 0.529 |
| `qwen36-35b` vs `qwen38-27b` | 11/18 | 0.611 |
| `qwen36-35b` vs `qwen38-flash` | 11/18 | 0.611 |
| `qwen38-27b` vs `qwen38-flash` | 12/18 | 0.667 |

## Speed and realized context

| model | KV | prompt tokens | prefill t/s | decode t/s | cache hit rate | out tokens |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | f16 | 81845 | 34.5 | 46.9 | 1.0 | 121371 |
| `gemma4-26b-a4b` | q8_0 | 81845 | 43.6 | 33.8 | 1.0 | 135175 |
| `gemma4-26b-a4b` | q4_0 | 81845 | 51.7 | 34.3 | 1.0 | 153433 |
| `ornith-35b` | f16 | 91357 | 67.4 | 48.8 | 1.0 | 47391 |
| `ornith-35b` | q8_0 | 91357 | 59.6 | 35.7 | 1.0 | 48370 |
| `ornith-35b` | q4_0 | 91357 | 59.7 | 35.2 | 1.0 | 45415 |
| `qwen36-35b` | f16 | 91357 | 57.2 | 38.7 | 1.0 | 49725 |
| `qwen36-35b` | q8_0 | 91357 | 50.8 | 31.2 | 1.0 | 48413 |
| `qwen36-35b` | q4_0 | 91357 | 51.6 | 30.8 | 1.0 | 49339 |
| `qwen38-27b` | f16 | 91357 | 17.9 | 12.4 | 1.0 | 42855 |
| `qwen38-27b` | q8_0 | 91357 | 16.3 | 10.3 | 1.0 | 42139 |
| `qwen38-27b` | q4_0 | 91357 | 15.6 | 9.7 | 1.0 | 43006 |
| `qwen38-flash` | f16 | 91357 | 21.5 | 16.6 | 1.0 | 43678 |
| `qwen38-flash` | q8_0 | 91357 | 20.4 | 12.6 | 1.0 | 43793 |
| `qwen38-flash` | q4_0 | 91357 | 21.2 | 12.3 | 1.0 | 39776 |

## How long this took

| phase | wall clock | when (UTC) |
|---|---|---|
| harness self-test + smoke runs | 40 m 00 s | 2026-11-01T00:02 → 2026-11-01T03:15 |
| fit probe (context + offload + depth) | 35 m 29 s | 2026-11-01T00:02 → 2026-11-01T03:50 |
| matrix: 15 arms, 540 requests | 14 h 12 m | 2026-11-01T03:53 → 2026-11-01T18:05 |
| report generation | 1 m 00 s | — → — |
| **total test time (phases summed, provisioning excluded)** | **15 h 27 m** | 2026-11-01T00:02 → 2026-11-01T18:05 |

Summed per-arm time is 13 h 47 m inside a 14 h 12 m matrix window: the difference is model load time and the one-off prefill of the shared document in every arm.

run1 (03:53Z) was stopped by an arm-start failure at 09:19Z; run2 resumed the 9 unfinished arms from their per-arm checkpoints, so no sample was regenerated.

Provenance: fit_probe = sum of wall_s over the five fit invocations (fit.json reports only the last); harness_selftest_and_smoke = three --smoke arm durations plus six selftest runs; matrix = min(arm.started) -> max(arm.finished) over the 15 arm checkpoints; per_arm = wall_s recorded by the orchestrator in each arm checkpoint; study_wall_s = sum of the work phases below (they partially overlapped the excluded downloads).

Per-arm wall clock (includes the document prefill and the server load):

| arm | wall clock | decode t/s | samples |
|---|---|---|---|
| `qwen36-35b__f16` | 27 m 31 s | 38.7 | 36 |
| `qwen36-35b__q8_0` | 31 m 50 s | 31.2 | 36 |
| `qwen36-35b__q4_0` | 32 m 30 s | 30.8 | 36 |
| `qwen38-27b__f16` | 1 h 06 m | 12.4 | 36 |
| `qwen38-27b__q8_0` | 1 h 18 m | 10.3 | 36 |
| `qwen38-27b__q4_0` | 1 h 27 m | 9.7 | 36 |
| `qwen38-flash__f16` | 55 m 26 s | 16.6 | 36 |
| `qwen38-flash__q8_0` | 1 h 09 m | 12.6 | 36 |
| `qwen38-flash__q4_0` | 1 h 05 m | 12.3 | 36 |
| `gemma4-26b-a4b__f16` | 56 m 37 s | 46.9 | 36 |
| `gemma4-26b-a4b__q8_0` | 1 h 24 m | 33.8 | 36 |
| `gemma4-26b-a4b__q4_0` | 1 h 35 m | 34.3 | 36 |
| `ornith-35b__f16` | 21 m 00 s | 48.8 | 36 |
| `ornith-35b__q8_0` | 27 m 33 s | 35.7 | 36 |
| `ornith-35b__q4_0` | 26 m 34 s | 35.2 | 36 |

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
