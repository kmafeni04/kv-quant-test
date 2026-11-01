# KV-cache quantization at long context — 5 models × f16 / q8_0 / q4_0

Context (identical for every arm): **-c 16384**. llama.cpp `2a667032dc78`, document sha256 `2b85355d504953ef`, reps {'R': 1, 'C': 1, 'L': 1}, seeds [1001, 1002, 1003], budgets {'C': 6144, 'L': 2048, 'R': 2048, 'tool_turn': 512}, thinking caps {'C': 3072, 'L': 256, 'R': 1024, 'tool_turn': 256}.

Total test wall clock (fit probe + matrix + harness validation, model downloads excluded): **—** (13 m 12 s of it the matrix).


## Verdict

**Best overall: `qwen36-35b` with a `q4_0` KV cache** — capability 81.20 of 100 (R 70.0% and C 92.4% averaged), 0/4 structural-loop samples, 45.6 t/s decode, 9541 MiB peak with {'ncmoe': 26} offloaded.

Ranking rule, fixed in `kvtest/report.py::verdict` so it can be recomputed: capability = mean(R %, C %) per arm, then fewer structural-loop samples, then higher decode throughput. **No arm differs from its own f16 arm at p < 0.05 on any metric**, so these ranks sit inside the noise this design can resolve — the reasons column says what the numbers actually support.

| model | best KV | capability | runner-up | structural loops | decode t/s | why |
|---|---|---|---|---|---|---|
| `qwen36-35b` | **q4_0** | 81.20 | None | 0/4 | 45.6 | 0/4 structural-loop samples, 45.6 t/s decode |

### All 15 arms, ranked by the same rule

| rank | arm | capability | structural loops | decode t/s |
|---|---|---|---|---|
| 1 | `qwen36-35b` **q4_0** | 81.20 | 0/4 | 45.6 |
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

## Paired tests vs f16 (exact sign-flip, paired by task and seed)

| model | KV | metric | level | change | p | direction |
|---|---|---|---|---|---|---|

α = 0.05. The test is exact — every sign pattern is enumerated, no sampling and no normal approximation — so the same `samples.jsonl` always produces the same numbers. With 6 C-tier tasks the smallest attainable p is 0.031 and with 12 loop samples it is 0.0005, so an arm must move nearly every paired sample the same way to be called different. 'Not significant' means the change is inside what a draw of this size produces; it is not proof of identity.

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
