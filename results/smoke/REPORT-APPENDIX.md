# Appendix — every table, method and provenance

## Appendix: full tables, method and provenance


## Hardware and build this ran on

The machine and the compiler settings decide how much of the model fits on the GPU and how fast it
answers, so they are recorded next to the scores. Everything below was captured on the host that ran
the matrix.

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

The table ranks the arms by capability, best first. The `p vs #1` column is the paired test against
the top arm and the last column says whether that test separated the two from each other; in this
run nothing separates.

| rank | arm | capability | structural loops | decode t/s | p vs #1 | distinguishable from #1? |
|---|---|---|---|---|---|---|
| 1 | `qwen36-35b` **q4_0** | 81.22 | 0/4 | 45.6 | 1.0000 | no — tied |
## Per-model offload (knobs are frozen across the three KV arms)

Knobs are identical for a model's three arms; the VRAM peak and speeds below are the **f16 arm's**. 

| model | weights off the GPU | VRAM peak MiB (f16) | realized prompt tokens | prefill t/s (f16) | decode t/s (f16) |
|---|---|---|---|---|---|
| `qwen36-35b` | 26 layers on CPU | 9541 | 10679 | 642.6 | 45.6 |

## R tier: retrieval, tracing, aggregation (partial credit, %)

Retrieval is scored per sub-value rather than per task, so a task asking for three values
contributes three points. The table gives each model's share of the 30 available points, with the
three reps already pooled, which is why there is no error bar here.

| model | f16 | q8_0 | q4_0 |
|---|---|---|---|
| `qwen36-35b` | — | — | 70.0 |

## C tier: hidden checks passed / total, and resolved tasks

Code is scored two ways from the same generations: checks passed out of 780 hidden checks, and whole
tasks resolved out of 6, where resolved means every check on that task passes. Each task got one
greedy pass, so these counts carry no sampling noise and no error bar.

| model | KV | checks | % | resolved | too-hard tasks |
|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 721/780 | 92.4 | 3/6 | — |

## Loop table: flagged samples / samples (reasoning and content split)

`union (L)` counts every verdict from the zlib-ratio heuristic or a structural signal; `hard repetition` counts only samples with a repeated line, n-gram, 200-char block or tail cycle. The gap between the two columns is formulaic prose, not looping.

| model | KV | gen_manual | copy_ledgers | enumerate_shards | tool_episode | union (L) | hard repetition | reasoning | content | any tier |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 0/1 | 0/1 | 0/1 | 0/1 | 0/4 | 0/4 | 0 | 0 | 1 |

## Paired tests vs f16 (exact sign-flip, paired by task and seed)

Each row compares one arm with its own model's f16 arm on one tier, and `change` is the arm minus
f16 in that tier's units. An arm therefore appears three times: once for retrieval points, once for
code checks, once for loop flags, because pooling those units into one number would let the
780-check tier outweigh the 0/1 flags.

| model | KV | metric | level | change | p | direction |
|---|---|---|---|---|---|---|

α = 0.05. The test is exact — every sign pattern is enumerated, no sampling and no normal approximation — so the same `samples.jsonl` always produces the same numbers. The smallest attainable p is 0.0000076 on 18 R pairs, 0.0312 on 6 C tasks and 0.0005 on 12 loop samples, so an arm must move nearly every paired sample the same way to be called different. 'Not significant' means the change is inside what a draw of this size produces; it is not proof of identity.

## Agreement with f16 per model (share of identical graded answers)

Agreement measures something pass rates cannot: whether the quantized arm produced the same answer
as f16 even when both were right. Each line gives one model and the share of graded answers that
were byte-identical to its f16 answer, over the samples the pair had in common. A share near 1.0
means the cache setting did not change the output at all, and the denominator after each share shows
how much the comparison rests on; Gemma's five samples are small for that reason.

## Cross-model answer agreement (same R task + seed, same KV)

Two models answering the same retrieval task, with the same seed and the same cache type, can agree
or disagree, and the share that agree says how much of the answer comes from the document rather
than from the model. Each block below is one cache type and each row is a model pair, so the f16
block is the baseline to read the other two against.

## Speed and realized context

The table prices the three cache settings against each other on the same work: prompt tokens
actually used after tokenization, prefill and decode rates at depth, the share of the prompt served
from the reused prefix, and total tokens generated. A high decode rate with a low generated count
means the arm answered less, not that it was faster.

| model | KV | prompt tokens | prefill t/s | decode t/s | cache hit rate | out tokens |
|---|---|---|---|---|---|---|
| `qwen36-35b` | q4_0 | 10679 | 642.6 | 45.6 | 1.0 | 31815 |

## How long this took

Wall clock is reported per phase because the phases cost different amounts: the matrix dominates,
and the fit probe pays for itself by resolving the context once instead of once per arm. The `when`
column gives UTC start and end times, which is what makes a resumed or restarted run auditable.

| phase | wall clock | when (UTC) |
|---|---|---|
| matrix: 1 arms, 16 requests | 13 m 12 s | 2026-11-01T02:01 → 2026-11-01T02:14 |

Summed per-arm time is 13 m 11 s inside a 13 m 12 s matrix window: the difference is model load time and the one-off prefill of the shared document in every arm.

Per-arm wall clock (includes the document prefill and the server load):

| arm | wall clock | decode t/s | samples |
|---|---|---|---|
| `qwen36-35b__q4_0` | 13 m 11 s | 45.6 | 16 |

## Threats to validity (literal)

Ten limits apply to every number in this report, ordered so that the ones deciding how far a
comparison reaches come first: one document is not one token sequence, and only some layers hold a
cache that grows with depth. The rest describe scoring artefacts that are visible in the artifacts,
and each names the column or flag that shows it.

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
