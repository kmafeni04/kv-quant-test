# Method

## Question

At one identical long context, on five models served by llama.cpp, does the KV cache quantization
(f16 → q8_0 → q4_0) degrade (a) hard, automatically verifiable task performance and (b) the
model's tendency to loop — repeat the same words, commands or tool calls without doing work?

## Hypotheses

- **H1 (capability):** q8_0 is indistinguishable from f16; q4_0 loses some points on the retrieval
  and code tiers. Prior art at 32k on the dense 27B read flat on capability and separated only in
  KL (0 → 0.00066 → 0.0032 mean KLD vs f16, same-top 100 → 99.1 → 97.4 %). This study pushes the
  same ladder to ~96k tokens, where every full-attention layer has seen 3× more positions.
- **H2 (degeneration):** q4_0 increases the loop rate. This is the metric the earlier study did not
  have: KV quantization perturbs the attention distribution a little, and a small perturbation near
  a repetition attractor can flip a sample into a loop without changing its answer.
- **H3 (structure):** any capability effect scales with the number of *global* attention layers
  (qwen38-27b: 16, Flash-Next: 12, the 35B MoEs: 10, Gemma 26B: 5), because only those layers hold
  KV that grows with depth. A flat q4_0 result on a model with few global layers is partly
  structural, not evidence that KV quantization is harmless in general.

## Design

| factor | levels |
|---|---|
| model | 5 (2 dense-hybrid, 3 MoE; see `kvtest/config.py`) |
| KV cache | f16/f16, q8_0/q8_0, q4_0/q4_0 (symmetric, `-fa on`) |
| context | one value for every arm, resolved by the fit probe from `102400 … 49152` |
| tier R | 6 retrieval/tracing tasks × 3 reps, temp 0.7 |
| tier C | 6 code tasks × 1 greedy pass, temp 0 |
| tier L | 4 loop probes × 3 reps, temp 0.7, repetition suppression OFF |

**Why one document.** All twelve graded tasks append to a single canonical ~96k-token document, so
llama-server reuses the KV prefix for every subsequent request (`timings.cache_n` is recorded on
every sample and `cache_hit_rate` is reported). That makes 3 reps of 6 tasks affordable: the
expensive prefill is paid once per arm, not once per task.

**Why these tasks.** Everything is graded by regex or by running hidden checks. No judge model, no
human reading. Tier R answers are short strings or integers; tier C is the SWE-bench contract
(hidden tests, partial credit, plus an all-or-nothing "resolved" count); tier L is the degenerate
output measurement itself.

**Why symmetric KV and flash attention.** A quantized V cache refuses to load without flash
attention, and the pinned build compiled CUDA flash-attention kernels for exactly
`q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`. Asymmetric K/V is not a production setting here.

**Why no speculative decoding.** Speculative decoding verifies against the target model and cannot
change the output distribution, so it is irrelevant to the question, it costs VRAM the f16 arm
needs, and the MTP drafters would be extra downloads.

**Per-request thinking cap.** Reasoning is on at `--reasoning-effort medium` for every tier, and the
server is started with `--reasoning-budget 8192`. On top of that, each request carries
`reasoning_budget_tokens`: 1024 for tier R, 3072 for tier C, 256 for tiers L and the tool turns.
Reason: measured on this build, an uncapped medium-effort template spends the entire generation
budget inside the thinking block — the code tier produced 21,298 characters of reasoning and **zero
characters of code** at a 6,144-token cap (the earlier 32k study hit the same wall at 32,489
completion tokens and 0 lines). The sampler forces the end-of-thinking tag at the cap, after which
the model answers: with a 3,072 cap the same request emitted 71 lines of code and scored 29/29 on
`csvlite`. The cap is identical for all 15 arms and all three KV types, so it cannot bias the KV
comparison; it only guarantees that every tier can emit its answer.

**Why repetition suppression is off in tier L.** A repeat penalty is exactly the thing that masks
looping. Tier L runs `repeat_penalty 1.0`, `dry_multiplier 0`; tiers R and C run the production
`repeat_penalty 1.05`. Both are per-request fields of the same server, so one server per arm serves
all tiers.

**Tier budgets were measured, not assumed.** R = 2,048 tokens, C = 6,144, L = 2,048, tool turn =
512. L was raised from the designed 1,536 after the 40-shard listing ran out of room at exactly
1,536 tokens (88 lines emitted, 39/40 shards present, `finish_reason=length`); a truncation artefact
must not read as "did no work". The same budget is used in every arm.

**Why the code tier is not repeated.** At temperature 0, on a fixed build with a fixed `-ub` and a
fixed seed, generation is bit-reproducible, so repeating it would only burn hours. `--code-reps 3
--code-temp 0.7` exists for pass@k over a stochastic decode. This is a deliberate choice, recorded
here so it is auditable.

**Loop instrument.** Per sample, on `reasoning_content` and `content` separately: longest run of a
repeated 1/2/4/8-gram, longest run of identical lines, largest repeated 200-char block, tail cycle
period, zlib compression ratio, 4-gram diversity. Thresholds are constants at the top of
`kvtest/loops.py`; the raw numbers are always reported next to the verdict so a reader can apply
their own. The tool tier additionally flags an identical consecutive call, three identical calls,
or four turns without a successful read.

**Grading calibration.** Hard tasks can be too hard: if no f16 arm scores above zero on a code
task, that task is labelled `too_hard_for_matrix` and excluded from the KV-delta reading while
staying in the raw data. The f16 column is always shown, so a q4_0 drop is never read as damage when
the task is simply beyond every arm.

**Statistics on the figures.** Every cell of the three Matrix figures prints the level and the
change against that model's own f16 arm, and the change is coloured only if an exact two-sided paired
sign-flip test rejects at α = 0.05 (all 2ⁿ sign patterns enumerated over integer differences — no
sampling, no normal approximation, so the same `samples.jsonl` always yields the same p). Each figure
states its n and the smallest p attainable at that n (R: 2/2¹⁸, C: 2/2⁶, L: 2/2¹²), and the
leader comparison in the verdict tests each tier separately with a Bonferroni factor of 3 rather than
pooling R points, C checks and L flags — pooling incommensurable units would let the 261-check
samples outweigh the 0/1 flags. Where a stored
`timing.json` is absent, the per-arm times come from the arm checkpoints.

**Agreement matrix.** Beyond pass/fail: for graded R samples, the share of answers where q4_0 gives
exactly the f16 answer for the same task and seed. This is the output-level analogue of the KL
instrument that resolved the 32k study, and it still has signal when nothing fails outright.

## Fixtures

The five codegen modules and the intervalkit repository came from the user's own
`~/.local/share/llm-models/bench/fixtures` (the earlier 32k harness) and are used **verbatim** so
the numbers remain comparable; they are not third-party SWE-bench data. Real SWE-bench instances
were considered and rejected: a SWE-bench-Verified image per instance is 1–2 GB and cannot be
graded offline inside the time budget, whereas the codegen/intervalkit tiers supply the same
contract (hidden tests, FAIL_TO_PASS, partial credit) at zero provisioning cost.

## Threats to validity

1. **Text-identical is not token-identical.** One document text, five tokenizers: realized prompt
   lengths differ (reported per arm). `-c` is identical everywhere; the within-model KV ladder is
   the experiment, cross-model comparison is descriptive.
2. **Only full-attention layers accumulate KV damage over depth.** qwen38-27b quantizes 16 layers,
   Flash-Next 12, the 35B MoEs 10, Gemma-26B 5 global (its other 25 layers are capped at a 1024
   token window). A "q4_0 is fine" result on the 27B is expected and partly structural.
3. **Greedy ≠ stochastic.** Tier C is one greedy pass, no error bar. Tiers R and L have 3 samples
   per task. Any delta smaller than one task's worth of points is noise, and the report says so.
4. **Hard-task pass rates are confounded with absolute difficulty.** Handled by the
   `too_hard_for_matrix` calibration and by always showing the f16 column.
5. **Tier L is more permissive than production** (`repeat_penalty 1.0` vs 1.05), so measured loop
   rates are an upper bound for production settings. Both values are printed in the report.
6. **Duplicated prefill within an arm does not leak between requests** — each request's KV is
   rewound to the common document prefix — except in `tool_episode`, whose transcript is
   deliberately sequential.
7. **The intervalkit fixture's third visible test** needs `coverage_report` to surface a degenerate
   point gap, which the hidden check suite does not score. `repo_fix` records both the hidden
   checks and the visible test; `resolved` requires both, so a model that patches only `abuts` is
   scored on the hidden suite and marked unresolved.
8. **Measured, after the fact, and kept in the report.** (a) `semver` behaved exactly as the prior
   32k study predicted: one prerelease-ordering decision moves the module between 24 and 261 of 261
   checks, so a C-tier delta of tens of points can be one bit. (b) A C-tier arm that hits the
   generation ceiling produces *truncated* code, not wrong code — Gemma-4-26B-A4B at q4_0 truncated
   on three tasks — so `out_tokens`, `finish_reason` and emitted line count are reported per sample
   and every re-run at a larger budget is marked `budget_raised`. (c) The zlib-ratio loop threshold
   fires on templated prose: 25 of 45 `gen_manual` samples cross 0.42 with no line/ngram/cycle
   signal, uniformly across arms. The report therefore prints two counts: `union (L)` (any verdict)
   and `hard repetition` (repeated line, n-gram, 200-char block or tail cycle only). Example measured
   in this run: qwen36-35b's f16 "loops" are two `gen_manual` samples at zip 0.424/0.420 with
   `line_run=1`, i.e. formulaic prose on the wrong side of a threshold, while its one structural loop
   is at q8_0.
9. **The GPU was left free.** The Strata service that had been holding it was already stopped by the
   operator before this study began, so nothing was restarted at the end; the final state is
   13 MiB used on the GPU and no `llama-server` process.
10. **Downloaded weights are pinned by sha256**; the three pre-existing GGUFs were already on this
   host and their upstream provenance was not re-verified, but their digests are recorded and
   checked on every run.
