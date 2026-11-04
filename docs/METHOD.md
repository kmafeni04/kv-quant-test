# Method

## Question

At one full-context rung per model, does quantizing the KV cache (the stored key/value attention
state for every token already read) from f16 to q4_0 degrade (a) the information a model can retrieve
from deep in a 92–95k-token document, (b) its ability to summarise that document for a later worker,
(c) whether long tool-loop work keeps its ledger state correct, where a lossy cache could drop a
mutation or apply it twice, and (d) its tendency to degenerate?

## Design

Only the cache type changes inside a model. Context rung, document, task text, guards and decode
settings are held fixed per model. The document is one canonical text; every arm of every model sees
the same bytes. The table below names the six factors that define an arm; the only one that changes
inside a model is the cache type.

| factor | levels |
|---|---|
| model | 3 in the matrix (Gemma-4-26B-A4B, Ornith-1.5-35B-A3B, Qwen3.6-35B-A3B); the dense 27B and Flash-Next remain configured for later cheap probes |
| KV cache | f16/f16 and q4_0/q4_0, symmetric, `-fa on` |
| context | the largest ladder rung (`102400, 98304, …, 49152`) that loads with a ≥ 91k-token prompt, resolved per model; all arms of a model share it |
| document | 92–95k tokens in the largest of the three tokenizers |
| decode | greedy, `temperature: 0.0` only, one draw per instance |
| response cap | 64k, runaway guard only; recorded as `halted_by=ceiling` |

The study uses one rung rather than a ladder because damage from a lossy cache scales with the number
of tokens in the cache. A ladder lets an arm that fails at 100k be credited with the 49k rung where it
succeeds, and makes the per-arm rung an uncontrolled variable inside every comparison. Each model
therefore runs at its own maximum affordable depth, and the report states that depth rather than
averaging over it.

Decode is greedy because sampling noise is the one cost the study can remove rather than pay for. A
stochastic decode needs several draws per instance to separate signal from noise; greedy is one draw
and the whole sample. The disclosed trade is that greedy is not a production setting and may be more
or less loop-prone, so every loop statement is scoped to greedy decode.

No reasoning budget is applied, because `--reasoning-budget` forces an end-of-thinking tag when it
fires. That is a harness-side loop breaker: a model that would rattle for 30k tokens is cut at 8,192
and answers from a coerced state, and the loop never reaches the transcript. The budget, the 4x
`budget_raised` retry and their plumbing are deleted.

The response record carries the information the budget used to hide:

- `thinking_tokens` and `thinking_share`, from streamed reasoning deltas over all generated deltas;
- time-to-first-visible-token on streaming single-response workflows (tool-loop requests are
  non-streaming and record per-step prompt tokens and wall time instead);
- the loop detector on `reasoning_content` and `content` separately, with the firing stream and
  token position;
- `halted_by=context_full` as a legitimate outcome with zero visible tokens.

The sampling audit is a server-side check rather than a promise. No sampler flags are passed to the
server and no sampler fields other than `temperature: 0.0` are sent. llama.cpp applies the model
file's `general.sampling.*` defaults for anything unset, so the server runs with `-v` and the resolved
`sampler params:` block is captured per arm as `sampler_resolved`; the report prints the resolved
temperature and whether greedy was in force.

## Workflows

The six workflows in the table are the whole study. `instances` is the count per arm, `guard` is the
hard stop that ends a run whether or not it finished, and `primary grading` is the measure the
workflow reports first.

| id | instances | guard | primary grading |
|---|---|---|---|
| W1 | 3 passes per arm | — | mean KLD vs the f16 logits, same-top, deep-context PPL ratio |
| W2 | 12 | 1 request | 64 keys correct in asked order, with `absent`/`mutated`/`misordered` per miss |
| W3 | 12 | 2 requests | brief retention out of 20; phase-B keys; degeneration |
| W4 | 6 | 120 tool calls | final ledger hash; five invariants |
| W5 | 6 | 60 tool calls | three multi-hop answer keys; search calls vs a floor of 9 |
| W6 | 6 | 64k ceiling | six sections present, six cross-references resolved |

The first workflow is the distribution instrument. It runs `llama-perplexity` from the pinned build.
Pass 1 writes the f16 baseline logits over a fixed slice (`-c 4000 --chunks 1`; the tool scores the
last half of the window, so about 2,000 tokens per chunk). Pass 2 runs every arm against that file
with `--kl-divergence`, printing mean KLD (Kullback–Leibler divergence), the PPL (perplexity) ratio,
the mean change in correct-token probability, the root-mean-square (RMS) change and same-top
agreement, each with propagated uncertainty. Pass 3 runs perplexity at depth with no KLD file.

The tool requires at least `2 × n_ctx` tokens in the file and accumulates host logits for the scored
half-window, `ctx/2 × vocab × 4` bytes. Slices are therefore built to `2 × ctx + 2048` tokens; the
extra tokens are a guard, identical across arms, and the scored window is the first `ctx`.

On this host the plan's `-c 102400` aborts with `std::bad_alloc` before scoring, because Gemma's
262,144-token vocabulary needs 50 GiB of host logits against 62 GiB of RAM. The default depth pass
therefore runs at 65,536 tokens (32 GiB, measured 82.9 s and PPL 177.16 on the pilot document, f16).
Each arm records `deep_ctx`, `--deep-ppl-ctx` raises it on a host with more RAM, and the report prints
the depth actually scored.

Baseline files are keyed by document sha256 and kept in `.cache/kv-quant-test` at the repository root,
out of the results tree; their sha256, byte count and slice sha256 are in the manifest.

The pinned tool's own `Mean PPL(base)` reconstruction is not used as evidence. An identical-model
check (f16 against its own logits) returned mean KLD `-0.000000 ± 0.000000`, same-top `100.000%`,
mean Δp `0.000%` and `Mean PPL(Q)` `801.133` — equal to that run's standalone PPL — while its
`Mean PPL(base)` printed `635.655`. The distributions were identical, so the reported base PPL is
inconsistent with the tool's own preceding run. The report therefore computes the ln PPL ratio from
each arm's own scored PPL against the f16 arm's PPL on the identical slice, and keeps the tool's KLD,
Δp and same-top outputs.

**W2.** All sixty-four planted shard codes, in the order the prompt names them, so the depths run
from 2% to 98% of the document and each answer position tests a different distance into the cache.
Each shard has a near-miss decoy: a code differing in one or two digits. The answer line is parsed for
code-shaped tokens in order, from `content` then from `reasoning_content`. Each position is `hit`,
`absent` (no value), `misordered` (the expected code appears at another position) or `mutated` (any
other wrong value, including the near-miss). Wrong, missing and duplicated positions all score zero,
so the score is monotone in what was lost.

The count is 64, not the 20 this probe asked for in `results/2026-11-02`. There, both cache types held
all twenty keys on every scored arm, and a probe nobody loses a key on answers no question about the
cache. Sixty-four keys give 768 key observations per twelve-instance arm and put every planted shard
under test. The 20 identifiers the W3 brief must carry are a separate set (`facts["w3_keys"]`),
spread across the same document, so the brief stays at a size a 1,500-word cap can hold.

**W3.** Phase A sends the full document and asks for a handover brief of at most 1,500 words carrying
twenty identifiers, the ledger rows a named tenant's mutations touch, and five invariants. Phase B
discards the document and the phase-A transcript, sends the brief verbatim, and asks five questions
answerable from it alone (three deep shard codes, one deep ledger amount, one tag). Retention is the
primary endpoint; phase B is the consequence channel; degeneration over both responses is behavioural.
The optional model-driven variant gives the model a `compact` tool with the same word cap and a wall
at 90% of the window, so an agent that never compacts is halted as `context_full`.

**W4.** The operation log is a file on disk: the shared document followed by 400 mutation records.
Twenty records per tenant are the target set; 280 records belong to non-study tenants. The prompt names
one tenant. The model reads the log through `read_ops(start, count)`, so the window fills as it reads,
and it applies the twenty records through `apply_append`, `apply_retract` and `apply_transfer` with
arguments copied from the record.

The harness keeps the authoritative ledger. Grading is the final state hash against the expected hash,
plus five invariants: the active total is conserved, retracted rows stay VOID, the row set matches the
log, every mutation is applied once in order, and no row changes tenant. Every applied mutation
increments the row's `rev`, so re-applying any mutation changes the hash.

**W5.** Sixty clusters in fifteen four-hop chains. `search(term)` returns the top-3 snippets by exact
and word overlap. Three questions each start at a chain head, follow `next` twice, and submit the
value through `submit_answer`. The answer is graded from tool state; calls are counted against a floor
of 9 (three searches per question) and a ceiling of 60.

**W6.** Six sections with given titles and a given cross-reference mapping, 120–220 words each, no
generation limit but the 64k ceiling. Grading parses section headings and `See Section <k>` lines;
the loop instrument runs over the whole output.

## Loop instrument

Per response, the detector runs on `reasoning_content` and `content` separately:

| metric | threshold |
|---|---|
| repeated identical line | 5 in a row |
| repeated single word | 8 in a row |
| repeated 4- or 8-word run | 6 in a row |
| repeated 200-character block | 3 occurrences |
| tail cycle | period 8 or more |
| compressed size / raw size | 0.42 or more, with 4-gram diversity 0.60 or less |
| monotone template run | 40 identical line templates |
| long low-diversity trace | 4,000 characters at 4-gram diversity 0.25 or less |

Thresholds are module constants in `kvtest/loops.py`; the raw numbers and firing span are recorded next
to every verdict. For tool loops, each step is scored for an identical consecutive call, an identical
call three times, oscillation between two calls (A, then B, then A again), and a six-step no-progress
window (unchanged state hash). A no-progress window excludes reads that advance the window, so a model
that reads new parts of the log is making progress.

## Statistics

The unit of analysis is a workflow instance, not a request, a token or a repetition. Because decode is
greedy, one draw per instance is the whole sample. The instance count per workflow is fixed by the
table above, so the smallest attainable two-sided p is 2/2¹² = 0.0005 for W2 and W3 and 0.031 for
W4–W6.

W2 and W3 are counts rather than verdicts. The report gives the mean out of 64 for W2 and out of 20 for
W3, each with its standard
deviation, the distribution across instances, and the paired difference against that model's f16 arm
with a percentile cluster bootstrap over instances (1,000 resamples, seeded). A one-key shift is
visible in the mean and in the per-depth curve even when no instance is dramatically wrong. Depth is
reported as a curve, not a total: W2 hit rate by planted-depth band with Wilson intervals.

Behaviour rates carry intervals but never ranking weight. Loop halts, context-full halts, ceiling
hits, thinking share, tokens and wall clock per instance are secondary and never the basis of a
ranking on their own.

The decision rule is pre-registered: an arm is worse if the lower bound of the paired interval
excludes 0 and the point estimate is worse; ties otherwise. Comparisons are only within a model, never
across models.

## Harness changes from the prior study

Nine changes separate this harness from the one that produced `results/2026-11-01/`; each removes a
way the old harness could decide a result before the model did.

1. Multi-turn tool runner with a step budget, transcript recorder (tool, args, ok, state hash before
   and after, token position) and per-step loop detection.
2. Compaction engine: forced two-phase pair, brief re-inserted verbatim, document removed, compaction
   events recorded; the optional model-driven `compact` tool with the 90% wall.
3. KLD/PPL runner that drives the pinned `llama-perplexity` with the arm's `-ctk`/`-ctv` and offload
   knobs, writes the f16 baseline once per model, and parses its metrics into the arm record.
4. Greedy sampling everywhere; no seeds or reps; `sampler_resolved` per arm; reasoning-budget machinery
   deleted; the 64k ceiling is the only cap.
5. 64-key task builder with per-key `absent`/`mutated`/`misordered` grading and the depth matrix.
6. State graders: ledger hash plus invariants, brief retention, search counts, long-form structure.
7. `extract_*` hardening: `ANSWER:` parsed from reasoning as well as content, and a wrapped or
   newline-separated answer block accepted for the 64-key line.
8. Manifest: one rung per model, model-file sha256, declared `general.sampling.*`, resolved sampler per
   arm, KLD baseline and slice hashes/offsets, hardware at matrix start.
9. Retired: codegen tier, repo tier, tier budgets, `REASONING_BUDGET_TOKENS`, `BUDGET_RETRY_MULT`,
   `SAMPLING_LOOP`, per-tier seed lists, the old codegen/intervalkit fixtures.

## Threats to validity

1. **Realized prompts differ across tokenizers.** One text, three tokenizers: prompt lengths differ
   (recorded per arm), though the f16-to-q4_0 comparison inside a model is unaffected.
2. **Greedy decode is one draw.** There is no sampling error bar; the paired bootstrap over instances
   accounts for instance-to-instance variance, and loop counts are scoped to greedy decoding.
3. **W4 pays its own prefill.** Its window fills through tool reads of an on-disk log, so it cannot
   reuse another workflow's cached prefix, though every W4 instance sees identical text across arms.
4. **Deep PPL stops at 65,536 tokens.** The pinned tool needs `2 × ctx` tokens in the file and
   allocates `ctx/2 × vocab × 4` bytes of host memory; at 102,400 that aborts on this 62 GiB machine,
   and every W1 record names the depth it scored.
5. **Phase B asks five questions at once.** A model that emits fewer than five `ANSWER:` lines loses
   the missing keys; that is short-context formatting, reported as phase-B keys and not folded into
   retention.
6. **W2 is at its ceiling.** The 20-key version was held in full by every scored arm in
   `results/2026-11-02`, so it measured format compliance. W2 now asks for all 64 planted shards; if
   every scored arm still holds all 64, the report flags saturation as a finding and the probe is not
   widened again, because a longer answer line would test the answer format rather than the cache.
7. **The rung can differ per model.** Each model runs at its own largest affordable rung, so
   cross-model rows are descriptive and never compared at unequal context.
8. **Reasoning is unbounded.** A model that spends the whole window thinking is a recorded
   `context_full` outcome with zero visible tokens; it is reported, never retried and never scored.
