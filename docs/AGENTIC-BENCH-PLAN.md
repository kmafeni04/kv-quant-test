# Next study: agentic workflows at full context, scored per key

This is a design document; nothing here was run. It replaces the current study, which measured short
answers at a shrinking context with sampler settings it imposed on the models.

## What the current probes cannot see

The table lists the harness limits that made the last study unable to answer its question. The left
column names the limit, the middle column gives the value that was in force, and the right column says
what that value did to the measurement; in most rows the harness, not the model, decided when a probe
ended.

| limit | number | consequence |
|---|---|---|
| Generation cap | 2,048 tokens (1,024 of them thinking) | ~1,000 visible tokens per probe |
| Task size | `copy_ledgers` = 12 rows, codegen fixtures = one short script | Long-horizon failures never appear |
| Multi-key breadth | 3 keys | 9/9 in every graded arm |
| Interaction | 5 fixed tool calls, one round | No re-read or oscillation observed |
| Context depth | samples ran on a ladder down to 49k | Half the dose, unequal tests |
| Sampling | harness-forced `temperature 0.7, top_k 20, top_p 0.95, repeat_penalty 1.05` | Arms sampled off-distribution |
| Thinking | `--reasoning-budget 8192` plus an end-of-thinking tag forced at the cap | Harness breaks thinking loops |
| Statistics | 12 samples/arm = 4 probes × 3 reps | p = 0.125 before correction |

Each short cell above is one reason the earlier study could not answer its question:

- The generation cap left about 1,000 visible tokens per probe, so a long answer reached the grader
  looking the same as a cut-off one.
- The tasks were short enough to finish or truncate before any long-horizon failure appeared;
  `enumerate_shards` hit the cap in most arms, so its "loop" flag is partly a truncation artefact.
- Three keys were answered 9/9 in every graded arm, so the probe could not separate arms or show a
  gradient.
- Five fixed tool calls in one round cannot show re-reading a file, oscillation between edits,
  re-running a failing test unchanged, or declaring done unverified.
- The ladder ran down to 49k, where cache damage is half the dose and every model gets a different
  test.
- Sampling was forced to `temperature 0.7`, `top_k 20`, `top_p 0.95` and `repeat_penalty 1.05`, which
  put three of five arms outside their declared distribution and put the repeat penalty on top of the
  loop detector.
- The reasoning budget forced an end-of-thinking tag at the cap, so a model that looped in thinking
  still answered normally and the loop never reached the output.
- Twelve samples per arm gave the smallest attainable p = 0.125 before correction; the earlier
  "three separable arms" was pseudo-replication, counting three reps of one task as independent.

## Decisions

The seven decisions below are the design's fixed choices. The right-hand column names what each one
replaces, so a reader can see which limit of the earlier study it removes.

| # | decision | replaces |
|---|---|---|
| D1 | 64-key retrieval, **scored 0–64** per sample | 3-key task with partial credit out of 3 |
| D2 | One full-context rung per model, no ladder | Ladder 49k–100k |
| D3 | Compaction workflow | — |
| D4 | KLD and perplexity tier as the fast instrument | Terminal-Bench |
| D5 | Greedy sampling only, one sample per instance | Forced 0.7 / 1.05 penalty, three seeds per instance |
| D6 | No reasoning budget, no retry path | `--reasoning-budget`, `budget_raised` retries |
| D7 | Three fast models, tranched to 12 h inside a 24 h cap | Flat five-model matrix |

Facts checked before writing this, so the plan rests on the build actually in use. GGUF is the model
file format llama.cpp loads, and the KV cache is the stored attention state for every token already
read.

- Every model file declares its own sampler block: `general.sampling.temp = 1.0` on all five,
  `top_p = 0.95`, `top_k = 20` (Gemma: `top_k = 64`). The run passed `temperature 0.7, top_k 20` on
  the loop tier anyway, so Gemma was sampled at 0.7 and top_k 20 against its declared 1.0 and 64.
- llama.cpp applies `general.sampling.*` as defaults for any option the command line does not set
  (`common/common.cpp` 1127–1145, gated on `common_params_sampling_config`), and the server prints
  `sampler chain:` and `sampler params:` per request at trace level
  (`tools/server/server-context.cpp` 1947–1948). The resolved config is therefore auditable with `-v`.
- All five models load at `102400` with a ≥ 91k prompt (fit.json). Prefill at that depth is 100 s for
  Ornith (912 t/s) and Gemma (916 t/s), 122 s for Qwen3.6-35B (750 t/s), 217 s for the dense 27B
  (420 t/s) and 299 s for Flash-Next (305 t/s); decode is 48.0, 44.9, 36.4, 12.2 and 17.4 t/s in the
  same order. Two of the five are slow in both directions, which is what D7 acts on.
- The pinned build's `llama-perplexity` supports `-ctk`/`-ctv`, `--save-all-logits` (the
  `--kl-divergence-base` file), `--kl-divergence`, `--chunks` and `--ppl-stride`, and with
  `--kl-divergence` it reports mean KLD (Kullback–Leibler divergence), the PPL
  (perplexity) ratio, the RMS probability change, the Pearson correlation and same-top agreement, each with uncertainty. Its own README puts the logits file at
  11 GiB for a 240k-token corpus, which is why D4 uses a 4k-token slice.

## D1 — W2: 64-key retrieval, scored out of 64

A lossy cache does one thing early: it drops or mutates specific values held at depth. The right
instrument for that is a count, not a verdict, because the count says *how much* went missing and the
verdict only says that something did.

**The task.** All sixty-four planted shard codes, in the answer order asked, so their depths spread
across the whole document: the shallowest sits near the opening and the deepest near the closing lines,
which means each answer position tests a different distance into the cache. Each shard has a near-miss
decoy, a code differing in one or two digits, so surface matching does not score.

**The score.** Each sample gets an integer out of 64: the number of positions holding the correct code.
Wrong, missing and duplicated all count as zero for that position, which makes the score monotone in
what was lost and comparable across arms. A sample that loses one deep key scores 63; one that keeps
only the shallow half scores 32.

**Why 64 and not 20.** The twenty-key version ran in `results/2026-11-02` and both cache types held
all twenty keys on every scored arm, so the probe could not see a loss. Sixty-four keys give 768 key
observations per twelve-instance arm, put every planted shard under test, and leave a distribution
with room above the ceiling. The price is output length: the answer line grows from roughly 180 to 580
characters, which is still one request and well inside the ceiling.

**What is reported.**

- per-arm mean and standard deviation of the score, and the paired difference in keys against that
  model's f16 arm, with a cluster bootstrap interval over instances;
- the full per-key hit matrix, so a loss can be read as a curve rather than a total: hit rate by depth,
  which turns "63 of 64" into "the miss was at 94% depth";
- the failure-mode split per missed key: `absent`, `mutated` (near-miss decoy or digit-altered code) or
  `misordered`. `mutated` is the worse of the three, because a confidently wrong value is cache damage
  while an empty slot is at least honest.

**Why not a gate.** A gate reports one bit per sample, and the earlier 3-key version reported one bit
for the whole arm. Sixty-four counted keys give sixty-four distinctions per sample and a distribution
to test, so an arm that loses one deep key in every instance shows up as a shifted mean with an
interval, not as a tie. The cost is one request per sample, so the probe stays the cheapest in the
study.

**Breadth.** Twenty keys saturate, and did: `results/2026-11-02` was held in full by both cache types
on every scored arm. All sixty-four planted shards are therefore asked for from the start. If every
scored arm still holds all sixty-four, the report flags saturation as a finding and the probe is not
widened again, because a longer answer line would test the answer format rather than the cache.

## D2 — One rung: the whole window, for every arm

Each model runs at the largest ladder entry it loads with a ≥ 91k-token prompt, and every arm of that
model runs at that same rung. From fit.json that is `102400` for all five models, so no model is
measured at a different depth and no cross-model row is compared at unequal context.

What this fixes: on a ladder, an arm that fails at 100k silently gets credited with the 49k rung where
it succeeds, and the ladder's per-arm rung differences were an uncontrolled variable inside every
comparison. Damage from a lossy cache grows with the number of tokens in it, so half the window is
half the dose — measuring at the bottom of the ladder systematically under-reports damage.

Operationally:

- the shared document is retargeted so that the document alone is 92–95k tokens in the *largest* of the
  three tokenizers, and every workflow's prompt reads it to the end;
- generation happens on top of that, so most workflows operate within a few thousand tokens of the
  window edge by construction. That is intended: exceeding the window is a recorded outcome
  (`halted_by=context_full`), and D3 exists to measure whether the model survives being forced to make
  room;
- `fit.json` still records which rungs load, because a model that needs a lower rung to start must be
  reported rather than moved down without a note and compared anyway.

## D3 — W3: compaction

The only operation that must read the entire cache in one pass is summarising it. If quantized KV
loses values at depth, a compaction step is where the loss becomes visible and permanent, because the
summary is written from the damaged read and then becomes the only context that survives.

**Phase A (forced, deterministic).** The model gets the full 100k document and one instruction:
produce a handover brief of at most 1,500 words for a worker who will not see the document. The brief
must carry the twenty identifiers, the ledger rows the mutations touch, and the five invariants.

**Phase B (compacted context).** The harness discards the document *and* the phase-A transcript and
sends only the brief verbatim, then asks questions answerable solely from facts that lived deep in the
discarded document. Nothing else changes: same model, same arm, same guard set.

**Grading, three separate channels.** Keeping them apart is the whole point, because they mean
different things:

| channel | measure | what a failure means |
|---|---|---|
| Brief retention | count of 20 planted facts present in the brief, with `absent`/`mutated` split per fact | A full-cache read that lost a deep value. This is the cache-damage channel; the brief stays at 20 identifiers because a 1,500-word cap cannot carry 64, and W2 carries the wider count. |
| Phase-B answers | keys correct given only the brief | If the fact is in the brief and the answer is still wrong, the failure is short-context reasoning, not cache depth. |
| Degeneration | loop detector over both responses, and where it fires | Summarising is a long generation under maximum attention span, so a loop here is the long-horizon symptom the earlier study could not reach. |

The primary endpoint is **brief retention count**, because it is measured on a full-depth read with no
chance of confusion from later context. Phase B is reported alongside it to show the consequence, not
to substitute for it.

**Secondary variant — model-driven compaction.** A `compact` tool with the same 1,500-word cap and a
harness wall at 90% of the window: an agent that never compacts is halted (`halted_by=context_full`),
one that compacts late loses whatever it silently dropped. This measures agentic context management
and is reported as behaviour with intervals, never folded into the primary endpoint, because the
compaction point then varies by arm and is no longer a controlled comparison.

**Cost.** Two requests, ~6k generated tokens per instance: the cheapest workflow with the highest
information per token, and the second one to run.

## D4 — W1: KLD and perplexity, the fast instrument

The workflow tiers give coarse counts and take hours. A distributional measurement answers the same
question in minutes, because it compares the model against itself: the f16 arm's logits are the
baseline, and each quantized arm is measured by how far its next-token distribution has moved from
them. That is the instrument that separated q4_0 in the earlier 32k study when capability did not move
at all (mean KLD 0, 0.00066, 0.0032; same-top agreement 100, 99.1, 97.4 %), and it is the only measure
in this study that has an error bar on every arm by construction.

**How it runs.** `llama-perplexity` from the pinned build, with the arm's `-ctk`/`-ctv` and the same
offload knobs the server used, runs three passes. KLD here is Kullback–Leibler divergence, and PPL is
perplexity.

1. The first pass writes the f16 baseline logits once per model with `--save-all-logits base.kld` over a
   fixed 4,000-token slice of the shared document. The file is tokens × vocabulary × 2 bytes, about
   1.2 GB at this slice length; the tool's README puts the same file at 11 GiB for a 240k-token corpus,
   which is why the slice is short.
2. The second pass runs every arm with `--kl-divergence --kl-divergence-base base.kld` on the same
   slice. It prints mean KLD, the PPL ratio and its log, the mean change in correct-token probability,
   the Pearson correlation of those probabilities, their RMS change and same-top agreement, each with
   propagated uncertainty.
3. The third pass runs perplexity with no KLD file over an 80k-token slice at `-c 102400`. Pass 2 tests
   the distribution and pass 3 tests the cache with 80k tokens in it; the two can disagree.

**What is reported.** Mean KLD with its interval and a test against zero; same-top agreement; the log
PPL ratio; and deep-context PPL per arm as a ratio to that model's own f16 arm. Perplexity is not
comparable between models, only within a model's own arms, and the report says so.

**Why it belongs here.** It is the cheapest thing in the plan — minutes per arm, no tool loop, no
grading ambiguity — and it produces a graded number where the workflow tiers produce counts. If the
cache is doing nothing on a given model, this says so immediately and cheaply, and the expensive
workflows can then be spent on the models where something moved.

## D5 — Greedy sampling, one sample per instance

Every request is sent with `temperature: 0.0` and nothing else: no `top_k`, `top_p`, `min_p` or
`repeat_penalty`, and no sampler flags on the server. llama.cpp treats a temperature of zero as greedy
decoding, so the sampled token is the argmax at every step.

**Why greedy.** Sampling noise is the one cost the study can remove rather than pay for. A stochastic
decode needs several draws per instance to separate signal from noise, and with three draws the run is
three times longer for the same number of instances. Greedy decode is bit-reproducible on a fixed build
with a fixed `-ub`, so one draw per instance is the whole sample and the entire variance in the results
comes from instance difficulty, which is what the instance counts are sized for.

The trade disclosed: greedy decoding is not the setting a user would run in production, and it can be
either more or less prone to looping than a sampled decode. That limits the loop counts to a statement
about greedy decoding, and the report says exactly that. It is the right trade here because the
alternative is a study that cannot afford enough instances to reach significance at all.

**What this removes from the last run.** The forced `temperature 0.7, top_k 20, top_p 0.95` and the
graded tier's `repeat_penalty 1.05`. All five model files declare temp 1.0; Gemma declares `top_k 64`.
The repeat penalty is the sharper error — the code comment next to it already says a repeat penalty is
"exactly the thing that masks looping" — and it was left on for every graded probe. Both of the
settings that produce degeneration were being controlled away from what the model ships with, and then
degeneration was measured.

**Audit.** The server runs with `-v` and its `sampler params:` line is captured per arm as
`sampler_resolved`, so the run records that greedy was actually in force and did not silently fall back
to a model default.

## D6 — No reasoning budget

`--reasoning-budget` is removed from the server, `reasoning_budget_tokens` from the request path, and
`BUDGET_RETRY_MULT` with its `budget_raised` retry path is deleted. Thinking is unbounded and fully
counted.

The reason is the same as the repeat penalty: the budget forces an end-of-thinking tag when it fires,
which is a harness-side loop breaker. A model that would rattle in its own reasoning for 30k tokens is
cut at 8,192 and then answers from a coerced state; the loop is gone from the transcript and the answer
usually looks acceptable. The retry path then re-ran the sample at 4× the budget and logged that it had
done so — a receipt for the artefact, not a fix.

What replaces it, per response:

- `thinking_tokens` and `thinking_share` (thinking as a fraction of all generated tokens);
- the loop detector applied to `reasoning_content` as well as `content`, with `halted_by=loop`
  recording which stream fired and at what token position;
- time-to-first-visible-token;
- `halted_by=context_full` with zero visible tokens is a legitimate outcome and its own counter: a
  model that spends the entire window thinking at depth has demonstrated exactly the failure this study
  is looking for. It is reported, never retried and never folded into a score.

Kept unchanged: the single 64k response ceiling, purely as a runaway guard, recorded as
`halted_by=ceiling` and reported separately.

## D7 — Three models that fit the night

The five-model matrix cannot be run at full context in any affordable time, and two of the five are the
reason. Measured at 102400: the dense 27B prefills at 420 t/s and decodes at 12.2 t/s, and Flash-Next
prefills at 305 t/s and decodes at 17.4 t/s. Both pay twice, once per prompt and again on every
generated token. The other three prefills in 100–122 s and decodes at 36–48 t/s, so they cost roughly a
third as much per instance.

| model | prefill at 92k | decode | role |
|---|---|---|---|
| Gemma-4-26B-A4B | 916 t/s (100 s) | 44.9 t/s | fewest global attention layers (5 of 30), and the only large effect in the last run |
| Ornith-1.5-35B-A3B | 912 t/s (100 s) | 48.0 t/s | fastest, so it carries the expensive workflows |
| Qwen3.6-35B-A3B | 750 t/s (122 s) | 36.4 t/s | second architecture for the deciding probes |

Dropped: the dense 27B and Flash-Next. Both are kept in `config.py` and can be run later on the two
cheap probes; neither is assumed in any claim here.

## The workflow set

Six workflows, all at the full rung. Success is graded from state or from counts, never from prose.

| id | workflow | instances | guard | primary grading |
|---|---|---|---|---|
| W1 | **KLD and perplexity** (D4) | 3 passes/arm | — | mean KLD vs f16, same-top, deep-context PPL ratio |
| W2 | **64-key retrieval** (D1) | 12 | 1 request | keys correct out of 64, in order |
| W3 | **Compaction** (D3) | 12 | 2 requests | brief retention out of 20 facts |
| W4 | **Long-horizon ledger** — 100k operation log on disk, read through tools so the window fills as it reads; 400 mutation records, 20 tenant-scoped mutations through `append`/`retract`/`transfer`, five invariants reported | 6 | 120 tool calls | final ledger hash + invariants; a re-applied mutation changes the hash |
| W5 | **Agentic search** — 60 fact clusters, `search(term)` returns top-3 snippets, three multi-hop questions | 6 | 60 tool calls | answer keys correct, with a search floor and ceiling |
| W6 | **One unbounded long-form response** — 6-section manual, section titles and cross-references given, no token limit | 6 | 64k ceiling only | sections present, cross-references resolved, loop instrument over the whole output |

Instance counts are per arm and set by information per hour, not by symmetry: the two counted probes
get twelve because a shifted mean needs instances to show up, and the long-horizon workflows get six
because each one costs a prefill plus 8–18k generated tokens. W4 and W5 run on the fast arm-sets first
and are the cells that drop when the governor fires; dropping them is logged rather than silent.

The old toy repo repair, the codegen tier and Terminal-Bench are all out. Terminal-Bench 4.0 exists and
would be the right long-horizon benchmark, but at 36–48 t/s decode a single task is one to two hours of
generation and the study cannot afford even a two-task subset on three models. W4 and W5 cover the
tool-loop ground that fits the budget.

Stopping rules: **completed**, **loop halt** (`halted_by=loop`, with token position and the offending
span), **context full** (`halted_by=context_full`), **ceiling**. No request carries `max_tokens` except
the 64k ceiling, and no per-arm budget exists. The asymmetric 4× retry that gave Gemma 8,192 tokens on
12–13 of 18 samples cannot recur because the machinery that produced it is gone.

## Statistics

The unit of analysis is a workflow instance, not a request or a repetition. No reps and no seeds are
needed because decode is greedy: one draw per instance is the whole sample. The instance count per
workflow is in the workflow table, so the smallest attainable p is 2/2¹² = 0.0005 for W2 and W3, and
0.031 for W4, W5 and W6.

W2 and W3 are counts rather than verdicts. The report gives the mean score out of 64 for W2 and out of
20 for W3, each with its standard
deviation, the distribution across instances, and the paired difference against that model's f16 arm
with a cluster bootstrap over instances (1,000 resamples, seeded). A one-key shift is visible in the
mean and in the per-depth curve even when no instance is dramatically wrong.

Depth is reported as a curve, not a total: per-depth hit rate for W2 and retention by depth band for
W3, with Wilson intervals. A loss at 90% depth is a different finding from a loss at 40%, and the
report says which it was. Behaviour rates (loop halts, context-full halts, no-progress windows,
thinking share, tokens and wall clock per instance) are reported with intervals and are never the
basis of a ranking on their own.

The decision rule is pre-registered: an arm is worse if the lower bound of the paired interval excludes
0 *and* the point estimate is worse. Ties otherwise. The power statement in the report header gives
the instance counts, the number of comparisons, and the smallest attainable p.

## Harness changes

Nine changes separate this harness from the earlier one; METHOD.md carries the detail.

1. Tool runner: step budget, transcript with state hashes, and per-step loop detection.
2. Compaction engine: fixed-boundary rewrite, verbatim brief, and recorded token positions.
3. KLD/PPL runner: f16 logits baseline once per model, arm metrics parsed into the arm record.
4. Greedy sampling only: no seeds, no reps, and `sampler_resolved` captured per arm.
5. 64-key builder: per-key grading, integer score out of 64, and a depth matrix.
6. State graders: ledger hash and invariants, brief retention, search counts, long-form structure.
7. Extraction hardening: `ANSWER:` from reasoning, and wrapped or newline-separated answer lists.
8. Manifest: rung, model hash, declared sampling, resolved sampler, baseline hashes, hardware.
9. Retirement of the codegen and repo tiers, tier budgets and reasoning-budget machinery.

## Budget: 12 h target, 24 h hard cap

Two terms, and the document prefill is paid once per arm, not once per instance: every instance in an
arm shares the same 92k document prefix, so the server reuses it and only the task text is prefilled
per request.

```
hours per arm ≈ document_prefill + Σ over instances ( task_text_prefill + generated_tokens / decode_tps )
```

Document prefill is 100–122 s for the three models in play (D7). Task-text prefill averages about 45 s
per instance, measured as the incremental rate the last run reported in its `prefill t/s` column. A
300-token answer at 45 t/s is 7 s, so the cheap probes cost almost exactly their pipeline overhead and
the long workflows cost their generation.

| workflow | instances | out tokens | Gemma | Ornith | Qwen3.6-35B |
|---|---|---|---|---|---|
| W1 KLD + PPL | 3 passes | ~4k + 80k scored | 5 m | 5 m | 6 m |
| W2 64-key | 12 | ~600 | 10 m | 10 m | 11 m |
| W3 compaction | 12 | ~6k | 36 m | 35 m | 42 m |
| W4 ledger | 6 | ~18k | 45 m | 44 m | 54 m |
| W5 search | 6 | ~8k | 22 m | 21 m | 27 m |
| W6 long-form | 6 | ~12k | 31 m | 29 m | 38 m |
| **per arm** | | | **2.6 h** | **2.5 h** | **3.0 h** |
| **both arms** | | | **5.2 h** | **5.0 h** | **6.0 h** |

Greedy accounting is already in those numbers: the last study spent three seeds per instance and this
one spends one, which is where a third of the saving comes from; dropping the two slow models and
Terminal-Bench is the rest.

### Tranches, fixed in advance

| tranche | cells | hours | running total | decision it buys |
|---|---|---|---|---|
| T1 | Gemma, f16 + q4_0, all six workflows | 5.2 | 5.2 | The whole result on the model with the only prior large effect, including the instrument tier. |
| T2 | Ornith, f16 + q4_0, all six workflows | 5.0 | 10.2 | The fastest model, so the long-horizon workflows actually finish: a clean second architecture. |
| T3 | Qwen3.6-35B, f16 + q4_0, W1 + W2 + W3 | 2.0 | 12.2 | The counted probes and the instrument on a third architecture. |
| T4 | Qwen3.6-35B, f16 + q4_0, W4 + W5 + W6 | 4.0 | 16.2 | The long-horizon workflows on the third model. Only if the governor has slack. |

The order is fixed in advance, so a deadline cut lands on the least informative remaining cell and never
on the deciding one. Anything cut is reported as cut, and the report header lists which cells ran with
every per-workflow n. Nothing here depends on a generation cap: a looping instance is cut by the
detector within a few thousand tokens, so failures cost less than successes.

Governor, resumability and per-arm checkpointing are reused unchanged.

## Stop rules

These are the conditions under which the study pauses instead of spending the night on a probe that
cannot answer. Each names what was observed and what to do about it; none relaxes a rule in favour of
getting a number.

- If f16 loses a key on any W2 instance, the task is mis-specified; fix the document or the prompt and
  do not proceed to the other arms.
- If the W2 score saturates at 64 for every arm including q4_0, the probe is at its ceiling: the report
  flags saturation as a finding and the task is not widened again, because a longer answer line would
  measure the answer format rather than cache damage.
- If `context_full` fires on more than half of an arm's runs, that is the finding; report it and do not
  move the rung down to make the arm finish.
- If W3 retention is at ceiling for f16, the compaction wall is too shallow; deepen the facts before
  starting T2.
- If T1 overruns 6 h, stop the tranche and report the cells that finished; T2 and later tranches are not
  started on the same night.
- If mean KLD comes back as zero with a tight interval on every arm, the instrument is not reaching the
  cache; check that `-ctk`/`-ctv` were passed and that the baseline slice matches token for token
  before running the workflows.

## Sequencing

1. No GPU: retire the old tiers, build the 64-key scorer and the compaction engine, wire the
   KLD/perplexity runner, delete the budget and sampler plumbing, switch every request to greedy.
2. Micro-pilot, ≤ 30 min, Gemma f16 only: one 92k prefill, one 64-key sample, one compaction pair, one
   KLD pass. Confirms the halt detector fires on a known loop, the score and the retention count
   discriminate, and calibrates prefill rate on this host. Not a scoring run.
3. T1, then T2, then T3, then T4 as the governor allows. Each arm-set checkpoints per instance, so a
   cut night resumes instead of restarting.
4. Report: one page (answer, W1 KLD and PPL by arm, W2 score out of 64 with the depth curve and the
   miss list, W3 retention, per-workflow results with intervals, halts with intervals, cost per
   instance, which cells ran) plus an appendix.

## What the current run still tells us

- `q4_0` did not obviously make these models loop more — a statement limited to ~1k visible tokens,
  with thinking capped, a repeat penalty on, and three of five arms sampled at a temperature they do not
  ship with. It is not a null result; it is a measurement of the harness.
- The one large effect (Gemma at q4_0, code 41 vs 89) is a generation-length/format failure: 3 of 6
  tasks hit the cap, and `repo_fix` emitted 20,459 tokens with no file block. Not evidence of cache
  depth damage, and Gemma has the fewest global attention layers (5 of 30), which is the opposite of
  what a depth effect predicts.
- Every paired comparison is task-clustered with a Holm correction; after that no arm separates
  (adjusted p = 1.00 for all 42 comparisons). The earlier "three separable arms" was pseudo-replication.
- The multikey probe was answered 9/9 by all arms, which is the clearest evidence that the tests, not
  the cache, were the limiting factor.
- Decode was 17–28 % slower with a quantized cache than at f16 on every model, so the setting buys
  context, not speed. W1's PPL ratio is the cheap way to confirm that this build still behaves that way.
