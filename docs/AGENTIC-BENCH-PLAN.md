# Next study: agentic workflows at full context

Design document only — nothing here was run. It replaces the current study, which measured short
answers at a shrinking context with sampler settings it imposed on the models.

## What the current probes cannot see

The table lists the harness limits that made the last study unable to answer its question. The left
column names the limit, the middle column gives the value that was in force, and the right column says
what that value did to the measurement — in most rows the harness, not the model, decided when a probe
ended.

| limit | number | consequence |
|---|---|---|
| Generation cap | 2,048 tokens (1,024 of them thinking) | ~1,000 visible tokens per probe. The harness, not the model, decides when a probe ends, so "finished" and "cut off" reach the grader looking the same. |
| Task size | `copy_ledgers` = 12 rows, codegen fixtures = one short script | Both finish or truncate before any long-horizon failure mode appears; `enumerate_shards` hit the cap in most arms, so its "loop" flag is partly a truncation artefact. |
| Multi-key breadth | 3 keys | 9/9 in every graded arm: the test cannot fail, so it cannot discriminate. |
| Interaction | 5 fixed tool calls, one round | Cannot show re-reading the same file, A↔B edit oscillation, re-running a failing test unchanged, or declaring done unverified. |
| Context depth | samples ran on a ladder down to 49k | KV damage scales with the number of cached tokens; a 49k rung is half the dose and every model gets a different test. |
| Sampling | harness-forced `temperature 0.7, top_k 20, top_p 0.95, repeat_penalty 1.05` | Three of five arms were sampled outside their own declared distribution, and the repeat penalty sits directly on top of the loop detector. |
| Thinking | `--reasoning-budget 8192` plus an end-of-thinking tag forced at the cap | The harness breaks reasoning loops for the model. A model that loops in thinking answers normally, and the loop never reaches the output. |
| Statistics | 12 samples/arm = 4 probes × 3 reps | Smallest attainable p = 0.125 before correction; the earlier "three separable arms" was pseudo-replication, counting three reps of one task as three independent observations. |

## Decisions

| # | decision | replaces |
|---|---|---|
| D1 | 20-key all-or-nothing retrieval gate | 3-key partial-credit task |
| D2 | One full-context rung per model, no ladder | Ladder 49k–100k |
| D3 | New compaction workflow (W2) | — |
| D4 | Terminal-Bench 2.0 subset via Harbor replaces the codegen/repo probes | 5 codegen fixtures, `repo_fix` |
| D5 | No sampler overrides; model default + one greedy arm | Forced 0.7 / 1.05 penalty |
| D6 | No reasoning budget, no retry path | `--reasoning-budget`, `budget_raised` retries |
| D7 | 12 h target, 24 h hard cap, tranches ordered by information per hour | Flat 3-model × 58-run matrix |

Facts checked before writing this, so the plan rests on the build actually in use. GGUF is the model
file format llama.cpp loads, and the KV cache is the stored attention state for every token already
read.

- Every model file declares its own sampler block: `general.sampling.temp = 1.0` on all five,
  `top_p = 0.95`, `top_k = 20` (Gemma: `top_k = 64`). The run passed `temperature 0.7, top_k 20` on
  the loop tier anyway, so Gemma was sampled at 0.7 and top_k 20 against its declared 1.0 and 64.
- llama.cpp applies `general.sampling.*` as defaults for any option the command line does not set
  (`common/common.cpp` 1127–1145, gated on `common_params_sampling_config`). Omitting `--temp`
  therefore yields the model's declared temperature, not llama.cpp's 0.80. "Model default" is
  obtainable exactly, and it is not the same number for every arm.
- The server prints `sampler chain:` and `sampler params:` per request at trace level
  (`tools/server/server-context.cpp` 1947–1948), so the resolved config is auditable with `-v`.
- All five models load at `102400` with a ≥ 91k prompt (fit.json): VRAM peak 11.2–15.9 GiB, prefill
  305–916 t/s, decode 12.2–44.9 t/s. Full context fits; nothing forces a ladder.
- Harbor is the official Terminal-Bench 2.0 harness, runs tasks in Docker, takes a model endpoint on
  `--model`, and registers a custom agent through the agent-import-path interface. Docker and a
  16 GiB GPU are on this host. Terminal-Bench is an agent benchmark built from real terminal tasks,
  which is what this study needs for long-horizon behaviour.

## D1 — W1: 20-key retrieval is a gate, not a score

A lossy KV cache does one thing well and early: it drops or mutates specific values held at depth.
That is binary damage, and the test for it must be binary.
**The task.** Twenty shard codes, in the answer order asked. The twenty are every second shard of the
forty planted, so their depths run from 2% of the document to 98% of it, and a miss points at a
position. Each shard has a near-miss decoy, either a code differing in one or two digits or a codeless
shard with a similar name, so surface matching does not pass.

**The rule.** A sample passes only if all twenty codes match position for position. One missing key,
one substituted key, one transposed pair — the sample fails, and the arm is damaged. There is no
partial-credit score in the ranking; nothing is averaged away.

**What is reported anyway, as diagnostics and never as score:**

- the per-key hit matrix per arm (which of the twenty survived, at what depth);
- the failure mode split: `absent`, `mutated` (near-miss decoy or digit-altered code), `misordered`;
  `mutated` is the worse of the three and is called out separately, because a confidently wrong value
  is cache damage and an empty slot is at least honest;
- failure depth histogram, so a loss confined to the last 20% of the document reads differently from
  a loss at mid-depth.

**Why a gate is the right instrument here.** f16 is expected at 100% — if it is not, the task is
mis-specified and the run stops. Against that baseline, a single missed key is a defect rather than a
noise sample, so one instance already carries information and twelve make it statistical:
0/12 vs 12/12 gives p = 2/2¹² = 0.0005 under the instance-clustered sign-flip test, unchanged from
the previous plan's arithmetic but now driven by a test that can actually fail. Partial credit would
have let an arm that loses one deep key in every sample average out to 0.95 and look healthy.

**Breadth.** Twenty keys, not three: three keys fit in a few hundred tokens of the cache and were
answered 9/9 by every arm including q4_0. If twenty also saturates at f16 *and* every arm — that is,
no arm ever drops a key — the task widens to all forty shards before any further models run; a gate
that nobody fails is a format test.

**Cost.** One request per sample. This is the cheapest workflow in the study and the one that decides
the result, so it runs first.

## D2 — One rung: the whole window, for every arm

Each model runs at the largest ladder entry it loads with a ≥ 91k-token prompt, and every arm of that
model runs at that same rung. From fit.json that is `102400` for all five models, so no model is
measured at a different depth and no cross-model row is compared at unequal context.

What this fixes: on a ladder, an arm that fails at 100k silently gets credited with the 49k rung where
it succeeds, and the ladder's per-arm rung differences were an uncontrolled variable inside every
comparison. Damage from a lossy cache grows with the number of tokens in it, so half the window is
half the dose — measuring at the bottom of the ladder systematically under-reports damage.

Operationally:

- the shared document is retargeted so that the document alone is 92–95k tokens in the *largest* of
  the five tokenizers, and every workflow's prompt reads it to the end;
- generation happens on top of that, so most workflows operate within a few thousand tokens of the
  window edge by construction. That is intended: exceeding the window is a recorded outcome
  (`halted_by=context_full`), and D3 exists to measure whether the model survives being forced to
  make room;
- `fit.json` still records which rungs load, because a model that needs a lower rung to start must be
  reported, not quietly moved down and compared anyway.

## D3 — W2: compaction

The only operation that must read the entire cache in one pass is summarising it. If quantized KV
loses values at depth, a compaction step is where the loss becomes visible and permanent, because the
summary is written from the damaged read and then becomes the only context that survives.

**Phase A (forced, deterministic).** The model gets the full 100k document and one instruction:
produce a handover brief of at most 1,500 words for a worker who will not see the document. The brief
must carry the twenty ident, the ledger rows the twenty mutations touch, and the five invariants.

**Phase B (compacted context).** The harness discards the document *and* the phase-A transcript and
sends only the brief verbatim, then asks questions answerable solely from facts that lived deep in the
discarded document. Nothing else changes: same model, same arm, same guard set.

**Grading, three separate channels.** Keeping them apart is the whole point, because they mean
different things:

| channel | measure | what a failure means |
|---|---|---|
| Brief retention | for each of 20 planted facts: `present` / `absent` / `mutated` | A full-cache read that lost a deep value. This is the KV-damage channel. |
| Phase-B answers | correctness given only the brief | If the fact is in the brief and the answer is still wrong, the failure is short-context reasoning, not cache depth. |
| Degeneration | loop detector over both responses, halt position | Summarising is a long generation under maximum attention span; loops here are the long-horizon symptom the current study could not reach. |

The primary endpoint is **brief retention**, because it is measured on a full-depth read with no
chance of confusion from later context. Phase B is reported alongside it to show the consequence, not
to substitute for it.

**Secondary variant — model-driven compaction.** A `compact` tool with the same 1,500-word cap and a
harness wall at 90% of the window: an agent that never compacts is halted (`halted_by=context_full`),
one that compacts late loses whatever it silently dropped. This measures agentic context management
and is reported as behaviour with intervals, never folded into the primary endpoint, because the
compaction point then varies by arm and is no longer a controlled comparison.

**Cost.** Two requests, ~6k generated tokens per instance: the cheapest workflow with the highest
information per token, and the second one to run.

## D4 — W3: Terminal-Bench 2.0 subset, because short scripts cannot show long-horizon damage

The codegen tier asked for a 40-line module; `repo_fix` asked for a five-line edit in a 60-file toy.
Neither produces the failure modes the study is about. Every workflow that remains in this plan is
either a depth probe (D1, W4, W5) or a degeneration probe (W6, W2). For real agentic behaviour the
study borrows an existing, verified benchmark instead of inventing one.

**Setup.** Terminal-Bench 2.0 through Harbor, with a custom `BaseAgent` adapter (~200 lines) that
drives llama-server's OpenAI-compatible endpoint and executes commands through Harbor's container.
The adapter is the same for every arm and every model; its prompt, tool schema and internal context
management are frozen and committed by hash in the manifest.

**Task selection, fixed before the first run.** Two of the tasks, chosen by three filters:

1. image builds from a pinned digest with no network at agent time;
2. oracle solution under 5 minutes, and reference traffic ≤ 120k tokens, so one run fits the window
   and finishes in 15–55 minutes at Gemma's decode rate;
3. verification is state-based (pytest over files), not prose.

Two, not six, because Terminal-Bench is the most expensive workflow per hour of information and the
only one whose cost is set by a model's decode rate rather than by instance count. Both ids and their
image digests go in the manifest before any arm runs, so the subset cannot be chosen after seeing
results.

**Only the fast arm-set runs it.** Gemma × f16/q4_0 × 2 tasks = 1.1–3.8 h. A second pair of tasks or a
third task is T4, and the dense 27B never runs W3: at 12 t/s the same two tasks are 3.9–13.9 h, more
than every other cell in its tranche combined.

**What it measures.** Solve rate per arm, steps to solve, tokens to solve, context-full halts and
compaction events inside the adapter's own loop. This is also the only workflow where a 100k window is
under real pressure for hours at a time, so `context_full` and adapter compaction events are
first-class results rather than cleanup.

**The fairness rule that matters most.** A model that cannot emit a valid tool call in this format has
an adapter problem, not a KV problem. Before scoring, the oracle agent runs as an infrastructure check,
then the adapter is exercised against the two tasks at f16. If it cannot get valid tool calls on two
consecutive tasks, the W3 cell is reported as `adapter_incompatible` and excluded from every
comparison. The current run's Gemma result — 20,459 tokens with no file block, scored as a code
failure — is exactly the artefact this rule exists to prevent.

**Retired.** All five codegen fixtures and the repo fixture leave the graded set with their budgets,
graders and the four fixture repairs planned for them. They stay in the tree as fixtures; they are no
longer measurements.

## D5 — Sampling is the model's, not the harness's

- No `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, and no sampling fields in any
  request payload. The effective configuration is the model file's own declaration, or llama.cpp's
  default where a model declares nothing. Gemma gets 1.0/64/0.95, the Qwen family gets 1.0/20/0.95 —
  each on its own terms.
- One additional profile, applied to D1 and W2 where determinism helps the depth curve: **greedy**
  (`temperature 0.0` and nothing else). Reported as its own row; never mixed with the model-default
  rows.
- **Sampler audit.** The server runs with `-v`; the first request's `sampler params:` line is captured
  into the arm record as `sampler_resolved`. The parity guard asserts two things: that no request in
  the arm carried a sampling key, and that the resolved params differ between models and never between
  arms of the same model.
- Reasoning effort is treated as the same class of override and dropped as well; the template default
  applies, and the resolved setting is recorded per model.

Why this is a correctness fix, not a preference: the current run forced `temperature 0.7, top_k 20,
top_p 0.95` on the loop tier for every model, and `repeat_penalty 1.05` on the graded tier. All five
files declare temp 1.0; Gemma declares top_k 64. The repeat penalty is the sharper error — the code
comment next to it already says a repeat penalty is "exactly the thing that masks looping", and it was
left on for every graded probe. Two of the three quantities that produce degeneration were being
controlled away from the value the model ships with, and then degeneration was measured.

## D6 — No reasoning budget

`--reasoning-budget` is removed from the server, `reasoning_budget_tokens` from the request path, and
`BUDGET_RETRY_MULT` with its `budget_raised` retry path is deleted. Thinking is unbounded and fully
counted.

The reason is the same as the repeat penalty: the budget forces an end-of-thinking tag when it fires,
which is a harness-side loop breaker. A model that would rattle in its own reasoning for 30k tokens is
cut at 8,192 and then answers from a coerced state; the loop is gone from the transcript and the answer
usually looks acceptable. The retry path then re-ran the sample at 4× the budget and logged the fact
that it had done so — a receipt for the artefact, not a fix.

What replaces it, per response:

- `thinking_tokens` and `thinking_share` (thinking as a fraction of all generated tokens);
- the loop detector applied to `reasoning_content` as well as `content`, with `halted_by=loop`
  recording which stream fired and at what token position;
- time-to-first-visible-token;
- `halted_by=context_full` with zero visible tokens is a legitimate outcome and its own counter: a
  model that spends the entire window thinking at depth has demonstrated exactly the failure this
  study is looking for. It is reported, never retried and never folded into a score.

Kept unchanged: the single 64k response ceiling, purely as a runaway guard, recorded as
`halted_by=ceiling` and reported separately.

## The workflow set

Six workflows; every one needs most of the window, and success is graded from state, never from prose.

| id | workflow | instances | guard | primary grading |
|---|---|---|---|---|
| W1 | **20-key retrieval gate** (D1) | 12 | 1 request | all 20 codes, exact, in order |
| W2 | **Compaction** (D3) | 12 | 2 requests | brief retention over 20 depth-spread facts |
| W3 | **Terminal-Bench subset** (D4) | 2 tasks | task timeout | benchmark verifier |
| W4 | **Long-horizon ledger** — 100k operation log on disk (read through tools, so the window fills as it reads), 400 mutation records, 20 tenant-scoped mutations through `append`/`retract`/`transfer`, five invariants reported | 8 | 120 tool calls | final ledger hash + invariants; a re-applied mutation changes the hash |
| W5 | **Agentic search** — 60 fact clusters, `search(term)` returns top-3 snippets, three multi-hop questions | 8 | 60 tool calls | answer correctness with a search floor and ceiling |
| W6 | **One unbounded long-form response** — 6-section manual, section titles and cross-references given, no token limit | 6 | 64k ceiling only | sections present, cross-references resolved, loop instrument over the whole output |

Instance counts are per arm and set by information per hour, not by symmetry: the gate gets twelve
because one missed key decides an arm, the long-form probe gets six because it costs a full prefill
plus ~12k generated tokens to say what it has to say. W5 and W4 run on the fast arm-sets only; they are
the cells that drop first when the governor fires, and dropping them is logged rather than silent.

The previous plan's toy repo repair and staged bug fix are deleted; W3 covers that ground with real
toolchains. W6 (long-form) stays because it is the only probe that isolates a single unbounded
generation from tool interaction.

Stopping rules are unchanged from the previous plan and now apply without exception: **completed**,
**loop halt** (`halted_by=loop`, with token position and the offending span), **context full**
(`halted_by=context_full`), **ceiling**. No request carries `max_tokens` except the 64k ceiling, no
per-arm budget exists, and the asymmetric 4× retry that gave Gemma 8,192 tokens on 12–13 of 18 samples
cannot recur because the machinery that produced it is gone.

## Statistics

- **Unit = workflow instance**, n per workflow as in the workflow table: 12 (W1), 12 (W2), 8 (W4), 8
  (W5), 6 (W6), 2 tasks (W3). Three seeds averaged *inside* an instance, never pooled as independent
  pairs. Smallest attainable p per workflow: 2/2¹² = 0.0005 at n = 12, 0.0078 at n = 8, 0.031 at n = 6.
  W3's two tasks carry no significance claim at all; it is descriptive evidence about long-horizon
  behaviour and is reported as such.
- **D1 is a gate**: per-instance pass/fail only. Instance-clustered sign-flip test against that model's
  f16 arm, plus the verbatim list of missed keys with their depths. The 20-item hit matrix is reported
  as a diagnostic table and is used for nothing else.
- **Curves, not scores, for depth.** D1 failure depth histogram and W2 retention by depth band, with
  Wilson intervals clustered by instance. A loss at 90% depth is a different finding from a loss at
  40%, and the report says which.
- **Rates with intervals for behaviour**: loop halts, context-full halts, no-progress windows, thinking
  share, tokens and wall clock per success. Secondary, reported with intervals, never the basis of a
  ranking on its own.
- **Pre-registered decision rule**: an arm is worse if the lower bound of the paired success-rate
  interval excludes 0 *and* the point estimate is worse. Ties otherwise. Power statement in the report
  header: instances, comparisons, smallest attainable p.

## Harness changes

1. **Multi-turn tool runner** with step budget, transcript recorder (tool, args, ok, state diff), and
   per-step loop detection: identical consecutive calls, identical calls ≥3, no-progress windows,
   A↔B oscillation.
2. **Compaction engine**: prompt rewrite at a fixed boundary, brief re-inserted verbatim, document
   removed, compaction event recorded with token positions; plus the model-driven `compact` tool and
   the 90% wall.
3. **Harbor adapter** (D4) with pinned image digests, both task ids in the manifest, and the
   oracle/incompatibility pilot.
4. **Sampler audit and parity guard**: no sampling keys in any payload, `sampler_resolved` captured
   from the server's trace line, reasoning budget machinery deleted, 64k ceiling the only cap.
5. **20-key task builder**: every second shard, per-key grading with the `absent`/`mutated`/
   `misordered` split, exact-order gate.
6. **State graders**: repository tree hash + hidden suite (W3 via Harbor), ledger hash + invariants,
   brief retention, search counts, "did the tests run after the last edit" parsed from the tool
   transcript.
7. **`extract_*` hardening**: reject `../` paths, parse `ANSWER:` from reasoning as well as content
   (the docstring promises it, the code does not), and treat a 20-value answer line as valid when
   comma- or newline-separated.
8. **Manifest**: single rung per model, model-file sha256, the GGUF `general.sampling.*` block as
   declared, `sampler_resolved` per arm, Harbor task digests, adapter commit hash, hardware captured
   at matrix start.
9. **Retirements**: codegen tier, repo tier, tier budgets, `REASONING_BUDGET_TOKENS`,
   `BUDGET_RETRY_MULT`, `SAMPLING_LOOP`.

## Budget: 12 h target, 24 h hard cap (D7)

Cost has two terms and the prefill is the larger one. Per arm-set:

```
hours = Σ over instances ( 92,000 / prefill_tps  +  generated_tokens / decode_tps )
```

A 92k prompt is 100 s at Gemma's 916 t/s prefill and 300 s at Flash's 305 t/s, and a 300-token answer
at 45 t/s is 7 s. The gate therefore costs almost exactly one prefill per instance, and no instance
count is free: each arm pays its own prefill because the KV cache is rebuilt per arm, which is the
variable under test. Instance counts are the only lever, so they are set per workflow rather than made
uniform.

Measured rates from fit.json, all five models loaded at `102400`:

| model | prefill / decode | W1 + W2 (24 inst) | cheap set, 1 arm | cheap set, 2 arms | W3 (2 tasks) |
|---|---|---|---|---|---|
| Gemma-4-26B-A4B | 916 / 44.9 | 1.1 h | 3.5 h | 7.0 h | 0.6–1.9 h |
| Qwen3.6-35B-A3B | 750 / 36.4 | 1.4 h | 4.3 h | 8.6 h | 0.7–2.4 h |
| dense 27B | 420 / 12.2 | 3.2 h | 10.9 h | 21.8 h | 1.9–6.9 h |
| Flash-Next | 305 / 17.4 | 3.2 h | 9.5 h | 19.1 h | 1.4–5.0 h |

cheap set = W1 + W2 + W4 + W5 + W6. W3 = Terminal-Bench, 40–150k generated tokens per task.

This is why a flat matrix lands at 70–120 h and why the one below is asymmetric: the dense 27B
at 12 t/s cannot run the full set inside any sensible budget, and Flash-Next pays twice — slow prefill
*and* slow decode. Cheap models carry the expensive workflows; slow models carry only the two probes
that decide the question.

### Tranches, fixed in advance

| tranche | cells | hours | decision it buys |
|---|---|---|---|
| T1 | Gemma, f16 + q4_0: W1, W2, W4, W5, W6 + W3 (2 tasks) | **8.1–10.8** | The whole result on the model with the fewest global attention layers (5 of 30) and the only prior large effect. Gate + compaction + long-horizon + TB, both arms. |
| T2a | Qwen3.6-35B-A3B, f16 + q4_0: W1 gate only | **0.9** | The deciding probe on a second architecture. Running total 9.0–11.7 h — inside the 12 h target even if T1 runs long. |
| T2b | Qwen3.6-35B-A3B, f16 + q4_0: W2 compaction | **1.9** | Compaction retention on that second architecture. Running total 10.9–13.6 h. |
| T3 | dense 27B, f16 + q4_0: W1, W2 at 8 instances | **4.2** | Worst-case decoder at the full window; the slowest arm cannot hide behind a big budget. Running total 15.1–17.8 h. |
| T4 | only if the governor has slack | — | W5 and W4 on Qwen3.6, second W3 task pair, a third TB task; not assumed in any claim. |

Fixed, pre-registered order, so a deadline cut always lands on the least informative remaining cell and
never on the deciding one. Anything cut is reported as cut; the report header lists which cells ran and
every per-workflow n. Three rules keep the night inside the target: the gate and the compaction step are
split from the expensive probes so they can reach a second model cheaply, no workflow runs on more than
one model unless it costs under ~1.5 h per arm-set, and W3 is 2 tasks rather than 6.

Nothing in the budget depends on a generation cap: a looping instance is cut by the detector within a
few thousand tokens, so failures cost less than successes — the asymmetry that makes this plan affordable
is the same one that made the old run's budgets unfair.

Governor, resumability and per-arm checkpointing are reused unchanged.

## Stop rules

These are the conditions under which the study pauses instead of spending the night on a probe that
cannot answer. Each one names what was observed and what to do about it; none of them relaxes a rule in
favour of getting a number.

- If f16 fails the 20-key gate on any instance, the task is mis-specified; fix the document or the
  prompt, and do not proceed to the other arms.
- If 20 keys saturate for every arm including q4_0, widen to all 40 shards before T2a, because a gate
  nobody fails measures format compliance rather than cache damage.
- If `context_full` fires on more than half of an arm's runs, that is the finding; report it and do not
  move the rung down to make the arm finish.
- If W2 retention is at ceiling for f16, the compaction wall is too shallow; deepen the facts before
  starting T2b.
- If the W3 adapter cannot produce valid tool calls on two consecutive tasks, report that cell as
  `adapter_incompatible` and exclude it rather than scoring it.
- If T1 overruns its 10.8 h upper bound, stop the tranche and report the cells that finished; T2b and
  T3 are not started on the same night, and T2a runs only if two hours remain.

## Sequencing

1. No GPU: retire the old tiers, build the 20-key gate and the compaction engine, delete the budget
   and sampler plumbing, write the Harbor adapter, record the two task digests.
2. Micro-pilot, ≤ 30 min, Gemma f16 only: one prefill at 92k, one 20-key sample, one compaction pair,
   one Harbor task. Confirms the halt detector fires on a known loop, the gate and the retention
   grader discriminate, and calibrates prefill rate on this host. Not a scoring run.
3. T1, then T2a, T2b, T3 as the governor allows. Each arm-set checkpoints per instance, so a cut night
   resumes instead of restarting.
4. Report: one page (answer, W1 gate result with missed keys by depth, W2 retention curve, per-workflow
   success with intervals, halts with intervals, cost per success, which cells ran) plus an appendix.

## What the current run still tells us

- `q4_0` did not obviously make these models loop more — a statement limited to ~1k visible tokens,
  with thinking capped, a repeat penalty on, and three of five arms sampled at a temperature they do
  not ship with. It is not a null result; it is a measurement of the harness.
- The one large effect (Gemma at q4_0, code 41 vs 89) is a generation-length/format failure: 3 of 6
  tasks hit the cap, and `repo_fix` emitted 20,459 tokens with no file block. Not evidence of KV-depth
  damage, and Gemma has the fewest global attention layers (5 of 30), which is the opposite of what a
  depth effect predicts.
- Every paired comparison is task-clustered with a Holm correction; after that no arm separates
  (adjusted p = 1.00 for all 42 comparisons). The earlier "three separable arms" was pseudo-replication.
- The multikey probe was answered 9/9 by all arms, which is the clearest evidence that the tests, not
  the cache, were the limiting factor.
