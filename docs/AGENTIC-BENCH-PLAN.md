# Next study: agentic workflows at 100k context

Design document only — nothing here was run. It replaces the loop/degeneration part of the current
study, which could not measure what it claimed to measure.

## Why the current probes cannot see anything

| limit | number | consequence |
|---|---|---|
| L-tier generation cap | 2,048 tokens (1,024 of them thinking) | ~1,000 visible tokens per probe: 12 short rows, 40 lines, or two paragraphs. The harness, not the model, decides when a probe ends, so "finished" and "cut off" reach the grader looking the same. |
| Task size | `copy_ledgers` = 12 rows, `enumerate_shards` = 40 lines | Both finish or truncate before any long-horizon failure mode appears; `enumerate_shards` hit the cap in most arms, so its "loop" flag is partly a truncation artefact. |
| Interaction | 5 fixed tool calls, one round | Cannot show the agentic failure modes: re-reading the same file, alternating between two edits, re-running a failing test without changing anything, declaring done without verifying. |
| Context use | fixed 91k document, tasks answerable from a few hundred tokens | Nothing forces the workflow to use the depth it pays for; a model that never reads past the first page scores the same. |
| Statistics | 12 samples/arm = 4 probes × 3 reps | Hypotheses are compared on 4 clusters. Smallest attainable p = 2/2⁴ = 0.125 before correction, so no loop effect could ever be established. |
| Tiers R/C | saturated or floored | `niah_multi` 9/9 everywhere, `agg_sum` 0/6 in 11 of 15 arms, `count_proj` degenerate (every project has the same count). The graded tiers are not measuring capability at depth, they are measuring format compliance. |

A loop that needs 6,000 tokens to appear is not detectable with a 2,000-token window, and a
4-cluster design cannot put a confidence interval on it even if it were. The next study has to fix
both, and measure behaviour rather than text.

## Stopping rules: no generation cap, halt only on completion or on a loop

A workflow runs until it finishes. No request carries a `max_tokens` (the value is `-1`, unlimited):
the model decides when it is done, and a long answer is data rather than a truncation. The only things
that end a run are:

1. **Completed** — the workflow's terminal action (submit the report, finish the tool sequence) and a
   graded end state. This is the outcome being measured.
2. **Loop halt** — the detector fires: a repeated 200-char block or 8-gram run inside one response, a
   repeated line run, the same tool call twice in a row or ≥3 times in total, or k = 4 consecutive
   steps that change no graded state. The halt records `halted_by=loop`, the token position and the
   offending span. The halt *is* the measurement: "looped after 3,100 tokens" is a result, not a
   cleanup.
3. **Context full** — the transcript reaches the model's 100k window. Deliberately not prevented:
   managing its own context (summarising, re-reading selectively) is part of the work, and failing at
   it is a recorded outcome (`halted_by=context_full`). Unbounded generation makes that the realistic
   risk it is in production.
4. **Identical safety guards** — tool-call budget per instance, wall clock per instance, and a single
   response ceiling of 64k tokens that exists only to catch a runaway generator. If it fires it is
   `halted_by=ceiling`, reported separately, never folded into a score.

Because there is no per-arm budget, the asymmetry that invalidated the current run cannot recur: its
4× retry gave Gemma 8,192 tokens on 12–13 of 18 samples while every other arm got 2,048.

## What to measure

Primary endpoint, pre-registered, deterministic: **task success** (the workflow reaches a verifiable
end state) and **tool-call efficiency** (calls to completion). Secondary: **tokens-to-first-repeat**
inside one generation, **no-progress windows** (≥k consecutive calls that change no graded state),
and **degeneration onset** (first sample position where a 200-char block or 8-gram repeats).

Everything is graded from state, not prose: a final repository tree, a database hash, a set of
files, a JSON state. No judge model, no regex on free text except where the task is literally
"emit this string".

Three counters are reported per arm alongside success: **completed**, **loop-halted** (with the token
position) and **context-full**. Finishing at 9,000 tokens is not better than finishing at 40,000; the
second is only evidence about behaviour at depth.

## The workflows (five, each requires using most of a 100k context)

**W1 — repo repair over a long transcript.** A package of 60 files (~95k tokens) laid out across the
whole document; three seeded defects in three modules, and the fix requires reading files at 5%, 40%
and 90% depth. Tools: `read_file`, `grep`, `write_file`, `run_tests`. Graded state: the tree hash
plus 120 hidden cases; test files are hashed and rejected if edited. What it detects: edit-loop
oscillation, re-reading without progress, "done" without running the tests. Guard: 120 tool calls.

**W2 — long-horizon ledger maintenance.** A 100k operations log with 400 mutation records; the agent
applies 20 tenant-scoped mutations through `append`/`retract`/`transfer`, then reports five
invariants. The records each mutation references sit far apart in the log, so the state it must hold
comes from clauses read long before it acts.
Graded state: final ledger hash + invariant answers; a re-applied mutation changes the hash. This is
the direct KV-depth task: the state it must keep is defined by clauses read long before the action.
Guard: 120 tool calls.

**W3 — agentic search.** 60 planted fact clusters spread through the 100k document; `search(term)`
returns top-3 snippets without ranking help. Three multi-hop questions. Graded: answer correctness
plus the number of searches — a floor (too few = guessing) and a ceiling (too many = search loop).
Guard: 60 tool calls.

**W4 — one unbounded long-form response.** A 12-section technical manual with given section titles and
cross-references, produced in a single response with no token limit. Graded: sections present,
cross-references resolved, and the loop instrument over the whole output, with the halt position
recorded when it fires. This is the direct replacement for `gen_manual`: the run ends when the model
stops or when it starts repeating, never because the harness cut it.

**W5 — plan, execute, verify.** A bug fix that must be proven: graded on whether the transcript
contains a *successful test run after the last edit* (test-run log entries, not prose). This is the
classic agentic failure at lower cache precision — declaring success without verifying. Guard: 60
tool calls.

None of the five is answerable from one paragraph, and "context used" is measured — unique prompt
tokens of the files the agent actually read and the depth spread of those reads — so an arm that
ignores the document cannot score well by luck.

## Statistics

- **Unit of analysis = the workflow instance, not the rep.** Each tier has ≥12 independent instances;
  three seeds per instance are averaged *inside* the instance, never pooled as independent pairs.
  Per-tier minimum attainable p is then 2/2¹² = 0.0005, and with a Holm correction over the arms
  actually compared (14 × 3 tiers = 42) the threshold is still reachable for a real effect.
- **Effect sizes with intervals.** Report per-arm success rate with a Wilson interval across
  instances, and the paired difference against that model's f16 arm with a cluster bootstrap
  (1,000 resamples over instances, seeded). The current report's exact sign-flip test stays for the
  paired tiers, but clustered.
- **Primary vs secondary.** Success is the primary endpoint; loops/partial credit are secondary and
  are reported as rates with intervals, never as the basis of a ranking on their own.
- **Pre-registered decision rule**, in the plan before running: an arm is "worse" if the lower bound
  of the paired success-rate interval excludes 0 *and* the point estimate is worse; ties otherwise.
- **Power statement in the report header**: the smallest attainable p, the number of instances, and
  the number of comparisons. No claim stronger than those allow.

## Budgets and fairness

- **No generation budget.** Every arm runs until it completes, loops, or exhausts the window, under
  identical stopping rules and identical guards. Truncation disappears from the comparison, and so
  does the outcome-dependent retry path that gave Gemma 8,192 tokens on 12–13 of 18 samples while
  every other arm got 2,048.
- Thinking is capped per request and the cap is identical for every arm; a thinking cap that fires is
  recorded as `halted_by=thinking_cap` and kept separate from a loop halt in the output. (Gemma's
  failures in the current run are content-side loops, not thinking truncation; that report is being
  corrected.)
- Token and wall-clock cost per success are reported; an arm that "succeeds" by spending 4× the
  tokens is visible.
- Same seeds, same prompts, same document per cluster; the document hash is in the manifest.

## Harness changes required

1. **Multi-turn tool runner** with a step budget, a transcript recorder (tool, args, ok, state diff)
   and per-step loop detection: identical consecutive calls, identical calls ≥3, no-progress windows
   (no graded state change in k steps), and A↔B oscillation.
2. **State-based graders**: repo tree hash + hidden suite (already have `repo-run.py`), ledger hash +
   invariants, file-set comparison, and a "did it run the tests after the last edit" check parsed
   from the tool transcript (not from prose).
3. **`extract_*` hardening**: refuse edits to hashed test files (`test_visible.py`), reject `../`
   paths, and parse `ANSWER:` from reasoning as well as content (the current docstring promises it,
   the code does not).
4. **Fixture repairs before any run**: `csvlite` checks must contain real newlines (`\n`, `\r\n`,
   missing final newline) — today they contain literal backslash-n, so the README's central rule is
   untested; `count_proj` must use uneven per-project counts (it is a free point today);
   `repo_fix`'s visible test must be scoreable (the degenerate point-gap case).
5. **Parity guard**: the runner asserts every arm ran with the same thinking cap, the same response
   ceiling and the same guards, and fails the arm if any per-arm override fires.
6. **Manifest**: hardware captured at matrix start (not by a later process), the model files'
   sha256, per-arm token cost, and the exact command lines.

## Cost

Without a generation cap the cost is set by how long a run takes to finish or to loop, not by a
budget. A well-behaved instance is 10k–40k generated tokens (15–60 min at 12–48 t/s depending on the
model); a looping instance is cut by the detector within a few thousand tokens, so failures are
cheap. The full grid (5 models × 3 arms × 5 workflows × 12 instances) is still months of wall clock,
so the plan is:

- 3 models first (dense 27B, Flash-Next, Gemma — the ones that bound the effect), 2 arms (f16, q4_0);
- 12 instances for W1/W2/W5, 8 for W3/W4;
- workflows shared across KV arms (same instance, same document, same seeds) so the prefill is paid
  once per arm, as today;
- estimated **~20 h for that reduced matrix**, ~55 h for all five models.

The governor, resumability and per-arm checkpointing are reused unchanged, and an arm cut by the
deadline is reported as cut (the current `incomplete_arms` banner).

The governor, resumability and per-arm checkpointing from the current harness are reused unchanged.

## What the current run still tells us

- `q4_0` did not make these five models loop more on the probes that were run; that statement is
  limited to ~1k visible tokens, which is why this plan exists.
- The one large effect (Gemma at q4_0, code 41 vs 89) is a generation-length/format failure: 3 of 6
  tasks hit the cap, and `repo_fix` emitted 20,459 tokens with no file block. It is not evidence of
  KV-depth damage, and Gemma has the fewest global attention layers (5 of 30), which is the opposite
  of what a depth effect would predict.
- Every paired comparison in the current report is now task-clustered with a Holm correction; after
  that, no arm separates (adjusted p = 1.00 for all 42 comparisons). The earlier "three separable
  arms" was pseudo-replication.

## Sequencing

1. Fixture repairs and the tool runner (no GPU).
2. W1 + W4 only, on one model at f16/q4_0, to confirm the halt detector fires on a known loop and that
   the graders discriminate; measure the observed token distribution to size the wall-clock guard.
3. Add W2/W3/W5, then the remaining models.
4. Report: one page (answer, per-workflow success with intervals, loops with intervals, cost per
   success) plus an appendix; the current report's structure is the template, minus the grading bug
   and the pseudo-replicated tests, both fixed in the working tree already.
