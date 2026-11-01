# Results

Run: `results/2026-11-01/`. One shared context, `-c 102400`, five models × three symmetric KV types
(f16 / q8_0 / q4_0), **540 graded samples** (36 per arm, plus 39 re-runs at a larger budget where a
sample hit the cap), 15/15 arms complete, zero failed requests, no deadline cut. Document sha256 `944bb825ead422d7…` (305,234 chars, 91,284 tokens on the
Qwen tokenizer, 81,765 on Gemma's). llama.cpp `2a667032dc78` (build 11698), `-fa on`, no speculative
decoding.

**Test wall clock: 15 h 27 m** (14 h 12 m of it the matrix; fit probe 35 m; harness validation 40 m).
Model downloads are excluded. Per-arm times are in `REPORT.md` §"How long this took"; the slowest arm
was the dense 27B at q4_0 (1 h 27 m) and the fastest was Ornith at f16 (21 m).

## Verdict: which KV cache to use, per model and overall

Ranking rule, fixed in `kvtest/report.py::verdict` so anyone can recompute it: **capability =
mean(R %, C %)** per arm (each tier already on a 0–100 scale, so 780 code checks cannot swamp 30
retrieval points), then **fewer structural-loop samples**, then **higher decode t/s**.

| model | best KV | capability | runner-up | structural loops | decode t/s | why |
|---|---|---|---|---|---|---|
| `qwen38-flash` | **q4_0** | 84.95 | f16 (83.35) | 0/12 | 12.3 | best overall; +1 R point and −1 check vs its own f16 arm (p = 1.000): a tie resolved by nothing more than a one-outcome difference |
| `qwen38-27b` | **q8_0** | 83.35 | q4_0 (75.65) | 0/12 | 10.3 | +2 R points, +73 checks vs f16 (p = 0.500) — the +73 is the `semver` bit, not a trend |
| `qwen36-35b` | **q4_0** | 82.25 | f16 (79.05) | 0/12 | 30.8 | +2 R points, −2 checks vs f16 (p = 0.500/0.750); its f16 arm was the one truncated once at the code cap |
| `ornith-35b` | **f16** | 66.80 | q8_0 (63.20) | 0/12 | 48.8 | q4_0 loses 6.3 capability points; its one structural loop is at q8_0 |
| `gemma4-26b-a4b` | **f16** | 62.85 | q8_0 (56.60) | 3/12 | 46.9 | q4_0 loses 29.1 points because it hit the 6,144-token code cap on 3 of 6 tasks — truncated output, not wrong code; Gemma loops at every KV type |

**Best overall: `qwen38-flash` with a q4_0 KV cache** (84.95/100, 0/12 structural loops, 12.3 t/s
decode, 9.1 GiB peak with 48 MoE layers on the CPU). **Its lead is not established.** Testing every
arm against it on all paired samples pooled across the three tiers, 11 of the other 14 arms cannot be
distinguished from it (p ≥ 0.05) — including `qwen38-flash` f16, which differs by one retrieval
sub-value. The only arms that *are* separable are separable as **worse**: Gemma f16 (p = 0.047),
Gemma q4_0 (p = 0.0034) and Ornith q4_0 (p = 0.0078). So the run establishes that **q4_0 is not a
downgrade on four of five models** and that Gemma's q4_0 failure is verbosity/truncation rather than
wrong answers — not that any KV type is better than f16.

## How a lossy cache can score higher than f16

Three questions come up, and they have one answer each.

**1. Can q4 hold *more* information than f16? No.** A quantized cache stores each number with fewer
bits, so it is a slightly damaged copy of what the model computed at full precision. Nothing in it is
sharper than f16. So "q4 beat f16" can only mean *the damaged copy produced a different, slightly
lucky set of outputs* — never that it carried better information.

**2. Then why does the score go up?** Because scores here are discrete and thin. One retrieval answer
is 1 point out of 30; one code module is 261 checks that turn on a single design decision (`semver`
scores 118, 190 or 261 depending on how prerelease ordering is implemented). A tiny numerical
difference in the cache — invisible in the prose the model writes — can flip one of those decisions,
either way, exactly like re-rolling a near-tied dice. When the flip lands well the arm looks better;
when it lands badly it looks worse. Both happened in this run: `qwen38-flash` q4_0 beat its f16 arm by
1 point out of 30, and `gemma` q4_0 fell 376 checks because it would not stop writing.

**3. How do we tell luck from a real effect?** By testing it. Every arm was compared with its own
f16 arm on matched samples (same task, same seed, same document prefix) with an exact paired test
that asks "could a difference this big arise from nothing?". The answer here is **no arm is
significantly better than f16 on anything** (all p ≥ 0.05). The only arm differences that survive the
test are three arms that are significantly *worse* — Gemma f16, Gemma q4_0 and Ornith q4_0 — and even
those are separations from the leader, not demonstrations that f16 is better than q4_0 in general.

**What would count as evidence that q4 helps?** A repeat. Run the same arms with more seeds and more
tasks and see whether the advantage survives; or measure the quantity that resolves it (KL/perplexity
against the f16 distribution, which separated q4_0 at 32k in the earlier study). A single 1-point
difference on 30 points is not that evidence, and the report says so wherever the number appears.

## The one-line answer

**At ~91k tokens of context, q4_0 and q8_0 KV caches cost essentially nothing in capability on four
of these five models — the differences are the same size as the noise the code tier is known to
carry — and quantizing the KV cache did not make any model loop more. The one large effect is
Gemma-4-26B-A4B at q4_0, and it is not a wrong answer: the model ran past the generation cap on
three code tasks, so the arm produced truncated output. On this build, quantized KV is also 10–28 %
*slower* to decode at this depth than f16, so the reason to use it is context headroom, not speed.**

## R tier — retrieval, tracing, aggregation (30 points per arm, 3 reps)

| model | f16 | q8_0 | q4_0 |
|---|---|---|---|
| `qwen38-flash` | 76.7 | 80.0 | 80.0 |
| `qwen38-27b` | 70.0 | 76.7 | 70.0 |
| `qwen36-35b` | 60.0 | 63.3 | 66.7 |
| `ornith-35b` | 63.3 | 53.3 | 60.0 |
| `gemma4-26b-a4b` | 36.7 | 23.3 | 26.7 |

No model moves monotonically with the KV type, and the two directions are equally represented
(qwen38-27b and flash go *up* at q8_0; ornith and gemma go *down*; qwen36-35b goes up at q4_0).
With 30 points per arm, one sub-value of one sample is 3.3 points: every delta above except gemma's
is inside ±2 samples of noise.

The task breakdown explains why this tier barely separates anything:

| task | behaviour |
|---|---|
| `niah_multi`, `vt_trace` | saturated at 9/9 and 6/6 for all four non-Gemma models in all three arms |
| `niah_deep`, `count_proj` | 3/3 and 1–3/3 for four of five models — one model's misses are stable across arms |
| `agg_sum` (count+sum over 120 ledger rows) | floor: 0/6 in 11 of 15 arms; 1–4 of 6 only in the Flash arms and once for the 27B at q8_0 |
| `fwe_count` (exact token count with decoys) | floor: 0/3 in 13 of 15 arms (1/3 for 27B q8_0 and Flash f16) |

So the R tier is a *control*, not an instrument: it proves the arms read the same document at the
same depth and answer the same way, and it shows no depth-related collapse from quantization.

## C tier — hidden checks (780 per arm) and resolved tasks

| model | f16 | q8_0 | q4_0 | resolved (of 6) f16/q8_0/q4_0 |
|---|---|---|---|---|
| `qwen36-35b` | 765 (98.1 %) | 694 (89.0 %) | 763 (97.8 %) | 4 / 3 / 3 |
| `qwen38-27b` | 629 (80.6 %) | 702 (90.0 %) | 634 (81.3 %) | 4 / 4 / 4 |
| `qwen38-flash` | 702 (90.0 %) | 630 (80.8 %) | 701 (89.9 %) | 4 / 4 / 3 |
| `gemma4-26b-a4b` | 694 (89.0 %) | 701 (89.9 %) | 318 (40.8 %) | 4 / 3 / 3 |
| `ornith-35b` | 548 (70.3 %) | 570 (73.1 %) | 475 (60.9 %) | 4 / 3 / 3 |

Two things dominate these numbers, and neither is a KV effect:

1. **`semver` is bimodal, exactly as the earlier 32k study found.** One prerelease-ordering decision
   flips ~70–170 pairwise checks: every arm scores 24, 71, 83, 118, 182, 187, 190, 253 or 261 out of
   261, and the spread between arms of the same model is a re-draw of that single decision
   (qwen36-35b 261→187→253; qwen38-27b 118→190→118; flash 190→118→190). It is one bit, not 261
   measurements, and it is the single largest contributor to the C-tier variance.
2. **Gemma's q4_0 collapse is truncation, not wrong answers.** On `cg_intervals` (15/162),
   `cg_semver` (0/261, `SyntaxError` mid-file) and `repo_fix` (no `### FILE:` block at all, 20,459
   completion tokens) the response hit the 6,144-token ceiling with `finish_reason=length`; at f16
   the same three tasks finished at ~4,100 tokens. So q4_0 made Gemma generate longer and never
   finish. That is a genuine behavioural difference and it is reported as such — but it is a
   *budget interaction*, not proved wrong code, and the other 12 arms show nothing like it.

The four modules that did not saturate at 100 % everywhere (`csvlite`, `fixedpoint`, `intervals`,
`statistics`) never differ by more than one or two checks between arms of the same model — e.g.
`fixedpoint` 45/45 in 11 arms, 39–42 in the other four; `intervals` 162/162 in 14 arms.

`repo_fix` (54 hidden checks + the visible test) is the one task with a visible 0-vs-54 spread, and
it tracks *format compliance*: the arms that scored 0 emitted no parseable `### FILE:` block
(ornith f16 and q4_0, gemma q4_0), the rest landed 43–54. Note that no arm reached `resolved` on
`repo_fix`: the fixture's third visible test needs `coverage_report` to surface a degenerate
point gap, which the hidden suite does not score (documented in `docs/METHOD.md`), so `resolved`
requires both and is rarely earned.

## Loops — the direct test of the original question

L-tier probes, 12 samples per arm (3 probes × 3 reps + 3 tool episodes). Two counts are given,
because the difference between them is the whole point: **union** is every verdict the detector
raised, **hard** is only samples that contain a repeated line, n-gram, 200-char block or tail cycle.

| model | f16 union / hard | q8_0 union / hard | q4_0 union / hard |
|---|---|---|---|
| `gemma4-26b-a4b` | 4 / **3** | 2 / **2** | 3 / **3** |
| `ornith-35b` | 2 / 0 | 4 / **1** | 3 / 0 |
| `qwen36-35b` | 2 / 0 | 3 / **1** | 0 / 0 |
| `qwen38-27b` | 2 / 0 | 2 / 0 | 2 / 0 |
| `qwen38-flash` | 2 / 0 | 2 / 0 | 2 / 0 |

**Quantizing the KV cache did not increase looping.** Structural repetition is 0–3 of 12 per arm,
it is present at every KV type for the only model that repeats (Gemma), and the two arms with one
hard flag each are q8_0 (ornith, qwen36), not q4_0.

### Why the union column can look backwards

qwen36-35b reads 2/12 at f16, 3/12 at q8_0 and 0/12 at q4_0, which invites the reading "q4_0 fixed
it". The samples say otherwise:

- The f16 and q8_0 verdicts are all `gen_manual` samples flagged *only* by the zlib ratio, at
  0.424 / 0.420 (f16) and 0.421 / 0.429 (q8_0) against a 0.42 threshold. No repeated line, no
  repeated n-gram, no cycle: those three samples have `line_run=1` and `word_run_4=1`, i.e. nothing
  repeats. A 12-paragraph manual generated from twelve fixed rules is formulaic by construction, so
  it compresses to ~0.42 whether or not anything is wrong.
- q4_0's manuals landed at 0.419 / 0.416 / 0.391 — three thousandths under the line, from **longer**
  text (6,354–9,245 characters vs 4,803–6,149 at f16). Longer, more varied text compresses less, so
  the arm scored "cleaner" mainly by being more verbose. One of those q4_0 samples ran to the
  2,048-token cap while narrating its own process ("I will scan the document to find exactly where
  each rule is stated…"), which is sloppier behaviour, not better.
- The single structural repetition in qwen36-35b's whole ladder is at **q8_0**: a 75-character tail
  repeated 3× in the shard listing. So the model's one real loop sits in the middle arm.

This is why the raw metrics are printed beside every verdict, and why `docs/METHOD.md` records the
threshold as a heuristic with a measured false-positive rate on templated text.

What the flags actually are matters more than the rate:

- **One model loops, and it is Gemma.** 8 of the 10 hard repetition events in the whole run are
  Gemma on `enumerate_shards`: identical lines repeated up to 159× in a row, 200-char blocks
  repeated 4×, 3-cycles of 105–300 characters in the tail. It does this at f16 (2 of 3 reps),
  q8_0 (2 of 3) and q4_0 (3 of 3) alike, and it fails the task while doing it (4–11 of 40 shards
  correct vs 39/40 for the other models). This is a model property, not a quantization effect.
- Single cycles also appear once each in ornith-35b q8_0 and qwen36-35b q8_0 on the same task.
- **25 of the 45 `gen_manual` samples cross the compression threshold** (zip ratio 0.42–0.44) with
  no line, n-gram or cycle signal. A 12-paragraph manual written from twelve fixed rules is
  formulaic by construction, so this is a threshold artefact of a templated task, and it is
  distributed evenly across arms (2–3 of 3 everywhere) — it inflates the "loop rate" to
  17–25 % uniformly without carrying any KV signal. It is left in the table, raw numbers beside the
  verdict, so a reader can discount it.
- **Tool episodes: 45/45 clean.** Every episode used exactly 5 calls (4 `read_shard` + 1
  `submit_report`), every report was accepted, and no identical-consecutive call, no triple call and
  no four-turn stall occurred in any arm — including Gemma. The one tool-tier flag in the table is
  a 3-cycle inside Gemma's *reasoning text*, not in its calls.
- Thinking-side loops: no arm produced a reasoning loop beyond the same cycle signals above
  (`reasoning` column of the loop table is 0–1 per arm).

## Agreement — the low-level signal

Share of R samples where the arm gives byte-identical answers to its own f16 arm (same task, seed):

| model | q8_0 | q4_0 |
|---|---|---|
| `qwen38-27b` | 0.72 (13/18) | 0.72 (13/18) |
| `qwen36-35b` | 0.72 (13/18) | 0.56 (10/18) |
| `qwen38-flash` | 0.61 (11/18) | 0.67 (12/18) |
| `ornith-35b` | 0.50 (9/18) | 0.71 (12/17) |
| `gemma4-26b-a4b` | 0.20 (1/5) | 0.20 (1/5) |

The Qwen-family models reproduce 56–72 % of their f16 answers exactly under quantization; the rest
differ in wording or in a digit, and only some of those differences change the score. Gemma's 0.20
is measured on 5 comparable samples (its answers were often absent — it spent the budget thinking),
so it is weak evidence; it reflects a model that is highly unstable at this depth even at f16.

Cross-model agreement on the same task and seed is reported in `REPORT.md`. At f16 the four
Qwen-family models agree on 50–72 % of answers, and Gemma agrees with them on 33–44 % — the models
are genuinely different, so cross-model comparison is descriptive and the within-model ladder is the
experiment.

## Speed at ~91k depth

| model | prefill t/s (f16) | decode t/s f16 → q8_0 → q4_0 |
|---|---|---|
| `ornith-35b` | 67.4 | 48.8 → 35.7 → 35.2 |
| `gemma4-26b-a4b` | 34.5 | 46.9 → 33.8 → 34.3 |
| `qwen36-35b` | 57.2 | 38.7 → 31.2 → 30.8 |
| `qwen38-flash` | 21.5 | 16.6 → 12.6 → 12.3 |
| `qwen38-27b` | 17.9 | 12.4 → 10.3 → 9.7 |

**f16 is the fastest KV type to decode at this depth, on every model**, by 10–28 %; q8_0 and q4_0
are within ~3 % of each other. Prefill is flat. This is a property of the pinned build's CUDA
flash-attention kernels (the f16 path is the hot one), not a property of quantization in general,
but it is what a user of this build would measure. Prompt cache reuse was 100 % (prefix hits of
81k–91k tokens on every post-first request), so the per-arm cost is one 91k prefill.

## What this means in practice

- **Use q4_0 if you need the context.** At ~91k tokens it cost no measurable capability on four of
  five models, and it never increased looping. Gemma-4-26B-A4B is the exception to watch: it needs a
  larger generation budget at q4_0, because it writes longer and truncates.
- **Do not expect q4_0 to be faster.** On this build it is slower to decode than f16 at depth; it
  buys memory, not throughput.
- **Do not read small C-tier deltas as KV damage.** The `semver` module carries a single design
  decision worth up to 170 checks, and the code tier is one greedy pass per task. Both are printed
  beside every number for this reason.

## Caveats specific to this run

1. Gemma truncated on 3 of 6 code tasks at q4_0 and hit the extended 8,192-token ceiling on 12–13 of
   36 requests in **all three** of its arms (`budget_raised` in `samples.jsonl`): it is the only
   model here whose thinking does not fit the tier budgets, so its numbers carry that artefact in
   every arm, not only q4_0.
2. `repo_fix` cannot be `resolved` without the visible test, and the visible test's third case is
   unscorable by the hidden suite (fixture quirk, documented).
3. The R tier saturates on 4 tasks and floors on 2, so it is a control for depth, not a
   discriminating instrument — 30 points per arm is too coarse to resolve small effects.
4. The zip-ratio loop threshold fires on templated `gen_manual` text (25/45 samples); the real
   repetition events are counted separately in this document.
5. VRAM peaks in the report include transient compute buffers, so they are not a clean measurement
   of KV-cache size (the KV scaling is in `fit.json`'s geometry block: 64/34/18 KiB per token for
   the dense 27B, 20/11/5.6 KiB for the 35B MoEs).
