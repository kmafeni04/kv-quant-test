# Results — the prior study (archived)

These are the findings of the 15-arm, short-answer study that [AGENTIC-BENCH-PLAN.md](AGENTIC-BENCH-PLAN.md)
replaces. They are kept because they define what the new harness had to fix: a generation cap, a
reasoning budget, a repeat penalty and three reps at 0.7 sampled away the very failure modes the new
study measures. Nothing below was produced by the current harness and none of it is comparable with a
run of it.

Run `results/2026-11-01/`: five models at `-c 102400`, each with f16, q8_0 and q4_0 KV caches (the
key/value cache holding every earlier token's attention state), 540 requests, no failed requests, no
deadline cut. Grading and statistics were corrected after the run: `repo_fix` now needs its visible
test, and every paired test clusters the reps of a task and Holm-corrects across comparisons. Every
number below is from the corrected artifacts.

**The loop probes are too short to settle the looping question.** They see about 1,000 visible tokens
per probe and four independent probes per arm. The design that replaces them is in
[AGENTIC-BENCH-PLAN.md](AGENTIC-BENCH-PLAN.md).

## Answer

Quantizing the KV cache made no resolvable difference on four of the five models. Gemma is the
exception, and it failed by writing too long, not by answering wrong: at q4_0 it hit the code tier's
token ceiling on three of six tasks and its code score fell from 89 to 41. On this build the quantized
cache is also 17–28 % slower to decode at depth, so it buys context, not speed.

The table ranks all 15 arms by capability, which is the mean of the retrieval and code percentages,
with ties going to the arm with fewer repeating samples and then to the faster one.

| arm | capability | retrieval | code | structural loops | decode t/s |
|---|---|---|---|---|---|
| `qwen38-flash` q4_0 | 84.9 | 80.0 % | 89.9 % | 0/12 | 12.3 |
| `qwen38-flash` f16 | 83.3 | 76.7 % | 90.0 % | 0/12 | 16.6 |
| `qwen38-27b` q8_0 | 83.3 | 76.7 % | 90.0 % | 0/12 | 10.3 |
| `qwen36-35b` q4_0 | 82.2 | 66.7 % | 97.8 % | 0/12 | 30.8 |
| `qwen38-flash` q8_0 | 80.4 | 80.0 % | 80.8 % | 0/12 | 12.6 |
| `qwen36-35b` f16 | 79.0 | 60.0 % | 98.1 % | 0/12 | 38.7 |
| `qwen36-35b` q8_0 | 76.2 | 63.3 % | 89.0 % | 1/12 | 31.2 |
| `qwen38-27b` q4_0 | 75.6 | 70.0 % | 81.3 % | 0/12 | 9.7 |
| `qwen38-27b` f16 | 75.3 | 70.0 % | 80.6 % | 0/12 | 12.4 |
| `ornith-35b` f16 | 66.8 | 63.3 % | 70.3 % | 0/12 | 48.8 |
| `ornith-35b` q8_0 | 63.2 | 53.3 % | 73.1 % | 1/12 | 35.7 |
| `gemma4-26b-a4b` f16 | 62.8 | 36.7 % | 89.0 % | 3/12 | 46.9 |
| `ornith-35b` q4_0 | 60.5 | 60.0 % | 60.9 % | 0/12 | 35.2 |
| `gemma4-26b-a4b` q8_0 | 56.6 | 23.3 % | 89.9 % | 2/12 | 33.8 |
| `gemma4-26b-a4b` q4_0 | 33.7 | 26.7 % | 40.8 % | 3/12 | 34.3 |

Capability is the mean of the retrieval and code percentages. The rank order of the top three arms
sits inside one graded outcome, so read the table as levels, not as a podium.

## Nothing here is statistically separable

Each tier has six task clusters (four for the loop probes), and the three reps of a task share its
prompt, document and difficulty. Counting them as independent pairs is what made an earlier version of
this report claim three separable arms: with 18 pairs the sign-flip test returned p = 0.0007, and with
the reps summed per task the same data gives p = 0.0625. After correcting over the 42 arm × tier
comparisons, every adjusted p is 1.00.

The design can therefore rule out only a large, consistent difference. "No difference" in this report
means "not resolvable at six clusters", not "equal".

## How a lossy cache can outscore f16

It can't be better informed: q4 stores fewer bits, so the cache is a damaged copy. A small change in a
damaged copy can still flip a discrete outcome either way, the way re-rolling a near-tied die can. One
retrieval answer is 1 point of 30; `semver` is 261 checks that hinge on one ordering decision. Both
directions happened: `qwen38-flash` q4_0 beat its own f16 arm by one retrieval sub-value (p = 1.00),
and Gemma q4_0 lost 376 checks to truncation.

Evidence that quantization helps would be an advantage that survives more seeds and more tasks, or a
distributional measure (KL against the f16 model, which separated q4_0 at 32k in the earlier study).
Here KL means Kullback–Leibler divergence, a measure of how far one next-token distribution has moved
from another. A one-point difference on 30 points is neither.

## What each tier actually measured

The task names below (`niah_multi`, `vt_trace`, `agg_sum`, `fwe_count`, `count_proj`, `semver`,
`intervals`, `statistics`, `csvlite`, `repo_fix`) are the harness task ids; `csvlite` is a small
CSV-parsing fixture and `repo_fix` is the repository-repair fixture.

### Retrieval: 30 points per arm

Four of the six tasks saturate or floor: `niah_multi` is 9/9 in all
15 arms, `vt_trace` 6/6 in 14, `agg_sum` 0/6 in 11, `fwe_count` 0/3 in 13. `count_proj` is degenerate
— every project has the same record count, so guessing 3 scores. This tier is a control for depth, not
an instrument.

### Code: 780 hidden checks

Three modules carry 84 % of it and two of those are saturated in 14 of 15
arms; `semver` alone is 261 checks that move as one bit (118, 190 or 261 depending on how prerelease
ordering is implemented). The four arms that resolved the most tasks resolved four of six; `repo_fix`
was resolved by no arm because its visible test is not scoreable by the hidden suite (fixture defect,
recorded).

### Loops: 12 samples per arm

Seven of the ten structural repetition events are Gemma on
`enumerate_shards`, at every cache type: identical lines up to 159 times, 200-character blocks up to
15 times, tail cycles of 105–300 characters. Gemma's retrieval failures are the same failure in another
form — 447 repeated lines and no `ANSWER` line — not truncated thinking. A further 25 of 45
`gen_manual` samples trip the compression heuristic with nothing repeating; that is a template-prose
artefact and it is excluded from the structural count.

## What this run cannot tell you

The limits below are structural, not incidental: each one follows from how the prior study was built,
and each one is why the replacement study changes that part of the harness.

Looping at depth is unmeasured. The probes stop at roughly 1,000 visible tokens by construction, which
is too short for a model to reach a repetition attractor that needs thousands of tokens to form.

Whether q4_0 is safe in agentic use is unmeasured. Nothing here runs a tool loop longer than five
calls, and the one model that did fail did so by writing, not by looping.

Cross-model comparisons are not available. One document was tokenized five ways (81,765 to 91,357
prompt tokens), and every claim in this file is about a model's own KV ladder rather than about one
model against another.

`csvlite` is not a clean CSV measurement. Its fixture checks contain literal `\n` text, so the newline
rules the spec centres on were never tested.

The full tables, per-task breakdown, cross-model agreement, timing and hardware are in
[`REPORT-APPENDIX.md`](../results/2026-11-01/REPORT-APPENDIX.md); the method and threats are in
[METHOD.md](METHOD.md).
