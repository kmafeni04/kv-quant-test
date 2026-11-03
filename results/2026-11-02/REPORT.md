# Full-context KV quantization: agentic workflows, zero–20 key scores

Compressing the cache to q4_0 changed what all three models predicted, but most task probes could not turn that into a measurable loss. Against each model's own f16 run, mean KLD was 0.2052 on Gemma, 0.0313 on Ornith and 0.0095 on Qwen3.6, and same-top agreement fell to 81.2%, 88.9% and 90.7%. Deep-context perplexity moved by +3.1%, -3.8% and +1.1%, so the effect is model-specific rather than a uniform penalty.

The counted workflows mostly saturated or failed for reasons unrelated to the cache. Ornith and Qwen3.6 scored 20/20 on both the 20-key retrieval and the compaction brief in both cache types. Gemma produced no scored 20-key answer in either arm because it looped in reasoning, and both arms failed the same way.

No arm reconciled the ledger (0/36 attempts), and Qwen3.6 wrote all six manual sections in both arms. The one graded loss that tracks the cache is Ornith's search arm: q4_0 answered 12/18 keys against 18/18 at f16 and made six searches instead of nine, skipping a hop.

The distribution shift is real and largest on Gemma, but the study can attribute a task-level loss to q4_0 only on that search probe; it cannot say compression made the other workflows worse.

This report compares an uncompressed 16-bit key/value cache (f16) with a cache compressed to 4 bits
per value (q4_0) inside three models, at one full-context rung each. The key/value cache — the KV
cache — is the stored attention state for every token the model has already read; the study asks what
compressing it costs the work. Six workflows per arm measure that cost from different angles.

## Premise, power and disclosure

The run gives each arm 12 instances of the two counted workflows (W2, the 20-key retrieval, and W3,
compaction), 6 instances of the three long workflows (W4, the ledger; W5, search; W6, long-form), and
3 passes of W1, the distributional instrument. A workflow instance, not a request or a repetition, is
the unit of analysis.

Decode was greedy: every request carried `temperature: 0.0` and no other sampling key, with no seeds
and no repetitions. Greedy decode is not a production setting and can be more or less loop-prone than
a sampled one, so every loop count below describes greedy decoding. No reasoning budget and no retry
path applied; the only cap was a 65,536-token response ceiling, recorded as `ceiling`.

The smallest attainable two-sided p is 2/2¹² = 0.0005 for W2 and W3 and 0.031 for W4–W6. Pairwise
tests are the paired difference against that model's f16 arm, with a cluster bootstrap over instances
(1,000 resamples, seeded); an arm counts as worse only when the interval excludes 0 and the point
estimate is worse. The document was 313,025 characters, sha256 `11e504b3f33395bc`, and tokenized to
84,113 tokens under Gemma and 94,170 under both other models.

## Cells run

The table lists every arm that ran and the wall time it took, including the server load. All six arms
are present; a cut or unrun cell would be named here rather than dropped.

| arm | wall |
|---|---|
| `gemma4-26b-a4b__f16` | 4 h 10 m |
| `gemma4-26b-a4b__q4_0` | 5 h 12 m |
| `ornith-35b__f16` | 1 h 42 m |
| `ornith-35b__q4_0` | 2 h 40 m |
| `qwen36-35b__f16` | 3 h 40 m |
| `qwen36-35b__q4_0` | 3 h 48 m |

## W1: KLD and perplexity (the fast instrument)

W1 compares each arm against the f16 arm's own next-token probabilities. Mean KLD is the average
Kullback–Leibler divergence in nats, same-top agreement is the share of positions where both runs rank
the same token first, and PPL is perplexity, the exponentiated average negative log-likelihood. The
table gives one row per arm; an f16 row is a self-check and should read 0 and 100%. The paragraphs
below the table explain the deep-context depth and the one place where the pinned tool's own PPL
output is not trusted.

| model | KV | mean KLD ± unc | same top p | ln PPL ratio | deep PPL | deep PPL / f16 |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | 0 ± 0 | 100.0% | 0.00000 | 177.164 | 1.0000 |
| `gemma4-26b-a4b` | `q4_0` | 0.2052 ± 0.0084 | 81.2% | -0.08176 | 182.718 | 1.0313 |
| `ornith-35b` | `f16` | 0 ± 0 | 100.0% | 0.00000 | 19.560 | 1.0000 |
| `ornith-35b` | `q4_0` | 0.0313 ± 0.0013 | 88.9% | -0.03999 | 18.821 | 0.9622 |
| `qwen36-35b` | `f16` | 0 ± 0 | 100.0% | 0.00000 | 14.500 | 1.0000 |
| `qwen36-35b` | `q4_0` | 0.0095 ± 0.0006 | 90.7% | 0.00357 | 14.656 | 1.0107 |

The ln PPL ratio is each arm's own scored PPL divided by that model's f16 arm's PPL on the identical slice. On an identical-model check the pinned tool printed `Mean PPL(Q)` equal to the baseline run's own PPL while its `Mean PPL(base)` reconstruction read 26% lower with mean KLD 0 and same-top 100%, so the tool's own base-PPL ratio is not used as evidence; its KLD, Δp and same-top outputs are.

Perplexity is comparable only within a model, never across models. A zero mean KLD with a tight interval on every arm means the instrument did not reach the cache: check that `-ctk`/`-ctv` were passed and that the baseline slice matches token for token.

## W2: 20-key retrieval

Each instance asks for the live code of twenty shards in a fixed order. The grader reads the final
`ANSWER:` line and awards one point per position holding the correct code; missing, wrong and
near-miss decoy values all score zero. A response halted by the loop detector, the window or the
ceiling is counted as a halt and excluded from the mean, because a halt is not a wrong key. The table
shows the mean out of 20, the score range, how many instances were scored out of how many were
attempted, and the paired result against f16.

| model | KV | mean /20 | sd | min | max | scored | halts | q4_0 − f16 | interval | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | — | — | — | — | 0/12 | 12 | — | [—, —] | no paired instances |
| `gemma4-26b-a4b` | `q4_0` | — | — | — | — | 0/12 | 12 | — | [—, —] | no paired instances |
| `ornith-35b` | `f16` | 20.00 | 0.00 | 20 | 20 | 12/12 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |
| `ornith-35b` | `q4_0` | 20.00 | 0.00 | 20 | 20 | 12/12 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |
| `qwen36-35b` | `f16` | 20.00 | 0.00 | 20 | 20 | 12/12 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |
| `qwen36-35b` | `q4_0` | 20.00 | 0.00 | 20 | 20 | 12/12 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |

A scored instance is one that produced an `ANSWER:` line. Instances halted by the loop detector, the window or the ceiling are counted in `halts` and excluded from the score, per the rule that a halt is its own counter and is never folded into a score.

### Hit rate by planted depth

Every scored arm hit all keys in all five depth bands; Gemma produced no scored instance, so its cells would be empty.

The f16 arm produced no scored W2 instance: every instance was halted by the loop detector, the window or the ceiling. Halts are their own counter, so no key was scored as lost, but the probe has no reference data for that model.

The scored arms saturated: every one holds all twenty keys. The stop rule says the probe widens to all forty shards before the second model runs, because otherwise it measures format compliance rather than cache damage.

## W3: compaction

Phase A asks for a handover brief of at most 1,500 words carrying twenty shard identifiers, the ledger
rows a named tenant's mutations touch, and five invariants. Phase B discards the document and the
phase-A transcript, sends the brief verbatim, and asks five questions answerable only from the brief.
The table reports how many of the twenty identifiers survived in the brief, the `mutated` and `absent`
splits, and the phase-B keys.

| model | KV | retention /20 | sd | scored | halts | mutated | absent | phase-B keys /5 | briefs within cap | paired vs f16 | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | — | — | 0/12 | 12 | 0 | 0 | — | 0/0 | — [—, —] | no paired instances |
| `gemma4-26b-a4b` | `q4_0` | 20.00 | 0.00 | 1/12 | 11 | 0 | 0 | 0.00 | 1/1 | — [—, —] | no paired instances |
| `ornith-35b` | `f16` | 20.00 | 0.00 | 12/12 | 0 | 0 | 0 | 3.00 | 12/12 | 0.00 [0.00, 0.00] | tie (interval includes 0) |
| `ornith-35b` | `q4_0` | 20.00 | 0.00 | 12/12 | 0 | 0 | 0 | 3.00 | 12/12 | 0.00 [0.00, 0.00] | tie (interval includes 0) |
| `qwen36-35b` | `f16` | 20.00 | 0.00 | 12/12 | 0 | 0 | 0 | 3.00 | 12/12 | 0.00 [0.00, 0.00] | tie (interval includes 0) |
| `qwen36-35b` | `q4_0` | 20.00 | 0.00 | 12/12 | 0 | 0 | 0 | 3.00 | 12/12 | 0.00 [0.00, 0.00] | tie (interval includes 0) |

### Brief retention by planted depth

The depth bands below repeat the retention count by document depth. A brief that keeps the shallow
identifiers but drops the deep ones produces a curve that slopes down from left to right.

Every scored brief kept all twenty identifiers in all five depth bands; Gemma's f16 arm produced no scored brief.

The primary endpoint is brief retention: a count out of 20 measured on a full-depth read. Phase B is reported alongside it as the consequence, not as a substitute.

## W4: long-horizon ledger

The model reads a 100k-token operation log through a windowed read tool and applies twenty tenant
mutations with append, retract and transfer calls. The grade compares the final ledger hash with the
expected hash, counts the five invariants it satisfies, and records whether the model reported the
right invariants. Re-applying a mutation changes the hash, so a double application cannot hide.

Identical on all 6 arms: 0/6 reached the expected final hash, 0/6 satisfied all five invariants, and 0 double and 0 foreign mutations were applied.

## W5: agentic search

Sixty fact clusters form fifteen four-hop chains, and three questions each require three searches to
reach the answer value. The table gives answer keys correct out of 18, the Wilson interval, the number
of search calls, and how many instances reached the floor of nine calls (three per question). A model
that answers without searching can still score, but its call count reveals the shortcut.

| model | KV | answer keys | interval | search calls | instances at/above floor | instances |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |
| `gemma4-26b-a4b` | `q4_0` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |
| `ornith-35b` | `f16` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |
| `ornith-35b` | `q4_0` | 12/18 | [43.7%, 83.7%] | 6–6 | 0/6 | 6 |
| `qwen36-35b` | `f16` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |
| `qwen36-35b` | `q4_0` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |

## W6: unbounded long-form

The model writes a six-section manual with given titles and a given cross-reference mapping, with no
token limit except the 64k ceiling. The table counts section headings present, cross-references that
resolve to the required section, instances that got both fully right, and loop halts.

| model | KV | sections present | cross-references resolved | fully correct | loop halts |
|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | 0/36 | 0/36 | 0/6 | 6 |
| `gemma4-26b-a4b` | `q4_0` | 0/36 | 0/36 | 0/6 | 6 |
| `ornith-35b` | `f16` | 0/36 | 0/36 | 0/6 | 0 |
| `ornith-35b` | `q4_0` | 6/36 | 0/36 | 0/6 | 0 |
| `qwen36-35b` | `f16` | 36/36 | 36/36 | 6/6 | 0 |
| `qwen36-35b` | `q4_0` | 36/36 | 36/36 | 6/6 | 0 |

## Behaviour, halts and cost

| arm | completed | loop [95%] | context full [95%] | ceiling [95%] | error | thinking share | out tokens (W2, W3, W6) | mean wall / instance |
|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | 7 | 31 [58.9%, 84.7%] | 0 [0.0%, 8.4%] | 4 [3.8%, 22.1%] | 0 | 100.0% | 327452 | 5 m 55 s |
| `gemma4-26b-a4b__q4_0` | 7 | 31 [58.9%, 84.7%] | 0 [0.0%, 8.4%] | 4 [3.8%, 22.1%] | 0 | 100.0% | 375290 | 7 m 24 s |
| `ornith-35b__f16` | 35 | 1 [0.4%, 12.3%] | 6 [6.7%, 27.8%] | 0 [0.0%, 8.4%] | 0 | 86.1% | 112410 | 2 m 24 s |
| `ornith-35b__q4_0` | 34 | 1 [0.4%, 12.3%] | 6 [6.7%, 27.8%] | 1 [0.4%, 12.3%] | 0 | 89.4% | 117069 | 3 m 46 s |
| `qwen36-35b__f16` | 42 | 0 [0.0%, 8.4%] | 0 [0.0%, 8.4%] | 0 [0.0%, 8.4%] | 0 | 89.8% | 156610 | 5 m 4 s |
| `qwen36-35b__q4_0` | 42 | 0 [0.0%, 8.4%] | 0 [0.0%, 8.4%] | 0 [0.0%, 8.4%] | 0 | 90.2% | 154787 | 5 m 23 s |

`out tokens (W2, W3, W6)` covers only the workflows that record a completion count; W4 and W5 do not, so their tokens are absent from this column.

The intervals are wide with 42 instances per arm, so these rates describe behaviour and never rank arms.

### Mean wall per instance

| arm | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | — | 2 m 29 s | 4 m 29 s | 20 m 21 s | 22 s | 6 m 45 s |
| `gemma4-26b-a4b__q4_0` | — | 6 m 22 s | 5 m 46 s | 20 m 7 s | 25 s | 6 m 59 s |
| `ornith-35b__f16` | — | 44 s | 1 m 13 s | 9 m 13 s | 27 s | 3 m 13 s |
| `ornith-35b__q4_0` | — | 1 m 16 s | 1 m 31 s | 15 m 41 s | 25 s | 4 m 46 s |
| `qwen36-35b__f16` | — | 1 m 32 s | 3 m 46 s | 20 m 54 s | 47 s | 3 m 12 s |
| `qwen36-35b__q4_0` | — | 1 m 51 s | 3 m 54 s | 22 m 18 s | 42 s | 3 m 11 s |

## Decision rule

An arm is worse if the lower bound of its paired interval excludes 0 and its point estimate is worse than that model's f16 arm; ties otherwise. With 1000 resamples the interval is a percentile interval over instances, and the smallest attainable p at these instance counts is in the premise above. No comparison in this report is a ranking on a behaviour count alone.

## Appendix: per-instance records

The first table summarises each arm's workflows; the second lists only the instances that did not reach full score.

| arm | workflow | instances | score range | halted |
|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | W2 | 12 | 0–0 | 12 |
| `gemma4-26b-a4b__f16` | W3 | 12 | 0–0 | 12 |
| `gemma4-26b-a4b__f16` | W4 | 6 | 0–0 | 5 |
| `gemma4-26b-a4b__f16` | W5 | 6 | 3–3 | 0 |
| `gemma4-26b-a4b__f16` | W6 | 6 | 0–0 | 6 |
| `gemma4-26b-a4b__q4_0` | W2 | 12 | 0–0 | 12 |
| `gemma4-26b-a4b__q4_0` | W3 | 12 | 0–20 | 11 |
| `gemma4-26b-a4b__q4_0` | W4 | 6 | 0–0 | 6 |
| `gemma4-26b-a4b__q4_0` | W5 | 6 | 3–3 | 0 |
| `gemma4-26b-a4b__q4_0` | W6 | 6 | 0–0 | 6 |
| `ornith-35b__f16` | W2 | 12 | 20–20 | 0 |
| `ornith-35b__f16` | W3 | 12 | 20–20 | 0 |
| `ornith-35b__f16` | W4 | 6 | 0–0 | 1 |
| `ornith-35b__f16` | W5 | 6 | 3–3 | 0 |
| `ornith-35b__f16` | W6 | 6 | 0–0 | 6 |
| `ornith-35b__q4_0` | W2 | 12 | 20–20 | 0 |
| `ornith-35b__q4_0` | W3 | 12 | 20–20 | 0 |
| `ornith-35b__q4_0` | W4 | 6 | 0–0 | 2 |
| `ornith-35b__q4_0` | W5 | 6 | 2–2 | 0 |
| `ornith-35b__q4_0` | W6 | 6 | 1–1 | 6 |
| `qwen36-35b__f16` | W2 | 12 | 20–20 | 0 |
| `qwen36-35b__f16` | W3 | 12 | 20–20 | 0 |
| `qwen36-35b__f16` | W4 | 6 | 0–0 | 0 |
| `qwen36-35b__f16` | W5 | 6 | 3–3 | 0 |
| `qwen36-35b__f16` | W6 | 6 | 12–12 | 0 |
| `qwen36-35b__q4_0` | W2 | 12 | 20–20 | 0 |
| `qwen36-35b__q4_0` | W3 | 12 | 20–20 | 0 |
| `qwen36-35b__q4_0` | W4 | 6 | 0–0 | 0 |
| `qwen36-35b__q4_0` | W5 | 6 | 3–3 | 0 |
| `qwen36-35b__q4_0` | W6 | 6 | 12–12 | 0 |

| arm | workflow | instance | score | halt | out tokens | wall |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | W2 | 0 | 0/20 | loop | 10752 | 5 m 26 s |
| `gemma4-26b-a4b__f16` | W2 | 1 | 0/20 | loop | 5888 | 2 m 11 s |
| `gemma4-26b-a4b__f16` | W2 | 2 | 0/20 | loop | 5888 | 2 m 10 s |
| `gemma4-26b-a4b__f16` | W2 | 3 | 0/20 | loop | 5888 | 2 m 10 s |
| `gemma4-26b-a4b__f16` | W2 | 4 | 0/20 | loop | 5888 | 2 m 12 s |
| `gemma4-26b-a4b__f16` | W2 | 5 | 0/20 | loop | 5888 | 2 m 12 s |
| `gemma4-26b-a4b__f16` | W2 | 6 | 0/20 | loop | 5888 | 2 m 12 s |
| `gemma4-26b-a4b__f16` | W2 | 7 | 0/20 | loop | 5888 | 2 m 13 s |
| `gemma4-26b-a4b__f16` | W2 | 8 | 0/20 | loop | 5888 | 2 m 14 s |
| `gemma4-26b-a4b__f16` | W2 | 9 | 0/20 | loop | 5888 | 2 m 14 s |
| `gemma4-26b-a4b__f16` | W2 | 10 | 0/20 | loop | 5888 | 2 m 14 s |
| `gemma4-26b-a4b__f16` | W2 | 11 | 0/20 | loop | 5888 | 2 m 16 s |
| `gemma4-26b-a4b__f16` | W3 | 0 | 0/20 | loop | — | 2 m 7 s |
| `gemma4-26b-a4b__f16` | W3 | 1 | 0/20 | loop | — | 1 m 37 s |
| `gemma4-26b-a4b__f16` | W3 | 2 | 0/20 | loop | — | 7 m 40 s |
| `gemma4-26b-a4b__f16` | W3 | 3 | 0/20 | loop | — | 5 m 45 s |
| `gemma4-26b-a4b__f16` | W3 | 4 | 0/20 | loop | — | 7 m 57 s |
| `gemma4-26b-a4b__f16` | W3 | 5 | 0/20 | loop | — | 4 m 49 s |
| `gemma4-26b-a4b__f16` | W3 | 6 | 0/20 | loop | — | 7 m 16 s |
| `gemma4-26b-a4b__f16` | W3 | 7 | 0/20 | loop | — | 1 m 34 s |
| `gemma4-26b-a4b__f16` | W3 | 8 | 0/20 | loop | — | 4 m 26 s |
| `gemma4-26b-a4b__f16` | W3 | 9 | 0/20 | loop | — | 1 m 36 s |
| `gemma4-26b-a4b__f16` | W3 | 10 | 0/20 | loop | — | 4 m 18 s |
| `gemma4-26b-a4b__f16` | W3 | 11 | 0/20 | loop | — | 4 m 43 s |
| `gemma4-26b-a4b__f16` | W4 | 0 | 0/1 | ceiling | — | 24 m 39 s |
| `gemma4-26b-a4b__f16` | W4 | 1 | 0/1 | ceiling | — | 25 m 50 s |
| `gemma4-26b-a4b__f16` | W4 | 2 | 0/1 | completed | — | 24 m 40 s |
| `gemma4-26b-a4b__f16` | W4 | 3 | 0/1 | ceiling | — | 21 m 32 s |
| `gemma4-26b-a4b__f16` | W4 | 4 | 0/1 | ceiling | — | 22 m 50 s |
| `gemma4-26b-a4b__f16` | W4 | 5 | 0/1 | loop | — | 2 m 33 s |
| `gemma4-26b-a4b__f16` | W6 | 0 | 0/12 | loop | 17152 | 7 m 59 s |
| `gemma4-26b-a4b__f16` | W6 | 1 | 0/12 | loop | 16896 | 6 m 19 s |
| `gemma4-26b-a4b__f16` | W6 | 2 | 0/12 | loop | 16896 | 6 m 19 s |
| `gemma4-26b-a4b__f16` | W6 | 3 | 0/12 | loop | 16896 | 6 m 33 s |
| `gemma4-26b-a4b__f16` | W6 | 4 | 0/12 | loop | 16896 | 6 m 39 s |
| `gemma4-26b-a4b__f16` | W6 | 5 | 0/12 | loop | 16896 | 6 m 40 s |
| `gemma4-26b-a4b__q4_0` | W2 | 0 | 0/20 | loop | 1280 | 2 m 16 s |
| `gemma4-26b-a4b__q4_0` | W2 | 1 | 0/20 | loop | 13056 | 6 m 44 s |
| `gemma4-26b-a4b__q4_0` | W2 | 2 | 0/20 | loop | 13056 | 6 m 49 s |
| `gemma4-26b-a4b__q4_0` | W2 | 3 | 0/20 | loop | 13056 | 6 m 50 s |
| `gemma4-26b-a4b__q4_0` | W2 | 4 | 0/20 | loop | 13056 | 6 m 40 s |
| `gemma4-26b-a4b__q4_0` | W2 | 5 | 0/20 | loop | 13056 | 6 m 39 s |
| `gemma4-26b-a4b__q4_0` | W2 | 6 | 0/20 | loop | 13056 | 6 m 43 s |
| `gemma4-26b-a4b__q4_0` | W2 | 7 | 0/20 | loop | 13056 | 6 m 42 s |
| `gemma4-26b-a4b__q4_0` | W2 | 8 | 0/20 | loop | 13056 | 6 m 42 s |
| `gemma4-26b-a4b__q4_0` | W2 | 9 | 0/20 | loop | 13056 | 6 m 45 s |
| `gemma4-26b-a4b__q4_0` | W2 | 10 | 0/20 | loop | 13056 | 6 m 49 s |
| `gemma4-26b-a4b__q4_0` | W2 | 11 | 0/20 | loop | 13056 | 6 m 50 s |
| `gemma4-26b-a4b__q4_0` | W3 | 1 | 0/20 | loop | — | 5 m 20 s |
| `gemma4-26b-a4b__q4_0` | W3 | 2 | 0/20 | loop | — | 5 m 26 s |
| `gemma4-26b-a4b__q4_0` | W3 | 3 | 0/20 | loop | — | 1 m 10 s |
| `gemma4-26b-a4b__q4_0` | W3 | 4 | 0/20 | loop | — | 5 m 21 s |
| `gemma4-26b-a4b__q4_0` | W3 | 5 | 0/20 | loop | — | 7 m 44 s |
| `gemma4-26b-a4b__q4_0` | W3 | 6 | 0/20 | loop | — | 10 m 14 s |
| `gemma4-26b-a4b__q4_0` | W3 | 7 | 0/20 | loop | — | 7 m 17 s |
| `gemma4-26b-a4b__q4_0` | W3 | 8 | 0/20 | loop | — | 1 m 43 s |
| `gemma4-26b-a4b__q4_0` | W3 | 9 | 0/20 | loop | — | 9 m 19 s |
| `gemma4-26b-a4b__q4_0` | W3 | 10 | 0/20 | loop | — | 2 m 7 s |
| `gemma4-26b-a4b__q4_0` | W3 | 11 | 0/20 | loop | — | 7 m 27 s |
| `gemma4-26b-a4b__q4_0` | W4 | 0 | 0/1 | ceiling | — | 28 m 5 s |
| `gemma4-26b-a4b__q4_0` | W4 | 1 | 0/1 | ceiling | — | 28 m 1 s |
| `gemma4-26b-a4b__q4_0` | W4 | 2 | 0/1 | loop | — | 2 m 15 s |
| `gemma4-26b-a4b__q4_0` | W4 | 3 | 0/1 | ceiling | — | 32 m 22 s |
| `gemma4-26b-a4b__q4_0` | W4 | 4 | 0/1 | loop | — | 1 m 33 s |
| `gemma4-26b-a4b__q4_0` | W4 | 5 | 0/1 | ceiling | — | 28 m 28 s |
| `gemma4-26b-a4b__q4_0` | W6 | 0 | 0/12 | loop | 6400 | 3 m 10 s |
| `gemma4-26b-a4b__q4_0` | W6 | 1 | 0/12 | loop | 15360 | 7 m 45 s |
| `gemma4-26b-a4b__q4_0` | W6 | 2 | 0/12 | loop | 15360 | 7 m 45 s |
| `gemma4-26b-a4b__q4_0` | W6 | 3 | 0/12 | loop | 15360 | 7 m 45 s |
| `gemma4-26b-a4b__q4_0` | W6 | 4 | 0/12 | loop | 15360 | 7 m 45 s |
| `gemma4-26b-a4b__q4_0` | W6 | 5 | 0/12 | loop | 15360 | 7 m 45 s |
| `ornith-35b__f16` | W4 | 0 | 0/1 | completed | — | 9 m 8 s |
| `ornith-35b__f16` | W4 | 1 | 0/1 | completed | — | 9 m 0 s |
| `ornith-35b__f16` | W4 | 2 | 0/1 | completed | — | 9 m 15 s |
| `ornith-35b__f16` | W4 | 3 | 0/1 | loop | — | 7 m 38 s |
| `ornith-35b__f16` | W4 | 4 | 0/1 | completed | — | 11 m 1 s |
| `ornith-35b__f16` | W4 | 5 | 0/1 | completed | — | 9 m 17 s |
| `ornith-35b__f16` | W6 | 0 | 0/12 | context_full | 8055 | 4 m 43 s |
| `ornith-35b__f16` | W6 | 1 | 0/12 | context_full | 8055 | 2 m 54 s |
| `ornith-35b__f16` | W6 | 2 | 0/12 | context_full | 8055 | 2 m 55 s |
| `ornith-35b__f16` | W6 | 3 | 0/12 | context_full | 8055 | 2 m 54 s |
| `ornith-35b__f16` | W6 | 4 | 0/12 | context_full | 8055 | 2 m 56 s |
| `ornith-35b__f16` | W6 | 5 | 0/12 | context_full | 8055 | 2 m 55 s |
| `ornith-35b__q4_0` | W4 | 0 | 0/1 | completed | — | 12 m 50 s |
| `ornith-35b__q4_0` | W4 | 1 | 0/1 | completed | — | 15 m 41 s |
| `ornith-35b__q4_0` | W4 | 2 | 0/1 | loop | — | 14 m 54 s |
| `ornith-35b__q4_0` | W4 | 3 | 0/1 | completed | — | 10 m 59 s |
| `ornith-35b__q4_0` | W4 | 4 | 0/1 | completed | — | 11 m 14 s |
| `ornith-35b__q4_0` | W4 | 5 | 0/1 | ceiling | — | 28 m 28 s |
| `ornith-35b__q4_0` | W5 | 0 | 2/3 | completed | — | 24 s |
| `ornith-35b__q4_0` | W5 | 1 | 2/3 | completed | — | 31 s |
| `ornith-35b__q4_0` | W5 | 2 | 2/3 | completed | — | 24 s |
| `ornith-35b__q4_0` | W5 | 3 | 2/3 | completed | — | 23 s |
| `ornith-35b__q4_0` | W5 | 4 | 2/3 | completed | — | 24 s |
| `ornith-35b__q4_0` | W5 | 5 | 2/3 | completed | — | 23 s |
| `ornith-35b__q4_0` | W6 | 0 | 1/12 | context_full | 8055 | 6 m 26 s |
| `ornith-35b__q4_0` | W6 | 1 | 1/12 | context_full | 8055 | 4 m 20 s |
| `ornith-35b__q4_0` | W6 | 2 | 1/12 | context_full | 8055 | 4 m 18 s |
| `ornith-35b__q4_0` | W6 | 3 | 1/12 | context_full | 8055 | 4 m 13 s |
| `ornith-35b__q4_0` | W6 | 4 | 1/12 | context_full | 8055 | 4 m 15 s |
| `ornith-35b__q4_0` | W6 | 5 | 1/12 | context_full | 8055 | 5 m 2 s |
| `qwen36-35b__f16` | W4 | 0 | 0/1 | completed | — | 29 m 51 s |
| `qwen36-35b__f16` | W4 | 1 | 0/1 | completed | — | 24 m 51 s |
| `qwen36-35b__f16` | W4 | 2 | 0/1 | completed | — | 16 m 10 s |
| `qwen36-35b__f16` | W4 | 3 | 0/1 | completed | — | 18 m 46 s |
| `qwen36-35b__f16` | W4 | 4 | 0/1 | completed | — | 18 m 47 s |
| `qwen36-35b__f16` | W4 | 5 | 0/1 | completed | — | 17 m 2 s |
| `qwen36-35b__q4_0` | W4 | 0 | 0/1 | completed | — | 22 m 40 s |
| `qwen36-35b__q4_0` | W4 | 1 | 0/1 | completed | — | 14 m 50 s |
| `qwen36-35b__q4_0` | W4 | 2 | 0/1 | completed | — | 18 m 48 s |
| `qwen36-35b__q4_0` | W4 | 3 | 0/1 | completed | — | 18 m 51 s |
| `qwen36-35b__q4_0` | W4 | 4 | 0/1 | completed | — | 19 m 33 s |
| `qwen36-35b__q4_0` | W4 | 5 | 0/1 | completed | — | 39 m 3 s |

All other instances scored full marks and completed; their rows are in `raw.json`.

## Provenance

The record below names the exact build, sampler configuration and hashes for this run, so a later run
can be checked against it. The sampling section is the audit: the request sent only `temperature: 0.0`,
and the server's resolved block confirms greedy decoding on every arm.

- llama.cpp `2a667032dc78` · 0.7.0-dev (build 11698) · flags `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`
- sampling sent: `{"temperature": 0.0}`; resolved per arm: `gemma4-26b-a4b__f16` temp=0.000 greedy=True, `gemma4-26b-a4b__q4_0` temp=0.000 greedy=True, `ornith-35b__f16` temp=0.000 greedy=True, `ornith-35b__q4_0` temp=0.000 greedy=True, `qwen36-35b__f16` temp=0.000 greedy=True, `qwen36-35b__q4_0` temp=0.000 greedy=True.
- rungs: `{"gemma4-26b-a4b": 102400, "ornith-35b": 102400, "qwen36-35b": 102400}`
- model sha256: `gemma4-26b-a4b` `d208665ab1cd`, `ornith-35b` `42739874cc2c`, `qwen36-35b` `55983c5a75a1`
- KLD baselines: `gemma4-26b-a4b__f16` `3af73ca491c6` ctx=4000, `gemma4-26b-a4b__q4_0` `3af73ca491c6` ctx=4000, `ornith-35b__f16` `1d340ea536e4` ctx=4000, `ornith-35b__q4_0` `1d340ea536e4` ctx=4000, `qwen36-35b__f16` `afb73afa9879` ctx=4000, `qwen36-35b__q4_0` `afb73afa9879` ctx=4000 (stored outside the results tree).
- document sha256 `11e504b3f33395bc55eae9a1b669a19cc18bf18c0fbf47792b2854aa08fa3cc2`; instances `{"W1": 3, "W2": 12, "W3": 12, "W4": 6, "W5": 6, "W6": 6}`

