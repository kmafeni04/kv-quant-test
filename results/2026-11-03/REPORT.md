# Full-context KV quantization: agentic workflows, zero–64 key scores

Compressing the key/value (KV) cache, the stored attention state for every token the model has already read, from sixteen bits per value (f16) to four (q4_0) moves what all three models predict, and it moves Gemma most. Against the same model's f16 run, q4_0 produced a mean Kullback–Leibler divergence (KLD, the average distance between the two next-token distributions) of 0.1868 on gemma4-26b-a4b, 0.0276 on ornith-35b and 0.0100 on qwen36-35b, and the two runs named the same token first in 81.5%, 90.6% and 90.9% of positions. Perplexity (PPL, the exponentiated average negative log-likelihood) at 65,536 tokens moved +4.5%, +2.7% and -0.2%.

None of that turned into a lost key. Where a model answered at all, both cache types scored 64 of 64 on the retrieval probe and kept all twenty brief identifiers on the compaction task: ornith-35b and qwen36-35b, twelve instances each in both arms. gemma4-26b-a4b answered neither probe, looping in reasoning on all twelve instances of both arms, which is counted as a halt and not as a lost key. No arm reconciled the ledger (0 of 36 attempts), and qwen36-35b wrote all six manual sections in both arms.

One workflow did separate the arms. In agentic search, ornith-35b q4_0 answered 12 of 18 keys where its f16 arm answered 18 of 18 and made nine searches against six, so it skipped a hop; that is the only task-level loss this study can attribute to the cache. Against each model's own f16 arm, q4_0 is worse on none of the two paired retrieval arms and none of the two compaction arms and ties on the rest, and across the six arms 65 instances were halted by the loop detector and 17 by a full context.

## Premise, power and disclosure

Each arm runs twelve instances of W2 (64-key retrieval) and W3 (compaction), and six each of W4 (the ledger), W5 (agentic search) and W6 (long-form writing). Decode was one greedy draw per instance (`temperature: 0.0`, no other sampling key, no seeds, no reps); greedy decode can be more or less loop-prone than a sampled setting, so every loop count here describes greedy decoding. No reasoning budget and no retry path applied, and the only cap was the 65,536-token response ceiling, recorded as `ceiling`.

A workflow instance, not a request, is the unit of analysis, so the smallest attainable two-sided p is 0.0005 for W2 and W3 and 0.0312 for W4 to W6. Pairwise tests are the paired difference against that model's f16 arm, a cluster bootstrap over instances (1000 resamples, seeded), and an arm counts as worse only when the interval excludes 0 and the point estimate is worse. The document was 312,347 characters, sha256 `ef61e342dc7f5245`, and tokenized to 84,279 tokens under gemma4-26b-a4b and 94,222 under both other models.

## Cells run

| arm | wall |
|---|---|
| `gemma4-26b-a4b__f16` | 3 h 26 m |
| `gemma4-26b-a4b__q4_0` | 3 h 25 m |
| `ornith-35b__f16` | 2 h 15 m |
| `ornith-35b__q4_0` | 2 h 40 m |
| `qwen36-35b__f16` | 2 h 50 m |
| `qwen36-35b__q4_0` | 3 h 15 m |

## W1: KLD and perplexity (the fast instrument)

| model | KV | mean KLD ± unc | same top p | ln PPL ratio | deep PPL | deep PPL / f16 |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | 0 ± 0 | 100.0% | 0.00000 | 170.350 | 1.0000 |
| `gemma4-26b-a4b` | `q4_0` | 0.1868 ± 0.0074 | 81.5% | 0.04886 | 178.019 | 1.0450 |
| `ornith-35b` | `f16` | 0 ± 0 | 100.0% | 0.00000 | 18.722 | 1.0000 |
| `ornith-35b` | `q4_0` | 0.0276 ± 0.0017 | 90.6% | -0.02017 | 19.224 | 1.0268 |
| `qwen36-35b` | `f16` | 0 ± 0 | 100.0% | 0.00000 | 14.644 | 1.0000 |
| `qwen36-35b` | `q4_0` | 0.0100 ± 0.0005 | 90.9% | -0.00121 | 14.611 | 0.9978 |

The ln PPL ratio is each arm's own scored PPL divided by that model's f16 arm's PPL on the identical slice. On an identical-model check the pinned tool printed `Mean PPL(Q)` equal to the baseline run's own PPL while its `Mean PPL(base)` reconstruction read 26% lower with mean KLD 0 and same-top 100%, so the tool's own base-PPL ratio is not used as evidence; its KLD, Δp and same-top outputs are.

Perplexity is comparable only within a model, never across models. A zero mean KLD with a tight interval on every arm means the instrument did not reach the cache: check that `-ctk`/`-ctv` were passed and that the baseline slice matches token for token.

## W2: 64-key retrieval

| model | KV | mean /64 | sd | min | max | scored | halts | q4_0 − f16 | interval | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | — | — | — | — | 0/12 | 12 | — | [—, —] | no paired instances |
| `gemma4-26b-a4b` | `q4_0` | — | — | — | — | 0/12 | 12 | — | [—, —] | no paired instances |
| `ornith-35b` | `f16` | 64.00 | 0.00 | 64 | 64 | 12/12 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |
| `ornith-35b` | `q4_0` | 64.00 | 0.00 | 64 | 64 | 12/12 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |
| `qwen36-35b` | `f16` | 64.00 | 0.00 | 64 | 64 | 12/12 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |
| `qwen36-35b` | `q4_0` | 64.00 | 0.00 | 64 | 64 | 12/12 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |

A scored instance produced an `ANSWER:` line; instances halted by the loop detector, the window or the ceiling are counted in `halts` and excluded from the score.

### Hit rate by planted depth

Every scored arm hit all keys in all five depth bands; arms with no scored instance have empty cells instead.

**UNSCORED f16.** Every `gemma4-26b-a4b__f16` instance was halted before it answered, so no key was scored as lost, but the probe has no reference data for that model.

**SATURATED.** Every scored arm holds all 64 keys. Per the stop rules the probe is at its ceiling and the study does not widen it again: a longer answer line would test the answer format, not the cache. The saturation is reported as a finding.

## W3: compaction

The task is a handover brief of at most 1,500 words carrying twenty shard identifiers, the ledger rows a named tenant's mutations touch, and five invariants; the document and the phase-A transcript are then discarded and five questions answerable only from the brief are put to the model. The table reports how many of the twenty identifiers survived, how many came back wrong (`mutated`, including the near-miss decoy) or missing (`absent`), and how many of the five phase-B keys were correct.

| model | KV | retention /20 | sd | scored | halts | mutated | absent | phase-B keys /5 | briefs within cap | paired vs f16 | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | — | — | 0/12 | 12 | 0 | 0 | — | 0/0 | — [—, —] | no paired instances |
| `gemma4-26b-a4b` | `q4_0` | — | — | 0/12 | 12 | 0 | 0 | — | 0/0 | — [—, —] | no paired instances |
| `ornith-35b` | `f16` | 20.00 | 0.00 | 12/12 | 0 | 0 | 0 | 3.00 | 12/12 | 0.00 [0.00, 0.00] | tie (interval includes 0) |
| `ornith-35b` | `q4_0` | 20.00 | 0.00 | 12/12 | 0 | 0 | 0 | 3.17 | 12/12 | 0.00 [0.00, 0.00] | tie (interval includes 0) |
| `qwen36-35b` | `f16` | 20.00 | 0.00 | 12/12 | 0 | 0 | 0 | 3.00 | 12/12 | 0.00 [0.00, 0.00] | tie (interval includes 0) |
| `qwen36-35b` | `q4_0` | 20.00 | 0.00 | 12/12 | 0 | 0 | 0 | 3.00 | 12/12 | 0.00 [0.00, 0.00] | tie (interval includes 0) |

### Brief retention by planted depth

Every scored brief kept all twenty identifiers in all five depth bands; arms with no scored brief have empty cells instead.

## W4: long-horizon ledger

Identical on all 6 arms: 0/6 reached the expected final hash, 0/6 satisfied all five invariants, and 0 double and 0 foreign mutations were applied.

## W5: agentic search

Sixty fact clusters form fifteen four-hop chains, and each instance asks three questions that need three searches each, so nine calls is the floor. The table gives answer keys correct out of 18, a 95% Wilson interval on that proportion, the searches made, and how many instances reached the floor; a model that answers without searching can still score, but its call count shows the shortcut.

| model | KV | answer keys | interval | search calls | instances at/above floor | instances |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |
| `gemma4-26b-a4b` | `q4_0` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |
| `ornith-35b` | `f16` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |
| `ornith-35b` | `q4_0` | 12/18 | [43.7%, 83.7%] | 6–6 | 0/6 | 6 |
| `qwen36-35b` | `f16` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |
| `qwen36-35b` | `q4_0` | 18/18 | [82.4%, 100.0%] | 9–9 | 6/6 | 6 |

## W6: unbounded long-form

Each instance asks for a six-section operator's manual with given titles, a given cross-reference mapping and 120–220 words per section. The table counts headings present, cross-references that resolve to the required section, instances that got both fully right, and loop halts.

| model | KV | sections present | cross-references resolved | fully correct | loop halts |
|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | 0/36 | 0/36 | 0/6 | 5 |
| `gemma4-26b-a4b` | `q4_0` | 0/36 | 0/36 | 0/6 | 6 |
| `ornith-35b` | `f16` | 0/36 | 0/36 | 0/6 | 0 |
| `ornith-35b` | `q4_0` | 0/36 | 0/36 | 0/6 | 0 |
| `qwen36-35b` | `f16` | 36/36 | 36/36 | 6/6 | 0 |
| `qwen36-35b` | `q4_0` | 36/36 | 36/36 | 6/6 | 0 |

## Behaviour, halts and cost

| arm | completed | loop [95%] | context full [95%] | ceiling [95%] | error | thinking share | out tokens | mean wall / instance |
|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | 9 | 29 [54.0%, 80.9%] | 3 [2.5%, 19.0%] | 1 [0.4%, 12.3%] | 0 | 100.0% | 531583 | 5 m 5 s |
| `gemma4-26b-a4b__q4_0` | 6 | 32 [61.5%, 86.5%] | 1 [0.4%, 12.3%] | 3 [2.5%, 19.0%] | 0 | 100.0% | 422636 | 4 m 51 s |
| `ornith-35b__f16` | 33 | 1 [0.4%, 12.3%] | 6 [6.7%, 27.8%] | 2 [1.3%, 15.8%] | 0 | 85.5% | 360393 | 3 m 17 s |
| `ornith-35b__q4_0` | 33 | 1 [0.4%, 12.3%] | 7 [8.3%, 30.6%] | 1 [0.4%, 12.3%] | 0 | 79.4% | 336656 | 3 m 46 s |
| `qwen36-35b__f16` | 41 | 1 [0.4%, 12.3%] | 0 [0.0%, 8.4%] | 0 [0.0%, 8.4%] | 0 | 86.1% | 361814 | 4 m 0 s |
| `qwen36-35b__q4_0` | 41 | 1 [0.4%, 12.3%] | 0 [0.0%, 8.4%] | 0 [0.0%, 8.4%] | 0 | 89.0% | 334561 | 4 m 29 s |

The intervals are wide with this many instances, so the rates describe behaviour and never rank arms.

Loop detection runs on `reasoning_content` and `content` separately for single responses and per step on tool transcripts (identical consecutive call, an identical call three times, A↔B oscillation, a no-progress window). `context_full` with zero visible tokens is a recorded outcome, never retried.

### Mean wall per instance

Mean wall clock per instance, server load included. W1 is the dash on every arm: it runs `llama-perplexity` once per arm, timed in the W1 record.

| arm | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | — | 1 m 56 s | 5 m 45 s | 13 m 24 s | 22 s | 6 m 26 s |
| `gemma4-26b-a4b__q4_0` | — | 1 m 11 s | 4 m 34 s | 15 m 51 s | 42 s | 5 m 50 s |
| `ornith-35b__f16` | — | 1 m 29 s | 1 m 2 s | 14 m 7 s | 31 s | 3 m 20 s |
| `ornith-35b__q4_0` | — | 1 m 57 s | 1 m 29 s | 15 m 9 s | 21 s | 4 m 1 s |
| `qwen36-35b__f16` | — | 2 m 46 s | 3 m 13 s | 12 m 53 s | 38 s | 2 m 28 s |
| `qwen36-35b__q4_0` | — | 3 m 53 s | 3 m 36 s | 12 m 40 s | 39 s | 3 m 6 s |

## Decision rule

With 1000 resamples the interval is a percentile interval over instances, and the smallest attainable p at these instance counts is in the premise above. No comparison in this report is a ranking on a behaviour count alone.

## Appendix: per-instance records

The first table summarises each arm's workflows; the second lists only instances that missed a full score, so a miss can be traced without paging rows that all report full marks.

| arm | workflow | instances | score range | halted |
|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | W2 | 12 | 0–0 | 12 |
| `gemma4-26b-a4b__f16` | W3 | 12 | 0–0 | 12 |
| `gemma4-26b-a4b__f16` | W4 | 6 | 0–0 | 3 |
| `gemma4-26b-a4b__f16` | W5 | 6 | 3–3 | 0 |
| `gemma4-26b-a4b__f16` | W6 | 6 | 0–0 | 6 |
| `gemma4-26b-a4b__q4_0` | W2 | 12 | 0–0 | 12 |
| `gemma4-26b-a4b__q4_0` | W3 | 12 | 0–0 | 12 |
| `gemma4-26b-a4b__q4_0` | W4 | 6 | 0–0 | 6 |
| `gemma4-26b-a4b__q4_0` | W5 | 6 | 3–3 | 0 |
| `gemma4-26b-a4b__q4_0` | W6 | 6 | 0–0 | 6 |
| `ornith-35b__f16` | W2 | 12 | 64–64 | 0 |
| `ornith-35b__f16` | W3 | 12 | 20–20 | 0 |
| `ornith-35b__f16` | W4 | 6 | 0–0 | 3 |
| `ornith-35b__f16` | W5 | 6 | 3–3 | 0 |
| `ornith-35b__f16` | W6 | 6 | 0–0 | 6 |
| `ornith-35b__q4_0` | W2 | 12 | 64–64 | 0 |
| `ornith-35b__q4_0` | W3 | 12 | 20–20 | 0 |
| `ornith-35b__q4_0` | W4 | 6 | 0–0 | 3 |
| `ornith-35b__q4_0` | W5 | 6 | 2–2 | 0 |
| `ornith-35b__q4_0` | W6 | 6 | 0–0 | 6 |
| `qwen36-35b__f16` | W2 | 12 | 64–64 | 0 |
| `qwen36-35b__f16` | W3 | 12 | 20–20 | 0 |
| `qwen36-35b__f16` | W4 | 6 | 0–0 | 1 |
| `qwen36-35b__f16` | W5 | 6 | 3–3 | 0 |
| `qwen36-35b__f16` | W6 | 6 | 12–12 | 0 |
| `qwen36-35b__q4_0` | W2 | 12 | 64–64 | 0 |
| `qwen36-35b__q4_0` | W3 | 12 | 20–20 | 0 |
| `qwen36-35b__q4_0` | W4 | 6 | 0–0 | 1 |
| `qwen36-35b__q4_0` | W5 | 6 | 3–3 | 0 |
| `qwen36-35b__q4_0` | W6 | 6 | 12–12 | 0 |

| arm | workflow | instance | score | halt | out tokens | wall |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | W2 | 0 | 0/64 | loop | 3584 | 3 m 20 s |
| `gemma4-26b-a4b__f16` | W2 | 1 | 0/64 | loop | 3584 | 3 m 4 s |
| `gemma4-26b-a4b__f16` | W2 | 2 | 0/64 | loop | 4096 | 1 m 51 s |
| `gemma4-26b-a4b__f16` | W2 | 3 | 0/64 | loop | 4096 | 1 m 48 s |
| `gemma4-26b-a4b__f16` | W2 | 4 | 0/64 | loop | 4096 | 1 m 33 s |
| `gemma4-26b-a4b__f16` | W2 | 5 | 0/64 | loop | 4096 | 1 m 40 s |
| `gemma4-26b-a4b__f16` | W2 | 6 | 0/64 | loop | 4096 | 1 m 39 s |
| `gemma4-26b-a4b__f16` | W2 | 7 | 0/64 | loop | 4096 | 1 m 42 s |
| `gemma4-26b-a4b__f16` | W2 | 8 | 0/64 | loop | 4096 | 1 m 36 s |
| `gemma4-26b-a4b__f16` | W2 | 9 | 0/64 | loop | 4096 | 1 m 38 s |
| `gemma4-26b-a4b__f16` | W2 | 10 | 0/64 | loop | 4096 | 1 m 38 s |
| `gemma4-26b-a4b__f16` | W2 | 11 | 0/64 | loop | 4096 | 1 m 42 s |
| `gemma4-26b-a4b__f16` | W3 | 0 | 0/20 | loop | 15104 | 6 m 7 s |
| `gemma4-26b-a4b__f16` | W3 | 1 | 0/20 | loop | 9216 | 3 m 10 s |
| `gemma4-26b-a4b__f16` | W3 | 2 | 0/20 | context_full | 22696 | 8 m 39 s |
| `gemma4-26b-a4b__f16` | W3 | 3 | 0/20 | loop | 15360 | 6 m 7 s |
| `gemma4-26b-a4b__f16` | W3 | 4 | 0/20 | loop | 16640 | 6 m 18 s |
| `gemma4-26b-a4b__f16` | W3 | 5 | 0/20 | loop | 19968 | 7 m 22 s |
| `gemma4-26b-a4b__f16` | W3 | 6 | 0/20 | context_full | 22696 | 8 m 27 s |
| `gemma4-26b-a4b__f16` | W3 | 7 | 0/20 | loop | 5888 | 2 m 25 s |
| `gemma4-26b-a4b__f16` | W3 | 8 | 0/20 | loop | 5888 | 1 m 57 s |
| `gemma4-26b-a4b__f16` | W3 | 9 | 0/20 | loop | 5888 | 1 m 58 s |
| `gemma4-26b-a4b__f16` | W3 | 10 | 0/20 | loop | 13568 | 6 m 2 s |
| `gemma4-26b-a4b__f16` | W3 | 11 | 0/20 | loop | 19968 | 10 m 34 s |
| `gemma4-26b-a4b__f16` | W4 | 0 | 0/1 | completed | 8304 | 4 m 10 s |
| `gemma4-26b-a4b__f16` | W4 | 1 | 0/1 | completed | 63912 | 26 m 7 s |
| `gemma4-26b-a4b__f16` | W4 | 2 | 0/1 | loop | 578 | 1 m 6 s |
| `gemma4-26b-a4b__f16` | W4 | 3 | 0/1 | loop | 5654 | 3 m 0 s |
| `gemma4-26b-a4b__f16` | W4 | 4 | 0/1 | completed | 58251 | 22 m 17 s |
| `gemma4-26b-a4b__f16` | W4 | 5 | 0/1 | ceiling | 67520 | 23 m 45 s |
| `gemma4-26b-a4b__f16` | W6 | 0 | 0/12 | context_full | 17940 | 8 m 14 s |
| `gemma4-26b-a4b__f16` | W6 | 1 | 0/12 | loop | 16384 | 6 m 4 s |
| `gemma4-26b-a4b__f16` | W6 | 2 | 0/12 | loop | 16384 | 6 m 4 s |
| `gemma4-26b-a4b__f16` | W6 | 3 | 0/12 | loop | 16384 | 6 m 4 s |
| `gemma4-26b-a4b__f16` | W6 | 4 | 0/12 | loop | 16384 | 6 m 4 s |
| `gemma4-26b-a4b__f16` | W6 | 5 | 0/12 | loop | 16384 | 6 m 4 s |
| `gemma4-26b-a4b__q4_0` | W2 | 0 | 0/64 | loop | 5632 | 4 m 21 s |
| `gemma4-26b-a4b__q4_0` | W2 | 1 | 0/64 | loop | 1792 | 53 s |
| `gemma4-26b-a4b__q4_0` | W2 | 2 | 0/64 | loop | 1792 | 53 s |
| `gemma4-26b-a4b__q4_0` | W2 | 3 | 0/64 | loop | 1792 | 53 s |
| `gemma4-26b-a4b__q4_0` | W2 | 4 | 0/64 | loop | 1792 | 54 s |
| `gemma4-26b-a4b__q4_0` | W2 | 5 | 0/64 | loop | 1792 | 54 s |
| `gemma4-26b-a4b__q4_0` | W2 | 6 | 0/64 | loop | 1792 | 54 s |
| `gemma4-26b-a4b__q4_0` | W2 | 7 | 0/64 | loop | 1792 | 54 s |
| `gemma4-26b-a4b__q4_0` | W2 | 8 | 0/64 | loop | 1792 | 54 s |
| `gemma4-26b-a4b__q4_0` | W2 | 9 | 0/64 | loop | 1792 | 55 s |
| `gemma4-26b-a4b__q4_0` | W2 | 10 | 0/64 | loop | 1792 | 55 s |
| `gemma4-26b-a4b__q4_0` | W2 | 11 | 0/64 | loop | 1792 | 55 s |
| `gemma4-26b-a4b__q4_0` | W3 | 0 | 0/20 | loop | 15104 | 7 m 16 s |
| `gemma4-26b-a4b__q4_0` | W3 | 1 | 0/20 | loop | 12288 | 6 m 14 s |
| `gemma4-26b-a4b__q4_0` | W3 | 2 | 0/20 | loop | 7168 | 3 m 16 s |
| `gemma4-26b-a4b__q4_0` | W3 | 3 | 0/20 | loop | 8192 | 3 m 50 s |
| `gemma4-26b-a4b__q4_0` | W3 | 4 | 0/20 | loop | 8448 | 3 m 58 s |
| `gemma4-26b-a4b__q4_0` | W3 | 5 | 0/20 | loop | 4096 | 1 m 36 s |
| `gemma4-26b-a4b__q4_0` | W3 | 6 | 0/20 | context_full | 20648 | 10 m 50 s |
| `gemma4-26b-a4b__q4_0` | W3 | 7 | 0/20 | loop | 4096 | 1 m 33 s |
| `gemma4-26b-a4b__q4_0` | W3 | 8 | 0/20 | loop | 5888 | 2 m 56 s |
| `gemma4-26b-a4b__q4_0` | W3 | 9 | 0/20 | loop | 17664 | 9 m 0 s |
| `gemma4-26b-a4b__q4_0` | W3 | 10 | 0/20 | loop | 5376 | 2 m 43 s |
| `gemma4-26b-a4b__q4_0` | W3 | 11 | 0/20 | loop | 4096 | 1 m 38 s |
| `gemma4-26b-a4b__q4_0` | W4 | 0 | 0/1 | ceiling | 65987 | 29 m 24 s |
| `gemma4-26b-a4b__q4_0` | W4 | 1 | 0/1 | loop | 1375 | 1 m 22 s |
| `gemma4-26b-a4b__q4_0` | W4 | 2 | 0/1 | loop | 545 | 1 m 6 s |
| `gemma4-26b-a4b__q4_0` | W4 | 3 | 0/1 | ceiling | 66704 | 31 m 0 s |
| `gemma4-26b-a4b__q4_0` | W4 | 4 | 0/1 | loop | 1012 | 1 m 15 s |
| `gemma4-26b-a4b__q4_0` | W4 | 5 | 0/1 | ceiling | 66969 | 30 m 59 s |
| `gemma4-26b-a4b__q4_0` | W6 | 0 | 0/12 | loop | 8448 | 4 m 12 s |
| `gemma4-26b-a4b__q4_0` | W6 | 1 | 0/12 | loop | 12032 | 6 m 4 s |
| `gemma4-26b-a4b__q4_0` | W6 | 2 | 0/12 | loop | 12032 | 6 m 17 s |
| `gemma4-26b-a4b__q4_0` | W6 | 3 | 0/12 | loop | 12032 | 6 m 22 s |
| `gemma4-26b-a4b__q4_0` | W6 | 4 | 0/12 | loop | 12032 | 6 m 4 s |
| `gemma4-26b-a4b__q4_0` | W6 | 5 | 0/12 | loop | 12032 | 6 m 4 s |
| `ornith-35b__f16` | W4 | 0 | 0/1 | completed | 22062 | 9 m 55 s |
| `ornith-35b__f16` | W4 | 1 | 0/1 | ceiling | 66221 | 20 m 55 s |
| `ornith-35b__f16` | W4 | 2 | 0/1 | completed | 37748 | 15 m 29 s |
| `ornith-35b__f16` | W4 | 3 | 0/1 | loop | 3352 | 2 m 11 s |
| `ornith-35b__f16` | W4 | 4 | 0/1 | completed | 31979 | 12 m 30 s |
| `ornith-35b__f16` | W4 | 5 | 0/1 | ceiling | 66465 | 23 m 40 s |
| `ornith-35b__f16` | W6 | 0 | 0/12 | context_full | 8003 | 5 m 10 s |
| `ornith-35b__f16` | W6 | 1 | 0/12 | context_full | 8003 | 3 m 1 s |
| `ornith-35b__f16` | W6 | 2 | 0/12 | context_full | 8003 | 2 m 58 s |
| `ornith-35b__f16` | W6 | 3 | 0/12 | context_full | 8003 | 2 m 59 s |
| `ornith-35b__f16` | W6 | 4 | 0/12 | context_full | 8003 | 2 m 56 s |
| `ornith-35b__f16` | W6 | 5 | 0/12 | context_full | 8003 | 2 m 56 s |
| `ornith-35b__q4_0` | W4 | 0 | 0/1 | loop | 3702 | 2 m 13 s |
| `ornith-35b__q4_0` | W4 | 1 | 0/1 | ceiling | 67149 | 28 m 49 s |
| `ornith-35b__q4_0` | W4 | 2 | 0/1 | completed | 34707 | 17 m 13 s |
| `ornith-35b__q4_0` | W4 | 3 | 0/1 | context_full | 10395 | 6 m 24 s |
| `ornith-35b__q4_0` | W4 | 4 | 0/1 | completed | 63838 | 28 m 12 s |
| `ornith-35b__q4_0` | W4 | 5 | 0/1 | completed | 13470 | 8 m 2 s |
| `ornith-35b__q4_0` | W5 | 0 | 2/3 | completed | 947 | 22 s |
| `ornith-35b__q4_0` | W5 | 1 | 2/3 | completed | 947 | 20 s |
| `ornith-35b__q4_0` | W5 | 2 | 2/3 | completed | 947 | 20 s |
| `ornith-35b__q4_0` | W5 | 3 | 2/3 | completed | 947 | 20 s |
| `ornith-35b__q4_0` | W5 | 4 | 2/3 | completed | 947 | 20 s |
| `ornith-35b__q4_0` | W5 | 5 | 2/3 | completed | 947 | 21 s |
| `ornith-35b__q4_0` | W6 | 0 | 0/12 | context_full | 8003 | 4 m 2 s |
| `ornith-35b__q4_0` | W6 | 1 | 0/12 | context_full | 8003 | 4 m 1 s |
| `ornith-35b__q4_0` | W6 | 2 | 0/12 | context_full | 8003 | 4 m 1 s |
| `ornith-35b__q4_0` | W6 | 3 | 0/12 | context_full | 8003 | 4 m 1 s |
| `ornith-35b__q4_0` | W6 | 4 | 0/12 | context_full | 8003 | 4 m 1 s |
| `ornith-35b__q4_0` | W6 | 5 | 0/12 | context_full | 8003 | 4 m 1 s |
| `qwen36-35b__f16` | W4 | 0 | 0/1 | loop | 7235 | 4 m 0 s |
| `qwen36-35b__f16` | W4 | 1 | 0/1 | completed | 27133 | 13 m 24 s |
| `qwen36-35b__f16` | W4 | 2 | 0/1 | completed | 27698 | 13 m 33 s |
| `qwen36-35b__f16` | W4 | 3 | 0/1 | completed | 34520 | 16 m 35 s |
| `qwen36-35b__f16` | W4 | 4 | 0/1 | completed | 33771 | 16 m 7 s |
| `qwen36-35b__f16` | W4 | 5 | 0/1 | completed | 27758 | 13 m 40 s |
| `qwen36-35b__q4_0` | W4 | 0 | 0/1 | loop | 587 | 21 s |
| `qwen36-35b__q4_0` | W4 | 1 | 0/1 | completed | 24484 | 14 m 38 s |
| `qwen36-35b__q4_0` | W4 | 2 | 0/1 | completed | 26513 | 15 m 42 s |
| `qwen36-35b__q4_0` | W4 | 3 | 0/1 | completed | 23857 | 14 m 23 s |
| `qwen36-35b__q4_0` | W4 | 4 | 0/1 | completed | 27883 | 16 m 18 s |
| `qwen36-35b__q4_0` | W4 | 5 | 0/1 | completed | 24340 | 14 m 36 s |

All other instances scored full marks and completed; their rows are in `raw.json`.


## Provenance

Everything ran on llama.cpp `2a667032dc78` (0.7.0-dev, build 11698), built with flash attention for the `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16` cache pairs. Each request carried `{"temperature": 0.0}` and nothing else; the server's resolved sampler reads temp=0.000 and greedy=True on four arms and records no resolved sampler on `ornith-35b__f16` and `qwen36-35b__q4_0`, where the verbose line did not parse. All three models ran at a 102,400-token context, `ornith-35b`'s weights hash to `42739874cc2c`, and each model keeps one f16 logits baseline at a 4,000-token context outside the results tree (`3a7626a66402` gemma4-26b-a4b, `aa5abdfef653` ornith-35b, `eae4369b1270` qwen36-35b, shared by that model's two arms). The shared document hashes to `ef61e342dc7f52453b53def29de60b0955c7840fab58212c8d034fd61f751e8c`, and instances were `{"W1": 3, "W2": 12, "W3": 12, "W4": 6, "W5": 6, "W6": 6}`.

