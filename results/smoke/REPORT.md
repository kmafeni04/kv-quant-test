# Full-context KV quantization: a smoke run of the 20-key retrieval probe

Both Ornith arms answered the one 20-key instance on this short control document and scored 20/20, so the probe is answerable and the per-key grader works. One instance per arm cannot show whether f16 or q4_0 is better, and the probe saturated at its twenty-key ceiling. This run skipped the distribution instrument (W1) and the compaction (W3), ledger (W4), search (W5) and long-form (W6) workflows. It therefore says nothing about how compression affects deep retrieval or the other five workflows.

The run answered one question before the full matrix started. Can the harness put a 20-key retrieval task in front of a model at 32k, grade the answer per key, and record a clean halt or a loop? Two arms of Ornith-1.5-35B-A3B ran the task once each: one with an uncompressed 16-bit key/value cache (f16), one with a 4-bit cache (q4_0). A key/value cache, the KV cache, is the stored attention state for every token the model has already read. Compressing it is the treatment this study measures. A single instance on a short control document cannot tell the two cache types apart.

## Premise, power and disclosure

- Instances per arm: W2 n=1.
- One greedy draw per instance (`temperature: 0.0`, no other sampling key, no seeds, no reps). Greedy decode is not a production setting and can be more or less loop-prone than a sampled one; every loop count below is a statement about greedy decoding.
- No reasoning budget and no retry path. The only cap is the 65536-token response ceiling, recorded as `ceiling`.
- Unit = workflow instance. Smallest attainable two-sided p: W2 1.0000.
- Pairwise tests are the paired difference against that model's f16 arm, cluster bootstrap over instances (1000 resamples, seeded). An arm is worse only when the interval excludes 0 and the point estimate is worse.
- Document 42,837 characters, sha256 `4f9468d0ba794767`; tokens per model: not measured for this stamp.

## Cells run

| arm | wall |
|---|---|
| `ornith-35b__q4_0` | 50 s |

## W1: KLD and perplexity (the fast instrument)

No arm ran W1 in this stamp.

## W2: 20-key retrieval

| model | KV | mean /20 | sd | min | max | scored | halts | q4_0 − f16 | interval | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| `ornith-35b` | `f16` | 20.00 | 0.00 | 20 | 20 | 1/1 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |
| `ornith-35b` | `q4_0` | 20.00 | 0.00 | 20 | 20 | 1/1 | 0 | 0.00 | [0.00, 0.00] | tie (interval includes 0) |

A scored instance is one that produced an `ANSWER:` line. Instances halted by the loop detector, the window or the ceiling are counted in `halts` and excluded from the score, per the rule that a halt is its own counter and is never folded into a score.

### Hit rate by planted depth

Every scored arm hit all keys in all five depth bands; arms with no scored instance have empty cells instead.


**SATURATED.** Every scored arm holds all twenty keys. Per the stop rules the probe widens to all forty shards before the second model runs; otherwise it is measuring format compliance rather than cache damage.

## W3: compaction

No arm ran W3 in this stamp.

## W4: long-horizon ledger

No arm ran W4 in this stamp.

## W5: agentic search

No arm ran W5 in this stamp.

## W6: unbounded long-form

No arm ran W6 in this stamp.

## Behaviour, halts and cost

| arm | completed | loop [95%] | context full [95%] | ceiling [95%] | error | thinking share | out tokens | mean wall / instance |
|---|---|---|---|---|---|---|---|---|
| `ornith-35b__f16` | 1 | 0 [0.0%, 79.3%] | 0 [0.0%, 79.3%] | 0 [0.0%, 79.3%] | 0 | 83.6% | 2122 | 49 s |
| `ornith-35b__q4_0` | 1 | 0 [0.0%, 79.3%] | 0 [0.0%, 79.3%] | 0 [0.0%, 79.3%] | 0 | 80.2% | 1758 | 46 s |

The intervals are wide with this many instances, so the rates describe behaviour and never rank arms.

Loop detection runs on `reasoning_content` and `content` separately for single responses and per step on tool transcripts (identical consecutive call, an identical call three times, A↔B oscillation, a no-progress window). `context_full` with zero visible tokens is a recorded outcome, never retried.

### Mean wall per instance

| arm | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| `ornith-35b__f16` | — | 49 s | — | — | — | — |
| `ornith-35b__q4_0` | — | 46 s | — | — | — | — |

## Decision rule

An arm is worse if the lower bound of its paired interval excludes 0 and its point estimate is worse than that model's f16 arm; ties otherwise. With 1000 resamples the interval is a percentile interval over instances, and the smallest attainable p at these instance counts is in the premise above. No comparison in this report is a ranking on a behaviour count alone.

## Appendix: per-instance records

The first table summarises each arm's workflows; the second lists only the instances that did not reach a full score, so a miss can be traced without paging through rows that all say `20` or `complete`.

| arm | workflow | instances | score range | halted |
|---|---|---|---|---|
| `ornith-35b__f16` | W2 | 1 | 20–20 | 0 |
| `ornith-35b__q4_0` | W2 | 1 | 20–20 | 0 |

## Provenance

- llama.cpp `2a667032dc78` · 0.7.0-dev (build 11698) · flags `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`
- sampling sent: `{"temperature": 0.0}`; resolved per arm: `ornith-35b__f16` temp=0.000 greedy=True, `ornith-35b__q4_0` temp=0.000 greedy=True.
- rungs: `{"ornith-35b": 32768}`
- model sha256: `ornith-35b` `42739874cc2c`
- KLD baselines: `ornith-35b__f16` `` ctx=None, `ornith-35b__q4_0` `` ctx=None (stored outside the results tree).
- document sha256 `4f9468d0ba794767849ed5d41f0a077c08a00f0b8cdf753de26f9de7eafe9908`; instances `{"W2": 1}`
