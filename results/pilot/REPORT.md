# Full-context KV quantization: a micro-pilot on Gemma at 84k tokens

Because the pilot ran only the f16 arm, it cannot say how the cache affected the results. It does show that the distribution instrument works: the f16 arm against its own logits read mean KLD (Kullback–Leibler divergence) 0 and same-top agreement 100%, and the deep pass produced perplexity 177.16 at 65,536 tokens. The 20-key and compaction workflows produced no scored answer because Gemma looped in reasoning and was halted in both, so the pilot proves the halt detector works but gives no retrieval score. The cache comparison is in the full-matrix report.

Its job was to prove three workflows end to end at the full 102,400-token rung before the study spent a night on the matrix: the distributional instrument (W1), the 20-key retrieval probe (W2) and compaction (W3). One arm ran: Gemma-4-26B-A4B with an uncompressed 16-bit key/value cache (f16), over the canonical 313,025-character document. That document tokenizes to 84,113 tokens under Gemma's tokenizer. The KV cache is the stored attention state for every token the model has already read. The pilot's outcome is that the instruments execute and the halt detector fires on real degeneration.

## Premise, power and disclosure

- Instances per arm: W2 n=1, W3 n=1.
- One greedy draw per instance (`temperature: 0.0`, no other sampling key, no seeds, no reps). Greedy decode is not a production setting and can be more or less loop-prone than a sampled one; every loop count below is a statement about greedy decoding.
- No reasoning budget and no retry path. The only cap is the 65536-token response ceiling, recorded as `ceiling`.
- Unit = workflow instance. Smallest attainable two-sided p: W2 1.0000, W3 1.0000.
- Pairwise tests are the paired difference against that model's f16 arm, cluster bootstrap over instances (1000 resamples, seeded). An arm is worse only when the interval excludes 0 and the point estimate is worse.
- Document 313,025 characters, sha256 `11e504b3f33395bc`; tokens per model: gemma4-26b-a4b 84,113; ornith-35b 94,170; qwen36-35b 94,170.

## Cells run

| arm | wall |
|---|---|
| `gemma4-26b-a4b__f16` | 3 m 52 s |

## W1: KLD and perplexity (the fast instrument)

| model | KV | mean KLD ± unc | same top p | ln PPL ratio | deep PPL | deep PPL / f16 |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | 0 ± 0 | 100.0% | 0.00000 | 177.164 | 1.0000 |

The ln PPL ratio is each arm's own scored PPL divided by that model's f16 arm's PPL on the identical slice. On an identical-model check the pinned tool printed `Mean PPL(Q)` equal to the baseline run's own PPL while its `Mean PPL(base)` reconstruction read 26% lower with mean KLD 0 and same-top 100%, so the tool's own base-PPL ratio is not used as evidence; its KLD, Δp and same-top outputs are.

Perplexity is comparable only within a model, never across models. A zero mean KLD with a tight interval on every arm means the instrument did not reach the cache: check that `-ctk`/`-ctv` were passed and that the baseline slice matches token for token.

## W2: 20-key retrieval

| model | KV | mean /20 | sd | min | max | scored | halts | q4_0 − f16 | interval | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | — | — | — | — | 0/1 | 1 | — | [—, —] | no paired instances |

A scored instance is one that produced an `ANSWER:` line. Instances halted by the loop detector, the window or the ceiling are counted in `halts` and excluded from the score, per the rule that a halt is its own counter and is never folded into a score.

### Hit rate by planted depth

No arm scored a key, so no depth band has an observation.


**UNSCORED f16.** `gemma4-26b-a4b__f16` produced no scored W2 instance: every instance was halted by the loop detector, the window or the ceiling. Halts are their own counter, so no key was scored as lost, but the probe has no reference data for that model.

## W3: compaction

| model | KV | retention /20 | sd | scored | halts | mutated | absent | phase-B keys /5 | briefs within cap | paired vs f16 | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | `f16` | — | — | 0/1 | 1 | 0 | 0 | — | 0/0 | — [—, —] | no paired instances |

### Brief retention by planted depth

No arm produced a scored brief, so no depth band has an observation.


The primary endpoint is brief retention: a count out of 20 measured on a full-depth read. Phase B is reported alongside it as the consequence, not as a substitute.

## W4: long-horizon ledger

No arm ran W4 in this stamp.

## W5: agentic search

No arm ran W5 in this stamp.

## W6: unbounded long-form

No arm ran W6 in this stamp.

## Behaviour, halts and cost

| arm | completed | loop [95%] | context full [95%] | ceiling [95%] | error | thinking share | out tokens | mean wall / instance |
|---|---|---|---|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | 0 | 2 [34.2%, 100.0%] | 0 [0.0%, 65.8%] | 0 [0.0%, 65.8%] | 0 | 100.0% | 18157 | 4 m 35 s |

The intervals are wide with this many instances, so the rates describe behaviour and never rank arms.

Loop detection runs on `reasoning_content` and `content` separately for single responses and per step on tool transcripts (identical consecutive call, an identical call three times, A↔B oscillation, a no-progress window). `context_full` with zero visible tokens is a recorded outcome, never retried.

### Mean wall per instance

| arm | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | — | 5 m 23 s | 3 m 47 s | — | — | — |

## Decision rule

An arm is worse if the lower bound of its paired interval excludes 0 and its point estimate is worse than that model's f16 arm; ties otherwise. With 1000 resamples the interval is a percentile interval over instances, and the smallest attainable p at these instance counts is in the premise above. No comparison in this report is a ranking on a behaviour count alone.

## Appendix: per-instance records

The first table summarises each arm's workflows; the second lists only the instances that did not reach a full score, so a miss can be traced without paging through rows that all say `20` or `complete`.

| arm | workflow | instances | score range | halted |
|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | W2 | 1 | 0–0 | 1 |
| `gemma4-26b-a4b__f16` | W3 | 1 | 0–0 | 1 |

| arm | workflow | instance | score | halt | out tokens | wall |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b__f16` | W2 | 0 | 0/20 | loop | 10752 | 5 m 23 s |
| `gemma4-26b-a4b__f16` | W3 | 0 | 0/20 | loop | — | 3 m 47 s |

All other instances scored full marks and completed; their rows are in `raw.json`.


## What the pilot establishes

It establishes that the harness can prefill the full document, run the KLD instrument and the deep PPL pass, stream and grade a 20-key response, run both compaction phases, and stop degenerate output at a recorded token position rather than letting it consume the whole window. It also establishes that sample counts, timings, halts and the resolved sampler configuration land in the checkpoint. What it cannot show is whether compression helps or hurts: there is no q4_0 arm, and whether this model can answer W2 and W3 at this depth needs the multi-instance arms.

## Provenance

- llama.cpp `2a667032dc78` · 0.7.0-dev (build 11698) · flags `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`
- sampling sent: `{"temperature": 0.0}`; resolved per arm: `gemma4-26b-a4b__f16` temp=0.000 greedy=True.
- rungs: `{"gemma4-26b-a4b": 102400}`
- model sha256: `gemma4-26b-a4b` `d208665ab1cd`
- KLD baselines: `gemma4-26b-a4b__f16` `3af73ca491c6` ctx=4000 (stored outside the results tree).
- document sha256 `11e504b3f33395bc55eae9a1b669a19cc18bf18c0fbf47792b2854aa08fa3cc2`; instances `{"W1": 1, "W2": 1, "W3": 1}`
