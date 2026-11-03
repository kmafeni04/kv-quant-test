# kv-quant-test: what a compressed KV cache costs at full context

A language model working on a long document keeps its attention state in the KV cache: a stored copy
of every earlier token's keys and values, so it does not have to re-read the document for each new
word. That cache can be compressed to save memory. This study measures what the compression costs
when the model actually has to work at the edge of its context window.

The earlier study in `docs/RESULTS.md` answered short questions at a shrinking context with a
generation cap, a reasoning budget and sampler settings imposed by the harness. It could not see the
failure modes that matter at depth. This harness replaces it: one full-context rung per model, greedy
decode, no reasoning budget, and six workflows whose success is graded from counts, hashes and
structure rather than from prose. The design document is
[docs/AGENTIC-BENCH-PLAN.md](docs/AGENTIC-BENCH-PLAN.md).

## Design

The table fixes what changes between arms and what is held constant. Only the cache type differs
inside a model; the context rung, the document and the decode settings are the same for both arms.

| factor | levels |
|---|---|
| models | Gemma-4-26B-A4B, Ornith-1.5-35B-A3B, Qwen3.6-35B-A3B (the dense 27B and Flash-Next stay configured but are not in the trenched matrix) |
| KV cache | **f16** (uncompressed) and **q4_0** (4-bit), symmetric, with flash attention on |
| context | one rung per model: the largest ladder entry it loads with a ≥ 91k-token prompt, `102400` for all three here. All arms of a model run at that same rung. |
| document | one canonical text, 92–95k tokens in the largest of the three tokenizers, shared byte-for-byte by every arm |
| decode | greedy: `temperature: 0.0` and nothing else; one draw per instance, no seeds, no reps |
| cap | none per request except a 64k response ceiling, recorded as `halted_by=ceiling` |

In the model names, A3B and A4B mean the model activates roughly 3 or 4 billion parameters per token; the number before it is the total parameter count.

Six workflows, each with a fixed instance count per arm:

| id | workflow | instances | guard | what is graded |
|---|---|---|---|---|
| W1 | KLD and perplexity | 3 passes | — | mean KLD (Kullback–Leibler divergence) against the f16 logits, same-top agreement, deep-context PPL (perplexity) ratio |
| W2 | 20-key retrieval | 12 | 1 request | codes correct out of 20, per key, in order; hit rate by planted depth |
| W3 | compaction | 12 | 2 requests | how many of twenty brief facts survive, answers from the brief alone, loop detection over both responses |
| W4 | long-horizon ledger | 6 | 120 tool calls | final ledger hash and five invariants |
| W5 | agentic search | 6 | 60 tool calls | three multi-hop answer keys, search calls against a floor |
| W6 | unbounded long-form | 6 | 64k ceiling | six sections present, six cross-references resolved, loop instrument |

W1's KLD is the Kullback–Leibler divergence, a measure of how far the arm's next-token
distribution has moved from the f16 arm's, and PPL is perplexity, the exponentiated average
negative log-likelihood. The `guard` column is the hard stop for a workflow: one request for W2, two
for W3, a tool-call count for W4 and W5, and only the response ceiling for W6.

W2 and W3 are counts, not verdicts: a sample that loses one deep key scores 19, and the report shows
which depth the miss sat at and whether it was `absent`, `mutated` or `misordered`. W1 is the fast
instrument: it compares the quantized arm's next-token distribution with the f16 arm's own logits, so
it produces a graded number with an error bar in minutes.

The tranches are fixed in advance — T1 Gemma all six workflows, T2 Ornith all six, T3 Qwen3.6 the
counted probes, T4 Qwen3.6 the long workflows — so a deadline cut lands on the least informative
remaining cell and never on the deciding one. Cut cells are printed in the report, not hidden.

## Quickstart

Needs Python 3.11+ (stdlib only), a built llama.cpp with `GGML_CUDA_FA_QUANTS` covering
`f16-f16;q4_0-q4_0`, and the GGUF files (GGUF is the model-file format llama.cpp loads). No network
after the models are downloaded. The commands below move from a fast no-GPU check to the overnight
run; the fit step must come before either run because it resolves the rung each model can hold.

```bash
python3 -m kvtest.selftest                 # ~3 s, no GPU, no models: "selftest OK"
python3 -m kvtest.fit --stamp STAMP        # resolves one rung per model and the 92–95k document
python3 -m kvtest.run --stamp STAMP --pilot   # micro-pilot: Gemma f16, W1 + one W2 + one W3
python3 -m kvtest.report results/STAMP     # regenerate the tables, HTML, CSV and figures
```

The full night is `./scripts/run_all.sh`, which runs the preflight, the fit probe, the four tranches
under a 24 h governor, and the report. Individual steps and the exact commands are in
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Headline

Each row is one arm: mean W2 key score out of 20, mean W3 brief retention out of 20, mean KLD
against that model's f16 logits, and loop and context-full halt counts.
`mean (scored/attempted)` shows how many instances produced a score; `—` means none did, so the mean
is absent rather than zero, and the W2/W3 columns are not comparable across models when halt counts
differ.

<!-- HEADLINE:START -->
| model | KV | W2 mean /20 | W3 retention /20 | mean KLD | loop halts | context full |
|---|---|---|---|---|---|---|
| `gemma4-26b-a4b` | f16 | — (0/12) | — (0/12) | 0 | 31 | 0 |
| `gemma4-26b-a4b` | q4_0 | — (0/12) | 20.00 (1/12) | 0.2052 | 31 | 0 |
| `ornith-35b` | f16 | 20.00 | 20.00 | 0 | 1 | 6 |
| `ornith-35b` | q4_0 | 20.00 | 20.00 | 0.0313 | 1 | 6 |
| `qwen36-35b` | f16 | 20.00 | 20.00 | 0 | 0 | 0 |
| `qwen36-35b` | q4_0 | 20.00 | 20.00 | 0.0095 | 0 | 0 |
<!-- HEADLINE:END -->

## What is here

The tree lists the source modules and the generated directories so a reader can tell the harness from
its outputs. `kvtest/` is the only code; everything under `results/` is produced by a run.

```
kvtest/        harness: config, server lifecycle, client (streaming + tool calls), document
               builder, six workflow modules, tool runner with loop detection, KLD/PPL runner,
               statistics, fit probe, orchestrator, report, selftest
scripts/       preflight, build, download, run_all
docs/          AGENTIC-BENCH-PLAN.md (design), METHOD.md (method + threats),
               REPRODUCE.md (steps), RESULTS.md (the prior study, archived)
results/       <UTC-stamp>/{fit.json,document.txt,samples.jsonl,arms/,logs/,REPORT.md,report.html}
models.lock.json   pinned GGUF sha256 + llama.cpp commit + build flags + document hash
```

## Reading a result

- The experiment is the f16 → q4_0 difference **inside one model at one rung**, never a race between
  models. The same text tokenizes differently on each tokenizer, and only full-attention layers
  accumulate cache damage with depth.
- Perplexity is comparable only within a model's own arms.
- Every table in `REPORT.md` shows the f16 arm. The decision rule is pre-registered: an arm is worse
  only when the paired bootstrap interval over instances excludes 0 and the point estimate is worse.

Greedy decode is not what a user would run in production, and it can be more or less loop-prone than
a sampled decode. Every loop count in the report is therefore a statement about greedy decoding. That
trade is what buys enough instances for a paired interval at all.
