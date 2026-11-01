# Appendix — every table, method and provenance

## Appendix — full tables, method and provenance


## Hardware and build this ran on

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16384 MiB total, compute 8.6 |
| driver / CUDA | 615.78.08 / 13.4 |
| CPU | 11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz (16 logical cores) |
| RAM | 62.5 GiB |
| OS / kernel | Arch Linux / 7.2.11-arch1-1 |
| llama.cpp | 2a667032dc78 (clean working tree) |
| build flags | `CMAKE_BUILD_TYPE=Release`, `CMAKE_CUDA_ARCHITECTURES=86`, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_FA_ALL_QUANTS=OFF`, `GGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`, `GGML_CUDA_NO_VMM=OFF`, `GGML_NATIVE=ON` |

Recorded: captured on the same machine, recorded in hardware.json. Per-arm GPU peak and the offload knobs are in the per-model offload table below; those are what the memory budget was spent on.
The long form lives in `REPORT-APPENDIX.md` next to this file; it is the same data, unsorted for reading.

| model | best KV | capability | runner-up | structural loops | decode t/s | why |
|---|---|---|---|---|---|---|
| `qwen38-flash` | **q4_0** | 84.94 | f16 | 0/12 | 12.3 | q4_0 beats its own f16 arm on capability (84.9 vs 83.3); vs f16: R Δ+1 points (p=1.000), C Δ-1 checks (p=1.000); 0/12 structural-loop samples, 12.3 t/s decode |
| `qwen38-27b` | **q8_0** | 83.33 | q4_0 | 0/12 | 10.3 | q8_0 beats its own f16 arm on capability (83.3 vs 75.3); vs f16: R Δ+2 points (p=0.500), C Δ+73 checks (p=0.500); 0/12 structural-loop samples, 10.3 t/s decode |
| `qwen36-35b` | **q4_0** | 82.24 | f16 | 0/12 | 30.8 | q4_0 beats its own f16 arm on capability (82.2 vs 79.0); the arm f16 hit the 6144-token code cap on 1 of 6 tasks, so its checks are truncated output, not wrong code; vs f16: R Δ+2 points (p=1.000), C Δ-2 checks (p=0.750); 0/12 structural-loop samples, 30.8 t/s decode |
| `ornith-35b` | **f16** | 66.79 | q8_0 | 0/12 | 48.8 | f16 keeps the lead; q4_0 loses 6.3 capability points; 0/12 structural-loop samples, 48.8 t/s decode |
| `gemma4-26b-a4b` | **f16** | 62.82 | q8_0 | 3/12 | 46.9 | f16 keeps the lead; q4_0 loses 29.1 capability points; the arm q4_0 hit the 6144-token code cap on 3 of 6 tasks, so its checks are truncated output, not wrong code; 3/12 structural-loop samples, 46.9 t/s decode |

### All 15 arms, ranked by the same rule

| rank | arm | capability | structural loops | decode t/s | p vs #1 | distinguishable from #1? |
|---|---|---|---|---|---|---|
| 1 | `qwen38-flash` **q4_0** | 84.94 | 0/12 | 12.3 | 1.0000 | no — tied |
| 2 | `qwen38-flash` **f16** | 83.33 | 0/12 | 16.6 | 1.0000 | no — tied |
| 3 | `qwen38-27b` **q8_0** | 83.33 | 0/12 | 10.3 | 1.0000 | no — tied |
| 4 | `qwen36-35b` **q4_0** | 82.24 | 0/12 | 30.8 | 1.0000 | no — tied |
| 5 | `qwen38-flash` **q8_0** | 80.38 | 0/12 | 12.6 | 1.0000 | no — tied |
| 6 | `qwen36-35b` **f16** | 79.04 | 0/12 | 38.7 | 1.0000 | no — tied |
| 7 | `qwen36-35b` **q8_0** | 76.15 | 1/12 | 31.2 | 1.0000 | no — tied |
| 8 | `qwen38-27b` **q4_0** | 75.64 | 0/12 | 9.7 | 1.0000 | no — tied |
| 9 | `qwen38-27b` **f16** | 75.32 | 0/12 | 12.4 | 1.0000 | no — tied |
| 10 | `ornith-35b` **f16** | 66.79 | 0/12 | 48.8 | 1.0000 | no — tied |
| 11 | `ornith-35b` **q8_0** | 63.21 | 1/12 | 35.7 | 1.0000 | no — tied |
| 12 | `gemma4-26b-a4b` **f16** | 62.82 | 3/12 | 46.9 | 1.0000 | no — tied |
| 13 | `ornith-35b` **q4_0** | 60.45 | 0/12 | 35.2 | 1.0000 | no — tied |
| 14 | `gemma4-26b-a4b` **q8_0** | 56.60 | 2/12 | 33.8 | 1.0000 | no — tied |
| 15 | `gemma4-26b-a4b` **q4_0** | 33.72 | 3/12 | 34.3 | 1.0000 | no — tied |

The comparison is made **one tier at a time** (R, C, L tested separately, smallest p multiplied by 3 as a Bonferroni correction), because pooling raw R points, C checks and L flags into one sum would let the 261-check samples outweigh the 0/1 flags — a scale artefact, not evidence. The arms that are **not distinguishable from the winner** are: `qwen38-flash` f16 (83.33), `qwen38-27b` q8_0 (83.33), `qwen36-35b` q4_0 (82.24), `qwen38-flash` q8_0 (80.38), `qwen36-35b` f16 (79.04), `qwen36-35b` q8_0 (76.15), `qwen38-27b` q4_0 (75.64), `qwen38-27b` f16 (75.32), `ornith-35b` f16 (66.79), `ornith-35b` q8_0 (63.21), `gemma4-26b-a4b` f16 (62.82), `ornith-35b` q4_0 (60.45), `gemma4-26b-a4b` q8_0 (56.60), `gemma4-26b-a4b` q4_0 (33.72). The 0 that are separable are all *worse* than it: . On this run the lead is therefore not established.

Sensitivity, disclosed rather than hidden: normalising every sample to its own fraction (points ÷ possible, checks ÷ checks) before pooling separates eight arms instead of three. Every alternative pooling scheme that was tried separates only arms that are *worse* than the leader, never a better one, so no reading of this data supports 'quantized KV is an improvement'.

## Per-model offload (knobs are frozen across the three KV arms)

Knobs are identical for a model's three arms; the VRAM peak and speeds below are the **f16 arm's**. Measured peaks differ between a model's arms by up to 3142 MiB, and the ordering is not the same on every model, so the verdict table's numbers are the arm that actually won, not this one (peaks include transient buffers and are not a clean cache-size measurement).

| model | weights off the GPU | VRAM peak MiB (f16) | realized prompt tokens | prefill t/s (f16) | decode t/s (f16) |
|---|---|---|---|---|---|
| `qwen38-flash` | 48 layers on CPU | 11295 | 91357 | 21.5 | 16.6 |
| `qwen38-27b` | 8 layers on CPU | 15899 | 91357 | 17.9 | 12.4 |
| `qwen36-35b` | 26 layers on CPU | 11619 | 91357 | 57.2 | 38.7 |
| `ornith-35b` | 16 layers on CPU | 15468 | 91357 | 67.4 | 48.8 |
| `gemma4-26b-a4b` | 8 layers on CPU | 13757 | 81845 | 34.5 | 46.9 |

## R tier — retrieval / tracing / aggregation (partial credit, %)

| model | f16 | q8_0 | q4_0 |
|---|---|---|---|
| `qwen38-flash` | 76.7 | 80.0 | 80.0 |
| `qwen38-27b` | 70.0 | 76.7 | 70.0 |
| `qwen36-35b` | 60.0 | 63.3 | 66.7 |
| `ornith-35b` | 63.3 | 53.3 | 60.0 |
| `gemma4-26b-a4b` | 36.7 | 23.3 | 26.7 |

## C tier — hidden checks passed / total, and resolved tasks

| model | KV | checks | % | resolved | too-hard tasks |
|---|---|---|---|---|---|
| `qwen38-flash` | f16 | 702/780 | 90.0 | 4/6 | — |
| `qwen38-flash` | q8_0 | 630/780 | 80.8 | 4/6 | — |
| `qwen38-flash` | q4_0 | 701/780 | 89.9 | 3/6 | — |
| `qwen38-27b` | f16 | 629/780 | 80.6 | 3/6 | — |
| `qwen38-27b` | q8_0 | 702/780 | 90.0 | 4/6 | — |
| `qwen38-27b` | q4_0 | 634/780 | 81.3 | 3/6 | — |
| `qwen36-35b` | f16 | 765/780 | 98.1 | 4/6 | — |
| `qwen36-35b` | q8_0 | 694/780 | 89.0 | 3/6 | — |
| `qwen36-35b` | q4_0 | 763/780 | 97.8 | 3/6 | — |
| `ornith-35b` | f16 | 548/780 | 70.3 | 4/6 | — |
| `ornith-35b` | q8_0 | 570/780 | 73.1 | 2/6 | — |
| `ornith-35b` | q4_0 | 475/780 | 60.9 | 3/6 | — |
| `gemma4-26b-a4b` | f16 | 694/780 | 89.0 | 4/6 | — |
| `gemma4-26b-a4b` | q8_0 | 701/780 | 89.9 | 2/6 | — |
| `gemma4-26b-a4b` | q4_0 | 318/780 | 40.8 | 3/6 | — |

## Loop table — flagged samples / samples (reasoning and content split)

`union (L)` counts every verdict from the zlib-ratio heuristic or a structural signal; `hard repetition` counts only samples with a repeated line, n-gram, 200-char block or tail cycle. The gap between the two columns is formulaic prose, not looping.

| model | KV | gen_manual | copy_ledgers | enumerate_shards | tool_episode | union (L) | hard repetition | reasoning | content | any tier |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen38-flash` | f16 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 2 |
| `qwen38-flash` | q8_0 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 2 |
| `qwen38-flash` | q4_0 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 2 |
| `qwen38-27b` | f16 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 4 |
| `qwen38-27b` | q8_0 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 2 |
| `qwen38-27b` | q4_0 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 3 |
| `qwen36-35b` | f16 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 3 |
| `qwen36-35b` | q8_0 | 2/3 | 0/3 | 1/3 | 0/3 | 3/12 | 1/12 | 0 | 3 | 4 |
| `qwen36-35b` | q4_0 | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 | 0/12 | 0 | 0 | 2 |
| `ornith-35b` | f16 | 2/3 | 0/3 | 0/3 | 0/3 | 2/12 | 0/12 | 0 | 2 | 4 |
| `ornith-35b` | q8_0 | 3/3 | 0/3 | 1/3 | 0/3 | 4/12 | 1/12 | 0 | 4 | 4 |
| `ornith-35b` | q4_0 | 3/3 | 0/3 | 0/3 | 0/3 | 3/12 | 0/12 | 0 | 3 | 5 |
| `gemma4-26b-a4b` | f16 | 1/3 | 0/3 | 2/3 | 1/3 | 4/12 | 3/12 | 1 | 3 | 14 |
| `gemma4-26b-a4b` | q8_0 | 0/3 | 0/3 | 2/3 | 0/3 | 2/12 | 2/12 | 0 | 2 | 13 |
| `gemma4-26b-a4b` | q4_0 | 0/3 | 0/3 | 3/3 | 0/3 | 3/12 | 3/12 | 0 | 3 | 16 |

## Paired tests vs f16 (exact sign-flip, paired by task and seed)

| model | KV | metric | level | change | p | direction |
|---|---|---|---|---|---|---|
| `qwen38-flash` | q8_0 | R | 80.0 % of 30 points | +1 points | 1.0000 | higher · not significant |
| `qwen38-flash` | q8_0 | C | 80.8 % of 780 checks | -72 checks | 1.0000 | lower · not significant |
| `qwen38-flash` | q8_0 | L | 0.0 % of 12 samples | +0 samples | 1.0000 | equal · not significant |
| `qwen38-flash` | q4_0 | R | 80.0 % of 30 points | +1 points | 1.0000 | higher · not significant |
| `qwen38-flash` | q4_0 | C | 89.9 % of 780 checks | -1 checks | 1.0000 | lower · not significant |
| `qwen38-flash` | q4_0 | L | 0.0 % of 12 samples | +0 samples | 1.0000 | equal · not significant |
| `qwen38-27b` | q8_0 | R | 76.7 % of 30 points | +2 points | 0.5000 | higher · not significant |
| `qwen38-27b` | q8_0 | C | 90.0 % of 780 checks | +73 checks | 0.5000 | higher · not significant |
| `qwen38-27b` | q8_0 | L | 0.0 % of 12 samples | +0 samples | 1.0000 | equal · not significant |
| `qwen38-27b` | q4_0 | R | 70.0 % of 30 points | +0 points | 1.0000 | equal · not significant |
| `qwen38-27b` | q4_0 | C | 81.3 % of 780 checks | +5 checks | 1.0000 | higher · not significant |
| `qwen38-27b` | q4_0 | L | 0.0 % of 12 samples | +0 samples | 1.0000 | equal · not significant |
| `qwen36-35b` | q8_0 | R | 63.3 % of 30 points | +1 points | 1.0000 | higher · not significant |
| `qwen36-35b` | q8_0 | C | 89.0 % of 780 checks | -71 checks | 0.6250 | lower · not significant |
| `qwen36-35b` | q8_0 | L | 8.3 % of 12 samples | +1 samples | 1.0000 | higher · not significant |
| `qwen36-35b` | q4_0 | R | 66.7 % of 30 points | +2 points | 1.0000 | higher · not significant |
| `qwen36-35b` | q4_0 | C | 97.8 % of 780 checks | -2 checks | 0.7500 | lower · not significant |
| `qwen36-35b` | q4_0 | L | 0.0 % of 12 samples | +0 samples | 1.0000 | equal · not significant |
| `ornith-35b` | q8_0 | R | 53.3 % of 30 points | -3 points | 0.5000 | lower · not significant |
| `ornith-35b` | q8_0 | C | 73.1 % of 780 checks | +22 checks | 1.0000 | higher · not significant |
| `ornith-35b` | q8_0 | L | 8.3 % of 12 samples | +1 samples | 1.0000 | higher · not significant |
| `ornith-35b` | q4_0 | R | 60.0 % of 30 points | -1 points | 1.0000 | lower · not significant |
| `ornith-35b` | q4_0 | C | 60.9 % of 780 checks | -73 checks | 0.5000 | lower · not significant |
| `ornith-35b` | q4_0 | L | 0.0 % of 12 samples | +0 samples | 1.0000 | equal · not significant |
| `gemma4-26b-a4b` | q8_0 | R | 23.3 % of 30 points | -4 points | 0.7500 | lower · not significant |
| `gemma4-26b-a4b` | q8_0 | C | 89.9 % of 780 checks | +7 checks | 0.7500 | higher · not significant |
| `gemma4-26b-a4b` | q8_0 | L | 16.7 % of 12 samples | -1 samples | 1.0000 | lower · not significant |
| `gemma4-26b-a4b` | q4_0 | R | 26.7 % of 30 points | -3 points | 0.7500 | lower · not significant |
| `gemma4-26b-a4b` | q4_0 | C | 40.8 % of 780 checks | -376 checks | 0.2500 | lower · not significant |
| `gemma4-26b-a4b` | q4_0 | L | 25.0 % of 12 samples | +0 samples | 1.0000 | equal · not significant |

α = 0.05. The test is exact — every sign pattern is enumerated, no sampling and no normal approximation — so the same `samples.jsonl` always produces the same numbers. The smallest attainable p is 0.0000076 on 18 R pairs, 0.0312 on 6 C tasks and 0.0005 on 12 loop samples, so an arm must move nearly every paired sample the same way to be called different. 'Not significant' means the change is inside what a draw of this size produces; it is not proof of identity.

## Agreement with f16 per model (share of identical graded answers)

- `qwen38-flash` — q8_0: 0.611 of 18 samples; q4_0: 0.667 of 18 samples
- `qwen38-27b` — q8_0: 0.722 of 18 samples; q4_0: 0.722 of 18 samples
- `qwen36-35b` — q8_0: 0.722 of 18 samples; q4_0: 0.556 of 18 samples
- `ornith-35b` — q8_0: 0.5 of 18 samples; q4_0: 0.706 of 17 samples
- `gemma4-26b-a4b` — q8_0: 0.2 of 5 samples; q4_0: 0.2 of 5 samples

## Cross-model answer agreement (same R task + seed, same KV)


**f16**

| pair | identical / compared | share |
|---|---|---|
| `gemma4-26b-a4b` vs `ornith-35b` | 4/9 | 0.444 |
| `gemma4-26b-a4b` vs `qwen36-35b` | 3/9 | 0.333 |
| `gemma4-26b-a4b` vs `qwen38-27b` | 4/9 | 0.444 |
| `gemma4-26b-a4b` vs `qwen38-flash` | 4/9 | 0.444 |
| `ornith-35b` vs `qwen36-35b` | 13/18 | 0.722 |
| `ornith-35b` vs `qwen38-27b` | 10/18 | 0.556 |
| `ornith-35b` vs `qwen38-flash` | 10/18 | 0.556 |
| `qwen36-35b` vs `qwen38-27b` | 9/18 | 0.5 |
| `qwen36-35b` vs `qwen38-flash` | 9/18 | 0.5 |
| `qwen38-27b` vs `qwen38-flash` | 12/18 | 0.667 |

**q8_0**

| pair | identical / compared | share |
|---|---|---|
| `gemma4-26b-a4b` vs `ornith-35b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen36-35b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen38-27b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen38-flash` | 1/7 | 0.143 |
| `ornith-35b` vs `qwen36-35b` | 8/18 | 0.444 |
| `ornith-35b` vs `qwen38-27b` | 8/18 | 0.444 |
| `ornith-35b` vs `qwen38-flash` | 9/18 | 0.5 |
| `qwen36-35b` vs `qwen38-27b` | 10/18 | 0.556 |
| `qwen36-35b` vs `qwen38-flash` | 9/18 | 0.5 |
| `qwen38-27b` vs `qwen38-flash` | 11/18 | 0.611 |

**q4_0**

| pair | identical / compared | share |
|---|---|---|
| `gemma4-26b-a4b` vs `ornith-35b` | 2/7 | 0.286 |
| `gemma4-26b-a4b` vs `qwen36-35b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen38-27b` | 1/7 | 0.143 |
| `gemma4-26b-a4b` vs `qwen38-flash` | 1/7 | 0.143 |
| `ornith-35b` vs `qwen36-35b` | 10/17 | 0.588 |
| `ornith-35b` vs `qwen38-27b` | 9/17 | 0.529 |
| `ornith-35b` vs `qwen38-flash` | 9/17 | 0.529 |
| `qwen36-35b` vs `qwen38-27b` | 11/18 | 0.611 |
| `qwen36-35b` vs `qwen38-flash` | 11/18 | 0.611 |
| `qwen38-27b` vs `qwen38-flash` | 12/18 | 0.667 |

## Speed and realized context

| model | KV | prompt tokens | prefill t/s | decode t/s | cache hit rate | out tokens |
|---|---|---|---|---|---|---|
| `qwen38-flash` | f16 | 91357 | 21.5 | 16.6 | 1.0 | 43678 |
| `qwen38-flash` | q8_0 | 91357 | 20.4 | 12.6 | 1.0 | 43793 |
| `qwen38-flash` | q4_0 | 91357 | 21.2 | 12.3 | 1.0 | 39776 |
| `qwen38-27b` | f16 | 91357 | 17.9 | 12.4 | 1.0 | 42855 |
| `qwen38-27b` | q8_0 | 91357 | 16.3 | 10.3 | 1.0 | 42139 |
| `qwen38-27b` | q4_0 | 91357 | 15.6 | 9.7 | 1.0 | 43006 |
| `qwen36-35b` | f16 | 91357 | 57.2 | 38.7 | 1.0 | 49725 |
| `qwen36-35b` | q8_0 | 91357 | 50.8 | 31.2 | 1.0 | 48413 |
| `qwen36-35b` | q4_0 | 91357 | 51.6 | 30.8 | 1.0 | 49339 |
| `ornith-35b` | f16 | 91357 | 67.4 | 48.8 | 1.0 | 47391 |
| `ornith-35b` | q8_0 | 91357 | 59.6 | 35.7 | 1.0 | 48370 |
| `ornith-35b` | q4_0 | 91357 | 59.7 | 35.2 | 1.0 | 45415 |
| `gemma4-26b-a4b` | f16 | 81845 | 34.5 | 46.9 | 1.0 | 121371 |
| `gemma4-26b-a4b` | q8_0 | 81845 | 43.6 | 33.8 | 1.0 | 135175 |
| `gemma4-26b-a4b` | q4_0 | 81845 | 51.7 | 34.3 | 1.0 | 153433 |

## How long this took

| phase | wall clock | when (UTC) |
|---|---|---|
| harness self-test + smoke runs | 40 m 00 s | 2026-11-01T00:02 → 2026-11-01T03:15 |
| fit probe (context + offload + depth) | 35 m 29 s | 2026-11-01T00:02 → 2026-11-01T03:50 |
| matrix: 15 arms, 540 requests | 14 h 12 m | 2026-11-01T03:53 → 2026-11-01T18:05 |
| report generation | 1 m 00 s | — → — |
| **total test time (phases summed, provisioning excluded)** | **15 h 27 m** | 2026-11-01T00:02 → 2026-11-01T18:05 |

Summed per-arm time is 13 h 47 m inside a 14 h 12 m matrix window: the difference is model load time and the one-off prefill of the shared document in every arm.

run1 (03:53Z) was stopped by an arm-start failure at 09:19Z; run2 resumed the 9 unfinished arms from their per-arm checkpoints, so no sample was regenerated.

Provenance: fit_probe = sum of wall_s over the five fit invocations (fit.json reports only the last); harness_selftest_and_smoke = three --smoke arm durations plus six selftest runs; matrix = min(arm.started) -> max(arm.finished) over the 15 arm checkpoints; per_arm = wall_s recorded by the orchestrator in each arm checkpoint; study_wall_s = sum of the work phases below (they partially overlapped the excluded downloads).

Per-arm wall clock (includes the document prefill and the server load):

| arm | wall clock | decode t/s | samples |
|---|---|---|---|
| `qwen36-35b__f16` | 27 m 31 s | 38.7 | 36 |
| `qwen36-35b__q8_0` | 31 m 50 s | 31.2 | 36 |
| `qwen36-35b__q4_0` | 32 m 30 s | 30.8 | 36 |
| `qwen38-27b__f16` | 1 h 06 m | 12.4 | 36 |
| `qwen38-27b__q8_0` | 1 h 18 m | 10.3 | 36 |
| `qwen38-27b__q4_0` | 1 h 27 m | 9.7 | 36 |
| `qwen38-flash__f16` | 55 m 26 s | 16.6 | 36 |
| `qwen38-flash__q8_0` | 1 h 09 m | 12.6 | 36 |
| `qwen38-flash__q4_0` | 1 h 05 m | 12.3 | 36 |
| `gemma4-26b-a4b__f16` | 56 m 37 s | 46.9 | 36 |
| `gemma4-26b-a4b__q8_0` | 1 h 24 m | 33.8 | 36 |
| `gemma4-26b-a4b__q4_0` | 1 h 35 m | 34.3 | 36 |
| `ornith-35b__f16` | 21 m 00 s | 48.8 | 36 |
| `ornith-35b__q8_0` | 27 m 33 s | 35.7 | 36 |
| `ornith-35b__q4_0` | 26 m 34 s | 35.2 | 36 |

## Threats to validity (literal)

- Text-identical is not token-identical: one document, five tokenizers; realized prompt lengths are reported and `-c` is identical everywhere.
- Only full-attention layers accumulate KV damage over depth, so a model with few global attention layers can only show a small effect; the per-model layer counts for this run are in fit.json's geometry block.
- The C tier is one greedy pass per task, so it has no error bar; R and L have reps samples each. Any delta smaller than one task's worth of points is noise.
- Hard-task pass rates are confounded with absolute difficulty; tasks with no f16 success are labelled too_hard_for_matrix and excluded from the delta read.
- `repeat_penalty 1.0` in the L tier is more permissive than the production 1.05, so measured loop rates are an upper bound for production settings.
- Duplicated prefill within an arm means a bad early answer cannot contaminate later requests; `tool_episode` is the exception, its transcript is deliberately sequential.
- The repository fixture's third visible test requires `coverage_report` to surface a degenerate point gap, which the hidden check suite does not score; `repo_fix` records both, and `resolved` requires both.
- The `semver` module is bimodal on one prerelease-ordering decision (24/71/83/118/182/187/190/253/261 of 261): a C-tier delta of tens of points can be that single bit, and it is the largest single contributor to C-tier variance here.
- A C-tier arm that hits the generation cap produces truncated code, not wrong code. The report prints `out_tokens`, `finish_reason` and emitted line count per sample so the two are distinguishable; `budget_raised` marks every sample that was re-run at a larger budget.
- The zlib-ratio loop threshold fires on templated text: 25 of 45 `gen_manual` samples cross 0.42 with no line/ngram/cycle signal, uniformly across arms. Real repetition is the line-run, n-gram, block and cycle counters.
