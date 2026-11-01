# KV-cache quantization at long context — 5 models × f16 / q8_0 / q4_0

One context for every arm: **-c 102400** · 15 arms · 540 samples · llama.cpp `2a667032d`. Seeds, budgets and the document hash are in the appendix.

Total test wall clock (fit probe + matrix + harness validation, model downloads excluded): **15 h 27 m** (14 h 12 m of it the matrix).


## Verdict

No model scored differently from its own f16 arm by more than this test can resolve. gemma4-26b-a4b is the exception: at q4_0 its code score was 41 of 100 against 89 at f16, and it ran out of room on 3 of 6 coding tasks, so it was cut off rather than wrong. On this build the quantized cache was also 17–28 % slower to answer with than f16, so it buys memory, not speed. No cache setting made a model repeat itself more often by a difference this test can resolve.

**Best overall: `qwen38-flash` with a `q4_0` KV cache** — capability 84.94 of 100 (R 80.00% and C 89.87%, averaged before rounding), 0/12 structural-loop samples, 12.3 t/s decode, 9107 MiB of graphics memory with 48 layers of the model kept in system RAM. **Its lead is not established: 14 other arms, including its own f16 arm, cannot be distinguished from it (task-clustered pairs, Holm over 42 comparisons).**

The next arms are `qwen38-flash` f16 (83.33), `qwen38-27b` q8_0 (83.33) — the top three sit within 1.6 capability points of each other, which is one or two graded outcomes.

Ranking rule: capability = mean(retrieval %, code %); ties to the arm with fewer repeating samples, then the faster one. Every paired test clusters the three reps of a task into one observation and then applies a Holm correction across all 42 arm x tier comparisons, so nothing here can separate on a handful of correlated samples.

### Every arm, one line each

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

### What this does not show

- **No arm is statistically separable.** Each tier has 6 task clusters (4 loop probes), so the smallest attainable p is 0.031; correcting across the 42 arm x tier comparisons puts every adjusted p at 1.00. The test can exclude a large, consistent difference and nothing smaller. 'No difference' here means 'not resolvable at this sample size'.
- **Gemma's exception is a length/format failure, not wrong code.** At q4_0 it hit the 6,144-token cap on 3 of 6 coding tasks; on `repo_fix` it wrote 20,459 tokens with no `### FILE:` block at all. Its retrieval failures are content-side repetition loops, not truncated thinking.
- **Two tasks carry most of the code score.** `semver` is 261 of 780 checks and moves on a single decision; `intervals` and `statistics` are saturated in 14 of 15 arms. `count_proj` is degenerate (every project has the same count, so guessing 3 scores).
- **The loop probes only see the first ~2,000 tokens**, so a loop that starts later is invisible; `enumerate_shards` hit that cap in most arms. `csvlite`'s fixtures contain literal `\n` text, so newline handling was not actually tested.
