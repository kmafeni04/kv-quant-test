# KV-cache quantization at long context: 1 models × 3 cache types

One context for every arm: **-c 16384** · 1 arms · 16 samples · llama.cpp `2a667032d`. Seeds, budgets and the document hash are in the appendix.

Total test wall clock (fit probe + matrix + harness validation, model downloads excluded): **—** (13 m 12 s of it the matrix).


## Verdict

**Best overall: `qwen36-35b` with a `q4_0` KV cache** — capability 81.22 of 100 (R 70.00% and C 92.44%, averaged before rounding), 0/4 structural-loop samples, 45.6 t/s decode, 9541 MiB of graphics memory with 26 layers of the model kept in system RAM. **No other arm is comparable on this result, so the ranking is descriptive only.**

Ranking rule: capability = mean(retrieval %, code %); ties to the arm with fewer repeating samples, then the faster one. Every paired test clusters the three reps of a task into one observation and then applies a Holm correction across all 42 arm x tier comparisons, so nothing here can separate on a handful of correlated samples.

### Every arm, one line each

| arm | capability | retrieval | code | structural loops | decode t/s |
|---|---|---|---|---|---|
| `qwen36-35b` q4_0 | 81.2 | 70.0 % | 92.4 % | 0/4 | 45.6 |

### What this does not show

- **No arm is statistically separable.** Each tier has 6 task clusters (4 loop probes), so the smallest attainable p is 0.031; correcting across the 42 arm x tier comparisons puts every adjusted p at 1.00. The test can exclude a large, consistent difference and nothing smaller. 'No difference' here means 'not resolvable at this sample size'.
- **Gemma's exception is a length/format failure, not wrong code.** At q4_0 it hit the 6,144-token cap on 3 of 6 coding tasks; on `repo_fix` it wrote 20,459 tokens with no `### FILE:` block at all. Its retrieval failures are content-side repetition loops, not truncated thinking.
- **Two tasks carry most of the code score.** `semver` is 261 of 780 checks and moves on a single decision; `intervals` and `statistics` are saturated in 14 of 15 arms. `count_proj` is degenerate (every project has the same count, so guessing 3 scores).
- **The loop probes only see the first ~2,000 tokens**, so a loop that starts later is invisible; `enumerate_shards` hit that cap in most arms. `csvlite`'s fixtures contain literal `\n` text, so newline handling was not actually tested.
