# statistics.py

Implement the module described below. Pure standard library only (`math` is allowed).

Every function takes a sequence of numbers. Where the spec says "population", the divisor is
`n`; where it says "sample", the divisor is `n - 1`. All results are floats.

| function | behaviour |
|---|---|
| `mean(xs)` | arithmetic mean |
| `median(xs)` | middle value after sorting; for an even count, the mean of the two middle values |
| `variance(xs, sample=False)` | population variance by default, sample variance when `sample=True` |
| `stdev(xs, sample=False)` | square root of the matching variance |
| `percentile(xs, p)` | the **linear-interpolation-between-closest-ranks** method (numpy's default): sort, let `k = (n-1) * p/100`, and interpolate between the values at `floor(k)` and `ceil(k)`. `p` is in `0..100`. |
| `mode(xs)` | the most frequent value; **on a tie return the smallest such value** |
| `normalize(xs)` | z-scores, `(x - mean) / stdev`, using the **population** stdev |
| `covariance(xs, ys, sample=False)` | population by default; `sample=True` uses `n - 1` |
| `correlation(xs, ys)` | `covariance(xs, ys) / (stdev(xs) * stdev(ys))` |

Rules:
- An empty input raises `ValueError`.
- `sample=True` with fewer than 2 values raises `ValueError`.
- Mismatched lengths in `covariance`/`correlation` raise `ValueError`.
- `percentile` with `p` outside `0..100` raises `ValueError`.

Return **only** the complete file contents in a single ```python code block, nothing else.
