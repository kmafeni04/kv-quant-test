# statistics.py

Implement the statistics module described below, using the Python standard library only (`math` is
allowed). Do not import numpy, and do not write the numerical shortcuts it provides.

Every function takes a sequence of numbers and returns a float. Where the description says
"population", the divisor is `n`; where it says "sample", the divisor is `n - 1`. Each function states
which one it uses by default and how to ask for the other.

- `mean(xs)` is the arithmetic mean.
- `median(xs)` is the middle value after sorting; with an even count it is the mean of the two middle
  values.
- `variance(xs, sample=False)` is the population variance by default and the sample variance when
  `sample=True`.
- `stdev(xs, sample=False)` is the square root of the matching variance, so the flag means the same
  thing here as it does in `variance`.
- `percentile(xs, p)` uses the linear-interpolation-between-closest-ranks method, which is numpy's
  default: sort the values, set `k = (n-1) * p/100`, and interpolate between the values at `floor(k)`
  and `ceil(k)`. `p` is in `0..100`.
- `mode(xs)` is the most frequent value, and on a tie it returns the smallest of the tied values.
- `normalize(xs)` returns z-scores computed as `(x - mean) / stdev`, using the **population** stdev
  so the divisor does not depend on a flag.
- `covariance(xs, ys, sample=False)` is the population covariance by default, and `sample=True` makes
  it divide by `n - 1`.
- `correlation(xs, ys)` is `covariance(xs, ys) / (stdev(xs) * stdev(ys))`.

Input rules, all of which raise `ValueError` rather than returning a sentinel:

- an empty input;
- `sample=True` with fewer than two values, since `n - 1` would be zero or negative;
- mismatched lengths in `covariance` or `correlation`;
- `p` outside `0..100` in `percentile`.

Return **only** the complete file contents in a single ```python code block, nothing else.
