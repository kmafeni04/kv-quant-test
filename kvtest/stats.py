"""Statistics for counted probes: Wilson intervals, paired cluster bootstrap, decision rule.

One draw per instance (greedy decode), so the instance is the cluster. W2 and W3 are counts out
of 20, which is why the paired difference is a mean of per-instance key differences with a
bootstrap interval rather than a test over samples.
"""
import math
import random

from . import config


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    if len(xs) < 2:
        return 0.0 if xs else None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def wilson(hits, n, z=1.96):
    """Wilson score interval for a proportion; returns (p, lo, hi)."""
    if not n:
        return None, None, None
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _percentile(sorted_xs, q):
    if not sorted_xs:
        return None
    i = q * (len(sorted_xs) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] * (hi - i) + sorted_xs[hi] * (i - lo)


def bootstrap_mean(xs, resamples=None, seed=None, alpha=0.05):
    """Percentile bootstrap interval for the mean."""
    resamples = resamples or config.BOOTSTRAP_RESAMPLES
    rng = random.Random(config.BOOTSTRAP_SEED if seed is None else seed)
    if not xs:
        return dict(mean=None, lo=None, hi=None, n=0)
    means = sorted(mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(resamples))
    return dict(mean=mean(xs), lo=_percentile(means, alpha / 2),
                hi=_percentile(means, 1 - alpha / 2), n=len(xs))


def paired_bootstrap(diffs, resamples=None, seed=None, alpha=0.05):
    """Paired cluster bootstrap over instances: diffs[i] is (arm - f16) for instance i."""
    resamples = resamples or config.BOOTSTRAP_RESAMPLES
    rng = random.Random(config.BOOTSTRAP_SEED if seed is None else seed)
    if not diffs:
        return dict(n=0, mean=None, lo=None, hi=None, worse=False, better=False)
    stats = sorted(mean([diffs[rng.randrange(len(diffs))] for _ in diffs])
                   for _ in range(resamples))
    lo = _percentile(stats, alpha / 2)
    hi = _percentile(stats, 1 - alpha / 2)
    m = mean(diffs)
    # Pre-registered rule: worse if the interval excludes 0 AND the point estimate is worse.
    return dict(n=len(diffs), mean=m, lo=lo, hi=hi,
                worse=hi < 0 and m < 0, better=lo > 0 and m > 0)


def paired_diff(a, b):
    """Per-instance a - b, dropping instances missing from either side."""
    return [x - y for x, y in zip(a, b) if x is not None and y is not None]


def decision(paired):
    if paired.get("n", 0) == 0:
        return "no paired instances"
    if paired["worse"]:
        return "worse than f16"
    if paired["better"]:
        return "better than f16"
    return "tie (interval includes 0)"


def smallest_p(n):
    """Smallest attainable two-sided sign-flip p with n instances."""
    return 2.0 / (2 ** n) if n else None
