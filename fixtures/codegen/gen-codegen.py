#!/usr/bin/env python3
"""Generate fixtures/codegen/<module>/{README.md,checks.json} - the p6 coding-quality test.

WHY THIS EXISTS
    p1h grades tool calling on 94 binary cases, so its 1-sd error is ~2.5 points and a 3-5
    point quant difference is invisible. Resolving that with more binary puzzles needs ~1000
    items per arm. This instead asks for ONE long-form implementation per module and grades it
    with 40-70 hidden checks, so a single generation yields hundreds of graded outcomes at
    ~1-2% resolution.

    The modules are chosen to stress what quantisation actually damages: float precision
    (statistics), exact decimal arithmetic (fixedpoint), long spec adherence (semver),
    byte-exact string handling (csvlite), and multi-branch logic (intervals).

HOW IT STAYS FAIR
    Expected values are computed by RUNNING the reference implementation here, at generation
    time, so the checks cannot drift from the spec. Each reference is then asserted to pass
    its own checks, so a broken reference fails the build rather than the models.

    The spec shown to the model is the README; the checks are hidden. Everything the checks
    require is stated in the README - including the exact rounding rule, the interpolation
    method, and the tie-breaks, because those are the things a fair test cannot leave implied.

Run:  python3 gen-codegen.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ============================================================ reference implementations

REF_STATS = '''
import math


def _f(xs):
    xs = [float(x) for x in xs]
    if not xs:
        raise ValueError("empty input")
    return xs


def mean(xs):
    xs = _f(xs)
    return sum(xs) / len(xs)


def median(xs):
    xs = sorted(_f(xs))
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def variance(xs, sample=False):
    xs = _f(xs)
    n = len(xs)
    if sample and n < 2:
        raise ValueError("sample variance needs n >= 2")
    m = sum(xs) / n
    denom = n - 1 if sample else n
    return sum((x - m) ** 2 for x in xs) / denom


def stdev(xs, sample=False):
    return math.sqrt(variance(xs, sample))


def percentile(xs, p):
    """Linear interpolation between closest ranks (the numpy default)."""
    xs = sorted(_f(xs))
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be in 0..100")
    n = len(xs)
    if n == 1:
        return xs[0]
    k = (n - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mode(xs):
    """Most frequent value; on a tie return the SMALLEST such value."""
    xs = _f(xs)
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    return min(k for k, v in counts.items() if v == best)


def normalize(xs):
    """Z-scores using the POPULATION stdev."""
    xs = _f(xs)
    m = mean(xs)
    s = stdev(xs)
    return [(x - m) / s for x in xs]


def covariance(xs, ys, sample=False):
    xs, ys = _f(xs), _f(ys)
    if len(xs) != len(ys):
        raise ValueError("lengths differ")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = n - 1 if sample else n
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / denom


def correlation(xs, ys):
    return covariance(xs, ys) / (stdev(xs) * stdev(ys))
'''

REF_FIXED = '''
SCALE = 10000
DP = 4


def parse(text):
    """Exact decimal string -> scaled integer. Rejects malformed input."""
    s = str(text).strip()
    if not s:
        raise ValueError("empty")
    neg = s.startswith("-")
    if s[0] in "+-":
        s = s[1:]
    if s.count(".") > 1:
        raise ValueError("bad number")
    whole, _, frac = s.partition(".")
    if not whole and not frac:
        raise ValueError("bad number")
    if whole and not whole.isdigit():
        raise ValueError("bad number")
    if frac and not frac.isdigit():
        raise ValueError("bad number")
    if len(frac) > DP:
        raise ValueError("more than 4 decimal places")
    v = int(whole or "0") * SCALE + int((frac + "0" * DP)[:DP] or "0")
    return -v if neg else v


def format(value):
    """Scaled integer -> decimal string with at least one decimal place, trailing zeros kept."""
    neg = value < 0
    v = abs(int(value))
    whole, frac = divmod(v, SCALE)
    out = "%d.%04d" % (whole, frac)
    return "-" + out if neg else out


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def _round_half_away(num, den):
    """Round num/den to the nearest integer, halves AWAY FROM ZERO."""
    neg = (num < 0) != (den < 0)
    num, den = abs(num), abs(den)
    q, r = divmod(num, den)
    if 2 * r >= den:
        q += 1
    return -q if neg else q


def mul(a, b):
    return _round_half_away(a * b, SCALE)


def div(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return _round_half_away(a * SCALE, b)


def to_float(value):
    return value / SCALE
'''

REF_SEMVER = '''
import re

_NUM = re.compile(r"^(0|[1-9][0-9]*)$")


def parse(text):
    """-> (major, minor, patch, prerelease_tuple, build_tuple). Prerelease identifiers that
    are all digits compare numerically; others compare as ASCII strings; numeric < alphanumeric."""
    s = str(text).strip()
    if s.startswith("v"):
        s = s[1:]
    build = ()
    if "+" in s:
        s, _, b = s.partition("+")
        if not b:
            raise ValueError("empty build")
        build = tuple(b.split("."))
    pre = ()
    if "-" in s:
        s, _, p = s.partition("-")
        if not p:
            raise ValueError("empty prerelease")
        pre = tuple(p.split("."))
        for ident in pre:
            if ident == "":
                raise ValueError("empty identifier")
    parts = s.split(".")
    if len(parts) != 3:
        raise ValueError("need major.minor.patch")
    for p in parts:
        if not _NUM.match(p):
            raise ValueError("bad number: %r" % p)
    return (int(parts[0]), int(parts[1]), int(parts[2]), pre, build)


def _pre_cmp(a, b):
    if not a and not b:
        return 0
    if not a:
        return 1          # a release outranks a prerelease
    if not b:
        return -1
    for x, y in zip(a, b):
        xd, yd = x.isdigit(), y.isdigit()
        if xd and yd:
            if int(x) != int(y):
                return -1 if int(x) < int(y) else 1
        elif xd != yd:
            return -1 if xd else 1      # numeric identifiers rank lower
        elif x != y:
            return -1 if x < y else 1
    if len(a) != len(b):
        return -1 if len(a) < len(b) else 1
    return 0


def compare(a, b):
    """-1 / 0 / 1 by semver precedence. Build metadata is IGNORED."""
    pa, pb = parse(a), parse(b)
    for i in range(3):
        if pa[i] != pb[i]:
            return -1 if pa[i] < pb[i] else 1
    return _pre_cmp(pa[3], pb[3])


def _cmp_ok(v, op, t):
    c = compare(v, t)
    return {"<": c < 0, "<=": c <= 0, ">": c > 0, ">=": c >= 0, "=": c == 0, "==": c == 0}[op]


def _bounds(spec):
    """^x.y.z -> >=x.y.z <(x+1).0.0 ; ~x.y.z -> >=x.y.z <x.(y+1).0"""
    if spec.startswith("^"):
        v = parse(spec[1:])
        return ">=%d.%d.%d" % v[:3], "<%d.0.0" % (v[0] + 1)
    if spec.startswith("~"):
        v = parse(spec[1:])
        return ">=%d.%d.%d" % v[:3], "<%d.%d.0" % (v[0], v[1] + 1)
    return None


def satisfies(version, spec):
    """Space-separated AND of comparators. Supports >= <= > < = and ^ and ~."""
    for clause in str(spec).split():
        if not clause:
            continue
        if clause[0] in "^~":
            lo, hi = _bounds(clause)
            if not (_cmp_ok(version, lo[0:2], lo[2:]) and _cmp_ok(version, hi[0], hi[1:])):
                return False
            continue
        for op in (">=", "<=", "==", ">", "<", "="):
            if clause.startswith(op):
                if not _cmp_ok(version, op, clause[len(op):]):
                    return False
                break
        else:
            raise ValueError("bad clause: %r" % clause)
    return True
'''

REF_CSV = '''
def parse(text):
    """RFC4180-ish: comma separator, double-quote quoting, "" escapes a quote inside a quoted
    field, quoted fields may contain commas and newlines. LF, CRLF and a missing trailing
    newline are all accepted. A blank line yields an empty row. Empty input -> []."""
    if text == "":
        return []
    rows, row, field = [], [], []
    in_quotes = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            field.append(ch)
            i += 1
            continue
        if ch == '"' and not field:
            in_quotes = True
            i += 1
            continue
        if ch == ",":
            row.append("".join(field))
            field = []
            i += 1
            continue
        if ch in "\\r\\n":
            if ch == "\\r" and i + 1 < n and text[i + 1] == "\\n":
                i += 1
            row.append("".join(field))
            field = []
            rows.append(row)
            row = []
            i += 1
            continue
        field.append(ch)
        i += 1
    if field or row:
        row.append("".join(field))
        rows.append(row)
    return rows


def _needs_quote(f):
    if f == "":
        return False
    if any(c in f for c in ',"\\n\\r'):
        return True
    return f != f.strip()


def format(rows):
    """Minimal quoting: quote only when the field contains , " CR LF, or leading/trailing
    whitespace. Rows joined with \\n, no trailing newline."""
    out = []
    for row in rows:
        cells = []
        for f in row:
            f = str(f)
            if _needs_quote(f):
                cells.append('"' + f.replace('"', '""') + '"')
            else:
                cells.append(f)
        out.append(",".join(cells))
    return "\\n".join(out)
'''

REF_INTERVALS = '''
def _norm(intervals):
    out = []
    for iv in intervals:
        s, e = iv[0], iv[1]
        if s > e:
            raise ValueError("start after end")
        if s == e:
            continue
        out.append((s, e))
    return sorted(out)


def merge(intervals):
    """Sort, then merge overlapping AND touching (half-open [start, end)). Empty spans dropped."""
    ivs = _norm(intervals)
    out = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def intersect(a, b):
    out = []
    for s1, e1 in _norm(a):
        for s2, e2 in _norm(b):
            s, e = max(s1, s2), min(e1, e2)
            if s < e:
                out.append((s, e))
    return merge(out)


def subtract(a, b):
    """a minus b, as a merged list of half-open spans."""
    pieces = []
    for s1, e1 in _norm(a):
        cur = [(s1, e1)]
        for s2, e2 in _norm(b):
            nxt = []
            for s, e in cur:
                if e2 <= s or s2 >= e:
                    nxt.append((s, e))
                    continue
                if s < s2:
                    nxt.append((s, s2))
                if e2 < e:
                    nxt.append((e2, e))
            cur = nxt
        pieces.extend(cur)
    return merge(pieces)


def coverage(intervals):
    return sum(e - s for s, e in merge(intervals))


def contains(intervals, point):
    for s, e in merge(intervals):
        if s <= point < e:
            return True
    return False
'''

# ============================================================ specs (shown to the model)

SPEC_STATS = """# statistics.py

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
"""

SPEC_FIXED = """# fixedpoint.py

Implement exact decimal arithmetic for money, using **scaled integers**. Pure standard library.

Fixed point with `SCALE = 10000` (4 decimal places). A value is stored as an integer equal to
the real value times 10000. For example `12.3456` is stored as `123456`.

| function | behaviour |
|---|---|
| `parse(text)` | decimal string to scaled integer. Accepts an optional leading `+`/`-`, an optional single `.`, and **at most 4** decimal places. Raises `ValueError` on anything else (empty string, non-digits, more than 4 decimals, more than one `.`). `".5"` and `"5."` are both valid. |
| `format(value)` | scaled integer back to a string with **exactly 4** decimal places, e.g. `123456 -> "12.3456"`, `-5 -> "-0.0005"` |
| `add(a, b)` / `sub(a, b)` | scaled-integer addition / subtraction |
| `mul(a, b)` | exact product, then rounded to the nearest scaled integer, **halves away from zero** |
| `div(a, b)` | exact quotient, then rounded the same way. Raises `ZeroDivisionError` when `b == 0` |
| `to_float(value)` | the value as a Python float |

The point of the module is exactness: `parse("0.1") + parse("0.2")` must equal `parse("0.3")`
**exactly**, with no floating-point error anywhere in `add`/`sub`/`mul`/`div`.

Return **only** the complete file contents in a single ```python code block, nothing else.
"""

SPEC_SEMVER = """# semver.py

Implement Semantic Versioning 2.0.0 parsing and comparison. Pure standard library.

`parse(text)` returns a 5-tuple `(major, minor, patch, prerelease, build)` where `prerelease`
and `build` are **tuples of strings** (empty tuple when absent). An optional leading `v` is
accepted. `major`, `minor` and `patch` must be `0` or a digit string with no leading zero.
Raises `ValueError` for: missing parts, a non-numeric core part, a leading zero, an empty
prerelease or build section, or an empty prerelease identifier.

`compare(a, b)` returns `-1`, `0` or `1` using semver precedence:
- Compare major, then minor, then patch numerically.
- A version **with** a prerelease has **lower** precedence than one without.
- Prerelease identifiers are compared left to right: **numeric identifiers compare
  numerically, and a numeric identifier always ranks lower than an alphanumeric one**;
  alphanumeric identifiers compare by ASCII order. A longer prerelease wins when all the
  shared identifiers are equal.
- **Build metadata is ignored.**

`satisfies(version, spec)` takes a space-separated list of comparators which must **all** hold:
- `>=`, `<=`, `>`, `<`, `=`, `==`
- `^x.y.z` means `>=x.y.z` and `<(x+1).0.0`
- `~x.y.z` means `>=x.y.z` and `<x.(y+1).0`

An unknown comparator raises `ValueError`.

Return **only** the complete file contents in a single ```python code block, nothing else.
"""

SPEC_CSV = """# csvlite.py

Implement a small CSV reader/writer. Pure standard library. Do not use the `csv` module.

`parse(text)` returns a list of rows, each a list of strings:
- `,` separates fields.
- A field starting with `"` is quoted. Inside a quoted field, `""` means a literal `"`, and
  commas and newlines are ordinary characters.
- `\\n`, `\\r\\n` and a **missing final newline** all end a row.
- A blank line produces an empty row (a row with one empty field).
- Empty input returns `[]`.
- The result always has the same number of rows as the input has lines.

`format(rows)` turns rows back into text:
- Fields are quoted **only when necessary**: when the field contains `,`, `"`, `\\r` or `\\n`,
  or has leading or trailing whitespace. An empty field is never quoted.
- A literal `"` inside a quoted field is written `""`.
- Rows are joined with `\\n`; there is **no trailing newline**.

`format(parse(t))` must reproduce `t` for canonical input.

Return **only** the complete file contents in a single ```python code block, nothing else.
"""

SPEC_IV = """# intervals.py

Implement half-open interval arithmetic. Pure standard library. Every interval is a pair
`(start, end)` representing `[start, end)`.

| function | behaviour |
|---|---|
| `merge(intervals)` | sort by start, then merge intervals that overlap **or merely touch**. A zero-length interval (`start == end`) is dropped. Raises `ValueError` if `start > end`. Returns a sorted list of tuples. |
| `intersect(a, b)` | every non-empty overlap of an interval in `a` with one in `b`, merged |
| `subtract(a, b)` | the parts of `a` not covered by `b`, merged |
| `coverage(intervals)` | total length covered, counting overlaps once |
| `contains(intervals, point)` | `True` when the point falls in any interval, honouring the half-open rule |

Because the intervals are half-open, `(1, 2)` and `(2, 3)` merge into `(1, 3)`, but `(1, 2)`
does **not** contain `2`.

Return **only** the complete file contents in a single ```python code block, nothing else.
"""

# ============================================================ check generation

def build_checks():
    """Return {module: [(func, args, kwargs), ...]}; expected values are filled in by the
    caller from the reference, so they cannot drift from the spec."""
    ck = {}

    s = []
    sets = [[1, 2, 3], [1, 2, 3, 4], [5], [2, 2, 2], [1, 1, 2, 3], [-1, 0, 1],
            [1.5, 2.5, 3.5], [10, 20, 30, 40, 50], [0, 0, 0, 1], [3, 1, 4, 1, 5, 9, 2, 6],
            [7, 7, 7, 7, 2, 2, 2], [-5, -10, 0, 5], [100, 1, 50]]
    for xs in sets:
        s += [("mean", (xs,), {}), ("median", (xs,), {}), ("variance", (xs,), {}),
              ("stdev", (xs,), {}), ("mode", (xs,), {})]
        # normalize divides by the stdev, so a constant series would be a ZeroDivisionError
        # rather than a value. Only ask for it where the stdev is non-zero.
        if len(set(xs)) > 1:
            s.append(("normalize", (xs,), {}))
        for p in (0, 10, 25, 50, 75, 90, 100, 33.3, 66.7):
            s.append(("percentile", (xs, p), {}))
        # sample statistics need n >= 2; with n == 1 the reference correctly raises
        if len(xs) >= 2:
            s.append(("variance", (xs,), {"sample": True}))
            s.append(("stdev", (xs,), {"sample": True}))
    for xs, ys in [([1, 2, 3], [4, 5, 6]), ([1, 2, 3, 4], [2, 4, 4, 8]),
                   ([0, 1, 2], [2, 1, 0]), ([2, 4, 6, 8], [1, 2, 3, 4])]:
        s += [("covariance", (xs, ys), {}), ("covariance", (xs, ys), {"sample": True})]
        # correlation divides by both stdevs, so both series must vary
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            s.append(("correlation", (xs, ys), {}))
    ck["statistics"] = s

    f = []
    for text in ["0", "1", "-1", "12.3456", "0.0001", "-0.0001", ".5", "5.", "123.4",
                 "1000000", "-999.9999", "3.1415", "-2.7182", "+42.42",
                 # invalid: the spec mandates ValueError for each of these
                 "", "abc", "1.2.3", "1.23456", "--1", "1e5", " "]:
        f.append(("parse", (text,), {}))
    for a, b in [("0.1", "0.2"), ("1.0001", "2.0002"), ("-1.5", "1.5"), ("0", "0"),
                 ("99.9999", "0.0001"), ("-3.3333", "-6.6667")]:
        f += [("add", (a, b), {}), ("sub", (a, b), {})]
    for a, b in [("1.5", "2.5"), ("0.3333", "3"), ("-2.5", "4"), ("1.00005", "2"),
                 ("0.0001", "0.0001"), ("-1.2345", "-1.2345")]:
        f += [("mul", (a, b), {}), ("div", (a, b), {})]
    ck["fixedpoint"] = f

    v = []
    versions = ["0.0.0", "1.2.3", "v1.2.3", "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta",
                "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0",
                "2.0.0+build.1", "1.0.0+build", "10.20.30", "1.0.0-0", "1.0.0-1.2.3",
                "1.0.0-x.7.z.92", "3.4.5-pre.1+exp.sha.5114f85"]
    for t in versions:
        v.append(("parse", (t,), {}))
    for i, a in enumerate(versions):
        for b in versions[i:]:
            v.append(("compare", (a, b), {}))
    for ver in ["1.2.3", "1.2.4", "1.9.9", "2.0.0", "1.0.0-alpha", "1.0.0", "2.1.0", "1.2.0"]:
        for spec in [">=1.0.0", ">=1.0.0 <2.0.0", "^1.2.3", "~1.2.3", "=1.2.3", ">1.2.3",
                     "^1.0.0", "~1.0.0", ">=1.0.0-alpha"]:
            v.append(("satisfies", (ver, spec), {}))
    ck["semver"] = v

    c = []
    csvs = [
        'a,b,c',
        'a,b,c\\n',
        'a,b\\nc,d\\n',
        'a,,c',
        '',
        '\\n',
        'a\\n\\nb',
        '"a,b",c',
        '"say ""hi""",x',
        '"multi\\nline",y',
        'a,"b\\r\\nc",d',
        ' spaced , x ',
        '"",empty',
        'a,b,',
        'x',
        '"quoted"',
        'a,b\\r\\nc,d\\r\\n',
        '"a""b",c',
        '1,2,3\\n4,5,6',
        '"trailing "',
    ]
    for t in csvs:
        c.append(("parse", (t,), {}))
    rows = [
        [["a", "b"], ["c", "d"]],
        [["has,comma", "plain"]],
        [['has"quote', "x"]],
        [["multi\\nline", "y"]],
        [["", "empty"]],
        [[" leading", "trailing "]],
        [["plain"]],
        [],
        [["a"], ["b"], ["c"]],
    ]
    for r in rows:
        c.append(("format", (r,), {}))
    ck["csvlite"] = c

    iv = []
    sets2 = [
        [[1, 3], [2, 4]],
        [[1, 2], [2, 3]],
        [[5, 6], [1, 2]],
        [[1, 10]],
        [],
        [[3, 3]],
        [[1, 2], [3, 4], [5, 6]],
        [[0, 5], [1, 2], [2, 3]],
        [[10, 20], [15, 25], [30, 40]],
        [[-5, -1], [-3, 2]],
    ]
    for a in sets2:
        iv += [("merge", (a,), {}), ("coverage", (a,), {})]
        for pt in (-6, -1, 0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40):
            iv.append(("contains", (a, pt), {}))
    pairs = [([[1, 5]], [[2, 3]]), ([[1, 5]], [[5, 9]]), ([[1, 10]], [[3, 4], [7, 8]]),
             ([[0, 10]], [[-5, 5]]), ([[1, 2], [4, 5]], [[1, 5]]), ([[5, 6]], [[1, 2]])]
    for a, b in pairs:
        iv += [("intersect", (a, b), {}), ("subtract", (a, b), {})]
    ck["intervals"] = iv
    return ck


MODULES = {
    "statistics": (REF_STATS, SPEC_STATS),
    "fixedpoint": (REF_FIXED, SPEC_FIXED),
    "semver": (REF_SEMVER, SPEC_SEMVER),
    "csvlite": (REF_CSV, SPEC_CSV),
    "intervals": (REF_INTERVALS, SPEC_IV),
}


def _close(a, b, tol=1e-9):
    """Compare a model's result to the reference. Sequences are compared STRUCTURALLY, so a
    list is accepted where the reference returned a tuple: JSON cannot express the difference,
    and it is not a meaningful difference in behaviour."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=tol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    return a == b


def main():
    checks = build_checks()
    total = 0
    for name, (ref, spec) in MODULES.items():
        d = HERE / name
        d.mkdir(parents=True, exist_ok=True)
        ns = {}
        exec(compile(ref, f"<ref:{name}>", "exec"), ns)     # noqa: S102 - our own source
        out = []
        for func, args, kwargs in checks[name]:
            rec = {"func": func, "args": list(args), "kwargs": kwargs}
            try:
                rec["expected"] = ns[func](*args, **kwargs)
            except Exception as e:
                # The spec mandates specific errors; record which one so the model is graded
                # on raising it rather than on returning a wrong value.
                rec["expect_error"] = type(e).__name__
            out.append(rec)
        (d / "checks.json").write_text(json.dumps(out, indent=1))
        (d / "README.md").write_text(spec)
        total += len(out)
        print(f"  {name:11} {len(out):3} checks")

    # Sanity: the reference must pass its own checks, or the build is broken, not the models.
    print("\nverifying each reference against its own checks")
    bad = 0
    for name, (ref, _) in MODULES.items():
        ns = {}
        exec(compile(ref, f"<ref:{name}>", "exec"), ns)
        cks = json.loads((HERE / name / "checks.json").read_text())
        fails = []
        for c in cks:
            try:
                got = ns[c["func"]](*c["args"], **c["kwargs"])
            except Exception as e:
                if c.get("expect_error") == type(e).__name__:
                    continue
                fails.append(f"{c['func']}{c['args']}: raised {type(e).__name__}: {e}")
                continue
            if c.get("expect_error"):
                fails.append(f"{c['func']}{c['args']}: expected {c['expect_error']}, got {got!r}")
                continue
            if not _close(got, c["expected"]):
                fails.append(f"{c['func']}{c['args']}: {got!r} != {c['expected']!r}")
        status = "OK" if not fails else f"{len(fails)} FAIL"
        print(f"  {name:11} {len(cks) - len(fails)}/{len(cks)}  {status}")
        for f in fails[:5]:
            print(f"      {f}")
        bad += len(fails)
    print(f"\n{total} checks across {len(MODULES)} modules")
    if bad:
        raise SystemExit(f"reference implementation failed {bad} of its own checks - fix first")


if __name__ == "__main__":
    main()
