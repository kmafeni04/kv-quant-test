#!/usr/bin/env python3
"""Generate the p7 fixture: a small multi-file project with a REAL bug, plus a hidden suite.

Why this exists
---------------
p6 (write a module from a spec) saturates: 4 of its 5 modules score 100% for every arm, and
the fifth carries ~1 independent bit. p1h is 94 binary cases, i.e. +/-8.4 points at 95% - too
coarse to separate these models. Neither resembles what the published benchmarks measure.

p7 is the missing axis: READ an existing codebase, find a defect from a failing test, and fix
it. That is what SWE-bench Verified / Terminal-Bench measure, and it is where a larger model
should pull away.

The bug
-------
For two intervals that meet at a point (a.hi == b.lo) they should be merged into one iff that
point is covered, i.e. iff AT LEAST ONE of the two touching bounds is closed:

    [0,1) + [1,2]  -> merge, 1 is in the second            -> [0,2]
    [0,1] + (1,2]  -> merge, 1 is in the first             -> [0,2]
    [0,1) + (1,2]  -> DO NOT merge, 1 is in neither        -> two pieces, gap at 1

`merge._mergeable` calls `a.abuts(b)`, and the buggy `abuts` returns True for ANY touching
pair regardless of closedness. So it bridges the third case and claims to cover 1.

The defect is in core.py/merge.py but only shows up through store.gaps() and report.py, so
fixing it requires reading across the package rather than editing one function in isolation.

Grading
-------
checks.json is generated from a REFERENCE implementation, so it cannot drift from the spec.
The model is shown a failing test that demonstrates the symptom, never the hidden suite.
"""
import json
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "repofixture"

# --------------------------------------------------------------------------- buggy package
CORE = '''"""Interval primitives. Closed/open bounds are (lo_closed, hi_closed)."""


class Interval:
    __slots__ = ("lo", "hi", "lo_closed", "hi_closed")

    def __init__(self, lo, hi, lo_closed=True, hi_closed=True):
        if hi < lo:
            raise ValueError("hi < lo")
        self.lo, self.hi = lo, hi
        self.lo_closed, self.hi_closed = lo_closed, hi_closed

    def __repr__(self):
        lb = "[" if self.lo_closed else "("
        rb = "]" if self.hi_closed else ")"
        return f"{lb}{self.lo},{self.hi}{rb}"

    def __eq__(self, other):
        return (isinstance(other, Interval) and self.lo == other.lo and self.hi == other.hi
                and self.lo_closed == other.lo_closed and self.hi_closed == other.hi_closed)

    def __hash__(self):
        return hash((self.lo, self.hi, self.lo_closed, self.hi_closed))

    def contains(self, x):
        """True if the point x lies in this interval."""
        if x < self.lo or x > self.hi:
            return False
        if x == self.lo and not self.lo_closed:
            return False
        if x == self.hi and not self.hi_closed:
            return False
        return True

    def is_empty(self):
        """True when the interval contains no points at all."""
        return self.lo == self.hi and not (self.lo_closed and self.hi_closed)

    def length(self):
        """Measure of the interval."""
        return 0.0 if self.is_empty() else float(self.hi - self.lo)

    def touches(self, other):
        """True when the two intervals share at least one point."""
        if self.hi < other.lo or other.hi < self.lo:
            return False
        if self.hi == other.lo and not (self.hi_closed and other.lo_closed):
            return False
        if other.hi == self.lo and not (other.hi_closed and self.lo_closed):
            return False
        return True

    def abuts(self, other):
        """True when the two intervals meet at a point without overlapping.

        [0,1) and (1,2] meet at 1. Whether that means they can be collapsed depends on
        whether the meeting point is actually covered - see merge._mergeable.
        """
        if self.hi == other.lo:
            return True
        if other.hi == self.lo:
            return True
        return False
'''

MERGE = '''"""Merging a list of intervals into a minimal covering list."""

from .core import Interval


def _mergeable(a, b):
    """Should a and b be collapsed into one interval?

    They should when they overlap, and when they meet at a point that is COVERED - i.e. when
    at least one of the two touching bounds is closed. [0,1) and (1,2] meet at 1, which
    neither includes, so their union is not one interval.

    BUG: `abuts` reports that any meeting pair "abuts", so this bridges that gap.
    """
    if a.touches(b):
        return True
    return a.abuts(b)


def merge_all(intervals):
    """Return the minimal list of intervals covering the same set of points."""
    live = [i for i in intervals if not i.is_empty()]
    if not live:
        return []
    live = sorted(live, key=lambda i: (i.lo, not i.lo_closed))
    out = [live[0]]
    for cur in live[1:]:
        last = out[-1]
        if _mergeable(last, cur):
            lo, lo_closed = last.lo, last.lo_closed
            if cur.hi > last.hi or (cur.hi == last.hi and cur.hi_closed):
                hi, hi_closed = cur.hi, cur.hi_closed
            else:
                hi, hi_closed = last.hi, last.hi_closed
            out[-1] = Interval(lo, hi, lo_closed, hi_closed)
        else:
            out.append(cur)
    return out
'''

STORE = '''"""An interval store with coverage accounting."""

from .core import Interval
from .merge import merge_all


class Store:
    def __init__(self, intervals=None):
        self._items = list(intervals or [])

    def add(self, interval):
        self._items.append(interval)
        return self

    def merged(self):
        return merge_all(self._items)

    def coverage(self):
        """Total measure covered by the store, counting overlaps once."""
        return sum(i.length() for i in self.merged())

    def gaps(self, lo, hi):
        """Sub-intervals of [lo, hi] not covered by the store."""
        covered = self.merged()
        out = []
        cursor = lo
        cursor_closed = True
        for iv in covered:
            if iv.hi < cursor or (iv.hi == cursor and not (iv.hi_closed and cursor_closed)):
                continue
            if iv.lo > cursor or (iv.lo == cursor and not iv.lo_closed):
                out.append(Interval(cursor, iv.lo, cursor_closed, not iv.lo_closed))
            cursor, cursor_closed = iv.hi, iv.hi_closed
        if cursor < hi or (cursor == hi and cursor_closed):
            out.append(Interval(cursor, hi, not cursor_closed, True))
        return [g for g in out if not g.is_empty()]
'''

REPORT = '''"""Human-readable coverage reporting."""

from .store import Store


def coverage_report(store, lo, hi):
    """Lines describing coverage and the gaps within [lo, hi]."""
    cov = store.coverage()
    gaps = store.gaps(lo, hi)
    lines = [f"coverage={cov:g}"]
    if not gaps:
        lines.append("no gaps")
    else:
        lines.append("gaps=" + ",".join(repr(g) for g in gaps))
    return lines
'''

INIT = '''from .core import Interval
from .store import Store
from .merge import merge_all
from .report import coverage_report

__all__ = ["Interval", "Store", "merge_all", "coverage_report"]
'''

VISIBLE_TEST = '''"""Run this. It fails. Make it pass without changing this file."""

from intervalkit import Interval, Store, coverage_report


def test_both_open_does_not_bridge_the_gap():
    # [0,1) and (1,2] meet at 1, which NEITHER includes. The union is not one interval.
    s = Store([Interval(0, 1, True, False), Interval(1, 2, False, True)])
    assert len(s.merged()) == 2, f"expected 2 pieces, got {len(s.merged())}"


def test_one_closed_still_merges():
    # [0,1) and [1,2] meet at 1, which the SECOND includes, so they DO collapse.
    s = Store([Interval(0, 1, True, False), Interval(1, 2, True, True)])
    assert len(s.merged()) == 1, f"expected 1 piece, got {len(s.merged())}"


def test_report_shows_the_both_open_gap():
    s = Store([Interval(0, 1, True, False), Interval(1, 2, False, True)])
    lines = coverage_report(s, 0, 2)
    assert "gaps=" in lines[1], f"expected a gap, got {lines}"


if __name__ == "__main__":
    fails = 0
    for name in sorted(list(globals())):
        if name.startswith("test_") and callable(globals()[name]):
            try:
                globals()[name]()
                print("PASS " + name)
            except AssertionError as e:
                fails += 1
                print("FAIL " + name + ": " + str(e))
    print()
    print(str(fails) + " failing")
    raise SystemExit(1 if fails else 0)
'''


def build_reference():
    """The CORRECT implementation, used only to compute the hidden expectations."""

    class Interval:
        __slots__ = ("lo", "hi", "lo_closed", "hi_closed")

        def __init__(self, lo, hi, lo_closed=True, hi_closed=True):
            self.lo, self.hi = lo, hi
            self.lo_closed, self.hi_closed = lo_closed, hi_closed

        def is_empty(self):
            return self.lo == self.hi and not (self.lo_closed and self.hi_closed)

        def length(self):
            return 0.0 if self.is_empty() else float(self.hi - self.lo)

        def touches(self, other):
            if self.hi < other.lo or other.hi < self.lo:
                return False
            if self.hi == other.lo and not (self.hi_closed and other.lo_closed):
                return False
            if other.hi == self.lo and not (other.hi_closed and self.lo_closed):
                return False
            return True

        def abuts(self, other):
            """CORRECT: merge when AT LEAST ONE touching bound is closed."""
            if self.hi == other.lo:
                return self.hi_closed or other.lo_closed
            if other.hi == self.lo:
                return other.hi_closed or self.lo_closed
            return False

    def merge_all(intervals):
        live = [i for i in intervals if not i.is_empty()]
        if not live:
            return []
        live = sorted(live, key=lambda i: (i.lo, not i.lo_closed))
        out = [live[0]]
        for cur in live[1:]:
            last = out[-1]
            if last.touches(cur) or last.abuts(cur):
                lo, lo_closed = last.lo, last.lo_closed
                if cur.hi > last.hi or (cur.hi == last.hi and cur.hi_closed):
                    hi, hi_closed = cur.hi, cur.hi_closed
                else:
                    hi, hi_closed = last.hi, last.hi_closed
                out[-1] = Interval(lo, hi, lo_closed, hi_closed)
            else:
                out.append(cur)
        return out

    def coverage(intervals):
        return sum(i.length() for i in merge_all(intervals))

    def merged_count(intervals):
        return len(merge_all(intervals))

    def gaps(intervals, lo, hi):
        """CORRECT gap list within [lo, hi] - where the defect becomes visible."""
        covered = merge_all(intervals)
        out, cursor, cursor_closed = [], lo, True
        for iv in covered:
            if iv.hi < cursor or (iv.hi == cursor and not (iv.hi_closed and cursor_closed)):
                continue
            if iv.lo > cursor or (iv.lo == cursor and not iv.lo_closed):
                out.append((cursor, iv.lo, cursor_closed, not iv.lo_closed))
            cursor, cursor_closed = iv.hi, iv.hi_closed
        if cursor < hi or (cursor == hi and cursor_closed):
            out.append((cursor, hi, not cursor_closed, True))
        return [g for g in out if not (g[0] == g[1] and not (g[2] and g[3]))]

    return {"Interval": Interval, "merge_all": merge_all, "coverage": coverage,
            "merged_count": merged_count, "gaps": gaps}


# Cases chosen so that BOTH-OPEN adjacency appears often - that is the only shape the bug
# changes, so a suite full of closed adjacency would not separate a fix from no fix.
SPECS = [
    ("adjacent_closed",     [(0, 1, True, True),  (1, 2, True, True)]),
    ("open_then_closed",    [(0, 1, True, False), (1, 2, True, True)]),
    ("closed_then_open",    [(0, 1, True, True),  (1, 2, False, True)]),
    ("both_open",           [(0, 1, True, False), (1, 2, False, True)]),
    ("both_open_rev",       [(0, 1, False, True), (1, 2, True, False)]),
    ("overlapping",         [(0, 3, True, True),  (1, 2, True, True)]),
    ("disjoint",            [(0, 1, True, True),  (5, 6, True, True)]),
    ("empty_point",         [(2, 2, False, False), (0, 1, True, True)]),
    ("closed_point",        [(2, 2, True, True),  (0, 3, True, False)]),
    ("nested_open",         [(0, 10, True, True), (2, 3, False, False)]),
    ("three_chain_closed",  [(0, 1, True, True),  (1, 2, True, True),  (2, 3, True, True)]),
    ("three_chain_bothopen", [(0, 1, True, False), (1, 2, False, False), (2, 3, False, True)]),
    ("chain_mixed",         [(0, 1, True, False), (1, 2, True, True),  (2, 3, False, True)]),
    ("unsorted",            [(5, 6, True, True),  (0, 1, True, True),  (1, 2, True, False)]),
    ("chain_open_mid",      [(0, 2, True, True),  (2, 4, False, False), (4, 6, True, True)]),
    ("mixed_ends",          [(0, 1, False, True), (1, 2, True, False), (2, 3, False, True)]),
    ("zero_width_mid",      [(0, 1, True, True),  (1, 1, False, False), (1, 2, True, True)]),
    ("four_open_chain",     [(0, 1, True, False), (1, 2, False, False), (2, 3, False, False), (3, 4, False, True)]),
]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    pkg = OUT / "intervalkit"
    pkg.mkdir(parents=True)
    for name, text in (("__init__.py", INIT), ("core.py", CORE), ("merge.py", MERGE),
                       ("store.py", STORE), ("report.py", REPORT)):
        (pkg / name).write_text(text)
    (OUT / "test_visible.py").write_text(VISIBLE_TEST)
    (OUT / "README.md").write_text(
        "# intervalkit\n\n"
        "Interval arithmetic with open/closed bounds, a merging store and coverage reporting.\n\n"
        "`python3 test_visible.py` fails. Fix the library - do not edit the test.\n\n"
        "Modules:\n"
        "- `intervalkit/core.py`   - `Interval`, `contains`, `touches`, `abuts`, `length`\n"
        "- `intervalkit/merge.py`  - `merge_all` collapses a list into a minimal cover\n"
        "- `intervalkit/store.py`  - `Store.coverage()`, `Store.gaps(lo, hi)`\n"
        "- `intervalkit/report.py` - `coverage_report(store, lo, hi)`\n"
    )

    R = build_reference()
    I, cov, mcount, gaps = R["Interval"], R["coverage"], R["merged_count"], R["gaps"]
    cases = []
    for name, spec in SPECS:
        ivs = [I(*s) for s in spec]
        lo, hi = min(s[0] for s in spec), max(s[1] for s in spec) + 1
        cases.append({"case": name, "kind": "coverage", "intervals": [list(s) for s in spec],
                      "expect": cov(ivs)})
        cases.append({"case": name, "kind": "merged_count", "intervals": [list(s) for s in spec],
                      "expect": mcount(ivs)})
        cases.append({"case": name, "kind": "gaps", "lo": lo, "hi": hi,
                      "intervals": [list(s) for s in spec],
                      "expect": [list(g) for g in gaps(ivs, lo, hi)]})
    (OUT / "checks.json").write_text(json.dumps({"cases": cases}, indent=1))
    print(f"wrote {OUT}")
    print(f"  package: {sum(len((pkg/f).read_text().splitlines()) for f in ('__init__.py','core.py','merge.py','store.py','report.py'))} lines across 5 modules")
    print(f"  hidden checks: {len(cases)}")


if __name__ == "__main__":
    main()
