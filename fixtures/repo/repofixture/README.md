# intervalkit

Interval arithmetic with open and closed bounds, a merging store, and coverage reporting.

`python3 test_visible.py` fails. Fix the library until it passes, and do not edit the test: the test is
the specification you are working from, so changing it would hide the bug rather than fix it.

The package is five modules, each with one job:

- `intervalkit/core.py` holds the `Interval` type plus `contains`, `touches`, `abuts` and `length`,
  which are the primitives every other module builds on.
- `intervalkit/merge.py` has `merge_all`, which collapses a list of intervals into a minimal cover.
- `intervalkit/store.py` holds `Store`, whose `coverage()` reports the covered length and whose
  `gaps(lo, hi)` reports the uncovered stretches inside a range.
- `intervalkit/report.py` has `coverage_report(store, lo, hi)`, which renders the store's state as a
  report.
