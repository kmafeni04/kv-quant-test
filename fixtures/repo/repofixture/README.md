# intervalkit

Interval arithmetic with open/closed bounds, a merging store and coverage reporting.

`python3 test_visible.py` fails. Fix the library - do not edit the test.

Modules:
- `intervalkit/core.py`   - `Interval`, `contains`, `touches`, `abuts`, `length`
- `intervalkit/merge.py`  - `merge_all` collapses a list into a minimal cover
- `intervalkit/store.py`  - `Store.coverage()`, `Store.gaps(lo, hi)`
- `intervalkit/report.py` - `coverage_report(store, lo, hi)`
