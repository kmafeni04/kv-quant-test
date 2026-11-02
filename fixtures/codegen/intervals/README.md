# intervals.py

Implement half-open interval arithmetic with the Python standard library only. Every interval is a
pair `(start, end)` representing `[start, end)`: the start is included and the end is not.

- `merge(intervals)` sorts by start and then merges intervals that overlap or merely touch. It drops
  a zero-length interval (`start == end`), because a half-open interval of zero width covers nothing.
  It raises `ValueError` if `start > end`, and returns a sorted list of tuples.
- `intersect(a, b)` returns every non-empty overlap between an interval in `a` and one in `b`, merged.
- `subtract(a, b)` returns the parts of `a` that `b` does not cover, merged.
- `coverage(intervals)` returns the total length covered, counting an overlap once rather than twice.
- `contains(intervals, point)` returns `True` when the point falls inside any interval, applying the
  half-open rule so an end point is excluded.

The two consequences of the half-open rule are the ones the hidden checks press on:
`(1, 2)` and `(2, 3)` merge into `(1, 3)` because they touch, while `(1, 2)` does **not** contain `2`
because the end is excluded.

Return **only** the complete file contents in a single ```python code block, nothing else.
