# intervals.py

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
