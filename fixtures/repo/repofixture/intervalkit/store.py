"""An interval store with coverage accounting."""

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
