"""Merging a list of intervals into a minimal covering list."""

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
