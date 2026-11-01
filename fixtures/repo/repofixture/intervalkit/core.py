"""Interval primitives. Closed/open bounds are (lo_closed, hi_closed)."""


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
