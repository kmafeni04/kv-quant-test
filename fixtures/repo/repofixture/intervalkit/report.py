"""Human-readable coverage reporting."""

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
