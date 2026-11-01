"""Run this. It fails. Make it pass without changing this file."""

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
