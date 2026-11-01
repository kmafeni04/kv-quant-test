#!/usr/bin/env python3
"""Grade a p7 attempt: load the model's fixed package and run the hidden checks.

Usage: repo-run.py <candidate_dir> <checks.json>

<candidate_dir> must contain an `intervalkit/` package. The model's files are loaded from
there; nothing else about the repo is trusted. Prints one JSON line:

    {"load_error": ..., "passed": N, "of": M, "results": [...]}

Isolated in a subprocess on purpose: a candidate that loops or raises at import time must not
take the harness with it, and each attempt must start from a clean module cache.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path


def _load(cand_dir):
    """Import the candidate's intervalkit package fresh, from cand_dir only."""
    pkg_init = Path(cand_dir) / "intervalkit" / "__init__.py"
    if not pkg_init.exists():
        raise FileNotFoundError(f"no intervalkit/__init__.py in {cand_dir}")
    spec = importlib.util.spec_from_file_location(
        "intervalkit",
        pkg_init,
        submodule_search_locations=[str(Path(cand_dir) / "intervalkit")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["intervalkit"] = mod
    spec.loader.exec_module(mod)
    return mod


def _norm(iv):
    """A gap as a comparable list, tolerating an Interval or a plain tuple."""
    if hasattr(iv, "lo"):
        return [iv.lo, iv.hi, iv.lo_closed, iv.hi_closed]
    return list(iv)


def main():
    cand_dir, checks_path = sys.argv[1], sys.argv[2]
    checks = json.loads(Path(checks_path).read_text())["cases"]
    out = {"load_error": None, "passed": 0, "of": len(checks), "results": []}
    try:
        pkg = _load(cand_dir)
        Interval, Store = pkg.Interval, pkg.Store
    except Exception:
        out["load_error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
        print(json.dumps(out))
        return

    for c in checks:
        got, err = None, None
        try:
            store = Store([Interval(*s) for s in c["intervals"]])
            if c["kind"] == "coverage":
                got = store.coverage()
            elif c["kind"] == "merged_count":
                got = len(store.merged())
            else:
                got = [_norm(g) for g in store.gaps(c["lo"], c["hi"])]
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        ok = (err is None and got == c["expect"])
        out["passed"] += 1 if ok else 0
        out["results"].append({"case": c["case"], "kind": c["kind"], "ok": ok,
                               "got": got, "expect": c["expect"], "error": err})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
