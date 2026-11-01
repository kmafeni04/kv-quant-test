#!/usr/bin/env python3
"""Run a hidden check suite against one candidate module. Isolated in a subprocess.

Isolation matters: the candidate is model-written code, so it can hang, recurse, or call
sys.exit. Running it in-process would take the whole benchmark down with it. This exits with
JSON on stdout and nothing else.

    codegen-run.py <candidate.py> <checks.json>
"""
import importlib.util
import json
import math
import sys


def _close(a, b, tol=1e-9):
    """Structural comparison; a list is accepted where the reference returned a tuple, since
    JSON cannot express the difference and it is not a behavioural difference."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=tol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    return a == b


def main():
    mod_path, checks_path = sys.argv[1], sys.argv[2]
    checks = json.loads(open(checks_path).read())
    module, load_error = None, None
    try:
        spec = importlib.util.spec_from_file_location("candidate", mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except BaseException as e:                      # noqa: BLE001 - a model can raise anything
        load_error = f"{type(e).__name__}: {e}"[:200]
    results = []
    for c in checks:
        if module is None:
            results.append({"ok": False, "why": "module did not import"})
            continue
        fn = getattr(module, c["func"], None)
        if not callable(fn):
            results.append({"ok": False, "why": f"no function {c['func']}"})
            continue
        try:
            got = fn(*c["args"], **c["kwargs"])
        except BaseException as e:                  # noqa: BLE001
            if c.get("expect_error") == type(e).__name__:
                results.append({"ok": True, "why": "raised as specified"})
            else:
                results.append({"ok": False,
                                "why": f"raised {type(e).__name__}: {e}"[:140]})
            continue
        if c.get("expect_error"):
            results.append({"ok": False,
                            "why": f"expected {c['expect_error']}, got {got!r}"[:140]})
            continue
        ok = _close(got, c["expected"])
        results.append({"ok": ok, "why": "" if ok else f"{got!r} != {c['expected']!r}"[:140]})
    passed = sum(1 for r in results if r["ok"])
    print(json.dumps({"load_error": load_error, "passed": passed,
                      "of": len(results), "results": results}))


if __name__ == "__main__":
    main()
