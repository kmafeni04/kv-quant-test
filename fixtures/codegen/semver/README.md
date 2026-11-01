# semver.py

Implement Semantic Versioning 2.0.0 parsing and comparison. Pure standard library.

`parse(text)` returns a 5-tuple `(major, minor, patch, prerelease, build)` where `prerelease`
and `build` are **tuples of strings** (empty tuple when absent). An optional leading `v` is
accepted. `major`, `minor` and `patch` must be `0` or a digit string with no leading zero.
Raises `ValueError` for: missing parts, a non-numeric core part, a leading zero, an empty
prerelease or build section, or an empty prerelease identifier.

`compare(a, b)` returns `-1`, `0` or `1` using semver precedence:
- Compare major, then minor, then patch numerically.
- A version **with** a prerelease has **lower** precedence than one without.
- Prerelease identifiers are compared left to right: **numeric identifiers compare
  numerically, and a numeric identifier always ranks lower than an alphanumeric one**;
  alphanumeric identifiers compare by ASCII order. A longer prerelease wins when all the
  shared identifiers are equal.
- **Build metadata is ignored.**

`satisfies(version, spec)` takes a space-separated list of comparators which must **all** hold:
- `>=`, `<=`, `>`, `<`, `=`, `==`
- `^x.y.z` means `>=x.y.z` and `<(x+1).0.0`
- `~x.y.z` means `>=x.y.z` and `<x.(y+1).0`

An unknown comparator raises `ValueError`.

Return **only** the complete file contents in a single ```python code block, nothing else.
