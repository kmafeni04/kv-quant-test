# semver.py

Implement Semantic Versioning 2.0.0 parsing and comparison, using the Python standard library only.
Nothing here is a formatting exercise: the work is in the ordering rules, which are what the hidden
checks test.

`parse(text)` returns a 5-tuple `(major, minor, patch, prerelease, build)`, where `prerelease` and
`build` are **tuples of strings**, empty when the section is absent. An optional leading `v` is
accepted. Each of `major`, `minor` and `patch` must be `0` or a digit string with no leading zero.
`ValueError` is raised for a missing part, a non-numeric core part, a leading zero, an empty prerelease
or build section, and an empty prerelease identifier.

`compare(a, b)` returns `-1`, `0` or `1` following semver precedence:

- Compare major, then minor, then patch, numerically at each step.
- A version **with** a prerelease has **lower** precedence than one without, so `1.0.0-rc1` sorts
  before `1.0.0`.
- Prerelease identifiers are compared left to right. Numeric identifiers compare numerically, and a
  numeric identifier always ranks lower than an alphanumeric one; alphanumeric identifiers compare by
  ASCII order. When all shared identifiers are equal, the longer prerelease wins.
- **Build metadata is ignored**, so two versions that differ only in build metadata compare equal.

`satisfies(version, spec)` takes a space-separated list of comparators, all of which must hold:

- `>=`, `<=`, `>`, `<`, `=`, `==`, which compare directly against the version;
- `^x.y.z`, meaning `>=x.y.z` and `<(x+1).0.0`;
- `~x.y.z`, meaning `>=x.y.z` and `<x.(y+1).0`.

An unknown comparator raises `ValueError`.

Return **only** the complete file contents in a single ```python code block, nothing else.
