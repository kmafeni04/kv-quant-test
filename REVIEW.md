# Review — chart ordering (round 3), plus one table the ordering fix missed

Scope: `git diff 8c73eb1 -- kvtest/report.py` and the regenerated artifacts. The three findings from
the last round are fixed; one table of the same class still runs worst to best, and it is now the only
surface left pointing the wrong way.

## Fixed and verified — do not revisit

Each item below names the surface that changed and the artifact that shows it, so the list can be
checked without re-reading the diff.

- **Charts read best to worst.** `_ranked_models(out)` (`report.py:145-150`) uses `verdict.order`, and
  it is wired into the three grids, the 16 task dot plots, the speed chart, the loop chart and every
  markdown table. Verified in the artifacts: `fig1.svg` and `fig3.svg` now run `qwen38-flash`, then
  `qwen38-27b`, `qwen36-35b`, `ornith-35b` and `gemma4-26b-a4b`, which is the order capability gives;
  the Tasks tab's dot plots start with `qwen38-flash`; the structural-repetition bars ascend by rate,
  from `ornith q8_0` at 8.3 % to `gemma q4_0` at 25 % with the cleanest arm on top, which matches
  their `green = clean` colouring.
- **Row-order convention is stated**: all three grid captions now end "rows ordered best overall
  first".
- **Title fixed**: `report.html` reads "…: 5 models at f16 / q8_0 / q4_0", and the smoke page
  correctly says "1 model at q4_0" (singular handled).
- **Provenance wall clock fixed**: "wall clock 15 h 27 m" in `results/2026-11-01/report.html`,
  "wall clock 13 m 12 s" in `results/smoke/report.html` — no more em dash.
- **Consistency after regeneration**: 21 SVGs on the page, 21 files, `figures.json` lists 21, 15
  arms; `fig2` (C), `fig5` (loop bars) and `fig6…21` (dot plots) all re-sorted, `fig4` (speed)
  unchanged as it should be; README headline table ranked flash, 27B, 36B, ornith, gemma across 15
  rows; both HTML files well-formed; selftest green (63 checks).

## must-fix

### 1. `kvtest/report.py:264-267` — must-fix — the per-model verdict table is still alphabetical, so it reads worst to best and now contradicts the ranking table directly below it

```python
def _verdict_rows(out):
    """(model, best KV, capability, runner-up, loops, decode, reason) for the verdict section."""
    v = out.get("verdict") or {}
    rows = []
    for model, d in v.get("per_model", {}).items():
```

`per_model` is built by iterating `out["models"]`, which `load()` produces alphabetically, so both
artifacts print the table below. It is ascending by capability, which means worst first.

| model | best KV | capability |
|---|---|---|
| `gemma4-26b-a4b` | f16 | 62.82 |
| `ornith-35b` | f16 | 66.79 |
| `qwen36-35b` | q4_0 | 82.24 |
| `qwen38-27b` | q8_0 | 83.33 |
| `qwen38-flash` | q4_0 | 84.94 |

`REPORT.md:33-38` and the Verdict card in `report.html` — the two most prominent tables on the page,
and the one a reader hits before any chart. These are ascending by capability, i.e. worst first,
and they sit immediately above "### All 15 arms, ranked by the same rule", which runs best to worst.
Before this round every table was alphabetical so the inconsistency was invisible; now this is the
single surface the ordering pass missed, in exactly the direction you objected to.

Fix — iterate the ranked order, keeping the fallback for a model missing from `out["models"]`:

```python
def _verdict_rows(out):
    """(model, best KV, capability, runner-up, loops, decode, reason), best overall first."""
    v = out.get("verdict") or {}
    per_model = v.get("per_model", {})
    known = set(out.get("models") or [])
    order = ([m for m in _ranked_models(out) if m in per_model]
             + [m for m in per_model if m not in known])
    rows = []
    for model in order:
        d = per_model[model]
```

One change fixes both writers (`write_markdown:733` and `_verdict_html:1089`). Verify:

```
python3 -m kvtest.report results/2026-11-01 && python3 -m kvtest.report results/smoke
python3 - <<'EOF'
import re
h = open("results/2026-11-01/report.html").read()
print(re.findall(r"<tr><td>([a-z0-9-]+)</td><td><b>", h[h.find("<th>model</th>"):])[:5])
EOF
# expect: qwen38-flash, qwen38-27b, qwen36-35b, ornith-35b, gemma4-26b-a4b
grep -n "^| \`" results/2026-11-01/REPORT.md | sed -n 1,5p   # same order, before "All 15 arms"
```

## Note (not a finding) — how strict should "best to worst" be per grid?

Because rows are ordered once, by overall capability, each individual grid is only *roughly*
monotone in its own column: on the R grid `qwen36-35b` (60.0) sits above `ornith-35b` (63.3), and on
the C grid `qwen38-27b` (90.0) sits below `qwen38-flash` (89.9). The caption ("rows ordered best
overall first") explains it. The alternative, sorting every grid by its own column, would give a model
a different row in each figure, so a reader comparing the three cells of a row across the grids would
be comparing three different models; that is the reading the grids exist for, and it is why I
recommend keeping one order. If you would rather each grid be strictly descending, that is a two-line
change in `_svg_grids` (sort by `get(c)` per card) plus dropping that caption clause; say so and it
goes in as a finding next round.

## Verdict

The ordering rework is correct everywhere it was applied — grids, dot plots, loop bars, speed chart
and all markdown tables now read best to worst, the row-order convention is captioned, and the title
and wall-clock regressions are gone. The one remaining defect is the same ordering question in the
per-model verdict table, which no longer matches anything around it.

Fix these
