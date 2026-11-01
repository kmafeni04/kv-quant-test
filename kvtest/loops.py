"""Loop / degeneration instrument.

Per the operational definition: repeating the same words, commands, tool calls or actions without
doing work. Metrics run on reasoning_content and content SEPARATELY (thinking loops are the common
failure and concatenation would hide them) and on the tool transcript.

Thresholds (single module constants so a reader can substitute their own): a line repeated 5x, a
4-gram repeated 6x consecutively, an 8x word run, an >=8-char tail cycle, a 200-char block 3x, or
a zlib ratio >= 0.42 over >=1 KiB is degenerate text by any reasonable reading. The raw numbers are
always reported beside the verdict.
"""
import re
import zlib
from collections import Counter

LINE_RUN_FLAG = 5
NGRAM_RUN_FLAG = 6
WORD_RUN_FLAG = 8
CYCLE_FLAG = 8
BLOCK_FLAG = 3
ZIP_FLAG = 0.42
ZIP_MIN_CHARS = 1024
SPAN_CHARS = 600


def words(t):
    return re.findall(r"\S+", t or "")


def ngram_max_run(toks, k):
    """Longest run of one k-gram repeated CONSECUTIVELY -> (reps, start_index, kgram)."""
    best = (1, 0, None)
    n = len(toks)
    for i in range(n - k):
        gram = tuple(toks[i:i + k])
        reps, j = 1, i + k
        while j + k <= n and tuple(toks[j:j + k]) == gram:
            reps += 1
            j += k
        if reps > best[0]:
            best = (reps, i, gram)
    return best


def line_max_run(t):
    best, run, prev = 0, 0, None
    for raw in (t or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        run = run + 1 if line == prev else 1
        prev = line
        best = max(best, run)
    return best


def repeated_blocks(t, size=200, step=50, min_reps=3):
    """Any >=size-char window occurring >=min_reps times anywhere -> (text, count)."""
    t = t or ""
    seen = Counter()
    for i in range(0, max(len(t) - size, 0), step):
        seen[t[i:i + size]] += 1
    if not seen:
        return None
    blk, n = seen.most_common(1)[0]
    return (blk, n) if n >= min_reps else None


def suffix_cycle(t, max_period=300, min_reps=3):
    """Largest p such that the last m*p chars are m identical blocks -> (p, m)."""
    t = t or ""
    for p in range(min(max_period, len(t) // min_reps), 7, -1):
        tail = t[-3 * p:]
        if len(tail) == 3 * p and tail[:p] == tail[p:2 * p] == tail[2 * p:]:
            return (p, 3)
    return None


def _span(t, toks, per_k, lines):
    """First SPAN_CHARS of the offending region, so the HTML gallery can mark it."""
    t = t or ""
    if not t:
        return None
    worst_k, worst = None, (1, 0, None)
    for k in (1, 2, 4, 8):
        if per_k[k][0] > worst[0]:
            worst = per_k[k]
            worst_k = k
    if worst_k and worst[0] >= 4:
        tok_start = worst[1]
        prefix = " ".join(toks[:tok_start])
        start = len(prefix)
        return t[start:start + SPAN_CHARS]
    if lines >= LINE_RUN_FLAG:
        prev, run, offset = None, 0, 0
        for raw in t.splitlines(True):
            line = raw.strip()
            if line:
                run = run + 1 if line == prev else 1
                prev = line
                if run == LINE_RUN_FLAG:
                    return t[max(0, offset - SPAN_CHARS // 2):offset + SPAN_CHARS]
            offset += len(raw)
    return t[:SPAN_CHARS]


def metrics(t):
    t = t or ""
    toks = words(t)
    per_k = {k: ngram_max_run(toks, k) for k in (1, 2, 4, 8)}
    lines = line_max_run(t)
    blk = repeated_blocks(t)
    cyc = suffix_cycle(t)
    raw = t.encode()
    zr = (len(zlib.compress(raw, 6)) / max(len(raw), 1)) if raw else 0.0
    grams = Counter(tuple(toks[i:i + 4]) for i in range(max(len(toks) - 3, 0)))
    tot = sum(grams.values())
    dist = (len(grams) / tot) if tot else 1.0
    return dict(word_run_1=per_k[1][0], word_run_2=per_k[2][0],
                word_run_4=per_k[4][0], word_run_8=per_k[8][0],
                line_run=lines, block=list(blk) if blk else None,
                cycle=list(cyc) if cyc else None, zip_ratio=round(zr, 3),
                distinct_4=round(dist or 1.0, 3), chars=len(t),
                span=_span(t, toks, per_k, lines))


def verdict(m, graded=False, got_answer=True, gen_tokens=0):
    reasons = []
    if m["line_run"] >= LINE_RUN_FLAG:
        reasons.append(f"line_run={m['line_run']}")
    if m["word_run_8"] >= NGRAM_RUN_FLAG or m["word_run_4"] >= NGRAM_RUN_FLAG \
            or m["word_run_1"] >= WORD_RUN_FLAG:
        reasons.append("ngram_run=%d" % max(m["word_run_1"], m["word_run_4"], m["word_run_8"]))
    if m["cycle"] and m["cycle"][0] >= CYCLE_FLAG:
        reasons.append(f"cycle={m['cycle']}")
    if m["block"] and m["block"][1] >= BLOCK_FLAG:
        reasons.append(f"block_x{m['block'][1]}")
    if m["chars"] >= ZIP_MIN_CHARS and m["zip_ratio"] >= ZIP_FLAG:
        reasons.append(f"zip={m['zip_ratio']}")
    if graded and gen_tokens >= 1024 and not got_answer:
        reasons.append("no_answer_after_long_gen")
    return (bool(reasons), "; ".join(reasons))
