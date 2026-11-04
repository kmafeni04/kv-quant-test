"""Single-response workflows: W2 (64-key retrieval) and W6 (unbounded long-form).

W2 is a count out of 64, not a verdict: every position holds either the correct code or one of
`absent` / `mutated` / `misordered`, so a loss of one deep key reads as 63 and a lost deep half
reads as 32. W6 grades structure from the output text; the loop instrument runs over the whole
output separately.
"""
import re

ANSWER_RE = re.compile(r"^ANSWER:\s*(.+?)\s*$", re.M)
CODE_RE = re.compile(r"\b[A-Z]{2}-\d{4}\b")


def answer_block(text):
    """Everything after the last ANSWER: tag up to the next blank line (values may wrap)."""
    idx = (text or "").rfind("ANSWER:")
    if idx < 0:
        return None
    return (text[idx + len("ANSWER:"):].split("\n\n")[0]).strip()


def norm(s):
    return re.sub(r"\s+", "", (s or "")).upper()


def parse_codes(value):
    """Code-like tokens in order from a comma-, space- or newline-separated answer line."""
    return CODE_RE.findall((value or "").upper())


# ---------------------------------------------------------------- W2
def w2_prompt(facts):
    names = ", ".join(s["name"] for s in facts["w2_keys"])
    n = len(facts["w2_keys"])
    return (
        "The document records shard codes at many depths. For each of the "
        f"{n} shards named below, in exactly this order, report its live code in its "
        f"XY-#### form:\n{names}\n\n"
        "Do not re-read, re-check or verify any shard, and do not deliberate once you have the "
        f"{n} codes: stop and emit the answer. Your response must end with one "
        "comma-separated list on a single final line, in the same order:\n"
        f"ANSWER: <code1>, <code2>, ..., <code{n}>"
    )


def grade_w2(facts, content, reasoning=""):
    """-> {score, of, keys:[{name, depth, expected, got, status}], answer, source}"""
    value, source = None, None
    for label, text in (("content", content), ("reasoning", reasoning)):
        block = answer_block(text)
        if block is not None:
            value, source = block, label
            break
    codes = parse_codes(value) if value else []
    keys = []
    score = 0
    for i, shard in enumerate(facts["w2_keys"]):
        expected = norm(shard["code"])
        got = codes[i] if i < len(codes) else None
        if got == expected:
            status = "hit"
            score += 1
        elif got is None:
            status = "absent"
        elif expected in codes:
            status = "misordered"
        else:
            status = "mutated"
        keys.append(dict(name=shard["name"], depth=shard["frac"], expected=expected,
                         got=got, status=status))
    return dict(score=score, of=len(facts["w2_keys"]), keys=keys,
                answer=value, source=source, codes=len(codes), extras=max(0, len(codes) - len(keys)))


# ---------------------------------------------------------------- W6
SECTIONS = [("Custody", 4), ("Settlement", 6), ("Containers", 2),
            ("Audit", 5), ("Escalation", 1), ("Reporting", 3)]


def w6_prompt(facts):
    lines = "\n".join(f"{i + 1}. {title}" for i, (title, _) in enumerate(SECTIONS))
    refs = "; ".join(f"{i + 1} -> {target}" for i, (_, target) in enumerate(SECTIONS))
    return (
        "Write an operator's manual with exactly six sections, in this order:\n"
        f"{lines}\n\n"
        "Each section starts with a heading of the form `## <n>. <title>`, is 120-220 words, and "
        "ends with one cross-reference line of the form `See Section <k>.` Use these "
        f"cross-references, one per section: {refs}.\n"
        f"Ground the manual in the document's {len(facts['invariants'])} invariant statements and "
        "its ledger rules. Do not add a seventh section, a preface or an appendix."
    )


def _section_bodies(text):
    text = text or ""
    marks = []
    for m in re.finditer(r"(?im)^\s*#{0,6}\s*(\d{1,2})\.\s*([A-Za-z][A-Za-z ]*?)\s*$", text):
        marks.append((int(m.group(1)), m.group(2).strip().lower(), m.start(), m.end()))
    bodies = {}
    for i, (num, title, _, end) in enumerate(marks):
        stop = marks[i + 1][2] if i + 1 < len(marks) else len(text)
        bodies[num] = dict(title=title, body=text[end:stop])
    return marks, bodies


def grade_w6(content):
    marks, bodies = _section_bodies(content)
    present, resolved, details = 0, 0, []
    for i, (title, target) in enumerate(SECTIONS):
        num = i + 1
        body = bodies.get(num)
        ok = bool(body) and title.lower() in body["title"]
        present += 1 if ok else 0
        ref = None
        if body:
            m = re.search(r"(?i)see\s+section\s+(\d{1,2})", body["body"])
            ref = int(m.group(1)) if m else None
        good_ref = ref == target
        resolved += 1 if good_ref else 0
        details.append(dict(section=num, title=title, present=ok, expected_ref=target,
                            ref=ref, ref_ok=good_ref,
                            words=len(re.findall(r"\S+", body["body"])) if body else 0))
    return dict(sections_present=present, of_sections=len(SECTIONS),
                refs_resolved=resolved, of_refs=len(SECTIONS),
                score=present + resolved, of=2 * len(SECTIONS),
                ok=present == len(SECTIONS) and resolved == len(SECTIONS),
                sections=details)
