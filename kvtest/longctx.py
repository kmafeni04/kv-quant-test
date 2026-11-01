"""The one long document every task appends to.

A single canonical text is built once per run and reused verbatim by every arm: filler plus
planted, uniquely identifiable records at fixed depths, each with a near-miss decoy so retrieval
cannot succeed on fuzzy surface match. Text identity across arms is what makes the comparison
valid, so the document is a pure function of (seed, filler_target_chars, repository files) and its
sha256 is recorded in raw.json.
"""
import hashlib
import random
import re

from . import config

ROOTS = """harbor vellum quill ember quartz lattice tinder marrow cobalt saffron wicker basalt
glacier prickle thistle mantle socket furrow cadence lantern rivet chisel gadfly willow
sirocco plinth coppice gantry bramble flagon gibbet hearth inkwell jetty kettle loam mistral
nimbus oxbow pinnace quartern runnel steeple tally umbral vane whetstone yarrow zephyr abutment
bellows cinder drift elver fathom gullet huskings ingot jamb keel lanyard mizzen netting oarlock
pennant quarterdeck reef sailcloth tiller undercroft vault wale yardarm anchor bilge capstan
davit ensign fender grapple halyard isthmus jib kelson leeward mainmast nautilus ogee pintle
quickwork ratline scupper thole uphaul victual warp yoke zany abacus bevel chamfer dowel escarp
fillet gasket hinge inlay joinery kerf lapidary mitre nosing ogee plumb quoin rabbet scarf
tenon undercut veneer wainscot axonometric bressummer coffer dorter eaves fanlight gable header
impost joggle kneeler lintel mullion newel oriel parapet quatrefoil rafter sill transom
voussoir wainscot yale ziggurat alcove baluster cames dentil entasis fluting guttae hypotrachelion
iota kalpis lecythus metope necking ovolo plinth regula sima taenia undula volute abrade blanch
calcine deliquesce effloresce flux granulate hydrate infuse jaundice kindle lixiviate macerate
neutralize oxidize precipitate quench rectify sublime titrate unctuous vitrify wash xenolith
yield zest amber beryl chert dolomite epidote feldspar gabbro hornfels ignimbrite jasper
""".split()

SUFFIXES = ["", "ery", "ing", "ous"]


def _vocab():
    out = []
    for r in ROOTS:
        for s in SUFFIXES:
            w = r + s
            if w not in out:
                out.append(w)
    # deterministic top-up if suffixes collided
    i = 0
    while len(out) < 400:
        out.append(f"{ROOTS[i % len(ROOTS)]}{i}")
        i += 1
    return out[:400]


VOCAB = _vocab()

SHARD_NAMES = ["orion", "vega", "lyra", "draco", "cygnus", "altair", "rigel", "sirius",
               "capella", "procyon", "arcturus", "bellatrix", "castor", "deneb", "elara",
               "fomalhaut", "gemma", "hadar", "izar", "jabbah", "kochab", "lucida", "mira",
               "nashira", "okab", "pollux", "quintet", "rasalgethi", "saiph", "tarazed",
               "unukalhai", "vindemiatrix", "wasat", "xerxes", "ytterbium", "zubenelgenubi",
               "achernar", "betelgeuse", "canopus", "diphda"]
CODE_PREFIX = ["AK", "BQ", "CZ", "DF", "EG", "HN", "JP", "KM", "LR", "NT"]
PROJECTS = [f"PRJ-{i:02d}" for i in range(1, 21)]
RULES = [
    "Rule 1: Before any shard is moved, its custodian must sign the manifest.",
    "Rule 2: A ledger row tagged RED is never settled against an OPEN shard.",
    "Rule 3: Two shards sharing a custodian may travel in one container only if closed.",
    "Rule 4: Voided rows are retained for audit and never re-tagged.",
    "Rule 5: A custodian who leaves hands every badge to the manager above them.",
    "Rule 6: Projects ending in an even number require a second signature.",
    "Rule 7: The order of record in the ledger is the order of settlement.",
    "Rule 8: A code suffixed with a dash and four digits is unique for the epoch.",
    "Rule 9: Badge numbers are never reused, not even after a custodian retires.",
    "Rule 10: Any shard at ninety percent depth or deeper is re-sealed on read.",
    "Rule 11: A void ledger row cancels the shard it references, not the project.",
    "Rule 12: Every report ends with the count of rows it did not settle.",
]


def _sentence(rng, n):
    ws = [rng.choice(VOCAB) for _ in range(n)]
    ws[0] = ws[0].capitalize()
    return " ".join(ws) + "."


def _filler(rng, n_chars):
    parts, n = [], 0
    while n < n_chars:
        line = " ".join(_sentence(rng, rng.randint(8, 16)) for _ in range(rng.randint(2, 4)))
        parts.append(line)
        n += len(line) + 2
    text = "\n\n".join(parts)
    return text[:n_chars]


def _code_section():
    """The intervalkit package + failing test, dumped as ### FILE blocks (repo_fix input)."""
    root = f"{config.FIXTURES}/repo/repofixture"
    out = ["The following package is under maintenance and one of its behaviours is wrong.\n"]
    for name in ("README.md",):
        out.append(f"### FILE: {name}\n{open(f'{root}/{name}').read().strip()}\n")
    for name in sorted(("core.py", "merge.py", "store.py", "report.py", "__init__.py")):
        body = open(f"{root}/intervalkit/{name}").read()
        out.append(f"### FILE: intervalkit/{name}\n```python\n{body}```\n")
    test = open(f"{root}/test_visible.py").read()
    out.append(f"### FILE: test_visible.py\n```python\n{test}```\n")
    out.append("--- end of package listing ---\n")
    return "\n".join(out)


def build_document(seed=config.DOC_SEED, filler_chars=None, include_code=True):
    """-> (text, facts). Deterministic in (seed, filler_chars, fixture files on disk)."""
    rng = random.Random(seed)
    filler_chars = config.DOC_FILLER_TARGET if filler_chars is None else filler_chars

    # --- planted records, decided before they are placed so tasks can be built from facts
    shards = []
    for i, name in enumerate(SHARD_NAMES):
        code = f"{CODE_PREFIX[i % 10]}-{1000 + rng.randrange(8000)}"
        shards.append(dict(name=name, code=code, custodian=f"EMP-{rng.randrange(100, 999):04d}",
                           frac=round(0.02 + i * (0.96 / (len(SHARD_NAMES) - 1)), 4)))
    seen = {s["code"] for s in shards}
    for s in shards:                       # ensure unique codes
        while s["code"] in seen and list(seen).count(s["code"]) > 1:
            s["code"] = f"{s['code'][:2]}-{1000 + rng.randrange(8000)}"
            seen.add(s["code"])

    ledgers = []
    for i in range(120):
        ledgers.append(dict(id=f"L-{i + 1:04d}",
                            status=rng.choice(["OPEN", "CLOSED", "VOID", "OPEN", "CLOSED"]),
                            tag=rng.choice(["RED", "BLUE", "GOLD", "RED"]),
                            amount=rng.randrange(11, 9999),
                            frac=round(0.01 + i * (0.98 / 119), 4)))

    heads = [42, 118, 205, 291, 377, 463, 549, 635, 721, 807, 893, 979]
    records = []
    for c, head in enumerate(heads):
        chain = [f"EMP-{head + k:04d}" for k in range(5)]
        for k, emp in enumerate(chain):
            records.append(dict(
                emp=emp, name=f"{rng.choice(SHARD_NAMES)}-{rng.randrange(10, 99)}",
                manager=chain[k + 1] if k + 1 < len(chain) else "NONE",
                badge=f"B-{1000 + c * 10 + k}",
                project=PROJECTS[(c * 5 + k) % len(PROJECTS)],
                frac=round(0.03 + (c * 5 + k) * (0.80 / 59), 4)))

    # --- blocks: (fraction of final text, rendered text)
    blocks = []
    decoy_words = 0
    for i, s in enumerate(shards):
        blocks.append((s["frac"], (
            f"SHARD {s['name']} code={s['code']} custodian={s['custodian']}\n"
            f"  archive-note: revision {s['code'][:2]}-{int(s['code'][3:]) + 1} superseded; "
            f"index tag ZQ-{7000 + rng.randrange(999)}")))
        if i % 2 == 0:
            blocks.append((s["frac"] + 0.002, "ZQ-7731 flagged for re-seal under Rule 10."))
        if i % 3 == 0:
            blocks.append((s["frac"] + 0.004, "ZQ-7732 cleared; ZQ-77310 quarantined; "
                                              "ZQ-7731X retired."))
        decoy_words += 3
    for lg in ledgers:
        blocks.append((lg["frac"], (
            f"LEDGER {lg['id']} status={lg['status']} tag={lg['tag']} amount={lg['amount']}")))
        if lg["id"].endswith("7"):
            blocks.append((min(lg["frac"] + 0.003, 0.995),
                           f"LEDGER {lg['id']}-B status={lg['status']} tag=BLUE amount=1 "
                           f"(superseded copy of {lg['id']})"))
    for r in records:
        blocks.append((r["frac"], (
            f"RECORD {r['emp']} name={r['name']} manager={r['manager']} "
            f"badge={r['badge']} project={r['project']}")))
    for i, rule in enumerate(RULES):
        blocks.append((round(0.500 + i * (0.100 / max(len(RULES) - 1, 1)), 4), rule))
    if include_code:
        blocks.append((0.85, _code_section()))

    # dedupe/shift collisions so ordering stays monotone
    blocks.sort(key=lambda b: b[0])
    total_chars = filler_chars + sum(len(b) + 6 for _, b in blocks)
    filler = _filler(rng, filler_chars)

    out, fi = [], 0
    for frac, block in blocks:
        target = int(frac * total_chars)
        want = max(0, target - sum(len(x) + 2 for x in out))
        if want > 0:
            take = filler[fi:fi + want]
            fi += len(take)
            out.append(take)
        out.append(block)
    out.append(filler[fi:])

    body = "\n\n".join(out)
    text = body + "\n\n--- END OF DOCUMENT ---\n"
    # exact-token occurrences only: ZQ-77310 and ZQ-7731X are decoys, not hits
    fwe = len(re.findall(r"(?<![A-Za-z0-9-])ZQ-7731(?![A-Za-z0-9])", text))
    facts = dict(shards=shards, ledgers=ledgers, records=records, rules=RULES,
                 fwe_count=fwe,
                 chains=[dict(head=f"EMP-{h:04d}", hops=[
                     f"EMP-{h + k:04d}" for k in range(5)]) for h in heads],
                 chars=len(text))
    return text, facts


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def answers(facts):
    """Every answerable quantity a graded task could ask for, computed from the facts."""
    out = {}
    reds = [lg for lg in facts["ledgers"] if lg["status"] == "OPEN" and lg["tag"] == "RED"]
    out["agg_sum"] = (len(reds), sum(lg["amount"] for lg in reds))
    out["count_proj"] = {}
    for r in facts["records"]:
        out["count_proj"][r["project"]] = out["count_proj"].get(r["project"], 0) + 1
    out["fwe_count"] = facts["fwe_count"]
    out["shard_by_name"] = {s["name"]: s for s in facts["shards"]}
    out["by_emp"] = {r["emp"]: r for r in facts["records"]}
    return out


if __name__ == "__main__":
    import json
    import sys
    fc = int(sys.argv[1]) if len(sys.argv) > 1 else None
    text, facts = build_document(filler_chars=fc)
    print(len(text), sha256(text))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in facts.items()},
                     indent=1)[:800])
