"""The one long document every workflow shares, plus the on-disk operation log.

A single canonical text is built once per run and reused verbatim by every arm: filler plus
planted, uniquely identifiable records at fixed depths, each with a near-miss decoy so retrieval
cannot succeed on fuzzy surface match. Text identity across arms is what makes the comparison
valid, so the document is a pure function of (seed, filler_target_chars) and its sha256 is
recorded in raw.json.

Planted for the six workflows:
  64 shards, every one of them a W2 key, each with a digit-altered near-miss decoy,
  20 of the 64 are the W3 brief identifiers, spanned across the document,
  120 ledger rows with tenants, 60 employee records, 12 rules, 5 invariants,
  60 search clusters in 15 four-hop chains,
  400 mutation log records, 20 per tenant for six tenants (W4 reads this log on disk).
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
               "achernar", "betelgeuse", "canopus", "diphda",
               "alkaid", "alnilam", "alnitak", "alphard", "ankaa", "arneb", "ascella",
               "atria", "avior", "bunda", "celaeno", "chort", "dziban", "edasich",
               "electra", "enif", "errai", "gienah", "gomeisa", "heze", "keid",
               "lesath", "maia", "megrez"]
CODE_PREFIX = ["AK", "BQ", "CZ", "DF", "EG", "HN", "JP", "KM", "LR", "NT"]
PROJECTS = [f"PRJ-{i:02d}" for i in range(1, 21)]
TENANTS = [f"TEN-{i:02d}" for i in range(1, 7)]

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

INVARIANTS = [
    "Invariant I1: The active ledger total is conserved by every transfer.",
    "Invariant I2: A retracted row keeps its record and changes status to VOID.",
    "Invariant I3: An appended row must carry a row id that has never existed.",
    "Invariant I4: Every mutation is applied exactly once and in log order.",
    "Invariant I5: A mutation tagged for one tenant never touches another tenant's rows.",
]

W3_KEYS = 20                  # brief identifiers, spread across the 64 planted shards
SEARCH_CHAINS = 15            # 15 chains x 4 clusters = 60 search clusters
OPS_PER_TENANT = 20
OP_NOISE = 280


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


def _build_shards(rng):
    shards = []
    used = set()
    for i, name in enumerate(SHARD_NAMES):
        code = f"{CODE_PREFIX[i % 10]}-{1000 + rng.randrange(8000)}"
        while code in used:
            code = f"{CODE_PREFIX[i % 10]}-{1000 + rng.randrange(8000)}"
        used.add(code)
        n = int(code[3:])                            # digit-altered near miss
        decoy = f"{code[:2]}-{(n + 1 + rng.randrange(8)) % 10000:04d}"
        while decoy in used or decoy == code:
            decoy = f"{code[:2]}-{rng.randrange(1000, 9999):04d}"
        used.add(decoy)
        shards.append(dict(
            name=name, code=code, custodian=f"EMP-{rng.randrange(100, 999):04d}",
            frac=round(0.02 + i * (0.96 / (len(SHARD_NAMES) - 1)), 4),
            decoy=decoy))
    span = len(shards) - 1
    brief = {round(k * span / (W3_KEYS - 1)) for k in range(W3_KEYS)}
    for i, s in enumerate(shards):
        s["w3_index"] = i if i in brief else None
    return shards


def _build_ledgers(rng):
    ledgers = []
    for i in range(120):
        ledgers.append(dict(
            id=f"L-{i + 1:04d}", tenant=TENANTS[i % len(TENANTS)],
            status=rng.choice(["OPEN", "CLOSED", "VOID", "OPEN", "CLOSED"]),
            tag=rng.choice(["RED", "BLUE", "GOLD", "RED"]),
            amount=rng.randrange(11, 9999),
            frac=round(0.01 + i * (0.98 / 119), 4)))
    return ledgers


def _build_records(rng):
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
    return records


def _build_clusters(rng):
    """15 chains of 4 clusters; a multi-hop question means searching three cluster ids."""
    words = [w for w in VOCAB if len(w) > 5][:160]
    clusters = []
    for c in range(SEARCH_CHAINS):
        chain = []
        for k in range(4):
            cid = f"KX-{c * 4 + k + 1:04d}"
            chain.append(dict(
                id=cid, term=f"{rng.choice(SHARD_NAMES)}-{rng.choice(words)}",
                value=f"{rng.choice(words).upper()}-{rng.randrange(1000, 9999)}",
                next=None,
                frac=round(0.36 + (c * 4 + k) * (0.34 / 59), 4)))
        for k in range(3):
            chain[k]["next"] = chain[k + 1]["id"]
        chain[3]["next"] = "NONE"
        clusters.extend(chain)
    return clusters


def multihop_questions(facts):
    """Three questions, each two `next` hops from its chain head: search floor is 3 calls."""
    out = []
    for c in range(3):
        chain = facts["clusters"][c * 4:c * 4 + 4]
        out.append(dict(
            id=f"multihop_{c + 1}",
            head=chain[0]["id"], hops=[chain[1]["id"], chain[2]["id"]],
            expected=chain[2]["value"],
            prompt=(f"Starting from search cluster {chain[0]['id']}, follow the `next` field twice. "
                    f"Report the `value` of the cluster you land on. Search first; do not guess.")))
    return out


def _build_ops(rng, ledgers):
    """400 mutation records: 20 per tenant for six tenants, plus 280 foreign-tenant noise."""
    by_tenant = {t: [lg for lg in ledgers if lg["tenant"] == t] for t in TENANTS}
    ops, used_ids = [], set()
    for t_i, tenant in enumerate(TENANTS):
        rows = by_tenant[tenant]
        kinds = (["append"] * 7 + ["retract"] * 6 + ["transfer"] * 7)
        rng.shuffle(kinds)
        append_n = 0
        for k, kind in enumerate(kinds):
            op_id = f"OP-{len(ops) + 1:04d}"
            src = rows[(k * 3) % len(rows)]
            if kind == "append":
                append_n += 1
                row = f"L-{2000 + t_i * 100 + append_n:04d}"
                while row in used_ids:
                    append_n += 1
                    row = f"L-{2000 + t_i * 100 + append_n:04d}"
                used_ids.add(row)
                tag = rng.choice(["RED", "BLUE", "GOLD"])
                amount = rng.randrange(11, 999)
                ops.append(dict(id=op_id, tenant=tenant, kind="append", row=row,
                                tag=tag, amount=amount,
                                line=f"OP {op_id} tenant={tenant} kind=append row={row} "
                                     f"tag={tag} amount={amount}"))
            elif kind == "retract":
                ops.append(dict(id=op_id, tenant=tenant, kind="retract", row=src["id"],
                                line=f"OP {op_id} tenant={tenant} kind=retract row={src['id']}"))
            else:
                dst = rows[(k * 3 + 1) % len(rows)]
                amount = rng.randrange(5, 200)
                ops.append(dict(id=op_id, tenant=tenant, kind="transfer", row=src["id"],
                                to=dst["id"], amount=amount,
                                line=f"OP {op_id} tenant={tenant} kind=transfer row={src['id']} "
                                     f"to={dst['id']} amount={amount}"))
    for i in range(OP_NOISE):
        tenant = f"TEN-{90 + i % 8:02d}"          # not a study tenant: pure background noise
        op_id = f"OP-{len(ops) + 1:04d}"
        kind = ["append", "retract", "transfer"][i % 3]
        row = f"L-{3000 + i:04d}" if kind == "append" else \
            ledgers[i % len(ledgers)]["id"]
        if kind == "transfer":
            dst = ledgers[(i + 3) % len(ledgers)]["id"]
            line = (f"OP {op_id} tenant={tenant} kind=transfer row={row} to={dst} "
                    f"amount={rng.randrange(5, 200)} (foreign batch)")
        elif kind == "append":
            line = (f"OP {op_id} tenant={tenant} kind=append row={row} tag=BLUE "
                    f"amount={rng.randrange(11, 999)} (foreign batch)")
        else:
            line = f"OP {op_id} tenant={tenant} kind=retract row={row} (foreign batch)"
        ops.append(dict(id=op_id, tenant=tenant, kind=kind, row=row, line=line, foreign=True))
    rng.shuffle(ops)
    return ops


def build_document(seed=config.DOC_SEED, filler_chars=None):
    """-> (text, facts). Deterministic in (seed, filler_chars)."""
    rng = random.Random(seed)
    filler_chars = config.DOC_FILLER_TARGET if filler_chars is None else filler_chars

    shards = _build_shards(rng)
    ledgers = _build_ledgers(rng)
    records = _build_records(rng)
    clusters = _build_clusters(rng)
    ops = _build_ops(random.Random(seed + 1), ledgers)

    blocks = []
    for s in shards:
        blocks.append((s["frac"], (
            f"SHARD {s['name']} code={s['code']} custodian={s['custodian']}\n"
            f"  superseded-note: revision {s['decoy']} was retired and is not the live code")))
    for lg in ledgers:
        blocks.append((lg["frac"], (
            f"LEDGER {lg['id']} tenant={lg['tenant']} status={lg['status']} "
            f"tag={lg['tag']} amount={lg['amount']}")))
        if lg["id"].endswith("7"):
            blocks.append((min(lg["frac"] + 0.003, 0.995),
                           f"LEDGER {lg['id']}-B tenant={lg['tenant']} status={lg['status']} "
                           f"tag=BLUE amount=1 (superseded copy of {lg['id']})"))
    for r in records:
        blocks.append((r["frac"], (
            f"RECORD {r['emp']} name={r['name']} manager={r['manager']} "
            f"badge={r['badge']} project={r['project']}")))
    for i, rule in enumerate(RULES):
        blocks.append((round(0.500 + i * (0.060 / max(len(RULES) - 1, 1)), 4), rule))
    for i, inv in enumerate(INVARIANTS):
        blocks.append((round(0.565 + i * (0.020 / max(len(INVARIANTS) - 1, 1)), 4), inv))
    for c in clusters:
        blocks.append((c["frac"], (
            f"CLUSTER {c['id']} term={c['term']} value={c['value']} next={c['next']} "
            f"note={_sentence(rng, 12)}")))

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

    text = "\n\n".join(out) + "\n\n--- END OF DOCUMENT ---\n"
    w2_keys = list(shards)
    w3_keys = sorted((s for s in shards if s["w3_index"] is not None),
                     key=lambda s: s["w3_index"])
    facts = dict(shards=shards, ledgers=ledgers, records=records, rules=RULES,
                 invariants=INVARIANTS, clusters=clusters, ops=ops,
                 tenants=TENANTS, w2_keys=w2_keys, w3_keys=w3_keys,
                 multihop=multihop_questions(dict(clusters=clusters)),
                 chains=[dict(head=f"EMP-{h:04d}", hops=[f"EMP-{h + k:04d}" for k in range(5)])
                         for h in (42, 118, 205, 291, 377, 463, 549, 635, 721, 807, 893, 979)],
                 chars=len(text))
    return text, facts


def op_log(facts, doc):
    """The W4 operation log on disk: the shared document followed by the mutation records."""
    return doc + "\n\n--- BEGIN MUTATION LOG ---\n" + \
        "\n".join(op["line"] for op in facts["ops"]) + "\n--- END MUTATION LOG ---\n"


def shard_by_name(facts):
    return {s["name"]: s for s in facts["shards"]}


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def by_depth(items, key="frac"):
    return sorted(items, key=lambda d: d[key])


def exact_occurrences(text, token):
    return len(re.findall(rf"(?<![A-Za-z0-9-]){re.escape(token)}(?![A-Za-z0-9])", text))


if __name__ == "__main__":
    import json
    import sys
    fc = int(sys.argv[1]) if len(sys.argv) > 1 else None
    text, facts = build_document(filler_chars=fc)
    print(len(text), sha256(text))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in facts.items()},
                     indent=1)[:900])
