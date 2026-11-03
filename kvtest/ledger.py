"""W4: the long-horizon ledger workflow.

The operation log is a file on disk. The model can only see it through `read_ops`, so the window
fills as it reads, and it applies the twenty mutation records for its tenant through
`apply_append` / `apply_retract` / `apply_transfer`. Grading is the final ledger hash plus the
five invariants; re-applying a mutation changes the hash.
"""
import hashlib
import json
import os

from . import longctx, toolrun

INVARIANTS = [
    ("I1", "the active ledger total is conserved by every transfer"),
    ("I2", "a retracted row is retained with status VOID"),
    ("I3", "no ledger row exists that the log never created, and none is missing"),
    ("I4", "every mutation is applied exactly once, in log order"),
    ("I5", "no mutation changes a row belonging to another tenant"),
]


def initial_rows(facts):
    return {lg["id"]: dict(id=lg["id"], tenant=lg["tenant"], status=lg["status"],
                           tag=lg["tag"], amount=lg["amount"], rev=0)
            for lg in facts["ledgers"]}


def target_ops(facts, tenant):
    """The tenant's records in the order they appear in the log."""
    return [op for op in facts["ops"] if op["tenant"] == tenant and not op.get("foreign")]


def apply_op(rows, op):
    if op["kind"] == "append":
        row = rows.setdefault(op["row"], dict(id=op["row"], tenant=op["tenant"],
                                               status="OPEN", tag=op["tag"],
                                               amount=op["amount"], rev=0))
        row["rev"] += 1
    elif op["kind"] == "retract":
        rows[op["row"]]["status"] = "VOID"
        rows[op["row"]]["rev"] += 1
    elif op["kind"] == "transfer":
        rows[op["row"]]["amount"] -= op["amount"]
        rows[op["row"]]["rev"] += 1
        rows[op["to"]]["amount"] += op["amount"]


def state_hash(rows):
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def expected_rows(facts, tenant):
    rows = initial_rows(facts)
    for op in target_ops(facts, tenant):
        apply_op(rows, op)
    return rows


def five_invariants(rows, expected):
    """I4 lives on the applied log, not on row values, so `grade` adds it."""
    def active(rs):
        return sum(r["amount"] for r in rs.values() if r["status"] != "VOID")

    return {
        "I1": active(rows) == active(expected),
        "I2": all(rows[i]["status"] == "VOID" for i in rows
                  if expected.get(i, {}).get("status") == "VOID"),
        "I3": set(rows) == set(expected),
        "I5": all(rows[i]["tenant"] == expected[i]["tenant"] for i in rows if i in expected),
    }


TOOLS = [
    {"type": "function", "function": {
        "name": "read_ops",
        "description": "Read a window of the operation log on disk. One line per record.",
        "parameters": {"type": "object", "properties": {
            "start": {"type": "integer", "description": "0-based line offset"},
            "count": {"type": "integer", "description": "lines to return (1-250)"}},
            "required": ["start", "count"]}}},
    {"type": "function", "function": {
        "name": "ledger_row",
        "description": "Read the current state of one ledger row by id.",
        "parameters": {"type": "object", "properties": {
            "row": {"type": "string"}}, "required": ["row"]}}},
    {"type": "function", "function": {
        "name": "ledger_summary",
        "description": "Current ledger row count, active total and state hash.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "apply_append",
        "description": "Apply an append mutation record. Arguments must match the record exactly.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string"}, "row": {"type": "string"},
            "tag": {"type": "string"}, "amount": {"type": "integer"}},
            "required": ["op", "row", "tag", "amount"]}}},
    {"type": "function", "function": {
        "name": "apply_retract",
        "description": "Apply a retract mutation record. Arguments must match the record exactly.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string"}, "row": {"type": "string"}},
            "required": ["op", "row"]}}},
    {"type": "function", "function": {
        "name": "apply_transfer",
        "description": "Apply a transfer mutation record. Arguments must match the record exactly.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string"}, "row": {"type": "string"},
            "to": {"type": "string"}, "amount": {"type": "integer"}},
            "required": ["op", "row", "to", "amount"]}}},
    {"type": "function", "function": {
        "name": "report",
        "description": ("Finish the episode with the final ledger hash and the truth value of "
                        "each invariant I1..I5."),
        "parameters": {"type": "object", "properties": {
            "hash": {"type": "string"},
            "invariants": {"type": "object",
                           "description": "map of I1..I5 to true/false",
                           "additionalProperties": {"type": "boolean"}}},
            "required": ["hash", "invariants"]}}},
]


class LedgerEnv:
    """Authoritative ledger state + windowed log access for one instance."""

    def __init__(self, facts, doc, tenant, log_path):
        self.facts, self.tenant, self.log_path = facts, tenant, log_path
        self.rows = initial_rows(facts)
        self.expected = expected_rows(facts, tenant)
        self.expected_hash = state_hash(self.expected)
        self.ops = target_ops(facts, tenant)
        self.by_id = {op["id"]: op for op in self.ops}
        self.applied, self.foreign, self.double = [], [], []
        self.report = None
        self.reads = 0
        self.lines_read = 0
        self.tools = TOOLS
        if doc:
            with open(log_path, "w") as f:
                f.write(doc)

    # ---------------------------------------------------------- tool surface
    def state_hash(self):
        return state_hash(self.rows)

    def progress_hash(self):
        return hashlib.sha256(f"{self.state_hash()}|{self.reads}|{self.lines_read}"
                              .encode()).hexdigest()

    def __call__(self, name, args, messages):
        if name == "read_ops":
            return self._read_ops(args)
        if name == "ledger_row":
            row = self.rows.get(str(args.get("row", "")).strip())
            return dict(ok=bool(row), payload=json.dumps(row) if row else "no such row", diff=None)
        if name == "ledger_summary":
            return dict(ok=True, payload=json.dumps(dict(
                rows=len(self.rows),
                active_total=sum(r["amount"] for r in self.rows.values() if r["status"] != "VOID"),
                hash=self.state_hash())), diff=None)
        if name in ("apply_append", "apply_retract", "apply_transfer"):
            return self._apply(name, args)
        if name == "report":
            self.report = dict(hash=str(args.get("hash") or ""),
                               invariants=args.get("invariants") or {})
            return dict(ok=True, payload="report recorded; episode over", end=True)
        return dict(ok=False, payload=f"unknown tool {name}", diff=None)

    def _read_ops(self, args):
        try:
            with open(self.log_path) as f:
                lines = f.read().splitlines()
            total = len(lines)
            start = int(args.get("start", 0))
            count = max(1, min(int(args.get("count", 50)), 250))
        except Exception as e:                                     # noqa: BLE001
            return dict(ok=False, payload=f"bad read_ops arguments: {e}", diff=None)
        if start < 0 or start >= total:
            return dict(ok=False, payload=f"start {start} outside 0..{total - 1}", diff=None)
        chunk = lines[start:start + count]
        self.reads += 1
        self.lines_read += len(chunk)
        payload = (f"lines {start + 1}-{start + len(chunk)} of {total}\n" +
                   "\n".join(chunk))
        return dict(ok=bool(chunk), payload=payload,
                    diff={"read": [start, start + len(chunk)]})

    def _apply(self, name, args):
        op_id = str(args.get("op") or "").strip()
        op = self.by_id.get(op_id)
        kind = {"apply_append": "append", "apply_retract": "retract",
                "apply_transfer": "transfer"}[name]
        if not op:
            if any(o["id"] == op_id for o in self.facts["ops"]):
                self.foreign.append(op_id)
                return dict(ok=False, payload=f"{op_id} is not a {self.tenant} mutation", diff=None)
            return dict(ok=False, payload=f"no mutation record {op_id}", diff=None)
        if op["kind"] != kind:
            return dict(ok=False, payload=f"{op_id} is a {op['kind']} record, not {kind}", diff=None)
        fields = {"row": "row", "tag": "tag", "amount": "amount", "to": "to"}
        for arg, key in fields.items():
            if key in op and str(args.get(arg)) != str(op[key]):
                return dict(ok=False, payload=f"{op_id}: argument {arg} does not match the record",
                            diff=None)
        before = {i: dict(self.rows[i]) for i in ([op["row"]] + ([op["to"]] if op.get("to") else []))
                  if i in self.rows}
        if op_id in self.applied:
            self.double.append(op_id)
        apply_op(self.rows, op)
        self.applied.append(op_id)
        after = {i: self.rows.get(i) for i in before}
        return dict(ok=True, payload=f"{op_id} {kind} applied; hash={self.state_hash()[:16]}",
                    diff={"before": before, "after": after})


def build_instance(facts, doc, tenant, out_dir, instance):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"oplog-t{instance}.txt")
    env = LedgerEnv(facts, longctx.op_log(facts, doc), tenant, log_path)
    return env, log_path


def prompt(facts, tenant, log_name):
    inv = "; ".join(f"{i} {text}" for i, text in INVARIANTS)
    return (
        f"You are reconciling tenant {tenant}. The operation log is on disk at `{log_name}` and "
        "you cannot see it except through `read_ops(start, count)`; read it in windows.\n\n"
        f"Apply the twenty mutation records tagged tenant={tenant} exactly once each, in log "
        "order, using `apply_append`, `apply_retract` and `apply_transfer`. Ignore every other "
        "tenant's records. Arguments must be copied from the record; a mismatch is rejected.\n\n"
        "When the ledger is fully reconciled, call `report` with the final ledger hash from "
        "`ledger_summary` and the truth value of each invariant:\n"
        f"{inv}\n\n"
        "Do not guess a mutation record. Every one of the twenty is in the log.")


def grade(env):
    inv = five_invariants(env.rows, env.expected)
    order_ok = env.applied == [op["id"] for op in env.ops]
    inv["I4"] = order_ok and not env.double
    reported = (env.report or {}).get("invariants") or {}
    reported_hits = sum(1 for k, v in inv.items() if bool(reported.get(k)) == v)
    return dict(state_ok=env.state_hash() == env.expected_hash,
                final_hash=env.state_hash(), expected_hash=env.expected_hash,
                invariants=inv, invariants_ok=sum(inv.values()),
                order_ok=order_ok, applied=len(env.applied), expected_ops=len(env.ops),
                double_applied=env.double, foreign_applied=env.foreign,
                report=env.report, reported_invariants_match=reported_hits,
                reads=env.reads, lines_read=env.lines_read,
                tools_used=env.reads + len(env.applied))


def run(chat, facts, doc, tenant, out_dir, instance, guard=120, ctx=None):
    env, log_path = build_instance(facts, doc, tenant, out_dir, instance)
    messages = [{"role": "user",
                 "content": prompt(facts, tenant, os.path.basename(log_path))}]
    box = toolrun.ToolBox(env.tools, env, env.progress_hash)
    res = toolrun.run_tool_loop(chat, messages, env.tools, box, guard=guard, ctx=ctx)
    grade_ = grade(env)
    return dict(workflow="W4", instance=instance, tenant=tenant,
                out_tokens=res.get("out_tokens"),
                content=res["content"], reasoning=res["reasoning"],
                transcript=res["transcript"], halted_by=res["halted_by"],
                halt_token_pos=res["halt_token_pos"], loop=res["loop"],
                prompt_tokens=res["prompt_tokens"], tools_used=res["tools_used"],
                grade=grade_, errors=res["errors"])
