"""Tier C — hard, automatically verifiable, SWE-flavoured tasks.

Five module-from-spec tasks plus one repository repair, all graded by hidden checks in a
subprocess: `codegen-run.py` for a single module, `repo-run.py` for the package. The extraction
helpers are ports of the earlier harness because they encode measured failure modes (see the
docstrings), and the fixtures are the user's own, copied verbatim so results stay comparable.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

from . import config

CODEGEN_DIR = os.path.join(config.FIXTURES, "codegen")
REPO_DIR = os.path.join(config.FIXTURES, "repo", "repofixture")
CODEGEN_RUN = os.path.join(config.FIXTURES, "codegen-run.py")
REPO_RUN = os.path.join(config.FIXTURES, "repo-run.py")
PKG = "intervalkit"

CODE_RE = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```", re.S)
FILE_RE = re.compile(r"^\s*#{2,4}\s*FILE:\s*(\S+)\s*$", re.M | re.I)

MODULES = ["csvlite", "fixedpoint", "intervals", "semver", "statistics"]


def extract_code(text):
    """Pick the largest fenced block that COMPILES.

    Models often emit a short example before the real file, so the first block is wrong; and a
    naive fence regex can treat an annotated closing line as an opening fence, concatenating two
    blocks into garbage. Compiling every candidate fixes both. Falls back to the largest block.
    """
    text = text or ""
    cands = CODE_RE.findall(text)
    if not cands:
        cands = re.findall(r"```[^\n]*\n(.*?)```", text, re.S)
    good = []
    for b in cands:
        try:
            ast.parse(b)
            good.append(b)
        except SyntaxError:
            pass
    if good:
        return max(good, key=len)
    if cands:
        return max(cands, key=len)
    return text


def extract_files(text):
    """Parse `### FILE: path` headers into {path: source}.

    Two shapes are accepted because models drift between them: a header followed by a fenced
    block, and a header followed by raw source until the next header. A fenced body is used as-is;
    only an unfenced one is dedented, because dedenting the fenced case would flatten real
    indentation.
    """
    out = {}
    text = text or ""
    parts = list(FILE_RE.finditer(text))
    for i, m in enumerate(parts):
        end = parts[i + 1].start() if i + 1 < len(parts) else len(text)
        body = text[m.end():end]
        fences = CODE_RE.findall(body)
        body = max(fences, key=len) if fences else textwrap.dedent(body)
        name = m.group(1).strip().strip("`\"'").replace("\\", "/")
        if f"{PKG}/" in name:
            name = re.sub(r"^.*?(intervalkit/)", r"\1", name)
        name = name.lstrip("./").strip()
        if not name.endswith(".py"):
            continue
        out[name] = body.strip("\n") + "\n"
    return out


def run_codegen(cand_path, checks_path, timeout=180):
    return _run_json([sys.executable, CODEGEN_RUN, cand_path, checks_path], timeout)


def run_repo(cand_dir, checks_path, timeout=180):
    return _run_json([sys.executable, REPO_RUN, cand_dir, checks_path], timeout)


def run_visible_test(cand_dir, timeout=180):
    try:
        p = subprocess.run([sys.executable, "test_visible.py"], cwd=cand_dir,
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0 and "FAIL" not in p.stdout, (p.stdout + p.stderr)[-400:]
    except Exception as e:                                         # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _run_json(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"load_error": f"timed out after {timeout}s", "passed": 0, "of": 0, "results": []}
    except Exception as e:                                         # noqa: BLE001
        return {"load_error": f"runner failed: {type(e).__name__}: {e}", "passed": 0, "of": 0,
                "results": []}


def c_tasks():
    """Six tasks. `of` is the hidden-check count, read from the fixture."""
    out = []
    for mod in MODULES:
        spec = open(os.path.join(CODEGEN_DIR, mod, "README.md")).read()
        checks = json.load(open(os.path.join(CODEGEN_DIR, mod, "checks.json")))
        out.append(dict(id=f"cg_{mod}", tier="C", kind="codegen", module=mod,
                        prompt=spec, of=len(checks)))
    checks = json.load(open(os.path.join(REPO_DIR, "checks.json")))["cases"]
    out.append(dict(
        id="repo_fix", tier="C", kind="repo", of=len(checks), prompt=REPO_PROMPT))
    return out


REPO_PROMPT = (
    "The document contains, at about 85% depth, the full listing of a Python package "
    "`intervalkit` (files `intervalkit/core.py`, `merge.py`, `store.py`, `report.py`, "
    "`__init__.py`) and its test file `test_visible.py`.\n\n"
    "`python3 test_visible.py` fails. Find the root cause(s) and fix them. There may be more than "
    "one, in more than one module. Do not change the test.\n\n"
    "Reply with ONLY the files you change, each preceded by a line of exactly the form\n"
    "`### FILE: intervalkit/<name>.py`, then the complete new contents of that file in a python "
    "code block. Output the whole file, not a diff."
)


def grade_c(task, text, workdir):
    """Run the hidden checks for one C-tier answer. Never raises."""
    if task["kind"] == "codegen":
        code = extract_code(text)
        lines = code.count("\n") + 1
        if len(code.strip()) < 40:
            return dict(passed=0, of=task["of"], score=0.0, resolved=False, lines=lines,
                        load_error="no code block emitted", fails=[])
        os.makedirs(workdir, exist_ok=True)
        cand = os.path.join(workdir, f"{task['module']}.py")
        with open(cand, "w") as f:
            f.write(code)
        res = run_codegen(cand, os.path.join(CODEGEN_DIR, task["module"], "checks.json"))
        fails = [check_id(task, i) for i, r in enumerate(res.get("results") or []) if not r["ok"]]
        return dict(passed=res.get("passed", 0), of=res.get("of") or task["of"],
                    score=round((res.get("passed", 0) / max(res.get("of") or task["of"], 1)), 4),
                    resolved=(res.get("passed", 0) == (res.get("of") or task["of"])),
                    lines=lines, load_error=res.get("load_error"), fails=fails[:12])

    edits = extract_files(text)
    lines = sum(v.count("\n") for v in edits.values())
    if not edits:
        return dict(passed=0, of=task["of"], score=0.0, resolved=False, lines=lines,
                    load_error="no FILE blocks emitted", fails=[], files_changed=[])
    os.makedirs(workdir, exist_ok=True)
    dest = os.path.join(workdir, "repo")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(REPO_DIR, dest)
    for rel, src in edits.items():
        path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(src)
    res = run_repo(dest, os.path.join(REPO_DIR, "checks.json"))
    visible_ok, visible_out = run_visible_test(dest)
    fails = [r.get("case") for r in (res.get("results") or []) if not r["ok"]]
    return dict(passed=res.get("passed", 0), of=res.get("of") or task["of"],
                score=round((res.get("passed", 0) / max(res.get("of") or task["of"], 1)), 4),
                # resolved means the task is actually done: every hidden check passes AND the
                # visible test passes, which is what METHOD.md and the plan promise
                resolved=(res.get("passed", 0) == (res.get("of") or task["of"]) and visible_ok),
                visible_test=visible_ok, visible_out=visible_out if not visible_ok else None,
                lines=lines, load_error=res.get("load_error"), fails=fails[:12],
                files_changed=sorted(edits))


def is_resolved(task_id, grade):
    """A task counts as resolved only when every hidden check passes AND, for repo_fix, the
    visible test passes too. Applied to stored records as well, so an artifact regenerated from an
    older samples.jsonl cannot keep an inflated count."""
    if not grade or grade.get("passed") != grade.get("of"):
        return False
    if task_id == "repo_fix":
        return bool(grade.get("visible_test"))
    return True


def check_id(task, i):
    if task["kind"] != "codegen":
        return str(i)
    try:
        checks = json.load(open(os.path.join(CODEGEN_DIR, task["module"], "checks.json")))
        c = checks[i]
        return f"{c['func']}({', '.join(map(repr, c.get('args') or []))})"[:80]
    except Exception:                                              # noqa: BLE001
        return str(i)


def workdir(base, name):
    d = os.path.join(base, "scratch", re.sub(r"[^A-Za-z0-9_.-]", "_", name))
    os.makedirs(d, exist_ok=True)
    return d


def reference_solution(module):
    """Self-test helper: the fixture's own reference module, when one ships next to the spec."""
    for cand in (f"{CODEGEN_DIR}/{module}/reference.py", f"{CODEGEN_DIR}/{module}/{module}.py"):
        if os.path.exists(cand):
            return open(cand).read()
    return None


if __name__ == "__main__":
    for t in c_tasks():
        print(t["id"], t["of"])
