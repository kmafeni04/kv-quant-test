"""W5: agentic search over 60 planted fact clusters.

`search(term)` returns the top-3 matching cluster snippets, and three multi-hop questions each
need three searches (head -> next -> next). The instance reports calls used against a floor of 9
(three cites per question) and a ceiling of 60, plus the three answer keys graded from the
`submit_answer` tool.
"""
import json
import re

from . import toolrun

SEARCH_FLOOR_PER_QUESTION = 3


def _terms(cluster):
    return f"{cluster['id']} {cluster['term']} {cluster['value']}".lower()


def search(facts, term, k=3):
    needle = (term or "").strip().lower()
    words = [w for w in re.findall(r"[a-z0-9-]+", needle) if len(w) > 2]
    scored = []
    for c in facts["clusters"]:
        hay = _terms(c)
        score = 0
        if needle and needle in hay:
            score += 10
        score += sum(1 for w in words if w in hay)
        if score:
            scored.append((score, c["id"], c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    hits = [dict(id=c["id"], term=c["term"], value=c["value"], next=c["next"])
            for _, _, c in scored[:k]]
    return hits


def _snippet(hit):
    return (f"CLUSTER {hit['id']} term={hit['term']} value={hit['value']} "
            f"next={hit['next']}")


TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the 60 fact clusters. Returns the top-3 matching snippets.",
        "parameters": {"type": "object", "properties": {
            "term": {"type": "string", "description": "term or cluster id to search for"}},
            "required": ["term"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the answer to the current question and end the episode.",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "string"}}, "required": ["value"]}}},
]


class SearchEnv:
    def __init__(self, facts):
        self.facts = facts
        self.calls = 0
        self.submitted = None
        self.terms = []

    def __call__(self, name, args, messages):
        if name == "search":
            term = str(args.get("term") or "")
            hits = search(self.facts, term)
            self.calls += 1
            self.terms.append(term)
            payload = ("no cluster matched" if not hits else
                       "\n".join(_snippet(h) for h in hits))
            return dict(ok=bool(hits), payload=payload,
                        diff={"search": term, "hits": [h["id"] for h in hits]})
        if name == "submit_answer":
            self.submitted = str(args.get("value") or "").strip()
            return dict(ok=True, payload="answer recorded; episode over", end=True)
        return dict(ok=False, payload=f"unknown tool {name}", diff=None)

    def progress_hash(self):
        return f"search:{self.calls}:{len(self.terms)}"

    def state_hash(self):
        return self.progress_hash()


def question_prompt(q):
    return (
        "You have a search tool over 60 planted fact clusters. Each snippet carries `term`, "
        "`value` and `next`. Do not guess: search for every id you need.\n\n"
        f"{q['prompt']}\n\n"
        "When you have the value, call `submit_answer` with it.")


def grade(answers, expected, calls):
    """answers: per-question submitted values; expected: strings."""
    got = [re.sub(r"\s+", "", (a or "")).upper() for a in answers]
    want = [re.sub(r"\s+", "", e).upper() for e in expected]
    hits = [g == w for g, w in zip(got, want)]
    return dict(keys_correct=sum(hits), of=len(want), hits=hits, answers=answers,
                expected=want, calls=calls, floor=SEARCH_FLOOR_PER_QUESTION * len(want),
                floor_met=calls >= SEARCH_FLOOR_PER_QUESTION * len(want))


def run(chat, facts, instance, guard=60, ctx=None):
    env = SearchEnv(facts)
    expected = [q["expected"] for q in facts["multihop"]]
    answers, episodes = [], []
    for q in facts["multihop"]:
        guard_left = max(guard - env.calls, 1)
        env.submitted = None
        messages = [{"role": "user", "content": question_prompt(q)}]
        box = toolrun.ToolBox(TOOLS, env, env.progress_hash)
        res = toolrun.run_tool_loop(chat, messages, TOOLS, box, guard=guard_left, ctx=ctx)
        answers.append(env.submitted)
        episodes.append(dict(question=q["id"], submitted=env.submitted,
                             halted_by=res["halted_by"], loop=res["loop"],
                             halt_token_pos=res["halt_token_pos"],
                             transcript=res["transcript"], content=res["content"],
                             reasoning=res["reasoning"], prompt_tokens=res["prompt_tokens"],
                             out_tokens=res.get("out_tokens"),
                             tools_used=res["tools_used"], errors=res["errors"]))
    return dict(workflow="W5", instance=instance, episodes=episodes,
                out_tokens=sum(e.get("out_tokens") or 0 for e in episodes),
                grade=grade(answers, expected, env.calls),
                halted_by=next((e["halted_by"] for e in episodes
                                if e["halted_by"] != "completed"), "completed"),
                calls=env.calls)
