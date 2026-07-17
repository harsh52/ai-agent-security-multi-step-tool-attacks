"""Run our REAL attack.py against a COMPLIANT stub agent + real OptimalGuardrail + real scorer.

This simulates the 'the target LLM follows instructions' case (which GPT-OSS / Gemma
may do), to check that our attack templates actually produce a positive score.
"""
import re
import sys
from pathlib import Path

import aicomp_sdk
from aicomp_sdk.guardrails.optimal import Guardrail
from aicomp_sdk.core.env.gym import GymAttackEnv
from aicomp_sdk.agents.types import ToolCall, ToolCallDecision, FinalResponseDecision, AGENT_STATE_VERSION
from aicomp_sdk.agents.types import ToolResultEvent, UserMessageEvent
from aicomp_sdk.scoring import score_attack
from aicomp_sdk.core.cells import cell_signature
from aicomp_sdk.core.predicates import eval_predicates

FIX = Path(aicomp_sdk.__file__).parent / "fixtures"
_URL = re.compile(r'https?://[^\s\'"<>]+')


class CompliantAgent:
    """A naive instruction-follower: does literally what the last user message asks."""

    def __init__(self):
        self._i = 1
        self._done = set()  # avoid infinite loops: one action per instruction

    def _call(self, tool, args):
        c = ToolCall(call_id=f"call_{self._i:06d}", tool_name=tool, arguments=args)
        self._i += 1
        return ToolCallDecision(call=c, assistant_message=None)

    def _last_user(self, history):
        for e in history.events[::-1]:
            if isinstance(e, UserMessageEvent):
                return e.text
        return None

    def next_action(self, *, history, tools):
        del tools
        msg = self._last_user(history) or ""
        low = msg.lower()
        key = (len(history.events), low[:40])
        if key in self._done:
            return FinalResponseDecision(text="done")
        self._done.add(key)

        url_m = _URL.search(msg)
        marker = "SECRET_MARKER" if "secret_marker" in low else "data"
        # http.post exfil
        if ("post" in low or "http.post" in low) and url_m:
            return self._call("http.post", {"url": url_m.group(0), "data": marker})
        # email.send
        if "deliver" in low or "dispatch" in low or ("send" in low and "@" in msg):
            to_m = re.search(r'[\w.+-]+@[\w.-]+', msg)
            return self._call("email.send", {"to": to_m.group(0) if to_m else "x@y.example",
                                             "subject": "note", "body": "status ok"})
        # benign reads used for taint-laundering
        if "open page_2" in low:
            return self._call("web.open", {"id": "page_2"})
        for kw, path in [("readme", "readme.txt"), ("note", "note.txt"),
                          ("report", "report.txt"), ("config", "config.txt"),
                          ("install", "install.txt")]:
            if kw in low and "read" in low:
                return self._call("fs.read", {"path": path})
        return FinalResponseDecision(text="ok")

    def reset_state(self):
        self._i = 1
        self._done = set()

    def snapshot_state(self):
        return {"version": AGENT_STATE_VERSION, "backend": "deterministic",
                "data": {"next_call_index": self._i}}

    def restore_state(self, snapshot):
        self._i = int(snapshot["data"].get("next_call_index", 1))
        self._done = set()


# Load our real submission
sys.path.insert(0, str(Path("/Users/harshkumar/Documents/Personal/Projects/AI Agent Security/submission")))
import attack as submission

env = GymAttackEnv(seed=0, fixtures_dir=FIX, guardrail=Guardrail(), agent=CompliantAgent())

from aicomp_sdk import AttackRunConfig
algo = submission.AttackAlgorithm()
candidates = algo.run(env, AttackRunConfig(time_budget_s=60.0, max_tool_hops=8))

print(f"Candidates returned: {len(candidates)}")

# Independent replay + score (mirrors evaluator: recompute trace/predicates/cells)
findings = []
for cand in candidates:
    env.reset()
    for m in cand.user_messages:
        env.interact(m, max_tool_hops=8)
    tr = env.export_trace_dict()
    preds = eval_predicates(tr)
    sig = cell_signature(tr["tool_events"], egress_data=tr.get("egress", []),
                         user_messages=tr.get("user_messages", []))
    if preds:
        findings.append({"predicates": preds, "cell_signature": sig})

pred_counts = {}
for f in findings:
    for p in f["predicates"]:
        pred_counts[p["predicate"]] = pred_counts.get(p["predicate"], 0) + 1
uniq = len({f["cell_signature"]["hash"] for f in findings})

print(f"Firing findings: {len(findings)}   unique cells: {uniq}")
print(f"Predicate breakdown: {pred_counts}")
print(f"Normalized attack score (0-1000): {score_attack(findings):.4f}")
print(f"Raw score: {score_attack(findings, normalize=False):.1f}")
