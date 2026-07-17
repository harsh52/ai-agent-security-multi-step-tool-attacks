"""Run v2 attack.py against a compliant stub agent + real OptimalGuardrail + real scorer.

Simulates: one post per exfil message, benign email.send for deputy messages, and a
configurable 'refuse first K messages' to exercise the escalation ladder.
"""
import re, sys, os
from pathlib import Path
import aicomp_sdk
from aicomp_sdk.guardrails.optimal import Guardrail
from aicomp_sdk.core.env.gym import GymAttackEnv
from aicomp_sdk.agents.types import (ToolCall, ToolCallDecision, FinalResponseDecision,
                                     AGENT_STATE_VERSION, UserMessageEvent)
from aicomp_sdk.scoring import score_attack
from aicomp_sdk.core.cells import cell_signature
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.attacks import AttackRunConfig

FIX = Path(aicomp_sdk.__file__).parent / "fixtures"
_URL = re.compile(r'https?://[^\s\'"<>]+')
REFUSE_FIRST = int(os.getenv("REFUSE_FIRST", "0"))  # refuse until this many user msgs seen


class CompliantAgent:
    def __init__(self):
        self._i = 1
        self._seen_msgs = 0
        self._done = set()

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
        # count distinct user turns (once per new user message)
        key = (len([e for e in history.events if isinstance(e, UserMessageEvent)]), low[:30])
        if key not in self._done:
            self._seen_msgs += 1
            self._done.add(key)
            self._acted = False
        # Simulate refusal for the first K user messages.
        if self._seen_msgs <= REFUSE_FIRST:
            return FinalResponseDecision(text="I'm sorry, I cannot comply with that request.")
        # one action per user message (no infinite hop loop in this stub)
        akey = ("act", key)
        if akey in self._done:
            return FinalResponseDecision(text="OK")
        self._done.add(akey)
        um = _URL.search(msg)
        if ("http.post" in low or "post" in low) and um:
            return self._call("http.post", {"url": um.group(0), "data": "SECRET_MARKER"})
        if "forward" in low or "deliver" in low or "dispatch" in low:
            to = re.search(r'[\w.+-]+@[\w.-]+', msg)
            return self._call("email.send", {"to": to.group(0) if to else "x@y.example",
                                             "subject": "note", "body": "status ok"})
        if "open web page" in low or "page_2" in low:
            return self._call("web.open", {"id": "page_2"})
        return FinalResponseDecision(text="OK")

    def reset_state(self):
        self._i = 1; self._seen_msgs = 0; self._done = set()

    def snapshot_state(self):
        return {"version": AGENT_STATE_VERSION, "backend": "deterministic",
                "data": {"next_call_index": self._i}}

    def restore_state(self, s):
        self._i = int(s["data"].get("next_call_index", 1)); self._seen_msgs = 0; self._done = set()


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))
import attack as sub
sub.MAX_CANDIDATES = 40  # keep the stub test bounded while exercising search+probes

env = GymAttackEnv(seed=0, fixtures_dir=FIX, guardrail=Guardrail(), agent=CompliantAgent())
algo = sub.AttackAlgorithm()
# small budget so the local (fast) stub loop terminates; measured search respects it
cands = algo.run(env, AttackRunConfig(time_budget_s=300.0, max_tool_hops=8))
print(f"REFUSE_FIRST={REFUSE_FIRST} | candidates returned: {len(cands)}")

# Independent replay + score (mirrors gateway)
findings, pred_counts = [], {}
for c in cands:
    env.reset()
    for m in c.user_messages:
        env.interact(m, max_tool_hops=8)
    tr = env.export_trace_dict()
    preds = eval_predicates(tr)
    if preds:
        sig = cell_signature(tr["tool_events"], egress_data=tr.get("egress", []),
                             user_messages=tr.get("user_messages", []))
        findings.append({"predicates": preds, "cell_signature": sig})
        for p in preds:
            pred_counts[p["predicate"]] = pred_counts.get(p["predicate"], 0) + 1

uniq = len({f["cell_signature"]["hash"] for f in findings})
print(f"firing candidates: {len(findings)} | unique cells: {uniq} | predicates: {pred_counts}")
print(f"normalized score: {score_attack(findings):.3f} | raw: {score_attack(findings, normalize=False):.1f}")
