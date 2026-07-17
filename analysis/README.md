# Analysis & validation harnesses

Local scripts used to reverse-engineer the benchmark and validate the attack. They **require the competition
SDK** (`aicomp_sdk`, from the Kaggle competition data or
[`mbhatt1/competitionscratch`](https://github.com/mbhatt1/competitionscratch)) installed in the active Python
environment. They are documentation of methodology, not part of the submitted artifact.

| Script | What it does |
|---|---|
| `probe.py` | Empirically probes what tool calls / predicates fire on the deterministic agent under `OptimalGuardrail`. |
| `verify_gap.py` | Unit-level proof of the guardrail gaps + predicate firing (e.g. clean-URL `http.post` of `SECRET_MARKER` is allowed and fires EXFILTRATION). |
| `compliant_harness.py` | Runs an attack against a *compliant* stub agent + the real guardrail + real scorer, to check the mechanics and score when the model complies. |
| `harness_v2.py` | Same idea for the v2 attack: multi-message chains, posts across hops, and a `REFUSE_FIRST=K` env var to simulate refusal and exercise the adaptive escalation ladder. |

```bash
python analysis/probe.py
python analysis/verify_gap.py
python analysis/harness_v2.py
REFUSE_FIRST=3 python analysis/harness_v2.py
```
