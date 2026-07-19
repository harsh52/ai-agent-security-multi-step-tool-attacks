# Private-Eval Proxy (community MAJ0RT0M) — future work

Runs our attack.py against a *hypothesized* private guard (content-DLP + LLM-judge) with
anti-gaming scoring: coarse (source, sink, predicate) cells + per-cell LOG-DECAY
(S·(1+ln k)), so N duplicate exfil posts collapse to ~one cell's credit.

STATUS: the git package (`git+https://github.com/aduriseti/kaggle-aas-private-eval-proxy`)
pulls a heavy dep tree (llm-guard -> transformers/torch) that **OOMs the Kaggle T4 kernel**
during install (tried kaggle_gguf backend twice; llama_cpp preinstall didn't help — it's the
proxy package + llm-guard that exhaust RAM). To revive: use the OpenRouter backend (needs an
OPENROUTER_API_KEY + phone-verified internet, ~cents/run) which avoids the local GGUF load, or
extract just the guard/scoring_v2 logic from the repo and run it against our own test bench.

KEY INSIGHT ALREADY EXTRACTED (don't need to run it to know this):
- PUBLIC board rewards THROUGHPUT (repetition) -> v12 single-post throughput engine.
- PRIVATE board (this proxy's guess) rewards DIVERSITY: log-decay caps repetition, so fire
  VARIED predicates (EXFIL + CONFUSED_DEPUTY + others) across distinct (source,sink,predicate)
  cells. github repo: aduriseti/kaggle-aas-private-eval-proxy
