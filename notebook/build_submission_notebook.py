"""Build the Kaggle submission notebook from submission/attack.py.

Cells: setup -> write attack.py -> py_compile -> SELF-CHECK (commit-time visibility)
-> placeholder submission.csv + conditional serve().

The self-check runs at COMMIT time (visible in the notebook log) and prints the config,
candidate count, projected posts, and projected score of whatever attack.py is being
submitted. The real scoring rerun's logs are hidden by Kaggle, so this commit-time
summary is the only at-a-glance confirmation of "what am I actually submitting".

Usage:  python notebook/build_submission_notebook.py [title]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTACK = ROOT / "submission" / "attack.py"
OUT = ROOT / "notebook" / "ai-agent-security-first-submission.ipynb"


def _md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def _code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.splitlines(keepends=True)}


SETUP = (
    "import sys, glob\n"
    "from pathlib import Path\n"
    "sys.argv = [sys.argv[0]]\n"
    "for cand in glob.glob('/kaggle/input/**/kaggle_evaluation', recursive=True):\n"
    "    root = str(Path(cand).parent)\n"
    "    if root not in sys.path:\n        sys.path.insert(0, root)\n"
    "    print('Dataset root:', root)\n    break\n"
    "WORKING = Path('/kaggle/working'); WORKING.mkdir(parents=True, exist_ok=True)\n"
    "print('Setup complete')\n"
)

COMPILE = (
    "import py_compile\n"
    "py_compile.compile(str(attack_path), doraise=True)\n"
    "print('attack.py compiled OK')\n"
)

# The self-check never raises (a broken self-check must NOT fail the commit/submission).
SELFCHECK = (
    "# --- SELF-CHECK (commit-time visibility; real scoring runs hidden during rerun) ---\n"
    "print('=' * 60)\n"
    "print('[SELF-CHECK] confirming what is being submitted')\n"
    "try:\n"
    "    import importlib.util as _iu\n"
    "    _sp = _iu.spec_from_file_location('selfcheck_attack', str(attack_path))\n"
    "    _mod = _iu.module_from_spec(_sp); _sp.loader.exec_module(_mod)\n"
    "    assert hasattr(_mod, 'AttackAlgorithm'), 'AttackAlgorithm missing'\n"
    "    _M   = getattr(_mod, 'M', None)\n"
    "    _RS  = getattr(_mod, 'REPLAY_SAFE', None)\n"
    "    _LAT = getattr(_mod, 'ASSUMED_REPLAY_LAT_S', None)\n"
    "    _RB  = getattr(_mod, 'REPLAY_BUDGET_S', 9000.0)\n"
    "    _FN  = getattr(_mod, 'FORCE_N', None)\n"
    "    _PPM = getattr(_mod, 'POSTS_PER_MSG', 1)\n"
    "    print(f'[SELF-CHECK] AttackAlgorithm present OK | M={_M} msgs/chain, '\n"
    "          f'posts/msg={_PPM}, FORCE_N={_FN}, REPLAY_SAFE={_RS}, assumed={_LAT}s/post')\n"
    "    _N = _FN if _FN else (int(_RS * _RB / (_M * _LAT)) if (_M and _RS and _LAT) else None)\n"
    "    if _M and _N:\n"
    "        _posts = _N * _M * (_PPM or 1)\n"
    "        _proj_cell = min(1000, _posts * 16 / 200)\n"
    "        print(f'[SELF-CHECK] will return N={_N} candidates x {_M} msgs x {_PPM} posts/msg = {_posts} posts/cell')\n"
    "        print(f'[SELF-CHECK] projected score ~{_proj_cell:.0f}/cell (if it lands)')\n"
    "    # sample the first user message so we can eyeball the wording\n"
    "    try:\n"
    "        _algo = _mod.AttackAlgorithm()\n"
    "        _msg = _algo._multi_msg() if hasattr(_algo, '_multi_msg') else _mod._VECTORS[0](_algo._next_url())\n"
    "        print(f'[SELF-CHECK] sample message: {_msg[:110]!r}')\n"
    "    except Exception as _e:\n"
    "        print('[SELF-CHECK] (sample-message note:', type(_e).__name__, str(_e)[:60], ')')\n"
    "    print('[SELF-CHECK] READY — packaged OK. Real attack runs during the hidden rerun.')\n"
    "except Exception as _e:\n"
    "    print('[SELF-CHECK] WARNING:', type(_e).__name__, str(_e)[:100])\n"
    "print('=' * 60)\n"
)

SERVE = (
    "import os, csv\n"
    "with open(WORKING / 'submission.csv', 'w', newline='') as f:\n"
    "    w = csv.writer(f); w.writerow(['Id', 'Score'])\n"
    "    for row in ['gpt_oss_public', 'gpt_oss_private', 'gemma_public', 'gemma_private']:\n"
    "        w.writerow([row, 0.0])\n"
    "print('placeholder submission.csv written')\n"
    "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    import kaggle_evaluation.jed_attack_134815.jed_attack_inference_server as inference_server\n"
    "    inference_server.JEDAttackInferenceServer().serve()\n"
)


def build(title: str) -> None:
    attack_src = ATTACK.read_text()
    assert "'''" not in attack_src, "triple-single-quote breaks the r''' embed"
    assert "<|" not in attack_src, "raw control tokens are unsafe"
    write_cell = (
        "attack_code = r'''\n" + attack_src + "\n'''\n"
        "attack_path = WORKING / 'attack.py'\n"
        "attack_path.write_text(attack_code)\n"
        "assert 'class AttackAlgorithm' in attack_code\n"
        "print('attack.py written:', len(attack_code), 'chars')\n"
    )
    nb = {
        "cells": [
            _md(title),
            _code(SETUP), _code(write_cell), _code(COMPILE), _code(SELFCHECK), _code(SERVE),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    # round-trip check
    nb2 = json.loads(OUT.read_text())
    wc = "".join(nb2["cells"][2]["source"])
    ns: dict = {"WORKING": Path("/tmp")}
    exec(wc.split("attack_path = ")[0], ns)
    assert ns["attack_code"].strip() == attack_src.strip(), "embed mismatch"
    print(f"built {OUT.name} | embed OK | cells={len(nb['cells'])} (incl. self-check)")


if __name__ == "__main__":
    title = sys.argv[1] if len(sys.argv) > 1 else "# AI Agent Security — Submission"
    build(title)
