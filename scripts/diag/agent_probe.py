"""Profiling tools for the optimizer agent (2 Nov, round 3): understand the teacher before editing it.

Usage: python scripts/diag/agent_probe.py map|heads|confusion [CONFIG.json]
Descriptive only, never a score: the agent's scores come from agent_opt.py --nested. Uses the dev
participants' dev_loso_all rows (P04 / P09 excluded) and teacher seeds 0-5 only.
  map        the teacher's logistic map fitted in each of the 8 leave-one-person-out folds: per head and
             class the mean weight, its spread across folds and stability = mean / sd (|stab| > 3:
             a weight every fold agrees on; small: the weight depends on who is in the training set).
  heads      per head ("neuron"), the class effects MI-SUB, MI-WG, SUB-WG in within-person z units,
             per participant, and how many of the 8 people share the group's sign (8/8 = general,
             4/8 = person-specific). Sorted by the strongest consistent effect.
  confusion  per participant, the confusion matrix (rows true MI / SUB / WG, %) of the full pipeline
             under CONFIG (default: the package) and of the teacher alone.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_opt as AO  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

P8 = AO.GROUPS["G1"] + AO.GROUPS["G2"]
Bc, H = np.load(AO.ROOT / "runs/diag/base_cache.npz"), np.load(AO.ROOT / "runs/diag/screenW5.npz")
idx = Bc["s05_dev_loso_all_idx"]
y, who = AO.y, AO.short


def probe_map(cfg):
    F, names = AO.heads(H, cfg, range(6))
    W = []
    for p in P8:
        tr = np.isin(np.arange(len(y)), idx) & (who != p) & AO.DEV
        W.append(LogisticRegression(C=cfg.get("C", 0.01), max_iter=3000, class_weight="balanced")
                 .fit(F[tr], y[tr]).coef_)
    W = np.array(W)
    m, sd = W.mean(0), W.std(0) + 1e-9
    print(f"{'head':38s}" + "".join(f"{c:>18s}" for c in ("MI mean/stab", "SUB mean/stab", "WG mean/stab")))
    for j, n in enumerate(names):
        print(f"{n:38s}" + "".join(f"{m[c, j]:10.3f}{m[c, j] / sd[c, j]:8.1f}" for c in range(3)))


def probe_heads(cfg):
    F, names = AO.heads(H, cfg, range(6))
    rows = []
    for j, n in enumerate(names):
        eff = {}
        for p in P8:
            r = idx[who[idx] == p]
            z = (F[r, j] - F[r, j].mean()) / (F[r, j].std() + 1e-9)
            mu = [z[y[r] == c].mean() for c in range(3)]
            eff[p] = (mu[0] - mu[1], mu[0] - mu[2], mu[1] - mu[2])
        E = np.array(list(eff.values()))
        g = E.mean(0)
        agree = (np.sign(E) == np.sign(g)).sum(0)
        rows.append((max(abs(g) * agree / 8), n, g, agree))
    print(f"{'head':38s} {'MI-SUB':>14s} {'MI-WG':>14s} {'SUB-WG':>14s}   (mean z effect, people agreeing /8)")
    for _, n, g, a in sorted(rows, reverse=True):
        print(f"{n:38s} " + " ".join(f"{g[k]:8.2f} {a[k]:d}/8 " for k in range(3)))


def probe_confusion(cfg):
    AO.run(Bc, H, cfg, "dev_loso_all", range(6), list(P8))
    L = AO.run.last
    for p in P8:
        r = who[L["ix"]] == p
        yt = y[L["ix"]][r]
        cm = lambda pr: (np.bincount(3 * yt + pr[r], minlength=9).reshape(3, 3)
                         / np.maximum(np.bincount(yt, minlength=3), 1)[:, None] * 100).round(0).astype(int).tolist()
        print(p, "pipeline", cm(L["e"]), "teacher alone", cm(L["t"]))


if __name__ == "__main__":
    cfg = AO.prep(json.loads(Path(sys.argv[2]).read_text())) if len(sys.argv) > 2 else {}
    {"map": probe_map, "heads": probe_heads, "confusion": probe_confusion}[sys.argv[1]](cfg)
