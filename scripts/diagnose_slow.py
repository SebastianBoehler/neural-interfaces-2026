"""Diagnose the slow-waveform block of v2 per participant/session (round 2).

Usage: python scripts/diagnose_slow.py OUT.json
Diagnosis only: it reads P04/P09 labels to explain all_loso and official
results, and nothing here selects a model. Reports
  blocks  - all_loso cell means of v2 refitted on covariance-only,
            slow-only and both feature blocks (same session-aligned features);
  cells   - per participant x session x interface scores of v1, v2 and the
            blocks for the multi-session participants and P09;
  drift   - per session statistics of the session-centred slow features:
            cosine of the class-contrast template with the template of the
            other participants, within-session 5-fold decodability, the
            share of window-DC variance, and the causal-centring lag.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import protocol  # noqa: E402
from models import TangentSpace, make_clf  # noqa: E402
from run_cv import CONFIGS, V2  # noqa: E402

N_SLOW = 41 * 8  # 41 channels x 8 samples of the 1 Hz waveform


def fold_blocks(d, tr, te):
    """Fit v2 features on ``tr``; return causal test predictions per block."""
    m = TangentSpace(d, **V2)
    idx = np.flatnonzero(tr)
    m.log_glob = [L[idx].mean(0) for L in m.logs]
    m.glob = None
    streams = d["stream"][idx]
    F = np.concatenate([m._session_feats(idx[streams == s], False)
                        for s in dict.fromkeys(streams)])
    y = np.concatenate([d["y"][idx[streams == s]] for s in dict.fromkeys(streams)])
    tidx = np.flatnonzero(te)
    tstreams = d["stream"][tidx]
    G = np.concatenate([m._session_feats(tidx[tstreams == s], True)
                        for s in dict.fromkeys(tstreams)])
    order = np.concatenate([tidx[tstreams == s] for s in dict.fromkeys(tstreams)])
    assert (order == tidx).all()
    cut = F.shape[1] - N_SLOW
    out = {}
    for name, sl in (("both", slice(None)), ("cov", slice(0, cut)),
                     ("slow", slice(cut, None))):
        clf = make_clf("lr", 0.1).fit(F[:, sl], y)
        out[name] = clf.predict(G[:, sl])
    return out


def cell_table(d, mask, pred):
    rows = {}
    y = d["y"][mask]
    for key in sorted(set(zip(d["subject"][mask], d["session"][mask],
                              d["interface"][mask]))):
        sel = ((d["subject"][mask] == key[0]) & (d["session"][mask] == key[1])
               & (d["interface"][mask] == key[2]))
        name = f"{key[0].split('/')[-1]}|s{key[1]}|{key[2]}"
        rows[name] = protocol.balanced_accuracy(y[sel], pred[sel])
    return rows


def cos(a, b):
    return float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))


def drift_stats(d, M, k=5.0):
    """Per-session slow-feature statistics (full-session centring)."""
    stats, contrast = {}, {}
    for s in dict.fromkeys(d["stream"]):
        sel = d["stream"] == s
        Z = M[sel] - M[sel].mean(0)
        y = d["y"][sel]
        contrast[s] = np.stack([Z[y == c].mean(0) for c in range(3)])
        W = M[sel].reshape(sel.sum(), 41, -1)
        dc = W.mean(-1)
        n = np.arange(1, sel.sum() + 1)[:, None]
        lag = np.cumsum(M[sel], 0) / (k + n) - M[sel].mean(0)
        clf = make_clf("lr", 0.1)
        stats[s] = {
            "within_session_cv_acc": float(cross_val_score(clf, Z, y, cv=5).mean()),
            "dc_variance_share": float(dc.var(0).sum() * W.shape[-1]
                                       / ((W - W.mean(0)) ** 2).mean(0).sum()),
            "feature_rms": float(np.sqrt((Z ** 2).mean())),
            "causal_lag_over_contrast": float(
                np.linalg.norm(lag, axis=1).mean()
                / np.linalg.norm(contrast[s], axis=1).mean()),
        }
    subj = {s: s.split("|")[0] for s in contrast}
    for s in contrast:
        others = [contrast[t] for t in contrast if subj[t] != subj[s]]
        ref = np.mean(others, 0)
        stats[s]["template_cosine_vs_others"] = cos(contrast[s], ref)
        same = [cos(contrast[s], contrast[t]) for t in contrast
                if subj[t] == subj[s] and t != s]
        if same:  # stability of the participant's own template over sessions
            stats[s]["template_cosine_vs_own_sessions"] = float(np.mean(same))
    return {s.replace("Dreyer2026Proteus/", ""): v for s, v in stats.items()}


def main():
    d = protocol.load()
    res = {"blocks_all_loso": {}, "cells_all_loso": {}}
    preds = {b: np.empty(len(d["y"]), np.int64) for b in ("both", "cov", "slow")}
    for name, tr, te in protocol.folds(d, "all_loso"):
        for b, p in fold_blocks(d, tr, te).items():
            preds[b][te] = p
    full = np.ones(len(d["y"]), bool)
    for b, p in preds.items():
        sc = protocol.score(d["y"], p, d, full)
        per = {}
        for s in sorted(set(d["subject"])):
            m = d["subject"] == s
            per[s.split("/")[-1]] = protocol.score(d["y"][m], p[m], d, m)[
                "balanced_accuracy_cell_mean"]
        res["blocks_all_loso"][b] = {"cell_mean": sc["balanced_accuracy_cell_mean"],
                                     "per_participant": per}
        res["cells_all_loso"][b] = cell_table(d, full, p)
    for v in ("v1", "v2"):  # saved by run_cv.py --protocols all_loso
        f = np.load(ROOT / f"runs/experiments/probs/{v}-all_loso-s0.npz")
        assert (f["index"] == np.arange(len(d["y"]))).all()
        res["cells_all_loso"][v] = cell_table(d, full, f["proba"].argmax(1))
        if v == "v2":  # the "both" block must reproduce the harness v2
            res["both_vs_v2_agreement"] = float(np.mean(
                f["proba"].argmax(1) == preds["both"]))
    keep = ("P04", "P09", "P10")
    for k in res["cells_all_loso"]:
        res["cells_all_loso"][k] = {c: round(v, 4) for c, v in
                                    res["cells_all_loso"][k].items()
                                    if c.startswith(keep)}
    M = CONFIGS["v2"](d).M
    res["drift"] = drift_stats(d, M)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
    for b, r in res["blocks_all_loso"].items():
        print(b, round(r["cell_mean"], 4),
              {k: round(v, 3) for k, v in r["per_participant"].items()})
    for s, r in res["drift"].items():
        print(s, {k: round(v, 3) for k, v in r.items()})


if __name__ == "__main__":
    main()
