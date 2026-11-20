"""Lane Z lever (b): nested leave-one-person-out mixer weights from the baseline member banks of final_z.py (job prof).
usage: python fit_weights.py CHAIN_DIR FEAT_NPZ   -> CHAIN_DIR/weights_ORDER.npz
For each held-out person q the weights are fitted on the other nine people only (their windows and labels), never on q.
  ll<pct>   EM for mixture weights (maximum likelihood of the true class under sum_k w_k p_k), start at the stock weights,
            shrunk toward the stock weights: w = (1 - pct/100) w_stock + pct/100 w_fit
  acc<pct>  w_k = w_stock_k * exp(10 * (acc_k - mean acc)) on the nine people's window accuracy of each member, same shrink
Weights are unnormalised mixer weights (the chain divides by their sum); keys follow the base arm's mixer items in order.
Members are the held-new members of every person (leave-one-person-out by construction), the fit uses other people's windows only."""
import json
import sys
from pathlib import Path

import numpy as np

STAGE = "base|member|"


def fit(P, y, w0, iters=300):
    w = w0 / w0.sum()
    lik = np.stack([p[np.arange(len(y)), y] for p in P], 1)  # n x k
    for _ in range(iters):
        r = lik * w
        r /= r.sum(1, keepdims=True)
        w = r.mean(0)
    return w * w0.sum()


def main(chain, feat):
    z = np.load(feat, allow_pickle=False)
    y, subject = z["y"].astype(int), np.asarray([s.rsplit("/", 1)[-1] for s in z["subject"]])
    people = sorted(set(subject))
    for order in ("full0", "within0"):
        bank = np.load(Path(chain) / f"{order}_prof.npz", allow_pickle=False)
        info = json.loads((Path(chain) / f"{order}_prof.json").read_text())
        out = {}
        for system in ("stock", "hardcb"):
            items = info["arms"][f"{system}|base"]["items"]
            keys = [k for k, _ in items]
            w0 = np.array([w for _, w in items])
            P = [bank[f"{system}|{STAGE}{k}"].astype(np.float64) for k in keys]
            acc = np.array([[np.mean(p[subject == q].argmax(1) == y[subject == q]) for p in P] for q in people])
            for pct in (50, 100):
                lam = pct / 100
                Wll, Wacc = [], []
                for i, q in enumerate(people):
                    tr = subject != q
                    wf = fit([p[tr] for p in P], y[tr], w0)
                    Wll.append((1 - lam) * w0 + lam * wf)
                    a = np.delete(acc, i, 0).mean(0)
                    wa = w0 * np.exp(10 * (a - a.mean()))
                    wa *= w0.sum() / wa.sum()
                    Wacc.append((1 - lam) * w0 + lam * wa)
                out[f"{system}|ll{pct}|W"], out[f"{system}|acc{pct}|W"] = np.array(Wll), np.array(Wacc)
                for v in ("ll", "acc"):
                    out[f"{system}|{v}{pct}|keys"] = np.array([str(k) for k in keys])
            print(order, system, "stock w", dict(zip(keys, np.round(w0, 2))))
            print("  ll100 mean w", dict(zip(keys, np.round(out[f"{system}|ll100|W"].mean(0), 2))))
        np.savez(Path(chain) / f"weights_{order}.npz", **out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
