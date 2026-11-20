"""Lane R1 wave 3: leave-one-person-out ridge-penalised mixer weights fitted on the tapped member posteriors of the base arm (pre-chain proxy).
w_k = w0_k exp(theta_k); minimise -class-balanced per-person mean log p_mix(y) + lam |theta|^2 on nine people, apply to the tenth. Extras (full0 only): prior weight 0.25.
usage: python proxy_fit.py DATA_DIR   -> prints held-out proxy BA / logloss vs stock weights, writes arms_w3_{order}.json (chain arms with per-person weights)"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

KEYS = ("c1", "hb", "1", "2", "3", "4", "10", "9", "sc_fbraw", "sc_fb", "sc_sinc", "sc_fbl", "sc_rest")
W0 = np.array([1, 1, .5, .5, 1, 1, 5, 1, 1, 1, 1, 2, 1.0])  # sc_fbl counted twice (FBL2)
EXTRAS = ("reve", "prat", "cbx", "xfam")
LAMS = (0.003, 0.01, 0.03, 0.1)


def load(d, order):
    z = np.load(Path(d) / f"{order}_w1_0_4.npz")
    M = np.stack([z[f"member|{k}"] for k in KEYS], 1).astype(np.float64)  # n,K,3 recording layout
    cols = []
    if order == "full0":
        for e, f in (("reve", "bank_reve"), ("prat", "bank_prat"), ("cbx", "bank_cbx"), ("xfam", "bank_xfam")):
            cols.append(np.load(Path(d) / f"{f}.npz")["full0"])
        M = np.concatenate([M, np.stack(cols, 1)], 1)
    person = np.array([str(p).rsplit("/", 1)[-1] for p in z["person"]])
    return M, z["y"].astype(int), person


def nll(theta, M, y, person, w0, lam):
    w = w0 * np.exp(theta)
    p = (M * w[None, :, None]).sum(1) / w.sum()
    ll = np.log(np.clip(p[np.arange(len(y)), y], 1e-9, None))
    tot = 0.0
    for q in set(person):
        m = person == q
        for c in range(3):
            mm = m & (y == c)
            if mm.any():
                tot += ll[mm].mean() / 3 / len(set(person))
    return -tot + lam * (theta ** 2).sum()


def ba(p, y, person):
    out = []
    for q in sorted(set(person)):
        m = person == q
        out.append(np.mean([(p[m & (y == c)].argmax(1) == c).mean() for c in range(3) if (m & (y == c)).any()]))
    return 100 * np.mean(out)


def main(d):
    people = sorted(set(load(d, "within0")[2]))
    plan = {}
    for order in ("full0", "within0"):
        M, y, person = load(d, order)
        K = M.shape[1]
        w0 = np.concatenate([W0, np.full(K - len(KEYS), 0.25)])
        base_p = (M * w0[None, :, None]).sum(1) / w0.sum()
        print(order, "proxy BA at stock weights", round(ba(base_p, y, person), 3))
        for lam in LAMS:
            held = np.zeros_like(base_p)
            W = {}
            for q in people:
                tr = person != q
                th = minimize(nll, np.zeros(K), args=(M[tr], y[tr], person[tr], w0, lam), method="L-BFGS-B").x
                w = w0 * np.exp(th)
                m = person == q
                held[m] = (M[m] * w[None, :, None]).sum(1) / w.sum()
                W[q] = (dict(zip(KEYS, np.exp(th[:len(KEYS)]).tolist())), dict(zip(EXTRAS, (w0[len(KEYS):] * np.exp(th[len(KEYS):])).tolist())) if K > len(KEYS) else {})
            ll = -np.mean(np.log(held[np.arange(len(y)), y]))
            ll0 = -np.mean(np.log(base_p[np.arange(len(y)), y]))
            print(f"  lam {lam}: held-out proxy BA {ba(held, y, person):.3f} (stock {ba(base_p, y, person):.3f}) logloss {ll:.4f} (stock {ll0:.4f}); mean theta kept: "
                  + " ".join(f"{k}={np.mean([W[q][0][k] for q in people]):.2f}" for k in KEYS))
            plan.setdefault(order, []).append(dict(name=f"learn_lam{lam}", W={q: W[q][0] for q in people}, Wadd={q: W[q][1] for q in people}))
    for o, arms in plan.items():
        Path(d, f"arms_w3_{o}.json").write_text(json.dumps(arms))


if __name__ == "__main__":
    main(sys.argv[1])
