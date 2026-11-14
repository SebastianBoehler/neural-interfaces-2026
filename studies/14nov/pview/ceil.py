"""PV1 (ledger PV1): ceilings of personal against population views, standalone, CPU, public data.
For every fold of data.folds(XS | R42 | IF): train with labels on the train rows, classify the test rows.
  sources  pers = the train rows (the person's own data); pop = the other nine people (all their sessions)
  views    gaze32, GE48 (SessionEM's view), pc128 (TSEM's view)
  methods  LR  = logistic regression (C fixed at 0.05) on per-session z-scored features
           TPL = SessionEM-style Gaussian class templates (class means, shared covariance shrunk 0.7 to scaled identity,
                 equal priors), i.e. what a template prior from window 0 would give if used alone
  test standardisation: causal (data.zcausal: windows < t plus the train prior as K=10 pseudo-windows) in each delivery
  order of data.ORDERS (rec, within 0-3, full 0-1, runs 0-1; global permutations of runs/shuffle_perms.npz restricted to
  the test rows); 'whole' = the test part's own full stats (non-causal reference).
  eye = the population eye net (stored, LOSO-trained) on the same test rows.
Usage: [PROTOS=XS,R42,IF] [PEOPLE=P04,P10] python studies/14nov/pview/ceil.py  -> runs/pview/ceil.npz + table"""
import os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402

PROTOS = os.environ.get("PROTOS", "XS,R42,IF").split(","); WHO = os.environ.get("PEOPLE", ",".join(D.PEOPLE)).split(",")
K, C_LR, SHR = 10, 0.05, 0.7
RANKS = {o: D.order_rank(o) for o in D.ORDERS}


class TPL:
    def fit(self, X, y):
        self.mu = np.stack([X[y == k].mean(0) for k in range(3)]); R = X - self.mu[y]
        C = np.cov(R.T); C = (1 - SHR) * C + SHR * np.eye(len(C)) * np.trace(C) / len(C); self.Ci = np.linalg.inv(C); return self

    def predict_proba(self, X):
        Wt = self.Ci @ self.mu.T; s = X @ Wt - 0.5 * (self.mu * Wt.T).sum(1); s -= s.max(1, keepdims=True); e = np.exp(s)
        return e / e.sum(1, keepdims=True)


def fit(method, X, y):
    return TPL().fit(X, y) if method == "TPL" else LogisticRegression(C=C_LR, max_iter=3000).fit(X, y)


def groups_of(rows):
    return [rows[(D.SESS[rows] == s)] for s in dict.fromkeys(D.SESS[rows])]


def test_preds(model, F, te, prior):
    """-> {order: predicted labels on te (te's layout)} plus 'whole'."""
    out = {}
    for o in D.ORDERS:
        pr = np.empty(len(te), int)
        for g in groups_of(te):  # one test session at a time, in its delivery order
            ix = np.flatnonzero(np.isin(te, g)); ix = ix[np.argsort(RANKS[o][te[ix]], kind="stable")]
            pr[ix] = model.predict_proba(D.zcausal(F[te[ix]], prior, K)).argmax(1)
        out[o] = pr
    pr = np.empty(len(te), int)
    for g in groups_of(te):
        ix = np.flatnonzero(np.isin(te, g)); Fz = D.ztrain(F[te[ix]], [np.arange(len(ix))]); pr[ix] = model.predict_proba(Fz).argmax(1)
    out["whole"] = pr; return out


def main():
    t0 = time.time(); R = {}; POP = {}
    for proto in PROTOS:
        for person, name, tr, te in D.folds(proto):
            if person not in WHO:
                continue
            for view, F in D.VIEWS.items():
                # personal: the person's own train rows; prior = their per-session stats
                gtr = groups_of(tr); Xtr = D.ztrain(F, gtr)[tr]
                for meth in ("LR", "TPL"):
                    m = fit(meth, Xtr, D.Y[tr]); prd = test_preds(m, F, te, D.sess_stats(F, gtr))
                    for o, pr in prd.items():
                        R[proto, person, name, view, meth, "pers", o] = D.ba(te, pr)
                    # population: other nine people in the coordinates of this person's rows (cached per person)
                    if (person, view, meth) not in POP:
                        prow, PF = D.pop_view(view, person); g = groups_of(prow); pos = {r: i for i, r in enumerate(prow)}
                        gi = [np.array([pos[r] for r in x]) for x in g]
                        POP[person, view, meth] = (fit(meth, D.ztrain(PF, gi), D.Y[prow]), D.sess_stats(PF, gi))
                    pm, ppr = POP[person, view, meth]
                    for o, pr in test_preds(pm, F, te, ppr).items():
                        R[proto, person, name, view, meth, "pop", o] = D.ba(te, pr)
            R[proto, person, name, "eye", "net", "pop", "rec"] = D.ba(te, D.EYE[te].argmax(1))
            print(f"{proto} {person} {name} done ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(D.OUT, exist_ok=True)
    np.savez(D.OUT + f"/ceil_{'-'.join(PROTOS)}_{len(WHO)}.npz", keys=np.array(["|".join(k) for k in R]), vals=np.array(list(R.values())))
    report(R)


def report(R):
    SH = [o for o in D.ORDERS if o != "rec"]
    for proto in PROTOS:
        ppl = [p for p in WHO if any(k[0] == proto and k[1] == p for k in R)]
        print(f"\n{proto}: cell-mean BA (%), mean over folds; causal standardisation, shuffled orders mean [rec] (whole)")
        print("person " + " ".join(f"{v[:6]:>6s}-{m[:3]:3s} pers/pop" for v in D.VIEWS for m in ("LR", "TPL")) + "   eye")
        for p in ppl + ["mean"]:
            P = ppl if p == "mean" else [p]
            def g(v, m, s, os_):
                return np.mean([np.mean([R[k] for k in R if k[0] == proto and k[1] == q and k[3] == v and k[4] == m and k[5] == s and k[6] in os_]) for q in P])
            row = " ".join(f"{g(v, m, 'pers', SH):6.1f}/{g(v, m, 'pop', SH):4.1f}     " for v in D.VIEWS for m in ("LR", "TPL"))
            print(f"{p:6s} {row} {g('eye', 'net', 'pop', ['rec']):5.1f}", flush=True)
        for v in D.VIEWS:
            for m in ("LR", "TPL"):
                def h(s, os_):
                    return np.mean([np.mean([R[k] for k in R if k[0] == proto and k[1] == q and k[3] == v and k[4] == m and k[5] == s and k[6] in os_]) for q in ppl])
                print(f"  {v} {m}: pers rec {h('pers', ['rec']):.1f} within {h('pers', SH[:4]):.1f} full {h('pers', SH[4:6]):.1f} runs {h('pers', SH[6:]):.1f} whole {h('pers', ['whole']):.1f} | "
                      f"pop rec {h('pop', ['rec']):.1f} within {h('pop', SH[:4]):.1f} full {h('pop', SH[4:6]):.1f} whole {h('pop', ['whole']):.1f}")


if __name__ == "__main__":
    main()
