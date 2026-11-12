"""PCROP (ledger s100): own-space T and C members fitted on public windows plus shifted crops; v5.1 replay, seen, six bases.
Usage: CROPS=start0.5[,start2] python studies/12nov/pcrop.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
F = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"))  # noqa: S102  (O, A, SC, ns, REC, NM, staged, members, MODELS, STREAMS)
CROPS = os.environ.get("CROPS", "start0.5").split(","); N = len(O.y); X0, CV0 = O.Xc, O.CV
assert all(np.allclose(np.load(f"{REPO}/data/proteus_windows_{c}.npz")["y"], O.y) for c in CROPS)
Xs = [X0] + [(lambda X: X - X.mean(1, keepdims=True))(np.load(f"{REPO}/data/proteus_windows_{c}.npz")["X"].astype(np.float64)) for c in CROPS]
O.Xc = np.concatenate(Xs); O.CV = [O._cov(*b) for b in O.BANDS]; O.y = np.concatenate([O.y] * len(Xs))  # covariances of all copies (rows k*N + i)
O.SL = np.concatenate([O.SL] * len(Xs)); O.PER = np.concatenate([O.PER] * len(Xs)); O.SES = np.concatenate([O.SES] * len(Xs))
assert np.allclose(O.CV[2][:N], CV0[2]), "the original rows changed"


class CropPersonal(O.Personal):
    """T and C on originals + crops (each copy of a session recentred on its own); S on the originals only."""
    def __init__(self, train_sessions):
        super().__init__(train_sessions + [r + k * N for k in range(1, len(Xs)) for r in train_sessions])
        S = np.concatenate([(O.SL[r] - O.SL[r].mean(0)) / (O.SL[r].std(0) + 1e-9) for r in train_sessions])
        self.S = O.LDA(solver="lsqr", shrinkage=0.7).fit(S, np.concatenate([O.y[r] for r in train_sessions]))
        self.prior = [np.mean([O.recentred(r)[1][b] for r in train_sessions], 0) for b in range(len(O.BANDS))]  # test prior from the originals


res = []
for bn in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    O.Personal_cls = O.Personal; MODELS.clear(); STREAMS.clear(); REC.clear(); out0, _ = ns["system4"](bn, "max", 10); v51, _ = staged(out0)
    keep = O.Personal; O.Personal = CropPersonal; MODELS.clear(); STREAMS.clear(); crop, _ = staged(out0); O.Personal = keep
    r = SC.compare(crop, v51); res.append(r); print(f"{bn}: crops {CROPS} against v5.1: seen {r['seen']:+.2f} (P04 {r['ds'][0]:+.2f}, P10 {r['ds'][1]:+.2f}) new {r['new']:+.3f}", flush=True)
ok = sum(r["seen"] >= 0.3 and r["ds"][1] >= -0.3 for r in res)
print(f"MEAN: seen {np.mean([r['seen'] for r in res]):+.2f} (P04 {np.mean([r['ds'][0] for r in res]):+.2f}, P10 {np.mean([r['ds'][1] for r in res]):+.2f}); bases meeting the lever {ok}/6")
