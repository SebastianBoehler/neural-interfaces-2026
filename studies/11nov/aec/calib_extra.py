"""CAL diagnostic (exploratory): are members comparable in confidence, and does a temperature-calibrated extra member
help where the raw one hurt? Temperature per member = argmin of the new-person NLL of p^(1/T) (one parameter, fitted
leave-one-person-out for the extra member). Usage: python calib_extra.py NPZ NAME [NPZ NAME ...]"""
import sys
import numpy as np
from scipy.optimize import minimize_scalar
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; L = A.L
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; G1 = [SC.PEOPLE.index(p) for p in SC.G1]; G2 = [SC.PEOPLE.index(p) for p in SC.G2]
y = {p: L.Z[f"y|{p}"] for p in A.PROTOS}; per = L.Z["person|all_loso"]


def temp(P, T):
    q = np.clip(P, 1e-9, 1) ** (1 / T); return q / q.sum(1, keepdims=True)


def fitT(P, yy):
    f = lambda t: -np.mean(np.log(temp(P, np.exp(t))[np.arange(len(yy)), yy]))  # noqa: E731
    return float(np.exp(minimize_scalar(f, bounds=(-2, 4), method="bounded").x))


print("member  acc   mean conf  fitted T (1 = calibrated, > 1 = over-confident)  NLL raw -> calibrated")
mem = A.members("B2_pool", "all_loso"); m0, _ = A.mix("B2_pool", "all_loso", A.FAMS); mem["mix"] = m0
for k, P in mem.items():
    T = fitT(P, y["all_loso"]); nll = lambda Q: -np.mean(np.log(np.clip(Q, 1e-9, 1)[np.arange(len(P)), y["all_loso"]]))  # noqa: E731
    print(f"{str(k):6s} {100 * np.mean(P.argmax(1) == y['all_loso']):5.1f} {P.max(1).mean():8.3f} {T:8.2f} {nll(P):10.3f} -> {nll(temp(P, T)):.3f}")
ref = {bn: A.summary(A.system(bn, A.FAMS)) for bn in env.EXPL}
for f, name in zip(sys.argv[1::2], sys.argv[2::2]):
    Z = np.load(f); h = "lr3"; P = {p: Z[f"{h}|{p}"].astype(np.float64) for p in A.PROTOS}
    for p in A.PROTOS:
        assert (Z[f"index|{p}"] == IDX[p]).all()
    Tall = fitT(P["all_loso"], y["all_loso"]); C = {"all_loso": np.zeros_like(P["all_loso"]), "seen_session": temp(P["seen_session"], Tall)}
    for q in np.unique(per):  # leave-one-person-out temperature for the new-person rows
        m = per == q; C["all_loso"][m] = temp(P["all_loso"][m], fitT(P["all_loso"][~m], y["all_loso"][~m]))
    print(f"\n{name}: acc {100 * np.mean(P['all_loso'].argmax(1) == y['all_loso']):.1f}, mean conf raw {P['all_loso'].max(1).mean():.3f}, T {Tall:.2f}, mean conf calibrated {C['all_loso'].max(1).mean():.3f}")
    for tag, Q in (("raw", P), ("calibrated", C)):
        A.EXTRA[name + tag] = Q; line = []
        for w in (0.5, 1.0, 2.0, 4.0):
            dn, ds = [], []
            for bn in env.EXPL:
                s = A.summary(A.system(bn, A.FAMS, extra=[(name + tag, w)])); dn.append(s["dn"] - ref[bn]["dn"]); ds.append(s["ds"] - ref[bn]["ds"])
            dn, ds = np.mean(dn, 0), np.mean(ds, 0); line.append(f"w{w}: {0.5 * dn.mean() + 0.5 * ds.mean():+.2f} (new {dn.mean():+.2f}: G1 {dn[G1].mean():+.2f}, G2 {dn[G2].mean():+.2f}; seen {ds.mean():+.2f})")
        print(f"  {tag:10s} " + " | ".join(line), flush=True)
