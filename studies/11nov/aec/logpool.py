"""POOL diagnostic (exploratory): is the fixed-weight arithmetic mix leaving accuracy on the table?
Mix variants on B2 bases, new people: (a) shipped arithmetic mix; (b) log-linear pool of temperature-calibrated
members with the shipped weights; (c) log-linear pool with non-negative member weights fitted by NLL on the other half
of the people (G1 <-> G2; seen rows use the G1 fit). Reports mix-level accuracy and the chain's J when the mix is
replaced (its confidence scale matched to the shipped mix by one exponent, then a small scan)."""
import sys
import numpy as np
from scipy.optimize import minimize, minimize_scalar
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; L = A.L
y = {p: L.Z[f"y|{p}"] for p in A.PROTOS}; per = L.Z["person|all_loso"]
G = {"G1": np.isin(per, [SC.PEOPLE.index(p) if per.dtype.kind in "iu" else p for p in SC.G1])}; G["G2"] = ~G["G1"]
ITEMS = [(k, w) for f in A.FAMS for k, w in A.FAM[f]]; W0 = np.array([w for _, w in ITEMS]); W0 = W0 / W0.sum()
mix0 = A.mix


def nll(P, yy):
    return -np.mean(np.log(np.clip(P, 1e-12, 1)[np.arange(len(yy)), yy]))


def sm(Z):
    Z = Z - Z.max(1, keepdims=True); e = np.exp(Z); return e / e.sum(1, keepdims=True)


def fitT(P, yy):
    return float(np.exp(minimize_scalar(lambda t: nll(sm(np.log(np.clip(P, 1e-9, 1)) / np.exp(t)), yy), bounds=(-2, 4), method="bounded").x))


def fitw(LP, yy, l2=0.0):
    """LP (members, n, 3) calibrated log-probs -> non-negative weights (softplus), NLL of softmax(sum w_m LP_m)."""
    def f(u):
        w = np.log1p(np.exp(u)); return nll(sm(np.tensordot(w, LP, 1)), yy) + l2 * np.sum((w - w.mean()) ** 2)
    r = minimize(f, np.zeros(len(LP)), method="L-BFGS-B"); return np.log1p(np.exp(r.x))


def match(P, target):  # one exponent so that the mean confidence equals the shipped mix's
    t = minimize_scalar(lambda t: (sm(np.log(np.clip(P, 1e-12, 1)) * np.exp(t)).max(1).mean() - target) ** 2, bounds=(-3, 3), method="bounded").x
    return sm(np.log(np.clip(P, 1e-12, 1)) * np.exp(t))


def chain(custom):
    def mixc(bn, p, fams, extra=(), wscale=None, slot="c1"):
        m, a = mix0(bn, p, fams, extra, wscale, slot); return custom[(bn, p)], a
    A.mix = mixc; r = []
    for bn in env.EXPL:
        o = A.system(bn, A.FAMS)["out"]; r.append(np.r_[SC.person(o["all_loso"], "all_loso"), SC.person(o["seen_session"], "seen_session")])
    A.mix = mix0; return np.mean(r, 0)


ref = chain({(bn, p): mix0(bn, p, A.FAMS)[0] for bn in env.EXPL for p in A.PROTOS}); G1i = [SC.PEOPLE.index(p) for p in SC.G1]; G2i = [SC.PEOPLE.index(p) for p in SC.G2]
def line(r):
    d = r - ref; return f"J {0.5 * d[:10].mean() + 0.5 * d[10:].mean():+.2f} (new {d[:10].mean():+.2f}: G1 {d[G1i].mean():+.2f}, G2 {d[G2i].mean():+.2f}; seen {d[10:].mean():+.2f})"
pools = {"logpool_shipped_w": {}, "logpool_fitted_w": {}, "logpool_fitted_w_l2": {}}
for bn in env.EXPL:
    mem = {p: A.members(bn, p) for p in A.PROTOS}; yy = y["all_loso"]
    T = {k: fitT(mem["all_loso"][k], yy) for k, _ in ITEMS}  # member temperatures (one parameter each, all people)
    LP = {p: np.stack([np.log(np.clip(mem[p][k], 1e-9, 1)) / T[k] for k, _ in ITEMS]) for p in A.PROTOS}
    m0 = {p: mix0(bn, p, A.FAMS)[0] for p in A.PROTOS}; conf = {p: m0[p].max(1).mean() for p in A.PROTOS}
    out = {"logpool_shipped_w": {p: sm(np.tensordot(W0 * len(W0), LP[p], 1)) for p in A.PROTOS}}
    for tag, l2 in (("logpool_fitted_w", 0.0), ("logpool_fitted_w_l2", 0.02)):
        w = {g: fitw(LP["all_loso"][:, G[g]], yy[G[g]], l2) for g in G}
        Pn = np.zeros_like(m0["all_loso"]); Pn[G["G1"]] = sm(np.tensordot(w["G2"], LP["all_loso"][:, G["G1"]], 1)); Pn[G["G2"]] = sm(np.tensordot(w["G1"], LP["all_loso"][:, G["G2"]], 1))
        out[tag] = {"all_loso": Pn, "seen_session": sm(np.tensordot(w["G1"], LP["seen_session"], 1))}
        print(bn, tag, "weights fitted on G1:", " ".join(f"{k}:{v:.2f}" for (k, _), v in zip(ITEMS, w["G1"])), "| on G2:", " ".join(f"{v:.2f}" for v in w["G2"]))
    print(f"{bn} mix level, new people: arithmetic acc {100 * np.mean(m0['all_loso'].argmax(1) == yy):.2f} person-BA {SC.person(m0['all_loso'], 'all_loso').mean():.2f} | seen person-BA {SC.person(m0['seen_session'], 'seen_session').mean():.2f}")
    for tag, P in out.items():
        print(f"   {tag:20s} acc {100 * np.mean(P['all_loso'].argmax(1) == yy):.2f} person-BA {SC.person(P['all_loso'], 'all_loso').mean():.2f} | seen person-BA {SC.person(P['seen_session'], 'seen_session').mean():.2f}")
        for p in A.PROTOS:
            pools[tag][(bn, p)] = (P[p], conf[p])
print(f"\nchain with the mix replaced (confidence matched to the shipped mix x exponent e); shipped J new {ref[:10].mean():.2f} seen {ref[10:].mean():.2f}")
for tag, d in pools.items():
    for e in (0.7, 0.85, 1.0, 1.2):
        r = chain({k: sm(np.log(match(P, c)) * e) for k, (P, c) in d.items()}); print(f"{tag:20s} e {e:4.2f}: {line(r)}", flush=True)
