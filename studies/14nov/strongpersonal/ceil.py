"""s247 ceiling: personal members with true training labels, the test session recentred on its whole-session mean
(non-causal EA), fused with the routed output ER x member^lambda; lambda on a grid, the best picked per fold on the test
session itself (oracle). Members:
  TSp<C>   core1's features + LR(C), trained on the person's training sessions only
  TSw<w>   the nine other people (all sessions) + the person's training sessions at sample weight w, LR(10)
  TSwpl    TSw5 + the test session labelled by the mean ER posterior (16 orders x 6 bases), whole session
  PFT      stored s29 nets (5 families x 3 seeds, pf60 / pf200; session 1 only, so folds s1>* only)
  PFT12_60 pft12.py's nets (s12>s3, if fetched)
  ALL      geometric mean of TSw5 and PFT60 (s1>*) / TSw5 and PFT12_60 (else TSp10) (s12>s3)
Usage: python studies/14nov/strongpersonal/ceil.py -> runs/strongpersonal/ceil.npz, ceil.txt"""
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import common as C

LAMS = (0.0, 0.25, 0.5, 1.0, 2.0)
t0 = time.time(); F, _ = C.ts_whole(); print(f"features {F.shape} {time.time() - t0:.0f}s", flush=True)
ER, T = C.routed("ER"), C.routed("T")
PER_ER = np.mean([P for v in ER.values() for P in v], 0)  # mean ER posterior, for the pseudo-label bound
P60, n60 = C.pft("pf60"); P200, n200 = C.pft("pf200"); print(f"PFT nets: pf60 {n60}, pf200 {n200}", flush=True)


def lr(Xtr, ytr, Xte, Cc=10.0, w=None):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=Cc, max_iter=3000))
    m.fit(Xtr, ytr, logisticregression__sample_weight=w); return m.predict_proba(Xte)


def members(q, tr_s, te):
    tr = np.concatenate([C.rows(q, s) for s in tr_s]); oth = np.flatnonzero(C.PER != q); M = {}
    for Cc in (0.1, 10.0):
        P = np.full((C.N, 3), 1 / 3); P[te] = lr(F[tr], C.Y[tr], F[te], Cc); M[f"TSp{Cc:g}"] = P
    for w in (2.0, 5.0):
        idx = np.r_[oth, tr]; sw = np.r_[np.ones(len(oth)), np.full(len(tr), w)]
        P = np.full((C.N, 3), 1 / 3); P[te] = lr(F[idx], C.Y[idx], F[te], 10.0, sw); M[f"TSw{w:g}"] = P
    idx = np.r_[oth, tr, te]; yy = np.r_[C.Y[oth], C.Y[tr], PER_ER[te].argmax(1)]
    sw = np.r_[np.ones(len(oth)), np.full(len(tr), 5.0), np.ones(len(te))]
    P = np.full((C.N, 3), 1 / 3); P[te] = lr(F[idx], yy, F[te], 10.0, sw); M["TSwpl"] = P
    if tr_s == ["1"]:
        M["PFT60"], M["PFT200"] = P60, P200; M["ALL"] = C.fuse(M["TSw5"], P60, 1.0) ** 0.5
    else:
        p12, n12 = C.pft12("pf60")
        if n12:
            M["PFT12_60"] = p12; M["ALL"] = C.fuse(M["TSw5"], p12, 1.0) ** 0.5
        else:
            M["ALL"] = C.fuse(M["TSw5"], M["TSp10"], 1.0) ** 0.5
    M["ALL"] /= M["ALL"].sum(1, keepdims=True)
    return M


def scheme_mean(fn, ref):
    """mean over schemes (equal) of the mean over each scheme's orders of the mean over bases."""
    v = {}
    for o, Ps in ref.items():
        v.setdefault(o.rstrip("0123456789"), []).append(np.mean([fn(P) for P in Ps]))
    return np.mean([np.mean(x) for x in v.values()]), {k: np.mean(x) for k, x in v.items()}


RES, lines, MEM = {}, [], {}
for q, name, tr_s, te_s in C.FOLDS:
    te = C.rows(q, te_s); t1 = time.time(); M = members(q, tr_s, te); MEM.update({f"{q}|{name}|{k}": v[te] for k, v in M.items()})
    er, _ = scheme_mean(lambda P: C.ba(te, P), ER); tt, _ = scheme_mean(lambda P: C.ba(te, P), T)
    lines.append(f"{q} {name}: ER {er:.2f}  T {tt:.2f}  (members {time.time() - t1:.0f}s; the routed path has session 1 public only)")
    for k, Pm in M.items():
        alone = C.ba(te, Pm); fz = []
        for lam in LAMS[1:]:
            fz.append(scheme_mean(lambda P: C.ba(te, C.fuse(P, Pm, lam)), ER)[0])
        best = int(np.argmax(fz)); RES[q, name, k] = (alone, er, tt, fz)
        lines.append(f"   {k:7s} alone {alone:6.2f} | ER x m^lam " + " ".join(f"{l:g}:{f - er:+.2f}" for l, f in zip(LAMS[1:], fz))
                     + f" | oracle {fz[best] - er:+.2f} (lam {LAMS[1:][best]:g})")
    print("\n".join(lines[-len(M) - 1:]), flush=True)

lines.append("\nCEILING vs the routed path (ER), best member at its oracle lambda, per protocol (mean of test sessions):")
for prot, names in (("B (s1>s2, s1>s3)", ("s1>s2", "s1>s3")), ("A (s12>s3)", ("s12>s3",)), ("A-like (s1>s3)", ("s1>s3",))):
    for q in C.SEEN:
        ks = sorted({k for (qq, n, k) in RES if qq == q and n in names})
        best = max(ks, key=lambda k: np.mean([max(RES[q, n, k][3]) - RES[q, n, k][1] for n in names if (q, n, k) in RES]))
        g = np.mean([max(RES[q, n, best][3]) - RES[q, n, best][1] for n in names]); er = np.mean([RES[q, n, best][1] for n in names])
        lines.append(f"  {prot:18s} {q}: ER {er:.2f}, best {best} {g:+.2f}")
open(f"{C.OUT}/ceil.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines[-7:]))
np.savez(f"{C.OUT}/ceil_members.npz", **MEM)
np.savez(f"{C.OUT}/ceil.npz", **{"|".join(k): np.r_[v[0], v[1], v[2], v[3]] for k, v in RES.items()})
