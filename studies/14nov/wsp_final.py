"""WSP FINAL (ledger s207, stage 2, edit families 1 and 2): a final-layer edit of one chain component, fitted per held-out
person on the other nine people's outputs (as the package would fit it on all ten public people).
The edit is z' = t (I + B) z + c on the component's logits z (= log posteriors up to a per-row constant). B's rows sum to
zero, so it is invariant to that constant; it is exactly W' = t (I + B) W, b' = t (I + B) b + c in the final layer.
Objective:
  acc   the component's own log loss (family 1)
  corr  log loss + LAM x (correlation of its true-class log-posterior residual with the rest of the members' residual
        + lag-1 autocorrelation of that residual within runs in recording order)                          (family 2)
Both carry a ridge towards the identity (REG). Saves runs/wsp/final_<comp>_<family><LAM>.npz: per person t, B, c, and
held-out diagnostics (accuracy, correlation, autocorrelation before / after); shufcombo OUTX applies it.
Usage: [LAM=1] [REG=0.1] python studies/14nov/wsp_final.py COMP acc|corr      COMP in s<k>, c1, rest, eye"""
import glob, os, sys
import numpy as np
from scipy.optimize import minimize
REPO = os.path.abspath(__file__ + "/../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; p0 = "all_loso"; Z = A.L.Z; IDX = A.pl.B[f"idx|{p0}"].astype(int)
COMP, FAM = sys.argv[1], sys.argv[2]; LAM, REG = float(os.environ.get("LAM", 1)), float(os.environ.get("REG", 0.1))
assert FAM in ("acc", "corr"), FAM
y, per = Z[f"y|{p0}"].astype(int), Z[f"person|{p0}"]; PEOPLE = list(np.unique(per))
with np.load(REPO + "/data/proteus_windows.npz") as w:
    RUN = np.char.add(w["task"].astype(str), np.char.add("|", w["run"].astype(str)))[IDX]  # as shuffle.py; + session below
RUN = np.char.add(Z[f"sess|{p0}"].astype(str), np.char.add("|", RUN))  # a run of one session of one person


def ld(path):
    z = np.load(path); assert (z["index"].astype(int) == IDX).all(), path; return z["proba"]


PS = Z[f"B2_pool|Ps|{p0}"]
if COMP[0] == "s":
    P = PS[int(COMP[1:])]
elif COMP == "c1":
    P = A.C1[p0]
elif COMP == "rest":
    P = np.mean([ld(f"{REPO}/runs/rest4/probs/rest4_sh_r1-{p0}-s{s}.npz") for s in range(3)], 0)
elif COMP == "eye":
    P = np.mean([ld(f) for f in sorted(glob.glob(f"{REPO}/runs/ocunet/probs/oc_nomix-{p0}-s*.npz"))], 0)
else:
    raise SystemExit(f"unknown component {COMP}")
assert np.allclose(P.sum(1), 1, atol=1e-3), f"{COMP}: rows are not posteriors"
Zl = np.log(np.clip(P, 1e-6, 1)); others = [k for k in range(PS.shape[0]) if f"s{k}" != COMP]
RM = np.mean([np.log(np.clip(PS[k], 1e-6, 1))[np.arange(len(y)), y] for k in others], 0)  # rest of the members, true class


def unpack(v):
    B = np.zeros((3, 3)); B[:, :2] = v[1:7].reshape(3, 2); B[:, 2] = -B[:, :2].sum(1); return v[0], B, np.r_[v[7:9], 0.0]


def apply(v, z):
    t, B, c = unpack(v); o = t * z @ (np.eye(3) + B).T + c; o = o - o.max(1, keepdims=True); return o - np.log(np.exp(o).sum(1, keepdims=True))


def resid(lp, r):  # true-class log posterior, minus its person mean (a person's level is not an error streak)
    v = lp[r, y[r]]; out = v.copy()
    for q in np.unique(per[r]):
        m = per[r] == q; out[m] = v[m] - v[m].mean()
    return out


def stats(lp, r):
    e, m = resid(lp, r), RM[r] - RM[r].mean(); corr = np.corrcoef(e, m)[0, 1]; ac = []
    for g in np.unique(RUN[r]):
        s = e[RUN[r] == g]
        if len(s) > 3:
            ac.append(np.corrcoef(s[:-1], s[1:])[0, 1])
    return corr, np.nanmean(ac), np.mean(lp[r].argmax(1) == y[r])


def loss(v, r):
    lp = apply(v, Zl[r]); L = -lp[np.arange(len(r)), y[r]].mean() + REG * ((v[0] - 1) ** 2 + (v[1:] ** 2).sum())
    if FAM == "corr":
        lpa = apply(v, Zl); c, a, _ = stats(lpa, r); L += LAM * (c + a)
    return L


v0 = np.r_[1.0, np.zeros(8)]; OUT = {}
print(f"WSP FINAL {COMP} {FAM} LAM {LAM} REG {REG}: per held-out person, fitted on the other nine (accuracy | corr | autocorr)")
for q in PEOPLE:
    tr, te = np.flatnonzero(per != q), np.flatnonzero(per == q)
    v = minimize(loss, v0, args=(tr,), method="Powell" if FAM == "corr" else "L-BFGS-B", options={"maxiter": 3000}).x
    b, a = stats(Zl, te), stats(apply(v, Zl), te); t, B, c = unpack(v); OUT[q] = (t, B, c)
    print(f"  {q}: acc {b[2]:.4f} -> {a[2]:.4f} | corr {b[0]:+.3f} -> {a[0]:+.3f} | autocorr {b[1]:+.3f} -> {a[1]:+.3f} | t {t:.2f} |B| {np.abs(B).max():.2f}", flush=True)
os.makedirs(f"{REPO}/runs/wsp", exist_ok=True); name = f"{REPO}/runs/wsp/final_{COMP}_{FAM}{LAM:g}.npz"
np.savez(name, people=np.array(PEOPLE), t=np.array([OUT[q][0] for q in PEOPLE]), B=np.array([OUT[q][1] for q in PEOPLE]), c=np.array([OUT[q][2] for q in PEOPLE]))
print("written", name)
