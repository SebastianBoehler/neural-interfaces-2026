"""Sub-8 Hz view as TSEM's template view (ledger s222, step 3), recording order, replay chain (check.py must print OK):
  oracle  TSEM's likelihood (grid.tsem_lik, rho 0.7, shrink 0.7) with one-hot TRUE labels as pseudo-labels: the
          template separability of the view (cell BA per person over the rows TSEM scores, t >= 15)
  tsem    the same with the chain's real TSEM input (B2_pool, all families): template-alone accuracy
  chain   J of A.system(B2_pool) with pc128 replaced by the view ('rep') and with the view appended ('app')
  phi     correlation of the view member's and the eye net's correctness vectors (new people)
ORACLE_ONLY=1 skips 'tsem' and 'chain' (printed as nan / +0.00).
Usage: VIEWS=runs/views/sub8_x.npz[,..] [CFGS=a,b] [BASE=B2_pool] python studies/14nov/views/tview.py"""
import os, sys, time
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path[:0] = [REPO + "/replay"]
import env  # noqa: E402
A = env.A; Z = A.L.Z; G = A.L.G; T0 = time.time(); BASE = os.environ.get("BASE", "B2_pool")
ORACLE = bool(os.environ.get("ORACLE_ONLY"))  # band views: template ceiling and phi only (no TSEM-input or chain runs)
PC0 = {p: Z[f"pc128|{p}"].copy() for p in A.PROTOS}


def cell_ba(L, proto):
    y, cell, per = Z[f"y|{proto}"].astype(int), Z[f"cell|{proto}"], Z[f"person|{proto}"]; ok = ~np.isnan(L[:, 0]); acc = {}
    for c in np.unique(cell[ok]):
        m = ok & (cell == c); pr = L[m].argmax(1)
        acc.setdefault(per[m][0], []).append(100 * np.mean([np.mean(pr[y[m] == k] == k) for k in np.unique(y[m])]))
    return np.mean([np.mean(v) for v in acc.values()])


def J(res):
    s = A.summary(res); return 0.5 * s["dn"].mean() + 0.5 * s["ds"].mean(), s["dn"].mean(), s["ds"].mean()


def chain(views):
    for p in A.PROTOS:
        Z[f"pc128|{p}"] = views[p].astype(PC0[p].dtype)
    try:
        return J(A.system(BASE, A.FAMS))
    finally:
        for p in A.PROTOS:
            Z[f"pc128|{p}"] = PC0[p]


def templ(views, pre):
    out = []
    for p in A.PROTOS:
        y = Z[f"y|{p}"].astype(int); oh = np.eye(3)[y]; Lo, _ = G.tsem_lik(oh, Z[f"sess|{p}"], views[p], rho=0.7)
        Lt = G.tsem_lik(pre[p], Z[f"sess|{p}"], views[p], rho=0.7)[0] if not ORACLE else np.full_like(Lo, np.nan)
        out += [cell_ba(Lo, p), cell_ba(Lt, p) if not ORACLE else np.nan]
    return out  # oracle new, tsem new, oracle seen, tsem seen


ref = A.system(BASE, A.FAMS); PRE = ref["pre"]; J0 = J(ref)
eye = np.mean([np.load(f"{REPO}/runs/ocunet/probs/oc_nomix-all_loso-s{s}.npz")["proba"] for s in range(3)], 0)
ye = Z["y|all_loso"].astype(int); ceye = (eye.argmax(1) == ye).astype(float)
o = templ(PC0, PRE)
print(f"base {BASE} J {J0[0]:.4f} (new {J0[1]:.2f} seen {J0[2]:.2f}); pc128: oracle new {o[0]:.2f} seen {o[2]:.2f}, "
      f"tsem new {o[1]:.2f} seen {o[3]:.2f}; eye net alone {100 * ceye.mean():.1f} % ({time.time() - T0:.0f}s)", flush=True)
print("view            dim | oracle new/seen | tsem new/seen | rep dJ (new, seen) | app dJ (new, seen) | phi eye", flush=True)
want = os.environ.get("CFGS")
for path in os.environ["VIEWS"].split(","):
    f = np.load(REPO + "/" + path if not path.startswith("/") else path)
    for cfg in sorted({k.split("|")[0] for k in f.files}):
        if want and cfg not in want.split(","):
            continue
        V = {p: f[f"{cfg}|view|{p}"][Z[f"idx|{p}"].astype(int)].astype(np.float64) for p in A.PROTOS}
        assert all(np.isfinite(v).all() for v in V.values()), f"{cfg}: view has NaN on replay rows"
        V = {p: (v - v.mean(0)) / np.maximum(v.std(0), 1e-9) for p, v in V.items()}  # scale only (TSEM standardises per session)
        t = templ(V, PRE); r, a = (J0, J0) if ORACLE else (chain(V), chain({p: np.c_[PC0[p], V[p]] for p in A.PROTOS}))
        cv = (f[f"{cfg}|all_loso"].argmax(1) == ye).astype(float); phi = np.corrcoef(cv, ceye)[0, 1]
        print(f"{cfg:12s} {V['all_loso'].shape[1]:4d} | {t[0]:5.2f} / {t[2]:5.2f} | {t[1]:5.2f} / {t[3]:5.2f} | "
              f"{r[0] - J0[0]:+.2f} ({r[1] - J0[1]:+.2f}, {r[2] - J0[2]:+.2f}) | {a[0] - J0[0]:+.2f} ({a[1] - J0[1]:+.2f}, "
              f"{a[2] - J0[2]:+.2f}) | {phi:.2f} ({time.time() - T0:.0f}s)", flush=True)
