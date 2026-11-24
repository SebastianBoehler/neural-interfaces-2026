"""Lane G tasks. python tasks.py ZIP OUT CMD ...   (ZIP = sealed cap250 package, OUT = run directory)

  feats PERSON VISIT ORDER      test visit streamed in the pinned delivery order: view u, own-space features, labels (to OUT/feats)
  train PERSON VISITS MODE      fit the arm on the training visits only -> OUT/arms/<person>_<visits>_<mode>.npz
  prod  PERSON MODE             fit on ALL of the person's public sessions -> OUT/prod/<person>_<mode>.npz
  parity                        stock recipe on P01 vs the package's stored row; OwnStage vs this file's own-space scoring
Every command prints what it wrote; nothing here reads labels of a test visit before scoring (feats stores them for score.py).
"""
import sys
import time
from pathlib import Path

import numpy as np

import common as K
import fit as F


def posteriors(a, u, C, sl, kref):
    """Per-window personal posteriors of one fitted row ``a`` over a delivered visit: h, T, C, S and the product log score."""
    sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    h = sm(F.causal_stdz(u) @ a["route_ps1_W"].T + a["route_ps1_c"])
    Cw = F.recentre(C, a["own_prior"], kref, True)
    pt = sm(((F.tang(Cw) - a["own_t_mu"]) / a["own_t_sd"]) @ a["own_t_W"].T + a["own_t_c"])
    pc = sm(F.csp_feat(a["own_csp"], Cw) @ a["own_c_W"].T + a["own_c_c"])
    ps = sm(F.causal_z(sl) @ a["own_s_W"].T + a["own_s_c"])
    own = sum(np.log(np.clip(p, 1e-3, 1)) for p in (pt, pc, ps)) / 3
    return dict(h=h, T=pt, C=pc, S=ps, own=own, prod=np.log(np.clip(h, 1e-9, 1)) + own)


def cmd_feats(zip_path, out, person, visit, order):
    D = K.public()
    mod, z = K.package(zip_path, Path(out) / "pkg")
    rows = K.delivery(D, order, person, int(visit))
    feat = K.OwnFeat(z)
    X = D["X"][rows].astype(np.float32)
    t0 = time.time()
    u, (C, sl) = K.views(mod, z, X), feat(X)
    d = Path(out) / "feats"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{person}_{visit}_{order}.npz"
    np.savez_compressed(f, rows=rows, u=u, C=C.astype(np.float32), sl=sl, y=D["y"][rows], itf=D["itf"][rows])
    print(f"wrote {f} n={len(rows)} {time.time() - t0:.0f}s", flush=True)


def cmd_train(zip_path, out, person, visits, mode, prod=False):
    D = K.public()
    mod, z = K.package(zip_path, Path(out) / "pkg")
    vs = tuple(int(c) for c in visits) if not prod else tuple(sorted(set(D["visit"][D["person"] == person].tolist())))
    t0 = time.time()
    a = F.fit(mod, z, K.OwnFeat(z), D, person, vs, mode)
    d = Path(out) / ("prod" if prod else "arms")
    d.mkdir(parents=True, exist_ok=True)
    f = d / (f"{person}_{mode}.npz" if prod else f"{person}_{visits}_{mode}.npz")
    np.savez(f, visits=np.array(vs), **a)
    print(f"wrote {f} visits={vs} {time.time() - t0:.0f}s", flush=True)


def cmd_parity(zip_path, out):
    """(1) this file's own-space scoring == the shipped OwnStage on a P01 stream with the package's own row;
    (2) the stock recipe refitted on P01 (single visit) vs the package's stored row."""
    D = K.public()
    mod, z = K.package(zip_path, Path(out) / "pkg")
    feat, kref = K.OwnFeat(z), float(z["own_kref"])
    rows = np.flatnonzero((D["person"] == "P01") & (D["visit"] == 1))[:60]
    X = D["X"][rows].astype(np.float32)
    stored = {k: z[k][0] for k in ("route_ps1_W", "route_ps1_c", "own_prior", "own_csp", "own_t_mu", "own_t_sd", "own_t_W",
                                   "own_t_c", "own_c_W", "own_c_c", "own_s_W", "own_s_c")}
    C, sl = feat(X)
    mine = posteriors(stored, K.views(mod, z, X), C, sl, kref)["own"]
    st, ref = mod.OwnStage(z), []
    for x in X:
        st.observe(x)
        ref.append(st.log_members(0))
    err = float(np.abs(np.asarray(ref) - mine).max())
    print(f"own-space scoring vs OwnStage max abs diff {err:.3e}")
    assert err < 1e-6, "own-space scoring differs from the shipped stage"
    a = F.fit(mod, z, feat, D, "P01", (1,), "stock")
    for k, v in stored.items():
        r = float(np.abs(a[k] - v).max() / max(np.abs(v).max(), 1e-12))
        print(f"stock recipe vs package row {k}: max rel diff {r:.3e}")
        if k.startswith("own_") and k not in ("own_csp",):
            assert r < 1e-3, f"stock recipe does not reproduce {k}"
    print("PARITY OK")


if __name__ == "__main__":
    zip_path, out, cmd, *args = sys.argv[1:]
    {"feats": cmd_feats, "train": cmd_train, "parity": cmd_parity,
     "prod": lambda zp, o, person, mode: cmd_train(zp, o, person, "", mode, prod=True)}[cmd](zip_path, out, *args)
