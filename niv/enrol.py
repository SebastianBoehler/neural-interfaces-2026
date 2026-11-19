"""Enrol one labelled visit of a NEW person into a package, the way the sealed package holds a public person.

enrol(zip_in, zip_out, X, y, name): appends one row to every per-person array of weights.npz:
  route_gal / route_ps1_W / route_ps1_c / route_names   contrast gallery + personal head on the 256-PC view (replay/rlib2, rlib3)
  bio_gal / bio_m                                       identity-gate gallery (experiments/bio_gate.fold_params, bio_mu fixed)
  own_*                                                 own-space members (studies/12nov/ownspace.Personal, scripts/add_own_stage.person_arrays)
  v6_tpri                                               personal TSEM template prior (lean/tools/fit_tpri.py), if the package has it
X (n, 41, 480) float32 in recording order, y (n,) int labels 0..2. Everything else is copied byte for byte.
Self-check (`python enrol.py check ZIP`): rebuilding a single-session public person reproduces its stored row.
"""
import ast
import importlib.util
import io
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("route_gal", "route_ps1_W", "route_ps1_c", "route_names", "bio_gal", "bio_m", "v6_tpri")


def stdz(X):  # replay/rlib3.stdz
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)


def _functions(path, names, ns):
    nodes = [n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), "helper anchors differ"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)


def views(directory, z, X):
    """(n, 256) core1 view of each window, streamed causally in the given order (the harness gallery convention)."""
    spec = importlib.util.spec_from_file_location("niv_pkg", Path(directory) / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m, V0, V1 = (np.asarray(z[k], np.float64) for k in ("tsem_m", "tsem_V", "route_V_hi"))
    if hasattr(mod, "TangentCore"):  # v6 packages
        core = mod.TangentCore(mod.with_prefix(z, "ts_"))
        step = lambda x: core.step(x)[1]
    else:  # v5 packages: AlignedTangent, view of the scaled features of the last window
        ts = {k[3:]: v for k, v in z.items() if k.startswith("ts_")}
        ts["log_glob"], ts["bands"] = list(ts["log_glob"]), [tuple(b) for b in ts["bands"]]
        core = mod.AlignedTangent(ts)
        step = lambda x: (core.proba(x), core.z)[1]
    core.reset_state()
    out = []
    for x in np.nan_to_num(np.asarray(X, np.float32)):
        d = step(x) - m
        out.append(np.r_[d @ V0, d @ V1])
    return np.asarray(out)


def own_arrays(X, y):
    """The own-space members of one person from one session (module ownspace run on these windows only)."""
    src = (ROOT / "studies/12nov/ownspace.py").read_text()
    head = 'z = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)'
    assert head in src
    src = src.replace(head, "z = INJECT").replace("PER = np.array([s[-3:] for s in z[\"subject\"]]); SES = z[\"session\"].astype(str)",
                                                 "PER = np.array(['P'] * len(z['y'])); SES = np.array(['1'] * len(z['y']))")
    ns = {"__file__": str(ROOT / "studies/12nov/ownspace.py"),
          "INJECT": dict(X=np.asarray(X, np.float32), y=np.asarray(y), interface=np.array(["i"] * len(y)),
                         subject=np.array(["s"] * len(y)), session=np.array(["1"] * len(y)))}
    exec(compile(src, "ownspace_inject", "exec"), ns)
    m = ns["Personal"]([np.arange(len(y))])
    sc, lr = m.T.steps[0][1], m.T.steps[1][1]
    return dict(own_prior=np.stack(m.prior), own_csp=np.stack(m.W), own_t_mu=sc.mean_, own_t_sd=sc.scale_, own_t_W=lr.coef_,
                own_t_c=lr.intercept_, own_c_W=m.C.coef_, own_c_c=m.C.intercept_, own_s_W=m.S.coef_, own_s_c=m.S.intercept_)


def gs_member(directory, z, X, y):
    """Lane G gS: own_s_W/c of this person from the g4 fit (noise-augmented replicas, causal statistics) on the enrolled visit only."""
    import sys
    sys.path.insert(0, str(ROOT / "laneG"))
    import common as K  # laneG
    import fit as F
    spec = importlib.util.spec_from_file_location("niv_gs_pkg", Path(directory) / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    D = dict(X=np.asarray(X, np.float32), y=np.asarray(y).astype(int), person=np.array(["N"] * len(y)), visit=np.ones(len(y), int))
    a = F.fit(mod, z, K.OwnFeat(z), D, "N", (1,), "g4")
    return {k: a[k] for k in ("own_s_W", "own_s_c")}


def person_rows(directory, z, X, y, own_s="stock"):
    ns = {"np": np, "CLIP": 4.0}
    _functions(ROOT / "replay/rlib2.py", ("rstats", "rz", "contrast", "onehot"), ns)
    y = np.asarray(y).astype(int)
    U = views(directory, z, X)
    rows = dict(route_gal=ns["contrast"](U, ns["onehot"](y), 256)[None],
                route_names=None)
    head = LogisticRegression(C=0.1, max_iter=3000).fit(stdz(U[:, :256]), y)
    rows["route_ps1_W"], rows["route_ps1_c"] = head.coef_[None], head.intercept_[None]
    class_means = lambda A: np.stack([A[y == k].mean(0) for k in range(3)])
    rows["v6_tpri"] = class_means(stdz(U[:, :128]))[None]
    win = np.hanning(120)

    def feat(x):  # IdentityGate.feat
        x = np.asarray(x, np.float64)
        x = x - x.mean(0, keepdims=True)
        x = x - x.mean(1, keepdims=True)
        seg = np.stack([x[:, s:s + 120] for s in range(0, 361, 60)], 1) * win
        L = np.log((np.abs(np.fft.rfft(seg, axis=-1)) ** 2).mean(1)[:, 2:41] + 1e-12)
        return (L - L.mean(1, keepdims=True)).ravel()

    sig = np.mean([feat(x) for x in X], 0)
    rows["bio_gal"] = (((sig - z["bio_mu"]) / z["bio_sd"]) @ z["bio_V"])[None]
    rows["bio_m"] = np.array([1])
    own = own_arrays(X, y)
    if own_s == "g4":  # opt-in (NIV_OWN_S=g4); default "stock" is the recipe of every earlier NI-V arm
        own.update(gs_member(directory, z, X, y))
    elif own_s != "stock":
        raise ValueError(f"own_s {own_s!r}: expected stock or g4")
    rows.update({k: v[None] for k, v in own.items()})
    return rows


def enrol(zip_in, zip_out, X, y, name, own_s="stock"):
    with TemporaryDirectory() as tmp, zipfile.ZipFile(zip_in) as zi:
        zi.extractall(tmp)
        z = dict(np.load(Path(tmp) / "weights.npz", allow_pickle=False))
        rows = person_rows(tmp, z, X, y, own_s)
    n0 = len(z["route_names"])
    for k, v in rows.items():
        if k == "route_names":
            z[k] = np.concatenate([z[k], np.array([name], dtype=z[k].dtype if z[k].dtype.kind == "U" else str)])
        elif k in z:
            assert len(z[k]) == n0, (k, z[k].shape)
            z[k] = np.concatenate([z[k], v.astype(z[k].dtype)])
        else:
            assert k == "v6_tpri", f"package lacks {k}"
    assert all(len(z[k]) == n0 + 1 for k in ("route_gal", "route_ps1_W", "bio_gal", "own_prior"))
    buf = io.BytesIO()
    np.savez(buf, **z)
    with zipfile.ZipFile(zip_in) as zi, zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zo:
        for info in zi.infolist():
            zo.writestr(info, buf.getvalue() if info.filename == "weights.npz" else zi.read(info.filename))
    return n0 + 1
