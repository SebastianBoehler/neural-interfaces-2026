"""Robustness stress test of a package on one real public session (31 Oct audit).

Usage: python scripts/diag/stress_package.py PKG_DIR [REF_PKG_DIR]
Streams P09 session 1 through PKG (and REF, for a label-identity check), then
degenerate inputs: zero / NaN session starts, dead EM-view channels, a 3e38
spike, identical channels, a permuted or reduced montage with matching
meta["ch_names"]. Every case must run without an exception (a sealed crash
costs the slot, with no log); label changes are reported, not gated."""
import sys, copy, importlib.util, traceback
from pathlib import Path
import numpy as np, torch
ROOT = Path("/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026")
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
d = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
CH = [str(c) for c in d["ch_names"]]
def load(pkg, names=CH):
    spec = importlib.util.spec_from_file_location(pkg.name, pkg / "submission.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    meta = {"submission_dir": pkg, "device": "cpu"}
    if names is not None:
        meta["ch_names"] = list(names)
    return m.Solver.__new__(m.Solver).load_model(meta)
NEW = Path(sys.argv[1]).resolve()
OLD = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else NEW
key = np.char.add(np.char.add(d["subject"].astype(str), "|"), d["session"].astype(str))
k = list(dict.fromkeys(key))[-1]; r = np.flatnonzero(key == k); S = r[np.lexsort((d["onset"][r], d["run"][r]))]
X0 = d["X"][S]
def run(model, X):
    m = copy.deepcopy(model); m.reset_state()
    return np.array([int(m.predict(torch.as_tensor(x[None]))[0]) for x in X])
new = load(NEW); L0 = run(new, X0); Lold = run(load(OLD), X0)
print("perm None on training order:", getattr(new, "perm", None) is None, "| new vs staged labels differ:", int((L0 != Lold).sum()), "/", len(L0), flush=True)
def case(name, X, model=new):
    try:
        L = run(model, X); n = min(len(L), len(L0))
        print(f"{name:40s} ok changed {(L[:n] != L0[:n]).sum()}/{n} counts {np.bincount(L, minlength=3)}", flush=True)
    except Exception as e:
        print(f"{name:40s} CRASH {type(e).__name__}: {e}", flush=True); traceback.print_exc()
case("all-zero session", np.zeros_like(X0))
X = X0.copy(); X[:10] = np.nan; case("first 10 windows NaN", X)
X = X0.copy(); X[:30] = np.nan; case("first 30 windows NaN", X)
fr = [CH.index(c) for c in ("F5","AF3","F3","F6","AF4","F4","P1","P2","Pz","C1","C2","Cz")]
X = X0.copy(); X[:, fr] = 0; case("all 12 EM-view channels dead", X)
X = X0.copy(); X[5, 3, 10] = 3e38; case("3e38 spike at t=5", X)
X = X0.copy(); X[:, :] = X0[:, :1]; case("identical channels", X)
rng = np.random.default_rng(0); p = rng.permutation(41)
case("permuted montage + permuted ch_names", X0[:, p], load(NEW, [CH[i] for i in p]))
case("permuted montage, names lower-case", X0[:, p], load(NEW, [CH[i].lower() for i in p]))
keep = [i for i in range(41) if CH[i] != "C5"]
case("C5 missing (40 ch)", X0[:, keep], load(NEW, [CH[i] for i in keep]))
case("ch_names None", X0, load(NEW, None))
spec = importlib.util.spec_from_file_location("x", NEW / "submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
if hasattr(mod, "channel_map"):
    print("channel_map(None) =", mod.channel_map(None), "| junk names ->", mod.channel_map(["a"] * 41))
