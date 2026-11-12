"""Where does v5.2's GPU time go? Streams public sessions through the v5.2 package one window at a time on the given
device and times the personal-net path (PersonalNets.proba) against the whole predict call; counts which personal
traces use the shared FIR bank. Usage: python scripts/profile_v52.py PKG_V52 [cuda|cpu] [n_sessions]"""
import sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
from add_rebalance import load_solver  # noqa: E402
pkg, dev = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "cuda"; ns = int(sys.argv[3]) if len(sys.argv) > 3 else 3
m = load_solver(pkg, "v52prof", dev)


def find(o, depth=0, seen=None):  # load_solver does not register the module: find the PersonalNets instance by walking
    seen = seen if seen is not None else set()
    if id(o) in seen or depth > 4:
        return None
    seen.add(id(o))
    if type(o).__name__ == "PersonalNets":
        return o
    for v in (vars(o).values() if hasattr(o, "__dict__") else []):
        r = find(v, depth + 1, seen)
        if r is not None:
            return r
    return None


PNI = find(m); assert PNI is not None, "no PersonalNets in the loaded package"; PN = type(PNI)
T = {"proba": 0.0, "calls": 0, "windows": 0, "predict": 0.0}; orig = PN.proba


def timed(self, j, t):
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter(); out = orig(self, j, t)
    if dev == "cuda":
        torch.cuda.synchronize()
    T["proba"] += time.perf_counter() - t0; T["calls"] += 1; return out


PN.proba = timed
kinds = [type(n).__name__ for row in PNI.pristine for n in row]; print("personal trace classes:", {k: kinds.count(k) for k in set(kinds)})
d = protocol.load(); st = d["stream"]; sessions = list(dict.fromkeys(st))[:ns]
for s in sessions:
    rows = np.flatnonzero(st == s); m.reset_state()
    for i in rows:
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); m.predict(d["X"][i:i + 1]) if hasattr(m, "predict") else m(d["X"][i:i + 1]); T["predict"] += time.perf_counter() - t0; T["windows"] += 1
print(f"{dev}: {T['windows']} windows, predict {1000 * T['predict'] / T['windows']:.1f} ms per window, of which personal nets "
      f"{1000 * T['proba'] / max(T['windows'], 1):.1f} ms ({T['calls']} calls, {1000 * T['proba'] / max(T['calls'], 1):.1f} ms per call)")
