"""v6.2 check (a): two packages streamed over every public session, gates as shipped (and optionally forced off), one
window per step on a fresh copy; asserts the outputs equal bit for bit (e.g. v6.2 with v6_routed = 0 against v6.1).
Usage: python studies/13nov/v6_compare.py PKG_A PKG_B [--gates-off] [--device cuda]"""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import load_model  # noqa: E402
DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"; OFF = "--gates-off" in sys.argv


def run(m, X):
    c = copy.deepcopy(m)
    if OFF:
        for r in (c.route, getattr(c, "routeB", None)):
            if r is not None:
                r.contrast.tau = r.identity.tau = np.inf
    c.reset_state()
    return np.array([c.step(np.nan_to_num(np.asarray(x, np.float32))) for x in X])


a, b = Path(sys.argv[1]), Path(sys.argv[2]); d = protocol.load(); torch.set_num_threads(1)
A, B = load_model(a, None, "a", DEV), load_model(b, None, "b", DEV); worst, n = 0.0, 0
for s in dict.fromkeys(d["stream"]):
    X = d["X"][d["stream"] == s]; oa, ob = run(A, X), run(B, X); diff = float(np.abs(oa - ob).max()); worst = max(worst, diff); n += len(X)
    print(json.dumps(dict(session=s.split("/")[-1], windows=len(X), max_diff=diff, bitwise=bool(np.array_equal(oa, ob)))), flush=True)
print("RESULT", json.dumps(dict(a=str(a), b=str(b), gates_off=OFF, device=DEV, windows=n, max_diff=worst, bitwise=worst == 0.0)), flush=True)
assert worst == 0.0, "outputs differ"
