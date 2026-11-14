"""s261 CPU identity: two packages stepped over the same unrecognised session on CPU; step outputs compared bitwise.
Usage: python -I studies/14nov/cpu_identity.py PKG_A PKG_B [--only K]   (session K of the dev stream order, as r15_reference.py)
Prints one JSON line per session (max_diff, bitwise_equal, refits of each) and RESULT; exit 1 on any difference."""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import ft_clock_restart, load_model  # noqa: E402


def run(m, X):
    c = copy.deepcopy(m)
    for r in (c.route, getattr(c, "routeB", None)):
        if r is not None:
            r.contrast.tau = r.identity.tau = np.inf  # unrecognised path, as v6_time.py
    c.reset_state(); ft_clock_restart(c)
    out = np.array([np.asarray(c.step(np.nan_to_num(np.asarray(x, np.float32)))) for x in X])
    return out, dict(refits=c.ft.refits, err=c.log6["err"], capped=getattr(c.ft, "capped", None))


def main():
    pa, pb = Path(sys.argv[1]), Path(sys.argv[2]); d = protocol.load(); torch.set_num_threads(1)
    ss = list(dict.fromkeys(d["stream"]))
    if "--only" in sys.argv:
        ss = [ss[int(sys.argv[sys.argv.index("--only") + 1])]]
    A, B = load_model(pa, None, "idA", "cpu"), load_model(pb, None, "idB", "cpu"); ok, worst = True, 0.0
    for s in ss:
        X = d["X"][np.flatnonzero(d["stream"] == s)]
        oa, la = run(A, X); ob, lb = run(B, X); df = float(np.abs(oa - ob).max()); eq = bool(np.array_equal(oa, ob))
        ok &= eq and la["err"] == 0 and lb["err"] == 0; worst = max(worst, df)
        print(json.dumps(dict(session=str(s), windows=len(X), max_diff=df, bitwise_equal=eq, A=la, B=lb)), flush=True)
    print("RESULT", json.dumps(dict(A=str(pa), B=str(pb), max_diff=worst, ok=bool(ok))), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
