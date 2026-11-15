"""s268 predict-level identity: packages A (v6.5b) and B (v6.6, cp off or on) streamed window by window through predict() as the
harness does (one window per call, fresh copy, reset_state); labels compared. A session is run as unrecognised (gates' tau
infinite, as v6_time.py / cpu_identity.py) with --unrec, else as recorded (public person, recognised).
Usage: python -I studies/14nov/cp_identity.py PKG_A PKG_B --expect equal|late  [--unrec] [--only K ...]
  equal: every label must agree (cp off, or recognised sessions); late: labels may differ only at windows >= 89 (cp on, unrecognised).
Prints per session: windows, labels differing, first differing window, cpers/err; exit 1 on a violated expectation."""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import ft_clock_restart, load_model  # noqa: E402


def run(m, X, unrec):
    c = copy.deepcopy(m)
    if unrec:
        for r in (c.route, getattr(c, "routeB", None)):
            if r is not None:
                r.contrast.tau = r.identity.tau = np.inf
    c.reset_state(); ft_clock_restart(c)
    lab = np.array([int(np.asarray(c.predict(torch.from_numpy(np.nan_to_num(np.asarray(x, np.float32))[None])))[0]) for x in X])
    cp = getattr(c, "cp", None)
    return lab, dict(err=c.log6["err"], cp_refits=None if cp is None else cp.refits, cp_fused=None if cp is None else cp.fused,
                     cp_err=None if cp is None else cp.log["err"], cp_stopped=None if cp is None else cp.stopped)


def main():
    a = sys.argv; pa, pb = Path(a[1]), Path(a[2]); expect = a[a.index("--expect") + 1]; unrec = "--unrec" in a
    only = [int(v) for v in a[a.index("--only") + 1:] if v.isdigit()] if "--only" in a else None
    d = protocol.load(); torch.set_num_threads(1); ss = list(dict.fromkeys(d["stream"])); ss = ss if only is None else [ss[k] for k in only]
    A, B = load_model(pa, None, "idA", "cpu"), load_model(pb, None, "idB", "cpu"); ok = True
    for s in ss:
        X = d["X"][np.flatnonzero(d["stream"] == s)]; la, ia = run(A, X, unrec); lb, ib = run(B, X, unrec)
        diff = np.flatnonzero(la != lb); first = int(diff[0]) if len(diff) else None
        good = ia["err"] == 0 and ib["err"] == 0 and (len(diff) == 0 if expect == "equal" else (first is None or first >= 89))
        if expect == "equal" and ib["cp_err"]:
            good = False
        ok &= good
        print(json.dumps(dict(session=str(s), windows=len(X), differ=int(len(diff)), first_diff=first, A=ia, B=ib, ok=bool(good))), flush=True)
    print("RESULT", json.dumps(dict(A=str(pa), B=str(pb), expect=expect, unrec=unrec, ok=bool(ok))), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
