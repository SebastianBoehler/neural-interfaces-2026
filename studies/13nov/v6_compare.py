"""v6.2 check (a): two packages streamed over every public session, gates as shipped (and optionally forced off), one
window per step on a fresh copy; asserts the outputs equal bit for bit (e.g. v6.2 with v6_routed = 0 against v6.1).
Usage: python studies/13nov/v6_compare.py PKG_A PKG_B [--gates-off] [--device cuda] [--fresh-clock]
       [--from N] [--budget S] [--sessions K | --only K]
v6.2g (module-level fine-tuning clock): --fresh-clock restarts each package's clock before each session, so a
  process longer than the budget still tests the guards' inertness session by session (identity runs, CPU);
  --from N asserts equality only before window N of each session and reports where they first differ (cap trigger:
  the first skipped refit); --budget S sets B's budget on its session copy (a mid-session budget trigger; no equality
  asserted, only errors 0 and frozen windows > 0). Each session reports both chains' errors, capped and frozen counts."""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import ft_clock_restart, load_model  # noqa: E402
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
DEV, OFF, FRESH = arg("--device", "cpu"), "--gates-off" in sys.argv, "--fresh-clock" in sys.argv
FROM, BUDGET = arg("--from", -1), arg("--budget", -1.0)


def run(m, X, budget=-1.0):
    c = copy.deepcopy(m)
    if OFF:
        for r in (c.route, getattr(c, "routeB", None)):
            if r is not None:
                r.contrast.tau = r.identity.tau = np.inf
    c.reset_state()
    ft = getattr(c, "ft", None); clock = ft_clock_restart(c) if FRESH else None
    if budget >= 0:
        assert hasattr(ft, "budget"), "--budget needs a v6.2g package as B"
        ft.budget = budget
    o = np.array([c.step(np.nan_to_num(np.asarray(x, np.float32))) for x in X])
    log = getattr(c, "log6", {}); info = dict(err=log.get("err"), capped=getattr(ft, "capped", None), frozen=getattr(ft, "frozen", None), over=getattr(ft, "over", None),
                                              refits=getattr(ft, "refits", None), clock_stopped=None if clock is None else clock["stopped"])
    assert not log.get("err"), log
    return o, info


a, b = Path(sys.argv[1]), Path(sys.argv[2]); d = protocol.load(); torch.set_num_threads(1)
A, B = load_model(a, None, "a", DEV), load_model(b, None, "b", DEV); worst, n, ok = 0.0, 0, True
SS = list(dict.fromkeys(d["stream"]))[:arg("--sessions", 99)]
for s in ([SS[arg("--only", 0)]] if "--only" in sys.argv else SS):  # --only K: the K-th session alone (one job per session)
    X = d["X"][d["stream"] == s]; (oa, ia), (ob, ib) = run(A, X), run(B, X, BUDGET); n += len(X)
    neq = np.flatnonzero(np.any(oa != ob, axis=1)); first = int(neq[0]) if len(neq) else None
    if BUDGET >= 0:
        ok &= bool(ib["frozen"])
    elif FROM >= 0:
        ok &= first is None or first >= FROM
    else:
        diff = float(np.abs(oa - ob).max()); worst = max(worst, diff); ok &= diff == 0.0
    print(json.dumps(dict(session=s.split("/")[-1], windows=len(X), max_diff=float(np.abs(oa - ob).max()), first_diff=first, a=ia, b=ib)), flush=True)
print("RESULT", json.dumps(dict(a=str(a), b=str(b), gates_off=OFF, device=DEV, fresh_clock=FRESH, frm=FROM, budget=BUDGET, windows=n,
                                max_diff=worst, ok=ok)), flush=True)
assert ok, "check failed"
