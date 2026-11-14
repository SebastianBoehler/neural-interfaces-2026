"""v6.2h forced-exception check: one part of the v6 additions raises at window K of each session.
Asserts per session: the stream completes; the [v6] error count is 1; windows before K equal the same package's run
without the raise, bit for bit; windows from K on equal v5.1b's output, bit for bit (both the inner v5.1b of the run and
the v5.1b package streamed on its own).
Usage: python studies/14nov/v62h_raise.py PKG_H PKG_V51B [--site ft|eye] [--at 30] [--gates-off] [--only K | --sessions N]
--site ft raises in FineTune.step (the 14 Nov worker crash: Adam's CUDA check inside a refit); eye raises in EyeNet.step,
which every live window calls."""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import ft_clock_restart, load_model  # noqa: E402
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
SITE, AT, OFF = arg("--site", "ft"), arg("--at", 30), "--gates-off" in sys.argv


def run(m, X, raise_at=None):
    c = copy.deepcopy(m)
    if OFF:
        for r in (c.route, getattr(c, "routeB", None)):
            if r is not None:
                r.contrast.tau = r.identity.tau = np.inf
    c.reset_state(); ft_clock_restart(c); fired = []
    if raise_at is not None:
        part = c.ft if SITE == "ft" else c.eye
        orig = part.step

        def boom(*a, **k):
            if c.log6["n"] - 1 == raise_at:  # log6["n"] counts this window already
                fired.append(raise_at)
                raise RuntimeError(f"forced test exception in {SITE}.step")
            return orig(*a, **k)
        part.step = boom
    o, o5 = [], []
    for x in X:
        o.append(np.asarray(c.step(np.nan_to_num(np.asarray(x, np.float32)))))
        o5.append(np.asarray(getattr(c, "trace", {}).get("out_v5", np.full(3, np.nan))))
    return np.array(o), np.array(o5), dict(getattr(c, "log6", {})), fired


h, p5 = Path(sys.argv[1]), Path(sys.argv[2]); d = protocol.load(); torch.set_num_threads(1)
H, V5 = load_model(h, None, "h"), load_model(p5, None, "v51b"); ok = True
SS = list(dict.fromkeys(d["stream"]))
for s in ([SS[arg("--only", 0)]] if "--only" in sys.argv else SS[:arg("--sessions", 3)]):
    X = d["X"][d["stream"] == s]
    clean, _, logc, _ = run(H, X); hit, inner5, logh, fired = run(H, X, AT)
    c5 = copy.deepcopy(V5)
    if OFF:
        c5.route.contrast.tau = c5.route.identity.tau = np.inf
    c5.reset_state(); o5 = np.array([np.asarray(c5.step(np.nan_to_num(np.asarray(x, np.float32)))) for x in X])
    r = dict(session=s.split("/")[-1], windows=len(X), fired=fired, err=logh.get("err"), first=logh.get("first", "")[:80],
             err_clean=logc.get("err"), before_equal=bool(np.array_equal(hit[:AT], clean[:AT])),
             after_equal_inner_v51b=bool(np.array_equal(hit[AT:], inner5[AT:])), after_equal_v51b_pkg=bool(np.array_equal(hit[AT:], o5[AT:])),
             clean_differs_from_v51b_after=float(np.abs(clean[AT:] - o5[AT:]).max()))
    good = fired == [AT] and r["err"] == 1 and r["err_clean"] == 0 and r["before_equal"] and r["after_equal_inner_v51b"] and r["after_equal_v51b_pkg"]
    ok &= good
    print(json.dumps(dict(r, ok=good)), flush=True)
print("RESULT", json.dumps(dict(pkg=str(h), site=SITE, at=AT, gates_off=OFF, ok=ok)), flush=True)
assert ok, "forced-exception check failed"
