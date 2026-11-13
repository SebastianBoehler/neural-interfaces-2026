"""Review 13 Nov, item 8: does predict's cost grow with the window index? (SessionEM._adapt, TSEM._adapt and Route3.scores
rebuild arrays over all earlier windows on every call.) Per package: a warm-up session, then a 190-window session (P10|1)
and a long session made by concatenating P10's three sessions (577 windows, streamed as one session), one window per
predict on a fresh deepcopy, CUDA-synchronised; prints ms per window in bins of 50 windows and the slope.
Usage: python studies/13nov/time_by_window.py DEVICE PKG [PKG ...]"""
import copy, json, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts"), str(WS / "2026-competition")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import protocol  # noqa: E402
from add_rebalance import load_solver  # noqa: E402


def stream(m, X, dev):
    c = copy.deepcopy(m); c.reset_state(); ts = []
    for x in X:
        xt = torch.from_numpy(np.ascontiguousarray(x[None], np.float32)).to(dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); c.predict(xt)
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return np.array(ts) * 1e3


def main():
    dev = sys.argv[1]; d = protocol.load(); st = d["stream"]
    one = np.flatnonzero(st == "Dreyer2026Proteus/P10|1"); long_ = np.concatenate([np.flatnonzero(st == f"Dreyer2026Proteus/P10|{k}") for k in (1, 2, 3)])
    warm = np.flatnonzero(st == "Dreyer2026Proteus/P01|1")
    for pkg in map(Path, sys.argv[2:]):
        m = load_solver(pkg, pkg.name, dev); stream(m, d["X"][warm], dev)
        for tag, rows in (("190", one), ("577", long_)):
            ms = stream(m, d["X"][rows], dev); bins = [round(float(ms[a:a + 50].mean()), 1) for a in range(0, len(ms), 50)]
            slope = float(np.polyfit(np.arange(len(ms))[30:], ms[30:], 1)[0])
            print(json.dumps(dict(pkg=pkg.name, session=tag, windows=len(ms), total_s=round(float(ms.sum()) / 1e3, 1),
                                  ms_per_50=bins, slope_ms_per_window=round(slope, 4))), flush=True)


if __name__ == "__main__":
    main()
