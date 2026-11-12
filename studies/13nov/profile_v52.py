"""Where does v5.2's GPU replay time go? (ledger s112; user task 13 Nov: profile the v5.2 runtime and bring it down)
Runs the official streamed evaluation (benchmark_utils.streaming.predict_streams over benchmark_utils.data.make_loader,
batches of 32 moved to the device, one window per predict, a deepcopy + reset_state per session) on the replay
sessions (the public test split: P04 sessions 1-3, P09 session 1; 755 windows) for each package given. Pass 1: plain
wall time. Pass 2: CUDA-synchronised timers around the components (deepcopy, input streams, base nets, tangent wait,
stages, Route3.step, PersonalNets.proba split into first call and later calls), so the sums attribute the time.
Pass 3: cProfile of one session, top 25 by cumulative time.
Usage: python studies/13nov/profile_v52.py DEVICE PKG [PKG ...]"""
import cProfile, copy, io, json, os, pstats, sys, time
from collections import defaultdict
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts"), str(WS / "2026-competition")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import protocol  # noqa: E402
from add_rebalance import load_solver  # noqa: E402
from benchmark_utils.data import make_loader  # noqa: E402
from benchmark_utils.streaming import predict_streams  # noqa: E402

T = defaultdict(float); N = defaultdict(int)


def timed(obj, name, key, dev):
    f = getattr(obj, name)

    def g(*a, **k):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); o = f(*a, **k)
        if dev == "cuda":
            torch.cuda.synchronize()
        T[key] += time.perf_counter() - t0; N[key] += 1; return o
    setattr(obj, name, g)


def loader(d, rows, dev):
    st = np.unique(d["stream"][rows], return_inverse=True)[1]
    return make_loader(torch.from_numpy(np.ascontiguousarray(d["X"][rows], np.float32)), torch.from_numpy(d["y"][rows]), batch_size=32,
                       record_id=st * 100 + d["run"][rows].astype(int), onset=np.arange(len(rows), dtype=np.float64), stream_id=st, device=dev)


def main():
    dev = sys.argv[1]; d = protocol.load(); rows = np.flatnonzero(d["split"] == "test")
    for pkg in map(Path, sys.argv[2:]):
        m = load_solver(pkg, pkg.name, dev)
        t0 = time.perf_counter(); y, p, _ = predict_streams(m, loader(d, rows, dev)); wall = time.perf_counter() - t0
        res = dict(pkg=pkg.name, device=dev, windows=len(rows), wall=round(wall, 1), acc=float(np.mean(np.asarray(p) == np.asarray(y))))
        # pass 2: component timers on the class methods (the per-session copies inherit them)
        T.clear(); N.clear(); E = type(m)
        for name, key in (("mixed_proba", "mixed_proba"), ("_neural_batch", "nets"), ("predict", "predict"), ("_pmix", "pmix")):
            if hasattr(E, name):
                timed(E, name, key, dev)
        timed(type(m.route), "step", "route_step", dev)
        if getattr(m, "pnets", None) is not None:
            P = type(m.pnets); f = P.proba; first = {}

            def proba(self, j, t):
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter(); o = f(self, j, t)
                if dev == "cuda":
                    torch.cuda.synchronize()
                k = "pn_first" if (id(self), j) not in first else "pn_later"; first[(id(self), j)] = 1
                T[k] += time.perf_counter() - t0; N[k] += 1; return o
            P.proba = proba
        t0 = time.perf_counter(); _ = [copy.deepcopy(m) for _ in range(4)]; T["deepcopy4"] = time.perf_counter() - t0
        t0 = time.perf_counter(); predict_streams(m, loader(d, rows, dev)); res["wall_timed"] = round(time.perf_counter() - t0, 1)
        res["components"] = {k: dict(sec=round(v, 2), calls=N[k], ms_per_call=round(1e3 * v / max(N[k], 1), 2)) for k, v in T.items()}
        print(json.dumps(res), flush=True)
        # pass 3: cProfile of the first session
        one = rows[d["stream"][rows] == d["stream"][rows][0]]; pr = cProfile.Profile(); pr.enable(); predict_streams(m, loader(d, one, dev)); pr.disable()
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25); print(s.getvalue()[:6000], flush=True)
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20); print(s.getvalue()[:5000], flush=True)


if __name__ == "__main__":
    main()
