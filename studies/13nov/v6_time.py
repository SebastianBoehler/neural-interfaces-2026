"""v6 run time on unrecognised sessions (worst case: every session unrecognised), warm and paired with v5.1b.
Both packages stream the same sessions window by window on a fresh deepcopy with both gates forced off (v5.1b then runs
its unrouted path; v6 runs v5.1b plus the v6 chain and fine-tuning on every window). Sessions: P02|1 (warm-up, not
reported), P01|1 (190 windows) and P04|1+2 streamed as one session (378). Order A B B A per session length.
Usage: python studies/13nov/v6_time.py DEVICE PKG_V51B PKG_V6 [--gates-on] [--long] [--mem-frac 40/68] -> JSON lines"""
import copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
GATES_ON = "--gates-on" in sys.argv  # time the recognised path instead (gates as shipped)
from record import ft_clock_restart, load_model  # noqa: E402


def run(m, X, dev):
    c = copy.deepcopy(m)
    if not GATES_ON:  # the unrecognised path; v6.2's main chain has gates of its own (routeB)
        for r in (c.route, getattr(c, "routeB", None)):
            if r is not None:
                r.contrast.tau = r.identity.tau = np.inf
    c.reset_state()
    ft = getattr(c, "ft", None); ft_clock_restart(c)  # v6.2g: each timed run gets the whole budget (the cap is what --long times)
    acc = [0.0, 0.0]  # refit wall time (synchronised), longest single refit
    if ft is not None and hasattr(ft, "refit"):
        orig = ft.refit

        def timed(t):
            if dev == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter(); orig(t)
            if dev == "cuda":
                torch.cuda.synchronize()
            d = time.perf_counter() - t1; acc[0] += d; acc[1] = max(acc[1], d)
        ft.refit = timed
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for x in X:
        c.step(np.nan_to_num(np.asarray(x, np.float32)))
    if dev == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    gb = dict(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2)) if dev == "cuda" else {}
    return round(sec, 1), (dict(refits=ft.refits, steps=ft.steps, v6=c.log6["v6"], err=c.log6["err"], capped=getattr(ft, "capped", None),
                                frozen=getattr(ft, "frozen", None), refit_s=round(acc[0], 2), refit_max_s=round(acc[1], 2), **gb) if ft else gb)


def main():
    dev, p5, p6 = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); d = protocol.load(); torch.set_num_threads(1)
    if "--mem-frac" in sys.argv:  # e.g. 40/68: a 40 GB worker card on a 68 GB A100 (the allocator raises past the share)
        f = sys.argv[sys.argv.index("--mem-frac") + 1]; a_, b_ = (f.split("/") + ["1"])[:2]
        torch.cuda.set_per_process_memory_fraction(float(a_) / float(b_)); print(json.dumps(dict(mem_frac=float(a_) / float(b_))), flush=True)
    f = lambda s: np.flatnonzero(d["stream"] == f"Dreyer2026Proteus/{s}")  # noqa: E731
    S = {"190": f("P01|1"), "378": np.concatenate([f("P04|1"), f("P04|2")])}
    if "--long" in sys.argv:  # v6.2g: one synthetic 800-window session (P04's three sessions, P10|1, the first 42 of P10|2)
        S = {"800": np.concatenate([f("P04|1"), f("P04|2"), f("P04|3"), f("P10|1"), f("P10|2")])[:800]}
    A, B = load_model(p5, None, "v51b", dev), load_model(p6, None, "v6", dev)
    run(A, d["X"][f("P02|1")], dev); run(B, d["X"][f("P02|1")], dev)  # warm-up
    for tag, rows in S.items():
        X = d["X"][rows]; a1, la = run(A, X, dev); b1, l1 = run(B, X, dev); b2, l2 = run(B, X, dev); a2, _ = run(A, X, dev)
        a, b = (a1 + a2) / 2, (b1 + b2) / 2
        print(json.dumps(dict(session=tag, windows=len(rows), v51b=[a1, a2], v6=[b1, b2], ratio=round(b / a, 3), added_s=round(b - a, 1),
                              log=l1, log_v51b=la, sealed_min_57=round(1287 * (1 + 0.57 * (b / a - 1)) / 60, 1), sealed_min_100=round(1287 * b / a / 60, 1))), flush=True)


if __name__ == "__main__":
    main()
