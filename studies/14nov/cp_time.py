"""s268 timing of predict() (count filter and personalisation included), unrecognised path, packages A and B streamed over the
same windows, warm-up first. Usage: python -I studies/14nov/cp_time.py PKG_A PKG_B DEVICE [--n 190 378]
Windows are the dev stream order concatenated (as refit_profile.py), the first n of them, one window per predict() call.
Prints per n: seconds A, seconds B, ms/window each, ratio B/A, cpers refits / fused / err of B. Labels of A and B are compared (late-only)."""
import copy, json, sys, time
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
            r.contrast.tau = r.identity.tau = np.inf
    c.reset_state(); ft_clock_restart(c); lab = []; t0 = time.perf_counter()
    for x in X:
        lab.append(int(np.asarray(c.predict(torch.from_numpy(np.nan_to_num(np.asarray(x, np.float32))[None])))[0]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cp = getattr(c, "cp", None)
    return time.perf_counter() - t0, np.array(lab), None if cp is None else dict(refits=cp.refits, fused=cp.fused, err=cp.log["err"]), c.log6["err"]


def main():
    a = sys.argv; pa, pb, dev = Path(a[1]), Path(a[2]), a[3]; ns = [int(v) for v in a[a.index("--n") + 1:]] if "--n" in a else [190, 378]
    d = protocol.load(); torch.set_num_threads(4); X = np.concatenate([d["X"][np.flatnonzero(d["stream"] == s)] for s in dict.fromkeys(d["stream"])])
    A, B = load_model(pa, None, "tA", dev), load_model(pb, None, "tB", dev); run(A, X[:25]); run(B, X[:25])
    for n in ns:
        ta, la, _, ea = run(A, X[:n]); tb, lb, cb, eb = run(B, X[:n]); diff = np.flatnonzero(la != lb)
        print(json.dumps(dict(n=n, device=dev, sec_A=round(ta, 2), sec_B=round(tb, 2), ms_A=round(1000 * ta / n, 2), ms_B=round(1000 * tb / n, 2),
                              ratio=round(tb / ta, 3), cp_ms_per_window=round(1000 * (tb - ta) / n, 2), cp=cb, err=[ea, eb],
                              first_label_diff=int(diff[0]) if len(diff) else None, n_diff=int(len(diff)))), flush=True)
    print("EXIT", flush=True)


if __name__ == "__main__":
    main()
