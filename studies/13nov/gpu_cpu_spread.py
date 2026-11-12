"""Tolerance for v5.2 GPU checks (ledger s112): the GPU-vs-CPU spread of the SHIPPED base SessionBN traces (v5.1b)
and of the personal traces, on real aligned session streams (the package's own input streams), run once as one batch
per session and once window by window. Usage: python studies/13nov/gpu_cpu_spread.py PKG_V52 -> JSON"""
import copy, json, os, sys
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import protocol  # noqa: E402
from add_rebalance import load_solver  # noqa: E402


def run(f, X, dev, batch):
    m = torch.jit.load(str(f), map_location=dev).eval()
    for n, b in m.named_buffers():
        if n.rsplit(".", 1)[-1] in ("s1", "s2", "n"):
            b.zero_()
    Xt = torch.from_numpy(X).to(dev)
    with torch.no_grad():
        z = m(Xt) if batch else torch.cat([m(Xt[i:i + 1]) for i in range(len(Xt))])
    return torch.softmax(z.double(), 1).cpu().numpy()


def main():
    pkg = Path(sys.argv[1]); d = protocol.load(); m = load_solver(pkg, "v52"); z = np.load(pkg / "weights.npz")
    res = {"base": [0.0, 0.0], "personal": [0.0, 0.0]}
    for s in list(dict.fromkeys(d["stream"]))[10:12]:
        rows = np.flatnonzero(d["stream"] == s); streams = {}
        for k in set(m.pnets.keys):
            st = copy.deepcopy(m.groups[k][0]); st.reset_state(); streams[k] = np.stack([st(np.asarray(d["X"][i], np.float32)) for i in rows])
        for mi, (i, k) in enumerate(zip(m.pnets.idx, m.pnets.keys)):
            for kind, f in (("base", pkg / str(z[f"net{i}_sjit_file"])), ("personal", pkg / str(z["pnet_files"][mi, 3]))):
                cpu = run(f, streams[k], "cpu", True)
                res[kind][0] = max(res[kind][0], float(np.abs(run(f, streams[k], "cuda", True) - cpu).max()))
                res[kind][1] = max(res[kind][1], float(np.abs(run(f, streams[k], "cuda", False) - cpu).max()))
    print(json.dumps(dict(gpu_batch_vs_cpu={k: v[0] for k, v in res.items()}, gpu_window_by_window_vs_cpu_batch={k: v[1] for k, v in res.items()})), flush=True)


if __name__ == "__main__":
    main()
