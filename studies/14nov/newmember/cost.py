"""NEWMEMBER cost (ledger s272): per-window forward time of the eegn and conf nets under causal SessionBN, as a ratio against the
shipped sinc / fbraw / fb / fbl nets. Same harness for all (experiments/session_bn.make, eager SessionNet in eval mode = the causal
formula, clean forward without grad, one session of t windows per call, as ft_bench2's `eager`); the shipped TorchScript StreamBN
traces are not used, so the ratio, not the absolute ms, is the reading (A100 absolutes need the cluster: `python cost.py cuda`).
Configs: eegn = eegnet_ea_ptm7r_rot05_sbn, conf = conformer_ea_ptm7r_rot05_sbn (studies/12nov/ttt5.py ARCHS), seed 0.
Usage: scripts/run_capped.sh LOG python studies/14nov/newmember/cost.py [cpu|cuda]   -> JSON lines + a ratio table"""
import json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path[:0] = [str(REPO / "experiments"), str(REPO / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
from deep import build  # noqa: E402

CFG = {"fb": "fb_sbn_rot05", "fbraw": "fbcnet_raw_sbn", "fbl": "fblight_ea_rot05_sbn", "sinc": "sincshallow_ea_ptm7r_rot05_sbn",
       "eegn": "eegnet_ea_ptm7r_rot05_sbn", "conf": "conformer_ea_ptm7r_rot05_sbn"}
K = {"fb": 2.0, "fbraw": 20.0, "fbl": 5.0, "sinc": 20.0, "eegn": 20.0, "conf": 20.0}


def net(cfg, k, d, dev):
    m = SB.make(cfg, d, 0); m.net = build(m.arch, d["X"].shape[1], d["X"].shape[2], m.pretrained, **m.net_kw); m._prepare(); m.set_k(k)
    n = m.net.to(dev).eval(); nsb = len(m.sbns()); assert nsb > 0, f"{cfg}: no SessionBN layer (a no-SBN net would not be causal-session adaptive)"
    return n, nsb, sum(p.numel() for p in n.parameters())


def bench(n, X, dev, reps):
    with torch.no_grad():
        for _ in range(2):
            n(X)
        if dev == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); n(X)
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; torch.set_num_threads(int(os.environ.get("NTHREADS", 1))); d = protocol.load(); res = {}
    print(json.dumps(dict(torch=torch.__version__, device=dev, threads=torch.get_num_threads())), flush=True)
    for t in (30, 110, 190):
        X = torch.randn(t, d["X"].shape[1], d["X"].shape[2], device=dev)
        for fam, cfg in CFG.items():
            n, nsb, npar = net(cfg, K[fam], d, dev); s = bench(n, X, dev, 5 if dev == "cpu" else 20); res[fam, t] = s
            print(json.dumps(dict(fam=fam, t=t, ms_per_window=round(1000 * s / t, 3), n_sessionbn=nsb, params=npar)), flush=True)
    print("ratio of ms per window (median over t in 30/110/190), per net vs the shipped nets:", flush=True)
    ms = {f: np.mean([1000 * res[f, t] / t for t in (30, 110, 190)]) for f in CFG}
    ship = {f: ms[f] for f in ("fb", "fbraw", "fbl", "sinc")}; shipped_sum = sum(ship.values())
    for f in ("eegn", "conf"):
        print(json.dumps(dict(fam=f, ms=round(ms[f], 3), vs_fbraw=round(ms[f] / ms["fbraw"], 3), vs_sinc=round(ms[f] / ms["sinc"], 3), vs_mean_shipped=round(ms[f] / np.mean(list(ship.values())), 3))), flush=True)
    print(json.dumps(dict(arm_A_forward_ratio=round((shipped_sum - ms["fbraw"] + ms["eegn"]) / shipped_sum, 3), arm_B_forward_ratio=round((shipped_sum + ms["eegn"] + ms["conf"]) / shipped_sum, 3),
                          note="per-net-family sums, same seed count per slot; forward only (fine-tuning cost of the shipped slots is not in this ratio and the new slots get none)")), flush=True)


if __name__ == "__main__":
    main()
