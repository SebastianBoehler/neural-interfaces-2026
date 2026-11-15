"""s280 REVE deployability (2): forward ms per window of REVE (base) (pretrained architecture, eager fp32, one session of t windows per call,
clean forward without grad) against the shipped families fb / fbraw / fbl / sinc built exactly as studies/14nov/newmember/cost.py.
The ratio and the added fraction of per-window forward cost are the reading; absolutes are for the device given (use cuda on the A100 node).
Usage: python studies/15nov/fm280/cost.py [cpu|cuda]"""
import json, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3])); sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
from deep import build  # noqa: E402

CFG = {"fb": ("fb_sbn_rot05", 2.0), "fbraw": ("fbcnet_raw_sbn", 20.0), "fbl": ("fblight_ea_rot05_sbn", 5.0), "sinc": ("sincshallow_ea_ptm7r_rot05_sbn", 20.0)}


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
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; d = protocol.load(); C, T = d["X"].shape[1:]; ms = {}
    print(json.dumps(dict(torch=torch.__version__, device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None)), flush=True)
    for t in (30, 110, 190):
        X = torch.randn(t, C, T, device=dev)
        for fam, (cfg, k) in CFG.items():
            m = SB.make(cfg, d, 0); m.net = build(m.arch, C, T, m.pretrained, **m.net_kw); m._prepare(); m.set_k(k); n = m.net.to(dev).eval()
            ms.setdefault(fam, []).append(1000 * bench(n, X, dev, 5 if dev == "cpu" else 20) / t)
        cb = build("reve", C, T, True).to(dev).eval(); npar = sum(p.numel() for p in cb.parameters())
        ms.setdefault("reve", []).append(1000 * bench(cb, X, dev, 3 if dev == "cpu" else 20) / t)
        print(json.dumps(dict(t=t, **{f: round(v[-1], 3) for f, v in ms.items()}, reve_params=npar)), flush=True)
    mm = {f: float(np.mean(v)) for f, v in ms.items()}; ship = sum(mm[f] for f in CFG)
    print(json.dumps(dict(mean_ms_per_window=mm, reve_vs_shipped_four_sum=round(mm["reve"] / ship, 3), reve_vs_fbraw=round(mm["reve"] / mm["fbraw"], 3),
                          note="forward only, one REVE net added to the shipped four per-family nets; fine-tuning refits of the shipped slots are not in this ratio")), flush=True)


if __name__ == "__main__":
    main()
