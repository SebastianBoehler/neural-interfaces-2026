"""Run named candidate configurations under one or more protocols.

Usage: python experiments/run_cv.py NAME [NAME ...] [--protocols p1,p2]
Results are appended as JSON lines to runs/experiments/results.jsonl.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import protocol
from deep import AlignedNet
from models import MeanLinear, TangentSpace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get("NI26_RESULTS", "runs/experiments/results.jsonl")
SEED = [0]  # set from --seed; used by stochastic (deep) configs
FB = ("theta", "alpha", "beta1", "beta2", "gamma", "high")

CONFIGS = {
    "mean_lr": lambda d: MeanLinear(d),
    "ts_none_ab": lambda d: TangentSpace(d, align="none"),
    "ts_sess_ab": lambda d: TangentSpace(d),
    "ts_sess_ab_tc": lambda d: TangentSpace(d, train_causal=True),
    "ts_none_fb": lambda d: TangentSpace(d, bands=FB, align="none"),
    "ts_sess_fb": lambda d: TangentSpace(d, bands=FB),
    "ts_sess_fb_tc": lambda d: TangentSpace(d, bands=FB, train_causal=True),
    "ts_sess_broad": lambda d: TangentSpace(d, bands=("broad",)),
    "ts_sess_fb_lda": lambda d: TangentSpace(d, bands=FB, clf="lda"),
    "ts_sess_fb_mraw": lambda d: TangentSpace(d, bands=FB, means="raw"),
    "ts_sess_fb_mctr": lambda d: TangentSpace(d, bands=FB, means="center"),
    "ts_sess_fb7_mctr": lambda d: TangentSpace(d, bands=("delta",) + FB, means="center"),
    "ts_sess_fb_mctr_st": lambda d: TangentSpace(d, bands=FB, means="center", stretch=True),
    "ts_sess_fb7_wctr": lambda d: TangentSpace(d, bands=("delta",) + FB, means="center", slow="wave"),
    "ts_sess_fb7_w1ctr": lambda d: TangentSpace(d, bands=("delta",) + FB, means="center", slow="wave1"),
}
CONFIGS["v1"] = CONFIGS["ts_sess_fb_mctr"]
CONFIGS["v2"] = CONFIGS["ts_sess_fb7_w1ctr"]
for _C in (0.01, 0.03, 0.3, 1.0):
    CONFIGS[f"ts_sess_fb_C{_C}"] = lambda d, C=_C: TangentSpace(d, bands=FB, C=C)
for _a in ("eegnet", "shallow"):
    CONFIGS[f"{_a}_raw"] = lambda d, a=_a: AlignedNet(d, arch=a, align=False, seed=SEED[0])
    CONFIGS[f"{_a}_ea"] = lambda d, a=_a: AlignedNet(d, arch=a, seed=SEED[0])
for _k in (1.0, 20.0):
    CONFIGS[f"ts_sess_fb_k{_k:g}"] = lambda d, k=_k: TangentSpace(d, bands=FB, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--protocols", default="dev_loso,dev_session")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    SEED[0] = args.seed
    d = protocol.load()
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        for proto in args.protocols.split(","):
            t0 = time.time()
            res = protocol.evaluate(lambda: CONFIGS[name](d), d, proto)
            res.update(name=name, protocol=proto, rev=rev, seed=args.seed,
                       seconds=round(time.time() - t0, 1))
            with OUT.open("a") as f:
                f.write(json.dumps(res) + "\n")
            folds = " ".join(f"{k}={v:.3f}" for k, v in res["per_fold"].items())
            print(f"{name:18s} s{args.seed} {proto:12s} cell={res['balanced_accuracy_cell_mean']:.4f} "
                  f"pooled={res['balanced_accuracy']:.4f} [{folds}] "
                  f"{res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
