"""Run named candidate configurations under one or more protocols.

Usage: python experiments/run_cv.py NAME [NAME ...] [--protocols p1,p2]
NAME@ext1+ext2 trains NAME with data/ext1/windows.npz etc. added to every
training fold (test folds stay PROTEUS-only; see protocol.py).
Results are appended as JSON lines to runs/experiments/results.jsonl.
Per-window class probabilities (models with ``predict_proba_stream``) go to
runs/experiments/probs/NAME-PROTOCOL-sSEED.npz for offline ensembling.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import protocol
import numpy as np
from deep import AlignedNet
from erp import ErpTangent
from ensemble import Ensemble
from models import MeanLinear, ReveLinear, TangentSpace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get("NI26_RESULTS", "runs/experiments/results.jsonl")
PROBS = ROOT / os.environ.get("NI26_PROBS", "runs/experiments/probs")
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
for _a in ("eegnet", "shallow", "deep4", "conformer", "atcnet", "eegnex",
           "ctnet", "fbcnet", "ifnet", "tidnet"):
    CONFIGS[f"{_a}_raw"] = lambda d, a=_a: AlignedNet(d, arch=a, align=False, seed=SEED[0])
    CONFIGS[f"{_a}_ea"] = lambda d, a=_a: AlignedNet(d, arch=a, seed=SEED[0])
for _a in ("shallow", "fbcnet"):
    CONFIGS[f"{_a}_ea_aug"] = lambda d, a=_a: AlignedNet(d, arch=a, seed=SEED[0], aug=True)
for _a in ("shallow", "fbcnet"):  # fine-tune 10 epochs on PROTEUS after external
    CONFIGS[f"{_a}_ea_ft"] = lambda d, a=_a: AlignedNet(d, arch=a, seed=SEED[0], ft=10)
    CONFIGS[f"{_a}_raw_ft"] = lambda d, a=_a: AlignedNet(d, arch=a, align=False,
                                                        seed=SEED[0], ft=10)
CONFIGS["shallow_raw_aug"] = lambda d: AlignedNet(d, arch="shallow", align=False,
                                                  seed=SEED[0], aug=True)
for _a in ("shallow", "fbcnet"):  # causal test-time batch-norm adaptation
    for _al in (False, True):
        for _k in (0, 5, 20, 100):
            CONFIGS[f"{_a}_{'ea' if _al else 'raw'}_bn{_k}"] = (
                lambda d, a=_a, al=_al, k=_k: AlignedNet(d, arch=a, align=al,
                                                         seed=SEED[0], adabn=k))
for _c, _nf in ((4.0, 4), (2.0, 4), (8.0, 4), (4.0, 2), (4.0, 8)):
    CONFIGS[f"erp_c{_c:g}_f{_nf}"] = lambda d, c=_c, nf=_nf: ErpTangent(d, cutoff=c, nf=nf)
CONFIGS["erp_noalign"] = lambda d: ErpTangent(d, align=False)
CONFIGS["reve_raw"] = lambda d: ReveLinear(d, center=False)
CONFIGS["reve_ctr"] = lambda d: ReveLinear(d)
CONFIGS["reve_ctr_C1"] = lambda d: ReveLinear(d, C=1.0)
CONFIGS["reve_ctr_C001"] = lambda d: ReveLinear(d, C=0.01)
V2 = dict(bands=("delta",) + FB, means="center", slow="wave1")
CONFIGS["ens_ea"] = lambda d: Ensemble([TangentSpace(d, **V2), CONFIGS["shallow_ea"](d)])
CONFIGS["ens_raw"] = lambda d: Ensemble([TangentSpace(d, **V2), CONFIGS["shallow_raw"](d)])
CONFIGS["ens_both"] = lambda d: Ensemble([TangentSpace(d, **V2), CONFIGS["shallow_ea"](d),
                                          CONFIGS["shallow_raw"](d)])
CONFIGS["v2_ws"] = lambda d: TangentSpace(d, **{**V2, "wscale": True})  # external data
for _k in (1.0, 20.0, 100.0):
    CONFIGS[f"v2_ks{_k:g}"] = lambda d, k=_k: TangentSpace(d, **{**V2, "k_slow": k})
for _C in (0.03, 0.3, 1.0, 3.0):
    CONFIGS[f"v2_C{_C:g}"] = lambda d, C=_C: TangentSpace(d, **{**V2, "C": C})
CONFIGS["v2_dc"] = lambda d: TangentSpace(d, **{**V2, "slow": "wave1d"})
for _k in (1.0, 20.0):
    CONFIGS[f"ts_sess_fb_k{_k:g}"] = lambda d, k=_k: TangentSpace(d, bands=FB, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--protocols", default="dev_loso,dev_session")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    SEED[0] = args.seed
    data = {}
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        base, _, ext = name.partition("@")
        if ext not in data:
            data[ext] = protocol.load(tuple(ext.split("+")) if ext else ())
        d = data[ext]
        for proto in args.protocols.split(","):
            t0 = time.time()
            res = protocol.evaluate(lambda: CONFIGS[base](d), d, proto)
            if "proba" in res:
                PROBS.mkdir(parents=True, exist_ok=True)
                np.savez(PROBS / f"{name}-{proto}-s{args.seed}.npz",
                         proba=res.pop("proba"), index=res.pop("index"))
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
