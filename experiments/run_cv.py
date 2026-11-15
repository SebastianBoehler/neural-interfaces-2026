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
import re
import subprocess
import time
from pathlib import Path

import protocol
import numpy as np
from deep import R9_ARCHS, AlignedNet
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
           "ctnet", "fbcnet", "ifnet", "tidnet", "rocket", *R9_ARCHS):
    CONFIGS[f"{_a}_raw"] = lambda d, a=_a: AlignedNet(d, arch=a, align=False, seed=SEED[0])
    CONFIGS[f"{_a}_ea"] = lambda d, a=_a: AlignedNet(d, arch=a, seed=SEED[0])
# s274: EEGNet one-factor sweep around braindecode 1.8.1 defaults (kernel_length 64, F1 8, D 2, drop_prob 0.25); family key -> net kwargs
EEGN_SWEEP = {"k30": dict(kernel_length=30), "k60": dict(kernel_length=60), "k90": dict(kernel_length=90), "f16": dict(F1=16),
              "d4": dict(D=4), "p25": dict(drop_prob=0.25), "p50": dict(drop_prob=0.5)}  # p25 equals the default (replicate)
for _t, _kw in EEGN_SWEEP.items():
    CONFIGS[f"eegnet{_t}_ea"] = lambda d, kw=_kw: AlignedNet(d, arch="eegnet", seed=SEED[0], net_kw=dict(kw))
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
for _al in (False, True):  # fine-tuned foundation model (CBraMod)
    _t = "ea" if _al else "raw"
    for _lr, _ep in ((1e-4, 30), (3e-4, 30), (1e-4, 60)):
        CONFIGS[f"cbramod_{_t}_lr{_lr:g}_e{_ep}"] = (
            lambda d, al=_al, lr=_lr, ep=_ep: AlignedNet(d, arch="cbramod", align=al,
                                                        lr=lr, epochs=ep, wd=5e-2,
                                                        seed=SEED[0]))
    CONFIGS[f"cbramod_{_t}_scratch"] = (
        lambda d, al=_al: AlignedNet(d, arch="cbramod", align=al, lr=1e-4, epochs=30,
                                     wd=5e-2, seed=SEED[0], pretrained=False))
_CB = dict(arch="cbramod", lr=1e-4, epochs=30, wd=5e-2)
for _n, _kw in (("ls", dict(ls=0.1)), ("lld", dict(lr=3e-4, lld=0.75)),
                ("lpft", dict(lp=5)), ("rec", dict(ls=0.1, lr=3e-4, lld=0.75, lp=5))):
    CONFIGS[f"cbramod_ea_{_n}"] = lambda d, kw=_kw: AlignedNet(
        d, **{**_CB, **kw}, seed=SEED[0])
for _a in ("shallow", "fbcnet"):  # extra training crops from the task period
    for _al in (False, True):
        for _n, _cr in (("c23", (2.0, 3.0)), ("c234", (2.0, 3.0, 4.0)), ("c05", (0.5,))):
            CONFIGS[f"{_a}_{'ea' if _al else 'raw'}_{_n}"] = (
                lambda d, a=_a, al=_al, cr=_cr: AlignedNet(d, arch=a, align=al,
                                                           seed=SEED[0], crops=cr))
CONFIGS["labram_ea"] = lambda d: AlignedNet(d, **{**_CB, "arch": "labram"}, seed=SEED[0])
CONFIGS["labram_ea_lld"] = lambda d: AlignedNet(  # LaBraM's own fine-tuning recipe
    d, **{**_CB, "arch": "labram", "lr": 5e-4, "lld": 0.65}, seed=SEED[0])
CONFIGS["reve_ea"] = lambda d: AlignedNet(d, **{**_CB, "arch": "reve"}, seed=SEED[0])
CONFIGS["reve_ea_lld"] = lambda d: AlignedNet(  # higher top lr, layer decay 0.8
    d, **{**_CB, "arch": "reve", "lr": 3e-4, "lld": 0.8}, seed=SEED[0])
for _al in (False, True):  # FBCNet variants (packageable: LogVar layer)
    for _n, _kw in (("s2", dict(stride_factor=2)), ("s8", dict(stride_factor=8)),
                    ("f16", dict(n_filters_spat=16)), ("f64", dict(n_filters_spat=64))):
        CONFIGS[f"fbcnet_{'ea' if _al else 'raw'}_{_n}"] = (
            lambda d, al=_al, kw=_kw: AlignedNet(d, arch="fbcnet", align=al,
                                                 seed=SEED[0], net_kw=kw))
for _n, _kw in (("lo", dict(first_freq=1.0)),  # SincShallowNet variants (R9):
                ("k65", dict(first_freq=1.0, time_filter_len=65)),  # slow band, longer
                ("n64", dict(first_freq=1.0, num_time_filters=64))):  # FIRs, more filters
    CONFIGS[f"sincshallow_ea_{_n}"] = lambda d, kw=_kw: AlignedNet(
        d, arch="sincshallow", seed=SEED[0], net_kw=kw)
CONFIGS["sincshallow_ea_e80"] = lambda d: AlignedNet(d, arch="sincshallow", seed=SEED[0],
                                                     epochs=80)
_REC = {"lr5e4": dict(lr=5e-4), "lr2e3": dict(lr=2e-3), "e20": dict(epochs=20),
        "e80": dict(epochs=80), "wd1e2": dict(wd=1e-2), "ls": dict(ls=0.1),
        "aug": dict(aug=True), "b32": dict(batch=32)}
_NKW = {"sincshallow": {"d25": dict(drop_prob=0.25), "d70": dict(drop_prob=0.7),
                        "bw8": dict(bandwidth=8.0), "p30": dict(pool_size=30, pool_stride=6),
                        "p110": dict(pool_size=110, pool_stride=24)},
        "fblight": {"w480": dict(win_len=480), "w240": dict(win_len=240),
                    "f64": dict(n_filters_spat=64)}}
for _a in ("sincshallow", "fblight"):  # recipe sweep of the R9 families (aligned)
    for _n, _kw in _REC.items():
        CONFIGS[f"{_a}_ea_r{_n}"] = lambda d, a=_a, kw=_kw: AlignedNet(
            d, arch=a, seed=SEED[0], **kw)
    for _n, _kw in _NKW[_a].items():
        CONFIGS[f"{_a}_ea_r{_n}"] = lambda d, a=_a, kw=_kw: AlignedNet(
            d, arch=a, seed=SEED[0], net_kw=kw)
for _a, _al in (("fbcnet", True), ("fbcnet", False), ("shallow", True),
                ("shallow", False), ("sincshallow", True)):  # time segments (R9)
    for _n, _tc in (("h1", (0, 240)), ("h2", (240, 480)), ("t3", (0, 360))):
        CONFIGS[f"{_a}_{'ea' if _al else 'raw'}_{_n}"] = (
            lambda d, a=_a, al=_al, tc=_tc: AlignedNet(d, arch=a, align=al, seed=SEED[0],
                                                       tcrop=tc))
for _a in ("shallow", "fbcnet", "sincshallow", "fblight"):  # session RMS only (R9)
    CONFIGS[f"{_a}_sn"] = lambda d, a=_a: AlignedNet(d, arch=a, align="scalar", seed=SEED[0])
CONFIGS["fbcnet_sn_tf"] = lambda d: AlignedNet(d, arch="fbcnet", align="scalar",
                                               seed=SEED[0], train_fir=True)
for _al in (False, True):  # FBCNet with a learned filter bank (R8)
    _t = "ea" if _al else "raw"
    CONFIGS[f"fbcnet_{_t}_tf"] = lambda d, al=_al: AlignedNet(
        d, arch="fbcnet", align=al, seed=SEED[0], train_fir=True)
    for _n, _kw in (("s2", dict(stride_factor=2)), ("s8", dict(stride_factor=8)),
                    ("f16", dict(n_filters_spat=16)), ("f64", dict(n_filters_spat=64))):
        CONFIGS[f"fbcnet_{_t}_{_n}_tf"] = (
            lambda d, al=_al, kw=_kw: AlignedNet(d, arch="fbcnet", align=al, seed=SEED[0],
                                                 net_kw=kw, train_fir=True))
for _a in ("shallow", "fbcnet"):  # participant-aware training objectives (R8)
    for _al in (False, True):
        _t = "ea" if _al else "raw"
        for _n, _kw in (("pbal", dict(pbal=True)), ("dro01", dict(dro=0.01)),
                        ("dro1", dict(dro=0.1))):
            CONFIGS[f"{_a}_{_t}_{_n}"] = lambda d, a=_a, al=_al, kw=_kw: AlignedNet(
                d, arch=a, align=al, seed=SEED[0], **kw)
CONFIGS["v2_lda"] = lambda d: TangentSpace(d, **{**V2, "clf": "lda"})  # generative head
CONFIGS["v2_pbal"] = lambda d: TangentSpace(d, **{**V2, "group": "pbal"})
for _e in (0.5, 2.0):
    CONFIGS[f"v2_dro{_e:g}"] = lambda d, e=_e: TangentSpace(d, **{**V2, "group": "dro", "dro_eta": e})
CONFIGS["cbramod_ea_c23"] = lambda d: AlignedNet(d, **_CB, crops=(2.0, 3.0), seed=SEED[0])
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
for _e, _t in ((0.02, "02"), (0.05, "05"), (0.1, "1")):  # R15: rotated training copies
    CONFIGS[f"v2_rot{_t}"] = lambda d, e=_e: TangentSpace(d, **{**V2, "rot": e})
for _k in (1.0, 20.0, 100.0):
    CONFIGS[f"v2_ks{_k:g}"] = lambda d, k=_k: TangentSpace(d, **{**V2, "k_slow": k})
for _C in (0.03, 0.3, 1.0, 3.0):
    CONFIGS[f"v2_C{_C:g}"] = lambda d, C=_C: TangentSpace(d, **{**V2, "C": C})
CONFIGS["v2_dc"] = lambda d: TangentSpace(d, **{**V2, "slow": "wave1d"})
for _k in (1.0, 20.0):
    CONFIGS[f"ts_sess_fb_k{_k:g}"] = lambda d, k=_k: TangentSpace(d, bands=FB, k=k)

PT = re.compile(r"(shallow|fbcnet|sincshallow|fblight|eegnex|ifnet|eegnet|deep4|conformer|atcnet)_(ea|raw)(_tf)?_pt([a-z0-9]+)(_lo|_lp)?")


def pt_config(base):
    """NAME_ALIGN[_tf]_ptTAG[_lo|_lp]: fine-tune the trunk pretrained by
    experiments/pretrain.py (runs/pretrain/NAME_ALIGN-TAG-sSEED.pt) on PROTEUS,
    at lr 1e-3, 3e-4 (_lo) or 3e-4 after a 5-epoch linear probe (_lp)."""
    a, t, tf, tag, rec = PT.fullmatch(base).groups()
    kw = {"_lo": dict(lr=3e-4), "_lp": dict(lr=3e-4, lp=5)}.get(rec, {})
    return lambda d: AlignedNet(d, arch=a, align=t == "ea", seed=SEED[0], train_fir=bool(tf),
                                init=f"{ROOT}/runs/pretrain/{a}_{t}-{tag}-s{{seed}}.pt", **kw)

# R11 training options as name suffixes on any AlignedNet config (experiments/dg.py)
MODS = {"swa": dict(swa=0.5), "swa25": dict(swa=0.25), "swa75": dict(swa=0.75),
        "cmix": dict(cmix=0.4), "cmix1": dict(cmix=1.0),
        "rot02": dict(rot=0.02), "rot05": dict(rot=0.05), "rot1": dict(rot=0.1),
        "fsc1": dict(fsc=0.1), "fsc2": dict(fsc=0.2),
        "ms5": dict(mstyle=0.5), "adv01": dict(adv=0.1), "adv03": dict(adv=0.3),  # R18
        "rb440": dict(ref_band=(4, 40)),  # R19
        "rc5": dict(ref_clip=5.0),  # R20: artifact-robust EA reference
        "ct03": dict(cotrain=0.3), "ct1": dict(cotrain=1.0), "ct3": dict(cotrain=3.0)}  # co-training (cotrain.py)


def resolve(base):
    if base in CONFIGS:
        return CONFIGS[base]
    head, _, mod = base.rpartition("_")
    if mod in MODS:
        inner = resolve(head)
        return lambda d: _with(inner(d), MODS[mod])
    return pt_config(base)


def _with(net, kw):
    for k, v in kw.items():
        setattr(net, k, v)
    return net



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
            make = resolve(base)
            res = protocol.evaluate(lambda: make(d), d, proto)
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
