"""Refit the Hilbert tangent member (export_ts2.py): reference, scaler and head on the source windows."""

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import ROOT, family_args, load_source

os.environ.setdefault("NI26_ROOT", str(ROOT))
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import ts_core2

CONFIG = "car_fb7_hfs2_t0.1_le_k2"
C = 3.0


def fit(args):
    data = load_source(args.people, args.first_session_only, args.data)
    cfg = ts_core2.parse(CONFIG)
    if cfg["cov"] != "hfs2":
        raise ValueError(f"unexpected Hilbert layout {cfg['cov']}")
    head = f"lr{C:g}"
    n = len(data["y"])
    features = ts_core2.Feats(data, cfg)
    core = ts_core2.Core(features, [head], {head: np.full((n, 3), np.nan)})
    core.fit(np.arange(n))
    scaler = core.models[0].named_steps["standardscaler"]
    lr = core.models[0].named_steps["logisticregression"]
    if list(lr.classes_) != [0, 1, 2] or lr.coef_.shape[1] != features.F.shape[1]:
        raise ValueError("Hilbert head missing a task class or wrong width")
    nsub = int(cfg["cov"][3:])
    ts2 = dict(bands=np.array(ts_core2.BANDSETS[cfg["bands"]], dtype=np.float64),
               log_glob=np.stack(core.priors), k=np.array(cfg["k"]), shrink=np.array(float(cfg["est"][1:])),
               mu=scaler.mean_, sd=scaler.scale_, W=lr.coef_, b=lr.intercept_,
               slow_cutoff=np.array(1.0), slow_step=np.array(60), car=np.array(1), nsub=np.array(nsub),
               hilbert=np.array(1))
    if ts2["log_glob"].shape[0] != len(ts2["bands"]) * (1 + nsub) or not np.iscomplexobj(ts2["log_glob"]):
        raise ValueError("Hilbert reference has the wrong shape")
    np.savez(args.out / "weights.npz", **{f"ts2_{k}": v for k, v in ts2.items()})


if __name__ == "__main__":
    fit(family_args(__doc__, (0,)))
