"""Train the v2 + ShallowConvNet ensemble and export a submission folder.

--train official : official train split only (honest warm-up replay).
--train all      : all public participants (candidate for upload).
--deep           : shallow_ea, shallow_raw or both; --seeds nets per kind.
--v2-weight      : share of the ensemble weight on v2; the rest is split
                   equally over the nets (0.5 with one kind = the harness
                   "v2 + seed-averaged member" ensemble).
Fits through the harness (experiments/ensemble.py), exports numpy weights,
then checks window by window that submissions/ensemble/submission.py
reproduces the harness on the official test streams (nets on CPU in both).
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))

import protocol  # noqa: E402
from deep import AlignedNet  # noqa: E402
from ensemble import Ensemble  # noqa: E402
from features import BANDS  # noqa: E402
from models import TangentSpace  # noqa: E402
from run_cv import V2  # noqa: E402

SRC = ROOT / "submissions/ensemble/submission.py"
NET_KEYS = {"conv_time_spat.conv_time.weight": "time_w",
            "conv_time_spat.conv_time.bias": "time_b",
            "conv_time_spat.conv_spat.weight": "spat_w",
            "bnorm.weight": "bn_w", "bnorm.bias": "bn_b",
            "bnorm.running_mean": "bn_mean", "bnorm.running_var": "bn_var",
            "final_layer.conv_classifier.weight": "cls_w",
            "final_layer.conv_classifier.bias": "cls_b"}


def tangent_params(m):
    scaler, lr = m.clf[0], m.clf[1]
    assert list(lr.classes_) == [0, 1, 2]
    return {"bands": np.array([BANDS[b] for b in m.bands], dtype=np.float64),
            "log_glob": np.stack(m.log_glob), "k": np.float64(m.k),
            "shrink": np.float64(1e-3), "mu": scaler.mean_, "sd": scaler.scale_,
            "W": lr.coef_, "b": lr.intercept_, "slow_cutoff": np.float64(1.0),
            "slow_step": np.int64(60)}


def net_params(m):
    sd = {k: v.detach().cpu().numpy() for k, v in m.net.state_dict().items()}
    out = {new: sd[old] for old, new in NET_KEYS.items()}
    assert set(sd) - set(NET_KEYS) == {"bnorm.num_batches_tracked"}, sorted(sd)
    out.update(align=np.bool_(m.align), k=np.float64(m.k), G=m.G,
               scale=np.float64(m.scale), shrink=np.float64(1e-3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=("official", "all"), default="official")
    ap.add_argument("--deep", choices=("shallow_ea", "shallow_raw", "both"),
                    default="shallow_raw")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--v2-weight", type=float, default=0.5)
    ap.add_argument("--rb-alpha", type=float, default=0.0,
                    help="causal class rebalancing exponent (0 = off)")
    ap.add_argument("--rb-k", type=float, default=5.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = protocol.load()
    tr = d["split"] == "train" if args.train == "official" else np.ones(len(d["y"]), bool)
    te = d["split"] == "test"
    kinds = ("shallow_ea", "shallow_raw") if args.deep == "both" else (args.deep,)
    nets = [AlignedNet(d, arch="shallow", align=kind == "shallow_ea", seed=s)
            for kind in kinds for s in range(args.seeds)]
    w = np.array([args.v2_weight] + [(1 - args.v2_weight) / len(nets)] * len(nets))
    model = Ensemble([TangentSpace(d, **V2)] + nets, weights=w,
                     alpha=args.rb_alpha, k=args.rb_k)
    t0 = time.time()
    model.fit(np.flatnonzero(tr))
    fit_s = time.time() - t0
    for m in nets:  # compare harness and submission on the same device
        m.net.to("cpu")
        m.dev = "cpu"
    z = {f"ts_{k}": v for k, v in tangent_params(model.members[0]).items()}
    for i, m in enumerate(nets):
        z.update({f"net{i}_{k}": v for k, v in net_params(m).items()})
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "weights.npz", n_nets=np.int64(len(nets)),
             member_weights=model.w, rb_alpha=np.float64(args.rb_alpha),
             rb_k=np.float64(args.rb_k), **z)
    shutil.copyfile(SRC, args.out / "submission.py")

    sys.path.insert(0, str(args.out))
    import submission  # noqa: E402
    solver = submission.Solver.__new__(submission.Solver)
    sub = solver.load_model({"submission_dir": args.out})
    ref = protocol.predict_streams(model, d, te, proba=True)
    got, probs, streams = [], [], d["stream"][te]
    t0 = time.time()
    for i, j in enumerate(np.flatnonzero(te)):
        if i == 0 or streams[i] != streams[i - 1]:
            sub.reset_state()
        got.append(int(sub.predict(d["X"][j:j + 1])[0]))
    per_window = (time.time() - t0) / len(got)
    got, ref_pred = np.array(got), ref.argmax(1)
    agree = float(np.mean(got == ref_pred))
    top2 = np.sort(ref, 1)[:, -2:]
    res = {"train": args.train, "deep": args.deep, "seeds": args.seeds,
           "rb_alpha": args.rb_alpha, "rb_k": args.rb_k,
           "member_weights": [round(float(x), 6) for x in model.w],
           "agreement_with_harness": agree,
           "min_harness_top2_margin": float((top2[:, 1] - top2[:, 0]).min()),
           "fit_seconds": round(fit_s, 1),
           "submission_seconds_per_window_cpu": round(per_window, 5),
           "harness": protocol.score(d["y"][te], ref_pred, d, te),
           "submission": protocol.score(d["y"][te], got, d, te),
           "weights_sha256": hashlib.sha256(
               (args.out / "weights.npz").read_bytes()).hexdigest(),
           "weights_bytes": (args.out / "weights.npz").stat().st_size}
    for r in (res["harness"], res["submission"]):
        r.pop("cell_scores")
    (args.out.parent / f"{args.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert agree == 1.0, "standalone submission diverges from harness"


if __name__ == "__main__":
    main()
