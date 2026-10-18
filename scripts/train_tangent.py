"""Train the aligned tangent-space decoder and export a submission folder.

--train official : official train split only (comparable to the baseline).
--train all      : all public participants (candidate for upload).
Checks that the standalone submission reproduces the harness predictions
window by window on the official test streams, then writes the folder.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))

import protocol  # noqa: E402
from features import BANDS  # noqa: E402
from models import TangentSpace  # noqa: E402

FB = ("theta", "alpha", "beta1", "beta2", "gamma", "high")
VARIANTS = {  # name -> TangentSpace keyword arguments
    "v1": dict(bands=FB, means="center"),
    "v2": dict(bands=("delta",) + FB, means="center", slow="wave1"),
}
SRC = ROOT / "submissions/tangent/submission.py"


def export(model, out, variant):
    scaler, lr = model.clf[0], model.clf[1]
    slow_kind = VARIANTS[variant].get("slow", "mean")
    slow = ({} if slow_kind == "mean" else
            {"slow_cutoff": np.float64(1.0 if slow_kind == "wave1" else 2.0),
             "slow_step": np.int64(60)})
    np.savez(out / "weights.npz",
             bands=np.array([BANDS[b] for b in model.bands], dtype=np.float64),
             log_glob=np.stack(model.log_glob), k=np.float64(model.k),
             shrink=np.float64(1e-3), mu=scaler.mean_, sd=scaler.scale_,
             W=lr.coef_, b=lr.intercept_, **slow)
    shutil.copyfile(SRC, out / "submission.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=("official", "all"), default="official")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="v2")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = protocol.load()
    tr = d["split"] == "train" if args.train == "official" else np.ones(len(d["y"]), bool)
    te = d["split"] == "test"
    model = TangentSpace(d, **VARIANTS[args.variant])
    model.fit(np.flatnonzero(tr))
    args.out.mkdir(parents=True, exist_ok=True)
    export(model, args.out, args.variant)

    sys.path.insert(0, str(args.out))
    import submission  # noqa: E402
    solver = submission.Solver.__new__(submission.Solver)
    sub = solver.load_model({"submission_dir": args.out})
    ref = protocol.predict_streams(model, d, te)
    got, streams = [], d["stream"][te]
    for i, j in enumerate(np.flatnonzero(te)):
        if i == 0 or streams[i] != streams[i - 1]:
            sub.reset_state()
        got.append(int(sub.predict(d["X"][j:j + 1])[0]))
    got = np.array(got)
    agree = float(np.mean(got == ref))
    res = {"train": args.train, "variant": args.variant,
           "agreement_with_harness": agree,
           "harness": protocol.score(d["y"][te], ref, d, te),
           "submission": protocol.score(d["y"][te], got, d, te),
           "weights_sha256": hashlib.sha256(
               (args.out / "weights.npz").read_bytes()).hexdigest()}
    for r in (res["harness"], res["submission"]):
        r.pop("cell_scores")
    (args.out.parent / f"{args.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert agree == 1.0, "standalone submission diverges from harness"


if __name__ == "__main__":
    main()
