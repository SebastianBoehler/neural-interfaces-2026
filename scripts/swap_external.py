"""Retrain package members with external data; keep all other members.

Usage: python scripts/swap_external.py SRC_DIR OUT_DIR --ext scherer2015~n~w0.1
       --train official|all [--kinds shallow_raw,fbcnet] [--seeds 5]
Slots of each kind (shallow_raw / shallow_ea: ShallowConvNets with align
False / True; fbcnet / fbcnet_raw: aligned / unaligned FBCNets) are retrained by the harness ``AlignedNet`` (seeds
0..n-1, in slot order) on the package's PROTEUS training windows plus the
external windows (``protocol.load``; test folds never contain them). Member
weights, v2 and the other nets are copied unchanged. Checks every
official-test window, streamed per session: the new package equals the
harness computation (source package mix of the kept members + harness
probabilities of the new nets, then ``experiments.ensemble.rebalance``).
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))

import protocol  # noqa: E402
from add_fbcnet import fbc_params  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402
from deep import AlignedNet  # noqa: E402
from ensemble import rebalance  # noqa: E402
from train_ensemble import net_params  # noqa: E402


def slot_kind(z, i):
    arch = "fbcnet" if f"net{i}_fbc_n_bands" in z else "shallow"
    if bool(z[f"net{i}_align"]):
        return "fbcnet" if arch == "fbcnet" else "shallow_ea"
    return f"{arch}_raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--train", choices=("official", "all"), required=True)
    ap.add_argument("--kinds", default="shallow_raw,fbcnet")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    kinds = args.kinds.split(",")
    d0, dx = protocol.load(), protocol.load(tuple(args.ext.split("+")))
    n0 = len(d0["y"])
    assert np.array_equal(dx["X"][:n0], d0["X"][:n0])
    pro = d0["split"] == "train" if args.train == "official" else np.ones(n0, bool)
    tr = np.concatenate([np.flatnonzero(pro), np.arange(n0, len(dx["y"]))])
    te = d0["split"] == "test"
    te_x = np.concatenate([te, np.zeros(len(dx["y"]) - n0, bool)])
    with np.load(args.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    n_nets, w = int(z["n_nets"]), z["member_weights"].astype(np.float64)
    w = w / w.sum()
    slots = {k: [i for i in range(n_nets) if slot_kind(z, i) == k] for k in kinds}
    assert all(len(s) == args.seeds for s in slots.values()), slots

    kept = z.copy()  # member weight 0 removes the swapped slots from the mix
    kept["member_weights"] = w.copy()
    swapped = [i for s in slots.values() for i in s]
    kept["member_weights"][[1 + i for i in swapped]] = 0.0
    tmp = args.out.parent / f"{args.out.name}_kept"
    tmp.mkdir(parents=True, exist_ok=True)
    np.savez(tmp / "weights.npz", **kept)
    shutil.copyfile(SRC, tmp / "submission.py")
    P_kept = stream(load_solver(tmp, "kept_pkg"), d0, te, lambda m, x: m.mixed_proba(x))
    shutil.rmtree(tmp)

    t0 = time.time()
    P = P_kept * kept["member_weights"].sum()
    for kind, idx in slots.items():
        for seed, i in enumerate(idx):
            fbc = kind.startswith("fbcnet")
            m = AlignedNet(dx, arch="fbcnet" if fbc else "shallow",
                           align=not kind.endswith("_raw"), seed=seed)
            m.fit(tr)
            m.net.to("cpu")
            m.dev = "cpu"
            for key in [k for k in z if k.startswith(f"net{i}_")]:
                del z[key]
            params = fbc_params(m) if fbc else net_params(m)
            z.update({f"net{i}_{k}": v for k, v in params.items()})
            P = P + w[1 + i] * protocol.predict_streams(m, dx, te_x, proba=True)
    fit_s = time.time() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "weights.npz", **z)
    shutil.copyfile(SRC, args.out / "submission.py")

    a, k, win = float(z["rb_alpha"]), float(z["rb_k"]), int(z.get("rb_w", 0))
    streams = d0["stream"][te]
    ref = np.concatenate([rebalance(P[streams == s], a, k, win).argmax(1)
                          for s in dict.fromkeys(streams)])
    t1 = time.time()
    got = stream(load_solver(args.out, "new_pkg"), d0, te, lambda m, x: m.predict(x))
    per_window = (time.time() - t1) / len(got)
    res = {"source": str(args.src), "ext": args.ext, "train": args.train,
           "swapped_slots": slots, "seeds": args.seeds,
           "n_train_windows": {"proteus": int(pro.sum()), "external": int(len(tr) - pro.sum())},
           "agreement_with_harness": float(np.mean(got == ref)),
           "fit_seconds": round(fit_s, 1),
           "submission_seconds_per_window_cpu": round(per_window, 5),
           "harness": protocol.score(d0["y"][te], ref, d0, te),
           "submission": protocol.score(d0["y"][te], got, d0, te),
           "weights_sha256": hashlib.sha256((args.out / "weights.npz").read_bytes()).hexdigest()}
    for r in (res["harness"], res["submission"]):
        r.pop("cell_scores")
    (args.out.parent / f"{args.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["agreement_with_harness"] == 1.0, "package diverges from harness"


if __name__ == "__main__":
    main()
