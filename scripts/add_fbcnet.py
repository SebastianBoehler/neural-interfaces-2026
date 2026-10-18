"""Append aligned FBCNets to an exported ensemble package.

Usage: python scripts/add_fbcnet.py SRC_DIR OUT_DIR --train official|all
       [--seeds 5] [--share 0.25] [--rb-alpha 0.5] [--rb-k 5]
Trains FBCNets (harness ``AlignedNet``, seeds 0..n-1) on the same windows
as the source package, scales the source members' weights by 1 - share and
gives the FBCNets ``share`` equally. Checks every official-test window,
streamed per session: the new package equals the harness computation
(source package member mix x (1 - share) + harness FBCNet probabilities,
then ``experiments.ensemble.rebalance``).
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
from add_rebalance import SRC, load_solver, stream  # noqa: E402
from deep import AlignedNet  # noqa: E402
from ensemble import rebalance  # noqa: E402


def fbc_params(m):
    net = m.net.to("cpu").eval()
    sd = {k: v.detach().double().numpy() for k, v in net.state_dict().items()}
    out = {f"fbc_fir{i}": f.detach().double().numpy()
           for i, f in enumerate(net.spectral_filtering.fir_list)}
    out.update(
        fbc_n_bands=np.int64(len(net.spectral_filtering.fir_list)),
        fbc_W=net.spatial_conv[0].weight.detach().double().numpy()[:, 0, :, 0],
        fbc_bw=sd["spatial_conv.0.bias"], fbc_rm=sd["spatial_conv.1.running_mean"],
        fbc_rv=sd["spatial_conv.1.running_var"], fbc_g=sd["spatial_conv.1.weight"],
        fbc_be=sd["spatial_conv.1.bias"], fbc_eps=np.float64(net.spatial_conv[1].eps),
        fbc_L=net.final_layer.weight.detach().double().numpy(),
        fbc_lb=sd["final_layer.bias"], fbc_stride=np.int64(net.stride_factor),
        align=np.bool_(m.align), k=np.float64(m.k), G=m.G,
        scale=np.float64(m.scale), shrink=np.float64(1e-3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--train", choices=("official", "all"), required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--share", type=float, default=0.25)
    ap.add_argument("--rb-alpha", type=float, default=0.5)
    ap.add_argument("--rb-k", type=float, default=5.0)
    args = ap.parse_args()
    d = protocol.load()
    tr = d["split"] == "train" if args.train == "official" else np.ones(len(d["y"]), bool)
    te = d["split"] == "test"
    with np.load(args.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    n_old = int(z["n_nets"])
    P_old = stream(load_solver(args.src, "src_pkg"), d, te, lambda m, x: m.mixed_proba(x))

    t0 = time.time()
    nets = [AlignedNet(d, arch="fbcnet", seed=s) for s in range(args.seeds)]
    for m in nets:
        m.fit(np.flatnonzero(tr))
        m.net.to("cpu")
        m.dev = "cpu"
    fit_s = time.time() - t0
    w_new = args.share / len(nets)
    z["member_weights"] = np.concatenate(
        [z["member_weights"] / z["member_weights"].sum() * (1 - args.share),
         np.full(len(nets), w_new)])
    for i, m in enumerate(nets):
        z.update({f"net{n_old + i}_{k}": v for k, v in fbc_params(m).items()})
    z.update(n_nets=np.int64(n_old + len(nets)), rb_alpha=np.float64(args.rb_alpha),
             rb_k=np.float64(args.rb_k))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "weights.npz", **z)
    shutil.copyfile(SRC, args.out / "submission.py")

    P = (1 - args.share) * P_old / P_old.sum(1, keepdims=True)
    P = P + sum(w_new * protocol.predict_streams(m, d, te, proba=True) for m in nets)
    streams = d["stream"][te]
    ref = np.concatenate([rebalance(P[streams == s], args.rb_alpha, args.rb_k).argmax(1)
                          for s in dict.fromkeys(streams)])
    t1 = time.time()
    got = stream(load_solver(args.out, "new_pkg"), d, te, lambda m, x: m.predict(x))
    per_window = (time.time() - t1) / len(got)
    res = {"source": str(args.src), "train": args.train, "fbc_seeds": args.seeds,
           "fbc_share": args.share, "rb_alpha": args.rb_alpha, "rb_k": args.rb_k,
           "agreement_with_harness": float(np.mean(got == ref)),
           "fbc_fit_seconds": round(fit_s, 1),
           "submission_seconds_per_window_cpu": round(per_window, 5),
           "harness": protocol.score(d["y"][te], ref, d, te),
           "submission": protocol.score(d["y"][te], got, d, te),
           "weights_sha256": hashlib.sha256((args.out / "weights.npz").read_bytes()).hexdigest()}
    for r in (res["harness"], res["submission"]):
        r.pop("cell_scores")
    (args.out.parent / f"{args.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["agreement_with_harness"] == 1.0, "package diverges from harness"


if __name__ == "__main__":
    main()
