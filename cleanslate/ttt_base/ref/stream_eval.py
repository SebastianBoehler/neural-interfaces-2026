"""Team C quick-eval: stream one held person's sessions through the package's model under shuffled delivery.

python cleanslate/teamC/stream_eval.py --pkg DIR --held P01 --orders 0,1 --variants A0,A1,A2 --out FILE.npz

Order k = runs/shuffle_perms.npz 'full|k|all_loso' (full-session shuffle; k=0 is the order of cap250's full0 bank,
so order 0 is paired with cap250 73.74). Every session runs on copy.deepcopy(model) + reset_state(), one window per
predict() call, exactly as benchmark_utils.streaming. Saves per variant/order: labels, posteriors, seconds per window.
Variants are cfg overrides of the package's TTT settings (VARIANTS below); unknown keys fail in TTTModel.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"

VARIANTS = {
    "A0": dict(bn="source", steps=0, balance=0),            # source ensemble, causal EA only
    "A1": dict(bn="prefix", steps=0, balance=0),            # + prefix BN statistics
    "A1b": dict(bn="prefix", steps=0, balance=1),           # + causal prefix Sinkhorn on the output
    "A2": dict(bn="prefix", steps=4, balance=0),            # + gradient TTT (all params, PL + IM + anchor)
    "A2b": dict(bn="prefix", steps=4, balance=1),
    "A2h": dict(bn="prefix", steps=8, lr=3e-4, balance=0),  # heavier TTT
    "A2n": dict(bn="prefix", steps=4, params="bn+fc", lr=1e-3, balance=0),  # BN affine + head only
    # ttt_base (registered after b0 collapsed under A2/A2h): gentler forms
    "A2l": dict(bn="prefix", steps=4, lr=2e-5, balance=0),
    "A2bn": dict(bn="prefix", steps=4, params="bn", lr=1e-3, balance=0),
    "A2f": dict(bn="prefix", steps=4, freeze_bd=1),  # A2 on the team C members only; braindecode members BN-refresh only
    "A2lf": dict(bn="prefix", steps=4, lr=2e-5, freeze_bd=1),
    # C2: heavier TTT around A2h (8 steps, lr 3e-4)
    "A3a": dict(bn="prefix", steps=16, lr=3e-4),
    "A3b": dict(bn="prefix", steps=8, lr=1e-3),
    "A3c": dict(bn="prefix", steps=8, lr=3e-4, anchor=0.0),
    "A3d": dict(bn="prefix", steps=8, lr=3e-4, lam_im=0.0),
    "A3e": dict(bn="prefix", steps=8, lr=3e-4, tau=0.5),
    "A3f": dict(bn="prefix", steps=8, lr=3e-4, every=2, dense_until=64),
}


def load_model(pkg, dev, cfg):
    sys.path.insert(0, str(Path(pkg).resolve()))
    import submission  # the package's own file
    assert Path(submission.__file__).resolve().parent == Path(pkg).resolve(), submission.__file__
    return submission.load_package(pkg, dev, cfg)


def p_seen(perms, k, query):
    """Returning dev order: the seen_session permutation of order k, restricted to the streamed rows."""
    with np.load(perms) as p:
        rows = p[f"full|{k}|seen_session"]
    return rows[np.isin(rows, query)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--orders", default="0")
    ap.add_argument("--variants", default="A0,A1,A2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", default="", help="returning dev: only these sessions of HELD (must be untrained)")
    ap.add_argument("--data", default=str(ROOT / "data/proteus_windows.npz"))
    ap.add_argument("--perms", default=str(ROOT / "runs/shuffle_perms.npz"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert hashlib.sha256(Path(args.perms).read_bytes()).hexdigest() == ORDER_SHA, "order archive changed"
    d = np.load(args.data)
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    tj = json.load(open(Path(args.pkg) / "train.json"))
    stream = np.char.add(np.char.add(person, "|"), d["session"])
    query = np.flatnonzero(person == args.held)
    if args.sessions:
        query = query[np.isin(d["session"][query], args.sessions.split(","))]
        assert not set(stream[query]) & set(tj["streams"]), "a streamed session was trained on"
    else:
        assert args.held not in tj["people"], "held person in the package's training people"
    out, meta = {}, []
    for k in map(int, args.orders.split(",")):
        with np.load(args.perms) as p:
            rows = p[f"full|{k}|all_loso"]
        rows = rows[np.isin(rows, query)] if not args.sessions else p_seen(args.perms, k, query)
        assert np.array_equal(np.sort(rows), query)
        out[f"rows|{k}"] = rows
        for v in args.variants.split(","):
            model = load_model(args.pkg, dev, VARIANTS[v])
            lab, post, sec = [], [], []
            for st in dict.fromkeys(stream[rows]):
                m = copy.deepcopy(model)
                m.reset_state()
                for r in rows[stream[rows] == st]:
                    x = torch.as_tensor(d["X"][r:r + 1])
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    y = m.predict(x)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    sec.append(time.perf_counter() - t0)
                    assert y.shape == (1,)
                    lab.append(int(y[0]))
                    post.append(m.last.detach().cpu().numpy())
            lab, post = np.array(lab), np.array(post)
            out[f"{v}|{k}|label"], out[f"{v}|{k}|post"], out[f"{v}|{k}|sec"] = lab, post, np.array(sec)
            acc = float((lab == d["y"][rows]).mean())
            meta.append(dict(variant=v, order=k, acc=round(acc, 4), ms=round(1e3 * float(np.mean(sec)), 2)))
            print("PASS", args.held, meta[-1], flush=True)
    np.savez(args.out, **out)
    json.dump(dict(held=args.held, pkg=str(args.pkg), runs=meta, dev=dev,
                   gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu"), open(args.out + ".json", "w"))


if __name__ == "__main__":
    main()
