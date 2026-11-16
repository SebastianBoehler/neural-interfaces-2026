"""§292D independent folds; reuse protocol, pinned orders and §288 backbone."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(WS / "experiments"), str(WS / "scripts"),
               str(WS / "studies/15nov/tsm288")]
import numpy as np
import torch
import protocol
import spd
from adapt import readout, stream

ORDERS = [("within", i) for i in range(4)] + [
    (s, i) for s in ("full", "runs") for i in range(2)]


def features(model, rows):
    x = torch.from_numpy(model._aligned(rows, causal=True)).to(model.dev)
    with torch.no_grad():
        net = model.net.eval()
        s = net.bimap(net.spdbn(net.spd_features(x)))
        expected = net(x).double().softmax(1).cpu().numpy()
        actual = readout(net, s, spd.logm, spd.sym_fn).double().softmax(1).cpu().numpy()
    assert np.allclose(actual, expected, atol=1e-7), "SPD extraction parity"
    return s.detach()


def guards(model, rows):
    rows = rows[:72]
    assert len(rows) > 40
    s = features(model, rows)
    prefix = features(model, rows[:40])
    assert torch.allclose(s[:40], prefix, atol=1e-7), "future feature influence"
    a = stream(model.net, s, "frozen", spd.logm, spd.sym_fn)[0]
    z = stream(model.net, s, "zero", spd.logm, spd.sym_fn)[0]
    assert np.array_equal(a, z), "zero bias differs"
    for mode in ("diag", "full"):
        p, stat = stream(model.net, s, mode, spd.logm, spd.sym_fn)
        short = stream(model.net, s[:40], mode, spd.logm, spd.sym_fn)[0]
        repeat = stream(model.net, s, mode, spd.logm, spd.sym_fn)[0]
        assert np.allclose(p[:40], short, atol=1e-7), "future update influence"
        assert np.array_equal(p, repeat), "fresh-session reset failed"
        assert np.max(np.abs(p[32:] - a[32:])) > 1e-8, "posterior no-op"
        print("GUARD", mode, stat, flush=True)
    print("GUARDS OK: extraction, prefix, zero, active, reset", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    d = protocol.load()
    folds = list(protocol.folds(d, "all_loso"))
    assert 0 <= args.task < 30
    seed, fold = divmod(args.task, 10)
    held, train, test = folds[fold]
    assert not set(d["subject"][train]) & set(d["subject"][test])
    out = WS / "runs/lane4"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{'smoke' if args.smoke else 'fold'}_s{seed}_{held}"
    assert not (out / (name + ".npz")).exists(), "output collision"
    started = time.perf_counter()
    m = spd.make(d, seed)
    m.fit(np.flatnonzero(train))
    spd.check_tsm(m, d, np.flatnonzero(test))
    m.set_k(20.0)
    for p in m.net.parameters():
        p.requires_grad_(False)
    with np.load(WS / "runs/shuffle_perms.npz") as f:
        perms = {f"{s}{i}": f[f"{s}|{i}|all_loso"].astype(int)
                 for s, i in ORDERS}
    first = perms["within0"]
    first = first[test[first]]
    one = first[d["stream"][first] == d["stream"][first[0]]]
    guards(m, one)
    if args.smoke:
        print("SMOKE OK", dict(held=held, seed=seed,
                               seconds=time.perf_counter() - started), flush=True)
        return
    result, metadata = {}, {}
    for order, global_rows in perms.items():
        assert np.array_equal(np.sort(global_rows), np.arange(len(d["y"])))
        rows = global_rows[test[global_rows]]
        for mode in ("frozen", "diag", "full"):
            probs = np.empty((len(rows), 3), dtype=np.float64)
            logs = []
            for sess in dict.fromkeys(d["stream"][rows]):
                local = np.flatnonzero(d["stream"][rows] == sess)
                s = features(m, rows[local])
                probs[local], stat = stream(m.net, s, mode, spd.logm, spd.sym_fn)
                logs.append(dict(session=str(sess), **stat))
            result[f"{order}|{mode}|index"] = rows
            result[f"{order}|{mode}|proba"] = probs
            metadata[f"{order}|{mode}"] = logs
            print("OUTPUT", held, seed, order, mode, len(rows), flush=True)
    metadata.update(held=held, seed=seed, fit_people=sorted(m.fit_subjects),
                    seconds=time.perf_counter() - started,
                    order_sha={k: hashlib.sha256(v.tobytes()).hexdigest()
                               for k, v in perms.items()})
    result["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(out / (name + ".npz"), **result)
    (out / (name + ".json")).write_text(json.dumps(metadata, indent=2))
    print("DONE", name, metadata["seconds"], flush=True)


if __name__ == "__main__":
    main()
