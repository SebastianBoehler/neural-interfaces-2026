"""Measure balanced Team B subsets against stored hard terminal trace rows."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from terminal_collect import C, cell_weights
from fixture import visits
from fit import DATA_SHA, D
from patch_terminal import verify_parent
from worker import DEVICE, load
from team_b_speed_bench import check_features

ORDERS = tuple(range(71300, 71308))
HISTORICAL_SHA = "6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9"


def chosen_subsets():
    rng = np.random.default_rng(71625)
    result = {}
    for each in (1, 2):
        combos = list(product(*(list(combinations(range(start, start + 3), each)) for start in (0, 3, 6))))
        result[3 * each] = np.asarray([sum(combos[i], ()) for i in rng.choice(len(combos), 20, replace=False)])
    result[9] = np.arange(9)[None, :]
    return result


def fold_stream(enc, signals, rows, pooled):
    enc.reset()
    out = []
    taps = [None] * 9

    def tap(i):
        def keep(_layer, _args, logits):
            taps[i] = logits.log_softmax(1)[-1]
        return keep

    hooks = [net.head.register_forward_hook(tap(i)) for i, net in enumerate(enc.nets)]
    try:
        for r in rows:
            taps[:] = [None] * 9
            p = enc.posterior(signals[r].to(DEVICE)).cpu().numpy()
            current = torch.stack(taps)
            q = (sum(taps) / len(taps)).softmax(0).cpu().numpy()
            np.testing.assert_allclose(q, p, atol=1e-7, rtol=0)
            out.append(current.cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
    check_features(enc, len(rows))
    if len(enc.windows) != len(rows):
        raise ValueError("Team B missed a window")
    out = np.asarray(out)
    logits = out.mean(1)
    new = softmax(logits)
    np.testing.assert_allclose(new, pooled, atol=2e-6, rtol=0)
    return out


def softmax(logits):
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def capture(run, data, fold, smoke):
    signals = torch.from_numpy(data["X"])
    trace = run.parent / "fable-round3-21nov" / "trace"
    out = {}
    with TemporaryDirectory() as tmp:
        model, receipt = load(run, fold, data, Path(tmp) / "fold", (), observe=False)
        for first in (ORDERS[:1] if smoke else ORDERS[::2]):
            with np.load(trace / f"{fold}_{first}.npz", allow_pickle=False) as tape:
                if (str(tape["weights_sha"]) != receipt["team_b_audit"]["weights_sha"] or
                        str(tape["data_sha"]) != DATA_SHA or str(tape["lead_sha"]) != HISTORICAL_SHA):
                    raise ValueError("size curve reference provenance differs")
                for o in (first,) if smoke else (first, first + 1):
                    for s, rows in D.plans(data, o, visits(data, fold)).items():
                        key = f"{o}_{s.split('|')[-1]}"
                        np.testing.assert_array_equal(tape[f"rows_{key}"], rows)
                        out[f"members_{key}"] = fold_stream(model.team_b, signals, rows, tape[f"team_{key}"])
                        out[f"rows_{key}"] = rows
                        out[f"hard_{key}"] = tape[f"hard_log_{key}"]
                        out[f"label_{key}"] = tape[f"hard_{key}"]
                        np.testing.assert_array_equal(out[f"hard_{key}"].argmax(1), out[f"label_{key}"])
                        print(f"PASS {fold} {o} {s} per-member and pooled trace parity", flush=True)
    if not smoke:
        path = run / "size"
        path.mkdir(exist_ok=True)
        with (path / f"{fold}.npz").open("xb") as f:
            np.savez_compressed(f, **out)


def report(run, data):
    subs = chosen_subsets()
    totals = {mode: {n: np.zeros(len(s)) for n, s in subs.items()} for mode in ("new", "ret")}
    counts = {"new": 0.0, "ret": 0.0}
    for mode, person in C.FOLDS:
        fold = f"{mode}_{person}"
        with np.load(run / "size" / f"{fold}.npz", allow_pickle=False) as tape:
            for o in ORDERS:
                for s, rows in D.plans(data, o, visits(data, fold)).items():
                    key = f"{o}_{s.split('|')[-1]}"
                    np.testing.assert_array_equal(tape[f"rows_{key}"], rows)
                    raw = tape[f"members_{key}"]
                    hard, label = softmax(tape[f"hard_{key}"]), tape[f"label_{key}"]
                    y, weights = data["y"][rows], cell_weights(data, rows)
                    counts[mode] += weights.sum()
                    for n, subsets in subs.items():
                        q = softmax(raw[:, subsets].mean(axis=2))
                        pred = (0.75 * hard[:, None, :] + 0.25 * q).argmax(-1)
                        gain = (pred == y[:, None]).astype(float) - (label == y)[:, None]
                        totals[mode][n] += (weights[:, None] * gain).sum(0)
    gain = {m: {n: v * 100 / counts[m] for n, v in x.items()} for m, x in totals.items()}
    j = {n: 0.5 * (gain["new"][n].mean() + gain["ret"][n].mean()) for n in subs}
    result = dict(orders=ORDERS, subsets={n: s.tolist() for n, s in subs.items()},
                  terminal_proxy_pp={m: {n: v.tolist() for n, v in x.items()} for m, x in gain.items()},
                  mean_j=j, nine_minus_six_j=j[9] - j[6], decision_train=bool(j[9] - j[6] >= 0.10))
    with (run / "size.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("mean_j", "nine_minus_six_j", "decision_train")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["smoke", "capture", "report"])
    p.add_argument("run", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("fold", nargs="?")
    a = p.parse_args()
    verify_parent(a.data, DATA_SHA)
    data = D.load(a.data)
    if a.mode == "report":
        report(a.run, data)
    else:
        capture(a.run, data, a.fold, a.mode == "smoke")


if __name__ == "__main__":
    main()
