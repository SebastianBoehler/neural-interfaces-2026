"""Capture label-blind stage scores from the pinned fused fold inference."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / name) for name in ("last_push", "leonardo_fill", "laneTB")]
from dose import DATA_SHA, V, sessions
from grow_fixture import open_fold
from grow_replay import check, load
from replay import pin
from tape import replay, tap

ORDERS = (71800, 71801, 71802, 71803, 71808, 71809, 71810, 71811)
DEVICE = V.DEVICE


def hook(owner, method, observe):
    original = getattr(owner, method)

    def call(*args, **kwargs):
        value = original(*args, **kwargs)
        observe(args, value)
        return value

    setattr(owner, method, call)


def top_gate(contrast, identity):
    if contrast is not None and (identity is None or contrast[2] >= identity[2]):
        return contrast
    return identity


def normalized(q, log_scores):
    if log_scores:
        q = np.exp(q - q.max())
    if not np.isfinite(q).all() or (q < 0).any() or q.sum() <= 0:
        raise ValueError("invalid recorded posterior")
    return q / q.sum()


def capture(template, signals, rows):
    storage = defaultdict(list)
    fresh_model = V.D.official()[1].fresh_model

    def save(key, value):
        storage[key].append(np.asarray(value).copy())

    def fresh(model):
        m = fresh_model(model)
        tail = m.routeB.tail
        hook(tail.rb, "step", lambda a, q: (save("routed", a[0]), save("rebalance", q)))
        hook(tail.em, "step", lambda _a, q: save("session_em", q))
        hook(tail.tsem, "step", lambda _a, q: save("tsem", q))
        hook(m.cf, "step", lambda _a, q: save("count", q))
        hook(m, "hard_probability", lambda _a, q: save("hard_actors", q))
        hook(m, "_tangents", lambda _a, q: save("core2", q[2]))
        original_cp = m.cp.step

        def cp(*args):
            before = m.cp.fused
            q = original_cp(*args)
            fused = m.cp.fused != before
            save("csp", normalized(q, fused))
            save("csp_active", fused)
            return q

        m.cp.step = cp
        heads = [None] * len(m.team_b.nets)

        def head(index):
            def observe(_args, logits):
                heads[index] = logits.log_softmax(1)[-1]
            return observe

        for i, net in enumerate(m.team_b.nets):
            hook(net.head, "forward", head(i))

        def keep_heads(_args, pooled):
            logits = torch.stack(heads)
            torch.testing.assert_close((sum(heads) / len(heads)).softmax(0), pooled, rtol=0, atol=1e-7)
            save("team_members", logits.exp().cpu().numpy())
            heads[:] = [None] * len(heads)

        hook(m.team_b, "posterior", keep_heads)

        def keep_frame(_args, _label):
            v6 = m.trace["v6"]
            save("raw", m.trace["mix"])
            save("blend", v6["mix6"])
            save("rest", v6["rest"])
            save("core1", m.trace["core1"])
            if storage["net_ids"] and not np.array_equal(storage["net_ids"][0], sorted(v6["members"])):
                raise ValueError("raw member set changed")
            save("net_ids", sorted(v6["members"]))
            save("net_members", [v6["members"][i] for i in sorted(v6["members"])])

        hook(m, "predict", keep_frame)
        return m

    items = tap(template, signals, rows, fresh, check, DEVICE)
    for item in items:
        gate = top_gate(item.contrast, item.identity)
        save("hard", item.base)
        save("team", item.team_b)
        save("own", item.route_out)
        save("recognised", item.recognised)
        save("gate_id", gate[0] if gate is not None and gate[2] > 0 else -1)
        save("gate_ramp", gate[2] if gate is not None else 0.0)
        if item.team_a is not None:
            save("manas", item.team_a)
        save("label", item.label)

    out = {key: np.stack(values) for key, values in storage.items()}
    if any(len(values) != len(rows) or not np.isfinite(values).all() for values in out.values()):
        raise ValueError("missing or non-finite stage capture")
    csp_active = out["csp_active"]
    out["terminal"] = np.stack([normalized(item.terminal, "manas" not in out and bool(on))
                                for item, on in zip(items, csp_active, strict=True)])
    np.testing.assert_array_equal(out["terminal"].argmax(1), out["label"])
    if "manas" in out:
        np.testing.assert_allclose(out["terminal"], 0.9 * out["csp"] + 0.1 * out["manas"], atol=1e-7, rtol=0)
    else:
        np.testing.assert_array_equal(out["terminal"], out["csp"])
    for item, mixed in zip(items, out["blend"], strict=True):
        np.testing.assert_array_equal(mixed, 0.75 * item.base + 0.25 * item.team_b)
    return out, items


def recognition_oracle(template, signals, rows, native, person_index):
    if person_index is None and not native["recognised"].any():
        return native["label"]
    m = V.D.official()[1].fresh_model(template)
    gate = None if person_index is None else (person_index, 1.0, 1.0)
    m.route.contrast.step = lambda *_: gate
    m.route.identity.step = lambda *_: None
    labels = []
    for row in rows:
        label = m.predict(signals[row:row + 1].to(DEVICE))
        if m.dead or m.log6["err"] or label.dtype != torch.int64 or label.shape != (1,):
            raise ValueError("invalid forced-recognition solver path")
        labels.append(int(label[0]))
    check(m, len(rows))
    if m.log6["rec"] != (0 if gate is not None else -1):
        raise ValueError(f"recognition intervention did not take effect: {m.log6}")
    return np.array(labels, dtype=np.int8)


def fixture_assets(folder, receipt):
    for name, sha in receipt["assets"].items():
        if name != "submission.py" and V.digest(folder / name) != sha:
            raise ValueError(f"fold asset differs: {folder / name}")
    if receipt["partner"] and V.digest(folder / "team_a.pt") != receipt["partner"]["weights_sha"]:
        raise ValueError("MANAS asset differs")
    with np.load(folder / "weights.npz", allow_pickle=False) as tape:
        return tape["route_names"].tolist()


def process(args):
    if not torch.cuda.is_available() or V.digest(args.data) != DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS export required")
    fold, roster = args.fold, args.roster
    data = V.D.load(args.data)
    signals = torch.from_numpy(data["X"])
    pkg, receipts = open_fold(args.fixtures, fold, "ba-confirm")
    receipt = receipts[roster]
    people = fixture_assets(pkg / roster, receipt)
    person_index = people.index(fold.split("_")[1]) if fold.startswith("ret_") else None
    template, _ = load(data, pkg, roster, receipt)
    streams = sessions(data, fold)
    out = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True)),
           "route_names": np.asarray(people), "gpu": np.asarray(torch.cuda.get_device_name())}
    for order in args.orders:
        for stream, rows in V.D.plans(data, order, streams).items():
            stages, items = capture(template, signals, rows)
            pin(args.fixtures, fold, roster, order, stream, rows, stages["label"], receipt)
            labels, _ = replay(template, items, "w25")
            np.testing.assert_array_equal(labels, stages["label"])
            if args.smoke:
                prefix, _ = capture(template, signals, rows[:32])
                for key in stages:
                    if np.issubdtype(stages[key].dtype, np.floating):
                        np.testing.assert_allclose(prefix[key], stages[key][:32], rtol=0, atol=2e-5, err_msg=key)
                    else:
                        np.testing.assert_array_equal(prefix[key], stages[key][:32], err_msg=key)
            stages["oracle_recognition"] = recognition_oracle(template, signals, rows, stages, person_index)
            key = f"{order}_{stream.split('|')[1]}"
            out[f"rows_{key}"] = rows
            out.update({f"{name}_{key}": array for name, array in stages.items()})
            print(f"PASS {fold} {roster} {key}: {len(rows)} pinned labels, exact zero replay, active capture", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out) as temp:
        filename = f"{roster}-{fold}-{args.orders[0]}.npz"
        partial = Path(temp) / filename
        np.savez_compressed(partial, **out)
        partial.rename(args.out / filename)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("fixtures", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("roster", choices=("nine25", "ba_six"))
    p.add_argument("fold")
    p.add_argument("orders", type=int, nargs="+", choices=ORDERS)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    process(args)


if __name__ == "__main__":
    main()
