"""Verify delivery rows and collect paired people-then-orders scheme-audit results."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from protocol import ARMS, ORDERS
from tapes import joint, metadata, paired, read_bank, score


def plans(data, orders):
    # Reuse the official cell and paired scoring helpers. Add the acquisition
    # metadata to their input only for this delivery-scheme audit.
    from tapes import protocol

    streams = np.unique(data["stream"])
    return {order: protocol.plans(data, order, streams) for order in orders}


def collect(captures, data_path, blocks, *, detector=None, urn=None):
    data = metadata(data_path)
    with np.load(data_path, allow_pickle=False) as raw:
        data.update(run=raw["run"], onset=raw["onset"])
    tapes, results = {}, {}
    for block in blocks:
        orders = ORDERS[block]
        delivery = plans(data, orders)

        def read(fold, order, key):
            path = captures / f"{fold}-{order - order % 2}.npz"
            if path not in tapes:
                with np.load(path, allow_pickle=False) as tape:
                    tapes[path] = dict(tape)
            tape = tapes[path]
            if str(tape["fold"]) != fold:
                raise ValueError(f"wrong fold marked in {path}")
            rows = tape[f"rows_{key}"]
            stream = f"{fold.split('_')[1]}|{key.split('_')[1]}"
            np.testing.assert_array_equal(rows, delivery[order][stream], err_msg=f"{fold}/{key}")
            scores = {arm: tape[f"{arm}_{key}"] for arm in ARMS}
            if any(x.shape != rows.shape or x.dtype != np.int8 or not np.isin(x, (0, 1, 2)).all()
                   for x in scores.values()):
                raise ValueError(f"invalid label vector: {path}/{key}")
            terminal, latch, recognised = (tape[f"{field}_{key}"] for field in ("terminal", "latch", "recognised"))
            np.testing.assert_array_equal(terminal.argmax(1), scores["quarter"])
            if (terminal.shape != (len(rows), 3) or (terminal < 0).any() or
                    not np.allclose(terminal.sum(1), 1, atol=1e-7, rtol=0) or
                    not (int(latch) == -1 or 1 <= int(latch) <= len(rows)) or
                    recognised.shape != rows.shape or recognised.dtype != bool or
                    (recognised[:-1] & ~recognised[1:]).any()):
                raise ValueError(f"invalid posterior or latch receipt: {path}/{key}")
            direct = bool(tape.get("direct", False))
            for arm in ("em_half", "quarter"):
                changes = (scores[arm] != scores["ba_six_fp16"]) & ~recognised
                if not direct and changes.any():
                    raise ValueError(f"{arm} replay changed a pre-recognition label: {path}/{key}")
                scores[f"{arm}_unrecognised_change"] = changes
            if detector is not None and detector[fold][key] != int(latch):
                raise ValueError(f"live latch differs from the signed EEG-only detector: {path}/{key}")
            scores["latch"] = np.full(len(rows), int(latch))
            if urn is not None:
                scores["quarter_urn"] = urn(fold, rows, terminal, int(latch))
            return rows, scores

        cases = read_bank(data, orders, read, deliveries=delivery)
        latch = Counter(int(case.scores["latch"][0]) for case in cases)
        out = {
            "orders": orders,
            "latch": dict(streams=len(cases), latched=len(cases) - latch[-1],
                          by_window_60=sum(n for age, n in latch.items() if 0 < age <= 60),
                          first_windows=sorted(latch.items())),
            "arms": {},
            "unrecognised_label_differences": {
                arm: sum(int(np.count_nonzero(case.scores[f"{arm}_unrecognised_change"])) for case in cases)
                for arm in ("em_half", "quarter")
            },
        }
        for arm in ARMS:
            predict = lambda c, name=arm: c.scores[name]
            ref = lambda c: c.scores["hss"]
            out["arms"][arm] = dict(
                BA=dict(new=score(cases, predict, "new"), ret=score(cases, predict, "ret"), J=joint(cases, predict)),
                delta=paired(cases, predict, ref, orders, nested_orders=True),
                returning={person: score([c for c in cases if c.person == person], predict, "ret") -
                                   score([c for c in cases if c.person == person], ref, "ret")
                           for person in ("P04", "P10")},
            )
        out["quarter_vs_half"] = paired(cases, lambda c: c.scores["quarter"], lambda c: c.scores["em_half"],
                                        orders, nested_orders=True)
        if urn is not None:
            out["urn_vs_quarter"] = paired(cases, lambda c: c.scores["quarter_urn"],
                                            lambda c: c.scores["quarter"], orders, nested_orders=True)
            out["urn_changed_labels"] = sum(np.count_nonzero(c.scores["quarter_urn"] != c.scores["quarter"])
                                            for c in cases)
        results[block] = out

    def digest(path):
        with path.open("rb") as f:
            return file_digest(f, "sha256").hexdigest()

    devices = sorted({str(tape["gpu"]) for tape in tapes.values()})
    if len(devices) != 1:
        raise ValueError(f"mixed device results: {devices}")
    return dict(device=devices[0], shards={p.name: digest(p) for p in sorted(tapes)}, blocks=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "captures", "output"):
        p.add_argument(arg, type=Path)
    p.add_argument("--blocks", nargs="+", choices=tuple(ORDERS), default=tuple(ORDERS))
    p.add_argument("--detector", type=Path)
    a = p.parse_args()
    detector = json.loads(a.detector.read_text()) if a.detector else None
    results = collect(a.captures, a.data, a.blocks, detector=detector)
    with a.output.open("x") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"{results['device']}: {len(results['shards'])} verified tapes, {a.output}")


if __name__ == "__main__":
    main()
