"""Full0 Lane C check on the authenticated corrected Friday cap250 instrument."""
import argparse
import json
from pathlib import Path

import numpy as np

from oracle import metadata, metric_weights, require, sha


def replay(banks, public):
    # Import the original Friday producers from their pinned PYTHONPATH.
    import outer
    from worker import run_tail

    members = []
    for bank, digest in banks:
        require(sha(bank) == digest, "D1 ensemble bank hash changed")
        with np.load(bank, allow_pickle=False) as inputs:
            rows, delivery, one = (inputs[k] for k in ("index", "full0_delivery", "full0"))
            np.testing.assert_array_equal(rows, np.arange(2655))
            require(one.shape == (len(rows), 3) and np.isfinite(one).all() and (one >= 0).all(),
                    "Invalid D1 member predictions")
            np.testing.assert_allclose(one.sum(1), 1, atol=2e-6, rtol=0)
            if members:
                np.testing.assert_array_equal(delivery, first)
            first = delivery
            members.append(one)
    member = np.mean(members, axis=0)  # several banks share one 1/6 slot as their mean

    owner = outer.full_parent
    original, mass = owner.candidate, outer.MASS
    arms, traces = {}, {}

    def evaluate(ctx, archive, patches, incumbent):
        np.testing.assert_array_equal(ctx.Pperm, delivery)
        stock, counts = original(ctx, archive, patches, incumbent)
        active = np.ones(len(member), dtype=bool)
        arms["stock"] = stock
        for kind, post in (("zero", None), ("laneC", member)):
            arms[kind], refits, traces[kind] = outer.replace(
                original, ctx, archive, patches, incumbent, post, active)
            require(refits == counts, f"{kind} changed the canonical shadow refits")
        np.testing.assert_array_equal(arms["zero"], stock)
        require(traces["zero"]["slot_movement"] == 0 and traces["laneC"]["slot_movement"] > 0,
                "mix patch inactive")
        for stage in ("mix", "rebalance", "em", "tsem"):
            require(traces["zero"][stage] != traces["laneC"][stage], f"D1 edit did not reach {stage}")
        return stock, counts

    owner.candidate, outer.MASS = evaluate, 1 / 6
    try:
        with owner.full0() as ctx:
            np.testing.assert_array_equal(ctx["delivery"], delivery)
            frame = ctx["matched"]["stock"]
            np.testing.assert_array_equal(arms["stock"], frame["raw"])
            sessions = tuple(ctx["count"].SESS)
            require(len(sessions) == 14, "missing sessions")
            checks = (sessions[0], sessions[-1], sessions[0])
            for arm in arms:
                arms[arm], traces[arm + "_tail"] = run_tail(ctx, "full0", arms[arm], sessions, checks)
            for key, values in frame.items():
                if key in arms["stock"]:
                    np.testing.assert_array_equal(values, arms["stock"][key])
                    np.testing.assert_array_equal(arms["zero"][key], arms["stock"][key])
    finally:
        owner.candidate, outer.MASS = original, mass

    pred = {k: arm["terminal"].argmax(1) for k, arm in arms.items()}
    pred["standalone"] = member.argmax(1)
    d = metadata(public)
    return d, pred, traces


def paired(d, predictions):
    y = d["y"]
    stock, candidate = predictions["stock"] == y, predictions["laneC"] == y
    w, _ = metric_weights(d, np.ones(len(y), bool))
    people = {}
    for name in np.unique(d["subject"]):
        wp, cells = metric_weights(d, d["subject"] == name)
        people[name] = dict(delta_pp=float(100 * wp @ (candidate.astype(int) - stock)), cells=cells)
    rng = np.random.default_rng(20261119)
    samples = rng.integers(len(people), size=(10000, len(people)))
    by_person = np.array([v["delta_pp"] for v in people.values()])
    counts = np.array([v["cells"] for v in people.values()])
    change = np.average(by_person[samples], weights=counts[samples], axis=1)
    return dict(ba={a: float(100 * w @ (p == y)) for a, p in predictions.items()},
                delta_pp=float(w @ (candidate.astype(int) - stock) * 100),
                paired_people_80=np.quantile(change, [0.1, 0.9]).tolist(),
                fixed=int((~stock & candidate).sum()), regressed=int((stock & ~candidate).sum()), people=people)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("public", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--bank", action="append", required=True, help="path:sha256 of an ensemble bank (repeatable)")
    p.add_argument("--cap250", type=Path, help="Friday full0.npz; stock terminal argmax must equal ours")
    a = p.parse_args()
    banks = [(Path(b.rsplit(":", 1)[0]), b.rsplit(":", 1)[1]) for b in a.bank]
    d, predictions, traces = replay(banks, a.public)
    if a.cap250:
        with np.load(a.cap250, allow_pickle=False) as z:
            np.testing.assert_array_equal(predictions["stock"], z["stock|terminal"].argmax(1))
    result = paired(d, predictions) | {"trace": traces, "banks": a.bank, "order": "full0",
                                       "cap250_checked": bool(a.cap250)}
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
