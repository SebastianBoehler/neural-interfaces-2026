"""Score the original official cells only after authenticating all blind F14 banks."""
import argparse
import json

import numpy as np

from held import ARMS, HERE, KINDS, paired
from replay_f13 import DATA, DATA_SHA, ORDER_SHA, ORDERS, ORIGIN_SHA, PEOPLE, ROOT, origin_report, sha


CONTRASTS = {
    "bounded_training": ("bounded_bounded", "causal_bounded"),
    "whole_decoder": ("bounded_bounded", "original_ordinary"),
    "causal_training_ordinary_eval": ("causal_ordinary", "original_ordinary"),
    **{f"{kind}_bounded_eval": (f"{kind}_bounded", f"{kind}_ordinary") for kind in KINDS},
}


def blind_bank(array, subject, people, stream):
    bank = np.full((2, len(ARMS), len(ORDERS), len(subject)), -1, dtype=np.int8)
    root = HERE / "folds"
    if {p.name for p in root.iterdir()} != {f"{held}_s{seed}.npz" for held in PEOPLE for seed in (0, 1)}:
        raise ValueError("incomplete blind held bank")
    digests = {}
    for seed in (0, 1):
        for i, held in enumerate(PEOPLE):
            output = root / f"{held}_s{seed}.npz"
            log = (HERE / f"held-{array}_{seed * len(PEOPLE) + i}.out").read_text().splitlines()
            digest = sha(output)
            if f"{digest}  {output.name}" not in log or "PASS F14 independent held inference" not in log:
                raise ValueError(f"held worker did not authenticate {output.name}")
            idx = np.flatnonzero(people == held)
            expected = dict(held=held, seed=seed, orders=list(ORDERS), order_sha=ORDER_SHA, dataset=DATA_SHA,
                            origin=ORIGIN_SHA, scorer=sha(HERE / "score.sha256"), actor=sha(HERE / "actor.sha256"),
                            held_labels_opened=False)
            with np.load(output, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                if (not np.array_equal(z["index"], idx) or any(meta[k] != v for k, v in expected.items())
                        or json.dumps(meta, sort_keys=True) not in log):
                    raise ValueError(f"held score bank provenance differs for {output.name}")
                sessions = len(set(stream[idx]))
                for cp, kind_receipt in paired(held, seed, subject, people, stream, False):
                    kind = kind_receipt["kind"]
                    guards = meta["guards"][kind]
                    if ((guards["checkpoint"], guards["receipt"], guards["source_state"], guards["source_rows"]) !=
                            (sha(cp), sha(cp.with_suffix(".json")), kind_receipt["state"], kind_receipt["source_rows"])
                            or guards["first_windows_exact"] != sessions * len(ORDERS)):
                        raise ValueError(f"independent held guard differs for {kind}/{held}/{seed}")
                for a, arm in enumerate(ARMS):
                    values = z[arm]
                    if values.shape != (len(ORDERS), len(idx)) or not np.isin(values, (0, 1, 2)).all():
                        raise ValueError(f"invalid decision bank for {arm}/{held}/{seed}")
                    bank[seed, a][:, idx] = values
            digests[output.name] = digest
    if (bank < 0).any():
        raise ValueError("missing canonical held rows after blind merge")
    path = HERE / "before_labels.sha256"
    with path.open("x") as handle:
        for name, digest in sorted(digests.items()):
            handle.write(f"{digest}  folds/{name}\n")
    return bank, sha(path)


def collect(array):
    report = origin_report()
    with np.load(DATA, allow_pickle=False) as z:
        subject, session = z["subject"].astype(str), z["session"].astype(str)
    people = np.array([s.rsplit("/", 1)[-1] for s in subject])
    stream = np.char.add(np.char.add(subject, "|"), session)
    predictions, lock = blind_bank(array, subject, people, stream)

    import protocol
    import score_s_v66p

    if (protocol.__file__ != str(ROOT / "experiments/protocol.py") or
            score_s_v66p.__file__ != str(ROOT / "studies/14nov/newmember/score_s_v66p.py")):
        raise ValueError("historical official metric imports differ")
    d = protocol.load()
    y = d["y"]
    if not np.array_equal(subject, d["subject"]) or not np.array_equal(stream, d["stream"]):
        raise ValueError("metric rows differ from the blind inference rows")
    cell = np.char.add(np.char.add(stream, "|"), d["interface"])
    counts = np.array([len(np.unique(cell[people == p])) for p in PEOPLE])
    groups = [np.flatnonzero([o.startswith(s) for o in ORDERS]) for s in ("within", "full", "runs")]

    def interval(values):
        return score_s_v66p.cell_weighted_summary(values, ORDERS, counts)

    result = {}
    for seed in (0, 1):
        table = {}
        for a, arm in enumerate(ARMS):
            decisions = predictions[seed, a]
            values = np.stack([score_s_v66p.person_accuracy(np.eye(3)[p], y, cell, people, PEOPLE) for p in decisions])
            official = [100 * protocol.score(y, p, d, np.ones(len(y), bool))["balanced_accuracy_cell_mean"]
                        for p in decisions]
            np.testing.assert_allclose(np.average(values, weights=counts, axis=1), official, atol=1e-10, rtol=0)
            table[arm] = values
        pairs = {}
        for name, (left, right) in CONTRASTS.items():
            delta = table[left] - table[right]
            per_person = np.mean([delta[g].mean(0) for g in groups], 0)
            order = np.average(delta, weights=counts, axis=1)
            pairs[name] = dict(**interval(delta), people=dict(zip(PEOPLE, per_person.tolist())),
                               orders=dict(zip(ORDERS, order.tolist())))
        result[str(seed)] = dict(absolute={arm: interval(table[arm]) for arm in ARMS}, paired=pairs)
    outcome = {}
    for name in ("bounded_training", "whole_decoder"):
        a, b = (result[str(seed)]["paired"][name]["interval"] for seed in (0, 1))
        outcome[name] = "supports" if min(a[0], b[0]) > 0 else "refuted" if max(a[1], b[1]) <= 0 else "undecided"
    output = HERE / "whole_results.json"
    data = dict(scope="held-new WHOLE DEV", source_array=array, actor=sha(HERE / "actor.sha256"),
                scorer=sha(HERE / "score.sha256"), blind_receipt=lock, seeds=result, outcome=outcome,
                historical_s1c=report["official_equal_cell_scheme_equal"]["S1C"])
    with output.open("x") as handle:
        json.dump(data, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"PASS F14 held whole-decoder score {sha(output)}", flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("array", type=int)
    collect(args.parse_args().array)
