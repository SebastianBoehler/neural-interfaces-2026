"""Seal matched F15 decisions, then reduce to the original §323 whole-model metric."""
import argparse
import json

import numpy as np

from evaluate import ARMS, F14, HERE, authenticate
from replay_f13 import DATA, DATA_SHA, ORDER_SHA, ORDERS, ORIGIN_SHA, PEOPLE, ROOT, origin_report, sha


COMPARISONS = {
    "dropout_vs_scaled_on": ("dropout_bounded_on", "scaled_bounded_on"),
    "dropout_vs_zero_zero": ("dropout_bounded_zero", "zero_bounded_zero"),
    "dropout_vs_original_on": ("dropout_bounded_on", "original_bounded_on"),
}
F14_RESULT = "82fe550621a8eb72a0762c507fb81ee1a9d4d3cc65983a3e8bbdf2d923aed420"


def frozen_decisions(array, subject, person, stream):
    directory = HERE / "folds"
    if {p.name for p in directory.iterdir()} != {f"{h}_s{s}.npz" for h in PEOPLE for s in (0, 1)}:
        raise ValueError("the full blind matrix is required before scoring")
    decisions = np.full((2, len(ARMS), len(ORDERS), len(subject)), -1, dtype=np.int8)
    manifest = []
    for seed in (0, 1):
        for i, held in enumerate(PEOPLE):
            output = directory / f"{held}_s{seed}.npz"
            line = f"{sha(output)}  {output.name}"
            log = (HERE / f"held-{array}_{seed * len(PEOPLE) + i}.out").read_text().splitlines()
            if line not in log or "PASS F15 independent held inference" not in log:
                raise ValueError(f"blind worker did not authenticate {output.name}")
            idx = np.flatnonzero(person == held)
            expected = dict(held=held, seed=seed, dataset=DATA_SHA, origin=ORIGIN_SHA, order_sha=ORDER_SHA,
                            orders=list(ORDERS), held_labels_opened=False, scorer=sha(HERE / "score.sha256"),
                            actor=sha(HERE / "actor.sha256"))
            with np.load(output, allow_pickle=False) as z:
                receipt = json.loads(str(z["metadata"]))
                if (any(receipt[k] != v for k, v in expected.items()) or not np.array_equal(z["index"], idx)
                        or set(z.files) != {"metadata", "index", *ARMS} or json.dumps(receipt, sort_keys=True) not in log):
                    raise ValueError("blind held receipt, rows or valid endpoints differ")
                sessions = len(set(stream[idx]))
                for p, meta in authenticate(held, seed, subject, person, stream, False):
                    evidence = receipt["evidence"][meta["kind"]]
                    if ((evidence["checkpoint"], evidence["receipt"], evidence["state"], evidence["first_windows_exact"]) !=
                            (sha(p), sha(p.with_suffix(".json")), meta["state"], sessions * len(ORDERS))):
                        raise ValueError("blind held scorer consumed a different source checkpoint")
                for a, arm in enumerate(ARMS):
                    values = z[arm]
                    if values.shape != (len(ORDERS), len(idx)) or not np.isin(values, (0, 1, 2)).all():
                        raise ValueError(f"invalid held decisions for {held}/{seed}/{arm}")
                    decisions[seed, a][:, idx] = values
            manifest.append(f"{sha(output)}  folds/{output.name}\n")
    if (decisions < 0).any():
        raise ValueError("canonical held merge lacks predictions")
    lock = HERE / "before_labels.sha256"
    with lock.open("x") as handle:
        handle.writelines(sorted(manifest))
    return decisions, sha(lock)


def collect(array):
    origin_report()
    with np.load(DATA, allow_pickle=False) as z:
        subject, session = z["subject"].astype(str), z["session"].astype(str)
    person = np.array([value.rsplit("/", 1)[-1] for value in subject])
    stream = np.char.add(np.char.add(subject, "|"), session)
    decisions, lock = frozen_decisions(array, subject, person, stream)

    import protocol
    import score_s_v66p

    if (protocol.__file__ != str(ROOT / "experiments/protocol.py") or
            score_s_v66p.__file__ != str(ROOT / "studies/14nov/newmember/score_s_v66p.py")):
        raise ValueError("original §323 metric imports differ")
    d = protocol.load()
    if not np.array_equal(d["subject"], subject) or not np.array_equal(d["stream"], stream):
        raise ValueError("metric rows do not match the blind decisions")
    y = d["y"]
    cells = np.char.add(np.char.add(stream, "|"), d["interface"])
    counts = np.array([len(np.unique(cells[person == h])) for h in PEOPLE])
    groups = [np.flatnonzero([order.startswith(s) for order in ORDERS]) for s in ("within", "full", "runs")]
    f14 = F14 / "whole_results.json"
    if sha(f14) != F14_RESULT:
        raise ValueError("original F14 score digest differs")
    f14 = json.loads(f14.read_text())
    score = {}
    for seed in (0, 1):
        raw, absolute = {}, {}
        for a, arm in enumerate(ARMS):
            raw[arm] = np.stack([score_s_v66p.person_accuracy(np.eye(3)[p], y, cells, person, PEOPLE)
                                 for p in decisions[seed, a]])
            official = [100 * protocol.score(y, p, d, np.ones(len(y), bool))["balanced_accuracy_cell_mean"]
                        for p in decisions[seed, a]]
            np.testing.assert_allclose(np.average(raw[arm], weights=counts, axis=1), official, atol=1e-10, rtol=0)
            absolute[arm] = float(np.mean([np.mean(np.asarray(official)[g]) for g in groups]))
        for normalizer in ("bounded", "ordinary"):
            stored = f14["seeds"][str(seed)]["absolute"][f"original_{normalizer}"]["mean"]
            if abs(absolute[f"original_{normalizer}_on"] - stored) > 1e-10:
                raise ValueError("authentic q-on control score differs from F14")
        paired = {}
        for name, (a, b) in COMPARISONS.items():
            delta = raw[a] - raw[b]
            who = np.mean([delta[g].mean(0) for g in groups], 0)
            paired[name] = dict(**score_s_v66p.cell_weighted_summary(delta, ORDERS, counts),
                                people=dict(zip(PEOPLE, who.tolist())),
                                orders=dict(zip(ORDERS, np.average(delta, weights=counts, axis=1).tolist())))
        score[str(seed)] = dict(absolute=absolute, paired=paired)
    decisions = {}
    for name in ("dropout_vs_scaled_on", "dropout_vs_zero_zero"):
        a, b = (score[str(seed)]["paired"][name]["interval"] for seed in (0, 1))
        decisions[name] = "supports" if min(a[0], b[0]) > 0 else "refuted" if max(a[1], b[1]) <= 0 else "undecided"
    output = HERE / "whole_results.json"
    with output.open("x") as handle:
        json.dump(dict(scope="held-new WHOLE DEV", seeds=score, outcome=decisions, blind_receipt=lock,
                       actor=sha(HERE / "actor.sha256"), scorer=sha(HERE / "score.sha256"), f14=F14_RESULT),
                  handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"PASS F15 held whole-decoder score {sha(output)}", flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("array", type=int)
    collect(args.parse_args().array)
