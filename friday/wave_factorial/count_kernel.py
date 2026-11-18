"""Replay the registered early-count and causal-kernel factorial on complete §430 traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


WORK = Path("/mnt/lustre/work/gehler/gfh527")
SCORER = WORK / "ni26_friday_wave_norm_20261118"
EARLY = WORK / "ni26_early455_v1_20261118"
STAGES = WORK / "ni26_match456_stages_20261118"
KERNEL = WORK / "ni26_memory454_20261118"
MATCH = WORK / "ni26_match456_v2_20261118"
PINS = {
    SCORER / "score.sha256": "2e3032e19ed3cbb361138599ebac577c69d870c00b4b0a62e303f27b56a578ba",
    EARLY / "source.sha256": "b22520ce11d59f8ba4d8e4780a42028e6b432be961628318f986ef1f69e9d5a3",
    STAGES / "source.sha256": "a66ce105e52714a85d5214307799e13dcd5b989a45f7acd96024c3a2d1f99f1b",
    KERNEL / "actor_source.sha256": "500f467d12477b46004c0a23aa03cd05ac4fdb0173aa9ca7566c5298f56c25ba",
    MATCH / "kernel_454_decisions.json": "ad7db8872cda9cbbc8784f95ecf9b5a38f6c2bff690d985449c0d8c7a62f42a4",
}
ARMS = ("stock", "early", "kernel", "combined", "combined_donor")
PAIRS = (("combined", "stock"), ("combined", "kernel"), ("combined", "early"),
         ("combined", "combined_donor"), ("early", "stock"), ("kernel", "stock"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(ok, why):
    if not ok:
        raise ValueError(why)


def verify_sources():
    for path, digest in PINS.items():
        check(sha(path) == digest, f"changed input {path}")
    for manifest in (SCORER / "score.sha256", EARLY / "source.sha256", STAGES / "source.sha256"):
        for line in manifest.read_text().splitlines():
            digest, file = line.split(maxsplit=1)
            path = manifest.parent / file
            check(sha(path) == digest, f"changed input source {path}")
    sys.path.insert(0, str(SCORER))


def early_probabilities(order, ages, stock, cap_package, early_module):
    path = STAGES / f"stages_{order}.npz"
    proof = json.loads(path.with_suffix(".json").read_text())
    anchor = STAGES / "stock" / f"{order}.npz"
    check(proof["order"] == order and proof["output"] == sha(path)
          and proof["experiment"] == PINS[STAGES / "source.sha256"]
          and proof["archived_stock"] == sha(anchor), f"changed early input receipt {order}")
    early_path = EARLY / f"{order}.npz"
    early_receipt = json.loads(early_path.with_suffix(".json").read_text())
    check(early_receipt["source"] == PINS[EARLY / "source.sha256"] and early_receipt["input"] == proof
          and early_receipt["output"] == sha(early_path), f"changed early endpoint receipt {order}")
    with np.load(path, allow_pickle=False) as arrays:
        raw, filtered, terminal = (arrays[k] for k in ("raw", "filtered", "terminal"))
    np.testing.assert_array_equal(terminal, stock)
    fixed, eligible = terminal.copy(), np.zeros(len(raw), dtype=bool)
    hypotheses = cap_package.J.hyp_cache(order, "wprior", cap_package.X, cap_package.LAM,
                                         sessions=cap_package.SESS, proto=cap_package.PROTO)
    latch = cap_package.latch(order)
    for session, (rows, lp, _, reset, cap) in hypotheses.items():
        np.testing.assert_array_equal(ages[rows], np.arange(len(rows)))
        p = raw[rows]
        _, corrected, urn = early_module.early(cap_package, p, lp, reset, cap)
        stop = len(rows) if latch[session] is None else latch[session]
        np.testing.assert_array_equal(p[:stop], filtered[rows[:stop]])
        np.testing.assert_allclose(urn[stop:], filtered[rows[stop:]], rtol=0, atol=1e-12)
        n = min(stop, 90)
        eligible[rows[:n]] = True
        np.testing.assert_array_equal(p[:n], terminal[rows[:n]])
        fixed[rows[:n]] = p[:n] + .20 * (corrected[:n] - p[:n])
        fixed[rows[0]] = terminal[rows[0]]
    np.testing.assert_array_equal(fixed[~eligible], terminal[~eligible])
    np.testing.assert_array_equal(fixed[ages == 0], terminal[ages == 0])
    with np.load(early_path, allow_pickle=False) as reference:
        np.testing.assert_array_equal(eligible, reference["eligible"])
        np.testing.assert_array_equal(terminal.argmax(1), reference["stock"])
        np.testing.assert_array_equal(fixed.argmax(1), reference["fixed"])
    return fixed, sha(early_path)


def score_bank(path, receipt, f9, meta, interval, metric):
    from score_metrics import COUNTS, paired, summarize

    check(sha(path) == receipt["decisions"], "blind bank changed")
    with np.load(path, allow_pickle=False) as data:
        bank = {name: data[name] for name in data}
    with np.load(f9.DATA, allow_pickle=False) as data:
        labels = data["y"]
    tables = {arm: np.empty((len(f9.PEOPLE), len(f9.ORDERS))) for arm in ARMS}
    for pi, person in enumerate(f9.PEOPLE):
        rows = meta["person"] == person
        for oi, order in enumerate(f9.ORDERS):
            for arm in ARMS:
                value = metric(labels[rows], bank[f"{person}|{order}|{arm}"], meta, rows)
                check(value["n_cells"] == COUNTS[pi], f"wrong official cells {person}")
                tables[arm][pi, oi] = 100 * value["balanced_accuracy_cell_mean"]
    result = summarize(tables, bank, PAIRS, interval, ())
    expected = (76.10391516713594, 76.1665, 76.321336)
    for arm, score in zip(ARMS, expected):
        check(abs(result["absolute"][arm]["pp"] - score) < 1e-4, f"changed original score {arm}")
    interaction = tables["combined"] - tables["early"] - tables["kernel"] + tables["stock"]
    result["contrasts"]["interaction"] = paired(interval, interaction)
    return result


def run(smoke):
    verify_sources()
    import friday_wave_score as f9

    owner, interval, metric = f9.upstream()
    meta, ages, _, stock = owner.inputs()
    early = owner.module(EARLY, "run", "5be6e071b70b130e36434dc0f1e8f90a2ad9ea4f3298eea4da72358a97171a1a")
    early.verify_sources()
    pkg = early.get_count_filter()
    check(pkg.J.TAU == 2.0 and pkg.J.FLOOR == 0.5, "wrong count model")
    kernel = owner.module(KERNEL, "corrections", "528f76f9d69d56e4be714bb0cf6d48bbff665b46dad1f7eeddcd0cb1307e1e07")
    path = MATCH / "kernel_454_decisions.npz"
    check(sha(path) == json.loads(path.with_suffix(".json").read_text())["bank"], "changed kernel reference bank")
    deltas, folds = {}, {}
    for person in f9.PEOPLE:
        rows = np.flatnonzero(meta["person"] == person)
        deltas[person], folds[person] = kernel.load(KERNEL, "kernel", 454, person, rows)
    bank, early_proof = {}, {}
    with np.load(path, allow_pickle=False) as archived:
        for i, order in enumerate(f9.ORDERS[:1] if smoke else f9.ORDERS):
            age = ages[order]
            fixed, early_proof[order] = early_probabilities(order, age, stock[order], pkg, early)
            base_log = owner.log_probability(stock[order], age)
            count_log = owner.log_probability(fixed, age)
            for person in f9.PEOPLE:
                rows = np.flatnonzero(meta["person"] == person)
                key = f"{person}|{order}"
                genuine, donor, zero = (deltas[person][arm][i] for arm in ("memory", "donor", "source"))
                first = age[rows] == 0
                check(not zero.any() and not genuine[first].any() and not donor[first].any(),
                      f"kernel first-window or zero changed {key}")
                paired = {
                    "stock": base_log[rows] + zero,
                    "early": count_log[rows] + zero,
                    "kernel": base_log[rows] + genuine,
                    "combined": count_log[rows] + genuine,
                    "combined_donor": count_log[rows] + donor,
                }
                np.testing.assert_array_equal(paired["stock"].argmax(1), archived[f"{key}|stock"])
                np.testing.assert_array_equal(paired["kernel"].argmax(1), archived[f"{key}|memory"])
                np.testing.assert_array_equal((base_log[rows] + donor).argmax(1), archived[f"{key}|donor"])
                np.testing.assert_array_equal(paired["early"].argmax(1), fixed[rows].argmax(1))
                for arm, value in paired.items():
                    decision = value.argmax(1).astype(np.uint8)
                    np.testing.assert_array_equal(decision[first], archived[f"{key}|stock"][first])
                    bank[f"{key}|{arm}"] = decision
    if smoke:
        print("PASS: within0, all ten people, exact early and kernel endpoints, ages, first rows, and zero")
        return
    path = Path("decisions.npz")
    np.savez_compressed(path, **bank)
    receipt = dict(decisions=sha(path), kernel=folds, early=early_proof, stock=owner.REPORT_SHA,
                   source=sha(Path("source.sha256")), replay=sha(Path("recording-replay.log")))
    Path("before_labels.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result = score_bank(path, receipt, f9, meta, interval, metric)
    Path("result.json").write_text(json.dumps({**receipt, **result}, indent=2) + "\n")
    print(json.dumps({arm: values["pp"] for arm, values in result["absolute"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
