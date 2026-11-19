"""Seal F31 blind held decisions and report the original whole-decoder metric."""
import json
from pathlib import Path

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, sha
from held_order import ACTOR, F16, paired
from order_pretext import ARMS, HERE, KINDS, MODES
from replay_f13 import DATA, ORDER_SHA, ORDERS, ROOT, origin_report
from source_targets import TOL


OUTPUT = Path(__file__).resolve().parent
BLIND_V1 = "a40d5e40ec256384aad788d6926e56e4183a048c93995918e81c37e09aa05b22"
F16_RESULT = "e91c377bd9560d7c02f01adc9ccaa5e82b87d347927c19e17fd1d217780a803e"
COMPARISONS = (("ordered", "hard"), ("ordered", "sham"), ("sham", "hard"))


def lock_decisions(subject, visit, person, streams):
    values = np.full((2, len(ARMS), len(ORDERS), len(subject)), -1, dtype=np.int8)
    manifest = []
    clips = windows = resets = 0
    prefix = 0.
    for seed in (0, 1):
        for held in PEOPLE:
            path = HERE / "folds" / f"{held}_s{seed}.npz"
            digest = sha(path)
            if digest != path.with_suffix(".sha256").read_text().strip():
                raise ValueError("held decisions changed")
            manifest.append(f"{digest}  folds/{path.name}\n")
            rows = np.flatnonzero(person == held)
            n = len(set(streams[rows])) * len(ORDERS)
            required = dict(held=held, seed=seed, actor=ACTOR, scorer=sha(HERE / "score.sha256"), data=DATA_SHA,
                            origin=ORIGIN_SHA, orders=list(ORDERS), order_sha=ORDER_SHA, held_labels_opened=False)
            with np.load(path, allow_pickle=False) as z, np.load(F16 / "folds" / path.name, allow_pickle=False) as prior:
                receipt = json.loads(str(z["metadata"]))
                if (any(receipt[k] != v for k, v in required.items()) or not np.array_equal(z["index"], rows) or
                        set(z.files) != set(ARMS) | {"metadata", "index"} or set(receipt["evidence"]) != set(KINDS) or
                        not np.array_equal(prior["index"], rows) or any(
                            not np.array_equal(z[f"hard_{m}"], prior[f"hard_{m}"]) for m in MODES)):
                    raise ValueError("blind F31 rows, original hard control or provenance differs")
                bank = np.stack([z[a] for a in ARMS])
            if bank.shape != (len(ARMS), len(ORDERS), len(rows)) or not np.isin(bank, (0, 1, 2)).all():
                raise ValueError("invalid held decision bank")
            for kind, (cp, source) in paired(held, seed, False, subject, visit).items():
                proof = receipt["evidence"][kind]
                expected = dict(checkpoint=source["checkpoint"], state=source["state"], tape=source["tape"],
                                receipt=sha(cp.with_suffix(".json")), first_windows_exact=n, resets=2 * n)
                if any(proof[k] != v for k, v in expected.items()) or not 0 <= proof["prefix_max_abs"] <= TOL:
                    raise ValueError("held archive or prefix controls differ")
                clips += proof["clips"]
                windows += n
                resets += 2 * n
                prefix = max(prefix, proof["prefix_max_abs"])
            values[seed][:, :, rows] = bank
    if (values < 0).any() or not clips:
        raise ValueError("full held matrix or bounded work missing")
    lock = HERE / "before_labels.sha256"
    signed = "".join(sorted(manifest))
    if OUTPUT == HERE:
        with lock.open("x") as out:
            out.write(signed)
    elif sha(lock) != BLIND_V1 or lock.read_text() != signed:
        raise ValueError("reconstructed F31 decisions differ from the original blind seal")
    return values, sha(lock), dict(first=windows, resets=resets, active_clips=clips, prefix_max_abs=prefix)


def order_schemes():
    groups = [np.flatnonzero([order.startswith(s) for order in ORDERS])
              for s in ("within", "full", "runs")]
    if [len(g) for g in groups] != [4, 2, 2] or not np.array_equal(
            np.sort(np.concatenate(groups)), np.arange(len(ORDERS))):
        raise ValueError("pinned order groups are incomplete or overlap")
    return groups


def statistics(values, person, streams, dataset, by_scheme):
    import protocol
    import score_s_v66p

    if (protocol.__file__ != str(ROOT / "experiments/protocol.py") or
            score_s_v66p.__file__ != str(ROOT / "studies/14nov/newmember/score_s_v66p.py")):
        raise ValueError("whole-decoder metric ancestry differs")
    labels = dataset["y"]
    cells = np.char.add(np.char.add(streams, "|"), dataset["interface"])
    counts = np.asarray([len(np.unique(cells[person == p])) for p in PEOPLE])
    ref = F16 / "whole_results.json"
    if sha(ref) != F16_RESULT:
        raise ValueError("F16 hard control results differ")
    original = json.loads(ref.read_text())["seeds"]
    results, outcome = {}, {}
    for seed, bank in enumerate(values):
        accuracy, across, paired_result = {}, {}, {}
        for a, decision in zip(ARMS, bank, strict=True):
            across[a] = np.stack([score_s_v66p.person_accuracy(np.eye(3)[p], labels, cells, person, PEOPLE) for p in decision])
            official = np.asarray([100 * protocol.score(labels, p, dataset, np.ones(len(labels), bool))[
                "balanced_accuracy_cell_mean"] for p in decision])
            np.testing.assert_allclose(np.average(across[a], weights=counts, axis=1), official, atol=1e-10, rtol=0)
            accuracy[a] = float(np.mean([official[g].mean() for g in by_scheme]))
            if not np.isfinite(accuracy[a]):
                raise ValueError(f"nonfinite official summary for {a}")
        for mode in MODES:
            key = f"hard_{mode}"
            if abs(accuracy[key] - original[str(seed)]["absolute"][key]) > 1e-10:
                raise ValueError("F16 sealed numeric hard score changed")
            for a, b in COMPARISONS:
                delta = across[f"{a}_{mode}"] - across[f"{b}_{mode}"]
                individuals = np.mean([delta[g].mean(0) for g in by_scheme], 0)
                paired_result[f"{mode}_{a}_minus_{b}"] = dict(
                    **score_s_v66p.cell_weighted_summary(delta, ORDERS, counts),
                    people=dict(zip(PEOPLE, individuals.tolist())),
                    orders=dict(zip(ORDERS, np.average(delta, weights=counts, axis=1).tolist())))
        results[str(seed)] = dict(absolute=accuracy, paired=paired_result)
    for control in ("hard", "sham"):
        k = f"bounded_ordered_minus_{control}"
        a, b = (results[str(seed)]["paired"][k]["interval"] for seed in (0, 1))
        outcome[k] = "supported" if min(a[0], b[0]) > 0 else "refuted" if max(a[1], b[1]) <= 0 else "undecided"
    return results, outcome


def run():
    import protocol

    by_scheme = order_schemes()
    origin_report()
    with np.load(DATA, allow_pickle=False) as z:
        subject, visit = z["subject"], z["session"]
    person = np.asarray([s.rsplit("/", 1)[-1] for s in subject])
    streams = np.char.add(np.char.add(subject, "|"), visit)
    values, seal, guards = lock_decisions(subject, visit, person, streams)
    d = protocol.load()
    if not np.array_equal(d["subject"], subject) or not np.array_equal(d["stream"], streams):
        raise ValueError("gold rows differ from sealed held decisions")
    diagnostic = {}
    for seed in (0, 1):
        folds = {}
        for held in PEOPLE:
            checked = paired(held, seed, False, subject, visit)
            keys=("true_source_loss","source_order_accuracy","padding_error","invariance_drift",
                  "maximum_pretext_score","body_gradient","head_gradient","sham_flips","orientation_examples")
            folds[held]={k:{field:m[field] for field in keys} for k,(_,m) in checked.items() if k!="hard"}
        means={k:{f:float(np.mean([v[k][f] for v in folds.values()])) for f in keys} for k in ("ordered","sham")}
        diagnostic[str(seed)] = dict(folds=folds, means=means)
    scores, outcome = statistics(values, person, streams, d, by_scheme)
    decision_scorer = sha(HERE / "score.sha256")
    scorer = decision_scorer if OUTPUT == HERE else sha(OUTPUT / "recovery.sha256")
    result = dict(scope="held-new WHOLE DEV", actor=ACTOR, scorer=scorer, decision_scorer=decision_scorer,
                  before_labels=seal, f16=F16_RESULT, guards=guards, source_order=diagnostic, seeds=scores, outcome=outcome)
    rendered = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    path = OUTPUT / "whole_results.json"
    with path.open("x") as out:
        out.write(rendered + "\n")
    print(f"PASS F31 whole score {sha(path)}", flush=True)


if __name__ == "__main__":
    run()
