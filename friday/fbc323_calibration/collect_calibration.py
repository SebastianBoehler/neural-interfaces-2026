"""Seal F24 blind held decisions and report the original whole-decoder metric."""
import json

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, sha
from calibration import ARMS, HERE, KINDS, MODES
from calibration_source import parent_roots
from held_calibration import ACTOR, compare_archives, paired
from replay_f13 import DATA, ORDER_SHA, ORDERS, ROOT, origin_report
from source_targets import TOL


PARENT_RESULTS = dict(hard="e91c377bd9560d7c02f01adc9ccaa5e82b87d347927c19e17fd1d217780a803e",
                      jitter="9c2ef83eb850d671950be72024393c43c2b2e9952303d5f5bf0718b618e62916")
COMPARISONS = (("cross", "jitter"), ("cross", "within"), ("cross", "hard"), ("within", "jitter"))


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
            with np.load(path, allow_pickle=False) as z:
                receipt = json.loads(str(z["metadata"]))
                if (any(receipt[k] != v for k, v in required.items()) or not np.array_equal(z["index"], rows) or
                        set(z.files) != set(ARMS) | {"metadata", "index"} or set(receipt["evidence"]) != set(KINDS)):
                    raise ValueError("blind F24 rows, original hard control or provenance differs")
                compare_archives(path.name, rows, z)
                bank = np.stack([z[a] for a in ARMS])
            if bank.shape != (len(ARMS), len(ORDERS), len(rows)) or not np.isin(bank, (0, 1, 2)).all():
                raise ValueError("invalid held decision bank")
            for kind, (cp, source) in paired(held, seed, False, subject, visit).items():
                proof = receipt["evidence"][kind]
                expected = dict(checkpoint=source["checkpoint"], state=source["state"], context_sha=source["context_sha"],
                                jitter_sha=source["jitter_sha"],
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
    with lock.open("x") as out:
        out.writelines(sorted(manifest))
    return values, sha(lock), dict(first=windows, resets=resets, active_clips=clips, prefix_max_abs=prefix)


def statistics(values, person, streams, dataset):
    import protocol
    import score_s_v66p

    if (protocol.__file__ != str(ROOT / "experiments/protocol.py") or
            score_s_v66p.__file__ != str(ROOT / "studies/14nov/newmember/score_s_v66p.py")):
        raise ValueError("whole-decoder metric ancestry differs")
    labels = dataset["y"]
    cells = np.char.add(np.char.add(streams, "|"), dataset["interface"])
    counts = np.asarray([len(np.unique(cells[person == p])) for p in PEOPLE])
    by_scheme = [np.flatnonzero([o.startswith(s) for o in ORDERS]) for s in ("within", "full", "runs")]
    originals = {}
    for k, root in parent_roots(True).items():
        path = root / "whole_results.json"
        if sha(path) != PARENT_RESULTS[k]:
            raise ValueError(f"final exact {k} result parent differs")
        originals[k] = json.loads(path.read_text())["seeds"]
    results, outcome = {}, {}
    for seed, bank in enumerate(values):
        accuracy, across, paired_result = {}, {}, {}
        for a, decision in zip(ARMS, bank, strict=True):
            across[a] = np.stack([score_s_v66p.person_accuracy(np.eye(3)[p], labels, cells, person, PEOPLE) for p in decision])
            official = np.asarray([100 * protocol.score(labels, p, dataset, np.ones(len(labels), bool))[
                "balanced_accuracy_cell_mean"] for p in decision])
            np.testing.assert_allclose(np.average(across[a], weights=counts, axis=1), official, atol=1e-10, rtol=0)
            accuracy[a] = float(np.mean([official[g].mean() for g in by_scheme]))
        for mode in MODES:
            for kind, original in originals.items():
                key = f"{kind}_{mode}"
                if abs(accuracy[key] - original[str(seed)]["absolute"][key]) > 1e-10:
                    raise ValueError(f"sealed numeric {kind} score changed")
            for a, b in COMPARISONS:
                delta = across[f"{a}_{mode}"] - across[f"{b}_{mode}"]
                individuals = np.mean([delta[g].mean(0) for g in by_scheme], 0)
                paired_result[f"{mode}_{a}_minus_{b}"] = dict(
                    **score_s_v66p.cell_weighted_summary(delta, ORDERS, counts),
                    people=dict(zip(PEOPLE, individuals.tolist())),
                    orders=dict(zip(ORDERS, np.average(delta, weights=counts, axis=1).tolist())))
        results[str(seed)] = dict(absolute=accuracy, paired=paired_result)
    for control in ("jitter", "within"):
        label = f"bounded_cross_minus_{control}"
        a, b = [results[str(seed)]["paired"][label]["interval"] for seed in (0, 1)]
        if min(a[0], b[0]) > 0:
            outcome[label] = "supported"
        elif max(a[1], b[1]) <= 0:
            outcome[label] = "refuted"
        else:
            outcome[label] = "undecided"
    return results, outcome


def run():
    import protocol

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
        keys = ("cross_js", "within_js", "between_js", "aux_steps", "aux_queries")
        for held in PEOPLE:
            checked = paired(held, seed, False, subject, visit)
            folds[held] = {k: {field: meta[field] for field in keys} for k, (_, meta) in checked.items()}
        means = {k: {field: float(np.mean([f[k][field] for f in folds.values()])) for field in keys} for k in KINDS}
        diagnostic[str(seed)] = dict(folds=folds, means=means,
                                     cross_vs_jitter=means["cross"]["cross_js"] / means["jitter"]["cross_js"] - 1,
                                     cross_vs_within=means["cross"]["cross_js"] / means["within"]["cross_js"] - 1)
    scores, outcome = statistics(values, person, streams, d)
    path = HERE / "whole_results.json"
    with path.open("x") as out:
        json.dump(dict(scope="held-new WHOLE DEV", actor=ACTOR, scorer=sha(HERE / "score.sha256"),
                       before_labels=seal, parents=PARENT_RESULTS, guards=guards, source_context=diagnostic, seeds=scores, outcome=outcome),
                  out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(f"PASS F24 whole score {sha(path)}", flush=True)


if __name__ == "__main__":
    run()
