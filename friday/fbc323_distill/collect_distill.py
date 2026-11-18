"""Seal all raw students before scoring their paired held-person decisions."""
import json

import numpy as np

from anchors import ARMS, DATA_SHA, HERE, KINDS, MODES, ORIGIN_SHA, PEOPLE, sha, split
from held_receipts import ACTOR_SHA, F15, source_receipts
from replay_f13 import DATA, ORDER_SHA, ORDERS, ROOT, origin_report
from source_targets import TOL


F15_RESULT = "eb06cd1fa8c52540706045850a4a76fa74113f7febe668e5229dc1a12984842e"
PAIRS = (("trial", "hard"), ("trial", "deranged"), ("deranged", "hard"))
PRIMARY = ("bounded_trial_minus_hard", "bounded_trial_minus_deranged")


def lock_decisions(subject, visit, person, stream):
    directory = HERE / "folds"
    decisions = np.full((2, len(ARMS), len(ORDERS), len(subject)), -1, dtype=np.int8)
    manifest = []
    first = resets = clips = 0
    max_prefix = 0.
    for seed in (0, 1):
        for held in PEOPLE:
            path = directory / f"{held}_s{seed}.npz"
            digest = sha(path)
            if digest != path.with_suffix(".sha256").read_text().strip():
                raise ValueError("blind held predictions changed")
            test = np.flatnonzero(person == held)
            n = len(ORDERS) * len(set(stream[test]))
            fixed = dict(held=held, seed=seed, actor=ACTOR_SHA, scorer=sha(HERE / "score.sha256"), data=DATA_SHA,
                         orders=list(ORDERS), order_sha=ORDER_SHA, origin=ORIGIN_SHA, held_labels_opened=False)
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                if (any(meta[k] != v for k, v in fixed.items()) or not np.array_equal(z["index"], test) or
                        set(z.files) != {"metadata", "index", *ARMS} or set(meta["evidence"]) != set(KINDS)):
                    raise ValueError("incomplete or unpaired raw held provenance")
                bank = np.stack([z[a] for a in ARMS])
            if bank.shape != (len(ARMS), len(ORDERS), len(test)) or not np.isin(bank, (0, 1, 2)).all():
                raise ValueError("invalid held decisions")
            rows, src, roster = split(subject, visit, held)
            for kind, (p, source) in source_receipts(held, seed, False, rows, src, roster).items():
                proof = meta["evidence"][kind]
                expected = dict(first_windows_exact=n, resets=2 * n, checkpoint=source["checkpoint"],
                                receipt=sha(p.with_suffix(".json")), state=source["state"],
                                teacher_bank=source["teacher_bank"], donor_plan=source["donor_plan"])
                if (any(proof[k] != v for k, v in expected.items()) or not 0 <= proof["prefix_max_abs"] <= TOL
                        or proof["clips"] < 0):
                    raise ValueError("held checkpoint, first-window or reset proof differs")
                first += n
                resets += 2 * n
                clips += proof["clips"]
                max_prefix = max(max_prefix, proof["prefix_max_abs"])
            decisions[seed][:, :, test] = bank
            manifest.append(f"{digest}  folds/{path.name}\n")
    if (decisions < 0).any() or clips == 0:
        raise ValueError("incomplete or inactive held replay")
    lock = HERE / "before_labels.sha256"
    with lock.open("x") as handle:
        handle.writelines(sorted(manifest))
    return decisions, sha(lock), dict(first=first, resets=resets, active_clips=clips, prefix_max_abs=max_prefix)


def score_whole(decisions, dataset, person, stream):
    import protocol
    import score_s_v66p

    if (protocol.__file__ != str(ROOT / "experiments/protocol.py") or
            score_s_v66p.__file__ != str(ROOT / "studies/14nov/newmember/score_s_v66p.py")):
        raise ValueError("original whole-model metric imports differ")
    y = dataset["y"]
    cells = np.char.add(np.char.add(stream, "|"), dataset["interface"])
    counts = np.asarray([len(np.unique(cells[person == h])) for h in PEOPLE])
    groups = [np.flatnonzero([order.startswith(s) for order in ORDERS]) for s in ("within", "full", "runs")]
    fixed = F15 / "whole_results.json"
    if sha(fixed) != F15_RESULT:
        raise ValueError("F15 teacher and hard reference result differs")
    archived = json.loads(fixed.read_text())["seeds"]
    result = {}
    for seed in (0, 1):
        raw, absolute, paired = {}, {}, {}
        for arm, bank in zip(ARMS, decisions[seed], strict=True):
            raw[arm] = np.stack([score_s_v66p.person_accuracy(np.eye(3)[p], y, cells, person, PEOPLE) for p in bank])
            official = [100 * protocol.score(y, p, dataset, np.ones(len(y), dtype=bool))["balanced_accuracy_cell_mean"]
                        for p in bank]
            np.testing.assert_allclose(np.average(raw[arm], weights=counts, axis=1), official, atol=1e-10, rtol=0)
            absolute[arm] = float(np.mean([np.mean(np.asarray(official)[g]) for g in groups]))
        reference = {}
        for mode in MODES:
            base = archived[str(seed)]["absolute"]
            if abs(absolute[f"hard_{mode}"] - base[f"zero_{mode}_zero"]) > 1e-10:
                raise ValueError("raw hard CE did not reproduce sealed F15 zero")
            reference[mode] = dict(full_q_teacher=base[f"dropout_{mode}_on"],
                                   trial_minus_full_q_teacher=absolute[f"trial_{mode}"] - base[f"dropout_{mode}_on"])
            for a, b in PAIRS:
                delta = raw[f"{a}_{mode}"] - raw[f"{b}_{mode}"]
                who = np.mean([delta[g].mean(0) for g in groups], 0)
                paired[f"{mode}_{a}_minus_{b}"] = dict(**score_s_v66p.cell_weighted_summary(delta, ORDERS, counts),
                    people=dict(zip(PEOPLE, who.tolist())),
                    orders=dict(zip(ORDERS, np.average(delta, weights=counts, axis=1).tolist())))
        result[str(seed)] = dict(absolute=absolute, paired=paired, teacher=reference)
    outcome = {}
    for contrast in PRIMARY:
        a, b = (result[str(s)]["paired"][contrast]["interval"] for s in (0, 1))
        outcome[contrast] = "supported" if min(a[0], b[0]) > 0 else "refuted" if max(a[1], b[1]) <= 0 else "undecided"
    return result, outcome


def run():
    import protocol

    origin_report()
    with np.load(DATA, allow_pickle=False) as z:
        subject, visit = z["subject"], z["session"]
    person = np.asarray([s.rsplit("/", 1)[-1] for s in subject])
    stream = np.char.add(np.char.add(subject, "|"), visit)
    decisions, lock, guards = lock_decisions(subject, visit, person, stream)
    dataset = protocol.load()
    if not np.array_equal(subject, dataset["subject"]) or not np.array_equal(stream, dataset["stream"]):
        raise ValueError("official metric rows differ from the sealed held rows")
    for held in PEOPLE:
        rows, src, roster = split(subject, visit, held)
        for seed in (0, 1):
            source_receipts(held, seed, False, rows, src, roster, source_y=dataset["y"][rows])
    score, outcome = score_whole(decisions, dataset, person, stream)
    output = HERE / "whole_results.json"
    with output.open("x") as handle:
        json.dump(dict(scope="held-new WHOLE DEV", seeds=score, outcome=outcome, blind_receipt=lock,
                       actor=ACTOR_SHA, scorer=sha(HERE / "score.sha256"), guards=guards, f15=F15_RESULT),
                  handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"PASS F16 held whole-decoder score {sha(output)}", flush=True)


if __name__ == "__main__":
    run()
